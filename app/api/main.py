"""
===============================================================================
Nom du script : main.py
Auteur : Nico H
Projet : Orientation et tri multimodal des demandeurs d'emploi 
Version : 1.0
===============================================================================

Fonctionnalités principales
---------------------------
- Chargement du modèle depuis un fichier PKL ou MLflow.
- Vérification de l'état de l'API.
- Consultation des métadonnées du modèle déployé.
- Prédiction du niveau de risque.
- Réentraînement monitoré à partir de feedbacks validés.
- Évaluation d'un modèle candidat avant promotion.
- Traçabilité des expériences avec MLflow lorsque disponible.

Endpoints
---------
GET /
    Endpoint racine.

GET /health
    Vérifie que l'API et le modèle sont disponibles.

GET /model-info
    Retourne les informations relatives au modèle actif.

POST /predict
    Réalise une prédiction à partir des données d'un usager.

POST /retrain
    Réentraîne un modèle candidat à partir d'un CSV contenant des feedbacks
    validés, l'évalue et le promeut uniquement si les garde-fous sont respectés.
"""

import io
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split



# ==============================================================================
# JOURNALISATION
# ==============================================================================


# ==============================================================================
# JOURNALISATION JSON
# ==============================================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class JsonFormatter(logging.Formatter):
    """
    Formateur JSON pour faciliter la traçabilité et l'audit.
    """

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(
                record,
                "%Y-%m-%dT%H:%M:%S"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)

        return json.dumps(
            log_record,
            ensure_ascii=False
        )


logger = logging.getLogger("employment-risk-api")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(
    LOG_DIR / "predictions.jsonl",
    encoding="utf-8"
)

file_handler.setFormatter(JsonFormatter())

logger.addHandler(file_handler)

# facultatif : affichage console
console_handler = logging.StreamHandler()
console_handler.setFormatter(JsonFormatter())
logger.addHandler(console_handler)






# ==============================================================================
# CONFIGURATION
# ==============================================================================


BASE_DIR = Path(__file__).resolve().parent


MODEL_BACKEND = os.getenv("MODEL_BACKEND", "pkl")

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "employment_risk_model.pkl"),
    )
)


RETRAIN_OUTPUT_DIR = Path(
    os.getenv(
        "RETRAIN_OUTPUT_DIR",
        "/app/api/models",
    )
)

TARGET_COLUMN = os.getenv(
    "TARGET_COLUMN",
    "classe_retour_emploi",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "employment_model_retraining",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "employment_class_model",
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion",
)

# Garde-fous de promotion.
# Ces valeurs sont des paramètres opérationnels, à justifier dans le rapport.
MAX_F1_MACRO_DROP = float(
    os.getenv("MAX_F1_MACRO_DROP", "0.01")
)

MIN_CLASS_2_RECALL = float(
    os.getenv("MIN_CLASS_2_RECALL", "0.50")
)

MAX_CRITICAL_ERRORS = int(
    os.getenv("MAX_CRITICAL_ERRORS", "5")
)

MIN_RETRAINING_ROWS = int(
    os.getenv("MIN_RETRAINING_ROWS", "30")
)

VALIDATION_SIZE = float(
    os.getenv("RETRAIN_VALIDATION_SIZE", "0.20")
)

RANDOM_STATE = int(
    os.getenv("RANDOM_STATE", "42")
)

ALLOWED_CLASSES = {0, 1, 2}

# Un seul réentraînement à la fois dans ce processus.
retraining_lock = threading.Lock()

# Protège le remplacement du modèle actif.
model_lock = threading.RLock()


# ==============================================================================
# APPLICATION FASTAPI
# ==============================================================================

app = FastAPI(
    title="Orientation et tri multimodal des demandeurs d'emploi",
    description=(
        "API d'aide à l'orientation. "
        "La prédiction ne remplace pas la décision du conseiller."
    ),
    version="1.1",
)


# ------------------------------------------------------------------
# Schéma des données d'entrée (pydantic)
# ------------------------------------------------------------------

class UserData(BaseModel):
    """
    Données du scénario 1 : approche multimodale complète.

    usager_id n'est pas transmis au modèle s'il ne figure pas dans la liste
    des features de l'artefact. Il peut néanmoins être utilisé pour la
    traçabilité des requêtes.
    """
    usager_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )
    age: float | None = Field(
        default=None,
        ge=16,
        le=100,
    )
    niveau_diplome: str | None = Field(
        default=None,
        max_length=100,
    )
    anciennete_poste_ans: float = Field(
        ...,
        ge=0,
        le=80,
    )
    code_rome_vise: str = Field(
        ...,
        min_length=5,
        max_length=5,
    )
    code_insee_commune: str = Field(
        ...,
        min_length=5,
        max_length=5,
    )
    est_allocataire: int | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    nationalite_hors_ue: int = Field(
        ...,
        ge=0,
        le=1,
    )
    synthese_entretien: str | None = Field(
        default=None,
        max_length=10_000,
    )


class RetrainingResponse(BaseModel):
    retraining_id: str
    status: str
    promoted: bool
    rejection_reasons: list[str]
    row_count: int
    train_row_count: int
    validation_row_count: int
    baseline_metrics: dict[str, Any]
    candidate_metrics: dict[str, Any]
    model_backend: str
    model_version: str | None = None
    mlflow_run_id: str | None = None
    completed_at: str



# ==============================================================================
# CHARGEMENT DU MODÈLE (selon le contexte)
# ==============================================================================

model: Any = None
features: list[str] | None = None
client: MlflowClient | None = None
active_model_version: str | None = None


def load_model() -> None:
    """Charge le modèle depuis Joblib ou depuis MLflow."""

    global model
    global features
    global client
    global active_model_version

    if MODEL_BACKEND == "pkl":
        logger.info("Chargement du modèle depuis %s", MODEL_PATH)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Artefact modèle introuvable : {MODEL_PATH}"
            )

        artifact = joblib.load(MODEL_PATH)

        if not isinstance(artifact, dict):
            raise ValueError(
                "L'artefact PKL doit être un dictionnaire."
            )

        if "model" not in artifact:
            raise ValueError(
                "La clé 'model' est absente de l'artefact."
            )

        model = artifact["model"]
        features = artifact.get("features")
        active_model_version = artifact.get("version")

        logger.info("Modèle PKL chargé avec succès")
        return

    logger.info("Chargement du modèle depuis MLflow")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )
    active_model_version = str(model_version.version)

    # Les features doivent idéalement être incluses dans la signature MLflow.
    features = None

    logger.info(
        "Modèle MLflow chargé : %s version %s",
        MODEL_NAME,
        active_model_version,
    )


load_model()

# ==============================================================================
# CHARGEMENT DES DONNEES
# ==============================================================================

def build_model_input(data, features):

    payload = data.model_dump()

    payload["departement"] = (payload["code_insee_commune"][:2])

    all_features = []
    all_features.extend(features.get("num_features", []))
    all_features.extend(features.get("cat_features", []))
    txt = features.get("txt_features")

    if isinstance(txt, str):
        all_features.append(txt)

    return pd.DataFrame([{f: payload.get(f) for f in all_features}])

# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def utc_now_iso() -> str:
    """Retourne la date UTC au format ISO 8601."""

    return datetime.now(timezone.utc).isoformat()



def audit_prediction(
    request_id: str,
    session_id: str,
    user_data: dict,
    prediction: int,
    probabilities: dict | None,
    confidence: float | None,
):
    logger.info(
        "prediction_completed",
        extra={
            "extra_data": {
                "request_id": request_id,
                "session_id": session_id,
                "inference_date": utc_now_iso(),
                "input": user_data,
                "prediction": prediction,
                "probabilities": probabilities,
                "confidence": confidence,
                "model_version": active_model_version
            }
        }
    )


    
def get_model_features(dataframe: pd.DataFrame) -> list[str]:
    """
    Détermine les colonnes utilisées pour l'entraînement.

    Si l'artefact contient explicitement les features, elles sont prioritaires.
    Sinon, toutes les colonnes sauf la cible sont utilisées.
    """

    if features:
        return list(features)

    return [
        column
        for column in dataframe.columns
        if column != TARGET_COLUMN
    ]


def validate_retraining_dataframe(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Valide le dataset de feedback avant réentraînement.

    Returns
    -------
    list[str]
        Liste ordonnée des features attendues.
    """

    if dataframe.empty:
        raise HTTPException(
            status_code=422,
            detail="Le fichier CSV est vide.",
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise HTTPException(
            status_code=422,
            detail=(
                f"La colonne cible '{TARGET_COLUMN}' est absente."
            ),
        )

    if len(dataframe) < MIN_RETRAINING_ROWS:
        raise HTTPException(
            status_code=422,
            detail=(
                "Nombre insuffisant de feedbacks validés : "
                f"{len(dataframe)} reçu(s), "
                f"{MIN_RETRAINING_ROWS} requis."
            ),
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise HTTPException(
            status_code=422,
            detail="La cible contient des valeurs manquantes.",
        )

    try:
        dataframe[TARGET_COLUMN] = dataframe[
            TARGET_COLUMN
        ].astype(int)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "La colonne cible doit contenir des entiers 0, 1 ou 2."
            ),
        ) from exc

    received_classes = set(
        dataframe[TARGET_COLUMN].unique().tolist()
    )

    invalid_classes = received_classes - ALLOWED_CLASSES

    if invalid_classes:
        raise HTTPException(
            status_code=422,
            detail=(
                "Classes invalides détectées : "
                f"{sorted(invalid_classes)}. "
                "Les classes autorisées sont 0, 1 et 2."
            ),
        )

    class_counts = dataframe[TARGET_COLUMN].value_counts()

    if len(class_counts) < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                "Le lot doit contenir au moins deux classes "
                "pour permettre une évaluation."
            ),
        )

    if class_counts.min() < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                "Chaque classe présente dans le lot doit contenir "
                "au moins deux observations pour le découpage stratifié."
            ),
        )

    expected_features = get_model_features(dataframe)

    missing_features = sorted(
        set(expected_features) - set(dataframe.columns)
    )

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Des variables attendues sont absentes.",
                "missing_features": missing_features,
            },
        )

    # L'identifiant ne doit pas devenir accidentellement une variable
    # prédictive si le pipeline initial ne l'utilisait pas.
    if "usager_id" in expected_features:
        logger.warning(
            "usager_id figure parmi les features du modèle. "
            "Vérifier le risque de mémorisation et de fuite de données."
        )

    return expected_features


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Calcule les métriques de validation et les erreurs critiques.

    Une erreur critique correspond à une classe réelle 2 prédite en classe 0.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    critical_errors = int(
        np.sum(
            (np.asarray(y_true) == 2)
            & (np.asarray(y_pred) == 0)
        )
    )

    class_2_support = int(
        np.sum(np.asarray(y_true) == 2)
    )

    critical_error_rate = (
        critical_errors / class_2_support
        if class_2_support > 0
        else None
    )

    return {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)),
            4,
        ),
        "f1_macro": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall_class_2": round(
            float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=[2],
                    average=None,
                    zero_division=0,
                )[0]
            ),
            4,
        ),
        "critical_errors_2_to_0": critical_errors,
        "critical_error_rate_2_to_0": (
            round(float(critical_error_rate), 4)
            if critical_error_rate is not None
            else None
        ),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_promotion(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Applique les règles de contrôle avant promotion du candidat.
    """

    reasons: list[str] = []

    allowed_f1 = (
        baseline_metrics["f1_macro"]
        - MAX_F1_MACRO_DROP
    )

    if candidate_metrics["f1_macro"] < allowed_f1:
        reasons.append(
            "Le F1 macro du candidat se dégrade au-delà "
            "du seuil autorisé."
        )

    if (
        candidate_metrics["recall_class_2"]
        < MIN_CLASS_2_RECALL
    ):
        reasons.append(
            "Le rappel de la classe 2 est inférieur "
            "au seuil minimal."
        )

    if (
        candidate_metrics["critical_errors_2_to_0"]
        > MAX_CRITICAL_ERRORS
    ):
        reasons.append(
            "Le nombre d'erreurs critiques 2 vers 0 "
            "dépasse le maximum autorisé."
        )

    if (
        candidate_metrics["critical_errors_2_to_0"]
        > baseline_metrics["critical_errors_2_to_0"]
    ):
        reasons.append(
            "Le candidat augmente les erreurs critiques "
            "par rapport au modèle actif."
        )

    return len(reasons) == 0, reasons


def save_candidate_locally(
    candidate_model: Any,
    candidate_features: list[str],
    metrics: dict[str, Any],
    retraining_id: str,
) -> Path:
    """
    Sauvegarde de façon atomique un artefact candidat en local.
    """

    RETRAIN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = (
        RETRAIN_OUTPUT_DIR
        / f"employment_class_model_{retraining_id}.pkl"
    )
    temporary_path = final_path.with_suffix(".tmp")

    artifact = {
        "model": candidate_model,
        "features": candidate_features,
        "metrics": metrics,
        "version": retraining_id,
        "created_at": utc_now_iso(),
    }

    joblib.dump(artifact, temporary_path)
    temporary_path.replace(final_path)

    return final_path


def promote_local_model(
    candidate_path: Path,
) -> None:
    """
    Remplace l'artefact actif par le candidat validé.

    os.replace fournit un remplacement atomique sur un même système
    de fichiers.
    """

    backup_path = MODEL_PATH.with_suffix(".previous.pkl")
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MODEL_PATH.exists():
        current_artifact = joblib.load(MODEL_PATH)
        joblib.dump(current_artifact, backup_path)

    candidate_artifact = joblib.load(candidate_path)
    temporary_active_path = MODEL_PATH.with_suffix(".new.pkl")

    joblib.dump(
        candidate_artifact,
        temporary_active_path,
    )
    temporary_active_path.replace(MODEL_PATH)


def log_retraining_to_mlflow(
    candidate_model: Any,
    X_train: pd.DataFrame,
    metrics: dict[str, Any],
    retraining_id: str,
    promote: bool,
) -> tuple[str | None, str | None]:
    """
    Journalise le run et enregistre le modèle candidat dans MLflow.

    Returns
    -------
    tuple[str | None, str | None]
        Run ID MLflow et version du modèle enregistré.
    """

    if MODEL_BACKEND == "pkl":
        return None, None

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"retraining-{retraining_id}"
    ) as run:
        mlflow.set_tags(
            {
                "retraining_id": retraining_id,
                "feedback_source": "conseiller_validated",
                "promotion_decision": str(promote).lower(),
                "human_supervision": "required",
            }
        )

        mlflow.log_params(
            {
                "row_count": len(X_train),
                "validation_size": VALIDATION_SIZE,
                "random_state": RANDOM_STATE,
                "max_f1_macro_drop": MAX_F1_MACRO_DROP,
                "min_class_2_recall": MIN_CLASS_2_RECALL,
                "max_critical_errors": MAX_CRITICAL_ERRORS,
            }
        )

        scalar_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
            and value is not None
        }
        mlflow.log_metrics(scalar_metrics)

        confusion_matrix_path = (
            RETRAIN_OUTPUT_DIR
            / f"confusion_matrix_{retraining_id}.json"
        )
        confusion_matrix_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        confusion_matrix_path.write_text(
            json.dumps(
                metrics["confusion_matrix"],
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(confusion_matrix_path))

        input_example = X_train.head(3)
        signature = infer_signature(
            X_train,
            candidate_model.predict(X_train),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=candidate_model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
        )

        model_version = getattr(
            model_info,
            "registered_model_version",
            None,
        )

        if promote and model_version is not None:
            mlflow_client = MlflowClient()
            mlflow_client.set_registered_model_alias(
                MODEL_NAME,
                MODEL_ALIAS,
                str(model_version),
            )

        return run.info.run_id, (
            str(model_version)
            if model_version is not None
            else None
        )


# ==============================================================================
# ENDPOINT RACINE
# ==============================================================================

@app.get("/")
def home() -> dict[str, str]:
    """Retourne une présentation minimale du service."""

    return {
        "message": (
            "API d'aide à l'orientation des demandeurs d'emploi"
        ),
        "documentation": "/docs",
    }

# ==============================================================================
# ENDPOINT HEALTH CHECK
# ==============================================================================

@app.get("/health")
def health() -> dict[str, Any]:
    """Vérifie que l'API et le modèle sont opérationnels."""

    return {
        "status": "UP" if model is not None else "DOWN",
        "model_loaded": model is not None,
        "model_backend": MODEL_BACKEND,
        "active_model_version": active_model_version,
        "retraining_in_progress": retraining_lock.locked(),
        "checked_at": utc_now_iso(),
    }


# ==============================================================================
# ENDPOINT MODEL INFO
# ==============================================================================

@app.get("/model-info")
def get_model_info() -> dict[str, Any]:
    """Retourne les informations du modèle actif."""

    if MODEL_BACKEND == "pkl":
        return {
            "source": "joblib",
            "model_path": str(MODEL_PATH),
            "model_type": type(model).__name__,
            "features": features,
            "version": active_model_version,
            "status": "loaded",
        }

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Le client MLflow n'est pas disponible.",
        )

    try:
        model_version = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS,
        )

        return {
            "model_name": model_version.name,
            "version": model_version.version,
            "alias": MODEL_ALIAS,
            "run_id": model_version.run_id,
            "source": model_version.source,
            "status": "loaded",
        }

    except Exception as exc:
        logger.exception(
            "Impossible de récupérer les métadonnées MLflow"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Impossible de récupérer les informations "
                "du modèle MLflow."
            ),
        ) from exc


# ==============================================================================
# ENDPOINT PREDICTION
# ==============================================================================

@app.post("/predict")
def predict(data: UserData) -> dict[str, Any]:
    """
    Réalise une prédiction multiclasse.

    La réponse constitue une aide à la décision et ne remplace pas
    l'analyse du conseiller.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas chargé.",
        )

    request_id = str(uuid4())

    # Construction du DataFrame attendu par le modèle
    input_df = build_model_input(data,features)

    print(input_df)
    print(input_df.isna().sum())
    print(type(model))

    try:
        with model_lock:
            
            prediction = int(model.predict(input_df)[0])

            probabilities = None
            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities_array = model.predict_proba(
                    input_df
                )[0]

                model_classes = getattr(
                    model,
                    "classes_",
                    np.arange(len(probabilities_array)),
                )

                probabilities = {
                    str(int(class_label)): round(
                        float(probability),
                        4,
                    )
                    for class_label, probability in zip(
                        model_classes,
                        probabilities_array,
                    )
                }

                confidence = round(
                    float(np.max(probabilities_array)),
                    4,
                )

        session_id = str(uuid4())

        audit_prediction(
            request_id=request_id,
            session_id=session_id,
            user_data=data.model_dump(),
            prediction=prediction,
            probabilities=probabilities,
            confidence=confidence
        )
        
        return {
            "request_id": request_id,
            "session_id": session_id,
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": confidence,
            "model_version": active_model_version,
            "human_review_required": True,
            "predicted_at": utc_now_iso(),
        }

    except Exception as exc:
        logger.exception(
            "Erreur de prédiction request_id=%s",
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="La prédiction n'a pas pu être réalisée.",
        ) from exc



# ==============================================================================
# ENDPOINT RÉENTRAÎNEMENT MONITORÉ
# ==============================================================================

@app.post(
    "/retrain",
    response_model=RetrainingResponse,
)
async def retrain(
    feedback_file: UploadFile = File(...),
    promote_if_valid: bool = False,
) -> RetrainingResponse:
    """
    Entraîne un modèle candidat à partir d'un fichier CSV de feedbacks.

    Le fichier doit contenir :
    - les variables attendues par le modèle ;
    - la cible `classe_retour_emploi` validée par un conseiller.

    Par défaut, le modèle candidat est évalué et enregistré, mais n'est
    pas promu. La promotion doit être explicitement demandée avec
    `promote_if_valid=true` et reste conditionnée aux garde-fous.
    """

    global model
    global features
    global active_model_version

    retraining_id = str(uuid4())

    if not retraining_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Un réentraînement est déjà en cours.",
        )

    try:
        if not feedback_file.filename:
            raise HTTPException(
                status_code=422,
                detail="Le nom du fichier est absent.",
            )

        if not feedback_file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=415,
                detail="Seuls les fichiers CSV sont acceptés.",
            )

        raw_content = await feedback_file.read()

        if not raw_content:
            raise HTTPException(
                status_code=422,
                detail="Le fichier transmis est vide.",
            )

        try:
            dataframe = pd.read_csv(
                io.BytesIO(raw_content)
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="Le fichier CSV est illisible.",
            ) from exc

        # ----------------------------------------------------------
        # validation des features
        # ----------------------------------------------------------
        expected_features = validate_retraining_dataframe(dataframe)

        X = dataframe[expected_features].copy()
        y = dataframe[TARGET_COLUMN].copy()

        # ----------------------------------------------------------
        # Découpage stratifié du nouveau jeu de données
        # ----------------------------------------------------------
        try:
            X_train, X_validation, y_train, y_validation = (
                train_test_split(
                    X,
                    y,
                    test_size=VALIDATION_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y,
                )
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Le découpage stratifié est impossible. "
                    "Vérifier le nombre d'observations par classe."
                ),
            ) from exc

        # ----------------------------------------------------------
        # Baseline = performance du champion actuel
        # ----------------------------------------------------------
        with model_lock:
            baseline_predictions = model.predict(
                X_validation
            )

        baseline_metrics = compute_metrics(
            y_validation,
            baseline_predictions,
        )

        # ----------------------------------------------------------
        # Entrainement du modèle candidat
        # ----------------------------------------------------------
        try:
            candidate_model = clone(model)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Le modèle actif ne peut pas être cloné. "
                    "Enregistrer le pipeline sklearn complet "
                    "dans l'artefact."
                ),
            ) from exc

        candidate_model.fit(
            X_train,
            y_train,
        )

        candidate_predictions = candidate_model.predict(
            X_validation
        )

        candidate_metrics = compute_metrics(
            y_validation,
            candidate_predictions,
        )

        # ----------------------------------------------------------
        # Vérification des garde-fous (comparaison champion vs. candidat)
        # ----------------------------------------------------------
        gates_passed, rejection_reasons = evaluate_promotion(
            baseline_metrics,
            candidate_metrics,
        )


        # ----------------------------------------------------------
        # Enregistrement du candidat dans MLflow
        # ----------------------------------------------------------
        run_id, model_version = (
            log_retraining_to_mlflow(
                candidate_model=candidate_model,
                X_train=X_train,
                metrics=candidate_metrics,
                retraining_id=retraining_id,
                promote=False
            )
        )

        promoted = False

        # ----------------------------------------------------------
        # Promotion éventuelle
        # ----------------------------------------------------------
        if (
            promote_if_valid
            and gates_passed
            and model_version is not None
        ):

            mlflow_client = MlflowClient()

            mlflow_client.set_registered_model_alias(
                MODEL_NAME,
                MODEL_ALIAS,
                str(model_version)
            )

            promoted = True

            logger.info(
                (
                    "Promotion du modèle %s "
                    "vers alias %s"
                ),
                model_version,
                MODEL_ALIAS
            )

            # recharge automatiquement
            load_model()        


        # ----------------------------------------------------------
        # Statut final
        # ----------------------------------------------------------

        if promoted:
            status = "promoted"
        elif gates_passed:
            status = "validated_not_promoted"
        else:
            status = "rejected"

        logger.info(
            (
                "Retraining completed "
                "retraining_id=%s "
                "status=%s "
                "run_id=%s "
                "model_version=%s"
            ),
            retraining_id,
            status,
            run_id,
            model_version
        )

        return RetrainingResponse(
            retraining_id=retraining_id,
            status=status,
            promoted=promoted,
            rejection_reasons=rejection_reasons,
            row_count=len(dataframe),
            train_row_count=len(X_train),
            validation_row_count=len(X_validation),
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            model_backend="mlflow",
            model_version=model_version,
            mlflow_run_id=run_id,
            completed_at=utc_now_iso()
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Échec du réentraînement %s",
            retraining_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Le réentraînement a échoué. "
                f"Identifiant de suivi : {retraining_id}"
            ),
        ) from exc

    finally:
        retraining_lock.release()


