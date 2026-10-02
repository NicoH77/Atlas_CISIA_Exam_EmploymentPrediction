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
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
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

from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# BASE_DIR = Path(__file__).resolve().parent
# ROOT_DIR = Path(__file__).resolve().parents[2]
# load_dotenv(ROOT_DIR / ".env")

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "pkl")

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(ROOT_DIR / "models" / "employment_risk_model.pkl"),
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
    "http://mlflow:5000",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "Employment_Prediction",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "employment_risk_model",
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


print(f"✅ Root : {ROOT_DIR}")
print(f"✅ Back-end : {MODEL_BACKEND}")

if MODEL_BACKEND.upper() == "PKL":
    print(f"✅ Model Path : {MODEL_PATH}")

elif MODEL_BACKEND.upper() == "MLFLOW":
    print(f"✅ MLflow Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"✅ MLflow Experiment : {MLFLOW_EXPERIMENT_NAME}")
    print(f"✅ Model Name : {MODEL_NAME}")
    print(f"✅ Model Alias : {MODEL_ALIAS}")


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
# INSTRUMENTATOR
# ==============================================================================
instrumentator = None # Define globally or manage scope appropriately

@asynccontextmanager
async def lifespan(app: FastAPI):

    logging.info("Startup complete.")
    yield
    logging.info("Shutdown complete.")



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
    lifespan=lifespan,
)

# Prometheus
Instrumentator().instrument(app).expose(app)

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
        print(features)
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
    preprocessor = model.named_steps["prep"]

    features = {}

    for name, _transformer, columns in preprocessor.transformers_:

        if name == "num":
            features["num_features"] = list(columns)
        elif name == "cat":
            features["cat_features"] = list(columns)
        elif name == "txt":
            if isinstance(columns, str):
                features["txt_features"] = columns
            else:
                features["txt_features"] = columns[0]
    
    print(features)
    
    logger.info(
        "Modèle MLflow chargé : %s version %s",
        MODEL_NAME,
        active_model_version,
    )

    

load_model()

# ==============================================================================
# CHARGEMENT DES DONNEES POUR LA PREDICTION
# ==============================================================================

def build_model_input(data, features):

    payload = data.model_dump()

    payload["departement"] = (
        payload["code_insee_commune"][:2]
    )

    all_features = []

    for feature_list in features.values():

        if isinstance(feature_list, list):
            all_features.extend(feature_list)

        elif isinstance(feature_list, str):
            all_features.append(feature_list)

    row = {
        feature: payload.get(feature)
        for feature in all_features
    }

    return pd.DataFrame([row])

# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def utc_now_iso() -> str:
    """Retourne la date UTC au format ISO 8601."""

    return datetime.now(UTC).isoformat()



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

    if not features:
        return [
            col
            for col in dataframe.columns
            if col != TARGET_COLUMN
        ]

    expected_features = [
        *features.get("num_features", []),
        *features.get("cat_features", []),
    ]

    txt_feature = features.get("txt_features")

    if isinstance(txt_feature, str):
        expected_features.append(txt_feature)
    elif isinstance(txt_feature, list):
        expected_features.extend(txt_feature)

    return list(dict.fromkeys(expected_features))


def validate_retraining_dataframe(dataframe: pd.DataFrame) -> list[str]:
    """Valide et prépare le dataset brut avant réentraînement."""
    feat_num = ["age", "anciennete_poste_ans"]
    feat_cat = [
        "niveau_diplome",
        "code_rome_vise",
        "departement",
        "est_allocataire",
        "nationalite_hors_ue",
    ]
    feat_txt = "synthese_entretien"
    target = "classe_retour_emploi"
    feat_sen = ["age", "nationalite_hors_ue"]

    required_input_columns = [
        "usager_id",
        "age",
        "niveau_diplome",
        "anciennete_poste_ans",
        "code_rome_vise",
        "code_insee_commune",
        "est_allocataire",
        "nationalite_hors_ue",
        "synthese_entretien",
        target,
    ]
    model_features = feat_num + feat_cat + [feat_txt]

    if dataframe.empty:
        raise HTTPException(status_code=422, detail="Le fichier CSV est vide.")

    missing_columns = [
        column
        for column in required_input_columns
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Des variables obligatoires sont absentes.",
                "missing_features": missing_columns,
                "required_features": required_input_columns,
            },
        )

    if TARGET_COLUMN != target:
        raise HTTPException(
            status_code=500,
            detail=(
                "Configuration incohérente : TARGET_COLUMN doit valoir "
                f"'{target}', valeur reçue : '{TARGET_COLUMN}'."
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

    # Préserve ou restaure les zéros initiaux supprimés par l'inférence CSV.
    insee_codes = (
        dataframe["code_insee_commune"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    valid_insee_mask = insee_codes.str.fullmatch(
        r"(?:\d{5}|2[AB]\d{3})",
        na=False,
    )
    if not valid_insee_mask.all():
        invalid_rows = dataframe.index[~valid_insee_mask].tolist()[:10]
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "La colonne 'code_insee_commune' contient des codes "
                    "invalides. Chaque code doit avoir cinq caractères."
                ),
                "invalid_row_indices": invalid_rows,
            },
        )

    dataframe["code_insee_commune"] = insee_codes
    dataframe["departement"] = np.where(
        insee_codes.str.startswith(("97", "98")),
        insee_codes.str[:3],
        insee_codes.str[:2],
    )

    if dataframe[target].isna().any():
        raise HTTPException(
            status_code=422,
            detail="La cible contient des valeurs manquantes.",
        )

    try:
        numeric_target = pd.to_numeric(dataframe[target], errors="raise")
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="La colonne cible doit contenir des entiers 0, 1 ou 2.",
        ) from exc

    if not np.equal(numeric_target, numeric_target.astype(int)).all():
        raise HTTPException(
            status_code=422,
            detail="La colonne cible doit contenir des entiers 0, 1 ou 2.",
        )
    dataframe[target] = numeric_target.astype(int)

    received_classes = set(dataframe[target].unique().tolist())
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

    class_counts = dataframe[target].value_counts()
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

    logger.info(
        "Retraining dataframe validated",
        extra={
            "extra_data": {
                "row_count": len(dataframe),
                "model_features": model_features,
                "sensitive_features": feat_sen,
                "target": target,
            }
        },
    )
    return model_features



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
    print("FEATURES =", features)
    input_df = build_model_input(data,features)

    print(input_df)
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
                        strict=True,
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
    feedback_file: Annotated[UploadFile, File(...)],
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

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle actif n'est pas chargé.",
        )

    if not retraining_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Un réentraînement est déjà en cours.",
        )

    run_id: str | None = None
    model_version: str | None = None

    try:
        # ----------------------------------------------------------
        # 1. Validation du fichier chargé
        # ----------------------------------------------------------

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
                io.BytesIO(raw_content),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="Le fichier CSV est illisible.",
            ) from exc

        logger.info(
            "retraining_file_loaded",
            extra={
                "extra_data": {
                    "retraining_id": retraining_id,
                    "filename": feedback_file.filename,
                    "row_count": len(dataframe),
                    "column_count": len(dataframe.columns),
                }
            },
        )

        print(
            f"✅ Données chargées : "
            f"{len(dataframe)} lignes, "
            f"{len(dataframe.columns)} colonnes"
        )
        
        # ----------------------------------------------------------
        # 2. Validation et préparation des variables
        # ----------------------------------------------------------

        expected_features = validate_retraining_dataframe(
            dataframe
        )

        x = dataframe[expected_features].copy()
        y = dataframe[TARGET_COLUMN].copy()

        print(
            f"✅ Dataset modèle préparé : "
            f"{x.shape[0]} lignes, "
            f"{x.shape[1]} features"
        )
        print(f"✅ Features utilisées : {expected_features}")
        print(
            f"✅ Répartition globale des classes : "
            f"{y.value_counts().sort_index().to_dict()}"
        )

        logger.info(
            "retraining_dataframe_validated",
            extra={
                "extra_data": {
                    "retraining_id": retraining_id,
                    "row_count": len(x),
                    "feature_count": x.shape[1],
                    "features": expected_features,
                    "class_distribution": (
                        y.value_counts()
                        .sort_index()
                        .to_dict()
                    ),
                }
            },
        )

        # ----------------------------------------------------------
        # 3. Découpage train/validation stratifié
        # ----------------------------------------------------------
        try:
            (
                x_train,
                x_validation,
                y_train,
                y_validation,
            ) = train_test_split(
                x,
                y,
                test_size=VALIDATION_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Le découpage stratifié est impossible. "
                    "Vérifier le nombre d'observations par classe "
                    "et la taille du jeu de validation."
                ),
            ) from exc

        print(
            f"✅ Split effectué : "
            f"total={len(x)} | "
            f"train={len(x_train)} "
            f"({len(x_train) / len(x):.1%}) | "
            f"validation={len(x_validation)} "
            f"({len(x_validation) / len(x):.1%})"
        )

        print(
            f"✅ Classes train : "
            f"{y_train.value_counts().sort_index().to_dict()}"
        )
        print(
            f"✅ Classes validation : "
            f"{y_validation.value_counts().sort_index().to_dict()}"
        )

        logger.info(
            "retraining_split_completed",
            extra={
                "extra_data": {
                    "retraining_id": retraining_id,
                    "train_row_count": len(x_train),
                    "validation_row_count": len(x_validation),
                    "validation_size": VALIDATION_SIZE,
                    "random_state": RANDOM_STATE,
                    "train_class_distribution": (
                        y_train.value_counts()
                        .sort_index()
                        .to_dict()
                    ),
                    "validation_class_distribution": (
                        y_validation.value_counts()
                        .sort_index()
                        .to_dict()
                    ),
                }
            },
        )

        
        # ----------------------------------------------------------
        # 4. Évaluation du modèle champion
        # ----------------------------------------------------------
        try:
            with model_lock:
                baseline_predictions = model.predict(
                    x_validation
                )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Le modèle actif n'a pas pu être évalué "
                    "sur le jeu de validation."
                ),
            ) from exc

        baseline_metrics = compute_metrics(
            y_validation,
            baseline_predictions,
        )

        print("========================================")
        print("📊 Évaluation du modèle champion")
        print(f"Accuracy          : {baseline_metrics['accuracy']}")
        print(f"F1 macro          : {baseline_metrics['f1_macro']}")
        print(
            "Recall classe 2   : "
            f"{baseline_metrics['recall_class_2']}"
        )
        print(
            "Erreurs critiques : "
            f"{baseline_metrics['critical_errors_2_to_0']}"
        )
        print(
            "Taux critique     : "
            f"{baseline_metrics['critical_error_rate_2_to_0']}"
        )
        print(
            "Matrice confusion : "
            f"{baseline_metrics['confusion_matrix']}"
        )
        print("========================================")
        

        # ----------------------------------------------------------
        # 5. Initialisation du tracking MLflow
        # ----------------------------------------------------------
        if 1 == 1:
            mlflow.set_tracking_uri(
                MLFLOW_TRACKING_URI
            )
            mlflow.set_experiment(
                MLFLOW_EXPERIMENT_NAME
            )

            run_context = mlflow.start_run(
                run_name=f"retraining-{retraining_id}"
            )
        else:
            # Aucun run distant pour le backend PKL.
            run_context = nullcontext(None)

        # Le run est ouvert avant clone(), fit() et predict().
        with run_context as active_run:
            if active_run is not None:
                run_id = active_run.info.run_id

                mlflow.set_tags(
                    {
                        "retraining_id": retraining_id,
                        "feedback_source": "conseiller_validated",
                        "human_supervision": "required",
                        "model_role": "candidate",
                        "active_model_version": (
                            active_model_version or "unknown"
                        ),
                        "promotion_requested": str(
                            promote_if_valid
                        ).lower(),
                        "status": "training",
                    }
                )

                mlflow.log_params(
                    {
                        "input_row_count": len(dataframe),
                        "feature_count": len(expected_features),
                        "train_row_count": len(x_train),
                        "validation_row_count": len(x_validation),
                        "validation_size": VALIDATION_SIZE,
                        "random_state": RANDOM_STATE,
                        "target_column": TARGET_COLUMN,
                        "max_f1_macro_drop": MAX_F1_MACRO_DROP,
                        "min_class_2_recall": MIN_CLASS_2_RECALL,
                        "max_critical_errors": MAX_CRITICAL_ERRORS,
                        "source_filename": (
                            feedback_file.filename or "unknown"
                        ),
                    }
                )

                mlflow.log_dict(
                    {
                        "features": expected_features,
                        "target": TARGET_COLUMN,
                    },
                    "dataset/feature_configuration.json",
                )

                mlflow.log_dict(
                    {
                        "global": (
                            y.value_counts()
                            .sort_index()
                            .to_dict()
                        ),
                        "train": (
                            y_train.value_counts()
                            .sort_index()
                            .to_dict()
                        ),
                        "validation": (
                            y_validation.value_counts()
                            .sort_index()
                            .to_dict()
                        ),
                    },
                    "dataset/class_distributions.json",
                )

                baseline_scalar_metrics = {
                    f"baseline_{key}": value
                    for key, value in baseline_metrics.items()
                    if isinstance(value, (int, float))
                    and value is not None
                }

                mlflow.log_metrics(
                    baseline_scalar_metrics
                )

                mlflow.log_dict(
                    {
                        "labels": [0, 1, 2],
                        "matrix": baseline_metrics[
                            "confusion_matrix"
                        ],
                    },
                    "metrics/baseline_confusion_matrix.json",
                )

        
            # ----------------------------------------------------------
            # 6. Clonage et entraînement du candidat
            # ----------------------------------------------------------
            try:
                with model_lock:
                    candidate_model = clone(model)
            except Exception as exc:
                if active_run is not None:
                    mlflow.set_tag("status", "clone_failed")

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Le modèle actif ne peut pas être cloné. "
                        "L'artefact doit contenir un estimateur ou "
                        "un pipeline Scikit-learn clonable."
                    ),
                ) from exc

            try:
                candidate_model.fit(
                    x_train,
                    y_train,
                )
            except Exception as exc:
                if active_run is not None:
                    mlflow.set_tag("status", "training_failed")

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "L'entraînement du modèle candidat "
                        "a échoué."
                    ),
                ) from exc

            # ----------------------------------------------------------
            # 7. Évaluation du candidat
            # ----------------------------------------------------------
            try:
                candidate_predictions = candidate_model.predict(
                    x_validation
                )
            except Exception as exc:
                if active_run is not None:
                    mlflow.set_tag("status", "evaluation_failed")

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Le modèle candidat n'a pas pu être évalué."
                    ),
                ) from exc

            candidate_metrics = compute_metrics(
                y_validation,
                candidate_predictions,
            )

            print("========================================")
            print("📊 Évaluation du modèle candidat")
            print(
                f"Accuracy          : "
                f"{candidate_metrics['accuracy']}"
            )
            print(
                f"F1 macro          : "
                f"{candidate_metrics['f1_macro']}"
            )
            print(
                f"Recall classe 2   : "
                f"{candidate_metrics['recall_class_2']}"
            )
            print(
                f"Erreurs critiques : "
                f"{candidate_metrics['critical_errors_2_to_0']}"
            )
            print(
                f"Taux critique     : "
                f"{candidate_metrics['critical_error_rate_2_to_0']}"
            )
            print(
                f"Matrice confusion : "
                f"{candidate_metrics['confusion_matrix']}"
            )
            print("========================================")
        

        # ----------------------------------------------------------
        # Vérification des garde-fous (comparaison champion vs. candidat)
        # ----------------------------------------------------------
        gates_passed, rejection_reasons = evaluate_promotion(
            baseline_metrics,
            candidate_metrics,
        )

        print(f"✅ Evalution promotion : gate passed ? {gates_passed}")

        print(f"promote_if_valid = {promote_if_valid}")
        print(f"gates_passed = {gates_passed}")
        print(f"model_version = {model_version}")
        
        # ----------------------------------------------------------
        # Promotion éventuelle
        # ----------------------------------------------------------
        promoted = False

        try:
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

                load_model()

                promoted = True

        except Exception as exc:
            logger.exception(
                "Échec de la promotion MLflow : %s",
                exc
            )

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
            train_row_count=len(x_train),
            validation_row_count=len(x_validation),
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

