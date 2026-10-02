"""
Configuration centralisée du projet Atlas CISIA.

Organisation :
- Settings       : infrastructure / environnement
- ProjectConfig  : features et dataset
- TrainingConfig : entraînement des modèles
- BusinessConfig : contraintes métier et critères de validation
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ==========================================================
# RACINE PROJET
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# SETTINGS INFRASTRUCTURE
# ==========================================================
class SystemConfig(BaseSettings):
    """
    Configuration globale du projet.
    """
    # ------------------------------------------------------
    # Projet
    # ------------------------------------------------------
    PROJECT_NAME: str = "Atlas CISIA"
    ENVIRONMENT: str = "dev"

    NO_PROXY: str = "localhost,127.0.0.1"

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------
    DATASET_PATH: str = "data/dataset_trajectoire_emploi.csv"

    # ------------------------------------------------------
    # Modèle
    # ------------------------------------------------------
    MODEL_NAME: str = "employment_risk_model"
    MODEL_PATH: str = "outputs/employment_risk_model.pkl"


    # ------------------------------------------------------
    # MLflow
    # ------------------------------------------------------
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "Employment_Prediction"

    # ------------------------------------------------------
    # API
    # ------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    PREDICTION_LOG_PATH: str = ("app/api/logs/predictions.jsonl")

    # ------------------------------------------------------
    # UI Streamlit
    # ------------------------------------------------------
    STREAMLIT_API_URL: str = ("http://api:8000")
    STREAMLIT_PORT: int = 8501
    
    # ------------------------------------------------------
    # SQLite
    # ------------------------------------------------------
    DATABASE_PATH: str = ("app/ui/database/history.db")

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


system_cfg = SystemConfig()




# ==========================================================
# CONFIGURATION PROJET
# ==========================================================

class ProjectConfig:

    """
    Définition du dataset.
    """

    TARGET = "classe_retour_emploi"

    FEATURE_TEXT = "synthese_entretien"

    FEATURES_NUMERIC = [
        "age",
        "anciennete_poste_ans",
    ]

    FEATURES_CATEGORICAL = [
        "niveau_diplome",
        "code_rome_vise",
        "departement",
        "est_allocataire",
        "nationalite_hors_ue",
    ]

    FEATURES_SENSITIVE = [
        "age",
        "nationalite_hors_ue",
    ]

    ALL_FEATURES = (
        FEATURES_NUMERIC
        + FEATURES_CATEGORICAL
        + [FEATURE_TEXT]
    )


project_cfg = ProjectConfig()


# ==========================================================
# CONFIGURATION ENTRAINEMENT
# ==========================================================

class TrainingConfig:

    """
    Paramètres ML reproductibles.
    """

    SEED = 42

    TEST_SIZE = 0.20

    N_SPLITS = 5

    SCORING_METRIC = "f1_macro"

    CV_STRATEGY = "StratifiedKFold"

    CLASS_WEIGHT_MODE = "balanced"

    RANDOM_SEARCH_ITERATIONS = 25

    MODEL_CANDIDATES = [
        "RandomForest",
        "LightGBM",
        "XGBoost",
    ]


training_cfg = TrainingConfig()


# ==========================================================
# CONFIGURATION METIER
# ==========================================================

class BusinessConfig:

    """
    Contraintes métier.
    """

    # ------------------------------------------------------
    # Classes
    # ------------------------------------------------------

    CLASSE_RETOUR_RAPIDE = 0
    CLASSE_RETOUR_MOYEN = 1
    CLASSE_RETOUR_DIFFICILE = 2

    # ------------------------------------------------------
    # Classe critique
    # ------------------------------------------------------

    CRITICAL_CLASS = 2

    # ------------------------------------------------------
    # Seuils de validation modèle
    # ------------------------------------------------------

    MIN_ACCURACY = 0.70

    MIN_MACRO_F1 = 0.65

    MIN_RECALL_CLASS_2 = 0.50

    MIN_PRECISION_CLASS_2 = 0.60

    # ------------------------------------------------------
    # Seuils d'alerte
    # ------------------------------------------------------

    MAX_CRITICAL_ERROR_RATE = 0.10

    # ------------------------------------------------------
    # Types d'erreurs critiques
    # ------------------------------------------------------

    CRITICAL_MISCLASSIFICATIONS = [
        (2, 0),
    ]

    # ------------------------------------------------------
    # Equité
    # ------------------------------------------------------

    FAIRNESS_GAP_THRESHOLD = 0.10


business_cfg = BusinessConfig()


