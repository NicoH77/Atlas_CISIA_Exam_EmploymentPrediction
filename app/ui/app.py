"""
===============================================================================
Nom du script : app.py
Auteur : Nico H
Projet : Orientation et tri multimodal des demandeurs d'emploi
Version : 1.0
===============================================================================

Fonctionnalités principales
---------------------------
- Initialisation de la base locale SQLite.
- Saisie des informations d'un usager dans un formulaire Streamlit.
- Envoi des données à l'API pour prédire le délai potentiel de retour à l'emploi.
- Affichage de la prédiction et du niveau de confiance
- Enregistrement de chaque inférence et de ses données dans l'historique.
- Consultation de l'historique des prédictions.
- Saisie et mise à jour d'un feedback humain sur la classe constatée.
- Export des prédictions revues au format CSV pour le réentraînement.
- Gestion des erreurs liées aux appels API, aux réponses invalides et à SQLite.

Interface
---------
Onglet « Nouvelle prédiction »
    Permet à un conseiller de renseigner les caractéristiques d'un usager,
    d'interroger l'API et de visualiser le résultat de la prédiction.

Onglet « Historique et feedback »
    Permet de consulter les inférences enregistrées, de sélectionner une
    prédiction et de renseigner la classe de retour à l'emploi constatée.

Persistance
-----------
Base SQLite
    Stocke les données d'entrée, la date d'inférence, la classe prédite,
    la confiance, les probabilités par classe et le feedback humain.

Export CSV
    Produit un jeu de données compatible avec le contrat de réentraînement,
    limité par défaut aux prédictions disposant d'un feedback validé.

Configuration
-------------
DB_PATH
    Chemin de la base SQLite. Valeur par défaut :
    /app/database/history.db

API_URL
    URL de l'endpoint de prédiction. Valeur par défaut :
    http://api:8000/predict

MODEL_INFO_URL
    URL de l'endpoint fournissant les métadonnées du modèle. Valeur par défaut :
    http://api:8000/model-info

Responsabilité et supervision humaine
-------------------------------------
Cette application fournit une aide à la décision. La prédiction du modèle ne
constitue pas une décision automatisée définitive. La classe constatée est
renseignée ultérieurement par un utilisateur habilité et la décision finale
reste sous la responsabilité du conseiller.
"""


from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# =====================================================
# CONFIGURATION
# =====================================================
load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "/app/database/history.db"))
API_URL = os.getenv("API_URL", "http://api:8000/predict")
MODEL_INFO_URL = os.getenv("MODEL_INFO_URL", "http://api:8000/model-info")

LABELS = {
    0: "Retour rapide (0-6 mois)",
    1: "Retour moyen (6-12 mois)",
    2: "Risque de chomage longue duree (>12 mois)",
}

EXPORT_COLUMNS = [
    "usager_id",
    "age",
    "niveau_diplome",
    "anciennete_poste_ans",
    "code_rome_vise",
    "code_insee_commune",
    "est_allocataire",
    "nationalite_hors_ue",
    "synthese_entretien",
    "classe_retour_emploi_predite",
    "classe_retour_emploi_revue",
]

st.set_page_config(
    page_title="Prediction retour a l'emploi",
    page_icon="📊",
    layout="wide",
)


st.markdown("""
<style>

.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0rem;
    max-width: 1400px;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.4rem;
}

header[data-testid="stHeader"]{
    visibility:hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 16px;
}

div[data-testid="stSidebar"] {
    width: 260px;
}

.blue-card {
    padding: 12px;
    border-left: 5px solid #0063CB;
    background-color: #F5F9FF;
    border-radius: 8px;
    margin-bottom: 12px;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# BASE SQLITE
# =====================================================
def get_connection() -> sqlite3.Connection:
    """Ouvre une connexion SQLite et retourne les lignes comme dictionnaires."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def create_database() -> None:
    """Cree la table et migre sans perte une ancienne version du schema."""
    required_columns = {
        "age": "INTEGER",
        "niveau_diplome": "TEXT",
        "anciennete_poste_ans": "REAL",
        "code_rome_vise": "TEXT",
        "code_insee_commune": "TEXT",
        "est_allocataire": "INTEGER",
        "nationalite_hors_ue": "INTEGER",
        "synthese_entretien": "TEXT",
        "reviewed_prediction": "INTEGER CHECK (reviewed_prediction IN (0, 1, 2))",
        "feedback_date": "TEXT",
    }

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                usager_id TEXT NOT NULL,
                age INTEGER,
                niveau_diplome TEXT,
                anciennete_poste_ans REAL,
                code_rome_vise TEXT,
                code_insee_commune TEXT,
                est_allocataire INTEGER,
                nationalite_hors_ue INTEGER,
                synthese_entretien TEXT,
                prediction INTEGER NOT NULL CHECK (prediction IN (0, 1, 2)),
                confidence REAL,
                proba_0 REAL NOT NULL,
                proba_1 REAL NOT NULL,
                proba_2 REAL NOT NULL,
                reviewed_prediction INTEGER
                    CHECK (reviewed_prediction IN (0, 1, 2)),
                feedback_date TEXT
            )
            """
        )

        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(predictions)")
        }
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE predictions ADD COLUMN "
                    f"{column_name} {column_type}"
                )

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_date "
            "ON predictions(prediction_date)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_usager "
            "ON predictions(usager_id)"
        )


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def save_prediction(
    features: dict[str, Any],
    prediction: int,
    confidence: float,
    probabilities: dict[str, float],
) -> int:
    """Enregistre une inference complete et retourne son identifiant."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                prediction_date,
                usager_id,
                age,
                niveau_diplome,
                anciennete_poste_ans,
                code_rome_vise,
                code_insee_commune,
                est_allocataire,
                nationalite_hors_ue,
                synthese_entretien,
                prediction,
                confidence,
                proba_0,
                proba_1,
                proba_2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                features["usager_id"],
                int(features["age"]),
                features["niveau_diplome"],
                float(features["anciennete_poste_ans"]),
                features["code_rome_vise"],
                features["code_insee_commune"],
                int(features["est_allocataire"]),
                int(features["nationalite_hors_ue"]),
                features["synthese_entretien"],
                int(prediction),
                float(confidence),
                float(probabilities["0"]),
                float(probabilities["1"]),
                float(probabilities["2"]),
            ),
        )
        return int(cursor.lastrowid)



def load_history() -> pd.DataFrame:
    """Charge toutes les predictions, de la plus recente a la plus ancienne."""
    with get_connection() as connection:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            connection,
        )

def load_prediction(prediction_id: int) -> dict[str, Any] | None:
    """Charge une prediction a partir de son identifiant technique."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
    return dict(row) if row else None

def save_feedback(prediction_id: int, reviewed_prediction: int) -> None:
    """Enregistre ou remplace la classe constatee par l'utilisateur."""
    if reviewed_prediction not in LABELS:
        raise ValueError("La classe constatee doit etre 0, 1 ou 2.")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE predictions
            SET reviewed_prediction = ?, feedback_date = ?
            WHERE id = ?
            """,
            (
                int(reviewed_prediction),
                datetime.now().astimezone().isoformat(timespec="seconds"),
                int(prediction_id),
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("Prediction introuvable.")

def build_training_export(history: pd.DataFrame) -> pd.DataFrame:
    """Construit le jeu CSV conforme au contrat de reentrainement."""
    export_df = history.rename(
        columns={
            "prediction": "classe_retour_emploi_predite",
            "reviewed_prediction": "classe_retour_emploi_revue",
        }
    )
    return export_df.reindex(columns=EXPORT_COLUMNS)


def validate_probabilities(probabilities: dict[str, Any]) -> dict[str, float]:
    """Valide et normalise les probabilites retournees par l'API."""
    normalized = {str(key): float(value) for key, value in probabilities.items()}
    missing = {"0", "1", "2"} - normalized.keys()
    if missing:
        raise ValueError(f"Probabilites absentes pour les classes : {sorted(missing)}")
    if any(not 0.0 <= normalized[key] <= 1.0 for key in ("0", "1", "2")):
        raise ValueError("Une probabilite est hors de l'intervalle [0, 1].")
    return normalized

create_database()


# =====================================================
# EN-TETE ET INFORMATIONS MODELE
# =====================================================
st.markdown("""
<div class="blue-card">
<h2 style="margin-bottom:2px;">
📊 Prédiction de retour à l'emploi
</h2>

<p style="margin-bottom:0px;color:#666;">
Aide à la décision • Validation humaine obligatoire
</p>
</div>
""", unsafe_allow_html=True)

try:
    model_response = requests.get(MODEL_INFO_URL, timeout=5)
    model_response.raise_for_status()
    model_info = model_response.json()
    st.sidebar.title("🤖 Modèle deployé")
    st.sidebar.write(f"**Nom :** {model_info.get('model_name', 'Inconnu')}")
    st.sidebar.write(f"**Version :** {model_info.get('version', 'Inconnue')}")
except (requests.RequestException, ValueError):
    st.sidebar.warning("Informations modèle indisponibles")


# =====================================================
# ONGLETS
# =====================================================

tab_prediction, tab_history = st.tabs(
    ["Nouvelle prédiction", "Historique et feedback"]
)

# =====================================================
# ONGLET PREDICTION
# =====================================================
with tab_prediction:

    with st.form("prediction_form"):

        st.markdown(
            '<div class="section-title">👤 Informations usager</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            usager_id = st.text_input(
                "Identifiant",
                placeholder="ID_0001"
            )

        with c2:
            age = st.number_input(
                "Âge",
                min_value=16,
                max_value=70,
                value=30
            )

        with c3:
            code_insee_commune = st.text_input(
                "Commune",
                placeholder="75000"
            )


        with c4:
            nationalite_hors_ue_txt = st.selectbox(
                "Nationalité hors UE",
                ["Non", "Oui"]
            )


        st.markdown(
            '<div class="section-title">💼 Situation professionnelle</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            code_rome_vise = st.text_input(
                "Code ROME",
                placeholder="M1805"
            )

        with c2:
            niveau_diplome = st.selectbox(
                "Diplôme",
                [
                    "Sans diplôme",
                    "Bac",
                    "Bac+2",
                    "Bac+5",
                ]
            )

        with c3:
            anciennete_poste_ans = st.number_input(
                "Ancienneté (ans)",
                min_value=0.0,
                value=0.0,
                step=0.5
            )

        with c4:
            est_allocataire_txt = st.selectbox(
                "Allocataire",
                ["Non", "Oui"]
            )

        st.markdown(
            '<div class="section-title">📝 Synthèse entretien</div>',
            unsafe_allow_html=True,
        )

        synthese_entretien = st.text_area(
            "",
            height=100,
            placeholder="""
Projet professionnel...
Freins identifiés...
Mobilité...
Compétences clés...
"""
        )

        col_left, col_center, col_right = st.columns([3,2,3])

        with col_center:
            submitted = st.form_submit_button(
                "🔍 Lancer la prédiction",
                use_container_width=True
            )

    est_allocataire = 1 if est_allocataire_txt == "Oui" else 0
    nationalite_hors_ue = 1 if nationalite_hors_ue_txt == "Oui" else 0

    if submitted:
        if not usager_id.strip():
            st.error("L'identifiant usager est obligatoire.")
        else:
            payload = {
                "usager_id": usager_id.strip(),
                "age": int(age),
                "niveau_diplome": niveau_diplome,
                "anciennete_poste_ans": float(anciennete_poste_ans),
                "code_rome_vise": code_rome_vise.strip(),
                "code_insee_commune": code_insee_commune.strip(),
                "est_allocataire": int(est_allocataire),
                "nationalite_hors_ue": int(nationalite_hors_ue),
                "synthese_entretien": synthese_entretien.strip(),
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()

                prediction = int(result["prediction"])
                if prediction not in LABELS:
                    raise ValueError("Classe predite invalide.")

                probabilities = validate_probabilities(result["probabilities"])
                confidence = float(
                    result.get("confidence", probabilities[str(prediction)])
                )

                prediction_id = save_prediction(
                    features=payload,
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                )

                st.success(
                    f"Prediction #{prediction_id} : {LABELS[prediction]}"
                )
                st.metric("Confiance du modele", f"{confidence:.1%}")

                if result.get("human_review_required", False):
                    st.warning(
                        "Cette prediction constitue une aide a la decision. "
                        "La decision finale reste sous la responsabilite du conseiller."
                    )

                proba_df = pd.DataFrame(
                    {
                        "Classe": [
                            "Retour rapide",
                            "Retour moyen",
                            "Risque longue duree",
                        ],
                        "Probabilite": [
                            probabilities["0"],
                            probabilities["1"],
                            probabilities["2"],
                        ],
                    }
                )
                st.subheader("Probabilites par classe")
                st.bar_chart(proba_df.set_index("Classe"))
                st.write(
                    {
                        "Retour rapide": f"{probabilities['0']:.1%}",
                        "Retour moyen": f"{probabilities['1']:.1%}",
                        "Risque longue duree": f"{probabilities['2']:.1%}",
                    }
                )

            except requests.RequestException as error:
                st.error(f"Erreur lors de l'appel API : {error}")
            except (KeyError, TypeError, ValueError) as error:
                st.error(f"Reponse API invalide : {error}")
            except sqlite3.Error as error:
                st.error(f"Erreur lors de l'enregistrement en base : {error}")


# =====================================================
# ONGLET HISTORIQUE ET FEEDBACK
# =====================================================
with tab_history:
    st.subheader("Historique des predictions")
    history = load_history()

    if history.empty:
        st.info("Aucune prediction enregistree.")
    else:
        summary = history[["id", "prediction_date", "usager_id", "prediction"]].copy()
        summary["prediction"] = summary["prediction"].map(LABELS)
        summary = summary.rename(
            columns={
                "id": "ID prediction",
                "prediction_date": "Date prediction",
                "usager_id": "Usager",
                "prediction": "Classe predite",
            }
        )

        st.caption("Selectionnez une ligne pour consulter et completer le feedback.")
        selection = st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="history_table",
        )

        selected_rows = selection.selection.rows
        if selected_rows:
            selected_position = selected_rows[0]
            prediction_id = int(history.iloc[selected_position]["id"])
            record = load_prediction(prediction_id)

            if record is None:
                st.error("La prediction selectionnee n'existe plus.")
            else:
                st.divider()
                st.subheader(f"Feedback de la prediction #{prediction_id}")

                display_data = {
                    "ID prediction": record["id"],
                    "Date prediction": record["prediction_date"],
                    "Usager": record["usager_id"],
                    "Age": record["age"],
                    "Niveau de diplome": record["niveau_diplome"],
                    "Anciennete du poste (ans)": record["anciennete_poste_ans"],
                    "Code ROME vise": record["code_rome_vise"],
                    "Code INSEE commune": record["code_insee_commune"],
                    "Allocataire": record["est_allocataire"],
                    "Nationalite hors UE": record["nationalite_hors_ue"],
                    "Synthese entretien": record["synthese_entretien"],
                    "Classe predite": LABELS.get(
                        record["prediction"], record["prediction"]
                    ),
                    "Probabilite classe 0": record["proba_0"],
                    "Probabilite classe 1": record["proba_1"],
                    "Probabilite classe 2": record["proba_2"],
                }
                st.dataframe(
                    pd.DataFrame(
                        display_data.items(),
                        columns=["Champ", "Valeur"],
                    ),
                    width="stretch",
                    hide_index=True,
                )

                current_review = record["reviewed_prediction"]
                default_index = int(current_review) if current_review is not None else 0

                with st.form(f"feedback_form_{prediction_id}"):
                    reviewed_prediction = st.selectbox(
                        "Classe retour emploi constatee",
                        options=[0, 1, 2],
                        index=default_index,
                        format_func=lambda value: f"{value} - {LABELS[value]}",
                    )
                    feedback_submitted = st.form_submit_button(
                        "💾 Enregistrer le feedback"
                    )

                if feedback_submitted:
                    try:
                        save_feedback(prediction_id, reviewed_prediction)
                        st.success("Feedback enregistre.")
                        st.rerun()
                    except (ValueError, sqlite3.Error) as error:
                        st.error(f"Impossible d'enregistrer le feedback : {error}")

        st.divider()
        st.subheader("Export pour reentrainement")
        only_reviewed = st.checkbox(
            "Exporter uniquement les predictions ayant un feedback",
            value=True,
        )
        export_source = history.copy()
        if only_reviewed:
            export_source = export_source[
                export_source["reviewed_prediction"].notna()
            ]

        export_df = build_training_export(export_source)
        st.caption(f"{len(export_df)} ligne(s) prete(s) a exporter.")
        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Telecharger le CSV de reentrainement",
            data=csv_data,
            file_name="feedback_reentrainement.csv",
            mime="text/csv",
            disabled=export_df.empty,
        )
