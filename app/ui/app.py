import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# =====================================================
# CONFIGURATION
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DB_PATH = ROOT_DIR / "app" / "ui" / "database" / "history.db"

API_URL = "http://localhost:8000/predict"


st.set_page_config(
    page_title="Prédiction retour à l'emploi",
    page_icon="📊",
    layout="wide",
)

# =====================================================
# BASE SQLITE
# =====================================================


def create_database():
    conn = sqlite3.connect(DB_PATH)

    # conn.execute("DROP TABLE IF EXISTS predictions")
    # conn.commit()
    
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            usager_id TEXT,
            prediction INTEGER,
            confidence REAL,
            proba_0 REAL,
            proba_1 REAL,
            proba_2 REAL
        )
        """
    )

    conn.commit()
    conn.close()




def save_prediction(
    usager_id,
    prediction,
    confidence,
    proba_0,
    proba_1,
    proba_2,
):
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            prediction_date,
            usager_id,
            prediction,
            confidence,
            proba_0,
            proba_1,
            proba_2
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            usager_id,
            prediction,
            confidence,
            proba_0,
            proba_1,
            proba_2,
        )
    )

    conn.commit()
    conn.close()


def load_history():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


create_database()

# =====================================================
# TITRE
# =====================================================

st.title("📊 Prédiction de retour à l'emploi")

st.markdown(
    """
    Cette application fournit une aide à la décision
    pour les conseillers de l'agence.
    """
)

# =====================================================
# ONGLETS
# =====================================================

tab_prediction, tab_history = st.tabs(
    [
        "Nouvelle prédiction",
        "Historique"
    ]
)

# =====================================================
# ONGLET PREDICTION
# =====================================================

with tab_prediction:

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            usager_id = st.text_input(
                "Identifiant usager"
            )

            age = st.number_input(
                "Âge",
                min_value=16,
                max_value=70
            )

            niveau_diplome = st.selectbox(
                "Niveau de diplôme",
                [
                    "Sans diplôme",
                    "Bac",
                    "Bac+2",
                    "Bac+5"
                ]
            )

            anciennete_poste_ans = st.number_input(
                "Ancienneté du dernier poste",
                min_value=0,
                max_value=50
            )

        with col2:

            code_rome_vise = st.text_input(
                "Code ROME visé"
            )

            code_insee_commune = st.text_input(
                "Code INSEE commune"
            )

            est_allocataire = st.selectbox(
                "Allocataire",
                [0, 1]
            )

            nationalite_hors_ue = st.selectbox(
                "Nationalité hors UE",
                [0, 1]
            )

        synthese_entretien = st.text_area(
            "Synthèse de l'entretien",
            height=200
        )

        submitted = st.form_submit_button(
            "🔍 Lancer la prédiction"
        )

    if submitted:

        payload = {
            "usager_id": usager_id,
            "age": age,
            "niveau_diplome": niveau_diplome,
            "anciennete_poste_ans": anciennete_poste_ans,
            "code_rome_vise": code_rome_vise,
            "code_insee_commune": code_insee_commune,
            "est_allocataire": est_allocataire,
            "nationalite_hors_ue": nationalite_hors_ue,
            "synthese_entretien": synthese_entretien,
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=10,
            )

            response.raise_for_status()

            result = response.json()

            prediction = result["prediction"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            labels = {
                0: "Retour rapide (0-6 mois)",
                1: "Retour moyen (6-12 mois)",
                2: "Risque de chômage longue durée (>12 mois)",
            }

            st.success(
                f"Prédiction : {labels.get(prediction, prediction)}"
            )

            st.metric(
                label="Confiance du modèle",
                value=f"{confidence:.1%}"
            )


            if result.get("human_review_required", False):
                st.warning(
                    """
                    Cette prédiction constitue une aide
                    à la décision.

                    La décision finale reste sous la
                    responsabilité du conseiller.
                    """
            )

            st.subheader("Probabilités par classe")

            proba_df = pd.DataFrame(
                {
                    "Classe": [
                        "Retour rapide",
                        "Retour moyen",
                        "Risque longue durée",
                    ],
                    "Probabilité": [
                        probabilities["0"],
                        probabilities["1"],
                        probabilities["2"],
                    ]
                }
            )

            st.bar_chart(
                proba_df.set_index("Classe")
            )

            st.write(
                {
                    "Retour rapide":
                        f"{probabilities['0']:.1%}",
                    "Retour moyen":
                        f"{probabilities['1']:.1%}",
                    "Risque longue durée":
                        f"{probabilities['2']:.1%}",
                }
            )
    
            save_prediction(
                usager_id=usager_id,
                prediction=prediction,
                confidence=confidence,
                proba_0=probabilities["0"],
                proba_1=probabilities["1"],
                proba_2=probabilities["2"],
            )

        except Exception as e:

            st.error(
                f"Erreur lors de l'appel API : {e}"
            )

# =====================================================
# ONGLET HISTORIQUE
# =====================================================

with tab_history:

    st.subheader("Historique des inférences")

    history = load_history()

    if len(history) > 0:

        st.dataframe(
            history,
            use_container_width=True
        )

        csv = history.to_csv(
            index=False
        )

        st.download_button(
            label="Télécharger CSV",
            data=csv,
            file_name="historique_predictions.csv",
            mime="text/csv",
        )

    else:

        st.info(
            "Aucune prédiction enregistrée."
        )