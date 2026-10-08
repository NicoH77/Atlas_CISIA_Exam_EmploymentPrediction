# Atlas CISIA - Employment Prediction

## Présentation

Atlas CISIA - Employment Prediction est un projet de classification supervisée développé dans le cadre de la certification CISIA (Concepteur Intégrateur de Solutions d'Intelligence Artificielle).

L'application vise à exploiter des données tabulaires et textuelles afin de prédire une catégorie de retour à l'emploi. Le projet couvre l'ensemble du cycle de vie d'une solution d'intelligence artificielle, depuis le développement du modèle jusqu'à son exposition au travers d'une API et d'une interface utilisateur.

Au-delà de l'entraînement du modèle, une attention particulière a été portée à l'industrialisation, à la traçabilité des expérimentations et à la mise à disposition de la solution dans un environnement exploitable.

---

## Objectif

L'objectif du projet est de construire une solution d'aide à la décision reposant sur le machine learning et capable de traiter des données hétérogènes :

- Données numériques
- Données catégorielles
- Données textuelles

Le projet intègre :

- Un pipeline de préparation des données
- Un moteur d'entraînement des modèles
- Un suivi des expérimentations avec MLflow
- Une API REST pour l'inférence et le réentraînement
- Une interface utilisateur Streamlit
- Une architecture conteneurisée avec Docker

---

## Fonctionnalités

### Data Science

- Préparation et transformation des données
- Gestion des variables numériques, catégorielles et textuelles
- Pipeline de prétraitement reproductible
- Entraînement et comparaison de plusieurs modèles de classification
- Sérialisation des modèles entraînés

### MLOps

- Suivi des expérimentations avec MLflow
- Gestion des artefacts de modèles
- Versionnage des exécutions
- Réentraînement du modèle depuis l'API

### API

- Service d'inférence REST
- Validation des données entrantes
- Journalisation des prédictions
- Vérification de l'état de santé de l'application

### Interface Utilisateur

- Formulaire de saisie des caractéristiques d'un usager
- Consultation du résultat d'une prédiction
- Historique des inférences réalisées

### Industrialisation

- Dockerisation des composants
- Orchestration avec Docker Compose
- Monitoring via Prometheus

---

## Architecture

```text
┌─────────────────┐
│    Streamlit    │
│ Interface Web   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     FastAPI     │
│   API REST ML   │
└────────┬────────┘
         │
         ├─────────────► Modèle entraîné
         │
         ▼
┌─────────────────┐
│     MLflow      │
│ Tracking Models │
└─────────────────┘
```

---

## Technologies utilisées

### Data & Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Joblib

### Backend

- FastAPI
- Pydantic

### MLOps

- MLflow

### Frontend

- Streamlit

### Infrastructure

- Docker
- Docker Compose
- Prometheus

### Qualité

- Pytest

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <repository-url>
cd Atlas_CISIA_Exam
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Endpoints de l'API

L'API est développée avec FastAPI et expose plusieurs services permettant l'exploitation du modèle.

### Vérification de l'état de santé

```http
GET /health
```

Permet de vérifier que l'API est disponible et opérationnelle.

---

### Informations sur le modèle

```http
GET /model-info
```

Retourne les informations disponibles concernant le modèle actuellement chargé.

---

### Prédiction

```http
POST /predict
```

Permet d'obtenir une prédiction à partir des données transmises dans la requête.

Exemple :

```json
{
  "age": 35,
  "niveau_diplome": "Bac+2",
  "anciennete_poste_ans": 5,
  "code_rome_vise": "M1805",
  "code_insee_commune": "92050",
  "est_allocataire": 1,
  "nationalite_hors_ue": 0,
  "synthese_entretien": "Texte libre décrivant la situation."
}
```

---

### Réentraînement du modèle

```http
POST /retrain
```

Permet de lancer un processus de réentraînement à partir d'un jeu de données fourni.

Cette route est destinée aux opérations d'administration et de maintenance du modèle.

---

## Structure du projet

```text
Atlas_CISIA_Exam/
│
├── app/
│   ├── api/                # API FastAPI et modèle déployé
│   ├── mlflow/             # Tracking et registre des modèles
│   ├── monitoring/         # Configuration Prometheus
│   └── ui/                 # Interface Streamlit
│
├── data/                   # Jeux de données
│
├── docs/                   # Documentation projet
│
├── notebooks/              # Analyses exploratoires et expérimentations
│
├── outputs/                # Modèles exportés
│
├── src/
│   ├── train_model.py      # Entraînement
│   ├── benchmark.py        # Comparaison de modèles
│   ├── evaluation.py       # Évaluation
│   └── config.py           # Configuration du projet
│
├── tests/
│   └── test_smoke.py       # Tests de validation
│
├── requirements.txt
├── requirements-ci.txt
├── pyproject.toml
└── README.md
```

---

## Monitoring et Observabilité

Le projet inclut plusieurs mécanismes de suivi :

- Journalisation des prédictions
- Historisation des inférences
- Suivi des expérimentations MLflow
- Supervision de l'application via Prometheus

---

## Améliorations futures

- Mise en place d'une chaîne CI/CD complète
- Renforcement de la couverture des tests
- Détection automatique de dérive des données
- Ajout de tableaux de bord de supervision avancés
- Déploiement sur une infrastructure de production

---

## Auteur

**NicoH**

Projet réalisé dans le cadre de la certification **CISIA – Concepteur Intégrateur de Solutions d'Intelligence Artificielle**.