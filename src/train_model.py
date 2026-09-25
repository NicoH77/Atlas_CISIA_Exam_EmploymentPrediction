"""
train_model.py - MLflow avec Scikit-Learn
=================================================

Fonction d'entraînement avec MLflow pour le modèle de classification de retour à l'emploi.

À appeler depuis le notebook après la création :
- du pipeline
- des jeux X_train, X_test, y_train, y_test

Prérequis:
---------
1. rendre ce script importable depuis le notebook

Exécution:
---------
1. appeler la fonction train_model()
  

Interface MLflow:
----------------
Après exécution, consultez http://localhost:5000 pour voir :
- Les runs dans l'expérience
- Les datasets loggés (onglet Overview)
- Les métriques loggées 
- Les artefacts comparables (matrices de confusion)
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import matplotlib.pyplot as plt



from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.pipeline import Pipeline

from mlflow.models import infer_signature


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


def train_model(model, preprocessor, X_train, X_test, y_train, y_test, model_name="retour-emploi-multimodal", stage="Staging"):
    """
    Entraîne le GridSearchCV et log les résultats dans MLflow.
    
    Cette fonction démontre le logging complet avec MLflow :
    1. Paramètres : hyperparamètres du modèle
    2. Métriques : performances mesurées
    3. Tags : métadonnées descriptives
    4. Artefacts : fichiers (graphiques, rapports)
    5. Dataset : données d'entraînement (traçabilité)
    6. Modèle : modèle sérialisé réutilisable
    
    Args:
        search (GridSearchCV): Objet de recherche d'hyperparamètres configuré avec le pipeline
            de prétraitement et de modélisation.
        X_train (pd.DataFrame): Variables du jeu d'entraînement
        X_test (pd.DataFrame): Variables du jeu de test.
        y_train (pd.Series): Variable cible du jeu d'entraînement.
        y_test (pd.Series): Variable cible du jeu d'évaluation.
        model_name (str, optional): Nom du modèle dans le MLflow Model Registry.
            Par défaut : "retour_emploi_multimodal".
        stage (str, optional): Stage MLflow dans lequel enregistrer la version du modèle
            Par défaut : "Staging".

    Returns:
        sklearn.pipeline.Pipeline:
            Le meilleur modèle obtenu après recherche des
            hyperparamètres.

    """
    # ========================================================================
    # DÉMARRAGE D'UN RUN MLFLOW
    # ========================================================================
    # mlflow.start_run() crée un nouveau run dans l'expérience active
    # run_name : nom affiché dans l'interface (optionnel mais recommandé)
    # Le context manager (with) garantit que le run est fermé proprement
    scenario = "S1"
    
    with mlflow.start_run(run_name=f"{scenario}_{model_name}"):

        # ====================================================================
        # ÉTAPE 1 : ENTRAÎNEMENT ET TEST DU MODÈLE
        # ====================================================================
        # fit() entraîne le modèle sur les données d'entrainement
        # Après cette étape, le modèle peut faire des prédictions
        
        print(f"\n🔄 Entraînement: {model_name}")

        pipeline = Pipeline([("prep", preprocessor),("clf", model)])
        
        pipeline.fit(X_train, y_train)

        
        # Prédictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # ====================================================================
        # ÉTAPE 2 : CALCUL DES MÉTRIQUES
        # ====================================================================
        # Les métriques quantifient la performance du modèle
        # 
        # accuracy  : % de prédictions correctes
        # precision : parmi les positifs prédits, combien sont vrais positifs
        # recall    : parmi les vrais positifs, combien sont détectés
        # f1_score  : moyenne harmonique de precision et recall
        
        # Capture des métriques d'évaluation        
        # accuracy = accuracy_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        # auc = roc_auc_score(y_test, y_proba)

        metriques = {
           "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
           "recall_macro": recall_score(y_test, y_pred, average="macro"),
           "f1_macro": f1_score(y_test, y_pred, average="macro"),
           "recall_classe_2": recall_score(y_test, y_pred, labels=[2], average=None)[0]
        }
        
        # ====================================================================
        # ÉTAPE 3 : LOG MLFLOW
        # ====================================================================

        # Log des paramètres
        mlflow.log_params(pipeline.get_params())
        mlflow.log_param("model_type", type(model).__name__)
        
        # Log des métriques
        mlflow.log_metric("balanced_accuracy", metriques['balanced_accuracy'])
        mlflow.log_metric("recall_macro", metriques['recall_macro'])
        mlflow.log_metric("f1_macro", metriques['f1_macro'])
        mlflow.log_metric("recall_classe_2", metriques['recall_classe_2'])
        

        # =====================================================
        # ÉTAPE 4 : CLASSIFICATION REPORT
        # =====================================================
        # Capture et log du rapport de classification
        report = classification_report(
            y_test,
            y_pred
        )

        with open(
            "classification_report.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(report)

        mlflow.log_artifact("classification_report.txt")

        
        # ====================================================================
        # ÉTAPE 5 : MATRICE DE CONFUSION
        # ====================================================================
        # Capture et log de la matrice de confusion

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        # Plot
        fig, ax = plt.subplots(figsize=(5, 5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot(ax=ax)

        # Log de la matrice dans MLFLow
        mlflow.log_figure(fig, "confusion_matrix.png")

        plt.close()        

        # =====================================================
        # ÉTAPE 6 : TAGS
        # =====================================================

        mlflow.set_tags(
            {
                "project": "CISIA",
                "scenario": scenario,
                "task": "multiclass_classification",
                "author": "Nico-H",
            }
        )
        
        # =========================
        # ÉTAPE 7 : Enregistrement du modèle
        # =========================

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            serialization_format="pickle"
        )


        # ====================================================================
        # ÉTAPE 8 : AFFICHAGE DES RÉSULTATS
        # ====================================================================
        print(f"✅ Modèle {model_name} entraîné et logué avec succès dans MLflow.")
        print(f"   - Balanced-accuracy: {metriques['balanced_accuracy']:.4f}")
        print(f"   - Recall-macro: {metriques['recall_macro']:.4f}")
        print(f"   - F1-Macro: {metriques['f1_macro']:.4f}")
        print(f"   - Recall-classe-2: {metriques['recall_classe_2']:.4f}")        
        

        # =========================
        # Registry
        # =========================

        client = MlflowClient()

        model_version = mlflow.register_model(model_info.model_uri, model_name)

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )

        print("Run ID :", mlflow.active_run().info.run_id)
        print("Model version :", model_version.version)
        print("F1-Macro :", round(metriques['f1_macro'], 4))

        return pipeline

