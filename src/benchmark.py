import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


def evaluate_model_cv(model, preprocessor, x, y, cv, scoring):
    """
    Exécute une validation croisée sur un pipeline complet.

    Parameters
    ----------
    model        : Estimateur sklearn
    preprocessor :  Transformer sklearn
    x            : Features
    y            : Cible
    cv           : Objet de validation croisée
    scoring      : Dictionnaire de métriques

    Returns
    -------
    dict
    """

    pipeline = Pipeline([("prep", preprocessor),("clf", model)])

    scores = cross_validate(estimator=pipeline, X=x, y=y, cv=cv, scoring=scoring, n_jobs=-1)

    return {
            "f1_macro": round(scores["test_f1_macro"].mean(), 3),
            "écart-type": round(scores["test_f1_macro"].std(), 3),
            "recall_macro": round(scores["test_recall_macro"].mean(), 3),
            "recall classe 2": round(scores["test_recall_classe_2"].mean(), 3),
            "bal. acc.": round(scores["test_balanced_accuracy"].mean(), 3),
            "temps fit (s)": round(scores["fit_time"].mean(), 3),
            # Temps de prédiction moyen calculé sur les folds CV
            "temps inf. (s)": round(scores["score_time"].mean(), 4)
            # "Score Métier": 0.4 * f1_macro + 0.6 * recall_classe_2
    }


def run_benchmark(models, scenarios, x_train, y_train, cv, scoring, build_preprocessor):
    """
    Benchmark de plusieurs modèles sur plusieurs scénarios.
    """

    results = []

    for scenario_name, scenario in scenarios.items():

        preprocessor = build_preprocessor(**scenario)

        for model_name, model in models.items():

            metrics = evaluate_model_cv(
                model=model,
                preprocessor=preprocessor,
                x=x_train,
                y=y_train,
                cv=cv,
                scoring=scoring
            )

            metrics["scen"] = scenario_name
            metrics["modele"] = model_name

            results.append(metrics)

    return pd.DataFrame(results)

