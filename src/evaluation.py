from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    classification_report
)


def evaluate_test_set(model, preprocessor, X_train, y_train, X_test, y_test):
    """
    Evaluation finale sur test set.
    """

    pipeline = Pipeline([("prep", preprocessor),("clf", model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy":
            accuracy_score(y_test, y_pred),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_test,
                y_pred
            ),

        "f1_macro":
            f1_score(
                y_test,
                y_pred,
                average="macro"
            ),

        "recall_classe_2":
            recall_score(
                y_test,
                y_pred,
                labels=[2],
                average=None
            )[0]
    }

    return pipeline, y_pred, metrics