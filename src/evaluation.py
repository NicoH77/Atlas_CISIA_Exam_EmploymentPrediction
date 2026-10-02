from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def evaluate_test_set(model, preprocessor, x_train, y_train, x_test, y_test):
    """
    Evaluation finale sur test set.
    """

    pipeline = Pipeline([("prep", preprocessor),("clf", model)])

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

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