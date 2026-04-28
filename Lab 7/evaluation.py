"""Evaluation utilities for multiclass deviation models."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute metrics and confusion matrices for each model."""
    rows = []
    confusion_matrices: Dict[str, np.ndarray] = {}
    predictions: Dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test) # type: ignore
        predictions[model_name] = y_pred

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        )
        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

    comparison_df = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    return comparison_df, confusion_matrices, predictions


def print_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    class_names: Iterable[str] | None = None,
) -> None:
    """Print confusion matrix per model in a readable format."""
    labels = list(class_names) if class_names is not None else None

    for model_name, matrix in confusion_matrices.items():
        print(f"\nConfusion Matrix - {model_name}")
        if labels is not None:
            matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
            print(matrix_df)
        else:
            print(matrix)
