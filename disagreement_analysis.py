"""Model disagreement analysis for uncertainty estimation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def build_prediction_table(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    label_encoder: Optional[object] = None,
) -> pd.DataFrame:
    """Collect predictions from each model into one table."""
    predictions = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test) # type: ignore
        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(y_pred.astype(int)) # type: ignore
        predictions[model_name] = y_pred

    return pd.DataFrame(predictions, index=X_test.index)


def disagreement_mask(prediction_table: pd.DataFrame) -> pd.Series:
    """Rows with at least one prediction mismatch across models."""
    return prediction_table.nunique(axis=1) > 1


def disagreement_percentage(disagreement_rows: pd.Series) -> float:
    """Percent of samples where models disagree."""
    if disagreement_rows.empty:
        return 0.0
    return float(disagreement_rows.mean() * 100.0)


def disagreement_by_grid_position(
    prediction_table_df: pd.DataFrame,
    X_test: pd.DataFrame,
    grid_col: str = "grid_position",
) -> pd.DataFrame:
    """Analyze grid positions with highest disagreement rates."""
    if grid_col not in X_test.columns:
        return pd.DataFrame(columns=[grid_col, "disagreement_pct", "samples"])

    mask = disagreement_mask(prediction_table_df)
    summary = pd.DataFrame(
        {
            grid_col: X_test.loc[prediction_table_df.index, grid_col],
            "is_disagreement": mask.astype(int),
        }
    )
    grouped = (
        summary.groupby(grid_col)["is_disagreement"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "disagreement_pct", "count": "samples"})
    )
    grouped["disagreement_pct"] = grouped["disagreement_pct"] * 100.0
    grouped = grouped.sort_values(["disagreement_pct", "samples"], ascending=[False, False]).reset_index(drop=True)
    return grouped


def run_disagreement_analysis(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    label_encoder: Optional[object] = None,
    grid_col: str = "grid_position",
) -> Dict[str, object]:
    """End-to-end disagreement analysis package."""
    preds = build_prediction_table(models=models, X_test=X_test, label_encoder=label_encoder)
    mask = disagreement_mask(preds)

    return {
        "prediction_table": preds,
        "disagreement_samples": preds[mask],
        "disagreement_percentage": disagreement_percentage(mask),
        "grid_position_disagreement": disagreement_by_grid_position(preds, X_test, grid_col=grid_col),
    }
