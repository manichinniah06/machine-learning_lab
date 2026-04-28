"""Preprocessing utilities for F1 deviation classification."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def deviation_to_class(deviation: float) -> str:
    """Map deviation value to class label."""
    if deviation >= 2:
        return "Outperform"
    if deviation <= -2:
        return "Underperform"
    return "Neutral"


def ensure_required_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create standardized columns used downstream."""
    df = dataframe.copy()

    if "driver" not in df.columns:
        if "driver_name" in df.columns:
            df["driver"] = df["driver_name"]
        elif "driver_number" in df.columns:
            df["driver"] = df["driver_number"].astype(str)

    if "constructor" not in df.columns and "team_name" in df.columns:
        df["constructor"] = df["team_name"]

    if "circuit" not in df.columns:
        if "circuit_short_name" in df.columns:
            df["circuit"] = df["circuit_short_name"]
        elif "race_name" in df.columns:
            df["circuit"] = df["race_name"]

    for col in ["grid_position", "finish_position", "deviation", "year", "meeting_key", "driver_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "dnf" not in df.columns:
        df["dnf"] = df["finish_position"].isna().astype(int) if "finish_position" in df.columns else 0
    else:
        df["dnf"] = pd.to_numeric(df["dnf"], errors="coerce").fillna(0).astype(int)

    return df


def create_or_update_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure deviation and class labels are present and synchronized."""
    df = dataframe.copy()

    if "deviation" not in df.columns and {"grid_position", "finish_position"}.issubset(df.columns):
        df["deviation"] = df["grid_position"] - df["finish_position"]

    if "deviation" in df.columns:
        df["class_label"] = df["deviation"].apply(lambda x: deviation_to_class(x) if pd.notna(x) else np.nan)

    return df


def apply_dnf_strategy(dataframe: pd.DataFrame, strategy: str = "mark") -> pd.DataFrame:
    """Remove DNFs or mark them with fallback finish position."""
    df = dataframe.copy()
    strategy = strategy.lower().strip()

    if strategy not in {"remove", "mark"}:
        raise ValueError("dnf strategy must be one of {'remove', 'mark'}")

    if "dnf" not in df.columns:
        df["dnf"] = df["finish_position"].isna().astype(int)

    if strategy == "remove":
        df = df[df["dnf"] == 0].copy()
        return create_or_update_labels(df)

    if "finish_position" in df.columns:
        if "meeting_key" in df.columns and "grid_position" in df.columns:
            fallback_finish = df.groupby("meeting_key")["grid_position"].transform("max") + 1
        elif "grid_position" in df.columns:
            fallback_finish = df["grid_position"].max() + 1
        else:
            fallback_finish = np.nan

        df["finish_position"] = df["finish_position"].fillna(fallback_finish)

    return create_or_update_labels(df)


def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Handle categorical and numeric missing values."""
    df = dataframe.copy()

    categorical_defaults: Dict[str, str] = {
        "driver": "Unknown Driver",
        "constructor": "Unknown Constructor",
        "circuit": "Unknown Circuit",
    }
    for col, default_value in categorical_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_value).astype(str)

    for col in ["grid_position", "finish_position", "deviation", "year"]:
        if col in df.columns:
            median_value = df[col].median()
            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)

    return df


def get_feature_columns(dataframe: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return available categorical and numerical columns for modeling."""
    base_categorical = ["driver", "constructor", "circuit"]
    base_numerical = [
        "grid_position",
        "year",
        "race_number_in_season",
        "rolling_mean_deviation",
        "driver_dnf_rate",
        "constructor_reliability",
        "dnf",
    ]

    categorical_cols = [col for col in base_categorical if col in dataframe.columns]
    numerical_cols = [col for col in base_numerical if col in dataframe.columns]
    return categorical_cols, numerical_cols


def build_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    """Build a reusable column transformer for model pipelines."""
    # Support both old and new scikit-learn OneHotEncoder constructor signatures.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("categorical", encoder, categorical_cols),
            ("numerical", StandardScaler(), numerical_cols),
        ],
        remainder="drop",
    )


def encode_target(labels: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """Label-encode multiclass target labels."""
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(labels.astype(str))
    return encoded, label_encoder # type: ignore


def preprocess_for_modeling(dataframe: pd.DataFrame, dnf_strategy: str = "mark") -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Full preprocessing pass before model training."""
    df = ensure_required_columns(dataframe)
    df = apply_dnf_strategy(df, strategy=dnf_strategy)
    df = handle_missing_values(df)
    df = create_or_update_labels(df)
    df = df.dropna(subset=["class_label"]).reset_index(drop=True)

    categorical_cols, numerical_cols = get_feature_columns(df)
    if not categorical_cols and not numerical_cols:
        raise ValueError("No usable feature columns found after preprocessing.")

    return df, categorical_cols, numerical_cols
