"""Feature engineering for F1 race outcome deviation modeling."""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd


def _time_sort_columns(dataframe: pd.DataFrame) -> List[str]:
    datetime_candidates = ["race_date", "date_start", "session_date"]
    sort_cols = []

    for col in datetime_candidates:
        if col in dataframe.columns:
            sort_cols.append(col)
            break

    for col in ["year", "meeting_key", "driver_number"]:
        if col in dataframe.columns and col not in sort_cols:
            sort_cols.append(col)

    return sort_cols


def _prepare_time_order(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    for col in ["race_date", "date_start", "session_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    sort_cols = _time_sort_columns(df)
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def add_race_number_in_season(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add race number index within each season."""
    df = _prepare_time_order(dataframe)

    if not {"year", "meeting_key"}.issubset(df.columns):
        df["race_number_in_season"] = np.arange(1, len(df) + 1)
        return df

    date_col = next((c for c in ["race_date", "date_start", "session_date"] if c in df.columns), None)
    race_keys = ["year", "meeting_key"]
    if date_col is not None:
        race_keys.append(date_col)

    unique_races = df[race_keys].drop_duplicates()
    sort_cols = [c for c in ["year", date_col, "meeting_key"] if c is not None and c in unique_races.columns]
    unique_races = unique_races.sort_values(sort_cols)
    unique_races["race_number_in_season"] = unique_races.groupby("year").cumcount() + 1

    df = df.merge(unique_races[["year", "meeting_key", "race_number_in_season"]], on=["year", "meeting_key"], how="left")
    return df


def add_rolling_mean_deviation(dataframe: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add leakage-safe rolling mean deviation from previous races only."""
    df = _prepare_time_order(dataframe)

    if not {"driver_number", "deviation"}.issubset(df.columns):
        df["rolling_mean_deviation"] = 0.0
        return df

    df["rolling_mean_deviation"] = (
        df.groupby("driver_number")["deviation"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        .fillna(0.0)
    )
    return df


def add_driver_dnf_rate(dataframe: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add leakage-safe rolling DNF rate per driver."""
    df = _prepare_time_order(dataframe)

    if "dnf" not in df.columns:
        if "finish_position" in df.columns:
            df["dnf"] = df["finish_position"].isna().astype(int)
        else:
            df["dnf"] = 0

    if "driver_number" not in df.columns:
        df["driver_dnf_rate"] = 0.0
        return df

    df["driver_dnf_rate"] = (
        df.groupby("driver_number")["dnf"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        .fillna(0.0)
    )
    return df


def add_constructor_reliability(dataframe: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add leakage-safe constructor reliability score (1 - rolling DNF rate)."""
    df = _prepare_time_order(dataframe)

    constructor_col = "constructor" if "constructor" in df.columns else "team_name" if "team_name" in df.columns else None
    if constructor_col is None:
        df["constructor_reliability"] = 1.0
        return df

    if "dnf" not in df.columns:
        if "finish_position" in df.columns:
            df["dnf"] = df["finish_position"].isna().astype(int)
        else:
            df["dnf"] = 0

    reliability = (
        df.groupby(constructor_col)["dnf"]
        .transform(lambda s: 1.0 - s.shift(1).rolling(window=window, min_periods=1).mean())
        .fillna(1.0)
    )
    df["constructor_reliability"] = reliability.clip(lower=0.0, upper=1.0)
    return df


def engineer_features(dataframe: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Apply all engineered features while preserving temporal causality."""
    df = _prepare_time_order(dataframe)
    df = add_race_number_in_season(df)
    df = add_rolling_mean_deviation(df, window=rolling_window)
    df = add_driver_dnf_rate(df, window=rolling_window)
    df = add_constructor_reliability(df, window=rolling_window)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineer features for F1 deviation dataset.")
    parser.add_argument("--input", type=str, default="f1_deviation_dataset_2022_2024.csv", help="Input CSV path.")
    parser.add_argument("--output", type=str, default="f1_deviation_features.csv", help="Output CSV path.")
    parser.add_argument("--rolling-window", type=int, default=5, help="Rolling window size for temporal features.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_df = pd.read_csv(args.input)
    featured_df = engineer_features(input_df, rolling_window=args.rolling_window)
    featured_df.to_csv(args.output, index=False)
    print(f"Feature-engineered dataset saved to {args.output} with shape: {featured_df.shape}")
