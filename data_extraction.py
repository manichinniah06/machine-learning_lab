"""Data extraction utilities for Formula 1 deviation analysis.

This module fetches sessions, identifies qualifying/race pairs, extracts grid
and finish positions, merges driver metadata, and computes deviation labels.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://api.openf1.org/v1"


class OpenF1Client:
    """Small API client with retry and rate-limit-safe pacing."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        sleep_seconds: float = 0.35,
        timeout_seconds: int = 30,
        max_retries: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sleep_seconds = sleep_seconds
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

        retries = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            backoff_factor=1.2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch JSON data from a single OpenF1 endpoint."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()

        payload = response.json()
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        if isinstance(payload, list):
            return payload
        return [payload]


def deviation_to_class(deviation: float) -> str:
    """Map numeric deviation to the project class labels."""
    if deviation >= 2:
        return "Outperform"
    if deviation <= -2:
        return "Underperform"
    return "Neutral"


def fetch_sessions(client: OpenF1Client, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch all sessions in the configured year range."""
    params = {"year>=": start_year, "year<=": end_year}
    sessions = client.get("sessions", params=params)
    sessions_df = pd.DataFrame(sessions)

    if sessions_df.empty:
        return sessions_df

    if "date_start" in sessions_df.columns:
        sessions_df["date_start"] = pd.to_datetime(sessions_df["date_start"], errors="coerce")

    return sessions_df


def identify_qualifying_race_pairs(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Identify qualifying/race session key pairs by meeting."""
    if sessions_df.empty:
        return pd.DataFrame()

    if "session_name" not in sessions_df.columns:
        raise ValueError("sessions dataframe must include 'session_name'.")

    working = sessions_df.copy()
    working["session_name_norm"] = working["session_name"].astype(str).str.strip().str.lower()

    qualifying = working[working["session_name_norm"] == "qualifying"].copy()
    race = working[working["session_name_norm"] == "race"].copy()

    if qualifying.empty or race.empty:
        return pd.DataFrame()

    sort_cols = [col for col in ["year", "meeting_key", "date_start"] if col in working.columns]
    key_cols = [col for col in ["year", "meeting_key"] if col in working.columns]

    if not key_cols:
        raise ValueError("sessions dataframe must include 'year' and/or 'meeting_key'.")

    if sort_cols:
        qualifying = qualifying.sort_values(sort_cols)
        race = race.sort_values(sort_cols)

    qualifying = qualifying.drop_duplicates(subset=key_cols, keep="last")
    race = race.drop_duplicates(subset=key_cols, keep="last")

    paired = qualifying.merge(race, on=key_cols, suffixes=("_qualifying", "_race"), how="inner")

    paired = paired.sort_values(key_cols).reset_index(drop=True)
    return paired


def fetch_final_positions(client: OpenF1Client, session_key: int) -> pd.DataFrame:
    """Fetch last available position per driver in a given session."""
    rows = client.get("position", params={"session_key": int(session_key)})
    positions = pd.DataFrame(rows)

    if positions.empty:
        return pd.DataFrame(columns=["driver_number", "position"])

    required = {"driver_number", "position"}
    if not required.issubset(positions.columns):
        return pd.DataFrame(columns=["driver_number", "position"])

    if "date" in positions.columns:
        positions["date"] = pd.to_datetime(positions["date"], errors="coerce")
        positions = positions.sort_values("date")

    final_snapshot = positions.groupby("driver_number", as_index=False).last()
    return final_snapshot[["driver_number", "position"]]


def fetch_driver_metadata(client: OpenF1Client, session_key: int) -> pd.DataFrame:
    """Fetch driver metadata for a session (name and constructor)."""
    rows = client.get("drivers", params={"session_key": int(session_key)})
    drivers = pd.DataFrame(rows)

    if drivers.empty:
        return pd.DataFrame(columns=["driver_number", "driver_name", "driver_code", "constructor"])

    rename_map = {
        "team_name": "constructor",
        "full_name": "driver_name",
        "name_acronym": "driver_code",
    }
    drivers = drivers.rename(columns=rename_map)

    for col in ["driver_number", "driver_name", "driver_code", "constructor"]:
        if col not in drivers.columns:
            drivers[col] = np.nan

    return drivers[["driver_number", "driver_name", "driver_code", "constructor"]].drop_duplicates(
        subset=["driver_number"]
    )


def _pick_value(row: pd.Series, candidates: List[str]) -> Any:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            return row[candidate]
    return np.nan


def build_deviation_dataset(
    start_year: int = 2022,
    end_year: int = 2024,
    sleep_seconds: float = 0.35,
) -> pd.DataFrame:
    """Create the full deviation dataset from OpenF1 API endpoints."""
    client = OpenF1Client(sleep_seconds=sleep_seconds)

    sessions_df = fetch_sessions(client, start_year=start_year, end_year=end_year)
    pairs_df = identify_qualifying_race_pairs(sessions_df)

    if pairs_df.empty:
        return pd.DataFrame()

    all_races: List[pd.DataFrame] = []

    for _, row in pairs_df.iterrows():
        qual_key = _pick_value(row, ["session_key_qualifying", "session_key_x"])
        race_key = _pick_value(row, ["session_key_race", "session_key_y"])

        if pd.isna(qual_key) or pd.isna(race_key):
            continue

        try:
            grid_df = fetch_final_positions(client, int(qual_key)).rename(columns={"position": "grid_position"})
            finish_df = fetch_final_positions(client, int(race_key)).rename(columns={"position": "finish_position"})
            drivers_df = fetch_driver_metadata(client, int(race_key))
        except requests.RequestException:
            # Skip only the problematic meeting and continue extraction.
            continue

        if grid_df.empty:
            continue

        race_df = grid_df.merge(finish_df, on="driver_number", how="left")
        race_df = race_df.merge(drivers_df, on="driver_number", how="left")

        race_df["year"] = _pick_value(row, ["year"])
        race_df["meeting_key"] = _pick_value(row, ["meeting_key"])
        race_df["qualifying_session_key"] = int(qual_key)
        race_df["race_session_key"] = int(race_key)
        race_df["race_name"] = _pick_value(row, ["meeting_name_race", "meeting_name_qualifying", "meeting_name"])
        race_df["circuit"] = _pick_value(
            row,
            ["circuit_short_name_race", "circuit_short_name_qualifying", "circuit_short_name"],
        )
        race_df["country_name"] = _pick_value(row, ["country_name_race", "country_name_qualifying", "country_name"])
        race_df["race_date"] = _pick_value(row, ["date_start_race", "date_start_qualifying", "date_start"])

        race_df["dnf"] = race_df["finish_position"].isna().astype(int)
        race_df["deviation"] = race_df["grid_position"] - race_df["finish_position"]
        race_df["class_label"] = race_df["deviation"].apply(
            lambda x: deviation_to_class(x) if pd.notna(x) else np.nan
        )

        all_races.append(race_df)

    if not all_races:
        return pd.DataFrame()

    dataset = pd.concat(all_races, ignore_index=True)

    numeric_cols = [
        "year",
        "meeting_key",
        "driver_number",
        "grid_position",
        "finish_position",
        "deviation",
    ]
    for col in numeric_cols:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    if "race_date" in dataset.columns:
        dataset["race_date"] = pd.to_datetime(dataset["race_date"], errors="coerce")

    sort_cols = [col for col in ["year", "race_date", "meeting_key", "driver_number"] if col in dataset.columns]
    if sort_cols:
        dataset = dataset.sort_values(sort_cols).reset_index(drop=True)

    return dataset


def save_dataset(dataframe: pd.DataFrame, output_path: str) -> None:
    """Persist extracted dataset to CSV."""
    dataframe.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract F1 deviation dataset from OpenF1 API.")
    parser.add_argument("--start-year", type=int, default=2022, help="Start year (inclusive).")
    parser.add_argument("--end-year", type=int, default=2024, help="End year (inclusive).")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.35,
        help="Delay between API requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="f1_deviation_dataset_2022_2024.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extracted_df = build_deviation_dataset(
        start_year=args.start_year,
        end_year=args.end_year,
        sleep_seconds=args.sleep_seconds,
    )
    save_dataset(extracted_df, args.output)
    print(f"Dataset saved to {args.output} with shape: {extracted_df.shape}")
