"""Normalization utilities for preparing training datasets."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "race_id",
    "date",
    "race_no",
    "lane_no",
    "rider_name",
    "score",
    "bank_code",
    "source",
]

ALL_COLUMNS = [
    "race_id",
    "date",
    "race_no",
    "stadium",
    "race_name",
    "grade",
    "class",
    "lane_no",
    "rider_id",
    "rider_name",
    "age",
    "prefecture",
    "score",
    "style",
    "backs",
    "homes",
    "starts",
    "win_rate",
    "quinella_rate",
    "top3_rate",
    "kimarite_nige",
    "kimarite_makuri",
    "kimarite_sashi",
    "kimarite_mark",
    "finish_pos",
    "line_id",
    "line_pos",
    "gear",
    "bank_code",
    "source",
    "field_size",
    "line_count",
    "line_pattern",
    "start_time",
    "weather",
    "wind",
    "kaizai_no",
    "term",
]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = "" if column not in {"score", "age", "win_rate"} else pd.NA
    return df


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = ["score", "age", "win_rate", "finish_pos", "line_pos", "field_size"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["lane_no"] = df["lane_no"].astype(str)
    return df


def to_training_csv(
    out_path: str,
    info_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    payout_df: pd.DataFrame,
) -> None:
    """Normalize DataFrames into the canonical races.csv format."""

    if entry_df.empty:
        LOGGER.warning("Entry dataframe is empty; output will still be generated with headers")
        normalized = pd.DataFrame(columns=ALL_COLUMNS)
    else:
        entry_df = _ensure_columns(entry_df, REQUIRED_COLUMNS)
        entry_df = _ensure_columns(entry_df, ALL_COLUMNS)
        entry_df = _normalize_types(entry_df)

        if not info_df.empty:
            info_subset = info_df[["race_id", "race_name", "stadium"]].drop_duplicates()
            entry_df = entry_df.merge(info_subset, on="race_id", how="left", suffixes=("", "_info"))
            if "race_name_info" in entry_df.columns and "race_name" in entry_df.columns:
                entry_df["race_name"] = entry_df["race_name"].fillna(entry_df["race_name_info"])
            if "stadium_info" in entry_df.columns and "stadium" in entry_df.columns:
                entry_df["stadium"] = entry_df["stadium"].fillna(entry_df["stadium_info"])
            entry_df = entry_df.drop(columns=[col for col in entry_df.columns if col.endswith("_info")])

        if not payout_df.empty and "finish_pos" not in entry_df.columns:
            LOGGER.info("finish_pos not found in entries; attempting to merge from payouts")
            payout_subset = payout_df[["race_id"]].drop_duplicates()
            entry_df = entry_df.merge(payout_subset, on="race_id", how="left")

        normalized = entry_df[ALL_COLUMNS]

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(path, index=False, encoding="utf-8-sig")
    LOGGER.info("Training dataset written to %s", path)


__all__ = ["to_training_csv"]
