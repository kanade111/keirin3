"""Utilities for converting scraped data into training-ready tables."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from utils import get_logger, save_csv, safe_to_numeric

TRAINING_COLUMNS = [
    "race_id",
    "date",
    "race_no",
    "stadium",
    "track",
    "title",
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

KIMARITE_COLUMNS = {
    "kimarite_nige": ("逃げ", "ニゲ", "Nige"),
    "kimarite_makuri": ("捲り", "マクリ", "Makuri"),
    "kimarite_sashi": ("差し", "差", "Sashi"),
    "kimarite_mark": ("マーク", "Mark"),
}


def to_training_csv(
    out_path: Path,
    info_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    payout_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join the scraped tables into the canonical training CSV."""

    logger = get_logger(__name__)

    if entry_df.empty:
        logger.warning("Entry dataframe is empty; nothing to normalise")
        empty_df = pd.DataFrame(columns=TRAINING_COLUMNS)
        save_csv(empty_df, Path(out_path))
        return empty_df

    merged = entry_df.copy()

    # Merge race-level metadata
    info_columns = [
        "race_id",
        "stadium",
        "track",
        "race_name",
        "grade",
        "field_size",
        "line_count",
        "line_pattern",
        "start_time",
        "weather",
        "wind",
        "bank_code",
        "term",
    ]
    if not info_df.empty:
        info_subset = info_df.copy()
        missing_cols = [col for col in info_columns if col not in info_subset.columns]
        for col in missing_cols:
            info_subset[col] = None
        merged = merged.merge(info_subset[info_columns], on="race_id", how="left")

    # Derive title/kaizai_no placeholders
    merged["title"] = merged.get("term")
    merged["kaizai_no"] = None

    # Normalise kimarite columns if they exist in entry_df (from results).
    for target, candidates in KIMARITE_COLUMNS.items():
        if target not in merged.columns:
            for candidate in candidates:
                if candidate in merged.columns:
                    merged[target] = safe_to_numeric(merged[candidate])
                    break
            else:
                merged[target] = None
        else:
            merged[target] = safe_to_numeric(merged[target])

    numeric_columns = [
        "lane_no",
        "age",
        "score",
        "backs",
        "homes",
        "starts",
        "win_rate",
        "quinella_rate",
        "top3_rate",
        "finish_pos",
        "line_pos",
        "gear",
        "field_size",
        "line_count",
    ]

    for column in numeric_columns:
        if column in merged.columns:
            merged[column] = safe_to_numeric(merged[column])
        else:
            merged[column] = None

    merged["source"] = merged.get("source", "chariloto")

    for column in TRAINING_COLUMNS:
        if column not in merged.columns:
            merged[column] = None

    ordered = merged[TRAINING_COLUMNS]
    save_csv(ordered, Path(out_path))
    logger.info("Normalised training rows: %s", len(ordered))
    return ordered
