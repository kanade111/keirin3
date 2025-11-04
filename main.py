"""CLI entry point for Chariloto data fetch, training, and prediction."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from chariloto_ids import fetch_race_ids_for_date
from compat_chariloto import fetch_results_for_ids
from model import Model
from scrape.normalize import to_training_csv

LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _split_race_ids(text: Optional[str]) -> List[str]:
    if not text:
        return []
    parts = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    return parts


def _normalize_probabilities(entries: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    if entries.empty:
        return np.array([])
    if probs.size != len(entries):
        LOGGER.warning(
            "Probability count mismatch (expected %s, got %s)",
            len(entries),
            probs.size,
        )
        return np.array([])
    series = pd.Series(probs, index=entries.index, dtype=float)
    normalized = np.zeros(len(entries), dtype=float)
    race_ids = (
        entries.get("race_id")
        if "race_id" in entries.columns
        else pd.Series(["" for _ in entries.index], index=entries.index)
    )
    if isinstance(race_ids, pd.Series):
        race_ids = race_ids.fillna("")
    grouped = race_ids.groupby(race_ids, sort=False) if isinstance(race_ids, pd.Series) else []
    any_group = False
    for _, idxs in grouped:
        any_group = True
        indices = idxs.index if isinstance(idxs, pd.Series) else idxs
        weights = series.loc[indices].astype(float)
        total = weights.sum()
        if total <= 0:
            values = np.full(len(indices), 1.0 / len(indices), dtype=float)
        else:
            values = (weights / total).to_numpy()
        positions = entries.index.get_indexer(indices)
        valid_mask = positions >= 0
        if not np.all(valid_mask):
            positions = positions[valid_mask]
            values = values[valid_mask]
        normalized[positions] = values
    if not any_group or not np.isfinite(normalized).all() or normalized.sum() <= 0:
        normalized = np.full(len(entries), 1.0 / len(entries), dtype=float)
    return normalized


def _prepare_cards(info_df: pd.DataFrame, entry_df: pd.DataFrame) -> pd.DataFrame:
    cards = entry_df.copy()
    if not info_df.empty and "race_id" in info_df.columns:
        meta = info_df[["race_id", "race_name", "stadium"]].drop_duplicates()
        cards = cards.merge(meta, on="race_id", how="left", suffixes=("", "_info"))
        for column in ("race_name", "stadium"):
            info_col = f"{column}_info"
            if info_col in cards.columns:
                cards[column] = cards[column].fillna(cards[info_col])
        cards = cards.drop(columns=[col for col in cards.columns if col.endswith("_info")])

    for column in ["race_id", "lane_no", "rider_name", "score", "stadium", "race_no", "date", "bank_code"]:
        if column not in cards.columns:
            cards[column] = "" if column not in {"score", "race_no"} else pd.NA
    cards["lane_no"] = cards["lane_no"].astype(str)
    return cards


def _resolve_race_ids(
    date: str,
    race_ids_arg: Optional[str],
    timeout: float,
    retries: int,
    rate_limit: float,
) -> List[str]:
    race_ids = _split_race_ids(race_ids_arg)
    if race_ids:
        return race_ids
    LOGGER.info("Race IDs not provided; attempting schedule discovery")
    discovered = fetch_race_ids_for_date(
        date,
        timeout=timeout,
        retries=retries,
        rate_limit=rate_limit,
    )
    if not discovered:
        LOGGER.warning("No race IDs discovered for %s", date)
    return discovered


# ---------------------------------------------------------------------------
# Fetch command
# ---------------------------------------------------------------------------

def _cmd_fetch(args: argparse.Namespace) -> None:
    date = args.date
    race_ids = _resolve_race_ids(date, args.race_ids, args.timeout, args.retries, args.rate_limit)
    info_df, entry_df, payout_df = fetch_results_for_ids(
        race_ids,
        timeout=args.timeout,
        retries=args.retries,
        rate_limit=args.rate_limit,
    )

    base_dir = Path(args.out)
    date_dir = base_dir / date
    date_dir.mkdir(parents=True, exist_ok=True)

    info_path = date_dir / "info.csv"
    entry_path = date_dir / "entries.csv"
    payout_path = date_dir / "payouts.csv"
    info_df.to_csv(info_path, index=False, encoding="utf-8-sig")
    entry_df.to_csv(entry_path, index=False, encoding="utf-8-sig")
    payout_df.to_csv(payout_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Raw data exported to %s", date_dir)

    races_path = date_dir / "races.csv"
    to_training_csv(str(races_path), info_df, entry_df, payout_df)


# ---------------------------------------------------------------------------
# Train command
# ---------------------------------------------------------------------------

def _cmd_train(args: argparse.Namespace) -> None:
    model = Model.train_from_csv(args.races)
    model.save(args.out)


# ---------------------------------------------------------------------------
# Today command
# ---------------------------------------------------------------------------

def _fallback_probabilities(cards_df: pd.DataFrame) -> np.ndarray:
    if cards_df.empty:
        return np.array([])
    scores = pd.to_numeric(cards_df.get("score"), errors="coerce").fillna(0.0)
    cards_df = cards_df.copy()
    cards_df["_score"] = scores
    weights: List[float] = []
    for _, group in cards_df.groupby("race_id"):
        total = group["_score"].sum()
        if total <= 0:
            weights.extend([1.0 / len(group)] * len(group))
        else:
            weights.extend((group["_score"] / total).tolist())
    return np.array(weights)


def _cmd_today(args: argparse.Namespace) -> None:
    date = args.date or dt.date.today().strftime("%Y-%m-%d")
    race_ids = _resolve_race_ids(date, args.race_ids, args.timeout, args.retries, args.rate_limit)
    info_df, entry_df, _ = fetch_results_for_ids(
        race_ids,
        timeout=args.timeout,
        retries=args.retries,
        rate_limit=args.rate_limit,
    )

    out_dir = Path(args.out) / date
    out_dir.mkdir(parents=True, exist_ok=True)

    cards_path = out_dir / "cards.csv"
    cards_df = _prepare_cards(info_df, entry_df)
    cards_df.to_csv(cards_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Cards saved to %s", cards_path)

    predictions_path = out_dir / f"predictions_{date}.csv"
    try:
        model = Model.load(args.model)
        probabilities = model.predict_proba(cards_df)
        source = "model"
    except Exception as exc:  # pragma: no cover - fallback for runtime issues
        LOGGER.error("Model load/predict failed (%s); falling back to score-based probabilities", exc)
        probabilities = _fallback_probabilities(cards_df)
        source = "fallback"

    if probabilities.size == 0:
        probabilities = _fallback_probabilities(cards_df)
        source = "fallback"

    normalized = _normalize_probabilities(cards_df, probabilities)
    if normalized.size == 0:
        normalized = _fallback_probabilities(cards_df)
        source = "fallback"

    required_cols = ["race_id", "lane_no", "rider_name", "race_name", "stadium"]
    for column in required_cols:
        if column not in cards_df.columns:
            cards_df[column] = ""
    predictions = cards_df[required_cols].copy()
    predictions["date"] = cards_df.get("date", date)
    predictions["bank_code"] = cards_df.get("bank_code", "")
    predictions["win_proba"] = normalized
    predictions["prob_source"] = source
    predictions["expected_odds"] = predictions["win_proba"].apply(lambda p: float("inf") if p <= 0 else 1.0 / p)

    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Predictions written to %s", predictions_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chariloto data pipeline")
    parser.add_argument("command", choices=["fetch", "train", "today"], help="Subcommand to run")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--rate-limit", type=float, default=0.6)

    parser.add_argument("--date", required=False, help="Target date YYYY-MM-DD")
    parser.add_argument("--race-ids", required=False, help="Comma separated race IDs")
    parser.add_argument("--out", required=False, help="Output directory")
    parser.add_argument("--races", required=False, help="Training races CSV path")
    parser.add_argument("--model", required=False, help="Model directory or file path")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    if args.command == "fetch":
        if not args.date or not args.out:
            parser.error("fetch requires --date and --out")
        _cmd_fetch(args)
    elif args.command == "train":
        if not args.races or not args.out:
            parser.error("train requires --races and --out")
        _cmd_train(args)
    elif args.command == "today":
        if not args.model or not args.out:
            parser.error("today requires --model and --out")
        _cmd_today(args)
    else:  # pragma: no cover - argparse ensures command validity
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
