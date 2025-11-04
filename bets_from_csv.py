"""Generate simple betting slips from cards and prediction CSV files."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate betting suggestions from predictions")
    parser.add_argument("--cards", required=True, help="Path to cards.csv")
    parser.add_argument("--pred", required=True, help="Path to predictions CSV")
    parser.add_argument("--out", required=True, help="Output CSV file")
    parser.add_argument("--budget", type=float, default=10000.0, help="Total budget in yen")
    parser.add_argument(
        "--policy",
        choices=["flat", "proportional", "kelly"],
        default="flat",
        help="Budget allocation policy",
    )
    parser.add_argument(
        "--ev-th",
        type=float,
        default=1.01,
        help="Minimum expected value threshold (EV >= EV_TH)",
    )
    return parser.parse_args()


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _kelly_fraction(prob: float, odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    edge = odds * prob - 1
    denom = odds - 1
    if denom <= 0:
        return 0.0
    fraction = edge / denom
    return max(0.0, fraction)


def _prepare_pairs(df: pd.DataFrame, ev_threshold: float) -> pd.DataFrame:
    records: List[dict] = []
    for race_id, group in df.groupby("race_id"):
        group = group.sort_values("win_proba", ascending=False).reset_index(drop=True)
        if len(group) < 2:
            continue
        first = group.iloc[0]
        second = group.iloc[1]
        # 2連単の簡易確率（独立近似）
        pair_prob = float(first["win_proba"]) * float(second["win_proba"]) / max(
            1e-6, 1.0 - float(first["win_proba"]) + 1e-6
        )
        pair_prob = max(min(pair_prob, 1.0), 1e-6)
        expected_odds = 1.0 / pair_prob
        ev = expected_odds * pair_prob
        if ev < ev_threshold:
            continue
        records.append(
            {
                "race_id": race_id,
                "stadium": first.get("stadium", ""),
                "race_name": first.get("race_name", ""),
                "date": first.get("date", ""),
                "bet_type": "2連単",
                "first_lane": first["lane_no"],
                "first_rider": first.get("rider_name", ""),
                "second_lane": second["lane_no"],
                "second_rider": second.get("rider_name", ""),
                "first_win_proba": float(first["win_proba"]),
                "second_win_proba": float(second["win_proba"]),
                "pair_probability": pair_prob,
                "expected_odds": expected_odds,
                "combination": f"{first['lane_no']}-{second['lane_no']}",
            }
        )
    return pd.DataFrame(records)


def _allocate_budget(df: pd.DataFrame, budget: float, policy: str) -> pd.DataFrame:
    if df.empty:
        df["stake"] = []
        return df
    weights = np.ones(len(df))
    if policy == "proportional":
        weights = df["pair_probability"].to_numpy()
    elif policy == "kelly":
        weights = df.apply(lambda row: _kelly_fraction(row["pair_probability"], row["expected_odds"]), axis=1).to_numpy()
    if np.allclose(weights, 0):
        weights = np.ones(len(df))
    total = weights.sum()
    if total <= 0:
        weights = np.ones(len(df))
        total = weights.sum()
    stakes = budget * (weights / total)
    df["stake"] = np.round(stakes, 0)
    return df


def build_bets(
    merged_df: pd.DataFrame,
    budget: float,
    policy: str,
    ev_threshold: float,
) -> pd.DataFrame:
    """Public function for tests and CLI reuse."""

    pairs = _prepare_pairs(merged_df, ev_threshold)
    pairs = _allocate_budget(pairs, budget, policy)
    columns = [
        "race_id",
        "stadium",
        "race_name",
        "date",
        "bet_type",
        "first_lane",
        "first_rider",
        "second_lane",
        "second_rider",
        "first_win_proba",
        "second_win_proba",
        "pair_probability",
        "expected_odds",
        "combination",
        "stake",
    ]
    return pairs[columns] if not pairs.empty else pd.DataFrame(columns=columns)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cards_df = _load_csv(args.cards)
    preds_df = _load_csv(args.pred)
    merged = cards_df.merge(
        preds_df,
        on=["race_id", "lane_no"],
        how="inner",
        suffixes=("_card", ""),
    )
    result = build_bets(merged, args.budget, args.policy, args.ev_th)
    if result.empty:
        LOGGER.warning("No bets generated")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")
    LOGGER.info("Bets written to %s", args.out)


if __name__ == "__main__":
    main()
