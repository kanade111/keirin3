"""Generate betting suggestions from cards and model predictions."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import get_logger, save_csv, setup_logging

BET_TYPE = "exacta"  # 2連単
DEFAULT_STAKE = 100
TOP_K = 3


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def estimate_pair_probability(first: pd.Series, second: pd.Series) -> float:
    p_first = float(first.get("p_win", 0))
    p_second = float(second.get("p_win", 0))
    if p_first <= 0 or p_second <= 0:
        return 0.0
    conditional_second = p_second / max(1e-6, 1 - p_first)
    return max(0.0, min(1.0, p_first * conditional_second))


def estimate_odds(probability: float) -> float:
    if probability <= 0:
        return 0.0
    base = 1.0 / probability
    return max(1.0, base * 0.85)


def generate_candidates(race_df: pd.DataFrame) -> List[Dict[str, float]]:
    race_df = race_df.sort_values("p_win", ascending=False).head(TOP_K)
    runners = race_df.to_dict("records")
    candidates = []
    for i, first in enumerate(runners):
        for j, second in enumerate(runners):
            if i == j:
                continue
            prob = estimate_pair_probability(pd.Series(first), pd.Series(second))
            odds = estimate_odds(prob)
            candidates.append(
                {
                    "lane_first": int(first.get("lane_no", 0)),
                    "lane_second": int(second.get("lane_no", 0)),
                    "p_pair": prob,
                    "odds": odds,
                    "runner_first": first,
                    "runner_second": second,
                }
            )
    candidates.sort(key=lambda x: x["p_pair"], reverse=True)
    return candidates[:TOP_K]


def allocate_budget(candidates: List[Dict[str, float]], budget: float, policy: str) -> List[float]:
    if not candidates:
        return []
    if policy == "flat":
        stakes = [DEFAULT_STAKE] * len(candidates)
        scale = min(1.0, budget / max(1, sum(stakes)))
        stakes = [stake * scale for stake in stakes]
    elif policy == "proportional":
        total_ev = sum(cand["ev"] for cand in candidates) or 1.0
        stakes = [budget * cand["ev"] / total_ev for cand in candidates]
    elif policy == "kelly":
        stakes = []
        for cand in candidates:
            p = cand["p_pair"]
            odds = cand["odds"]
            if odds <= 1:
                stakes.append(0.0)
                continue
            edge = (odds * p) - (1 - p)
            fraction = max(0.0, edge / (odds - 1))
            stakes.append(0.5 * fraction * budget)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    total = sum(stakes)
    if total == 0:
        return [0.0 for _ in stakes]
    if policy != "flat":
        stakes = [stake * budget / total for stake in stakes]
    # Round to nearest 10 and adjust residual
    stakes = [round(stake / 10) * 10 for stake in stakes]
    residual = budget - sum(stakes)
    if stakes and abs(residual) >= 10:
        adjustment = int(np.sign(residual) * 10)
        idx = np.argmax([cand["ev"] for cand in candidates])
        stakes[idx] = max(0, stakes[idx] + adjustment)
    return stakes


def build_bets(df: pd.DataFrame, budget: float, policy: str, ev_threshold: float) -> pd.DataFrame:
    bets: List[Dict[str, object]] = []
    for race_id, group in df.groupby("race_id"):
        race_name = group["race_name"].iloc[0]
        venue = group["stadium"].iloc[0]
        candidates = generate_candidates(group)
        for rank, cand in enumerate(candidates, start=1):
            ev = cand["p_pair"] * cand["odds"]
            if ev < ev_threshold:
                continue
            bets.append(
                {
                    "race_id": race_id,
                    "race_name": race_name,
                    "venue": venue,
                    "bet_type": BET_TYPE,
                    "combination": f"{cand['lane_first']}-{cand['lane_second']}",
                    "stake": 0.0,
                    "ev": ev,
                    "rank": rank,
                    "notes": f"p={cand['p_pair']:.3f},odds={cand['odds']:.2f}",
                    "p_pair": cand["p_pair"],
                    "odds": cand["odds"],
                }
            )

    if not bets:
        return pd.DataFrame(columns=["race_id", "race_name", "venue", "bet_type", "combination", "stake", "ev", "rank", "notes"])

    bets.sort(key=lambda x: (x["race_id"], x["rank"]))
    stakes = allocate_budget(bets, budget, policy)
    for bet, stake in zip(bets, stakes):
        bet["stake"] = max(0.0, stake)
        bet.pop("p_pair", None)
        bet.pop("odds", None)

    df = pd.DataFrame(bets)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chariloto betting suggestions")
    parser.add_argument("--cards", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--budget", type=float, default=10000)
    parser.add_argument("--policy", choices=["flat", "proportional", "kelly"], default="flat")
    parser.add_argument("--ev-th", type=float, default=1.01)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    logger = get_logger(__name__)

    cards_df = load_csv(args.cards)
    preds_df = load_csv(args.pred)
    if "p_win" not in preds_df.columns:
        raise ValueError("Predictions file must contain p_win column")

    merged = cards_df.merge(
        preds_df[["race_id", "lane_no", "p_win"]], on=["race_id", "lane_no"], how="inner"
    )

    bets_df = build_bets(merged, args.budget, args.policy, args.ev_th)
    save_csv(bets_df, args.out)
    logger.info("Generated %s bets", len(bets_df))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
