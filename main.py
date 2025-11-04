"""CLI entry point for the Chariloto end-to-end workflow."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from bets_from_csv import build_bets
from model import Model
from providers.chariloto import CharilotoProvider
from scrape.chariloto_cards import fetch_cards_for_ids as fetch_cards
from scrape.chariloto_results import fetch_results_for_ids as fetch_results
from scrape.normalize import to_training_csv
from utils import create_session, get_logger, save_csv, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chariloto data pipeline")
    parser.add_argument("command", choices=["fetch", "train", "today", "bets"], help="Subcommand to run")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--rate-limit", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)

    # fetch/today specific
    parser.add_argument("--from", dest="from_date")
    parser.add_argument("--to", dest="to_date")
    parser.add_argument("--date")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--races", type=Path)
    parser.add_argument("--budget", type=float, default=10000)
    parser.add_argument("--policy", choices=["flat", "proportional", "kelly"], default="flat")
    parser.add_argument("--ev-th", type=float, default=1.01)
    parser.add_argument("--cards", type=Path)
    parser.add_argument("--pred", type=Path)
    parser.add_argument("--midnight-only", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_fetch(args: argparse.Namespace) -> None:
    if not args.out:
        raise ValueError("--out is required for fetch")
    logger = get_logger(__name__)
    session = create_session(retries=args.retries, timeout=args.timeout)
    provider = CharilotoProvider(session=session, rate_limit=args.rate_limit, retries=args.retries, timeout=args.timeout)

    dates = list(_iter_dates(args))
    if not dates:
        raise ValueError("Specify --date or --from/--to")

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        logger.info("Processing %s", date_str)
        race_ids = provider.list_race_ids_for_date(date_str)
        if not race_ids:
            logger.info("No races scheduled for %s", date_str)
            continue

        out_dir = (args.out / date.strftime("%Y%m%d"))
        out_dir.mkdir(parents=True, exist_ok=True)

        cards_info, cards_entries = fetch_cards(
            race_ids,
            session=session,
            rate_limit=args.rate_limit,
            retries=args.retries,
            timeout=args.timeout,
        )
        results_info, results_entries, payouts = fetch_results(
            race_ids,
            session=session,
            rate_limit=args.rate_limit,
            retries=args.retries,
            timeout=args.timeout,
        )

        combined_info = _combine_info([cards_info, results_info])
        combined_entries = _combine_entries(cards_entries, results_entries)

        save_csv(cards_info, out_dir / "cards_info.csv")
        save_csv(cards_entries, out_dir / "cards_entries.csv")
        save_csv(results_info, out_dir / "results_info.csv")
        save_csv(results_entries, out_dir / "results_entries.csv")
        save_csv(payouts, out_dir / "payouts.csv")

        races_path = out_dir / "races.csv"
        to_training_csv(races_path, combined_info, combined_entries, payouts)


def cmd_train(args: argparse.Namespace) -> None:
    if not args.races or not args.out:
        raise ValueError("--races and --out are required for train")
    model = Model.train_from_csv(str(args.races))
    model.save(args.out)


def cmd_today(args: argparse.Namespace) -> None:
    if not args.model or not args.out:
        raise ValueError("--model and --out are required for today")

    logger = get_logger(__name__)
    session = create_session(retries=args.retries, timeout=args.timeout)
    provider = CharilotoProvider(session=session, rate_limit=args.rate_limit, retries=args.retries, timeout=args.timeout)

    date = dt.datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else dt.date.today()
    date_str = date.strftime("%Y-%m-%d")
    logger.info("Fetching cards for %s", date_str)

    if args.midnight_only:
        race_ids = provider.list_midnight_race_ids_for_date(date_str)
    else:
        race_ids = provider.list_race_ids_for_date(date_str)

    if not race_ids:
        logger.warning("No races found for %s", date_str)
        return

    cards_info, cards_entries = fetch_cards(
        race_ids,
        session=session,
        rate_limit=args.rate_limit,
        retries=args.retries,
        timeout=args.timeout,
    )

    out_dir = args.out / date.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(cards_info, out_dir / "cards_info.csv")
    save_csv(cards_entries, out_dir / "cards.csv")

    model = Model.load(args.model)
    probs = model.predict_proba(cards_entries)
    predictions = cards_entries[["race_id", "lane_no", "rider_name", "race_name", "stadium"]].copy()
    predictions["p_win"] = probs
    pred_path = out_dir / f"predictions_{date_str}.csv"
    save_csv(predictions, pred_path)
    logger.info("Predictions written to %s", pred_path)


def cmd_bets(args: argparse.Namespace) -> None:
    if not args.cards or not args.pred or not args.out:
        raise ValueError("--cards, --pred and --out are required for bets")
    cards_df = pd.read_csv(args.cards, encoding="utf-8-sig")
    preds_df = pd.read_csv(args.pred, encoding="utf-8-sig")
    merged = cards_df.merge(
        preds_df[["race_id", "lane_no", "p_win"]], on=["race_id", "lane_no"], how="inner"
    )
    bets_df = build_bets(merged, args.budget, args.policy, args.ev_th)
    save_csv(bets_df, args.out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_dates(args: argparse.Namespace) -> Iterable[dt.date]:
    if args.date:
        yield dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        return
    if args.from_date and args.to_date:
        start = dt.datetime.strptime(args.from_date, "%Y-%m-%d").date()
        end = dt.datetime.strptime(args.to_date, "%Y-%m-%d").date()
        delta = dt.timedelta(days=1)
        current = start
        while current <= end:
            yield current
            current += delta


def _combine_info(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in dfs if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["race_id"]).drop_duplicates("race_id", keep="last")
    return combined


def _combine_entries(cards_entries: pd.DataFrame, results_entries: pd.DataFrame) -> pd.DataFrame:
    if cards_entries is None or cards_entries.empty:
        return results_entries
    merged = cards_entries.copy()
    if results_entries is not None and not results_entries.empty:
        merged = merged.merge(
            results_entries[["race_id", "lane_no", "finish_pos"]],
            on=["race_id", "lane_no"],
            how="left",
        )
    return merged


COMMAND_HANDLERS = {
    "fetch": cmd_fetch,
    "train": cmd_train,
    "today": cmd_today,
    "bets": cmd_bets,
}


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        raise ValueError(f"Unknown command: {args.command}")
    handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
