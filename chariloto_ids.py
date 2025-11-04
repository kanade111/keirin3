"""Utility to derive Chariloto race IDs from schedule pages."""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from typing import Iterable, List, Optional, Set

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

SCHEDULE_URLS = [
    "https://www.chariloto.com/keirin/schedule?date={date}",
    "https://www.chariloto.com/keirin/schedule/keirin?date={date}",
    "https://www.chariloto.com/keirin/racelist?date={date_dash}",
]


def _normalize_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")


def _create_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; keirin-scraper/1.0)",
        "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.6,en;q=0.4",
    })
    session.timeout = timeout
    return session


def _fetch_html(session: requests.Session, url: str, timeout: float, retries: int) -> Optional[str]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                LOGGER.info("URL returned 404: %s", url)
                return None
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.text
        except requests.RequestException as exc:  # pragma: no cover
            last_exc = exc
            LOGGER.warning("Fetch error (attempt %s/%s): %s", attempt, retries, exc)
            time.sleep(min(2 ** attempt, 5))
    if last_exc:
        LOGGER.error("Failed to fetch %s: %s", url, last_exc)
    return None


def _extract_bank_codes(html_text: str) -> Set[str]:
    soup = BeautifulSoup(html_text, "lxml")
    codes: Set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        for pattern in [r"bank=(\d{2,3})", r"place=(\d{2,3})", r"stadium=(\d{2,3})"]:
            match = re.search(pattern, href)
            if match:
                codes.add(match.group(1).zfill(2))
        text = anchor.get_text(" ")
        match_text = re.search(r"(\d{2,3})", text)
        if match_text:
            codes.add(match_text.group(1).zfill(2))
    return codes


def _build_race_ids_for_bank(date: str, bank: str, max_races: int = 12) -> List[str]:
    date_raw = date.replace("-", "")
    return [f"{date_raw}CL{bank}{race:02d}" for race in range(1, max_races + 1)]


def _all_bank_codes() -> List[str]:
    return [f"{bank:02d}" for bank in range(1, 100)]


def _fallback_bank_race_ids(date: str, max_races: int = 12) -> List[str]:
    LOGGER.info("Falling back to exhaustive bank enumeration for %s", date)
    race_ids: List[str] = []
    for bank in _all_bank_codes():
        race_ids.extend(_build_race_ids_for_bank(date, bank, max_races=max_races))
    return race_ids


def fetch_race_ids_for_date(
    date: str,
    timeout: float = 10.0,
    retries: int = 3,
    rate_limit: float = 0.6,
) -> List[str]:
    normalized_date = _normalize_date(date)
    date_compact = normalized_date.replace("-", "")
    session = _create_session(timeout)
    race_ids: List[str] = []

    for index, url_template in enumerate(SCHEDULE_URLS):
        if index:
            time.sleep(max(rate_limit, 0.0))
        url = url_template.format(date=date_compact, date_dash=normalized_date)
        LOGGER.info("Fetching schedule: %s", url)
        html_text = _fetch_html(session, url, timeout, retries)
        if not html_text:
            continue
        bank_codes = _extract_bank_codes(html_text)
        if not bank_codes:
            LOGGER.info("No bank codes detected from %s", url)
            continue
        for bank in sorted(bank_codes):
            race_ids.extend(_build_race_ids_for_bank(normalized_date, bank))
        break

    if not race_ids:
        LOGGER.warning("Unable to discover race IDs for %s; using fallback", normalized_date)
        race_ids = _fallback_bank_race_ids(normalized_date)
    return race_ids


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch race IDs for a given date")
    parser.add_argument("date", help="Date in YYYY-MM-DD or YYYYMMDD format")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--rate-limit", type=float, default=0.6)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    race_ids = fetch_race_ids_for_date(
        args.date,
        timeout=args.timeout,
        retries=args.retries,
        rate_limit=args.rate_limit,
    )
    output = ",".join(race_ids)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
