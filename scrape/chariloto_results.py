"""Scraper for Chariloto race results and payouts."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from utils import (
    create_session,
    fetch_url,
    get_logger,
    normalize_whitespace,
    safe_to_numeric,
)

BASE_URL = "https://www.chariloto.com"
RESULT_PATH = "/keirin/result"
SOURCE = "chariloto"

RESULT_ENTRY_ALIASES: Dict[str, Iterable[str]] = {
    "finish_pos": ("着順", "順位", "着", "Rank"),
    "lane_no": ("車番", "枠番", "Lane", "枠番(車番)"),
    "rider_name": ("選手名", "氏名", "名前"),
    "rider_id": ("登番", "登録番号"),
    "time": ("上り", "タイム", "Time"),
    "odds": ("オッズ", "人気", "Odds"),
}

PAYOUT_ALIASES: Dict[str, Iterable[str]] = {
    "bet_type": ("式別", "賭式", "Bet"),
    "combination": ("組番", "組合せ", "番号"),
    "payout": ("払戻金", "払戻", "配当"),
    "popularity": ("人気", "人気順"),
    "unit": ("単位", "単価"),
    "n_tickets": ("的中", "票数", "口数"),
}

RESULT_ENTRY_COLUMNS = [
    "race_id",
    "date",
    "race_no",
    "lane_no",
    "rider_id",
    "rider_name",
    "finish_pos",
    "time",
    "odds",
    "bank_code",
    "source",
]

PAYOUT_COLUMNS = [
    "race_id",
    "bet_type",
    "combination",
    "payout",
    "popularity",
    "unit",
    "n_tickets",
    "source",
]


@dataclass
class RaceKey:
    date: str
    place_code: str
    race_no: int

    @property
    def race_id(self) -> str:
        return f"{self.date}CL{self.place_code}{self.race_no:02d}"

    @property
    def iso_date(self) -> str:
        return f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:8]}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_results_for_ids(
    race_ids: List[str],
    *,
    session=None,
    rate_limit: float = 0.5,
    retries: int = 3,
    timeout: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session = session or create_session(retries=retries, timeout=timeout)
    logger = get_logger(__name__)

    info_records: List[Dict[str, Optional[str]]] = []
    entry_frames: List[pd.DataFrame] = []
    payout_frames: List[pd.DataFrame] = []

    for race_id in sorted(set(race_ids)):
        key = _parse_race_id(race_id)
        params = {
            "date": key.date,
            "place": key.place_code,
            "race": f"{key.race_no:02d}",
        }
        url = urljoin(BASE_URL, RESULT_PATH)
        try:
            response = fetch_url(
                session,
                url,
                params=params,
                rate_limit=rate_limit,
                logger=logger,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to fetch race result %s: %s", race_id, exc)
            continue
        soup = BeautifulSoup(response.text, "lxml")
        meta = _parse_race_meta(soup, key)
        results_df = _parse_result_table(soup, key)
        payout_df = _parse_payout_table(soup, key)
        if results_df.empty:
            logger.warning("No result table parsed for %s", race_id)
            continue
        meta["field_size"] = int(results_df["lane_no"].count())
        info_records.append(meta)
        entry_frames.append(results_df)
        if not payout_df.empty:
            payout_frames.append(payout_df)

    info_df = pd.DataFrame(info_records)
    if not info_df.empty:
        info_df = info_df.reindex(
            columns=[
                "race_id",
                "date",
                "race_no",
                "stadium",
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
        )
    else:
        info_df = pd.DataFrame(
            columns=[
                "race_id",
                "date",
                "race_no",
                "stadium",
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
        )

    results_df = (
        pd.concat(entry_frames, ignore_index=True, sort=False)
        if entry_frames
        else pd.DataFrame(columns=RESULT_ENTRY_COLUMNS)
    )
    if not results_df.empty:
        results_df = results_df[RESULT_ENTRY_COLUMNS]

    payout_df = (
        pd.concat(payout_frames, ignore_index=True, sort=False)
        if payout_frames
        else pd.DataFrame(columns=PAYOUT_COLUMNS)
    )
    if not payout_df.empty:
        payout_df = payout_df[PAYOUT_COLUMNS]

    return info_df, results_df, payout_df


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_race_id(race_id: str) -> RaceKey:
    match = re.fullmatch(r"(\d{8})CL([A-Za-z0-9]+)(\d{2})", race_id)
    if not match:
        raise ValueError(f"Invalid race_id format: {race_id}")
    return RaceKey(date=match.group(1), place_code=match.group(2), race_no=int(match.group(3)))


def _parse_race_meta(soup: BeautifulSoup, key: RaceKey) -> Dict[str, Optional[str]]:
    meta: Dict[str, Optional[str]] = {
        "race_id": key.race_id,
        "date": key.iso_date,
        "race_no": key.race_no,
        "stadium": None,
        "race_name": None,
        "grade": None,
        "field_size": None,
        "line_count": None,
        "line_pattern": None,
        "start_time": None,
        "weather": None,
        "wind": None,
        "bank_code": key.place_code,
        "term": None,
    }

    header_text = normalize_whitespace(soup.get_text(" ", strip=True))
    header_match = re.search(r"(第\d+?回\s*)?(.*?)(\d{1,2}R)", header_text)
    if header_match:
        meta["race_name"] = header_match.group(2).strip()

    term = soup.select_one(".breadcrumb, .race-title, .event-title")
    if term:
        meta["term"] = normalize_whitespace(term.get_text(" ", strip=True))

    return meta


def _parse_result_table(soup: BeautifulSoup, key: RaceKey) -> pd.DataFrame:
    tables = _extract_tables(soup)
    for table in tables:
        df = _standardise_result_table(table, key)
        if not df.empty:
            return df
    return pd.DataFrame(columns=RESULT_ENTRY_COLUMNS)


def _parse_payout_table(soup: BeautifulSoup, key: RaceKey) -> pd.DataFrame:
    tables = _extract_tables(soup)
    for table in tables:
        df = _standardise_payout_table(table, key)
        if not df.empty:
            return df
    return pd.DataFrame(columns=PAYOUT_COLUMNS)


def _extract_tables(soup: BeautifulSoup) -> Iterable[pd.DataFrame]:
    html_content = str(soup)
    try:
        for table in pd.read_html(html_content, flavor="lxml"):
            yield table
    except ValueError:
        pass

    for table in soup.find_all("table"):
        headers = [normalize_whitespace(th.get_text(" ", strip=True)) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr"):
            cells = [normalize_whitespace(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if headers and rows:
            df = pd.DataFrame(rows[1:], columns=headers)
        elif rows:
            df = pd.DataFrame(rows)
        else:
            continue
        yield df


def _standardise_result_table(table: pd.DataFrame, key: RaceKey) -> pd.DataFrame:
    df = table.copy()
    df.columns = [normalize_whitespace(str(col)) for col in df.columns]

    rename_map = {}
    for target, candidates in RESULT_ENTRY_ALIASES.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = target
                break
    df = df.rename(columns=rename_map)

    if "finish_pos" not in df.columns or "lane_no" not in df.columns:
        return pd.DataFrame()

    df = df.assign(
        race_id=key.race_id,
        date=key.iso_date,
        race_no=key.race_no,
        bank_code=key.place_code,
        source=SOURCE,
    )

    for column in ("finish_pos", "lane_no", "odds"):
        if column in df.columns:
            df[column] = safe_to_numeric(df[column])

    if "rider_name" in df.columns:
        df["rider_name"] = df["rider_name"].map(lambda x: html.unescape(str(x)))

    for column in RESULT_ENTRY_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df[RESULT_ENTRY_COLUMNS]


def _standardise_payout_table(table: pd.DataFrame, key: RaceKey) -> pd.DataFrame:
    df = table.copy()
    df.columns = [normalize_whitespace(str(col)) for col in df.columns]

    rename_map = {}
    for target, candidates in PAYOUT_ALIASES.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = target
                break
    df = df.rename(columns=rename_map)

    if "bet_type" not in df.columns or "combination" not in df.columns:
        return pd.DataFrame()

    df = df.assign(
        race_id=key.race_id,
        source=SOURCE,
    )

    for column in ("payout", "popularity", "n_tickets"):
        if column in df.columns:
            df[column] = safe_to_numeric(df[column])

    for column in PAYOUT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df[PAYOUT_COLUMNS]
