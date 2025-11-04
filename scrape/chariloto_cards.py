"""Scraper for Chariloto race cards (pre-race entries)."""
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
RACECARD_PATH = "/keirin/racecard"
SOURCE = "chariloto"

ENTRY_COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "lane_no": ("車番", "枠番", "車番（枠番）", "Lane", "枠番(車番)"),
    "rider_name": ("選手名", "氏名", "名前", "name"),
    "class": ("級班", "級 班", "Class"),
    "age": ("年齢", "年令", "Age"),
    "prefecture": ("府県", "出身", "都道府県"),
    "score": ("競走得点", "得点", "Points"),
    "style": ("脚質", "脚質／得意", "脚質(得点)", "Style"),
    "backs": ("バック", "Backs"),
    "homes": ("ホーム", "Homes"),
    "starts": ("スタート", "Starts"),
    "win_rate": ("勝率", "1着", "Win%"),
    "quinella_rate": ("連対率", "2連対率", "連率"),
    "top3_rate": ("3連対率", "3連率", "入着率"),
    "gear": ("ギア", "ギヤ", "Gear"),
    "rider_id": ("登番", "登録番号", "選手番号"),
}

INFO_FIELD_CANDIDATES: Dict[str, Iterable[str]] = {
    "stadium": ("開催場", "場名", "バンク", "会場"),
    "race_name": ("レース名", "競走名", "タイトル"),
    "grade": ("グレード", "等級", "Grade"),
    "start_time": ("発走", "発走予定", "出走予定"),
    "weather": ("天候", "天気"),
    "wind": ("風速", "風力", "風"),
}

ENTRY_COLUMNS = [
    "race_id",
    "date",
    "race_no",
    "stadium",
    "track",
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
    "gear",
    "bank_code",
    "source",
]

INFO_COLUMNS = [
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


@dataclass
class RaceKey:
    date: str  # YYYYMMDD
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

def fetch_cards_for_ids(
    race_ids: List[str],
    *,
    session=None,
    rate_limit: float = 0.5,
    retries: int = 3,
    timeout: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch entry lists for the provided race identifiers."""

    session = session or create_session(retries=retries, timeout=timeout)
    logger = get_logger(__name__)

    info_records: List[Dict[str, Optional[str]]] = []
    entry_frames: List[pd.DataFrame] = []

    for race_id in sorted(set(race_ids)):
        key = _parse_race_id(race_id)
        params = {
            "date": key.date,
            "place": key.place_code,
            "race": f"{key.race_no:02d}",
        }
        url = urljoin(BASE_URL, RACECARD_PATH)
        try:
            response = fetch_url(
                session,
                url,
                params=params,
                rate_limit=rate_limit,
                logger=logger,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to fetch race card %s: %s", race_id, exc)
            continue
        soup = BeautifulSoup(response.text, "lxml")
        meta = _parse_race_meta(soup, key)
        entries = _parse_entry_table(soup, key, meta)
        if entries.empty:
            logger.warning("No entries parsed for %s", race_id)
            continue
        meta["field_size"] = int(entries["lane_no"].count())
        meta["line_count"] = meta.get("line_count")
        info_records.append(meta)
        entry_frames.append(entries)

    info_df = pd.DataFrame(info_records, columns=INFO_COLUMNS)
    entry_df = (
        pd.concat(entry_frames, ignore_index=True, sort=False)
        if entry_frames
        else pd.DataFrame(columns=ENTRY_COLUMNS)
    )
    if not entry_df.empty:
        entry_df = entry_df[ENTRY_COLUMNS]
    return info_df, entry_df


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
        "track": None,
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
    # Attempt to spot the race title.
    header_match = re.search(r"(第\d+?回\s*)?(.*?)(\d{1,2}R)", header_text)
    if header_match:
        meta.setdefault("race_name", header_match.group(2).strip())

    for label, candidates in INFO_FIELD_CANDIDATES.items():
        value = _find_field_value(soup, candidates)
        if value:
            meta[label] = value

    heading = soup.find(["h1", "h2", "h3"], string=re.compile("R", re.IGNORECASE))
    if heading:
        meta["race_name"] = normalize_whitespace(heading.get_text(" ", strip=True))

    # Term / series name often appears in breadcrumbs or subtitles.
    breadcrumb = soup.select_one(".breadcrumb, .race-title, .event-title")
    if breadcrumb:
        meta["term"] = normalize_whitespace(breadcrumb.get_text(" ", strip=True))

    return meta


def _parse_entry_table(
    soup: BeautifulSoup, key: RaceKey, meta: Dict[str, Optional[str]]
) -> pd.DataFrame:
    tables = _extract_tables(soup)
    for table in tables:
        df = _standardise_entry_table(table, key, meta)
        if not df.empty and {"rider_name", "lane_no"}.issubset(df.columns):
            return df
    return pd.DataFrame(columns=ENTRY_COLUMNS)


def _extract_tables(soup: BeautifulSoup) -> Iterable[pd.DataFrame]:
    html_content = str(soup)
    try:
        for table in pd.read_html(html_content, flavor="lxml"):
            yield table
    except ValueError:
        pass

    # BeautifulSoup fallback
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


def _standardise_entry_table(
    table: pd.DataFrame, key: RaceKey, meta: Dict[str, Optional[str]]
) -> pd.DataFrame:
    df = table.copy()
    df.columns = [normalize_whitespace(str(col)) for col in df.columns]

    rename_map = {}
    for target, candidates in ENTRY_COLUMN_ALIASES.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = target
                break
    df = df.rename(columns=rename_map)

    if "lane_no" not in df.columns or "rider_name" not in df.columns:
        return pd.DataFrame()

    df = df.assign(
        race_id=key.race_id,
        date=key.iso_date,
        race_no=key.race_no,
        stadium=meta.get("stadium"),
        track=meta.get("track"),
        race_name=meta.get("race_name"),
        grade=meta.get("grade"),
        bank_code=key.place_code,
        source=SOURCE,
    )

    for column in ("lane_no", "age", "score", "backs", "homes", "starts", "win_rate", "quinella_rate", "top3_rate", "gear"):
        if column in df.columns:
            df[column] = safe_to_numeric(df[column])

    if "rider_name" in df.columns:
        df["rider_name"] = df["rider_name"].map(lambda x: html.unescape(str(x)))

    for column in ENTRY_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[ENTRY_COLUMNS]


def _find_field_value(soup: BeautifulSoup, candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        label = soup.find(string=re.compile(candidate))
        if label:
            parent = label.parent
            if parent and parent.name in {"th", "dt"}:
                sibling = parent.find_next_sibling(["td", "dd"])
                if sibling:
                    return normalize_whitespace(sibling.get_text(" ", strip=True))
            elif hasattr(parent, "get_text"):
                return normalize_whitespace(parent.get_text(" ", strip=True))
    return None
