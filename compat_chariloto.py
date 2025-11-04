"""Compatibility layer for fetching and normalizing data from Chariloto."""
from __future__ import annotations

import html
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

CHARILOTO_URL = "https://www.chariloto.com/keirin/results/{bank}/{date}"

LOGGER = logging.getLogger(__name__)

ENTRY_KEYWORDS = {"車番", "車 番", "枠番", "枠 番", "選手", "選手名", "級班", "競走得点", "得点"}
PAYOUT_KEYWORDS = {"払戻", "払戻金", "配当", "人気", "組番", "組合せ", "オッズ"}


@dataclass(frozen=True)
class MeetingKey:
    """開催単位のキー。"""

    date: str  # YYYY-MM-DD
    bank_code: str  # zero padded string


def _normalize_text(value: object) -> str:
    """HTML解除＋NFKC正規化を行い、空白を圧縮する。"""

    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_digits(value: str) -> Optional[str]:
    match = re.search(r"(\d+)", value or "")
    return match.group(1) if match else None


def _to_date_str(raw: str) -> str:
    if re.fullmatch(r"\d{8}", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def _parse_race_id(race_id: str) -> Optional[Tuple[str, str, Optional[str]]]:
    match = re.fullmatch(r"(\d{8})CL(\d{2,3})(\d{2})", race_id)
    if not match:
        return None
    date_raw, bank, race = match.groups()
    return _to_date_str(date_raw), bank, race


def _build_race_id(date: str, bank: str, race_no: Optional[int]) -> str:
    if not race_no:
        return ""
    date_raw = date.replace("-", "")
    return f"{date_raw}CL{bank}{int(race_no):02d}"


def _create_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; keirin-bot/1.0; +https://example.com/bot)",
        "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.6,en;q=0.4",
    })
    session.timeout = timeout
    return session


def _get_with_retries(
    session: requests.Session, url: str, timeout: float, retries: int
) -> Optional[requests.Response]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                LOGGER.warning("URL not found (404): %s", url)
                return None
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response
        except requests.RequestException as exc:  # pragma: no cover - network errors
            last_exc = exc
            LOGGER.warning("Request error (attempt %s/%s): %s", attempt, retries, exc)
            time.sleep(min(2 ** attempt, 5))
    if last_exc:
        LOGGER.error("Failed to fetch %s: %s", url, last_exc)
    return None


def _collect_tables(html_text: str) -> Tuple[List[pd.DataFrame], BeautifulSoup]:
    soup = BeautifulSoup(html_text, "lxml")
    tables: List[pd.DataFrame] = []
    try:
        tables = pd.read_html(html_text, flavor="lxml")
    except ValueError:
        tables = []
    if not tables:
        for table in soup.find_all("table"):
            try:
                df_list = pd.read_html(str(table), flavor="lxml")
            except ValueError:
                continue
            for df in df_list:
                tables.append(df)
    return tables, soup


def _classify_table(columns: Sequence[str], keywords: Sequence[str]) -> bool:
    joined = " ".join(_normalize_text(col) for col in columns)
    for keyword in keywords:
        if keyword in joined:
            return True
    return False


def _find_heading_texts(soup: BeautifulSoup) -> List[str]:
    headings: List[str] = []
    current = ""
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "table"]):
        if element.name == "table":
            headings.append(_normalize_text(current))
        else:
            text = _normalize_text(element.get_text(" "))
            if text:
                current = text
    return headings


def _derive_race_no(heading: str, df: pd.DataFrame, fallback: int) -> int:
    candidates: List[int] = []
    if heading:
        match = re.search(r"(\d{1,2})\s*[RＲレ]\b", heading)
        if match:
            candidates.append(int(match.group(1)))
    for col in df.columns:
        col_norm = _normalize_text(col)
        match = re.search(r"(\d{1,2})\s*[RＲレ]\b", col_norm)
        if match:
            candidates.append(int(match.group(1)))
    if "レース" in df.columns:
        race_series = pd.to_numeric(df["レース"], errors="coerce").dropna()
        if not race_series.empty:
            candidates.append(int(race_series.iloc[0]))
    if candidates:
        return candidates[0]
    return fallback


def _clean_lane(value: object, fallback: int) -> str:
    lane = _normalize_text(value)
    if not lane:
        return str(fallback)
    digits = _extract_digits(lane)
    if digits:
        return digits
    return lane


def _clean_rider_name(value: object) -> str:
    name = _normalize_text(value)
    name = re.sub(r"[（(].*?[)）]", "", name)
    return name.strip()


def _normalize_entry_table(
    df: pd.DataFrame,
    heading: str,
    key: MeetingKey,
    default_race_no: int,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_text(col) for col in df.columns]

    race_no = _derive_race_no(heading, df, default_race_no)
    race_id = _build_race_id(key.date, key.bank_code, race_no)

    # 列名マッピング
    column_map: Dict[str, str] = {}
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col for keyword in {"車番", "枠番"}):
            column_map[col] = "lane_no"
        elif "選手" in col_lower or "氏名" in col_lower:
            column_map[col] = "rider_name"
        elif "級班" in col_lower or "級" in col_lower:
            column_map[col] = "class"
        elif "競走得点" in col or "得点" in col_lower:
            column_map[col] = "score"
        elif "年齢" in col_lower:
            column_map[col] = "age"
        elif "府県" in col_lower or "出身" in col_lower:
            column_map[col] = "prefecture"
        elif "着" in col_lower and "着順" not in column_map.values():
            column_map[col] = "finish_pos"
        elif "着順" in col_lower:
            column_map[col] = "finish_pos"
        elif "決まり手" in col_lower:
            column_map[col] = "kimarite"
        elif "ライン" in col_lower:
            column_map[col] = "line_id"
        elif "選手評価" in col_lower or "勝率" in col_lower:
            column_map[col] = "win_rate"
    df = df.rename(columns=column_map)

    if "lane_no" not in df.columns:
        df["lane_no"] = [str(i + 1) for i in range(len(df))]
    else:
        df["lane_no"] = [
            _clean_lane(value, idx + 1) for idx, value in enumerate(df["lane_no"].tolist())
        ]

    if "rider_name" not in df.columns:
        df["rider_name"] = ""
    else:
        df["rider_name"] = df["rider_name"].map(_clean_rider_name)

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    result = pd.DataFrame(
        {
            "race_id": race_id,
            "date": key.date,
            "bank_code": key.bank_code,
            "race_no": race_no,
            "lane_no": df["lane_no"],
            "rider_name": df["rider_name"],
            "score": df.get("score"),
            "class": df.get("class"),
            "age": df.get("age"),
            "prefecture": df.get("prefecture"),
            "finish_pos": df.get("finish_pos"),
            "kimarite": df.get("kimarite"),
            "line_id": df.get("line_id"),
            "win_rate": df.get("win_rate"),
            "source": "chariloto",
        }
    )
    result["race_name"] = heading
    return result


def _normalize_payout_table(
    df: pd.DataFrame,
    heading: str,
    key: MeetingKey,
    default_race_no: int,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_text(col) for col in df.columns]
    race_no = _derive_race_no(heading, df, default_race_no)
    race_id = _build_race_id(key.date, key.bank_code, race_no)
    df["race_id"] = race_id
    df["race_no"] = race_no
    df["date"] = key.date
    df["bank_code"] = key.bank_code
    df["source"] = "chariloto"
    return df


def _extract_info_records(
    key: MeetingKey,
    headings: Sequence[str],
    entry_tables: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for idx, table in enumerate(entry_tables):
        heading = headings[idx] if idx < len(headings) else ""
        race_no = _derive_race_no(heading, table, idx + 1)
        race_id = _build_race_id(key.date, key.bank_code, race_no)
        records.append(
            {
                "race_id": race_id,
                "race_no": race_no,
                "date": key.date,
                "bank_code": key.bank_code,
                "race_name": heading,
                "source": "chariloto",
            }
        )
    return pd.DataFrame(records)


def fetch_results_for_ids(
    race_ids: Iterable[str],
    timeout: float = 10.0,
    retries: int = 3,
    rate_limit: float = 0.6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charilotoの結果ページから情報を取得し正規化する。"""

    race_ids = [rid for rid in race_ids if rid]
    if not race_ids:
        LOGGER.warning("No race IDs provided")
        empty = pd.DataFrame()
        return empty, empty, empty

    key_to_race_ids: Dict[MeetingKey, List[str]] = {}
    for rid in race_ids:
        parsed = _parse_race_id(rid)
        if not parsed:
            LOGGER.warning("Invalid race_id format: %s", rid)
            continue
        date, bank, _ = parsed
        key = MeetingKey(date=date, bank_code=bank)
        key_to_race_ids.setdefault(key, []).append(rid)

    session = _create_session(timeout)

    info_frames: List[pd.DataFrame] = []
    entry_frames: List[pd.DataFrame] = []
    payout_frames: List[pd.DataFrame] = []

    for index, (key, key_race_ids) in enumerate(sorted(key_to_race_ids.items(), key=lambda x: x[0])):
        url = CHARILOTO_URL.format(bank=key.bank_code, date=key.date.replace("-", ""))
        if index:
            time.sleep(max(rate_limit, 0.0))
        LOGGER.info("Fetching %s (%s races)", url, len(key_race_ids))
        response = _get_with_retries(session, url, timeout, retries)
        if response is None:
            LOGGER.warning("Skipping meeting %s due to fetch failure", key)
            continue

        tables, soup = _collect_tables(response.text)
        if not tables:
            LOGGER.warning("No tables detected for %s", url)
            continue

        headings = _find_heading_texts(soup)
        meeting_entry_tables: List[pd.DataFrame] = []

        for idx, table in enumerate(tables):
            heading = headings[idx] if idx < len(headings) else ""
            columns = [str(col) for col in table.columns]
            if _classify_table(columns, ENTRY_KEYWORDS):
                normalized = _normalize_entry_table(table, heading, key, idx + 1)
                meeting_entry_tables.append(table)
                entry_frames.append(normalized)
            elif _classify_table(columns, PAYOUT_KEYWORDS):
                payout_frames.append(_normalize_payout_table(table, heading, key, idx + 1))

        if meeting_entry_tables:
            info_frames.append(_extract_info_records(key, headings, meeting_entry_tables))
        else:
            LOGGER.warning("No entry tables extracted for %s", url)

    info_df = pd.concat(info_frames, ignore_index=True) if info_frames else pd.DataFrame()
    entry_df = pd.concat(entry_frames, ignore_index=True) if entry_frames else pd.DataFrame()
    payout_df = pd.concat(payout_frames, ignore_index=True) if payout_frames else pd.DataFrame()

    return info_df, entry_df, payout_df


__all__ = [
    "CHARILOTO_URL",
    "MeetingKey",
    "fetch_results_for_ids",
]
