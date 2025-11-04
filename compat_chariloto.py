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

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    LOG.addHandler(handler)
LOG.propagate = False

_LAST_FETCH_TS: float = 0.0

ENTRY_KEYWORDS = {"車番", "車 番", "枠番", "枠 番", "選手", "選手名", "級班", "競走得点", "得点"}
PAYOUT_KEYWORDS = {"払戻", "払戻金", "配当", "人気", "組番", "組合せ", "オッズ"}

ENTRY_COLUMNS = [
    "race_id",
    "date",
    "bank_code",
    "race_no",
    "race_name",
    "stadium",
    "lane_no",
    "rider_id",
    "rider_name",
    "age",
    "prefecture",
    "score",
    "class",
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
    "field_size",
    "line_count",
    "line_pattern",
    "start_time",
    "weather",
    "wind",
    "source",
]

INFO_COLUMNS = [
    "race_id",
    "date",
    "bank_code",
    "race_no",
    "race_name",
    "stadium",
    "grade",
    "term",
    "start_time",
    "weather",
    "wind",
    "field_size",
    "line_count",
    "line_pattern",
    "source",
]

PAYOUT_COLUMNS = [
    "race_id",
    "race_no",
    "date",
    "bank_code",
    "source",
]


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


def _parse_key_from_race_id(race_id: str) -> Optional[MeetingKey]:
    match = re.fullmatch(r"(\d{8})CL(\d{2,3})(\d{2})", race_id)
    if not match:
        return None
    date_raw, bank, _ = match.groups()
    date = _to_date_str(date_raw)
    bank_code = bank.zfill(2)
    return MeetingKey(date=date, bank_code=bank_code)


def _parse_race_id(race_id: str) -> Optional[Tuple[MeetingKey, Optional[int]]]:
    match = re.fullmatch(r"(\d{8})CL(\d{2,3})(\d{2})", race_id)
    if not match:
        return None
    date_raw, bank, race = match.groups()
    key = MeetingKey(date=_to_date_str(date_raw), bank_code=bank.zfill(2))
    race_no = int(race) if race.isdigit() else None
    return key, race_no


def _build_race_id(date: str, bank: str, race_no: Optional[int]) -> str:
    if not race_no:
        return ""
    date_raw = date.replace("-", "")
    return f"{date_raw}CL{bank}{int(race_no):02d}"


def _normalize_date_hint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = _normalize_text(value)
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return _to_date_str(digits[:8])
    return text


def _create_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; keirin-bot/1.0; +https://example.com/bot)",
            "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.6,en;q=0.4",
        }
    )
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
                LOG.warning("URL not found (404): %s", url)
                return None
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response
        except requests.RequestException as exc:  # pragma: no cover - network errors
            last_exc = exc
            LOG.warning("Request error (attempt %s/%s): %s", attempt, retries, exc)
            time.sleep(min(2 ** attempt, 5))
    if last_exc:
        LOG.error("Failed to fetch %s: %s", url, last_exc)
    return None


def _fetch_html(
    date: str,
    bank: str,
    timeout: float,
    retries: int,
    rate_limit: float,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """Fetch HTML for a given meeting with retry and rate limiting."""

    global _LAST_FETCH_TS

    session = session or _create_session(timeout)
    url = CHARILOTO_URL.format(bank=bank, date=date.replace("-", ""))

    if rate_limit > 0:
        elapsed = time.time() - _LAST_FETCH_TS
        if elapsed < rate_limit:
            time.sleep(rate_limit - elapsed)

    response = _get_with_retries(session, url, timeout, retries)
    if response is None:
        return None

    _LAST_FETCH_TS = time.time()
    return response.text


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


def _split_kimarite(value: str) -> Dict[str, int]:
    text = _normalize_text(value)
    mapping = {
        "kimarite_nige": 0,
        "kimarite_makuri": 0,
        "kimarite_sashi": 0,
        "kimarite_mark": 0,
    }
    keywords = {
        "逃": "kimarite_nige",
        "捲": "kimarite_makuri",
        "まく": "kimarite_makuri",
        "差": "kimarite_sashi",
        "マーク": "kimarite_mark",
    }
    for key, column in keywords.items():
        if key in text:
            mapping[column] = 1
    if text and all(value == 0 for value in mapping.values()):
        mapping["kimarite_mark"] = 1 if text else 0
    return mapping


def _normalize_entry_table(
    df: pd.DataFrame,
    heading: str,
    key: MeetingKey,
    default_race_no: int,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_text(col) for col in df.columns]

    heading_norm = _normalize_text(heading)
    race_no = _derive_race_no(heading_norm, df, default_race_no)
    race_id = _build_race_id(key.date, key.bank_code, race_no)

    column_map: Dict[str, str] = {}
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col for keyword in {"車番", "枠番"}):
            column_map[col] = "lane_no"
        elif "選手" in col_lower or "氏名" in col_lower:
            column_map[col] = "rider_name"
        elif "登録番号" in col_lower or "選手番号" in col_lower:
            column_map[col] = "rider_id"
        elif "級班" in col_lower or "級" in col_lower:
            column_map[col] = "class"
        elif "競走得点" in col or "得点" in col_lower:
            column_map[col] = "score"
        elif "脚質" in col_lower or "脚 質" in col:
            column_map[col] = "style"
        elif "年齢" in col_lower:
            column_map[col] = "age"
        elif "府県" in col_lower or "出身" in col_lower:
            column_map[col] = "prefecture"
        elif "バック" in col_lower:
            column_map[col] = "backs"
        elif "ホーム" in col_lower:
            column_map[col] = "homes"
        elif "スタート" in col_lower or "Ｓ" in col:
            column_map[col] = "starts"
        elif "勝率" in col_lower:
            column_map[col] = "win_rate"
        elif "連対" in col_lower:
            column_map[col] = "quinella_rate"
        elif "３連対" in col_lower or "3連" in col_lower:
            column_map[col] = "top3_rate"
        elif "決まり手" in col_lower:
            column_map[col] = "kimarite"
        elif "逃げ" in col_lower:
            column_map[col] = "kimarite_nige"
        elif "捲り" in col_lower or "まくり" in col_lower:
            column_map[col] = "kimarite_makuri"
        elif "差し" in col_lower:
            column_map[col] = "kimarite_sashi"
        elif "マーク" in col_lower:
            column_map[col] = "kimarite_mark"
        elif "ライン" in col_lower and "位置" not in col_lower:
            column_map[col] = "line_id"
        elif "ライン位置" in col_lower or "位置" in col_lower:
            column_map[col] = "line_pos"
        elif "ギア" in col_lower:
            column_map[col] = "gear"
        elif "天候" in col_lower or "天気" in col_lower:
            column_map[col] = "weather"
        elif "風" in col_lower:
            column_map[col] = "wind"
        elif "発走" in col_lower or "開始" in col_lower:
            column_map[col] = "start_time"
        elif "競輪場" in col_lower or "会場" in col_lower or "track" in col_lower:
            column_map[col] = "stadium"
        elif "出走" in col_lower and "表" not in col_lower:
            column_map[col] = "field_size"
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

    numeric_columns = [
        "score",
        "age",
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
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    kimarite_map = {key: [0] * len(df) for key in [
        "kimarite_nige",
        "kimarite_makuri",
        "kimarite_sashi",
        "kimarite_mark",
    ]}
    if "kimarite" in df.columns:
        for idx, value in enumerate(df["kimarite"]):
            split = _split_kimarite(value)
            for key, indicator in split.items():
                kimarite_map[key][idx] = indicator
    for key in kimarite_map:
        if key in df.columns:
            continue
        df[key] = kimarite_map[key]

    data: Dict[str, List[object]] = {}
    length = len(df)
    for col in ENTRY_COLUMNS:
        if col in df.columns:
            data[col] = df[col].tolist()
        else:
            data[col] = [pd.NA] * length
    result = pd.DataFrame(data)
    result["race_id"] = race_id
    result["date"] = key.date
    result["bank_code"] = key.bank_code
    result["race_no"] = race_no
    result["race_name"] = heading_norm
    result["source"] = "chariloto"
    if length and ("stadium" not in result.columns or result["stadium"].isna().all()):
        result["stadium"] = [""] * length
    result = result[ENTRY_COLUMNS]
    return result


def _normalize_payout_table(
    df: pd.DataFrame,
    heading: str,
    key: MeetingKey,
    default_race_no: int,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_text(col) for col in df.columns]
    heading_norm = _normalize_text(heading)
    race_no = _derive_race_no(heading_norm, df, default_race_no)
    race_id = _build_race_id(key.date, key.bank_code, race_no)
    df["race_id"] = race_id
    df["race_no"] = race_no
    df["date"] = key.date
    df["bank_code"] = key.bank_code
    df["source"] = "chariloto"
    return df


def _extract_meeting_metadata(soup: BeautifulSoup) -> Dict[str, str]:
    metadata: Dict[str, str] = {"stadium": "", "grade": "", "term": ""}
    title_tag = soup.find("title")
    if title_tag:
        title_text = _normalize_text(title_tag.get_text(" "))
        metadata["term"] = title_text
    for element in soup.find_all(["h1", "h2", "h3", "p", "div", "span"]):
        text = _normalize_text(element.get_text(" "))
        if not text:
            continue
        if not metadata["stadium"]:
            match = re.search(r"([\w一-龠ぁ-んァ-ンー]+(?:競輪場|バンク))", text)
            if match:
                metadata["stadium"] = match.group(1)
        if not metadata["grade"]:
            match_grade = re.search(r"(G\d|F[IL]|ナイター|ミッドナイト)", text)
            if match_grade:
                metadata["grade"] = match_grade.group(1)
        if metadata["stadium"] and metadata["grade"]:
            break
    return metadata


def _extract_info_records(
    key: MeetingKey,
    headings: Sequence[str],
    entry_tables: Sequence[Tuple[pd.DataFrame, Optional[int]]],
    metadata: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for idx, (table, race_no_hint) in enumerate(entry_tables):
        heading = _normalize_text(headings[idx] if idx < len(headings) else "")
        race_no = race_no_hint or _derive_race_no(heading, table, idx + 1)
        race_id = _build_race_id(key.date, key.bank_code, race_no)
        records.append(
            {
                "race_id": race_id,
                "race_no": race_no,
                "date": key.date,
                "bank_code": key.bank_code,
                "race_name": heading,
                "stadium": metadata.get("stadium") if metadata else "",
                "grade": metadata.get("grade") if metadata else "",
                "term": metadata.get("term") if metadata else "",
                "source": "chariloto",
            }
        )
    return pd.DataFrame(records)


def _empty_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info = pd.DataFrame(columns=INFO_COLUMNS)
    entry = pd.DataFrame(columns=ENTRY_COLUMNS)
    payout = pd.DataFrame(columns=PAYOUT_COLUMNS)
    return info, entry, payout


def fetch_results_for_ids(
    race_ids: Optional[Iterable[str]] = None,
    date_hint: Optional[str] = None,
    timeout: float = 10.0,
    retries: int = 3,
    rate_limit: float = 0.6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charilotoの結果ページから情報を取得し正規化する。"""

    provided_ids = [rid for rid in (race_ids or []) if rid]

    key_to_race_ids: Dict[MeetingKey, List[Tuple[str, Optional[int]]]] = {}
    for rid in provided_ids:
        parsed = _parse_race_id(rid)
        if not parsed:
            LOG.warning("Invalid race_id format: %s", rid)
            continue
        key, race_no = parsed
        key_to_race_ids.setdefault(key, []).append((rid, race_no))

    keys_to_fetch: Dict[MeetingKey, List[Tuple[str, Optional[int]]]] = dict(key_to_race_ids)

    if not keys_to_fetch:
        normalized_hint = _normalize_date_hint(date_hint)
        if not normalized_hint:
            LOG.warning("No race IDs or valid date hint provided; returning empty frames")
            return _empty_frames()
        LOG.info(
            "No valid race IDs supplied; enumerating banks 01-99 for %s",
            normalized_hint,
        )
        for bank in range(1, 100):
            key = MeetingKey(date=normalized_hint, bank_code=f"{bank:02d}")
            keys_to_fetch.setdefault(key, [])

    session = _create_session(timeout)

    info_frames: List[pd.DataFrame] = []
    entry_frames: List[pd.DataFrame] = []
    payout_frames: List[pd.DataFrame] = []

    for key, key_race_ids in sorted(
        keys_to_fetch.items(), key=lambda item: (item[0].date, item[0].bank_code)
    ):
        LOG.info("Fetching meeting %s %s", key.date, key.bank_code)
        html_text = _fetch_html(
            key.date,
            key.bank_code,
            timeout,
            retries,
            rate_limit,
            session=session,
        )
        if html_text is None:
            LOG.warning("Skipping meeting %s due to fetch failure", key)
            continue

        tables, soup = _collect_tables(html_text)
        if not tables:
            LOG.warning("No tables detected for %s", CHARILOTO_URL.format(bank=key.bank_code, date=key.date.replace("-", "")))
            continue

        headings = _find_heading_texts(soup)
        metadata = _extract_meeting_metadata(soup)
        meeting_entry_tables: List[Tuple[pd.DataFrame, Optional[int]]] = []
        race_numbers = [
            race_no
            for _, race_no in sorted(key_race_ids, key=lambda item: (item[1] or 0))
            if race_no is not None
        ]
        requested_ids = {rid for rid, _ in key_race_ids if rid}

        entry_counter = 0
        payout_counter = 0

        for idx, table in enumerate(tables):
            heading = headings[idx] if idx < len(headings) else ""
            columns = [str(col) for col in table.columns]
            if _classify_table(columns, ENTRY_KEYWORDS):
                race_no_hint = race_numbers[entry_counter] if entry_counter < len(race_numbers) else None
                default_race_no = race_no_hint or (entry_counter + 1)
                normalized = _normalize_entry_table(table, heading, key, default_race_no)
                filtered = (
                    normalized
                    if not requested_ids
                    else normalized[
                        normalized["race_id"].isin(requested_ids)
                        | (normalized["race_id"] == "")
                    ]
                )
                meeting_entry_tables.append((table, default_race_no))
                if not filtered.empty:
                    entry_frames.append(filtered)
                entry_counter += 1
            elif _classify_table(columns, PAYOUT_KEYWORDS):
                race_no_hint = race_numbers[payout_counter] if payout_counter < len(race_numbers) else None
                default_race_no = race_no_hint or (payout_counter + 1)
                payout_df = _normalize_payout_table(table, heading, key, default_race_no)
                if requested_ids:
                    payout_df = payout_df[
                        payout_df["race_id"].isin(requested_ids)
                        | (payout_df["race_id"] == "")
                    ]
                if not payout_df.empty:
                    payout_frames.append(payout_df)
                payout_counter += 1

        if meeting_entry_tables:
            info_df = _extract_info_records(key, headings, meeting_entry_tables, metadata)
            if requested_ids:
                info_df = info_df[info_df["race_id"].isin(requested_ids)]
            if not info_df.empty or not requested_ids:
                info_frames.append(info_df)
        else:
            LOG.warning("No entry tables extracted for %s", CHARILOTO_URL.format(bank=key.bank_code, date=key.date.replace("-", "")))

    if info_frames:
        info_df = pd.concat(info_frames, ignore_index=True)
    else:
        info_df = pd.DataFrame(columns=INFO_COLUMNS)
    for column in INFO_COLUMNS:
        if column not in info_df.columns:
            info_df[column] = ""
    info_df = info_df[INFO_COLUMNS]
    if not info_df.empty:
        info_df = info_df.sort_values(["race_id"], ignore_index=True)

    if entry_frames:
        entry_df = pd.concat(entry_frames, ignore_index=True)
    else:
        entry_df = pd.DataFrame(columns=ENTRY_COLUMNS)
    for column in ENTRY_COLUMNS:
        if column not in entry_df.columns:
            entry_df[column] = pd.NA
    entry_df = entry_df[ENTRY_COLUMNS]
    if not entry_df.empty:
        entry_df = entry_df.sort_values(["race_id", "lane_no"], ignore_index=True)

    if payout_frames:
        payout_df = pd.concat(payout_frames, ignore_index=True)
    else:
        payout_df = pd.DataFrame(columns=PAYOUT_COLUMNS)
    for column in PAYOUT_COLUMNS:
        if column not in payout_df.columns:
            payout_df[column] = pd.NA
    payout_df = payout_df[PAYOUT_COLUMNS + [col for col in payout_df.columns if col not in PAYOUT_COLUMNS]]
    if not payout_df.empty and "race_id" in payout_df.columns:
        payout_df = payout_df.sort_values(["race_id"], ignore_index=True)

    return info_df, entry_df, payout_df


__all__ = [
    "CHARILOTO_URL",
    "MeetingKey",
    "fetch_results_for_ids",
    "_parse_key_from_race_id",
]
