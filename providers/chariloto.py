"""Utilities for discovering Chariloto race identifiers."""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup

from utils import create_session, fetch_url, get_logger, normalize_whitespace

BASE_URL = "https://www.chariloto.com"
SCHEDULE_PATH = "/keirin/schedule"
MIDNIGHT_KEYWORDS = ("ミッドナイト", "ナイター", "midnight", "night")


@dataclass
class RaceListing:
    place_name: str
    place_code: str
    race_numbers: Sequence[int]
    midnight: bool


class CharilotoProvider:
    """High-level helper to interact with the public Chariloto endpoints."""

    def __init__(
        self,
        *,
        session=None,
        rate_limit: float = 0.5,
        retries: int = 3,
        timeout: float = 10.0,
    ) -> None:
        self.logger = get_logger(__name__)
        self.session = session or create_session(retries=retries, timeout=timeout)
        self.rate_limit = rate_limit

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def list_race_ids_for_date(self, date_str: str) -> List[str]:
        """Return race identifiers scheduled for *date_str* (YYYY-MM-DD)."""

        listings = self._list_race_cards(date_str)
        return self._build_race_ids(date_str, listings)

    def list_midnight_race_ids_for_date(self, date_str: str) -> List[str]:
        """Return only midnight/night races for *date_str*."""

        listings = [
            listing
            for listing in self._list_race_cards(date_str)
            if listing.midnight
        ]
        return self._build_race_ids(date_str, listings)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _list_race_cards(self, date_str: str) -> List[RaceListing]:
        schedule_url = urljoin(BASE_URL, SCHEDULE_PATH)
        date_value = _normalize_date(date_str)
        response = fetch_url(
            self.session,
            schedule_url,
            params={"date": date_value},
            rate_limit=self.rate_limit,
            logger=self.logger,
        )
        soup = BeautifulSoup(response.text, "lxml")
        place_links = self._extract_place_links(soup)
        listings: List[RaceListing] = []
        for place_name, href in place_links:
            full_url = urljoin(schedule_url, href)
            params = parse_qs(urlparse(full_url).query)
            place_code = _extract_place_code(params, href, place_name)
            try:
                listing = self._fetch_place_listing(
                    place_name, place_code, full_url, params, date_value
                )
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning(
                    "Failed to parse place page %s (%s): %s", full_url, place_name, exc
                )
                continue
            listings.append(listing)
        return listings

    def _fetch_place_listing(
        self,
        place_name: str,
        place_code: str,
        full_url: str,
        params: dict,
        date_value: str,
    ) -> RaceListing:
        if not params.get("date"):
            params = {**params, "date": [date_value]}
        response = fetch_url(
            self.session,
            full_url,
            params={key: values[0] for key, values in params.items()},
            rate_limit=self.rate_limit,
            logger=self.logger,
        )
        soup = BeautifulSoup(response.text, "lxml")
        race_numbers = sorted(self._extract_race_numbers(soup))
        midnight = _is_midnight_event(soup, place_name)
        return RaceListing(place_name, place_code, race_numbers, midnight)

    def _extract_place_links(self, soup: BeautifulSoup) -> List[Tuple[str, str]]:
        links: List[Tuple[str, str]] = []
        for anchor in soup.find_all("a"):
            href = anchor.get("href")
            if not href:
                continue
            text = normalize_whitespace(anchor.get_text(" ", strip=True))
            if not text:
                continue
            if "racecard" not in href and "schedule" not in href and "place=" not in href:
                continue
            links.append((text, href))
        # remove duplicates preserving order
        seen = set()
        unique_links = []
        for item in links:
            if item[1] in seen:
                continue
            seen.add(item[1])
            unique_links.append(item)
        return unique_links

    def _extract_race_numbers(self, soup: BeautifulSoup) -> Iterable[int]:
        numbers = set()
        # Inspect anchor hrefs first.
        for anchor in soup.find_all("a"):
            href = anchor.get("href") or ""
            match = re.search(r"race(?:=|/)(\d{1,2})", href)
            if match:
                numbers.add(int(match.group(1)))
        # Fallback to raw text such as "1R" or "第5レース".
        text_content = soup.get_text(" ", strip=True)
        for match in re.finditer(r"(\d{1,2})\s*R", text_content, re.IGNORECASE):
            numbers.add(int(match.group(1)))
        for match in re.finditer(r"第\s*(\d{1,2})\s*レ", text_content):
            numbers.add(int(match.group(1)))
        return numbers

    def _build_race_ids(self, date_str: str, listings: Sequence[RaceListing]) -> List[str]:
        results: List[str] = []
        date_value = _normalize_date(date_str)
        for listing in listings:
            for race_no in listing.race_numbers:
                race_id = f"{date_value}CL{listing.place_code}{race_no:02d}"
                results.append(race_id)
        return results


# ----------------------------------------------------------------------
# Public convenience functions
# ----------------------------------------------------------------------

def list_race_ids_for_date(date_str: str) -> List[str]:
    """Wrapper that instantiates :class:`CharilotoProvider`."""

    provider = CharilotoProvider()
    return provider.list_race_ids_for_date(date_str)


def list_midnight_race_ids_for_date(date_str: str) -> List[str]:
    provider = CharilotoProvider()
    return provider.list_midnight_race_ids_for_date(date_str)


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _normalize_date(date_str: str) -> str:
    parsed = _dt.datetime.strptime(date_str, "%Y-%m-%d")
    return parsed.strftime("%Y%m%d")


def _extract_place_code(params: dict, href: str, fallback: str) -> str:
    if params.get("place"):
        return params["place"][0]
    match = re.search(r"place=([A-Za-z0-9]+)", href)
    if match:
        return match.group(1)
    # Fallback – use alphanumerics from the name.
    fallback_code = re.sub(r"[^A-Za-z0-9]", "", fallback)
    return fallback_code or "00"


def _is_midnight_event(soup: BeautifulSoup, place_name: str) -> bool:
    text = normalize_whitespace(soup.get_text(" ", strip=True)).lower()
    name = place_name.lower()
    return any(keyword.lower() in text or keyword.lower() in name for keyword in MIDNIGHT_KEYWORDS)
