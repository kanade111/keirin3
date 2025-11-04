"""Utility helpers for the Chariloto tooling."""
from __future__ import annotations

import logging
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER_NAME = "chariloto"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application.

    The configuration is idempotent – calling it multiple times will not
    install duplicate handlers. The format is designed to mirror the output
    that works well in terminals as well as Windows PowerShell.
    """

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name or LOGGER_NAME)


def ensure_directory(path: Path) -> Path:
    """Create *path* and all parents if they do not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def create_session(
    *,
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Optional[list[int]] = None,
    timeout: Optional[float] = 10,
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/117.0 Safari/537.36"
    ),
) -> requests.Session:
    """Return a :class:`requests.Session` configured with retry logic."""

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
        or [429, 500, 502, 503, 504],
        allowed_methods=("HEAD", "GET", "OPTIONS"),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Attach the default timeout to the session object for convenience.
    session.request = _wrap_request_with_timeout(session.request, timeout)
    return session


def _wrap_request_with_timeout(original_request, timeout: Optional[float]):
    def request_with_timeout(method, url, **kwargs):
        if timeout is not None:
            kwargs.setdefault("timeout", timeout)
        return original_request(method, url, **kwargs)

    return request_with_timeout


def fetch_url(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    rate_limit: float = 0.5,
    logger: Optional[logging.Logger] = None,
) -> requests.Response:
    """GET *url* with rate limiting and jitter."""

    logger = logger or get_logger(__name__)
    delay = max(rate_limit, 0)
    if delay:
        # Add a bit of jitter to avoid hammering the server rhythmically.
        sleep_time = delay + random.random() * 0.5
        time.sleep(sleep_time)

    response = session.get(url, params=params)
    response.raise_for_status()
    logger.debug("Fetched %s status=%s", response.url, response.status_code)
    return response


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Persist *df* as UTF-8 with BOM as mandated by the specification."""

    ensure_directory(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def backup_file(path: Path) -> None:
    """Create a timestamped backup copy of *path* if it already exists."""

    if not path.exists():
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".{timestamp}.bak")
    shutil.copy2(path, backup_path)


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric, coercing errors to NaN."""

    return pd.to_numeric(series, errors="coerce")


def normalize_whitespace(text: str) -> str:
    """Normalize weird whitespace to a single ASCII space."""

    return " ".join(text.replace("\u3000", " ").split())
