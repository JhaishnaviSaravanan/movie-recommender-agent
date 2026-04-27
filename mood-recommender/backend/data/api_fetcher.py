"""
api_fetcher.py
──────────────
Offline batch ingestion module — fetches data from all 4 API sources and
writes raw JSON to disk.  This script is run ONCE (or via n8n nightly refresh)
and is NEVER called during live user queries.

Sources:
    1. TMDB  — movie/show metadata, genres, overviews, cast, keywords
    2. OMDB  — IMDb ratings, plot summaries, awards, runtime
    3. Streaming Availability (RapidAPI) — platform catalog
    4. TVmaze — TV show details, genres, network info (no API key)

Usage:
    python -m backend.data.api_fetcher

Output:
    backend/data/raw/tmdb_data.json
    backend/data/raw/omdb_data.json
    backend/data/raw/streaming_data.json
    backend/data/raw/tvmaze_data.json
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from backend.config import TMDB_API_KEY, OMDB_API_KEY, RAPIDAPI_KEY, TVMAZE_BASE_URL

logger = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
_RAW_DIR = Path(__file__).resolve().parent / "raw"
_RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Request settings ──────────────────────────────────────────────────────────
_REQUEST_TIMEOUT = 15   # seconds per request
_RATE_LIMIT_DELAY = 0.25  # seconds between paginated requests


# ══════════════════════════════════════════════════════════════════════════════
# TMDB
# ══════════════════════════════════════════════════════════════════════════════

def fetch_tmdb(max_pages: int = 20) -> list[dict[str, Any]]:
    """
    Fetch popular movies and TV shows from TMDB.

    Args:
        max_pages (int): Number of pages to fetch per media type (20 results/page).

    Returns:
        list[dict[str, Any]]: Combined movie + TV show records from TMDB.
    """
    base = "https://api.themoviedb.org/3"
    results: list[dict[str, Any]] = []

    for media_type in ("movie", "tv"):
        for page in range(1, max_pages + 1):
            try:
                resp = requests.get(
                    f"{base}/discover/{media_type}",
                    params={
                        "api_key": TMDB_API_KEY,
                        "sort_by": "popularity.desc",
                        "page": page,
                        "include_adult": False,
                    },
                    timeout=_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                items = resp.json().get("results", [])
                for item in items:
                    item["_media_type"] = media_type  # tag for downstream use
                results.extend(items)
                time.sleep(_RATE_LIMIT_DELAY)
            except requests.RequestException as exc:
                logger.error("TMDB %s page %d failed: %s", media_type, page, exc)

    logger.info("TMDB: fetched %d records", len(results))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# OMDB
# ══════════════════════════════════════════════════════════════════════════════

def fetch_omdb(titles: list[str]) -> list[dict[str, Any]]:
    """
    Enrich a list of titles with OMDB metadata (ratings, awards, runtime).

    Args:
        titles (list[str]): Movie/show titles to query.

    Returns:
        list[dict[str, Any]]: OMDB records (only successful responses included).
    """
    base = "http://www.omdbapi.com/"
    results: list[dict[str, Any]] = []

    for title in titles:
        try:
            resp = requests.get(
                base,
                params={"apikey": OMDB_API_KEY, "t": title, "type": "movie"},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("Response") == "True":
                results.append(data)
            time.sleep(_RATE_LIMIT_DELAY)
        except requests.RequestException as exc:
            logger.error("OMDB '%s' failed: %s", title, exc)

    logger.info("OMDB: fetched %d records", len(results))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Streaming Availability (RapidAPI)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_streaming(country: str = "us", max_pages: int = 10) -> list[dict[str, Any]]:
    """
    Fetch catalog from the Streaming Availability API via RapidAPI.

    Args:
        country   (str): ISO 3166-1 alpha-2 country code (default "us").
        max_pages (int): Number of result pages to retrieve.

    Returns:
        list[dict[str, Any]]: Streaming availability records.
    """
    url = "https://streaming-availability.p.rapidapi.com/shows/search/filters"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com",
    }
    results: list[dict[str, Any]] = []
    cursor: str | None = None

    for _ in range(max_pages):
        params: dict[str, Any] = {"country": country, "orderBy": "popularity_alltime"}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            results.extend(payload.get("shows", []))
            cursor = payload.get("nextCursor")
            if not cursor:
                break
            time.sleep(_RATE_LIMIT_DELAY)
        except requests.RequestException as exc:
            logger.error("Streaming API page failed: %s", exc)
            break

    logger.info("Streaming: fetched %d records", len(results))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TVmaze
# ══════════════════════════════════════════════════════════════════════════════

def fetch_tvmaze(max_pages: int = 10) -> list[dict[str, Any]]:
    """
    Fetch TV show data from the public TVmaze REST API (no key required).

    Args:
        max_pages (int): Number of pages (250 shows per page).

    Returns:
        list[dict[str, Any]]: TVmaze show records.
    """
    results: list[dict[str, Any]] = []

    for page in range(max_pages):
        try:
            resp = requests.get(
                f"{TVMAZE_BASE_URL}/shows",
                params={"page": page},
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code == 404:
                break  # no more pages
            resp.raise_for_status()
            results.extend(resp.json())
            time.sleep(_RATE_LIMIT_DELAY)
        except requests.RequestException as exc:
            logger.error("TVmaze page %d failed: %s", page, exc)
            break

    logger.info("TVmaze: fetched %d records", len(results))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_all_fetchers(tmdb_pages: int = 20) -> None:
    """
    Run all four API fetchers and save results to the raw/ directory.

    Args:
        tmdb_pages (int): Number of TMDB pages per media type.

    Output files:
        raw/tmdb_data.json
        raw/omdb_data.json
        raw/streaming_data.json
        raw/tvmaze_data.json
    """
    logger.info("Starting batch data ingestion…")

    # ── TMDB ─────────────────────────────────────────────────────────────────
    tmdb = fetch_tmdb(max_pages=tmdb_pages)
    _save(_RAW_DIR / "tmdb_data.json", tmdb)

    # ── OMDB — enrich top TMDB titles ────────────────────────────────────────
    top_titles = [
        item.get("title") or item.get("name", "")
        for item in tmdb[:200]
        if item.get("title") or item.get("name")
    ]
    omdb = fetch_omdb(top_titles)
    _save(_RAW_DIR / "omdb_data.json", omdb)

    # ── Streaming Availability ────────────────────────────────────────────────
    streaming = fetch_streaming()
    _save(_RAW_DIR / "streaming_data.json", streaming)

    # ── TVmaze ────────────────────────────────────────────────────────────────
    tvmaze = fetch_tvmaze()
    _save(_RAW_DIR / "tvmaze_data.json", tvmaze)

    logger.info("Batch ingestion complete. Raw data saved to %s", _RAW_DIR)


def _save(path: Path, data: list[dict[str, Any]]) -> None:
    """
    Serialize and write data to a JSON file.

    Args:
        path (Path): Destination file path.
        data (list): Data to serialize.
    """
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %d records → %s", len(data), path.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_all_fetchers()
