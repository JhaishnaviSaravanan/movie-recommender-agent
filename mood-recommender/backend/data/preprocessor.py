"""
preprocessor.py
───────────────
Cleans and merges raw JSON data from all 4 API sources into a unified list of
movie/show dicts suitable for embedding and FAISS indexing.

Each output record has a consistent schema regardless of source:
    {
        "title":       str,
        "year":        str,
        "type":        str,   # "movie" | "tv"
        "genres":      list[str],
        "overview":    str,
        "cast":        list[str],
        "keywords":    list[str],
        "imdb_rating": str,
        "runtime":     str,
        "awards":      str,
        "platforms":   list[str],
        "network":     str,
        "poster_url":  str,
        "source":      str,   # primary source tag
    }

Usage:
    from backend.data.preprocessor import build_unified_dataset
    records = build_unified_dataset()
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RAW_DIR = Path(__file__).resolve().parent / "raw"


# ══════════════════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════════════════

def _load_json(filename: str) -> list[dict[str, Any]]:
    """
    Load a raw JSON file from the data/raw directory.

    Args:
        filename (str): Filename within data/raw/.

    Returns:
        list[dict[str, Any]]: Parsed records, or empty list if file missing.
    """
    path = _RAW_DIR / filename
    if not path.exists():
        logger.warning("Raw file not found: %s — skipping.", path)
        return []
    return json.loads(path.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# Source-specific normalizers
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_tmdb(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize TMDB records into the unified schema.

    Args:
        records (list[dict]): Raw TMDB API records.

    Returns:
        list[dict]: Normalized records.
    """
    out = []
    for r in records:
        title = r.get("title") or r.get("name", "")
        if not title:
            continue
        release = r.get("release_date") or r.get("first_air_date") or ""
        year = release[:4] if release else "N/A"
        genre_ids = r.get("genre_ids", [])
        out.append({
            "title": title.strip(),
            "year": year,
            "type": r.get("_media_type", "movie"),
            "genres": [],          # genre names resolved post-merge if needed
            "genre_ids": genre_ids,
            "overview": r.get("overview", "").strip(),
            "cast": [],
            "keywords": [],
            "imdb_rating": "N/A",
            "runtime": "N/A",
            "awards": "N/A",
            "platforms": [],
            "network": "N/A",
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{r['poster_path']}"
                if r.get("poster_path") else ""
            ),
            "source": "tmdb",
            "popularity": r.get("popularity", 0),
            "tmdb_id": r.get("id"),
        })
    return out


def _normalize_omdb(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Normalize OMDB records into a lookup dict keyed by lowercase title.

    Args:
        records (list[dict]): Raw OMDB API records.

    Returns:
        dict[str, dict]: Lookup map {title_lower: omdb_fields}.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for r in records:
        title = r.get("Title", "").strip()
        if not title:
            continue
        lookup[title.lower()] = {
            "imdb_rating": r.get("imdbRating", "N/A"),
            "runtime": r.get("Runtime", "N/A"),
            "awards": r.get("Awards", "N/A"),
            "cast": [a.strip() for a in r.get("Actors", "").split(",") if a.strip()],
            "genres": [g.strip() for g in r.get("Genre", "").split(",") if g.strip()],
            "overview": r.get("Plot", "").strip(),
            "year": r.get("Year", "N/A"),
        }
    return lookup


def _normalize_streaming(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """
    Build a lookup dict of title → platforms from streaming availability data.

    Args:
        records (list[dict]): Raw streaming availability records.

    Returns:
        dict[str, list[str]]: {title_lower: [platform1, platform2, ...]}.
    """
    lookup: dict[str, list[str]] = {}
    for r in records:
        title = r.get("title", "").strip()
        if not title:
            continue
        streaming_options = r.get("streamingOptions", {})
        platforms: list[str] = []
        for country_data in streaming_options.values():
            for svc in country_data:
                service_name = svc.get("service", {}).get("name", "")
                if service_name and service_name not in platforms:
                    platforms.append(service_name)
        lookup[title.lower()] = platforms
    return lookup


def _normalize_tvmaze(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Normalize TVmaze records into a lookup dict keyed by lowercase title.

    Args:
        records (list[dict]): Raw TVmaze API records.

    Returns:
        dict[str, dict]: Lookup map {title_lower: tvmaze_fields}.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for r in records:
        title = (r.get("name") or "").strip()
        if not title:
            continue
        genres = r.get("genres", [])
        network = (r.get("network") or {}).get("name", "N/A")
        premiered = r.get("premiered") or ""
        year = premiered[:4] if premiered else "N/A"
        summary_raw = r.get("summary", "") or ""
        # Strip HTML tags from TVmaze summaries
        import re
        summary = re.sub(r"<[^>]+>", "", summary_raw).strip()
        lookup[title.lower()] = {
            "genres": genres,
            "network": network,
            "year": year,
            "overview": summary,
            "type": "tv",
        }
    return lookup


# ══════════════════════════════════════════════════════════════════════════════
# Merge
# ══════════════════════════════════════════════════════════════════════════════

def _merge(
    tmdb_records: list[dict[str, Any]],
    omdb_lookup: dict[str, dict[str, Any]],
    streaming_lookup: dict[str, list[str]],
    tvmaze_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge all source-specific data into unified records.

    TMDB records are the base.  OMDB, Streaming, and TVmaze data are joined
    by lowercase title match.

    Args:
        tmdb_records    : Normalized TMDB records (list).
        omdb_lookup     : OMDB data indexed by title_lower.
        streaming_lookup: Streaming platforms indexed by title_lower.
        tvmaze_lookup   : TVmaze data indexed by title_lower.

    Returns:
        list[dict]: Merged, deduplicated records.
    """
    merged: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for record in tmdb_records:
        key = record["title"].lower()

        # ── Skip duplicates ───────────────────────────────────────────────────
        if key in seen_titles:
            continue
        seen_titles.add(key)

        # ── Overlay OMDB fields ───────────────────────────────────────────────
        if key in omdb_lookup:
            omdb = omdb_lookup[key]
            record["imdb_rating"] = omdb.get("imdb_rating", "N/A")
            record["runtime"] = omdb.get("runtime", "N/A")
            record["awards"] = omdb.get("awards", "N/A")
            if omdb.get("cast"):
                record["cast"] = omdb["cast"]
            if omdb.get("genres"):
                record["genres"] = omdb["genres"]
            if not record["overview"] and omdb.get("overview"):
                record["overview"] = omdb["overview"]

        # ── Overlay streaming platforms ───────────────────────────────────────
        if key in streaming_lookup:
            record["platforms"] = streaming_lookup[key]

        # ── Overlay TVmaze fields (for TV shows) ──────────────────────────────
        if key in tvmaze_lookup:
            tv = tvmaze_lookup[key]
            record["network"] = tv.get("network", "N/A")
            if not record["genres"] and tv.get("genres"):
                record["genres"] = tv["genres"]
            if not record["overview"] and tv.get("overview"):
                record["overview"] = tv["overview"]

        # ── Drop records with no usable overview ──────────────────────────────
        if len(record.get("overview", "")) < 10:
            continue

        # ── Clean up internal fields ──────────────────────────────────────────
        record.pop("genre_ids", None)
        record.pop("tmdb_id", None)
        record.pop("popularity", None)

        merged.append(record)

    logger.info("Merged dataset: %d unique records", len(merged))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_unified_dataset() -> list[dict[str, Any]]:
    """
    Load raw API data, normalize each source, and merge into a unified dataset.

    Returns:
        list[dict[str, Any]]: Merged, cleaned movie/show records ready for
                              embedding by embed_builder.py.
    """
    logger.info("Loading raw data files…")
    tmdb_raw = _load_json("tmdb_data.json")
    omdb_raw = _load_json("omdb_data.json")
    streaming_raw = _load_json("streaming_data.json")
    tvmaze_raw = _load_json("tvmaze_data.json")

    logger.info("Normalizing sources…")
    tmdb_records = _normalize_tmdb(tmdb_raw)
    omdb_lookup = _normalize_omdb(omdb_raw)
    streaming_lookup = _normalize_streaming(streaming_raw)
    tvmaze_lookup = _normalize_tvmaze(tvmaze_raw)

    logger.info("Merging sources…")
    unified = _merge(tmdb_records, omdb_lookup, streaming_lookup, tvmaze_lookup)
    return unified
