"""
conftest.py
───────────
Pytest configuration for the Mood-Adaptive Recommender test suite.

Adds the project root to sys.path so that `backend.*` imports resolve
correctly regardless of the directory from which pytest is invoked.

Also defines shared fixtures used across multiple test modules.
"""

import sys
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent  # mood-recommender/
sys.path.insert(0, str(_ROOT))

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_mood_data() -> dict:
    """
    A fully valid mood extraction response dict for use in multiple tests.

    Returns:
        dict: Mood object matching MoodExtractor output schema.
    """
    return {
        "interpreted_mood": "melancholic, reflective",
        "intensity": "medium",
        "themes": ["loss", "moving on", "healing"],
        "search_queries": [
            "emotional healing movies",
            "bittersweet drama series",
        ],
        "confidence": "high",
    }


@pytest.fixture
def sample_candidates() -> list[dict]:
    """
    A list of 10 fake movie metadata records simulating FAISS retrieval output.

    Returns:
        list[dict]: Fake records with all required metadata fields.
    """
    return [
        {
            "title": f"Test Movie {i}",
            "year": str(2010 + i),
            "type": "movie",
            "genres": ["Drama", "Romance"],
            "overview": (
                "A deeply moving story about human connection, loss, and "
                "the quiet beauty of moving forward after heartbreak."
            ),
            "cast": ["Actor A", "Actor B"],
            "imdb_rating": f"{7.0 + (i * 0.1):.1f}",
            "runtime": "110 min",
            "awards": "Nominated for 2 Oscars",
            "platforms": ["Netflix", "Prime"],
            "network": "N/A",
            "poster_url": "",
            "source": "tmdb",
            "_retrieval_score": round(0.95 - i * 0.02, 3),
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_session_id(tmp_path) -> str:
    """
    A deterministic session ID string for testing.

    Returns:
        str: Fixed test session ID.
    """
    return "test-session-00000000"
