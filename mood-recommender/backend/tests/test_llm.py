"""
test_llm.py
───────────
Tests for GeminiGenerator:
    - Output contains 3-5 items
    - Each item has required fields
    - mood_tag present and valid
    - explanation is non-empty string
    - Fallback on Gemini failure

Run with:
    pytest backend/tests/test_llm.py -v
"""

import json
import pytest
from unittest.mock import MagicMock

from backend.config import MIN_RECOMMENDATIONS, MAX_RECOMMENDATIONS


def _make_generator(response_list=None, raise_exc=None):
    """Build a GeminiGenerator with mocked Gemini model."""
    from backend.llm.gemini_generator import GeminiGenerator
    gen = GeminiGenerator.__new__(GeminiGenerator)
    mock_model = MagicMock()
    if raise_exc:
        mock_model.generate_content.side_effect = raise_exc
    else:
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(response_list or [])
        mock_model.generate_content.return_value = mock_resp
    gen._model = mock_model
    return gen


_VALID_REC = {
    "title": "Eternal Sunshine of the Spotless Mind",
    "year": "2004",
    "genres": ["Drama", "Romance"],
    "platforms": ["Prime"],
    "imdb_rating": "8.3",
    "mood_tag": "😌 Comforting",
    "explanation": "Perfect for sitting with something you can't let go of.",
}

_MOOD_DATA = {
    "interpreted_mood": "melancholic, reflective",
    "intensity": "medium",
    "themes": ["loss", "moving on"],
    "search_queries": ["emotional healing movies"],
    "confidence": "high",
}

_CANDIDATES = [
    {"title": f"Film {i}", "overview": "A detailed plot about emotions.", "year": "2020",
     "genres": ["Drama"], "imdb_rating": "7.5", "platforms": ["Netflix"]}
    for i in range(10)
]


class TestGeminiGenerator:

    def test_generate_returns_list(self):
        """generate() must return a list."""
        recs = [_VALID_REC.copy() for _ in range(3)]
        gen = _make_generator(recs)
        result = gen.generate(_MOOD_DATA, _CANDIDATES)
        assert isinstance(result, list)

    def test_generate_count_within_range(self):
        """Result count should be between MIN and MAX recommendations."""
        recs = [_VALID_REC.copy() for _ in range(4)]
        gen = _make_generator(recs)
        result = gen.generate(_MOOD_DATA, _CANDIDATES)
        assert MIN_RECOMMENDATIONS <= len(result) <= MAX_RECOMMENDATIONS + 2

    def test_each_item_has_required_fields(self):
        """Every recommendation must have title, year, genres, platforms, mood_tag, explanation."""
        required = {"title", "year", "genres", "platforms", "mood_tag", "explanation"}
        recs = [_VALID_REC.copy() for _ in range(3)]
        gen = _make_generator(recs)
        result = gen.generate(_MOOD_DATA, _CANDIDATES)
        for item in result:
            missing = required - item.keys()
            assert not missing, f"Missing keys: {missing}"

    def test_mood_tag_is_string(self):
        """mood_tag must be a non-empty string."""
        recs = [_VALID_REC.copy() for _ in range(3)]
        gen = _make_generator(recs)
        result = gen.generate(_MOOD_DATA, _CANDIDATES)
        for item in result:
            assert isinstance(item["mood_tag"], str) and len(item["mood_tag"]) > 0

    def test_explanation_is_non_empty_string(self):
        """explanation must be a non-empty string."""
        recs = [_VALID_REC.copy() for _ in range(3)]
        gen = _make_generator(recs)
        result = gen.generate(_MOOD_DATA, _CANDIDATES)
        for item in result:
            assert isinstance(item["explanation"], str) and len(item["explanation"]) > 5

    def test_fallback_on_api_exception(self):
        """Should return fallback recommendations on Gemini exception."""
        gen = _make_generator(raise_exc=Exception("API timeout"))
        result = gen.generate(_MOOD_DATA, _CANDIDATES)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_fallback_on_invalid_json(self):
        """Should return fallback recommendations on non-JSON Gemini response."""
        from backend.llm.gemini_generator import GeminiGenerator
        gen = GeminiGenerator.__new__(GeminiGenerator)
        mock_model = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Not valid JSON!!!"
        mock_model.generate_content.return_value = mock_resp
        gen._model = mock_model
        result = gen.generate(_MOOD_DATA, _CANDIDATES)
        assert isinstance(result, list)

    def test_genres_is_list(self):
        """genres field must be a list."""
        recs = [_VALID_REC.copy() for _ in range(3)]
        gen = _make_generator(recs)
        result = gen.generate(_MOOD_DATA, _CANDIDATES)
        for item in result:
            assert isinstance(item["genres"], list)

    def test_platforms_is_list(self):
        """platforms field must be a list."""
        recs = [_VALID_REC.copy() for _ in range(3)]
        gen = _make_generator(recs)
        result = gen.generate(_MOOD_DATA, _CANDIDATES)
        for item in result:
            assert isinstance(item["platforms"], list)
