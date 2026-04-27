"""
test_agent.py
─────────────
Tests for the agent layer components:
    - IntentDetector  : normal, vague, emoji, empty, gibberish inputs
    - MoodExtractor   : output schema validation, fallback on API failure
    - FeedbackHandler : session lifecycle, title tracking, feedback recording
    - RetrievalEvaluator: quality scoring logic

Run with:
    pytest backend/tests/test_agent.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.agent.intent_detector import IntentDetector
from backend.agent.retrieval_evaluator import RetrievalEvaluator
from backend.agent.feedback_handler import FeedbackHandler


# ══════════════════════════════════════════════════════════════════════════════
# IntentDetector
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentDetector:
    """Tests for IntentDetector.classify() and .is_interpretable()."""

    def setup_method(self):
        self.detector = IntentDetector()

    # ── Interpretable inputs ──────────────────────────────────────────────────

    def test_normal_input_is_interpretable(self):
        """Standard clear mood input should be interpretable."""
        assert self.detector.is_interpretable("I feel happy and energetic") is True

    def test_single_word_is_interpretable(self):
        """Single word inputs like 'idk' must be interpretable (not empty)."""
        assert self.detector.is_interpretable("idk") is True

    def test_emoji_only_is_interpretable(self):
        """Emoji-only inputs like '🥺' must be interpretable."""
        assert self.detector.is_interpretable("🥺") is True

    def test_gibberish_is_interpretable(self):
        """'aaaaaa' should still be classified as interpretable (not empty)."""
        assert self.detector.is_interpretable("aaaaaa") is True

    def test_vague_phrase_is_interpretable(self):
        """Vague phrases like 'something good' are interpretable."""
        assert self.detector.is_interpretable("something good") is True

    def test_burnt_out_is_interpretable(self):
        """Colloquial mood terms are interpretable."""
        assert self.detector.is_interpretable("burnt out") is True

    def test_pop_culture_ref_is_interpretable(self):
        """Pop-culture references like 'like Dark but easier' are interpretable."""
        assert self.detector.is_interpretable("like Dark but easier") is True

    def test_punctuation_only_is_interpretable(self):
        """Punctuation/symbols count as content."""
        assert self.detector.is_interpretable("...") is True

    def test_mixed_emoji_text_is_interpretable(self):
        """Mixed emoji + text is interpretable."""
        assert self.detector.is_interpretable("😢 sad day") is True

    # ── Empty inputs ──────────────────────────────────────────────────────────

    def test_empty_string_is_empty(self):
        """Truly empty string must return 'empty'."""
        assert self.detector.classify("") == "empty"

    def test_whitespace_only_is_empty(self):
        """Whitespace-only strings must be classified empty."""
        assert self.detector.classify("   ") == "empty"

    def test_tab_newline_only_is_empty(self):
        """Tab and newline-only strings must be empty."""
        assert self.detector.classify("\t\n\r") == "empty"

    # ── classify returns correct literals ────────────────────────────────────

    def test_classify_returns_literal_interpretable(self):
        assert self.detector.classify("happy") == "interpretable"

    def test_classify_returns_literal_empty(self):
        assert self.detector.classify("") == "empty"


# ══════════════════════════════════════════════════════════════════════════════
# MoodExtractor (with Gemini mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestMoodExtractor:
    """Tests for MoodExtractor output schema and fallback behaviour."""

    _VALID_RESPONSE = {
        "interpreted_mood": "melancholic, reflective",
        "intensity": "medium",
        "themes": ["loss", "moving on"],
        "search_queries": ["emotional healing movies", "bittersweet drama series"],
        "confidence": "high",
    }

    def _make_extractor(self, gemini_response=None, raise_exc=None):
        """
        Build a MoodExtractor with Gemini mocked.

        Args:
            gemini_response: dict to return (will be JSON-encoded)
            raise_exc: Exception to raise instead

        Returns:
            MoodExtractor with patched model
        """
        import json
        from backend.agent.mood_extractor import MoodExtractor

        extractor = MoodExtractor.__new__(MoodExtractor)

        mock_model = MagicMock()
        if raise_exc:
            mock_model.generate_content.side_effect = raise_exc
        else:
            mock_resp = MagicMock()
            mock_resp.text = json.dumps(gemini_response or self._VALID_RESPONSE)
            mock_model.generate_content.return_value = mock_resp

        extractor._model = mock_model
        return extractor

    def test_extract_returns_all_required_keys(self):
        """Successful extraction must include all required schema keys."""
        extractor = self._make_extractor(self._VALID_RESPONSE)
        result = extractor.extract("I feel so lost")
        required = {"interpreted_mood", "intensity", "themes", "search_queries", "confidence"}
        assert required.issubset(result.keys())

    def test_extract_search_queries_is_list(self):
        """search_queries must be a list."""
        extractor = self._make_extractor(self._VALID_RESPONSE)
        result = extractor.extract("I feel lost")
        assert isinstance(result["search_queries"], list)

    def test_extract_themes_is_list(self):
        """themes must be a list."""
        extractor = self._make_extractor(self._VALID_RESPONSE)
        result = extractor.extract("idk")
        assert isinstance(result["themes"], list)

    def test_fallback_on_gemini_failure(self):
        """Should return fallback mood dict when Gemini raises an exception."""
        extractor = self._make_extractor(raise_exc=Exception("API error"))
        result = extractor.extract("anything")
        assert "interpreted_mood" in result
        assert result["confidence"] == "low"

    def test_fallback_on_invalid_json(self):
        """Should return fallback mood dict when Gemini returns non-JSON."""
        from backend.agent.mood_extractor import MoodExtractor
        extractor = MoodExtractor.__new__(MoodExtractor)
        mock_model = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "This is not JSON at all!"
        mock_model.generate_content.return_value = mock_resp
        extractor._model = mock_model

        result = extractor.extract("🥺")
        assert "interpreted_mood" in result

    def test_intensity_is_valid_value(self):
        """intensity field must be one of: low, medium, high."""
        extractor = self._make_extractor(self._VALID_RESPONSE)
        result = extractor.extract("happy")
        assert result["intensity"] in ("low", "medium", "high")

    def test_empty_input_still_produces_output(self):
        """
        Even with empty input, extractor (when called directly, bypassing
        intent detection) should produce a result via fallback.
        """
        extractor = self._make_extractor(raise_exc=Exception("oops"))
        result = extractor.extract("")
        assert "interpreted_mood" in result


# ══════════════════════════════════════════════════════════════════════════════
# RetrievalEvaluator
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator scoring logic."""

    def setup_method(self):
        self.evaluator = RetrievalEvaluator(min_results=3)

    def _make_good_result(self, title="Movie A"):
        return {
            "title": title,
            "overview": "A long and detailed movie overview that describes the plot well.",
        }

    def test_good_results_produce_proceed_action(self):
        """Sufficient, complete results should produce 'proceed' action."""
        results = [self._make_good_result(f"Movie {i}") for i in range(5)]
        report = self.evaluator.evaluate(results)
        assert report["action"] == "proceed"
        assert report["quality"] == "good"

    def test_too_few_results_produce_retry_action(self):
        """Below threshold should produce 'retry_broader' action."""
        results = [self._make_good_result("Only One")]  # less than min_results=3
        report = self.evaluator.evaluate(results)
        assert report["action"] == "retry_broader"
        assert report["quality"] == "weak"

    def test_empty_results_produce_retry_action(self):
        """Zero results should always be weak."""
        report = self.evaluator.evaluate([])
        assert report["action"] == "retry_broader"

    def test_score_is_between_0_and_1(self):
        """Score must always be clamped to [0.0, 1.0]."""
        results = [self._make_good_result(f"M{i}") for i in range(10)]
        report = self.evaluator.evaluate(results)
        assert 0.0 <= report["score"] <= 1.0

    def test_missing_metadata_reduces_score(self):
        """Records missing required fields should lower the score."""
        results = [{"unknown_field": "x"} for _ in range(5)]
        report = self.evaluator.evaluate(results)
        assert report["score"] < 1.0

    def test_is_good_convenience_method(self):
        """is_good() should return True for good results."""
        results = [self._make_good_result(f"M{i}") for i in range(5)]
        assert self.evaluator.is_good(results) is True


# ══════════════════════════════════════════════════════════════════════════════
# FeedbackHandler
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedbackHandler:
    """Tests for session management and feedback tracking."""

    def setup_method(self):
        self.handler = FeedbackHandler()

    def test_create_session_returns_string(self):
        """create_session must return a non-empty string."""
        sid = self.handler.create_session()
        assert isinstance(sid, str) and len(sid) > 0

    def test_create_session_with_explicit_id(self):
        """create_session with explicit ID returns that exact ID."""
        sid = self.handler.create_session("my-session-123")
        assert sid == "my-session-123"

    def test_resuming_existing_session(self):
        """Calling create_session twice with same ID resumes (no reset)."""
        sid = self.handler.create_session("sess-abc")
        self.handler.record_shown_titles("sess-abc", ["Movie A"])
        self.handler.create_session("sess-abc")   # should NOT reset
        assert "Movie A" in self.handler.get_shown_titles("sess-abc")

    def test_record_and_get_shown_titles(self):
        """Shown titles are persisted and retrievable."""
        sid = self.handler.create_session()
        self.handler.record_shown_titles(sid, ["Inception", "Interstellar"])
        titles = self.handler.get_shown_titles(sid)
        assert "Inception" in titles
        assert "Interstellar" in titles

    def test_no_duplicate_shown_titles(self):
        """Same title recorded twice should appear only once."""
        sid = self.handler.create_session()
        self.handler.record_shown_titles(sid, ["Inception"])
        self.handler.record_shown_titles(sid, ["Inception"])
        titles = self.handler.get_shown_titles(sid)
        assert titles.count("Inception") == 1

    def test_record_feedback_stores_entry(self):
        """Feedback text is stored in session history."""
        sid = self.handler.create_session()
        self.handler.record_feedback(sid, "Too dark, want something lighter")
        session = self.handler.get_session(sid)
        assert len(session["feedback_history"]) == 1

    def test_rejected_titles_appear_in_excluded(self):
        """Rejected titles should be in get_excluded_titles result."""
        sid = self.handler.create_session()
        self.handler.record_feedback(sid, "didn't like it", rejected_titles=["Dark"])
        excluded = self.handler.get_excluded_titles(sid)
        assert "Dark" in excluded

    def test_shown_titles_appear_in_excluded(self):
        """Shown titles also appear in excluded list."""
        sid = self.handler.create_session()
        self.handler.record_shown_titles(sid, ["The Bear"])
        excluded = self.handler.get_excluded_titles(sid)
        assert "The Bear" in excluded

    def test_feedback_summary_empty_on_new_session(self):
        """New session should have empty feedback summary."""
        sid = self.handler.create_session()
        summary = self.handler.get_feedback_summary(sid)
        assert summary == ""

    def test_feedback_summary_includes_recent_feedback(self):
        """Feedback summary should include recently recorded feedback text."""
        sid = self.handler.create_session()
        self.handler.record_feedback(sid, "Too intense, want something calmer")
        summary = self.handler.get_feedback_summary(sid)
        assert "Too intense" in summary
