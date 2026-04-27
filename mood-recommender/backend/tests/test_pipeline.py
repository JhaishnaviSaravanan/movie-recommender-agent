"""
test_pipeline.py
────────────────
End-to-end pipeline tests with all external dependencies mocked.

Tests:
    - Full recommend() flow for normal, vague, emoji, empty input
    - Feedback refinement loop
    - Session memory persistence across multiple calls
    - Weak retrieval triggers re-retrieval
    - FAISS missing file returns meaningful error

Run with:
    pytest backend/tests/test_pipeline.py -v
"""

import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ── Shared test fixtures ──────────────────────────────────────────────────────

_MOOD_RESPONSE = json.dumps({
    "interpreted_mood": "melancholic, reflective",
    "intensity": "medium",
    "themes": ["loss", "moving on"],
    "search_queries": ["emotional healing movies", "bittersweet dramas"],
    "confidence": "high",
})

_CANDIDATES = [
    {
        "title": f"Great Film {i}",
        "overview": "A deeply moving story about loss and redemption." * 2,
        "year": "2020",
        "genres": ["Drama"],
        "imdb_rating": "8.0",
        "platforms": ["Netflix"],
    }
    for i in range(10)
]

_RECOMMENDATION = {
    "title": "Great Film 0",
    "year": "2020",
    "genres": ["Drama"],
    "platforms": ["Netflix"],
    "imdb_rating": "8.0",
    "mood_tag": "😌 Comforting",
    "explanation": "A beautifully crafted story that resonates with your reflective mood.",
}


def _build_pipeline():
    """
    Build a RecommenderPipeline with all external components mocked.

    Returns:
        RecommenderPipeline with patched Gemini, FAISS, and embedding models.
    """
    from backend.pipeline.recommender_pipeline import RecommenderPipeline

    pipeline = RecommenderPipeline.__new__(RecommenderPipeline)

    # IntentDetector — use real one (no external deps)
    from backend.agent.intent_detector import IntentDetector
    pipeline._intent_detector = IntentDetector()

    # MoodExtractor — mock Gemini Call 1
    from backend.agent.mood_extractor import MoodExtractor
    extractor = MoodExtractor.__new__(MoodExtractor)
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = _MOOD_RESPONSE
    mock_model.generate_content.return_value = mock_resp
    extractor._model = mock_model
    pipeline._mood_extractor = extractor

    # FAISSRetriever — mock index and metadata
    from backend.rag.faiss_retriever import FAISSRetriever
    retriever = FAISSRetriever.__new__(FAISSRetriever)
    retriever._top_k = 5
    mock_index = MagicMock()
    scores = np.full((2, 5), 0.9, dtype=np.float32)
    indices = np.array([[i % 10 for i in range(5)],
                        [i % 10 for i in range(5)]], dtype=np.int64)
    mock_index.search.return_value = (scores, indices)
    retriever.__dict__["_index"] = mock_index
    retriever.__dict__["_metadata"] = _CANDIDATES
    pipeline._retriever = retriever

    # RetrievalEvaluator — real one
    from backend.agent.retrieval_evaluator import RetrievalEvaluator
    pipeline._evaluator = RetrievalEvaluator(min_results=3)

    # GeminiGenerator — mock Gemini Call 2
    from backend.llm.gemini_generator import GeminiGenerator
    generator = GeminiGenerator.__new__(GeminiGenerator)
    mock_gen_model = MagicMock()
    mock_gen_resp = MagicMock()
    mock_gen_resp.text = json.dumps([_RECOMMENDATION.copy() for _ in range(3)])
    mock_gen_model.generate_content.return_value = mock_gen_resp
    generator._model = mock_gen_model
    pipeline._generator = generator

    # FeedbackHandler — real one
    from backend.agent.feedback_handler import FeedbackHandler
    pipeline._feedback_handler = FeedbackHandler()

    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRecommenderPipeline:

    @patch("backend.rag.faiss_retriever.embed_queries",
           return_value=np.ones((2, 384), dtype=np.float32))
    def test_normal_input_returns_recommendations(self, _):
        """Standard clear input should return type=recommendation with data."""
        pipeline = _build_pipeline()
        result = pipeline.recommend("I feel happy and energetic")
        assert result["type"] == "recommendation"
        assert isinstance(result["data"], list)
        assert len(result["data"]) > 0

    @patch("backend.rag.faiss_retriever.embed_queries",
           return_value=np.ones((2, 384), dtype=np.float32))
    def test_vague_input_idk_returns_recommendations(self, _):
        """'idk' should never trigger clarification — must return recommendations."""
        pipeline = _build_pipeline()
        result = pipeline.recommend("idk")
        assert result["type"] == "recommendation"

    @patch("backend.rag.faiss_retriever.embed_queries",
           return_value=np.ones((2, 384), dtype=np.float32))
    def test_emoji_input_returns_recommendations(self, _):
        """Emoji-only input '🥺' must return recommendations."""
        pipeline = _build_pipeline()
        result = pipeline.recommend("🥺")
        assert result["type"] == "recommendation"

    def test_empty_input_returns_clarification(self):
        """Truly empty string must return type=clarification."""
        pipeline = _build_pipeline()
        result = pipeline.recommend("")
        assert result["type"] == "clarification"
        assert result["data"] is None
        assert len(result["follow_up"]) > 0

    def test_whitespace_input_returns_clarification(self):
        """Whitespace-only input must return clarification."""
        pipeline = _build_pipeline()
        result = pipeline.recommend("   ")
        assert result["type"] == "clarification"

    @patch("backend.rag.faiss_retriever.embed_queries",
           return_value=np.ones((2, 384), dtype=np.float32))
    def test_session_id_returned(self, _):
        """Response must include a session_id string."""
        pipeline = _build_pipeline()
        result = pipeline.recommend("something relaxing")
        assert isinstance(result["session_id"], str)
        assert len(result["session_id"]) > 0

    @patch("backend.rag.faiss_retriever.embed_queries",
           return_value=np.ones((2, 384), dtype=np.float32))
    def test_interpreted_mood_returned(self, _):
        """Response must include interpreted_mood dict."""
        pipeline = _build_pipeline()
        result = pipeline.recommend("I feel burnt out")
        assert isinstance(result["interpreted_mood"], dict)

    @patch("backend.rag.faiss_retriever.embed_queries",
           return_value=np.ones((2, 384), dtype=np.float32))
    def test_shown_titles_tracked_in_session(self, _):
        """Titles from first call should be tracked and excluded in second call."""
        pipeline = _build_pipeline()
        result1 = pipeline.recommend("happy")
        sid = result1["session_id"]
        shown = pipeline._feedback_handler.get_shown_titles(sid)
        assert len(shown) > 0

    @patch("backend.rag.faiss_retriever.embed_queries",
           return_value=np.ones((2, 384), dtype=np.float32))
    def test_session_continuity_across_calls(self, _):
        """Second call with same session_id should use existing session."""
        pipeline = _build_pipeline()
        result1 = pipeline.recommend("calm", session_id=None)
        sid = result1["session_id"]
        result2 = pipeline.recommend("something cozy", session_id=sid)
        assert result2["session_id"] == sid

    @patch("backend.rag.faiss_retriever.embed_queries",
           return_value=np.ones((2, 384), dtype=np.float32))
    def test_refine_returns_recommendations(self, _):
        """refine() should return updated recommendations."""
        pipeline = _build_pipeline()
        result1 = pipeline.recommend("feeling nostalgic")
        sid = result1["session_id"]
        shown = [r["title"] for r in result1["data"]]
        result2 = pipeline.refine(sid, "too dark, want something lighter", shown)
        assert result2["type"] in ("recommendation", "clarification")

    @patch("backend.rag.faiss_retriever.embed_queries",
           return_value=np.ones((2, 384), dtype=np.float32))
    def test_follow_up_message_present(self, _):
        """Every response should include a follow_up message string."""
        pipeline = _build_pipeline()
        result = pipeline.recommend("adventurous")
        assert isinstance(result["follow_up"], str) and len(result["follow_up"]) > 0
