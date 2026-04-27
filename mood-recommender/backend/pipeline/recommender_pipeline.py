"""
recommender_pipeline.py
────────────────────────
Orchestrates the full Mood-Adaptive Recommendation pipeline end-to-end.

Flow:
    1. IntentDetector  — is the input interpretable or empty?
    2. MoodExtractor   — Gemini Call 1: interpret mood, generate queries
    3. FAISSRetriever  — multi-query semantic search
    4. RetrievalEvaluator — gate quality; retry with broad search if weak
    5. GeminiGenerator — Gemini Call 2: personalized recommendations
    6. FeedbackHandler — record shown titles in session

All components are instantiated once and reused (singleton pattern via module
level instances) to avoid repeated model loads.

Classes:
    RecommenderPipeline — single entry-point for /recommend and /feedback.
"""

import logging
import uuid
from typing import Any

from backend.agent.intent_detector import IntentDetector
from backend.agent.mood_extractor import MoodExtractor
from backend.agent.retrieval_evaluator import RetrievalEvaluator
from backend.agent.feedback_handler import FeedbackHandler
from backend.rag.faiss_retriever import FAISSRetriever
from backend.llm.llm_generator import LLMGenerator
from backend.llm.prompt_templates import CLARIFICATION_MESSAGE

logger = logging.getLogger(__name__)


class RecommenderPipeline:
    """
    Orchestrates all pipeline stages for a recommendation request.

    Maintains singleton instances of each component to amortise loading cost.
    FeedbackHandler holds session state across calls.

    Args:
        None — all configuration comes from backend.config.
    """

    def __init__(self) -> None:
        """Initialise all pipeline components."""
        self._intent_detector = IntentDetector()
        self._mood_extractor = MoodExtractor()
        self._retriever = FAISSRetriever()
        self._evaluator = RetrievalEvaluator()
        self._generator = LLMGenerator()
        self._feedback_handler = FeedbackHandler()
        logger.info("RecommenderPipeline initialised.")

    # ══════════════════════════════════════════════════════════════════════════
    # Primary recommend entry point
    # ══════════════════════════════════════════════════════════════════════════

    def recommend(
        self,
        user_input: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full recommendation pipeline for a user query.

        Args:
            user_input (str):        Raw text from the user.
            session_id (str | None): Existing session ID or None to create one.

        Returns:
            dict[str, Any]: Pipeline response with keys:
                - type          ("recommendation" | "clarification")
                - session_id    (str)
                - interpreted_mood (dict | None)
                - data          (list[dict] | None) — recommendations
                - follow_up     (str) — prompt for next interaction
        """
        # ── Ensure session ────────────────────────────────────────────────────
        sid = self._feedback_handler.create_session(session_id)
        logger.info("Pipeline run — session: %s, input: %r", sid, user_input[:80])

        # ── Step 1: Intent detection ──────────────────────────────────────────
        if not self._intent_detector.is_interpretable(user_input):
            return self._clarification_response(sid)

        # ── Step 2: Mood extraction (Gemini Call 1) ───────────────────────────
        mood_data = self._mood_extractor.extract(user_input)
        search_queries: list[str] = mood_data.get("search_queries", [])

        # ── Step 3: FAISS retrieval ───────────────────────────────────────────
        excluded = self._feedback_handler.get_excluded_titles(sid)
        candidates = self._retriever.search(search_queries, excluded_titles=excluded)

        # ── Step 4: Quality gate + optional re-retrieval ──────────────────────
        if not self._evaluator.is_good(candidates):
            logger.info("Retrieval quality weak — retrying with broader queries…")
            candidates = self._retriever.search_broad(excluded_titles=excluded)

        # ── Step 5: Generation (Gemini Call 2) ────────────────────────────────
        session_ctx = self._feedback_handler.get_feedback_summary(sid)
        recommendations = self._generator.generate(
            mood_data=mood_data,
            candidates=candidates,
            session_context=session_ctx,
        )

        # ── Step 6: Record shown titles ───────────────────────────────────────
        shown_titles = [r.get("title", "") for r in recommendations]
        self._feedback_handler.record_shown_titles(sid, shown_titles)

        return {
            "type": "recommendation",
            "session_id": sid,
            "interpreted_mood": mood_data,
            "data": recommendations,
            "follow_up": "💭 Does this feel right? Tell me what to adjust.",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Feedback / refinement entry point
    # ══════════════════════════════════════════════════════════════════════════

    def refine(
        self,
        session_id: str,
        feedback_text: str,
        shown_titles: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Process user feedback and return refined recommendations.

        The rejected titles are added to the session exclusion list so they
        never appear again.

        Args:
            session_id    (str): Existing session identifier.
            feedback_text (str): User's refinement request.
            shown_titles  (list[str] | None): Titles from the previous turn.

        Returns:
            dict[str, Any]: Same structure as recommend() output.
        """
        # ── Record feedback and rejections ────────────────────────────────────
        self._feedback_handler.record_feedback(
            session_id,
            feedback_text,
            rejected_titles=shown_titles,
        )

        # ── Re-run pipeline with feedback as the new user input ───────────────
        return self.recommend(feedback_text, session_id=session_id)

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _clarification_response(self, session_id: str) -> dict[str, Any]:
        """
        Build a clarification response for truly empty input.

        Args:
            session_id (str): Current session ID.

        Returns:
            dict[str, Any]: Clarification response dict.
        """
        return {
            "type": "clarification",
            "session_id": session_id,
            "interpreted_mood": None,
            "data": None,
            "follow_up": CLARIFICATION_MESSAGE,
        }
