"""
gemini_generator.py
────────────────────
Gemini Call 2 — takes retrieved movie/show candidates and the interpreted
mood from Call 1 and generates warm, personalized recommendation text.

This module is responsible ONLY for generation — retrieval happens upstream.
The LLM never invents titles; it selects and narrates from provided candidates.

Classes:
    GeminiGenerator — wraps Gemini API for recommendation generation.
"""

import json
import logging
from typing import Any

import google.generativeai as genai

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE
from backend.config import MIN_RECOMMENDATIONS, MAX_RECOMMENDATIONS
from backend.llm.prompt_templates import GENERATION_PROMPT

logger = logging.getLogger(__name__)


class GeminiGenerator:
    """
    Uses Gemini to generate personalized movie/show recommendations.

    Given:
        - Interpreted mood dict (from MoodExtractor)
        - Retrieved candidate records (from FAISSRetriever)
        - Session context (from FeedbackHandler)

    Produces:
        - A list of 3-5 structured recommendation dicts.

    Args:
        api_key (str | None): Override for GEMINI_API_KEY.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialise the Gemini generation client.

        Args:
            api_key (str | None): Override for GEMINI_API_KEY from config.
        """
        key = api_key or GEMINI_API_KEY
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=GEMINI_TEMPERATURE,
                response_mime_type="application/json",
            ),
        )

    def generate(
        self,
        mood_data: dict[str, Any],
        candidates: list[dict[str, Any]],
        session_context: str = "",
    ) -> list[dict[str, Any]]:
        """
        Run Gemini Call 2: generate personalized recommendations.

        Args:
            mood_data       (dict): Output from MoodExtractor.extract().
            candidates      (list): Retrieved movie/show metadata dicts.
            session_context (str):  Plain-text summary from FeedbackHandler.

        Returns:
            list[dict[str, Any]]: 3-5 recommendation dicts, each containing:
                - title, year, genres, platforms, imdb_rating
                - mood_tag (e.g. "😌 Comforting")
                - explanation (2-3 personalised sentences)

        Raises:
            ValueError: If Gemini returns unparseable JSON.
        """
        # ── Format the interpreted mood block ─────────────────────────────────
        mood_block = (
            f"Mood: {mood_data.get('interpreted_mood', 'open')}\n"
            f"Intensity: {mood_data.get('intensity', 'medium')}\n"
            f"Themes: {', '.join(mood_data.get('themes', []))}"
        )

        # ── Format candidates as a compact JSON block ─────────────────────────
        candidate_summary = json.dumps(
            [
                {
                    "title": c.get("title"),
                    "year": c.get("year"),
                    "genres": c.get("genres", []),
                    "overview": c.get("overview", "")[:300],   # truncate for prompt size
                    "imdb_rating": c.get("imdb_rating", "N/A"),
                    "platforms": c.get("platforms", []),
                    "awards": c.get("awards", ""),
                }
                for c in candidates
            ],
            ensure_ascii=False,
            indent=2,
        )

        # ── Build the full prompt ─────────────────────────────────────────────
        prompt = GENERATION_PROMPT.format(
            interpreted_mood_block=mood_block,
            candidates_block=candidate_summary,
            session_context=session_context or "No previous session context.",
            min_recs=MIN_RECOMMENDATIONS,
            max_recs=MAX_RECOMMENDATIONS,
        )

        # ── Call Gemini ───────────────────────────────────────────────────────
        try:
            response = self._model.generate_content(prompt)
            raw_text = response.text.strip()
            recommendations: list[dict[str, Any]] = json.loads(raw_text)

            if not isinstance(recommendations, list):
                raise ValueError("Expected JSON array from Gemini generation.")

            logger.info("Generated %d recommendations", len(recommendations))
            return recommendations

        except json.JSONDecodeError as exc:
            logger.error("Gemini Call 2 returned non-JSON: %s", exc)
            return self._fallback_recommendations(candidates)

        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini Call 2 failed: %s", exc)
            return self._fallback_recommendations(candidates)

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_recommendations(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Generate minimal fallback recommendations from raw candidates when
        Gemini is unavailable.

        Args:
            candidates (list[dict]): Retrieved metadata records.

        Returns:
            list[dict]: Simplified recommendation dicts (no explanation).
        """
        logger.warning("Using fallback recommendations from raw candidates.")
        top = candidates[:MAX_RECOMMENDATIONS] if candidates else []
        return [
            {
                "title": c.get("title", "Unknown"),
                "year": c.get("year", "N/A"),
                "genres": c.get("genres", []),
                "platforms": c.get("platforms", []),
                "imdb_rating": c.get("imdb_rating", "N/A"),
                "mood_tag": "🌀 Immersive",
                "explanation": c.get("overview", "A recommended title based on your mood.")[:200],
            }
            for c in top
        ]
