"""
llm_generator.py
────────────────
Generates personalized movie/show recommendations using an LLM (Groq or Gemini).
"""

import json
import logging
from typing import Any

import google.generativeai as genai
from groq import Groq

from backend.config import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE,
    GROQ_API_KEY, GROQ_MODEL,
    MAX_RECOMMENDATIONS, MIN_RECOMMENDATIONS
)
from backend.llm.prompt_templates import GENERATION_PROMPT

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Wraps an LLM provider to generate structured recommendations.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialise the LLM client. Prioritises Groq.
        """
        self.use_groq = bool(GROQ_API_KEY)
        
        if self.use_groq:
            logger.info("Using Groq for Generation (Model: %s)", GROQ_MODEL)
            self._groq_client = Groq(api_key=GROQ_API_KEY)
        else:
            logger.info("Using Gemini for Generation (Model: %s)", GEMINI_MODEL)
            key = api_key or GEMINI_API_KEY
            genai.configure(api_key=key)
            self._gemini_model = genai.GenerativeModel(
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
        session_context: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate recommendations from candidates.
        """
        if not candidates:
            logger.warning("No candidates provided to LLMGenerator.")
            return []

        # ── Format Prompt ──────────────────────────────────────────────────
        mood_block = (
            f"Mood: {mood_data.get('interpreted_mood', 'open')}\n"
            f"Intensity: {mood_data.get('intensity', 'medium')}\n"
            f"Themes: {', '.join(mood_data.get('themes', []))}"
        )

        candidate_summary = json.dumps(
            [
                {
                    "title": c.get("title"),
                    "year": c.get("year"),
                    "genres": c.get("genres", []),
                    "overview": c.get("overview", "")[:300],
                    "imdb_rating": c.get("imdb_rating", "N/A"),
                    "platforms": c.get("platforms", []),
                }
                for c in candidates if c
            ],
            indent=2
        )

        prompt = GENERATION_PROMPT.format(
            interpreted_mood_block=mood_block,
            candidates_block=candidate_summary,
            session_context=session_context or "No previous session context.",
            min_recs=MIN_RECOMMENDATIONS,
            max_recs=MAX_RECOMMENDATIONS,
        )

        if self.use_groq:
            recs = self._generate_groq(prompt, candidates)
        else:
            recs = self._generate_gemini(prompt, candidates)

        # Final safety check: if the LLM returned an empty list, use fallback
        if not recs and candidates:
            logger.info("LLM returned empty list. Using top-3 candidates as fallback.")
            recs = self._fallback_recommendations(candidates)
            
        return recs

    def _generate_groq(self, prompt: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            completion = self._groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a movie expert. You MUST suggest the best matches from the provided list. Return a JSON array of objects."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            raw_text = completion.choices[0].message.content.strip()
            logger.info("Groq Raw Response: %s", raw_text)
            
            data = json.loads(raw_text)
            if isinstance(data, dict):
                # Look for the first list in the dict (often 'recommendations' or 'movies')
                for val in data.values():
                    if isinstance(val, list) and len(val) > 0:
                        return val
                # If it's a single object that looks like a recommendation, wrap it
                if "title" in data:
                    return [data]
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc)
            return []

    def _generate_gemini(self, prompt: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            response = self._gemini_model.generate_content(prompt)
            if not response.parts:
                return []
            
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            recommendations = json.loads(raw_text)
            if isinstance(recommendations, list):
                return recommendations
            if isinstance(recommendations, dict):
                for val in recommendations.values():
                    if isinstance(val, list) and len(val) > 0:
                        return val
            return []
        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            return []

    @staticmethod
    def _fallback_recommendations(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Map raw candidate to the expected recommendation schema
        return [
            {
                "title": c.get("title"),
                "year": c.get("year"),
                "genres": c.get("genres", []),
                "platforms": c.get("platforms", []),
                "imdb_rating": c.get("imdb_rating", "N/A"),
                "mood_tag": "🌀 Immersive",
                "explanation": f"A highly rated {', '.join(c.get('genres', []))} title that matches your general vibe."
            }
            for c in candidates[:3]
        ]
