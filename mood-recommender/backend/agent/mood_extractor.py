"""
mood_extractor.py
─────────────────
Gemini Call 1 — interprets ANY user input (however vague) into a structured
mood object that downstream retrieval and generation modules can act on.

The extractor is deliberately permissive: it never refuses an input.
Even nonsense strings are turned into a best-effort mood interpretation.

Classes:
    MoodExtractor — wraps Gemini API for mood interpretation.

Typical output schema:
    {
        "interpreted_mood": "melancholic, reflective",
        "intensity": "medium",
        "themes": ["loss", "moving on"],
        "search_queries": [
            "emotional healing movies",
            "bittersweet drama series"
        ],
        "confidence": "high"
    }
"""

import json
import logging
from typing import Any

import google.generativeai as genai

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE
from backend.llm.prompt_templates import MOOD_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class MoodExtractor:
    """
    Uses Gemini to convert free-form user input into a structured mood object.

    The model is instructed to ALWAYS return valid JSON — even for completely
    unrecognisable inputs — by falling back to a relaxed / open mood state.

    Args:
        api_key (str | None): Override the GEMINI_API_KEY from config.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialise the Gemini client.

        Args:
            api_key (str | None): If provided, overrides the global config key.
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

    def extract(self, user_input: str) -> dict[str, Any]:
        """
        Run Gemini Call 1: interpret user input into a structured mood dict.

        Args:
            user_input (str): Raw, unprocessed text from the user.

        Returns:
            dict[str, Any]: Structured mood object with keys:
                - interpreted_mood (str)
                - intensity        (str: "low" | "medium" | "high")
                - themes           (list[str])
                - search_queries   (list[str], 2-3 items)
                - confidence       (str: "low" | "medium" | "high")

        Raises:
            ValueError: If Gemini returns unparseable JSON after retries.
        """
        # ── Build the prompt ─────────────────────────────────────────────────
        prompt = MOOD_EXTRACTION_PROMPT.format(user_input=user_input)

        try:
            response = self._model.generate_content(prompt)
            raw_text = response.text.strip()

            # ── Parse JSON response ──────────────────────────────────────────
            mood_data: dict[str, Any] = json.loads(raw_text)
            self._validate_mood_schema(mood_data)
            logger.info("Mood extracted successfully: %s", mood_data)
            return mood_data

        except json.JSONDecodeError as exc:
            logger.error("Gemini returned non-JSON response: %s", exc)
            return self._fallback_mood(user_input)

        except Exception as exc:  # noqa: BLE001 — broad catch for API errors
            logger.error("Gemini Call 1 failed: %s", exc)
            return self._fallback_mood(user_input)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_mood_schema(data: dict[str, Any]) -> None:
        """
        Ensure the required keys are present in the mood response.

        Args:
            data (dict[str, Any]): Parsed JSON from Gemini.

        Raises:
            ValueError: If any required key is missing.
        """
        required = {"interpreted_mood", "intensity", "themes", "search_queries", "confidence"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Mood response missing keys: {missing}")

    @staticmethod
    def _fallback_mood(user_input: str) -> dict[str, Any]:
        """
        Generate a safe fallback mood when Gemini is unavailable or fails.

        Args:
            user_input (str): The original user text.

        Returns:
            dict[str, Any]: A generic open/relaxed mood object.
        """
        logger.warning("Using fallback mood for input: %r", user_input)
        return {
            "interpreted_mood": "open, curious",
            "intensity": "medium",
            "themes": ["adventure", "discovery", "feel-good"],
            "search_queries": [
                "feel-good movies to watch",
                "popular entertaining films",
            ],
            "confidence": "low",
        }
