"""
mood_extractor.py
─────────────────
Uses an LLM (Groq or Gemini) to convert free-form user input into a structured 
mood object (JSON).

The module dynamically switches between providers based on available API keys, 
prioritising Groq for its high speed and rate limits.
"""

import json
import logging
from typing import Any

import google.generativeai as genai
from groq import Groq

from backend.config import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE,
    GROQ_API_KEY, GROQ_MODEL
)
from backend.llm.prompt_templates import MOOD_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class MoodExtractor:
    """
    Interprets natural language into structured mood JSON.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialise the LLM client.
        Prioritises Groq if GROQ_API_KEY is available.
        """
        self.use_groq = bool(GROQ_API_KEY)
        
        if self.use_groq:
            logger.info("Using Groq for Mood Extraction (Model: %s)", GROQ_MODEL)
            self._groq_client = Groq(api_key=GROQ_API_KEY)
        else:
            logger.info("Using Gemini for Mood Extraction (Model: %s)", GEMINI_MODEL)
            key = api_key or GEMINI_API_KEY
            genai.configure(api_key=key)
            self._gemini_model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=genai.types.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    response_mime_type="application/json",
                ),
            )

    def extract(self, user_input: str) -> dict[str, Any]:
        """
        Extract mood data from user input.
        """
        prompt = MOOD_EXTRACTION_PROMPT.format(user_input=user_input)

        if self.use_groq:
            return self._extract_groq(prompt, user_input)
        return self._extract_gemini(prompt, user_input)

    def _extract_groq(self, prompt: str, user_input: str) -> dict[str, Any]:
        try:
            completion = self._groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert mood analyzer. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5
            )
            raw_text = completion.choices[0].message.content.strip()
            mood_data = json.loads(raw_text)
            self._validate_mood_schema(mood_data)
            return mood_data
        except Exception as exc:
            logger.error("Groq extraction failed: %s", exc)
            return self._fallback_mood(user_input)

    def _extract_gemini(self, prompt: str, user_input: str) -> dict[str, Any]:
        try:
            response = self._gemini_model.generate_content(prompt)
            if not response.parts:
                return self._fallback_mood(user_input)
            
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            mood_data = json.loads(raw_text)
            self._validate_mood_schema(mood_data)
            return mood_data
        except Exception as exc:
            logger.error("Gemini extraction failed: %s", exc)
            return self._fallback_mood(user_input)

    def _validate_mood_schema(self, data: dict[str, Any]) -> None:
        required = ["interpreted_mood", "intensity", "themes", "confidence"]
        for field in required:
            if field not in data:
                data[field] = "unknown" if field != "themes" else []

    def _fallback_mood(self, user_input: str) -> dict[str, Any]:
        return {
            "interpreted_mood": "open, curious",
            "intensity": "medium",
            "themes": [],
            "search_queries": ["popular highly rated movies", "top trending shows"],
            "confidence": "low",
            "reasoning": f"Fallback due to LLM error on input: {user_input[:50]}..."
        }
