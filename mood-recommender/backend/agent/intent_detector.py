"""
intent_detector.py
──────────────────
Determines whether a user's input is interpretable or completely empty/null.

The system NEVER blocks on vague input — only truly empty inputs trigger a
clarification request.  Everything else (emojis, typos, slang, single words)
is treated as interpretable and forwarded to the mood extractor.

Classes:
    IntentDetector — stateless classifier with a single public method.
"""

import re
import unicodedata
from typing import Literal


class IntentDetector:
    """
    Classifies user input as 'interpretable' or 'empty'.

    The threshold for 'empty' is intentionally strict: only inputs that
    contain zero meaningful characters after stripping whitespace and
    control characters are considered empty.

    Attributes:
        None — completely stateless.
    """

    # Characters that count as "content" even if unconventional
    _EMOJI_RANGE_RE = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # Emoticons
        "\U0001F300-\U0001F5FF"   # Symbols & pictographs
        "\U0001F680-\U0001F6FF"   # Transport & map
        "\U0001F1E0-\U0001F1FF"   # Flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    def classify(self, raw_input: str) -> Literal["interpretable", "empty"]:
        """
        Classify the intent type of a raw user string.

        Args:
            raw_input (str): The unmodified text received from the frontend.

        Returns:
            Literal["interpretable", "empty"]:
                - "interpretable" — send to mood_extractor for Gemini processing.
                - "empty"         — ask the user one clarification question.
        """
        if not raw_input:
            return "empty"

        # ── Strip whitespace and control characters ──────────────────────────
        cleaned = raw_input.strip()
        cleaned = "".join(
            ch for ch in cleaned
            if not unicodedata.category(ch).startswith("C")  # remove control chars
        )

        # ── Anything left (including emojis and punctuation) is interpretable ─
        if not cleaned:
            return "empty"

        return "interpretable"

    def is_interpretable(self, raw_input: str) -> bool:
        """
        Convenience wrapper returning True when input can be processed.

        Args:
            raw_input (str): The unmodified text received from the frontend.

        Returns:
            bool: True if the input should proceed to mood extraction.
        """
        return self.classify(raw_input) == "interpretable"
