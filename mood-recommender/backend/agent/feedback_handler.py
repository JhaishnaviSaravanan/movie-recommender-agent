"""
feedback_handler.py
────────────────────
Manages per-session state: tracks shown titles, user feedback, and rejected
suggestions so the pipeline never repeats a title in the same session.

All state is stored in-memory (dict keyed on session_id).  For production
deployments replace with Redis or a persistent store.

Classes:
    FeedbackHandler — CRUD interface for session state.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackHandler:
    """
    In-memory session manager for feedback and shown-title tracking.

    Each session contains:
        - session_id       (str)
        - shown_titles     (list[str]) — all titles ever shown this session
        - rejected_titles  (list[str]) — titles the user explicitly disliked
        - feedback_history (list[dict]) — raw feedback messages with timestamps
        - created_at       (str)

    Note:
        State is process-local.  Restart the server and all sessions reset.
        Swap the _sessions dict for Redis integration if persistence is needed.
    """

    def __init__(self) -> None:
        """Initialise the in-memory session store."""
        self._sessions: dict[str, dict[str, Any]] = {}

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def create_session(self, session_id: str | None = None) -> str:
        """
        Create a new session or return an existing one unchanged.

        Args:
            session_id (str | None): Provide an existing ID to resume, or
                                     None to auto-generate a new UUID.

        Returns:
            str: The session ID (new or existing).
        """
        if session_id and session_id in self._sessions:
            logger.debug("Resuming existing session: %s", session_id)
            return session_id

        sid = session_id or str(uuid.uuid4())
        self._sessions[sid] = {
            "session_id": sid,
            "shown_titles": [],
            "rejected_titles": [],
            "feedback_history": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Created new session: %s", sid)
        return sid

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Retrieve the full session dict by ID.

        Args:
            session_id (str): The session identifier.

        Returns:
            dict[str, Any] | None: Session data or None if not found.
        """
        return self._sessions.get(session_id)

    # ── Title tracking ────────────────────────────────────────────────────────

    def record_shown_titles(self, session_id: str, titles: list[str]) -> None:
        """
        Add newly displayed titles to the session's shown-title history.

        Args:
            session_id (str): The session identifier.
            titles     (list[str]): Titles just shown to the user.
        """
        session = self._ensure_session(session_id)
        for title in titles:
            if title not in session["shown_titles"]:
                session["shown_titles"].append(title)
        logger.debug("Recorded %d titles for session %s", len(titles), session_id)

    def get_shown_titles(self, session_id: str) -> list[str]:
        """
        Return all titles shown so far in this session.

        Args:
            session_id (str): The session identifier.

        Returns:
            list[str]: Previously shown titles (may be empty).
        """
        session = self._ensure_session(session_id)
        return list(session["shown_titles"])

    def get_excluded_titles(self, session_id: str) -> list[str]:
        """
        Return union of shown and rejected titles to exclude from retrieval.

        Args:
            session_id (str): The session identifier.

        Returns:
            list[str]: All titles that should not appear again.
        """
        session = self._ensure_session(session_id)
        return list(set(session["shown_titles"]) | set(session["rejected_titles"]))

    # ── Feedback recording ────────────────────────────────────────────────────

    def record_feedback(
        self,
        session_id: str,
        feedback_text: str,
        rejected_titles: list[str] | None = None,
    ) -> None:
        """
        Store a user feedback message and mark any rejected titles.

        Args:
            session_id      (str): The session identifier.
            feedback_text   (str): Raw feedback from the user.
            rejected_titles (list[str] | None): Titles the user disliked.
        """
        session = self._ensure_session(session_id)
        entry = {
            "text": feedback_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        session["feedback_history"].append(entry)

        if rejected_titles:
            for title in rejected_titles:
                if title not in session["rejected_titles"]:
                    session["rejected_titles"].append(title)

        logger.info("Feedback recorded for session %s", session_id)

    def get_feedback_summary(self, session_id: str) -> str:
        """
        Build a concise feedback summary string for LLM context injection.

        Args:
            session_id (str): The session identifier.

        Returns:
            str: Plain-text summary of past feedback, or empty string.
        """
        session = self._ensure_session(session_id)
        history = session["feedback_history"]
        if not history:
            return ""
        lines = [f"- {entry['text']}" for entry in history[-5:]]  # last 5
        return "Previous feedback:\n" + "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_session(self, session_id: str) -> dict[str, Any]:
        """
        Retrieve or auto-create a session by ID.

        Args:
            session_id (str): The session identifier.

        Returns:
            dict[str, Any]: The session data dict (created if missing).
        """
        if session_id not in self._sessions:
            logger.warning("Session %s not found — creating implicitly.", session_id)
            self.create_session(session_id)
        return self._sessions[session_id]
