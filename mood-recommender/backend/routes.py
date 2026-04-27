"""
routes.py
─────────
FastAPI route definitions for the Mood-Adaptive Recommender API.

This module contains ONLY route declarations — zero business logic.
All processing is delegated to RecommenderPipeline and the data refresh
utility functions.

Routes:
    POST /recommend     — get personalized movie/show recommendations
    POST /feedback      — submit feedback and receive refined results
    POST /refresh-data  — trigger offline data re-ingestion (n8n target)
    GET  /health        — liveness check
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.pipeline.recommender_pipeline import RecommenderPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Singleton pipeline (shared across all requests in this process) ───────────
_pipeline = RecommenderPipeline()


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════════

class RecommendRequest(BaseModel):
    """Request body for POST /recommend."""

    input: str = Field(
        ...,
        min_length=0,
        max_length=1000,
        description="User mood or free-text input (can be vague, emoji, or single word).",
    )
    session_id: str | None = Field(
        default=None,
        description="Resume an existing session. Omit to start a new one.",
    )


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""

    session_id: str = Field(..., description="Session ID returned from /recommend.")
    feedback: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's refinement request or description of what felt off.",
    )
    shown_titles: list[str] = Field(
        default_factory=list,
        description="Titles displayed in the previous turn (will be excluded).",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/recommend", summary="Get mood-based recommendations")
async def recommend(request: RecommendRequest) -> dict[str, Any]:
    """
    Interpret the user's mood and return personalized movie/show recommendations.

    Args:
        request (RecommendRequest): User input and optional session ID.

    Returns:
        dict: Pipeline response with type, session_id, interpreted_mood,
              data (list of recommendations), and follow_up message.

    Raises:
        HTTPException 422: If input validation fails.
        HTTPException 500: If the pipeline encounters an unrecoverable error.
    """
    try:
        result = _pipeline.recommend(
            user_input=request.input,
            session_id=request.session_id,
        )
        return result
    except FileNotFoundError as exc:
        # FAISS index or metadata not found — data ingestion not run yet
        logger.error("FAISS data missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vector index not found. "
                "Please run api_fetcher.py and embed_builder.py first."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /recommend: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from exc


@router.post("/feedback", summary="Refine recommendations based on user feedback")
async def feedback(request: FeedbackRequest) -> dict[str, Any]:
    """
    Accept user feedback and return a refined set of recommendations.

    Rejected titles from shown_titles are excluded from future results.

    Args:
        request (FeedbackRequest): Session ID, feedback text, and shown titles.

    Returns:
        dict: Refined pipeline response (same structure as /recommend).

    Raises:
        HTTPException 500: If the refinement pipeline fails.
    """
    try:
        result = _pipeline.refine(
            session_id=request.session_id,
            feedback_text=request.feedback,
            shown_titles=request.shown_titles,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /feedback: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process feedback. Please try again.",
        ) from exc


@router.post("/refresh-data", summary="Trigger data re-ingestion (n8n target)")
async def refresh_data() -> dict[str, str]:
    """
    Re-run the full offline ingestion pipeline: fetch → preprocess → embed.

    Intended to be called by the n8n scheduled workflow, not by end users.
    Runs synchronously (may be slow — consider making async/background task
    for production use).

    Returns:
        dict: {"status": "refreshed"} on success.

    Raises:
        HTTPException 500: If ingestion fails.
    """
    try:
        # Import lazily to avoid loading heavy deps on every request
        from backend.data.api_fetcher import run_all_fetchers
        from backend.data.embeddings.embed_builder import build_faiss_index

        logger.info("Starting scheduled data refresh…")
        run_all_fetchers()
        build_faiss_index()
        logger.info("Data refresh complete.")
        return {"status": "refreshed"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Data refresh failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data refresh failed: {exc}",
        ) from exc


@router.get("/health", summary="Liveness check")
async def health() -> dict[str, str]:
    """
    Simple health-check endpoint.

    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}
