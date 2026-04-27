"""
retrieval_evaluator.py
───────────────────────
Scores the quality of FAISS retrieval results and decides whether to accept
them or trigger a broader re-retrieval attempt.

The evaluator uses a lightweight heuristic (no extra LLM call) to keep
latency low.  Scoring is based on result count and metadata completeness.

Classes:
    RetrievalEvaluator — assesses result quality and recommends actions.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
_MIN_RESULTS_THRESHOLD = 3       # below this → considered "weak"
_REQUIRED_METADATA_KEYS = {"title", "overview"}  # minimum fields expected


class RetrievalEvaluator:
    """
    Evaluates FAISS retrieval results and gates the recommendation pipeline.

    If results are deemed weak (too few items or missing metadata), the
    evaluator signals that a broader fallback query should be attempted.

    Args:
        min_results (int): Minimum number of results to consider a retrieval
                           successful. Defaults to _MIN_RESULTS_THRESHOLD.
    """

    def __init__(self, min_results: int = _MIN_RESULTS_THRESHOLD) -> None:
        """
        Initialise the evaluator with a minimum results threshold.

        Args:
            min_results (int): Minimum number of acceptable results.
        """
        self._min_results = min_results

    def evaluate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Score a list of retrieved movie/show metadata dicts.

        Args:
            results (list[dict[str, Any]]): Candidates from faiss_retriever.

        Returns:
            dict[str, Any]: Evaluation report with keys:
                - quality  (str): "good" | "weak"
                - score    (float): 0.0 – 1.0
                - reasons  (list[str]): Human-readable rationale
                - action   (str): "proceed" | "retry_broader"
        """
        reasons: list[str] = []
        score = 1.0

        # ── Check result count ────────────────────────────────────────────────
        if len(results) < self._min_results:
            deduction = 0.5
            score -= deduction
            reasons.append(
                f"Only {len(results)} result(s) returned "
                f"(threshold: {self._min_results})"
            )

        # ── Check metadata completeness ───────────────────────────────────────
        incomplete = [
            r.get("title", "unknown")
            for r in results
            if not _REQUIRED_METADATA_KEYS.issubset(r.keys())
        ]
        if incomplete:
            deduction = 0.3 * (len(incomplete) / max(len(results), 1))
            score -= deduction
            reasons.append(
                f"{len(incomplete)} result(s) missing required metadata fields"
            )

        # ── Check for meaningful overview text ────────────────────────────────
        empty_overviews = [
            r.get("title", "unknown")
            for r in results
            if len(str(r.get("overview", ""))) < 20
        ]
        if empty_overviews:
            score -= 0.1
            reasons.append(
                f"{len(empty_overviews)} result(s) have very short/empty overviews"
            )

        # ── Clamp score ───────────────────────────────────────────────────────
        score = max(0.0, min(1.0, score))
        quality = "good" if score >= 0.5 else "weak"
        action = "proceed" if quality == "good" else "retry_broader"

        report = {
            "quality": quality,
            "score": round(score, 2),
            "reasons": reasons,
            "action": action,
        }
        logger.info("Retrieval evaluation: %s", report)
        return report

    def is_good(self, results: list[dict[str, Any]]) -> bool:
        """
        Convenience method — returns True if retrieval quality is acceptable.

        Args:
            results (list[dict[str, Any]]): Candidates from faiss_retriever.

        Returns:
            bool: True when action is "proceed".
        """
        return self.evaluate(results)["action"] == "proceed"
