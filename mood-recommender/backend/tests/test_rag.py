"""
test_rag.py
───────────
Tests for the RAG pipeline: embed_query, embed_queries, FAISSRetriever.

Run with:
    pytest backend/tests/test_rag.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ── embed_query / embed_queries ───────────────────────────────────────────────

class TestEmbedQuery:

    @patch("backend.rag.embed_query._get_model")
    def test_embed_query_returns_2d_float32(self, mock_get_model):
        """embed_query must return shape (1, dim) float32 array."""
        mock_get_model.return_value.encode.return_value = np.ones((1, 384), dtype=np.float32)
        from backend.rag.embed_query import embed_query
        result = embed_query("emotional drama")
        assert result.ndim == 2 and result.shape[0] == 1
        assert result.dtype == np.float32

    @patch("backend.rag.embed_query._get_model")
    def test_embed_queries_batch_shape(self, mock_get_model):
        """embed_queries for N strings returns shape (N, dim)."""
        mock_get_model.return_value.encode.return_value = np.ones((3, 384), dtype=np.float32)
        from backend.rag.embed_query import embed_queries
        result = embed_queries(["q1", "q2", "q3"])
        assert result.shape[0] == 3

    def test_embed_queries_raises_on_empty(self):
        """embed_queries must raise ValueError for empty list."""
        from backend.rag.embed_query import embed_queries
        with pytest.raises(ValueError):
            embed_queries([])

    @patch("backend.rag.embed_query._get_model")
    def test_embed_query_normalized(self, mock_get_model):
        """Normalized output should have L2 norm ≈ 1.0."""
        vec = np.ones((1, 384), dtype=np.float32)
        vec /= np.linalg.norm(vec)
        mock_get_model.return_value.encode.return_value = vec
        from backend.rag.embed_query import embed_query
        result = embed_query("test")
        assert abs(float(np.linalg.norm(result)) - 1.0) < 0.01


# ── FAISSRetriever helpers ─────────────────────────────────────────────────────

def _make_mock_retriever(metadata, top_k=5):
    from backend.rag.faiss_retriever import FAISSRetriever
    retriever = FAISSRetriever.__new__(FAISSRetriever)
    retriever._top_k = top_k
    n = len(metadata)
    mock_index = MagicMock()
    scores = np.full((1, top_k), 0.9, dtype=np.float32)
    indices = np.array([[i % n for i in range(top_k)]], dtype=np.int64)
    mock_index.search.return_value = (scores, indices)
    retriever.__dict__["_index"] = mock_index
    retriever.__dict__["_metadata"] = metadata
    return retriever


_METADATA = [
    {"title": f"Movie {i}", "overview": f"Detailed overview for movie {i}." * 3}
    for i in range(20)
]


class TestFAISSRetriever:

    def test_search_returns_list(self):
        retriever = _make_mock_retriever(_METADATA)
        with patch("backend.rag.faiss_retriever.embed_queries",
                   return_value=np.ones((1, 384), dtype=np.float32)):
            results = retriever.search(["feel-good movies"])
        assert isinstance(results, list)

    def test_search_deduplicates_results(self):
        retriever = _make_mock_retriever(_METADATA, top_k=5)
        with patch("backend.rag.faiss_retriever.embed_queries",
                   return_value=np.ones((2, 384), dtype=np.float32)):
            results = retriever.search(["q1", "q2"])
        titles = [r["title"] for r in results]
        assert len(titles) == len(set(titles))

    def test_search_respects_excluded_titles(self):
        retriever = _make_mock_retriever(_METADATA, top_k=5)
        excluded = ["Movie 0", "Movie 1", "Movie 2"]
        with patch("backend.rag.faiss_retriever.embed_queries",
                   return_value=np.ones((1, 384), dtype=np.float32)):
            results = retriever.search(["drama"], excluded_titles=excluded)
        for r in results:
            assert r["title"] not in excluded

    def test_search_returns_empty_on_no_queries(self):
        retriever = _make_mock_retriever(_METADATA)
        assert retriever.search([]) == []

    def test_retrieval_score_attached(self):
        retriever = _make_mock_retriever(_METADATA, top_k=5)
        with patch("backend.rag.faiss_retriever.embed_queries",
                   return_value=np.ones((1, 384), dtype=np.float32)):
            results = retriever.search(["comedy"])
        for r in results:
            assert "_retrieval_score" in r

    def test_search_count_respects_top_n(self):
        retriever = _make_mock_retriever(_METADATA, top_k=5)
        with patch("backend.rag.faiss_retriever.embed_queries",
                   return_value=np.ones((1, 384), dtype=np.float32)):
            results = retriever.search(["movies"], top_n=2)
        assert len(results) <= 2

    def test_search_broad_returns_list(self):
        retriever = _make_mock_retriever(_METADATA, top_k=5)
        with patch("backend.rag.faiss_retriever.embed_queries",
                   return_value=np.ones((3, 384), dtype=np.float32)):
            results = retriever.search_broad()
        assert isinstance(results, list)
