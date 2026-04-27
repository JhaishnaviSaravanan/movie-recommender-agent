"""
embed_query.py
──────────────
Embeds one or more natural-language queries using the same sentence-transformer
model used during index construction, ensuring embedding space compatibility.

The module loads the model lazily (on first call) and caches it for the
lifetime of the process — avoiding repeated model loads during a session.

Functions:
    embed_query(text)        — embed a single query string
    embed_queries(texts)     — batch embed multiple query strings
"""

import logging
from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Load and cache the sentence-transformer model (singleton).

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string into a normalised float32 vector.

    Args:
        text (str): The natural-language query to embed.

    Returns:
        np.ndarray: Shape (1, embedding_dim), dtype float32.
    """
    model = _get_model()
    vec: np.ndarray = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.astype(np.float32)


def embed_queries(texts: list[str]) -> np.ndarray:
    """
    Batch-embed multiple query strings.

    Args:
        texts (list[str]): A list of query strings.

    Returns:
        np.ndarray: Shape (len(texts), embedding_dim), dtype float32.

    Raises:
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("embed_queries requires at least one query string.")

    model = _get_model()
    vecs: np.ndarray = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)
