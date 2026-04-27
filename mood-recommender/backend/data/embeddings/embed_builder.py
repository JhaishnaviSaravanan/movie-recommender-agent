"""
embed_builder.py
────────────────
Builds and saves a FAISS vector index from the unified movie/show dataset.

Uses sentence-transformers/all-MiniLM-L6-v2 to embed a rich text
representation of each record.  The resulting index and metadata store are
written to backend/data/embeddings/.

Usage:
    python -m backend.data.embeddings.embed_builder

Outputs:
    faiss_index.bin      — binary FAISS IVFFlat index
    metadata_store.json  — list of dicts matching each FAISS vector by position
"""

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, METADATA_STORE_PATH
from backend.data.preprocessor import build_unified_dataset

logger = logging.getLogger(__name__)


def _build_text(record: dict[str, Any]) -> str:
    """
    Construct a rich text representation of a movie/show for embedding.

    Concatenates the most semantically useful fields so the embedding
    captures mood, theme, genre, and narrative context — not just the title.

    Args:
        record (dict[str, Any]): A unified movie/show dict.

    Returns:
        str: Single text string to be embedded.
    """
    parts = [
        record.get("title", ""),
        f"({record.get('year', '')})",
        record.get("type", ""),
        "Genres: " + ", ".join(record.get("genres", [])),
        record.get("overview", ""),
        "Cast: " + ", ".join(record.get("cast", [])[:5]),
        "Keywords: " + ", ".join(record.get("keywords", [])),
        "Platforms: " + ", ".join(record.get("platforms", [])),
        "Rating: " + record.get("imdb_rating", ""),
        "Awards: " + record.get("awards", ""),
    ]
    return " | ".join(p for p in parts if p.strip() and p.strip() not in ("|", ""))


def build_faiss_index(records: list[dict[str, Any]] | None = None) -> None:
    """
    Build a FAISS index from the unified dataset and persist it to disk.

    Args:
        records (list[dict[str, Any]] | None): Pre-loaded records.  If None,
            the preprocessor is called to build the dataset from raw files.

    Raises:
        RuntimeError: If the dataset is empty after preprocessing.
    """
    # ── Load dataset ──────────────────────────────────────────────────────────
    if records is None:
        logger.info("Building unified dataset from raw files…")
        records = build_unified_dataset()

    if not records:
        raise RuntimeError(
            "No records found. Run api_fetcher.py first to populate raw data."
        )

    logger.info("Building embeddings for %d records…", len(records))

    # ── Embed each record ─────────────────────────────────────────────────────
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [_build_text(r) for r in records]
    embeddings: np.ndarray = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalization → cosine via inner product
    )

    dimension = embeddings.shape[1]
    logger.info("Embedding dimension: %d", dimension)

    # ── Build FAISS index (IVFFlat for speed at scale) ────────────────────────
    n_clusters = min(100, len(records) // 10 or 1)
    quantizer = faiss.IndexFlatIP(dimension)            # inner product = cosine (normalized)
    index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)

    # Train and populate
    index.train(embeddings.astype(np.float32))
    index.add(embeddings.astype(np.float32))
    index.nprobe = min(10, n_clusters)                  # search probes for recall/speed balance

    logger.info("FAISS index built: %d vectors", index.ntotal)

    # ── Persist index ─────────────────────────────────────────────────────────
    index_path = Path(FAISS_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved → %s", index_path)

    # ── Persist metadata store ────────────────────────────────────────────────
    meta_path = Path(METADATA_STORE_PATH)
    meta_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Metadata store saved → %s", meta_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_faiss_index()
