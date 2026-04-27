"""
config.py
─────────
Central configuration module for the Mood-Adaptive Recommender backend.

Loads all environment variables from a .env file using python-dotenv.
All other modules MUST import their secrets from here — never hardcode values.

Raises:
    RuntimeError: If a required API key is missing at startup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Resolve .env path relative to project root ────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=_ENV_PATH)

# ─── API Keys ───────────────────────────────────────────────────────────────
TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
OMDB_API_KEY: str = os.getenv("OMDB_API_KEY", "")
RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "")
TVMAZE_BASE_URL: str = os.getenv("TVMAZE_BASE_URL", "https://api.tvmaze.com")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ─── FAISS / Embeddings ──────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH: str = str(
    Path(__file__).resolve().parent / "data" / "embeddings" / "faiss_index.bin"
)
METADATA_STORE_PATH: str = str(
    Path(__file__).resolve().parent / "data" / "embeddings" / "metadata_store.json"
)

# ─── Retrieval settings ──────────────────────────────────────────────────────
TOP_K_RESULTS: int = 10          # candidates per query
MAX_RECOMMENDATIONS: int = 5     # final recs shown to user
MIN_RECOMMENDATIONS: int = 3

# ─── Gemini settings ─────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-1.5-flash"
GEMINI_TEMPERATURE: float = 0.7

# ─── CORS ────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]

# ─── Validation ──────────────────────────────────────────────────────────────
def validate_required_keys() -> None:
    """
    Validate that all critical environment variables are set.

    Raises:
        RuntimeError: If GEMINI_API_KEY is missing (hard requirement).
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )
