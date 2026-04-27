"""
main.py
───────
FastAPI application entry point for the CineMatch backend.

Responsibilities:
    - Create and configure the FastAPI app instance
    - Enable CORS for the Streamlit frontend
    - Register all API routers
    - Validate required environment variables at startup

Usage:
    uvicorn backend.main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import ALLOWED_ORIGINS, validate_required_keys
from backend.routes import router

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("backend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Validate environment on startup ───────────────────────────────────────────
validate_required_keys()

# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="CineMatch API",
    description=(
        "An emotionally intelligent AI recommendation system that interprets "
        "any form of mood input and returns personalized movie/show suggestions "
        "grounded in FAISS vector retrieval and Gemini 1.5 Flash generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow Streamlit frontend ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ───────────────────────────────────────────────────────────
app.include_router(router)

@app.get("/")
async def root():
    """Welcome message for the API."""
    return {
        "message": "Welcome to the CineMatch API!",
        "docs": "/docs",
        "health": "/health"
    }

logger.info("CineMatch API started.")
