# Mood-Adaptive Generative Movie/Show Recommender

An emotionally intelligent AI system that interprets **any form of mood input** — vague, emotional, emoji-only, or incomplete — and returns personalized movie/show recommendations grounded in FAISS semantic retrieval and Gemini 1.5 Flash generation.

> "I don't ask what you want to watch. I ask how you feel."

---

## Architecture

```
User Input (Streamlit)
        │
        ▼
┌─────────────────────┐
│  IntentDetector     │  ← Is input empty? If yes → clarify once
└─────────────────────┘
        │ interpretable
        ▼
┌─────────────────────┐
│  MoodExtractor      │  ← Gemini Call 1: interpret mood → JSON
│  (Gemini 1.5 Flash) │     {interpreted_mood, intensity, themes,
└─────────────────────┘      search_queries, confidence}
        │
        ▼
┌─────────────────────┐
│  FAISSRetriever     │  ← Multi-query semantic search (offline index)
│  (all-MiniLM-L6-v2) │     embeds queries → searches FAISS → merges results
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ RetrievalEvaluator  │  ← Score quality
│                     │     Good → proceed | Weak → retry broader query
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  GeminiGenerator    │  ← Gemini Call 2: generate warm, personalized
│  (Gemini 1.5 Flash) │     recommendations from retrieved candidates
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Streamlit UI       │  ← Display interpreted mood + recommendation cards
└─────────────────────┘
```

---

## COE Topic Coverage

| Topic | Where It Appears |
|---|---|
| **Generative AI** | Gemini 1.5 Flash — mood interpretation (Call 1) + recommendation generation (Call 2) |
| **Prompt Engineering** | 3-stage prompt chain in `backend/llm/prompt_templates.py` |
| **HuggingFace / Open Source** | `sentence-transformers/all-MiniLM-L6-v2` for FAISS embedding |
| **LLM + RAG** | FAISS multi-query retrieval feeds Gemini generation — LLM never retrieves directly |
| **Agentic AI** | Intent detection → mood extraction → quality gating → refinement loop (`pipeline/recommender_pipeline.py`) |
| **MCP Server** | `backend/mcp/mcp_server.py` exposes recommender as callable tool for other AI agents |
| **n8n** | `n8n/workflow_export.json` — scheduled daily FAISS refresh via `POST /refresh-data` |

---

## Folder Structure

```
mood-recommender/
│
├── backend/
│   ├── agent/
│   │   ├── intent_detector.py        # Never blocks on vague input
│   │   ├── mood_extractor.py         # Gemini Call 1 — free-form interpretation
│   │   ├── retrieval_evaluator.py    # Quality gate — proceed or retry broader
│   │   └── feedback_handler.py       # Session memory, title tracking
│   │
│   ├── data/
│   │   ├── api_fetcher.py            # Batch fetch from 4 APIs (run once)
│   │   ├── preprocessor.py           # Normalize + merge all sources
│   │   └── embeddings/
│   │       ├── embed_builder.py      # Build FAISS index
│   │       ├── faiss_index.bin       # Pre-built index (git-ignored)
│   │       └── metadata_store.json   # Metadata linked to vectors (git-ignored)
│   │
│   ├── rag/
│   │   ├── embed_query.py            # Embed single/batch queries
│   │   └── faiss_retriever.py        # Multi-query search, merge, deduplicate
│   │
│   ├── llm/
│   │   ├── prompt_templates.py       # 3-stage prompt chain
│   │   └── gemini_generator.py       # Gemini Call 2 — recommendation generation
│   │
│   ├── pipeline/
│   │   └── recommender_pipeline.py   # Orchestrates all components end-to-end
│   │
│   ├── mcp/
│   │   └── mcp_server.py             # MCP-compatible tool wrapper
│   │
│   ├── tests/
│   │   ├── test_agent.py
│   │   ├── test_rag.py
│   │   ├── test_llm.py
│   │   └── test_pipeline.py
│   │
│   ├── main.py                       # FastAPI app entry point
│   ├── routes.py                     # API route definitions
│   ├── config.py                     # Env var loader
│   └── requirements.txt
│
├── frontend/
│   └── app.py                        # Streamlit UI (dark glassmorphism theme)
│
├── n8n/
│   └── workflow_export.json          # Scheduled FAISS refresh workflow
│
├── notebooks/
│   ├── 01_api_exploration.ipynb
│   ├── 02_embedding_pipeline.ipynb
│   └── 03_end_to_end_demo.ipynb
│
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- API keys: TMDB, OMDB, RapidAPI (Streaming Availability), Google Gemini

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd mood-recommender
```

### 2. Create and configure `.env`

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

Required keys:
```
TMDB_API_KEY=your_key_here
OMDB_API_KEY=your_key_here
RAPIDAPI_KEY=your_key_here
GEMINI_API_KEY=your_key_here
TVMAZE_BASE_URL=https://api.tvmaze.com
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

> **Apple Silicon / CPU note:** If `faiss-cpu` fails, try:
> ```bash
> pip install faiss-cpu --no-cache-dir
> ```

---

## Data Ingestion (Run Once)

The system works from a pre-built FAISS index. APIs are **never** called during user queries.

### Step 1 — Fetch raw data from all 4 APIs

```bash
python -m backend.data.api_fetcher
```

This writes to `backend/data/raw/`:
- `tmdb_data.json`
- `omdb_data.json`
- `streaming_data.json`
- `tvmaze_data.json`

### Step 2 — Build the FAISS vector index

```bash
python -m backend.data.embeddings.embed_builder
```

This writes to `backend/data/embeddings/`:
- `faiss_index.bin`
- `metadata_store.json`

> Both files are git-ignored. Re-run these steps after any data refresh.

---

## Running the Application

### Backend (FastAPI)

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

Open: http://localhost:8501

### Both together (Docker)

```bash
docker-compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/recommend` | Get mood-based recommendations |
| `POST` | `/feedback` | Refine results based on user feedback |
| `POST` | `/refresh-data` | Trigger data re-ingestion (n8n target) |
| `GET` | `/health` | Liveness check |

### `POST /recommend`

```json
{
  "input": "I feel burnt out and need something easy",
  "session_id": null
}
```

**Response:**
```json
{
  "type": "recommendation",
  "session_id": "uuid-string",
  "interpreted_mood": {
    "interpreted_mood": "exhausted, comfort-seeking",
    "intensity": "high",
    "themes": ["escapism", "comfort", "low-effort"],
    "search_queries": ["cozy comfort shows", "easy feel-good series"],
    "confidence": "high"
  },
  "data": [
    {
      "title": "The Bear",
      "year": "2022",
      "genres": ["Drama"],
      "platforms": ["Disney+"],
      "imdb_rating": "8.7",
      "mood_tag": "⚡ Energizing",
      "explanation": "When you're burnt out, sometimes you need art that validates the feeling..."
    }
  ],
  "follow_up": "💭 Does this feel right? Tell me what to adjust."
}
```

### `POST /feedback`

```json
{
  "session_id": "uuid-string",
  "feedback": "Too intense, want something lighter",
  "shown_titles": ["The Bear", "Succession"]
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## MCP Server

The recommendation pipeline is also exposed as an MCP-compatible tool:

```bash
python -m backend.mcp.mcp_server
# Runs on http://localhost:8001
```

**List tools:** `GET /mcp/tools`

**Invoke:** `POST /mcp/tools/get_movie_recommendations`
```json
{
  "input": "something mysterious and slow-burn",
  "session_id": null
}
```

---

## Running Tests

```bash
# All tests
pytest backend/tests/ -v

# Individual test files
pytest backend/tests/test_agent.py -v
pytest backend/tests/test_rag.py -v
pytest backend/tests/test_llm.py -v
pytest backend/tests/test_pipeline.py -v
```

All external dependencies (Gemini, FAISS, sentence-transformers) are mocked in tests. No API keys required to run the test suite.

---

## n8n Workflow Import

1. Open your n8n instance
2. Go to **Workflows → Import from File**
3. Select `n8n/workflow_export.json`
4. Update the HTTP Request node URL if your backend runs on a different host/port
5. Activate the workflow

The workflow runs daily at midnight and calls `POST /refresh-data` to re-fetch all API data and rebuild the FAISS index.

---

## Input Examples

| Input | Interpreted As | Result Type |
|---|---|---|
| `"I feel melancholic"` | melancholic, reflective | 5 drama/emotional recs |
| `"idk"` | open, relaxed | 5 feel-good recs |
| `"🥺"` | sad, soft, emotional | 5 gentle comfort recs |
| `"aaaaaa"` | overwhelmed, stressed | 5 light escape recs |
| `"like Dark but easier"` | Dark as retrieval anchor | Sci-fi / thriller recs |
| `"burnt out"` | low-energy, comfort-seeking | 5 easy-watch recs |
| `""` (empty) | uninterpretable | One clarification question |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | FAISS (CPU, IVFFlat index) |
| LLM | Gemini 1.5 Flash (`google-generativeai`) |
| Agent | Custom agentic pipeline (intent → mood → retrieve → evaluate → generate) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit (dark glassmorphism theme) |
| Orchestration | n8n (scheduled FAISS refresh) |
| MCP | Custom FastAPI-based MCP server |
| Data Sources | TMDB · OMDB · Streaming Availability (RapidAPI) · TVmaze |

---

## Known Limitations

1. **In-memory sessions** — Session state resets on server restart. For production, replace `FeedbackHandler._sessions` dict with Redis.
2. **FAISS index is static** — New movies/shows only appear after running the ingestion pipeline again (automated via n8n nightly).
3. **Streaming platform data accuracy** — The Streaming Availability API catalog may not reflect real-time additions/removals.
4. **OMDB rate limits** — The free OMDB tier allows 1,000 requests/day; the ingestion script paginates slowly to stay within limits.
5. **Gemini availability** — If Gemini is unreachable, the system returns heuristic fallback recommendations (no personalized explanations).
6. **No poster images at runtime** — Poster URLs are stored in metadata but the Streamlit UI currently uses text-only cards.
7. **Docker setup** — The provided `docker-compose.yml` references `Dockerfile.backend` and `Dockerfile.frontend` which you'll need to create for containerised deployments.

---

## License

MIT
