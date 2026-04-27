# CineMatch | AI Movie Concierge

CineMatch is a premium, emotionally intelligent movie recommendation system. Unlike traditional search engines, it doesn't ask what you want to watch—it asks how you feel. 

Using **Retrieval-Augmented Generation (RAG)**, CineMatch interprets your mood (even from emojis or vague text) and provides personalized recommendations grounded in a curated database of 780+ movies and shows.

---

## ✨ Features

- **Mood-Adaptive Intelligence**: Interprets complex emotions, emojis, and slang using Groq's Llama 3.
- **RAG Architecture**: Recommendations are grounded in a local FAISS vector database to prevent AI hallucinations.
- **Premium Neutral UI**: A high-end, minimalist dark theme built with React and modern CSS.
- **Instant Results**: Powered by Groq LPU™ for near-zero latency generation.
- **Session Awareness**: Remembers your previous feedback to refine results in real-time.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18 + Vite (Vanilla CSS) |
| **Backend** | FastAPI (Python 3.11+) |
| **LLM** | Groq (Llama 3.3 70B) |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Data Sources** | TMDB · OMDB · RapidAPI · TVmaze |

---

## 🔌 Data Sources

To build its comprehensive movie library, CineMatch leverages four major APIs during the data ingestion phase:

1.  **TMDB (The Movie Database)**: The primary source for movie/show metadata, overviews, and genres.
2.  **OMDB (Open Movie Database)**: Provides accurate IMDb ratings and detailed plot summaries.
3.  **RapidAPI (Streaming Availability)**: Adds real-time catalog data for platforms like Netflix, Disney+, and Prime Video.
4.  **TVMaze**: Used to enrich TV show data with episode details and network information.

---

## 🏗️ Architecture Flowchart

```text
       ┌──────────────┐
       │  User Input  │ (Text/Emojis)
       └──────┬───────┘
              │ 1. Raw Input
              ▼
       ┌──────────────┐
       │   React UI   │ (State Management)
       └──────┬───────┘
              │ 2. POST /recommend
              ▼
       ┌──────────────┐
       │FastAPI Server│ (Orchestrator)
       └──────┬───────┘
              │ 3. Dispatch
      ┌───────┴───────┐
      │               │
      ▼               ▼
┌───────────┐   ┌───────────┐
│Mood Agent │   │ FAISS RAG │ (Semantic Search)
└─────┬─────┘   └─────┬─────┘
      │ 4. Mood       │ 5. Candidate
      │    Data       │    Movies
      └───────┬───────┘
              │ 6. Combined Context
              ▼
       ┌──────────────┐
       │Groq (Llama 3)│ (Generative AI)
       └──────┬───────┘
              │ 7. Personalized Explanation
              ▼
       ┌──────────────┐
       │Final Results │ (UI Render)
       └──────────────┘
```

### 🔄 System Flow Breakdown

1.  **Ingestion**: The user provides free-form text or emojis representing their current mood.
2.  **Analysis**: The **Mood Agent** (LLM Call 1) extracts emotional themes and intensity.
3.  **Retrieval**: The **FAISS RAG** engine performs a semantic vector search to find real movies from the database that match the extracted themes.
4.  **Synthesis**: **Groq (Llama 3)** receives the retrieved movies and the user's mood to generate warm, personalized "Why you'll like this" explanations.
5.  **Delivery**: The **React UI** renders the results as high-quality cards with streaming platform details.

---

## 📂 Project Structure

```text
mood-recommender/
├── backend/                # FastAPI logic
│   ├── agent/              # Mood interpretation & Intent logic
│   ├── data/               # Movie Library & Vector Database
│   ├── llm/                # Prompt templates & LLM Generator
│   ├── pipeline/           # Orchestration of the RAG flow
│   ├── rag/                # FAISS retrieval logic
│   └── main.py             # Server Entry Point
├── frontend/               # React Application
│   ├── src/                # Components, Hooks, and Styles
│   └── index.html          # Frontend Entry Point
└── .env                    # API Keys (TMDB, Groq, etc.)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js & npm
- API Keys: Groq, TMDB, OMDB, RapidAPI (Streaming)

### 1. Setup Backend
```bash
cd mood-recommender
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate  

pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### 2. Setup Frontend
```bash
cd mood-recommender/frontend
npm install
npm run dev
```

### 3. Open the App
Visit **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 🧠 How it Works (RAG Pipeline)

1.  **Intent Detection**: Checks if your input is interpretable.
2.  **Mood Extraction**: The AI converts "I've had a rough day 😔" into structured data (Themes: healing, intensity: high).
3.  **FAISS Retrieval**: The system searches the vector database for movies that match those themes mathematically.
4.  **LLM Generation**: Groq takes the results and writes a personalized explanation for why each movie fits your specific mood.
5.  **Refinement**: You can tell the AI "Make it more upbeat" to instantly get a new set of tailored results.

---

## 📜 License
MIT
