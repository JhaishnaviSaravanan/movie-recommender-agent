"""
app.py
──────
Streamlit frontend for the Mood-Adaptive Generative Movie/Show Recommender.

Single-file UI that:
    - Accepts any form of mood input (text, emoji, vague phrases)
    - Displays what the system interpreted (transparency)
    - Shows recommendation cards with mood tags and personalized explanations
    - Supports inline refinement via the feedback section
    - Tracks session state across interactions

Usage:
    streamlit run frontend/app.py
"""

import uuid
import requests
import streamlit as st

# ── Backend URL ───────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Moodflix · AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS — dark, premium, glassmorphism design
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f1a 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 900px; }

/* ── Hero header ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 600;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* ── Input area ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 16px !important;
    color: #e2e8f0 !important;
    font-size: 1.1rem !important;
    padding: 1rem 1.2rem !important;
    resize: none !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.8) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
}

/* ── Mood interpretation banner ── */
.mood-banner {
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.95rem;
    color: #c4b5fd;
    animation: fadeIn 0.5s ease;
}

/* ── Recommendation card ── */
.rec-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    transition: all 0.3s ease;
    animation: slideUp 0.4s ease;
    position: relative;
    overflow: hidden;
}
.rec-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #60a5fa, #34d399);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.rec-card:hover {
    background: rgba(255,255,255,0.06);
    border-color: rgba(167,139,250,0.3);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.rec-card:hover::before { opacity: 1; }

.rec-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
}
.rec-meta {
    font-size: 0.82rem;
    color: #64748b;
    margin-bottom: 0.7rem;
    letter-spacing: 0.4px;
}
.mood-tag {
    display: inline-block;
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #c4b5fd;
    margin-bottom: 0.9rem;
}
.rec-explanation {
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.65;
    font-weight: 300;
}
.platform-badge {
    display: inline-block;
    background: rgba(96,165,250,0.12);
    border: 1px solid rgba(96,165,250,0.25);
    border-radius: 8px;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    color: #93c5fd;
    margin-right: 0.4rem;
    margin-top: 0.5rem;
}

/* ── Clarification card ── */
.clarification-card {
    background: rgba(234,179,8,0.07);
    border: 1px solid rgba(234,179,8,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    color: #fde68a;
    font-size: 1rem;
    animation: fadeIn 0.5s ease;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2rem 0;
}

/* ── Refine section ── */
.refine-header {
    font-size: 1rem;
    font-weight: 500;
    color: #94a3b8;
    margin-bottom: 0.6rem;
}

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Rating pill ── */
.rating-pill {
    display: inline-block;
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 8px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    color: #6ee7b7;
    margin-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "shown_titles" not in st.session_state:
    st.session_state.shown_titles = []

if "loading" not in st.session_state:
    st.session_state.loading = False


# ══════════════════════════════════════════════════════════════════════════════
# API helpers
# ══════════════════════════════════════════════════════════════════════════════

def call_recommend(user_input: str) -> dict | None:
    """
    POST /recommend and return the response dict, or None on error.

    Args:
        user_input (str): Raw mood text from the user.

    Returns:
        dict | None: API response or None if request failed.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/recommend",
            json={
                "input": user_input,
                "session_id": st.session_state.session_id,
            },
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.")
        return None
    except requests.HTTPError as e:
        st.error(f"⚠️ Backend error {e.response.status_code}: {e.response.json().get('detail', str(e))}")
        return None
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
        return None


def call_feedback(feedback_text: str) -> dict | None:
    """
    POST /feedback and return the refined response dict, or None on error.

    Args:
        feedback_text (str): User's refinement request.

    Returns:
        dict | None: Refined recommendation response or None.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "session_id": st.session_state.session_id,
                "feedback": feedback_text,
                "shown_titles": st.session_state.shown_titles,
            },
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ══════════════════════════════════════════════════════════════════════════════

def render_mood_banner(mood_data: dict) -> None:
    """
    Display the interpreted mood in a subtle banner.

    Args:
        mood_data (dict): Mood dict from the pipeline (interpreted_mood, intensity, etc.)
    """
    mood = mood_data.get("interpreted_mood", "open")
    intensity = mood_data.get("intensity", "medium")
    confidence = mood_data.get("confidence", "medium")
    conf_icon = {"high": "✦", "medium": "◈", "low": "◌"}.get(confidence, "◈")

    st.markdown(
        f"""
        <div class="mood-banner">
            <span style="font-size:1.3rem">🎭</span>
            <span>I picked up: <strong>{mood}</strong>
            &nbsp;·&nbsp; {intensity} intensity
            &nbsp;·&nbsp; {conf_icon} {confidence} confidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(rec: dict, index: int) -> None:
    """
    Render a single recommendation as a styled card.

    Args:
        rec   (dict): Recommendation dict from the generator.
        index (int):  Card position (for animation delay).
    """
    title = rec.get("title", "Unknown")
    year = rec.get("year", "N/A")
    genres = ", ".join(rec.get("genres", []))
    platforms = rec.get("platforms", [])
    imdb = rec.get("imdb_rating", "N/A")
    mood_tag = rec.get("mood_tag", "🌀 Immersive")
    explanation = rec.get("explanation", "")

    # Build platform badges HTML
    platform_html = "".join(
        f'<span class="platform-badge">{p}</span>' for p in platforms
    ) or '<span class="platform-badge">N/A</span>'

    # IMDb rating pill
    rating_html = f'<span class="rating-pill">⭐ {imdb}</span>' if imdb != "N/A" else ""

    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-title">🎬 {title} {rating_html}</div>
            <div class="rec-meta">{year} &nbsp;·&nbsp; {genres}</div>
            <div class="mood-tag">{mood_tag}</div>
            <div class="rec-explanation">{explanation}</div>
            <div style="margin-top:0.8rem">{platform_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_clarification(message: str) -> None:
    """
    Render a clarification prompt card.

    Args:
        message (str): The clarification message from the pipeline.
    """
    st.markdown(
        f'<div class="clarification-card">💬 {message}</div>',
        unsafe_allow_html=True,
    )


def render_results(result: dict) -> None:
    """
    Render either a clarification or a full recommendation set.

    Args:
        result (dict): Pipeline response dict.
    """
    resp_type = result.get("type")

    if resp_type == "clarification":
        render_clarification(result.get("follow_up", "Could you give me a hint?"))
        return

    # ── Mood interpretation banner ────────────────────────────────────────────
    mood_data = result.get("interpreted_mood")
    if mood_data:
        render_mood_banner(mood_data)

    # ── Recommendation cards ──────────────────────────────────────────────────
    recs = result.get("data") or []
    if not recs:
        st.warning("No recommendations found. Try a different input.")
        return

    for i, rec in enumerate(recs):
        render_recommendation_card(rec, i)

    # ── Follow-up prompt ──────────────────────────────────────────────────────
    follow_up = result.get("follow_up", "")
    if follow_up:
        st.markdown(
            f'<div style="color:#64748b; font-size:0.9rem; margin-top:1rem; text-align:center;">'
            f'{follow_up}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">Moodflix</div>'
    '<div class="hero-sub">Your emotionally intelligent movie concierge. '
    'Tell me how you feel — however you want to say it.</div>',
    unsafe_allow_html=True,
)

# ── Mood input ────────────────────────────────────────────────────────────────
user_input = st.text_area(
    label="mood_input",
    placeholder='How are you feeling? Even "idk" or "🥺" works 😊',
    height=110,
    label_visibility="collapsed",
    key="mood_input_area",
)

col1, col2 = st.columns([1, 6])
with col1:
    submit = st.button("✨ Recommend", key="submit_btn", use_container_width=True)

# ── Trigger recommendation ────────────────────────────────────────────────────
if submit and user_input is not None:
    with st.spinner("Reading your vibe…"):
        result = call_recommend(user_input.strip())

    if result:
        # Store session state
        st.session_state.session_id = result.get("session_id")
        st.session_state.last_result = result
        st.session_state.shown_titles = [
            r.get("title", "") for r in (result.get("data") or [])
        ]

# ── Render last result ────────────────────────────────────────────────────────
if st.session_state.last_result:
    render_results(st.session_state.last_result)

    # ── Divider ───────────────────────────────────────────────────────────────
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Refine section ────────────────────────────────────────────────────────
    st.markdown('<div class="refine-header">💭 Tell me what felt off…</div>', unsafe_allow_html=True)

    feedback_text = st.text_input(
        label="refine_input",
        placeholder="e.g. 'Too dark, want something lighter' or 'More sci-fi please'",
        label_visibility="collapsed",
        key="feedback_input",
    )

    if st.button("🔄 Refine", key="refine_btn") and feedback_text.strip():
        with st.spinner("Finding better matches…"):
            refined = call_feedback(feedback_text.strip())

        if refined:
            st.session_state.last_result = refined
            st.session_state.shown_titles = [
                r.get("title", "") for r in (refined.get("data") or [])
            ]
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center; color:#334155; font-size:0.78rem; margin-top:3rem;">'
    'Powered by Gemini · FAISS · sentence-transformers · FastAPI'
    '</div>',
    unsafe_allow_html=True,
)
