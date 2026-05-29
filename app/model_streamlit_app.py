"""
model_streamlit_app.py
───────────────────────
Premium SaaS-level Streamlit frontend for the AI Adaptive Onboarding Engine.

Architecture
────────────
  UI (this file) → OnboardingEngine.analyze() → result dict → render dashboard

All analysis results are stored in st.session_state so they survive
Streamlit's top-to-bottom rerun cycle.  The model is processed exactly
ONCE per click via the "Analyze" button guard.
"""

import sys
import os
import tempfile

import streamlit as st

# ── Page config (first Streamlit call) ────────────────────────────────────────
st.set_page_config(
    page_title="AI Adaptive Onboarding Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Resolve project root and fix imports ──────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APP_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.inference import OnboardingEngine


# ═════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS
# ═════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
/* ── Import font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ──────────────────────────────────────────────────────── */
:root {
    --bg-primary:    #0f1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2333;
    --bg-card-hover: #222d3f;
    --border-subtle: #2d3748;
    --border-accent: #3b82f6;
    --text-primary:  #e2e8f0;
    --text-secondary:#94a3b8;
    --text-muted:    #64748b;
    --accent-blue:   #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-green:  #10b981;
    --accent-amber:  #f59e0b;
    --accent-red:    #ef4444;
    --accent-cyan:   #06b6d4;
    --gradient-main: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    --gradient-green:linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
    --shadow-card:   0 4px 24px rgba(0,0,0,0.25);
    --shadow-glow:   0 0 30px rgba(59,130,246,0.15);
    --radius:        12px;
    --radius-sm:     8px;
    --radius-xs:     6px;
}

/* ── Global Overrides ────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}

.block-container {
    padding-top: 2rem !important;
    max-width: 1300px !important;
}

/* ── Hide default Streamlit branding ─────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar        { width: 6px; }
::-webkit-scrollbar-track  { background: var(--bg-primary); }
::-webkit-scrollbar-thumb  { background: var(--border-subtle); border-radius: 3px; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-secondary);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    padding: 10px 24px;
    font-weight: 500;
    font-size: 14px;
    color: var(--text-secondary);
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-card);
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
div.stButton > button[kind="primary"],
div.stButton > button:first-child {
    border: none;
    border-radius: var(--radius-sm);
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.3px;
    padding: 12px 28px;
    transition: all 0.2s ease;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border-subtle);
    border-radius: var(--radius);
    padding: 12px;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue);
}

/* ── Text area ───────────────────────────────────────────────────────────── */
.stTextArea textarea {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 20px 24px;
    box-shadow: var(--shadow-card);
}
[data-testid="stMetricLabel"]  { color: var(--text-secondary) !important; font-weight: 500 !important; }
[data-testid="stMetricValue"]  { font-weight: 700 !important; font-size: 28px !important; }

/* ── Progress bar ────────────────────────────────────────────────────────── */
.stProgress > div > div {
    background: var(--gradient-main) !important;
    border-radius: 20px;
}
.stProgress > div {
    background: var(--bg-secondary) !important;
    border-radius: 20px;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 14px !important;
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

/* ── Hero Header ─────────────────────────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 40px 20px 30px;
    margin-bottom: 10px;
}
.hero-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    margin-top: 14px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    background: rgba(59,130,246,0.12);
    color: var(--accent-blue);
    border: 1px solid rgba(59,130,246,0.25);
}

/* ── Section Headers ─────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-subtle);
}
.section-header h2 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
}
.section-header .section-icon {
    font-size: 1.3rem;
}

/* ── Score Ring ───────────────────────────────────────────────────────────── */
.score-ring-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px;
}
.score-ring {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 12px;
    position: relative;
    box-shadow: var(--shadow-glow);
}
.score-ring::before {
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 50%;
    padding: 3px;
    background: var(--gradient-main);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
}
.score-label {
    font-size: 14px;
    color: var(--text-secondary);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Stat Cards ──────────────────────────────────────────────────────────── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    box-shadow: var(--shadow-card);
    transition: all 0.2s ease;
}
.stat-card:hover {
    border-color: var(--border-accent);
    transform: translateY(-2px);
    box-shadow: var(--shadow-glow);
}
.stat-number {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.stat-label {
    font-size: 13px;
    color: var(--text-secondary);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Skill Badges ────────────────────────────────────────────────────────── */
.skill-badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.skill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.skill-badge-green {
    background: rgba(16,185,129,0.1);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.25);
}
.skill-badge-red {
    background: rgba(239,68,68,0.1);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.25);
}
.skill-badge-blue {
    background: rgba(59,130,246,0.1);
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.25);
}
.skill-badge-amber {
    background: rgba(245,158,11,0.1);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.25);
}
.skill-badge:hover {
    transform: translateY(-1px);
}

/* ── Roadmap Cards ───────────────────────────────────────────────────────── */
.roadmap-phase-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 16px;
    padding: 14px 20px;
    background: var(--bg-secondary);
    border-radius: var(--radius);
    border-left: 4px solid var(--accent-blue);
}
.roadmap-phase-header h3 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
}
.roadmap-phase-header .phase-icon {
    font-size: 1.2rem;
}

.roadmap-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    gap: 18px;
    transition: all 0.2s ease;
}
.roadmap-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-glow);
    transform: translateX(4px);
}
.roadmap-card.highlight {
    border-color: var(--accent-amber);
    background: rgba(245,158,11,0.04);
    box-shadow: 0 0 20px rgba(245,158,11,0.08);
}

.roadmap-step-number {
    min-width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--gradient-main);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 15px;
    color: #fff;
    flex-shrink: 0;
}
.roadmap-step-number.prereq {
    background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%);
}

.roadmap-card-body {
    flex: 1;
}
.roadmap-card-body h4 {
    margin: 0 0 6px;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
}
.roadmap-card-meta {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 8px;
    flex-wrap: wrap;
}
.roadmap-meta-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 500;
}
.roadmap-link {
    color: var(--accent-blue);
    text-decoration: none;
    font-size: 12px;
    font-weight: 600;
    transition: color 0.15s;
}
.roadmap-link:hover {
    color: var(--accent-purple);
    text-decoration: underline;
}

.roadmap-type-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.type-required {
    background: rgba(59,130,246,0.12);
    color: var(--accent-blue);
}
.type-prerequisite {
    background: rgba(148,163,184,0.12);
    color: var(--text-secondary);
}

/* ── Roadmap Summary Bar ─────────────────────────────────────────────────── */
.roadmap-summary {
    display: flex;
    justify-content: space-around;
    padding: 20px;
    background: var(--bg-secondary);
    border-radius: var(--radius);
    border: 1px solid var(--border-subtle);
    margin-bottom: 24px;
}
.roadmap-summary-item {
    text-align: center;
}
.roadmap-summary-item .num {
    font-size: 1.5rem;
    font-weight: 800;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.roadmap-summary-item .label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
}

/* ── Next Skill Highlight ────────────────────────────────────────────────── */
.next-skill-banner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 20px;
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: var(--radius);
    margin-bottom: 20px;
}
.next-skill-banner .icon {
    font-size: 1.4rem;
}
.next-skill-banner .text {
    font-size: 14px;
    color: var(--text-primary);
}
.next-skill-banner .text strong {
    color: var(--accent-amber);
}

/* ── Input Section Card ──────────────────────────────────────────────────── */
.input-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 16px;
}
.input-card-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Method Badge ────────────────────────────────────────────────────────── */
.method-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.method-model {
    background: rgba(139,92,246,0.12);
    color: var(--accent-purple);
    border: 1px solid rgba(139,92,246,0.25);
}
.method-rule {
    background: rgba(6,182,212,0.12);
    color: var(--accent-cyan);
    border: 1px solid rgba(6,182,212,0.25);
}
</style>
"""


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════

_STATE_DEFAULTS = {
    "analysis_results": None,
    "error_message": None,
}

for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _clear_results() -> None:
    for key, default in _STATE_DEFAULTS.items():
        st.session_state[key] = default


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _save_upload_to_tempfile(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    try:
        data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            return tmp.name
    except Exception:
        return None


def _score_color(score: float) -> str:
    """Return a CSS color string based on match score."""
    if score >= 0.7:
        return "var(--accent-green)"
    if score >= 0.4:
        return "var(--accent-amber)"
    return "var(--accent-red)"


def _render_skill_badges(skills: list[str], css_class: str) -> str:
    """Generate HTML for a list of skill badges."""
    if not skills:
        return '<p style="color:var(--text-muted); font-size:14px; font-style:italic;">None detected</p>'
    badges = "".join(
        f'<span class="skill-badge {css_class}">{s}</span>' for s in skills
    )
    return f'<div class="skill-badge-container">{badges}</div>'


# ═════════════════════════════════════════════════════════════════════════════
# MODEL (loaded once)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _load_engine() -> OnboardingEngine:
    ckpt = os.path.join(_PROJECT_ROOT, "checkpoints", "best_model.pt")
    if not os.path.exists(ckpt):
        ckpt = None
    return OnboardingEngine(checkpoint=ckpt)


engine = _load_engine()


# ═════════════════════════════════════════════════════════════════════════════
# INJECT CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(_CSS, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# UI — HERO HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <h1>🧠 AI Adaptive Onboarding Engine</h1>
    <p>AI‑powered resume & job description analysis with personalized skill‑gap roadmaps</p>
    <span class="hero-badge">✦ POWERED BY CUSTOM TRANSFORMER</span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# UI — INPUTS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="section-header">
    <span class="section-icon">📂</span>
    <h2>Upload Documents</h2>
</div>
""", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    st.markdown("""
    <div class="input-card-title">📄 Resume</div>
    """, unsafe_allow_html=True)
    resume_file = st.file_uploader(
        "Upload Resume PDF", type=["pdf"], label_visibility="collapsed"
    )
    resume_text_input = st.text_area(
        "Or paste resume text", height=140, placeholder="Paste your resume text here…"
    )

with right:
    st.markdown("""
    <div class="input-card-title">📑 Job Description</div>
    """, unsafe_allow_html=True)
    jd_file = st.file_uploader(
        "Upload JD PDF", type=["pdf"], label_visibility="collapsed"
    )
    jd_text_input = st.text_area(
        "Or paste JD text", height=140, placeholder="Paste the job description here…"
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

btn_left, btn_right = st.columns(2, gap="medium")

with btn_left:
    analyze_clicked = st.button(
        "🚀  Analyze with AI", use_container_width=True, type="primary"
    )
with btn_right:
    st.button(
        "🗑️  Clear Results",
        use_container_width=True,
        on_click=_clear_results,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS (runs exactly once per click)
# ═════════════════════════════════════════════════════════════════════════════

if analyze_clicked:
    _clear_results()

    resume_text = resume_text_input.strip() or None
    resume_path = _save_upload_to_tempfile(resume_file) if not resume_text else None

    if not resume_text and not resume_path:
        st.warning("⚠️ Please provide a resume — upload a PDF or paste text.")
        st.stop()

    jd_text = jd_text_input.strip() or None
    jd_path = _save_upload_to_tempfile(jd_file) if not jd_text else None

    if not jd_text and not jd_path:
        st.warning("⚠️ Please provide a job description — upload a PDF or paste text.")
        st.stop()

    with st.spinner("🔍 Running AI analysis…"):
        try:
            result = engine.analyze(
                resume_text=resume_text,
                resume_pdf=resume_path,
                jd_text=jd_text,
                jd_pdf=jd_path,
            )
            st.session_state.analysis_results = result
            st.session_state.error_message = None
        except Exception as exc:
            st.session_state.analysis_results = None
            st.session_state.error_message = str(exc)

    for path in (resume_path, jd_path):
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# ERROR BANNER
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.error_message:
    st.error(f"❌ Analysis failed: {st.session_state.error_message}")


# ═════════════════════════════════════════════════════════════════════════════
# UI — RESULTS DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

result = st.session_state.analysis_results

if result:
    # ── Method badge ──────────────────────────────────────────────────────
    method = result.get("method", "unknown")
    badge_cls = "method-model" if method == "model" else "method-rule"
    badge_icon = "🤖" if method == "model" else "📏"
    st.markdown(f"""
    <div style="text-align:center; margin: 10px 0 4px;">
        <span class="method-badge {badge_cls}">{badge_icon} {method.upper()} INFERENCE</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_dashboard, tab_skills, tab_roadmap = st.tabs([
        "📊  Dashboard", "🧠  Skills Analysis", "🗺️  Learning Roadmap"
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — DASHBOARD
    # ══════════════════════════════════════════════════════════════════════
    with tab_dashboard:
        match_score = float(result.get("match_score", 0))
        pct = round(match_score * 100)
        color = _score_color(match_score)

        resume_skills = result.get("resume_skills", [])
        jd_skills = result.get("jd_skills", [])
        missing_skills = result.get("missing_skills", [])
        gap = result.get("gap_analysis", {})
        matched = gap.get("matched", [])

        # ── Row 1: Score ring + stat cards ────────────────────────────────
        score_col, stats_col = st.columns([1, 2], gap="large")

        with score_col:
            st.markdown(f"""
            <div class="score-ring-container">
                <div class="score-ring" style="background: var(--bg-card); color: {color};">
                    {pct}%
                </div>
                <div class="score-label">Match Score</div>
            </div>
            """, unsafe_allow_html=True)

        with stats_col:
            s1, s2, s3, s4 = st.columns(4, gap="small")
            stats = [
                (len(resume_skills), "Resume Skills", "var(--accent-blue)"),
                (len(jd_skills),     "JD Required",   "var(--accent-purple)"),
                (len(matched),       "Matched",       "var(--accent-green)"),
                (len(missing_skills),"Gaps Found",    "var(--accent-red)"),
            ]
            for col, (num, label, c) in zip([s1, s2, s3, s4], stats):
                col.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color:{c}">{num}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Row 2: Progress bar ───────────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.progress(max(0.0, min(match_score, 1.0)))

        # ── Row 3: Quick glance skills ────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        with st.expander("🔍 Quick Skill Glance", expanded=False):
            gl, gr = st.columns(2)
            with gl:
                st.markdown("**✅ Matched Skills**", unsafe_allow_html=True)
                st.markdown(
                    _render_skill_badges(matched, "skill-badge-green"),
                    unsafe_allow_html=True,
                )
            with gr:
                st.markdown("**❌ Missing Skills**", unsafe_allow_html=True)
                st.markdown(
                    _render_skill_badges(missing_skills, "skill-badge-red"),
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — SKILLS ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    with tab_skills:
        resume_skills = result.get("resume_skills", [])
        jd_skills = result.get("jd_skills", [])
        missing_skills = result.get("missing_skills", [])
        gap = result.get("gap_analysis", {})
        matched = gap.get("matched", [])
        extra = gap.get("extra", [])

        # ── Full breakdown ────────────────────────────────────────────────
        sk_left, sk_right = st.columns(2, gap="large")

        with sk_left:
            st.markdown("""
            <div class="section-header">
                <span class="section-icon">📄</span>
                <h2>Your Resume Skills</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                _render_skill_badges(resume_skills, "skill-badge-blue"),
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                <span class="section-icon">✅</span>
                <h2>Matched with JD</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                _render_skill_badges(matched, "skill-badge-green"),
                unsafe_allow_html=True,
            )

        with sk_right:
            st.markdown("""
            <div class="section-header">
                <span class="section-icon">📑</span>
                <h2>JD Required Skills</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                _render_skill_badges(jd_skills, "skill-badge-blue"),
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                <span class="section-icon">❌</span>
                <h2>Skills to Acquire</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                _render_skill_badges(missing_skills, "skill-badge-red"),
                unsafe_allow_html=True,
            )

        # ── Bonus skills (in resume but not required) ─────────────────────
        if extra:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            with st.expander("💡 Bonus Skills (in your resume but not required by JD)"):
                st.markdown(
                    _render_skill_badges(extra, "skill-badge-amber"),
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — LEARNING ROADMAP
    # ══════════════════════════════════════════════════════════════════════
    with tab_roadmap:
        roadmap = result.get("roadmap", {})

        if isinstance(roadmap, dict) and "learning_plan" in roadmap:
            learning_plan = roadmap.get("learning_plan", [])
            phases = roadmap.get("phases", {})
            total_hours = roadmap.get("total_hours", 0)
            message = roadmap.get("message", "")

            if message:
                st.success(message)
            elif not learning_plan:
                st.success("🎉 Your resume already covers all required skills!")
            else:
                # ── Summary bar ────────────────────────────────────────
                n_required = sum(1 for i in learning_plan if i.get("type") == "required")
                n_prereq = sum(1 for i in learning_plan if i.get("type") == "prerequisite")

                summary_html = (
                    '<div class="roadmap-summary">'
                    f'<div class="roadmap-summary-item"><div class="num">{len(learning_plan)}</div><div class="label">Total Skills</div></div>'
                    f'<div class="roadmap-summary-item"><div class="num">{n_required}</div><div class="label">Required</div></div>'
                    f'<div class="roadmap-summary-item"><div class="num">{n_prereq}</div><div class="label">Prerequisites</div></div>'
                    f'<div class="roadmap-summary-item"><div class="num">{total_hours}h</div><div class="label">Est. Time</div></div>'
                    f'<div class="roadmap-summary-item"><div class="num">{len(phases)}</div><div class="label">Phases</div></div>'
                    '</div>'
                )
                st.markdown(summary_html, unsafe_allow_html=True)

                # ── Next recommended skill highlight ──────────────────
                if learning_plan:
                    first = learning_plan[0]
                    skill_title = first["skill"].title()
                    hours = first.get("hours", 15)
                    banner_html = (
                        '<div class="next-skill-banner">'
                        '<span class="icon">⭐</span>'
                        '<span class="text">'
                        f'<strong>Start here →</strong> We recommend learning '
                        f'<strong>{skill_title}</strong> first (~{hours}h)'
                        '</span>'
                        '</div>'
                    )
                    st.markdown(banner_html, unsafe_allow_html=True)

                # ── Build a lookup: skill → learning_plan item ────────
                plan_lookup = {item["skill"]: item for item in learning_plan}

                # ── Phase icons ──────────────────────────────────────
                phase_icons = {
                    "Phase 1 — Foundations": "🏗️",
                    "Phase 2 — Core Skills": "⚡",
                    "Phase 3 — Advanced": "🚀",
                }

                # ── Render each phase ────────────────────────────────
                global_step = 0
                for phase_name, phase_skills in phases.items():
                    icon = phase_icons.get(phase_name, "📌")
                    phase_header = (
                        '<div class="roadmap-phase-header">'
                        f'<span class="phase-icon">{icon}</span>'
                        f'<h3>{phase_name}</h3>'
                        '</div>'
                    )
                    st.markdown(phase_header, unsafe_allow_html=True)

                    for skill_name in phase_skills:
                        global_step += 1
                        item = plan_lookup.get(skill_name, {})
                        skill_type = item.get("type", "required")
                        hours = item.get("hours", 15)
                        days = item.get("days", 3)
                        level = item.get("level", 2)
                        resources = item.get("resources", [])
                        deps = item.get("depends_on", [])

                        type_cls = "type-required" if skill_type == "required" else "type-prerequisite"
                        step_cls = "" if skill_type == "required" else "prereq"
                        highlight_cls = "highlight" if global_step == 1 else ""

                        # Level badge colors
                        level_colors = {
                            1: ("rgba(16,185,129,0.12)", "#34d399"),
                            2: ("rgba(59,130,246,0.12)", "#60a5fa"),
                            3: ("rgba(139,92,246,0.12)", "#a78bfa"),
                        }
                        lbg, lfg = level_colors.get(level, level_colors[2])

                        # Build resource links
                        links_parts = []
                        for res in resources[:2]:
                            title = res.get("title", "Learn")
                            url = res.get("url", "#")
                            links_parts.append(
                                f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="roadmap-link">📚 {title}</a>'
                            )
                        links_html = " ".join(links_parts)

                        # Build deps
                        deps_html = ""
                        if deps:
                            deps_str = ", ".join(d.title() for d in deps[:3])
                            more = f" +{len(deps)-3}" if len(deps) > 3 else ""
                            deps_html = f'<span class="roadmap-meta-item">🔗 Requires: {deps_str}{more}</span>'

                        card_html = (
                            f'<div class="roadmap-card {highlight_cls}">'
                            f'<div class="roadmap-step-number {step_cls}">{global_step}</div>'
                            '<div class="roadmap-card-body">'
                            f'<h4>{skill_name.title()}</h4>'
                            '<div style="margin-bottom:6px; display:flex; gap:8px; align-items:center; flex-wrap:wrap;">'
                            f'<span class="roadmap-type-badge {type_cls}">{skill_type}</span>'
                            f'<span class="roadmap-type-badge" style="background:{lbg}; color:{lfg};">L{level}</span>'
                            '</div>'
                            '<div class="roadmap-card-meta">'
                            f'<span class="roadmap-meta-item">⏱ {hours}h ({days} days)</span>'
                            f'{deps_html}'
                            '</div>'
                            '<div class="roadmap-card-meta" style="margin-top:6px;">'
                            f'{links_html}'
                            '</div>'
                            '</div>'
                            '</div>'
                        )
                        st.markdown(card_html, unsafe_allow_html=True)

        elif isinstance(roadmap, list):
            for i, step in enumerate(roadmap, 1):
                card_html = (
                    '<div class="roadmap-card">'
                    f'<div class="roadmap-step-number">{i}</div>'
                    '<div class="roadmap-card-body">'
                    f'<h4>{step}</h4>'
                    '</div>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.info("No roadmap data available.")