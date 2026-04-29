"""
app.py  —  Streamlit Frontend for the Adaptive Interview Agent
──────────────────────────────────────────────────────────────
Run with:
    streamlit run app.py

Architecture Note:
    LangGraph's graph.stream() runs continuously. For Streamlit we need
    a human-in-the-loop pause after ask_question. The cleanest solution:
    call the node functions directly (they're just Python functions) and
    store the full InterviewState in st.session_state. This avoids needing
    a LangGraph checkpointer while keeping all the business logic intact.

    Execution is split into two phases:
      Phase A — "run_to_question": retrieve → ask  (runs on init + after grading)
      Phase B — "process_answer":  grade → adjust → [loop or end]  (runs on submit)
"""

import streamlit as st
import pandas as pd
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Import node functions directly from agent ─────────────────────────────────
from agent import (
    create_initial_state,
    retrieve_question,
    ask_question,
    grade_answer,
    adjust_level,
    generate_final_report,
    should_continue,
    MAX_TURNS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adaptive Interview Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Overall background */
.stApp { background-color: #0f1117; }

/* Chat message styling */
.stChatMessage { border-radius: 12px; margin-bottom: 8px; }

/* Sidebar metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border: 1px solid #2e3350;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #8b95b0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #e8eaf6;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #6c7a9c;
    margin-top: 4px;
}

/* Level badge */
.level-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 6px;
}

/* Score bar */
.score-bar-wrap {
    background: #1a1f2e;
    border-radius: 8px;
    height: 8px;
    width: 100%;
    margin-top: 8px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
}

/* Weak area tags */
.weak-tag {
    display: inline-block;
    background: #2d1f3d;
    color: #c084fc;
    border: 1px solid #7c3aed;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 3px 2px;
}

/* Section headers */
.sidebar-section {
    font-size: 0.7rem;
    color: #4b5680;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 16px 0 8px 0;
    border-bottom: 1px solid #1e2130;
    padding-bottom: 4px;
}

/* Pulse animation for active session */
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
    70%  { box-shadow: 0 0 0 6px rgba(99, 102, 241, 0); }
    100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
}
.active-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #6366f1;
    border-radius: 50%;
    animation: pulse 1.8s infinite;
    margin-right: 6px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: merge partial state update
# ─────────────────────────────────────────────────────────────────────────────

def _apply(state: dict, updates: dict) -> dict:
    """Merge a node's partial update into the full state (handles messages list)."""
    for k, v in updates.items():
        if k == "messages":
            state["messages"] = state.get("messages", []) + v
        else:
            state[k] = v
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — run graph until a question is ready for the user
# ─────────────────────────────────────────────────────────────────────────────

def run_to_question(state: dict) -> dict:
    """retrieve_question → ask_question  (then pause for user input)."""
    state = _apply(state, retrieve_question(state))

    if state.get("session_complete"):
        state = _apply(state, generate_final_report(state))
        return state

    state = _apply(state, ask_question(state))
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — process user answer
# ─────────────────────────────────────────────────────────────────────────────

def process_answer(state: dict, user_answer: str) -> dict:
    """HumanMessage → grade_answer → adjust_level → [retrieve or final_report]."""
    state["messages"] = state.get("messages", []) + [HumanMessage(content=user_answer)]
    state = _apply(state, grade_answer(state))
    state = _apply(state, adjust_level(state))

    route = should_continue(state)
    if route == "generate_final_report":
        state = _apply(state, generate_final_report(state))
        state["session_complete"] = True
    else:
        state = run_to_question(state)

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_session(starting_level: int = 2):
    state = create_initial_state(starting_level=starting_level)
    state = run_to_question(state)
    st.session_state["interview_state"] = state
    st.session_state["awaiting_answer"] = True


if "interview_state" not in st.session_state:
    init_session(starting_level=2)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — level color & label
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_CONFIG = {
    1: {"color": "#22c55e", "bg": "#14532d", "label": "Beginner"},
    2: {"color": "#84cc16", "bg": "#365314", "label": "Intermediate"},
    3: {"color": "#f59e0b", "bg": "#451a03", "label": "Advanced"},
    4: {"color": "#f97316", "bg": "#431407", "label": "Expert"},
    5: {"color": "#ef4444", "bg": "#450a0a", "label": "Principal"},
}

def level_cfg(lvl):
    return LEVEL_CONFIG.get(lvl, LEVEL_CONFIG[2])

def score_color(score):
    if score >= 0.8: return "#22c55e"
    if score >= 0.6: return "#f59e0b"
    if score >= 0.4: return "#f97316"
    return "#ef4444"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Interview Agent")

    state = st.session_state["interview_state"]
    level       = state.get("current_level", 2)
    turn_count  = state.get("turn_count", 0)
    weak_areas  = state.get("weak_areas", [])
    score_hist  = state.get("score_history", [])
    last_score  = score_hist[-1]["score"] if score_hist else None
    cfg         = level_cfg(level)

    # ── Live status dot ───────────────────────────────────────────────────────
    if state.get("session_complete"):
        st.markdown("🔴 &nbsp; **Session complete**", unsafe_allow_html=True)
    else:
        st.markdown(
            f'<span class="active-dot"></span><span style="color:#8b95b0;font-size:0.85rem;">Session active</span>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # ── Current Level card ────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Current Difficulty</div>
            <div class="value" style="color:{cfg['color']};">{level} / 5</div>
            <span class="level-badge" style="background:{cfg['bg']};color:{cfg['color']};">
                {cfg['label']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Turn counter ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Questions Asked</div>
            <div class="value">{turn_count} <span style="font-size:1rem;color:#6c7a9c;">/ {MAX_TURNS}</span></div>
            <div class="sub">{"Interview complete" if state.get("session_complete") else f"{MAX_TURNS - turn_count} remaining"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Last score ────────────────────────────────────────────────────────────
    if last_score is not None:
        sc  = last_score
        clr = score_color(sc)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Last Score</div>
                <div class="value" style="color:{clr};">{sc:.0%}</div>
                <div class="score-bar-wrap">
                    <div class="score-bar-fill" style="width:{sc*100:.0f}%;background:{clr};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Weak areas ────────────────────────────────────────────────────────────
    if weak_areas:
        st.markdown('<div class="sidebar-section">Identified Weak Areas</div>', unsafe_allow_html=True)
        tags_html = "".join(f'<span class="weak-tag">{w.replace("_"," ")}</span>' for w in weak_areas)
        st.markdown(tags_html, unsafe_allow_html=True)

    # ── Score Progression Chart ───────────────────────────────────────────────
    if score_hist:
        st.markdown('<div class="sidebar-section">Score Progression</div>', unsafe_allow_html=True)
        df = pd.DataFrame([
            {
                "Turn": f"Q{h['turn']}",
                "Score": round(h["score"] * 100),
                "Level": h["level"],
            }
            for h in score_hist
        ])
        st.bar_chart(df.set_index("Turn")["Score"], use_container_width=True, color="#6366f1")

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Controls</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Restart", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with col2:
        start_lvl = st.selectbox("Start Level", [1, 2, 3, 4, 5], index=1, label_visibility="collapsed")

    if st.button("▶ New Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session(starting_level=start_lvl)
        st.rerun()

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem;color:#4b5680;line-height:1.8;">
        🔷 LangGraph &nbsp;·&nbsp; State machine<br>
        🟣 ChromaDB &nbsp;·&nbsp; Semantic RAG<br>
        🟠 all-MiniLM-L6-v2 &nbsp;·&nbsp; Embeddings<br>
        🟢 GPT-4o-mini &nbsp;·&nbsp; Grader LLM<br>
        🔵 Streamlit &nbsp;·&nbsp; Frontend
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<h2 style='margin-bottom:4px;'>Adaptive Technical Interview</h2>"
    "<p style='color:#6c7a9c;font-size:0.9rem;margin-top:0;'>Answer each question. The difficulty adapts in real-time based on your performance.</p>",
    unsafe_allow_html=True,
)
st.divider()

state      = st.session_state["interview_state"]
messages   = state.get("messages", [])
is_done    = state.get("session_complete", False)

# ── Render chat history ───────────────────────────────────────────────────────
for msg in messages:
    if isinstance(msg, SystemMessage):
        continue  # skip system prompt
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)

# ── Input box ─────────────────────────────────────────────────────────────────
if not is_done:
    placeholder = "Type your answer here and press Enter…"
    user_input  = st.chat_input(placeholder)

    if user_input and user_input.strip():
        # Show user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Process through the graph with a spinner
        with st.spinner("Evaluating your answer…"):
            updated_state = process_answer(
                st.session_state["interview_state"],
                user_input.strip()
            )
            st.session_state["interview_state"] = updated_state

        # Show new AI messages (everything added after user message)
        new_messages = updated_state["messages"][len(messages) + 1:]
        for msg in new_messages:
            if isinstance(msg, AIMessage):
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(msg.content)

        time.sleep(0.3)   # slight pause so the sidebar update feels intentional
        st.rerun()

else:
    # Session complete — show a clean CTA
    st.markdown("")
    st.info("✅ **Interview session complete.** See your score progression in the sidebar, or start a new session.", icon="🎯")
    if st.button("▶ Start New Session", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
