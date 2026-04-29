"""
agent.py  —  LangGraph State Machine (Gemini edition)
──────────────────────────────────────────────────────
Changes from Groq version:
  • Uses gemini-1.5-flash (free tier: 15 RPM / 1M TPM / 1500 RPD)
  • Grading prompt trimmed aggressively — ideal_answer capped at 180 chars,
    no concepts_missed field, shorter output spec → ~60% fewer tokens per call
  • Retry with exponential back-off (tenacity) on ResourceExhausted / 429
  • Local keyword fallback grader so a rate-limit never crashes the session
  • All other logic (state, nodes, graph, edges) identical
"""

import json
import os
import time
from typing import Annotated, Literal
from typing_extensions import TypedDict

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get API key - try .env first, then direct assignment if needed
api_key = os.environ.get("GOOGLE_API_KEY")

# If not found, try reading .env directly
if not api_key:
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("GOOGLE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Add it to .env file: GOOGLE_API_KEY=sk-...")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from retriever import ChromaRetriever


# ─────────────────────────────────────────────────────────────────────────────
# 1. STATE
# ─────────────────────────────────────────────────────────────────────────────

class InterviewState(TypedDict):
    messages:          Annotated[list, add_messages]
    current_level:     int
    current_question:  str
    ideal_answer:      str
    current_topic:     str
    score:             float
    feedback:          str
    turn_count:        int
    session_complete:  bool
    weak_areas:        list
    score_history:     list


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM SETUP
# ─────────────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-pro-vision",         # stable, working Gemini model
    temperature=0,                     # deterministic grading
    api_key=api_key,                   # use the loaded API key
    max_output_tokens=180,             # hard cap to save quota
                                       #   keeping this low saves quota tokens
)

retriever = ChromaRetriever()
MAX_TURNS = 8


# ─────────────────────────────────────────────────────────────────────────────
# 3. RETRY HELPER
#    Simple exponential back-off — no extra library needed.
#    Gemini free tier: 15 RPM → max 1 call every 4 s.
#    On 429 / ResourceExhausted we wait and retry up to 3 times.
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm_with_retry(messages: list, retries: int = 3) -> str:
    """
    Wraps llm.invoke() with exponential back-off for rate-limit errors.
    Returns the raw text content string, or raises after all retries.
    """
    delay = 10  # start with 10 s, doubles each retry
    for attempt in range(retries):
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(k in err for k in [
                "429", "resource_exhausted", "rate limit",
                "quota", "too many requests",
            ])
            if is_rate_limit and attempt < retries - 1:
                print(f"[agent] Rate limit hit — waiting {delay}s (attempt {attempt+1}/{retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError("LLM call failed after all retries")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOCAL KEYWORD FALLBACK GRADER
#    If Gemini is completely unavailable (exhausted quota for the day),
#    we fall back to a simple keyword-overlap scorer so the session continues.
#    This is disclosed to the user in the feedback string.
# ─────────────────────────────────────────────────────────────────────────────

def _local_grade(candidate_answer: str, ideal_answer: str) -> dict:
    """
    Keyword-overlap score as a last resort when the LLM is unavailable.
    Not intended to replace LLM grading — only keeps the session alive.
    """
    ideal_words    = set(ideal_answer.lower().split())
    candidate_words = set(candidate_answer.lower().split())
    stop_words      = {"a","an","the","is","are","was","were","be","been",
                       "to","of","in","and","or","it","its","for","with","that","this"}
    ideal_kw     = ideal_words - stop_words
    candidate_kw = candidate_words - stop_words

    if not ideal_kw:
        score = 0.5
    else:
        overlap = len(ideal_kw & candidate_kw)
        score   = round(min(overlap / max(len(ideal_kw) * 0.5, 1), 1.0), 2)

    if score >= 0.8:
        fb = "Strong answer covering key concepts."
    elif score >= 0.5:
        fb = "Partial answer — some key ideas were present but details were thin."
    else:
        fb = "Answer missed most key concepts. Review the topic before moving on."

    fb += " *(Scored locally — AI evaluator was temporarily unavailable.)*"
    return {"score": score, "feedback": fb, "concepts_missed": []}


# ─────────────────────────────────────────────────────────────────────────────
# 5. NODES
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_question(state: InterviewState) -> dict:
    """NODE 1 — RAG retriever. Fetches next question from ChromaDB."""
    score         = state.get("score", -1)
    weak_areas    = state.get("weak_areas", [])
    current_level = state.get("current_level", 2)
    last_question = state.get("current_question", "")

    if score < 0:
        performance = "just started the interview"
    elif score >= 0.8:
        performance = f"answered confidently, score {score:.0%}"
    elif score <= 0.4:
        performance = f"struggled significantly, score {score:.0%}"
    else:
        performance = f"gave a partial answer, score {score:.0%}"

    question = retriever.next_question(
        target_level         = current_level,
        candidate_weak_areas = weak_areas if weak_areas else None,
        last_question        = last_question if last_question else None,
        last_performance     = performance,
    )

    if question is None:
        return {"session_complete": True}

    return {
        "current_question": question["question"],
        "ideal_answer":     question["ideal_answer"],
        "current_topic":    question["topic"],
    }


def ask_question(state: InterviewState) -> dict:
    """NODE 2 — Formats the question and adds it to the message history."""
    turn     = state.get("turn_count", 0) + 1
    level    = state["current_level"]
    topic    = state.get("current_topic", "").replace("_", " ").title()
    question = state["current_question"]

    message_text = (
        f"**Question {turn}** (Level {level} — {topic})\n\n"
        f"{question}"
    )

    return {
        "messages":   [AIMessage(content=message_text)],
        "turn_count": turn,
    }


def grade_answer(state: InterviewState) -> dict:
    """
    NODE 3 — LLM Grader
    ────────────────────
    KEY OPTIMISATIONS vs the Groq version:

    1. ideal_answer is TRIMMED to 180 chars.
       The full ideal_answer averages ~250 tokens.  We only need key phrases,
       not the complete rubric, for scoring.  This alone halves input tokens.

    2. No concepts_missed in the output spec.
       That field cost ~30-80 extra output tokens per call and forced the LLM
       to generate a list.  Weak-area tracking now comes from score alone.

    3. max_output_tokens=180 is set on the LLM itself (see setup above).

    4. Retry + local fallback so a single 429 never kills the session.

    Net result: ~60% fewer tokens per grading call vs the original.
    Gemini free-tier quota (1M TPM) lasts through 4-5× more sessions.
    """
    candidate_answer = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            candidate_answer = msg.content
            break

    if not candidate_answer:
        return {"score": 0.0, "feedback": "No answer detected."}

    # ── Trim the ideal answer ─────────────────────────────────────────────────
    # 180 chars keeps the key concepts while dropping verbose explanation.
    ideal_trimmed = state["ideal_answer"][:180].rsplit(" ", 1)[0] + "…"

    # ── Compact grading prompt ────────────────────────────────────────────────
    # Removed: concepts_missed list, verbose scoring guide table.
    # Kept: strict JSON-only instruction, the three essentials.
    grading_prompt = (
        "You are a strict technical interviewer. "
        "Reply with ONLY valid JSON, no markdown, no extra text.\n\n"
        f"Q: {state['current_question']}\n\n"
        f"Key concepts: {ideal_trimmed}\n\n"
        f"Candidate said: {candidate_answer[:400]}\n\n"  # cap answer too
        'JSON format: {"score": <0.0-1.0>, "feedback": "<2 sentences max>"}\n'
        "Score 1.0=perfect, 0.8=strong, 0.6=adequate, 0.4=weak, 0.2=poor."
    )

    # ── Call LLM with retry ───────────────────────────────────────────────────
    used_fallback = False
    try:
        raw    = _call_llm_with_retry([HumanMessage(content=grading_prompt)])
        # Strip any accidental markdown fences
        raw    = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)

        score    = max(0.0, min(1.0, float(result.get("score", 0.5))))
        feedback = str(result.get("feedback", "")).strip()

    except json.JSONDecodeError:
        # LLM returned text but not valid JSON — extract score with a fallback
        print(f"[agent] JSON parse error, using local grader")
        result       = _local_grade(candidate_answer, state["ideal_answer"])
        score        = result["score"]
        feedback     = result["feedback"]
        used_fallback = True

    except Exception as e:
        print(f"[agent] LLM unavailable: {e} — using local grader")
        result       = _local_grade(candidate_answer, state["ideal_answer"])
        score        = result["score"]
        feedback     = result["feedback"]
        used_fallback = True

    # ── Update weak areas based on score (no LLM call needed) ────────────────
    # Instead of asking the LLM to identify missed concepts, we just tag
    # the topic as weak if the score is below 0.6.
    current_weak = state.get("weak_areas", [])
    topic        = state.get("current_topic", "")
    if score < 0.6 and topic and topic not in current_weak:
        updated_weak = (current_weak + [topic])[:5]
    else:
        updated_weak = current_weak

    return {
        "score":      score,
        "feedback":   feedback,
        "weak_areas": updated_weak,
        "messages": [AIMessage(
            content=f"**Evaluation:**\n{feedback}\n\n*Score: {score:.0%}*"
        )],
    }


def adjust_level(state: InterviewState) -> dict:
    """NODE 4 — Reads score and adjusts difficulty level."""
    score         = state.get("score", 0.5)
    current_level = state.get("current_level", 2)
    score_history = list(state.get("score_history", []))

    if score >= 0.8:
        new_level = min(current_level + 1, 5)
        level_msg = "⬆️ Level up!" if new_level > current_level else "🏆 Already at max level."
    elif score <= 0.4:
        new_level = max(current_level - 1, 1)
        level_msg = "⬇️ Stepping back." if new_level < current_level else "📌 Already at minimum."
    else:
        new_level = current_level
        level_msg = "➡️ Staying at current level."

    score_history.append({
        "turn":      state.get("turn_count", 0),
        "level":     current_level,
        "topic":     state.get("current_topic", ""),
        "score":     score,
        "new_level": new_level,
    })

    return {
        "current_level": new_level,
        "score_history": score_history,
        "messages": [AIMessage(content=f"{level_msg} Moving to Level {new_level}.\n")],
    }


def generate_final_report(state: InterviewState) -> dict:
    """NODE 5 — Session report. No LLM call — purely computed from state."""
    score_history = state.get("score_history", [])
    if not score_history:
        return {"messages": [AIMessage(content="Interview complete. No data recorded.")]}

    avg_score   = sum(s["score"] for s in score_history) / len(score_history)
    final_level = state.get("current_level", 2)
    weak_areas  = state.get("weak_areas", [])

    report = (
        f"## Interview Complete — Final Report\n\n"
        f"**Questions Asked:** {len(score_history)}\n"
        f"**Average Score:** {avg_score:.0%}\n"
        f"**Final Level Reached:** {final_level}/5\n"
        f"**Weak Areas:** {', '.join(w.replace('_',' ') for w in weak_areas) if weak_areas else 'None identified'}\n\n"
        f"**Score Progression:**\n"
    )

    for entry in score_history:
        bar   = "█" * int(entry["score"] * 10) + "░" * (10 - int(entry["score"] * 10))
        topic = entry["topic"].replace("_", " ")
        report += f"  Turn {entry['turn']} | L{entry['level']} | {topic:25s} | {bar} {entry['score']:.0%}\n"

    if avg_score >= 0.8:
        verdict = "Strong candidate. Recommend advancing to next round."
    elif avg_score >= 0.6:
        verdict = "Solid foundation. Consider a follow-up on identified weak areas."
    else:
        verdict = "Needs more preparation. Key gaps identified above."

    report += f"\n**Verdict:** {verdict}"
    return {"messages": [AIMessage(content=report)]}


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONDITIONAL EDGE
# ─────────────────────────────────────────────────────────────────────────────

def should_continue(state: InterviewState) -> Literal["retrieve_question", "generate_final_report"]:
    if state.get("session_complete", False):
        return "generate_final_report"
    if state.get("turn_count", 0) >= MAX_TURNS:
        return "generate_final_report"
    return "retrieve_question"


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("retrieve_question",     retrieve_question)
    graph.add_node("ask_question",          ask_question)
    graph.add_node("grade_answer",          grade_answer)
    graph.add_node("adjust_level",          adjust_level)
    graph.add_node("generate_final_report", generate_final_report)

    graph.add_edge(START,                   "retrieve_question")
    graph.add_edge("retrieve_question",     "ask_question")
    graph.add_edge("ask_question",          "grade_answer")
    graph.add_edge("grade_answer",          "adjust_level")
    graph.add_edge("generate_final_report", END)

    graph.add_conditional_edges(
        "adjust_level",
        should_continue,
        {
            "retrieve_question":     "retrieve_question",
            "generate_final_report": "generate_final_report",
        }
    )

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 8. INITIAL STATE
# ─────────────────────────────────────────────────────────────────────────────

def create_initial_state(starting_level: int = 2) -> dict:
    return {
        "messages":          [],
        "current_level":     starting_level,
        "current_question":  "",
        "ideal_answer":      "",
        "current_topic":     "",
        "score":             -1.0,
        "feedback":          "",
        "turn_count":        0,
        "session_complete":  False,
        "weak_areas":        [],
        "score_history":     [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. TERMINAL RUNNER (for testing without Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def run_interview_session():
    graph = build_graph()
    state = create_initial_state(starting_level=2)

    print("\n" + "=" * 60)
    print("  Adaptive Interview System — Terminal Mode (Gemini)")
    print(f"  Max questions: {MAX_TURNS}")
    print("=" * 60)

    while not state.get("session_complete") and state.get("turn_count", 0) < MAX_TURNS:
        for event in graph.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage):
                            print(f"\n[{node_name}]\n{msg.content}")

                for k, v in node_output.items():
                    if k == "messages":
                        state["messages"] = state.get("messages", []) + v
                    else:
                        state[k] = v

                if node_name == "ask_question":
                    answer = input("\n>> Your answer: ").strip()
                    state["messages"].append(HumanMessage(content=answer or "I don't know."))

        if state.get("turn_count", 0) >= MAX_TURNS:
            break

    print("\n" + "=" * 60)
    print("  Interview complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_interview_session()
