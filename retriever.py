"""
retriever.py  —  TRUE RAG Implementation
─────────────────────────────────────────
Retrieval is driven by SEMANTIC SIMILARITY, not tag filtering.
The agent passes its current interview context as a natural language string.
ChromaDB embeds it → finds nearest question vectors → level filter narrows results.

Flow:
    agent_context (str)
          │
          ▼
    SentenceTransformer embeds it → 384-dim query vector
          │
          ▼
    ChromaDB cosine similarity search across ALL question vectors
          │
          ▼
    Post-filter: keep only results matching target_level
          │
          ▼
    Exclude already-asked questions
          │
          ▼
    Return best semantic match
"""

import random
import json
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "./chroma_store"
COLLECTION_NAME = "interview_questions"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Context Builder  —  The part that makes retrieval SEMANTIC
# ─────────────────────────────────────────────────────────────────────────────

def build_query_context(
    target_level: int,
    topic_hint: Optional[str]             = None,
    candidate_weak_areas: Optional[list]  = None,
    last_question: Optional[str]          = None,
    last_performance: Optional[str]       = None,
) -> str:
    """
    Converts the agent's current state into a rich natural language query string.

    WHY THIS MATTERS:
    ─────────────────
    ChromaDB doesn't search by level or topic. It searches by MEANING.
    The richer and more specific this query string, the more semantically
    relevant the retrieved question will be to the candidate's real situation.

    This is the "R" in RAG — Retrieval quality entirely depends on this query.

    Examples of what ChromaDB will semantically match against:
      "candidate weak at recursion, needs level 2 question about trees"
        → retrieves questions about recursive tree traversal

      "candidate strong at arrays, ready for level 4, test graph algorithms"
        → retrieves Dijkstra, cycle detection, topological sort questions
    """
    parts = []

    level_descriptors = {
        1: "beginner fundamental concept",
        2: "intermediate problem solving",
        3: "advanced algorithm implementation",
        4: "expert optimization and design",
        5: "principal engineer system design and architecture",
    }
    parts.append(level_descriptors.get(target_level, f"level {target_level}"))

    if topic_hint:
        parts.append(f"about {topic_hint.replace('_', ' ')}")

    if candidate_weak_areas:
        weak_str = ", ".join(w.replace("_", " ") for w in candidate_weak_areas)
        parts.append(f"targeting weak areas in {weak_str}")

    if last_question:
        parts.append(f"different topic from: {last_question[:80]}")

    if last_performance:
        parts.append(f"candidate recently {last_performance}")

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Core RAG Retrieval Function
# ─────────────────────────────────────────────────────────────────────────────

def get_question(
    agent_context: str,
    target_level: int,
    exclude_ids: Optional[list] = None,
    n_candidates: int = 10,
) -> Optional[dict]:
    """
    TRUE RAG retrieval: semantic similarity search → then filter by level.

    Args:
        agent_context : natural language string describing current interview state
        target_level  : int 1-5, used as a POST-RETRIEVAL filter (NOT primary mechanism)
        exclude_ids   : question IDs already asked this session
        n_candidates  : how many semantic matches to retrieve before filtering

    Retrieval Pipeline:
    ───────────────────
    Step 1 — EMBED   : agent_context → 384-dim vector (SentenceTransformer)
    Step 2 — SEARCH  : cosine similarity against ALL question vectors in ChromaDB
    Step 3 — FILTER  : keep results where metadata.level == target_level
    Step 4 — EXCLUDE : remove already-asked question IDs
    Step 5 — PICK    : weighted random from top-3 by similarity score
    """
    exclude_ids = exclude_ids or []
    collection  = _get_collection()

    # ── Steps 1 & 2: Semantic vector search ───────────────────────────────────
    # ChromaDB calls embedding_fn on agent_context internally,
    # producing a query vector and running cosine similarity against all stored vectors.
    # n_results is large intentionally — we need enough candidates to survive
    # the level filter + exclude list downstream.
    raw = collection.query(
        query_texts=[agent_context],
        n_results=min(n_candidates, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    ids       = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]    # cosine distance: lower = more similar

    # ── Steps 3 & 4: Filter by level + exclude asked questions ───────────────
    candidates = []
    for i in range(len(ids)):
        if metadatas[i]["level"] != target_level:
            continue
        if ids[i] in exclude_ids:
            continue

        candidates.append({
            "id":           ids[i],
            "question":     documents[i],
            "level":        metadatas[i]["level"],
            "topic":        metadatas[i]["topic"],
            "ideal_answer": metadatas[i]["ideal_answer"],
            "similarity":   round(1 - distances[i], 4),   # distance → similarity score
        })

    if not candidates:
        print(f"[retriever] Level-{target_level} exhausted. Trying adjacent levels...")
        return _adjacent_level_fallback(agent_context, target_level, exclude_ids, collection)

    # ── Step 5: Weighted random pick from top-3 ───────────────────────────────
    # Why not always pick top-1?
    # Picking strictly top-1 means the same context always returns the same question.
    # Weighted random from top-3 adds diversity while still preferring
    # the most semantically relevant result.
    top_k   = candidates[:3]
    weights = [c["similarity"] for c in top_k]
    total   = sum(weights)

    if total == 0:
        chosen = random.choice(top_k)
    else:
        r, cumulative = random.uniform(0, total), 0
        chosen = top_k[-1]
        for candidate, w in zip(top_k, weights):
            cumulative += w
            if r <= cumulative:
                chosen = candidate
                break

    print(f"[retriever] ✅ '{chosen['topic']}' "
          f"(level={chosen['level']}, similarity={chosen['similarity']})")
    return chosen


def _adjacent_level_fallback(
    agent_context: str,
    target_level: int,
    exclude_ids: list,
    collection: chromadb.Collection,
) -> Optional[dict]:
    """
    When target level is fully exhausted, search semantically in adjacent levels.
    Tries level±1, then level±2. Still uses semantic search — just relaxes the level filter.
    """
    for delta in [1, -1, 2, -2]:
        lvl = target_level + delta
        if not (1 <= lvl <= 5):
            continue

        raw = collection.query(
            query_texts=[agent_context],
            n_results=min(10, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        for i in range(len(raw["ids"][0])):
            qid = raw["ids"][0][i]
            if raw["metadatas"][0][i]["level"] == lvl and qid not in exclude_ids:
                print(f"[retriever] Fallback: level-{lvl} question "
                      f"(originally requested level-{target_level})")
                return {
                    "id":           qid,
                    "question":     raw["documents"][0][i],
                    "level":        lvl,
                    "topic":        raw["metadatas"][0][i]["topic"],
                    "ideal_answer": raw["metadatas"][0][i]["ideal_answer"],
                    "similarity":   round(1 - raw["distances"][0][i], 4),
                    "fallback":     True,
                }

    print("[retriever] ❌ All questions exhausted.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stateful Agent-Facing Class
# ─────────────────────────────────────────────────────────────────────────────

class ChromaRetriever:
    """
    Stateful retriever the agent uses throughout an interview session.

    The agent's decision engine calls next_question() with its current state.
    This class handles:
      - building the semantic query from agent state
      - tracking asked questions (no repeats)
      - logging retrieval history for analytics/debugging
    """

    def __init__(self):
        self.asked_ids: list = []
        self.history:   list = []

    def next_question(
        self,
        target_level: int,
        topic_hint: Optional[str]            = None,
        candidate_weak_areas: Optional[list] = None,
        last_question: Optional[str]         = None,
        last_performance: Optional[str]      = None,
    ) -> Optional[dict]:
        """
        Called by the agent after every evaluation cycle.

        The agent passes what IT knows about the current state.
        This method converts that into a semantic query and retrieves.

        Example agent call:
            q = retriever.next_question(
                target_level         = 3,
                topic_hint           = "graphs",
                candidate_weak_areas = ["recursion", "DFS"],
                last_question        = "Explain BFS vs DFS",
                last_performance     = "gave partial answer, missed cycle detection",
            )
        """
        query = build_query_context(
            target_level         = target_level,
            topic_hint           = topic_hint,
            candidate_weak_areas = candidate_weak_areas,
            last_question        = last_question,
            last_performance     = last_performance,
        )

        print(f"\n[retriever] Semantic query → \"{query}\"")

        question = get_question(
            agent_context = query,
            target_level  = target_level,
            exclude_ids   = self.asked_ids,
        )

        if question:
            self.asked_ids.append(question["id"])
            self.history.append({
                "query":       query,
                "question_id": question["id"],
                "topic":       question["topic"],
                "level":       question["level"],
                "similarity":  question.get("similarity"),
                "fallback":    question.get("fallback", False),
            })

        return question

    def reset(self):
        self.asked_ids = []
        self.history   = []

    def session_summary(self) -> dict:
        return {
            "total_asked": len(self.asked_ids),
            "history":     self.history,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test — simulates a real adaptive agent session
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = ChromaRetriever()

    session_states = [
        {
            "target_level": 1,
            "topic_hint": None,
            "candidate_weak_areas": None,
            "last_question": None,
            "last_performance": "just started interview",
        },
        {
            "target_level": 2,
            "topic_hint": "arrays",
            "candidate_weak_areas": ["time complexity"],
            "last_question": "What is a hash map?",
            "last_performance": "answered correctly but skipped space complexity",
        },
        {
            "target_level": 3,
            "topic_hint": "graphs",
            "candidate_weak_areas": ["recursion", "DFS"],
            "last_question": "Explain two pointer technique",
            "last_performance": "strong answer with clear O(n) explanation",
        },
        {
            "target_level": 4,
            "topic_hint": "dynamic_programming",
            "candidate_weak_areas": ["memoization"],
            "last_question": "What is BFS vs DFS?",
            "last_performance": "missed cycle detection, partial credit",
        },
        {
            "target_level": 5,
            "topic_hint": "system_design",
            "candidate_weak_areas": [],
            "last_question": "Implement LRU Cache",
            "last_performance": "perfect answer with O(1) proof",
        },
    ]

    print("=" * 65)
    print("  TRUE RAG — Adaptive Interview Simulation")
    print("=" * 65)

    for i, state in enumerate(session_states, 1):
        q = retriever.next_question(**state)
        if q:
            print(f"\n  Round {i} | Level {q['level']} | {q['topic']}")
            print(f"  Q         : {q['question'][:90]}...")
            print(f"  Similarity: {q.get('similarity')}")
        print()

    print("── Session Summary ──")
    print(json.dumps(retriever.session_summary(), indent=2))
