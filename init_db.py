"""
init_db.py
──────────
Loads questions.json into a persistent ChromaDB collection.
Run this ONCE to seed your database before starting the interview agent.

Usage:
    python init_db.py
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

# ── Config ────────────────────────────────────────────────────────────────────
QUESTIONS_FILE = "questions.json"
CHROMA_DB_PATH = "./chroma_store"       # Persistent local folder
COLLECTION_NAME = "interview_questions"

# ── Embedding function ────────────────────────────────────────────────────────
# We use the default all-MiniLM-L6-v2 sentence-transformer (runs locally, free).
# It converts question text → 384-dim vector so ChromaDB can do semantic search.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def init_database() -> chromadb.Collection:
    """
    Creates (or resets) the ChromaDB collection and loads all questions into it.
    Each question is stored with:
      - document : the question text (what gets embedded into a vector)
      - metadata : level, topic, ideal_answer (filterable fields)
      - id       : unique string ID
    """
    # PersistentClient saves the DB to disk so it survives restarts
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # ── Idempotent reset ──────────────────────────────────────────────────────
    # Delete existing collection if re-running the script, so we don't duplicate.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"[init_db] Deleted existing collection '{COLLECTION_NAME}'")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        # cosine distance is best for sentence embeddings
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[init_db] Created collection '{COLLECTION_NAME}'")

    # ── Load questions ────────────────────────────────────────────────────────
    with open(QUESTIONS_FILE, "r") as f:
        questions = json.load(f)

    # ChromaDB batch API — much faster than adding one-by-one
    collection.add(
        ids=[q["id"] for q in questions],

        # The 'documents' field is what gets EMBEDDED into a vector.
        # We embed the question text so semantic search works later.
        documents=[q["question"] for q in questions],

        # Metadata is stored as-is (NOT embedded). Used for exact filtering.
        # ⚠️  ChromaDB only supports str, int, float, bool in metadata values.
        metadatas=[
            {
                "level":        q["level"],          # int  — used for difficulty filter
                "topic":        q["topic"],           # str  — used for topic filter
                "ideal_answer": q["ideal_answer"],    # str  — retrieved for LLM eval
            }
            for q in questions
        ],
    )

    print(f"[init_db] ✅ Inserted {len(questions)} questions into ChromaDB")
    print(f"[init_db] DB saved to: {os.path.abspath(CHROMA_DB_PATH)}")
    return collection


def verify_database(collection: chromadb.Collection) -> None:
    """Quick sanity check — print one question per level."""
    print("\n── Verification (1 question per level) ──────────────────────────")
    for level in range(1, 6):
        result = collection.get(
            where={"level": level},
            limit=1,
            include=["documents", "metadatas"],
        )
        if result["ids"]:
            q_id   = result["ids"][0]
            q_text = result["documents"][0]
            topic  = result["metadatas"][0]["topic"]
            print(f"  Level {level} | {topic:30s} | {q_text[:60]}...")
        else:
            print(f"  Level {level} | ⚠️  No questions found!")
    print("─────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    collection = init_database()
    verify_database(collection)
