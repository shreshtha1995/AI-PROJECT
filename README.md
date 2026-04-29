# AI-PROJECT

# 🧠 Autonomous AI Interview Agent

An agentic technical assessment platform that conducts dynamic, multi-turn technical interviews. This system leverages **LangGraph** for orchestration, a **ChromaDB** vector database for semantic RAG retrieval, and the **Gemini API** as an LLM-as-a-Judge to evaluate candidate responses in real-time.

## 🚀 The Problem It Solves
Standard AI interview chatbots are passive and repetitive. This project introduces an **algorithmic feedback loop**. Instead of asking a static list of questions, the system autonomously tracks candidate performance, identifies weak areas, and dynamically searches a vector database for the exact right question to ask next, adapting the difficulty (Levels 1-5) on the fly.

## 🏗️ System Architecture

* **The Orchestrator (LangGraph):** A custom Python state machine that manages the conversational flow, tracks the user's proficiency score, and handles decision-making nodes (Retrieve -> Ask -> Grade -> Adjust).
* **Semantic RAG Engine (ChromaDB):** Retrieves interview questions using cosine similarity (via `all-MiniLM-L6-v2` embeddings). The retrieval is Active—the agent queries the database using natural language context based on the user's real-time performance.
* **LLM-as-a-Judge (Gemini):** Evaluates free-text candidate answers. Utilizing strict Prompt Engineering and Structured Outputs (JSON), the LLM acts deterministically to assign mathematical scores rather than conversational feedback.
* **The Frontend (Streamlit):** A clean, interactive UI providing real-time telemetry (current difficulty level, identified weak areas, and score progression).

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **AI/ML:** LangChain, LangGraph, Google GenAI (Gemini 1.5)
* **Vector Database:** ChromaDB, Sentence-Transformers
* **Frontend:** Streamlit, Pandas
