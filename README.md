Markdown
# Autonomous Financial Analyst & Auditor Platform

An enterprise-grade, stateful multi-agent document intelligence platform built with **LangGraph**, **FastAPI**, **Pinecone**, and **Ollama**. The system executes hybrid RAG retrieval (Dense + Sparse), sandboxed Python quantitative execution, and adversarial self-correcting compliance auditing over complex SEC 10-K financial filings.

---

## Architecture Overview

                      ┌────────────────────────┐
                      │   Client / HTTP Request │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │   FastAPI Gateway      │
                      │   (REST & SSE Stream)  │
                      └───────────┬────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ LangGraph State Machine Loop │
                   └──────────────┬───────────────┘
                                  │
     ┌────────────────────────────┼────────────────────────────┐
     │                            │                            │
     ▼                            ▼                            ▼
┌─────────────────┐          ┌──────────────────┐        ┌──────────────────┐
│  RAG Specialist │          │  Code Executor   │        │ Compliance Auditor│
│  (Node 1)       │          │  (Node 2)        │        │  (Node 3)        │
└────────┬────────┘          └────────┬─────────┘        └────────┬─────────┘
         │                            │                           │
         ▼                            ▼                           ▼
┌──────────────────┐         ┌──────────────────┐        ┌──────────────────┐
│ Hybrid Retriever │         │ Safe Python REPL │        │ Self-Correction  │
│ (Pinecone + BGE) │         │ Scope Execution  │        │ Verification Loop│
└──────────────────┘         └──────────────────┘        └────────┬─────────┘
                                                                  │
                                                        [AUDIT_STATUS: PASSED?]
                                                            /
                                                      (Yes)/                    (No: Refine Search)
                                                          v                      v
                                                    ┌─────────────┐       ┌─────────────┐
                                                    │  END STATE  │       │ RAG Specialist│
                                                    └─────────────┘       └─────────────┘


---

## Core Features

* **Stateful Orchestration:** Utilizes LangGraph `StateGraph` primitives to enforce guarded, deterministic multi-agent loops over open-ended prompt chains.
* **Hybrid RAG & Cross-Encoder Reranking:** Combines Pinecone dense vector retrieval with BGE-reranker cross-encoders for precision chunk retrieval.
* **Sandboxed Code Execution:** Executes generated Python arithmetic scripts within isolated scopes, eliminating hallucinated financial calculations.
* **Adversarial Self-Healing Loop:** Evaluates report groundedness using an LLM Auditor node. If hallucinated facts or unverified estimations occur, the state graph automatically rewrites search queries and re-executes up to a maximum safety threshold.
* **Real-Time Event Streaming:** Implements Server-Sent Events (SSE) via `astream_events` to stream live graph state transitions directly to clients.
* **End-to-End Observability:** Integrated with LangSmith tracing for full execution visibility across node latencies, state payloads, and tool calls.

---

## Tech Stack

* **Frameworks & Agents:** FastAPI, LangGraph, LangChain, Pydantic V2, Instructor
* **LLM Engine:** Ollama (Mistral 7B local execution)
* **Vector Database:** Pinecone (Dense) + BGE Reranker
* **Caching & Observability:** Redis, LangSmith
* **Testing & Infrastructure:** Pytest, HTTPX, Docker (Multi-stage build)

---

## Quickstart Guide

### 1. Environment Setup

Clone the repository and configure your environment variables in `.env`:

```bash
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=financial-10k-index
REDIS_HOST=localhost
REDIS_PORT=6379
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key
LANGCHAIN_PROJECT=enterprise-doc-intelligence
2. Local Execution
Install dependencies and run the agent via CLI:

Bash
pip install -r requirements.txt
python scripts/run_agent.py
3. API & Real-Time Event Streaming
Start the production FastAPI server:

Bash
uvicorn main:app --reload
Test real-time SSE streaming updates:

Bash
curl -N -X POST "[http://127.0.0.1:8000/api/v1/agent/analyze/stream](http://127.0.0.1:8000/api/v1/agent/analyze/stream)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What were total operating expenses for 2015 and what is the calculated percentage change compared to 2014?"}'
Testing & Dockerization
Run the comprehensive unit, integration, and API test suite:

Bash
python -m pytest -v
Build and execute the production container:

Bash
docker build -t enterprise-doc-intelligence:latest .
docker run -d -p 8000:8000 --env-file .env enterprise-doc-intelligence:latest

---
