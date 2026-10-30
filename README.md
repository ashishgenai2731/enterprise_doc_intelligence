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

* **Stateful Orchestration:** Utilizes LangGraph `StateGraph` primitives to enforce guarded, deterministic multi-agent loops over open-ended prompt chains (max 3 rounds).
* **Hybrid RAG & Cross-Encoder Reranking:** Pinecone dense retrieval (BGE-large) and BM25 sparse retrieval are fused with Reciprocal Rank Fusion, then reranked by a BGE cross-encoder. The BM25 corpus lives in `data/corpus.jsonl` and uses the same chunk IDs as Pinecone.
* **Sandboxed Code Execution over MCP:** Generated Python arithmetic runs in a separate FastMCP server process with a restricted builtin scope and a hard timeout (`SANDBOX_TIMEOUT_SEC`); runaway code is killed with its process.
* **Structured, Double-Checked Auditing:** The auditor returns a typed verdict (`passed`, `unsupported_claims`, `retry_query`) instead of a magic string. A deterministic check then verifies every figure in the report appears in the retrieved sources or the sandbox output, so the LLM is not the only judge of its own numbers.
* **Targeted Self-Correction:** On a failed audit, the next round searches for the specific missing evidence (`retry_query`) and adds it to the earlier context instead of starting over.
* **Source Citations:** Every figure in the report cites its chunk (`[Source: <chunk id>]`); the API returns the verified `sources` list.
* **Real-Time Event Streaming:** Server-Sent Events (SSE) via `astream_events` stream live graph state transitions to clients.
* **End-to-End Observability:** Integrated with LangSmith tracing for node latencies, state payloads, and tool calls.

---

## Tech Stack

* **Frameworks & Agents:** FastAPI, LangGraph, LangChain, Pydantic V2, Instructor, MCP (FastMCP)
* **LLM Engine:** Ollama (Mistral 7B local execution)
* **Retrieval:** Pinecone (dense) + BM25 (sparse) + RRF + BGE cross-encoder reranker
* **Caching & Observability:** Redis, LangSmith
* **Testing & Infrastructure:** Pytest, HTTPX, Docker (multi-stage build)

---

## Quickstart Guide

### 1. Environment Setup

Configure your environment variables in `.env`:

```bash
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=financial-10k-index
REDIS_HOST=localhost
REDIS_PORT=6379
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key
LANGCHAIN_PROJECT=enterprise-doc-intelligence
```

### 2. Ingest a 10-K

```bash
pip install -r requirements.txt
ollama pull mistral
python -m scripts.ingest_doc          # parse, chunk, embed -> Pinecone + data/corpus.jsonl
python -m scripts.export_corpus       # only if the index was filled earlier: rebuild the BM25 corpus from Pinecone
```

### 3. API & Real-Time Event Streaming

```bash
uvicorn src.api.main:app --reload
```

```bash
curl -N -X POST "http://127.0.0.1:8000/api/v1/agent/analyze/stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "What was the percentage change in net income from 2014 to 2015?"}'
```

### 4. Tests & Docker

```bash
python -m pytest tests/unit tests/test_agent.py -v   # offline: no Pinecone/Ollama needed
python -m pytest -v                                   # full suite incl. live retrieval + API
```

```bash
docker build -t enterprise-doc-intelligence:latest .
docker run -d -p 8000:8000 --env-file .env enterprise-doc-intelligence:latest
```

---

## Known Limitations

* **Table parsing loses column headers.** Financial statements are split into one-row tables, so a chunk like `14,329 | 11,993 | 10,830` no longer says which column is which year. This is the main source of wrong answers and the next thing to fix.
* **Grounded is not the same as correct.** The auditor checks that every figure traces back to a source, not that the model picked the right line item. Measuring correctness needs a question set with known answers.
