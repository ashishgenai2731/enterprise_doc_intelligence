Markdown
# Autonomous Financial Analyst & Auditor Platform

An enterprise-grade, stateful multi-agent document intelligence platform built with **LangGraph**, **FastAPI**, **Pinecone**, and **Ollama**. The system executes hybrid RAG retrieval (Dense + Sparse), sandboxed Python quantitative execution, and adversarial self-correcting compliance auditing over complex SEC 10-K financial filings.

**Corpus:** 89 annual reports (10-K) from 30 US companies, fiscal years 2023–2026, pulled from SEC EDGAR, plus Apple's FY2015 10-K PDF: 5.2 million words, 98,137 indexed chunks. Every step's timing and result is in [`docs/PROJECT_METRICS.md`](docs/PROJECT_METRICS.md).

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
* **Table-Aware PDF Parsing:** Financial statements are rebuilt from word positions so every row carries its year columns (`Net income — 2015: 53,394; 2014: 39,510; 2013: 37,037`). This survives chunking, where a header-less table row would otherwise be meaningless.
* **EDGAR HTML Parsing:** Inline-XBRL 10-K tables split one figure over several cells (`$ | 391,035 | )`) and place columns with colspans. Cells are laid on a column grid and each value is matched to the header it overlaps, giving the same `row — year: value` format as the PDF parser.
* **Company / Year Filtered Search:** Each chunk starts with `[Apple Inc. (AAPL) 10-K, fiscal 2024]` and carries `ticker` / `fiscal_year` metadata. Company names and years in the question become a Pinecone metadata filter, and BM25 runs over the same filtered subset, so a question about Apple's 2024 net income is not answered from Microsoft's filing. If the filter finds nothing, search falls back to the whole index.
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
PINECONE_INDEX_NAME=doc-intelligence-edgar
CORPUS_PATH=data/edgar/corpus.jsonl
SEC_USER_AGENT="Your Name your@email.com"   # SEC EDGAR requires a contact in the User-Agent
REDIS_HOST=localhost
REDIS_PORT=6379
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key
LANGCHAIN_PROJECT=enterprise-doc-intelligence
```

### 2. Download and ingest the filings

```bash
pip install -r requirements.txt
ollama pull mistral
python -m scripts.edgar_download      # 3 latest 10-Ks + XBRL facts for each company in config/companies.py
python -m scripts.ingest_edgar        # parse, chunk, embed -> Pinecone + data/edgar/corpus.jsonl (resumable)
python -m scripts.ingest_edgar --pdf ~/Downloads/Form10K.pdf --ticker AAPL --fiscal-year 2015   # add a local PDF
python -m src.eval.edgar_golden       # build the XBRL golden set -> data/eval/edgar_golden_qa.jsonl
```

`edgar_download` and `ingest_edgar` skip work already done, so an interrupted run can simply be started again.

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

## Evaluation

### EDGAR golden set (1,074 questions, generated from XBRL)

Every 10-K is filed with machine-readable XBRL facts, so expected answers are read from the companies' own data instead of being written by hand: 592 lookups (revenue, net income, R&D, EPS, ...) and 482 computed metrics (growth, margins, R&D intensity, current ratio). A question is kept only if every input figure also appears in the parsed filing text.

```bash
python -m src.eval.golden_eval --golden data/eval/edgar_golden_qa.jsonl --sample 60 --out edgar_results
python -m src.eval.golden_eval --golden data/eval/edgar_golden_qa.jsonl --out edgar_results --rescore   # after fixing a golden answer
```

**60-question sample (seed 7, 23 companies) on the 98,137-chunk index, Mistral-7B:**

| Metric | Value |
|---|---|
| Accuracy (all 60) | **50%** |
| Accuracy, direct lookups | 53% |
| Accuracy, computed metrics | 47% |
| Correct company / year filter detected | 60 / 59 of 60 |
| Wrong answers caught by audit | 8 of 30 |
| Answers with valid citations | 85% |
| Avg. latency per question | 100s |

25 of the 30 misses are retrieval misses: the right statement row was not in any cited chunk. The other 5 had the right figure and picked the wrong one. On the original 25 Apple FY2015 questions, the big index scored **68%** against 52% on the single-PDF index, even though 17 of those questions name no company and search all 98,137 chunks. Full step-by-step timings and results: [`docs/PROJECT_METRICS.md`](docs/PROJECT_METRICS.md).

### Apple FY2015 golden set (25 questions, hand-written)

A 25-question golden set (`data/eval/golden_qa.jsonl`) built from the Apple FY2015 10-K: 13 direct lookups and 12 computed metrics (growth rates, margins, ratios, free cash flow). Every expected answer cites the parsed statement row it comes from. Scoring checks the number in the final report (±0.1 pts for percentages, ±0.5% for amounts) and whether the auditor caught the wrong ones.

```bash
python -m src.eval.golden_eval            # full run, writes data/eval/results.{json,md}
python -m src.eval.golden_eval --ids C02  # single question
```

**Baseline on the single-PDF index (Mistral-7B via Ollama, hybrid retrieval on Pinecone):**

| Metric | Value |
|---|---|
| Accuracy (all 25) | **52%** |
| Accuracy, direct lookups | 69% |
| Accuracy, computed metrics | 33% |
| Audit pass rate | 88% |
| False-pass rate (wrong but passed, of passed) | 45% |
| Wrong answers caught by audit | 2 of 12 |
| Answers with valid citations | 96% |
| Avg. graph iterations | 1.52 |
| Avg. latency per question | 148s |

**What the failures show** (per-question reports in `data/eval/results.json`):
* **Wrong inputs, right arithmetic.** Most computed-metric misses use a wrong row as input (e.g. a segment or quarterly figure instead of the consolidated one). The sandbox computes them correctly, so the grounding check passes them.
* **The auditor rarely catches wrong-but-plausible numbers** (2 of 12). It checks that figures exist in the sources, not that they are the right line item for the question.
* **Unit slips** such as "$23.09 million" for a $23,089 million change.

These are the targets for the next iteration: check that every input literal in the generated code appears in a retrieved row with the right year, and compare a stronger local model against Mistral-7B on this same set.

---

## Known Limitations

* **Grounded is not the same as correct.** The auditor checks that every figure traces back to a source, not that the model picked the right line item. Measuring correctness needs a question set with known answers.
* **Table parsing is tuned for SEC-style statements.** Rows are matched to year headers (`2015 2014 2013`) by position; tables whose columns are not years (e.g. geographic segments) are kept as plain text lines.
