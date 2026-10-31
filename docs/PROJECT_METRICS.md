# Project Metrics: SEC EDGAR Multi-Filing Pivot

Every number below was measured on one run and has a log or result file behind it
(`data/edgar/logs/`, `data/eval/*.json`). Nothing here is an estimate.

**Machine:** Apple M4 MacBook, 24 GB RAM. Embedding and reranking run on the M4 GPU (MPS);
the LLM is Mistral-7B served locally by Ollama. Vectors are stored in Pinecone (serverless, cosine).

**Models and settings:** `BAAI/bge-large-en-v1.5` embeddings (1024 dimensions),
`BAAI/bge-reranker-large` cross-encoder, chunks of 512 tokens with 50 overlap,
top 5 chunks after reranking, at most 3 agent rounds per question.

---

## 1. Scale: before and after

| | Before (one PDF) | After (EDGAR pivot) |
|---|---|---|
| Documents | 1 (Apple FY2015 10-K PDF) | 89 10-K filings + the same PDF |
| Companies | 1 | 30 |
| Fiscal years | 2015 | 2023–2026 (plus 2015) |
| Raw data | 80-page PDF | 268 MB filing HTML + 133 MB XBRL facts |
| Words | ~47,000 | 5,155,219 (HTML) + the PDF |
| Indexed chunks (vectors) | 715 | **98,137** |
| Golden questions | 25, written by hand | **1,074** generated from XBRL + the 25 originals |

About 110 times more text, and the test set grows with the corpus because nobody writes the answers by hand.

---

## 2. Pipeline steps and time taken

| # | Step | What it produced | Time | Log |
|---|---|---|---|---|
| 1 | Download from SEC EDGAR | 89 filings + XBRL facts for 30 companies, 401 MB | **55 s** | `logs/download.log` |
| 2 | Parse HTML to text | 281 MB HTML → 5,155,219 words | **49 s** (≈0.5 s per filing) | `logs/parse.log` |
| 3 | Chunk + embed + upsert to Pinecone | 97,422 chunks from 89 filings | **3 h 9 min** (11,356 s) | `logs/ingest.log` |
| 3b | Add the Apple FY2015 PDF | 715 chunks | 88 s | `logs/ingest.log` |
| 4 | Build golden Q&A from XBRL | 1,074 questions (592 lookup, 482 computed), 261 skipped | **61 s** first run, 291 s on a later re-run | `logs/golden_build.log` |
| 5 | Unit tests | 36 passed, 0 failed | **5.8 s** | `logs/unit_tests.log` |
| 6 | Agent eval, 60 sampled questions | scores in section 4 | **1 h 40 min** (5,996 s) | `logs/eval_60.log` |
| 7 | Agent eval, old 25 Apple questions on the new index | scores in section 5 | **42 min** (2,509 s) | `logs/eval_apple2015_on_big_index.log` |

**Indexing details (step 3):**
- Average 128 s per filing. The fastest was IBM at 27 s (its 10-K is short because the financial
  statements sit in a separate annual report). The slowest was Pfizer FY2025 at 262 s (1,883 chunks).
- Throughput: about 8.6 chunks per second, embedding on the M4 GPU.
- The run can resume. A chunk ID is `TICKER-FY{year}-chunk-{n}`, and a filing already in the local
  corpus is skipped, so a stopped run continues where it left off.

**Notes:**
- Step 1 was first run while the code was being built, so it was re-run into an empty folder to time it.
  The re-run produced the same 89-filing manifest.
- Step 6 was started once at 14:33. A Mac logout stopped it after 5 questions
  (`logs/eval_60_interrupted_by_logout.log`), so it was re-run from the start at 15:43.

---

## 3. The golden set (1,074 questions)

- **Where the answers come from:** each 10-K is filed with machine-readable XBRL facts
  (for example `NetIncomeLoss = 93,736,000,000` for the year ended 2024-09-28). The expected answer
  is read from those facts, not typed by hand.
- **Check against the text:** a question is kept only if every input figure (for example "391,035")
  appears in the parsed filing text. Otherwise the question would test the parser, not the agent.
  261 candidates were dropped this way.
- **Question types:**
  - 8 lookups: revenue, net income, operating income, R&D, operating cash flow, total assets, equity, diluted EPS
  - 7 computed: revenue growth, net income growth, gross margin, operating margin, net margin, R&D % of revenue, current ratio
- **Coverage:** 29 of 30 companies. IBM has none, for the same reason as above.
  By fiscal year: 2023: 237, 2024: 357, 2025: 370, 2026: 110.
- **Bug found and fixed in the golden set:** J&J reports a $483M item under the XBRL tag
  `ResearchAndDevelopmentExpense`, and its real total R&D ($15,085M) under
  `ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost`. The builder took the first tag.
  It now takes the largest one, as it already did for revenue.
  - 6 J&J answers changed. No questions were added or removed, so the 60-question sample stayed the same.
  - The saved answers were re-scored (`--rescore`) without re-running the agent. Accuracy went from 47% to 50%.

---

## 4. Agent results: 60 questions on the 98,137-chunk index

Sample: 30 lookups and 30 computed questions, fixed seed 7, covering 23 companies.

| Metric | Value |
|---|---|
| Accuracy (all 60) | **50%** (30 / 60) |
| Accuracy, direct lookups | 53% (16 / 30) |
| Accuracy, computed metrics | 47% (14 / 30) |
| Audit pass rate | 78% |
| False-pass rate (wrong but passed, of passed) | 47% |
| Wrong answers caught by the auditor | 8 of 30 |
| Answers with valid citations | 85% |
| Correct company filter detected | 60 / 60 |
| Correct fiscal-year filter detected | 59 / 60 |
| Agent rounds used (1 / 2 / 3) | 29 / 16 / 15 questions |
| Time per question (min / median / avg / max) | 35 s / 77 s / 100 s / 272 s |

**Why the 30 wrong answers failed.** For each one, I checked whether the expected figure
appears in any chunk the answer cited:
- **25 of 30: retrieval miss.** The right statement row was never among the chunks used. The agent
  answered from a nearby table instead, such as segment assets instead of total assets.
- **5 of 30: reasoning miss.** The right figure was in the sources, but the model picked another
  number or got the scale wrong (e.g. "$3.001 billion" for Alphabet's $73,795M net income).

So retrieval is the main bottleneck at this scale, not the LLM's arithmetic. The Python sandbox
computes correctly whenever it is given the right inputs.

---

## 5. Regression check: the old 25 Apple questions on the big index

The same 25 hand-written Apple FY2015 questions, first run on a 715-chunk index holding only that PDF,
were run again on the 98,137-chunk index.

| Metric | Old index (715 chunks) | New index (98,137 chunks) |
|---|---|---|
| Accuracy (all 25) | 52% | **68%** |
| Accuracy, lookups | 69% | 92% |
| Accuracy, computed | 33% | 42% |
| False-pass rate | 45% | 23% |
| Wrong answers caught by auditor | 2 of 12 | 3 of 8 |
| Avg. time per question | 148 s | 100 s |

- 17 of these 25 questions don't name the company ("What was diluted EPS for fiscal 2015?").
  So no filter applied, and they were answered against all 98,137 chunks from 90 filings.
- Adding 137 times more data did not drown the right answer.
- Part of the gain is likely the `[Apple Inc. (AAPL) 10-K, fiscal 2015]` header now on every chunk,
  which the reranker can match. Mistral-7B output also varies between runs, so a 25-question
  comparison is a direction, not a precise effect size.

---

## 6. Build timeline

| When | What |
|---|---|
| 2026-09-13 → 09-17 | Original project: LangGraph agents, FastAPI, Pinecone, MCP sandbox |
| 2026-10-30 23:23 | PR #1: real hybrid search (BM25 + dense + RRF + reranker), grounded audit, sandbox timeout, citations |
| 2026-10-31 00:04 | PR #2: table-aware PDF parsing (rows keep their year columns) |
| 2026-10-31 09:00 | PR #3: 25-question golden eval, baseline 52% |
| 2026-10-31 11:06 | EDGAR pivot started: company list, SEC client, download |
| 2026-10-31 11:19 | HTML table parser, XBRL golden builder, company/year filters |
| 2026-10-31 ≈11:20 → 14:31 | Full indexing (89 filings + PDF) |
| 2026-10-31 15:43 → 18:05 | Both agent evals |

---

## 7. Known gaps and next steps

- **Retrieval is the bottleneck** (25 of 30 misses). Next: a statement-aware retriever that always
  adds the consolidated income statement / balance sheet chunk for a filtered company-year, or
  query rewriting into the exact line-item name.
- **Year detection edge case:** Home Depot's fiscal 2025 ends February 1, 2026. The question mentions
  "2026", so the year filter fell back to company-only. This is the 1 of 60 missed year filters.
- **The auditor checks that figures exist in the sources, not that they are the right line item.**
  Only 8 of 30 wrong answers were caught.
- **A larger local model** (or a hosted one) on the same 60-question sample is the obvious next comparison.
