# Project Metrics: SEC EDGAR Multi-Filing Pivot

Every number below was measured on one run and has a log or result file behind it
(`data/edgar/logs/`, `data/eval/*.json`). Nothing here is an estimate.

**Machine:** Apple M4 MacBook, 24 GB RAM. Embedding and reranking run on the M4 GPU (MPS);
the LLM is Mistral-7B served locally by Ollama. Vectors are stored in Pinecone (serverless, cosine).

**Models and settings:** `BAAI/bge-large-en-v1.5` embeddings (1024 dimensions),
`BAAI/bge-reranker-large` cross-encoder, chunks of 512 tokens with 50 overlap,
top 5 chunks after reranking, at most 3 agent rounds per question.

---

## 1. Scale

| | Value |
|---|---|
| Documents | 90 10-K filings from SEC EDGAR (HTML) |
| Companies | 30 |
| Fiscal years | 2023–2026 for every company, plus Apple FY2015 (for the hand-written test set) |
| Raw data | 270 MB filing HTML + 133 MB XBRL facts |
| Words | 5,201,706 (5,155,219 in the 89 FY2023–2026 filings + 46,487 in Apple FY2015) |
| Indexed chunks (vectors) | **98,355** (97,422 + 933) |
| Golden questions | **1,074** generated from XBRL + 25 written by hand (Apple FY2015) |

The test set grows with the corpus because nobody writes the XBRL answers by hand.

---

## 2. Pipeline steps and time taken

| # | Step | What it produced | Time | Log |
|---|---|---|---|---|
| 1 | Download from SEC EDGAR | 89 filings + XBRL facts for 30 companies, 401 MB | **55 s** | `logs/download.log` |
| 2 | Parse HTML to text | 281 MB HTML → 5,155,219 words | **49 s** (≈0.5 s per filing) | `logs/parse.log` |
| 3 | Chunk + embed + upsert to Pinecone | 97,422 chunks from 89 filings | **3 h 9 min** (11,356 s) | `logs/ingest.log` |
| 3b | Add Apple FY2015 (EDGAR HTML, for the hand-written set) | 933 chunks | 116 s | `logs/ingest_aapl_fy2015.log` |
| 4 | Build golden Q&A from XBRL | 1,074 questions (592 lookup, 482 computed), 261 skipped | **61 s** first run, 291 s on a later re-run | `logs/golden_build.log` |
| 5 | Unit tests | 36 passed, 0 failed | **5.8 s** | `logs/unit_tests.log` |
| 6 | Agent eval, 60 sampled questions | scores in section 4 | **1 h 40 min** (5,996 s) | `logs/eval_60.log` |
| 7 | Agent eval, 25 hand-written Apple FY2015 questions | scores in section 5 | **61 min** (shared GPU) | `logs/eval_apple2015.log` |

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

## 4. Agent results: 60 questions on the 98,137-chunk index (before Apple FY2015 was re-indexed from HTML)

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

## 5. The 25 hand-written Apple FY2015 questions on the full index

The hand-written Apple FY2015 set, run against all 98,355 chunks.

| Metric | Value |
|---|---|
| Accuracy (all 25) | **64%** |
| Accuracy, lookups | 77% (10 of 13) |
| Accuracy, computed | 50% (6 of 12) |
| Accuracy, company named → filtered to Apple FY2015 | 75% (6 of 8) |
| Accuracy, no company named → all 90 filings searched | 59% (10 of 17) |
| False-pass rate | 26% |
| Wrong answers caught by auditor | 4 of 9 |
| Avg. time per question | 147 s |

- 17 of these 25 questions don't name the company ("What was diluted EPS for fiscal 2015?"),
  so no filter applies and they are answered against all 98,355 chunks from 90 filings.
- Failures: 5 wrong rows from the right filing, 2 answered from another company's filing
  (Home Depot, Microsoft), 1 unit slip (EPS 9.22 → 0.0092), 1 made-up input.
- The run shared the M4 GPU with a retrieval benchmark, which inflates the time per question.
  Mistral-7B output also varies between runs, so a 25-question score is a direction, not a precise effect size.
- Log: `logs/eval_apple2015.log`; per-question reports: `data/eval/apple2015.json`.

---

## 6. Build timeline

| When | What |
|---|---|
| 2026-09-13 → 09-17 | Original project on a single Apple 10-K: LangGraph agents, FastAPI, Pinecone, MCP sandbox |
| 2026-10-30 23:23 | PR #1: real hybrid search (BM25 + dense + RRF + reranker), grounded audit, sandbox timeout, citations |
| 2026-10-31 00:04 | PR #2: table-aware parsing (rows keep their year columns) |
| 2026-10-31 09:00 | PR #3: 25-question hand-written golden eval (Apple FY2015) |
| 2026-10-31 11:06 | EDGAR pivot started: company list, SEC client, download |
| 2026-10-31 11:19 | HTML table parser, XBRL golden builder, company/year filters |
| 2026-10-31 ≈11:20 → 14:31 | Full indexing (89 filings) |
| 2026-10-31 15:43 → 18:05 | 60-question agent eval |
| 2026-11-19 | Apple FY2015 re-indexed from its EDGAR HTML (one parser for every filing); unused first-version code removed; Docker API reaches Ollama and the BM25 corpus (PR #8); Apple FY2015 eval re-run |

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

---

## 8. Semantic cache (2026-11-18)

**Calibration** (`scripts/calibrate_semantic_cache.py`, bge-large-en-v1.5, 12 company-years, 16 everyday
rewordings each). Each row compares a new question with a cached one:

| Pair | Pairs | Cosine range | Cosine ≥ 0.80 alone | Scope key + cosine ≥ 0.80 |
|---|---|---|---|---|
| Rewording of the same question (should reuse) | 192 | 0.73 – 0.91 | 81.2% | **81.2%** |
| Other line item, same company and year | 2,608 | 0.58 – 0.90 | 14.0% | **0%** |
| Same line item, previous year | 192 | 0.69 – 0.88 | 57.8% | **0%** |
| Same line item, one segment ("in the Americas segment") | 192 | 0.72 – 0.88 | 51.0% | **0%** |
| Same line item, one quarter ("for the fourth quarter only") | 192 | 0.71 – 0.89 | 56.8% | **0%** |

Cosine similarity alone cannot be made safe: the wrong-figure pairs reach 0.90, above most true rewordings.
The scope key (company/year filter + years written + line items, segment/quarter/adjusted questions never
cached) removes every wrong reuse, and the threshold only has to separate what is left.
Line-item detection maps all 1,074 golden questions to the right item.

**End to end** (real API, Redis 7, Mistral-7B, Pinecone; `/api/v1/agent/analyze`):

| Question | Time | Cached | Similarity |
|---|---|---|---|
| What was Apple's total revenue in fiscal 2024? | 193.39 s (agent, audit passed, 1 round) | no | – |
| How much revenue did Apple make in FY2024? | 0.06 s | yes | 0.837 |
| Apple total net sales for fiscal year 2024? | 0.04 s | yes | 0.916 |

Tests: 30 new offline unit tests (`tests/unit/test_semantic_cache.py`) with a fake Redis and embedder.
