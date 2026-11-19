"""
Builds the QLoRA training set for the Code Executor step, from the XBRL golden set.

Why this task: in the 60-question eval, the sandbox computed every formula it was given correctly,
but the model often picked the wrong row (a segment or quarterly figure) or the wrong formula.
So the fine-tune teaches exactly that step: question + retrieved 10-K chunks -> Python code with the
right figures as literals and the right formula.

How each example is made:
1. The question comes from data/eval/edgar_golden_qa.jsonl, whose answers are XBRL facts.
2. The context is what the production retriever returns for it (same filters, same top 5), so the
   model learns to pick the right row among real distractors. If a needed figure is not in those
   5 chunks, the lowest-ranked chunk is swapped for the chunk of that filing that holds it, because
   training on context without the answer would teach the model to invent numbers.
3. The target code is written from the golden inputs and formula, run in the same restricted
   builtins as the sandbox, and kept only if it reproduces the golden answer.
4. The prompt is built by build_code_prompt(), the function production uses.

Leakage guard: only companies that do NOT appear in the 60-question eval sample (seed 7) are used,
so the before/after comparison on that sample is fair.

    python -m src.training.dataset_prep            # writes data/training/code_sft_{train,val}.jsonl
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.agents.nodes import build_code_prompt, format_context
from src.agents.tools import get_retriever
from src.eval.edgar_golden import COMPUTED, appears_in
from src.eval.golden_eval import sample_questions
from src.retrieval.query_filters import detect_filters
from src.retrieval.sparse_store import load_corpus

GOLDEN = Path("data/eval/edgar_golden_qa.jsonl")
OUT_DIR = Path("data/training")

# Python expression per computed metric; {y} = fiscal year, {p} = prior year.
# Must match the formulas in src/eval/edgar_golden.py COMPUTED (checked in build_code).
FORMULAS = {
    "revenue_growth": "round((revenue_{y} / revenue_{p} - 1) * 100, 2)",
    "net_income_growth": "round((net_income_{y} / net_income_{p} - 1) * 100, 2)",
    "gross_margin": "round(gross_profit_{y} / revenue_{y} * 100, 2)",
    "operating_margin": "round(operating_income_{y} / revenue_{y} * 100, 2)",
    "net_margin": "round(net_income_{y} / revenue_{y} * 100, 2)",
    "rd_intensity": "round(rd_{y} / revenue_{y} * 100, 2)",
    "current_ratio": "round(current_assets_{y} / current_liabilities_{y}, 3)",
}
SANDBOX_BUILTINS = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len,
                    "round": round, "float": float, "int": int}


def metric_of(item: Dict[str, Any]) -> str:
    return item["id"].split("-", 2)[2]


def literal(figure: str) -> str:
    """'391,035' -> '391035', '6.08' -> '6.08' (how the filing prints it, as a Python number)."""
    return figure.replace(",", "")


def build_code(item: Dict[str, Any]) -> str:
    """Target code: one named literal per input figure, then `result`."""
    metric, year = metric_of(item), item["fiscal_year"]
    if item["type"] == "lookup":
        name = f"{metric}_{year}"
        return f"{name} = {literal(item['evidence'][0])}\nresult = {name}"

    inputs = COMPUTED[metric][1]
    lines = []
    for name, figure in zip(inputs, item["evidence"]):
        var_year = year - 1 if name.endswith("_prior") else year
        lines.append(f"{name.replace('_prior', '')}_{var_year} = {literal(figure)}")
    lines.append("result = " + FORMULAS[metric].format(y=year, p=year - 1))
    return "\n".join(lines)


def run_like_sandbox(code: str) -> Optional[float]:
    scope: Dict[str, Any] = {}
    exec(code, {"__builtins__": SANDBOX_BUILTINS}, scope)
    return scope.get("result")


def close_enough(value: float, answer: float, unit: str) -> bool:
    if unit in ("percent", "ratio"):
        return abs(value - answer) <= 0.1
    return abs(value - answer) <= 0.005 * max(1.0, abs(answer))


def add_missing_evidence(chunks: List[Dict[str, Any]], item: Dict[str, Any],
                         filing_chunks: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Make sure every input figure is in the context; None if the filing has no chunk with it."""
    chunks = list(chunks)
    for figure in item["evidence"]:
        if any(appears_in(c["text"], figure) for c in chunks):
            continue
        holder = next((c for c in filing_chunks if appears_in(c["text"], figure)), None)
        if holder is None:
            return None
        # Replace the weakest chunk that is not itself needed for another figure
        for i in range(len(chunks) - 1, -1, -1):
            if not any(appears_in(chunks[i]["text"], f) for f in item["evidence"]):
                chunks[i] = {"id": holder["id"], "text": holder["text"], "score": None}
                break
        else:
            chunks.append({"id": holder["id"], "text": holder["text"], "score": None})
    return chunks


def main(val_fraction: float = 0.1, seed: int = 7, limit: int = None):
    golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    eval_tickers = {q["ticker"] for q in sample_questions(golden, 60, seed=7)}
    items = [q for q in golden if q["ticker"] not in eval_tickers]
    if limit:
        items = items[:limit]
    print(f"Training companies: {sorted({q['ticker'] for q in items})} ({len(items)} questions); "
          f"{len(eval_tickers)} eval companies excluded", flush=True)

    corpus = load_corpus(settings.CORPUS_PATH)
    retriever = get_retriever()
    examples, stats = [], {"questions": len(items), "kept": 0, "context_patched": 0,
                           "no_chunk_with_figure": 0, "code_mismatch": 0}
    for n, item in enumerate(items, 1):
        code = build_code(item)
        value = run_like_sandbox(code)
        if value is None or not close_enough(value, item["answer"], item["unit"]):
            stats["code_mismatch"] += 1
            continue

        retrieved = retriever.retrieve(query=item["question"], top_k=5, filters=detect_filters(item["question"]))
        filing_chunks = [c for c in corpus if c.get("metadata", {}).get("ticker") == item["ticker"]
                         and c.get("metadata", {}).get("fiscal_year") == item["fiscal_year"]]
        context = add_missing_evidence(retrieved, item, filing_chunks)
        if context is None:
            stats["no_chunk_with_figure"] += 1
            continue
        if [c["id"] for c in context] != [c["id"] for c in retrieved]:
            stats["context_patched"] += 1

        examples.append({
            "id": item["id"],
            "prompt": [{"role": "user", "content": build_code_prompt(item["question"], format_context(context))}],
            "completion": [{"role": "assistant", "content": f"```python\n{code}\n```"}],
        })
        stats["kept"] += 1
        if n % 20 == 0:
            print(f"  {n}/{len(items)} done", flush=True)

    random.Random(seed).shuffle(examples)
    n_val = max(1, round(len(examples) * val_fraction))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("val", examples[:n_val]), ("train", examples[n_val:])):
        with (OUT_DIR / f"code_sft_{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    stats.update({"train": len(examples) - n_val, "val": n_val,
                  "companies": sorted({q["ticker"] for q in items}),
                  "excluded_eval_companies": sorted(eval_tickers)})
    (OUT_DIR / "code_sft_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    args = parser.parse_args()
    main(args.val_fraction, limit=args.limit)
