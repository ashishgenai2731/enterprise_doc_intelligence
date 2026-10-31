"""
Exact-answer evaluation of the agent against a golden Q&A set built from the 10-K.

RAGAs (evaluate_ragas.py) scores how faithful and relevant an answer sounds; this script
checks whether the number in the final report is actually right, and whether the auditor
caught the wrong ones. The key metric is the false-pass rate: wrong answers the audit let through.

Usage:
    python -m src.eval.golden_eval                 # all questions
    python -m src.eval.golden_eval --ids C02 L05   # a subset
    python -m src.eval.golden_eval --golden data/eval/edgar_golden_qa.jsonl --sample 60 --out edgar_results
"""
import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

GOLDEN_PATH = Path("data/eval/golden_qa.jsonl")
OUT_DIR = Path("data/eval")
QUESTION_TIMEOUT_SEC = 900

# Absolute tolerance per unit; amounts in $ millions use a relative tolerance instead
ABS_TOLERANCE = {"percent": 0.1, "ratio": 0.01, "usd": 0.01}
REL_TOLERANCE = 0.005

_CITATION_RE = re.compile(r"\[Source:[^\]]*\]")
_FIGURE_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(billion|bn|million|mn)?", re.IGNORECASE)


def extract_figures(text: str, unit: str) -> List[float]:
    """Numbers in the report, converted to the golden answer's unit ('$233.7 billion' -> 233700)."""
    figures = []
    for raw, scale in _FIGURE_RE.findall(_CITATION_RE.sub("", text)):
        value = float(raw.replace(",", ""))
        if unit == "usd_millions" and scale.lower() in ("billion", "bn"):
            value *= 1000
        figures.append(value)
    return figures


def is_correct(report: str, answer: float, unit: str) -> bool:
    """True if any figure in the report matches the answer. Signs are compared loosely
    because reports often say 'decreased by 23.3%' instead of '-23.3%'."""
    tolerance = ABS_TOLERANCE.get(unit, REL_TOLERANCE * abs(answer))
    return any(abs(abs(fig) - abs(answer)) <= tolerance for fig in extract_figures(report, unit))


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    correct = [r for r in results if r["correct"]]
    passed = [r for r in results if r["audit_passed"]]
    false_pass = [r for r in passed if not r["correct"]]
    return {
        "questions": n,
        "accuracy": len(correct) / n,
        "accuracy_lookup": _rate([r for r in results if r["type"] == "lookup"]),
        "accuracy_computed": _rate([r for r in results if r["type"] == "computed"]),
        "audit_pass_rate": len(passed) / n,
        "false_pass_rate": len(false_pass) / max(1, len(passed)),  # of passed answers, share that were wrong
        "wrong_caught_by_audit": sum(1 for r in results if not r["correct"] and not r["audit_passed"]),
        "wrong_total": n - len(correct),
        "citation_rate": sum(1 for r in results if r["sources"]) / n,
        "avg_iterations": sum(r["iterations"] for r in results) / n,
        "avg_latency_sec": sum(r["latency_sec"] for r in results) / n,
    }


def _rate(subset: List[Dict[str, Any]]) -> float:
    return sum(r["correct"] for r in subset) / len(subset) if subset else 0.0


def to_markdown(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    lines = [
        "| Metric | Value |", "|---|---|",
        f"| Accuracy (all {summary['questions']}) | **{summary['accuracy']:.0%}** |",
        f"| Accuracy, direct lookups | {summary['accuracy_lookup']:.0%} |",
        f"| Accuracy, computed metrics | {summary['accuracy_computed']:.0%} |",
        f"| Audit pass rate | {summary['audit_pass_rate']:.0%} |",
        f"| False-pass rate (wrong but passed, of passed) | {summary['false_pass_rate']:.0%} |",
        f"| Wrong answers caught by audit | {summary['wrong_caught_by_audit']} of {summary['wrong_total']} |",
        f"| Answers with valid citations | {summary['citation_rate']:.0%} |",
        f"| Avg. graph iterations | {summary['avg_iterations']:.2f} |",
        f"| Avg. latency per question | {summary['avg_latency_sec']:.0f}s |",
        "", "| ID | Expected | Correct | Audit | Iter. |", "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['id']} | {r['answer']} | {'✅' if r['correct'] else '❌'} | "
                     f"{'pass' if r['audit_passed'] else 'fail'} | {r['iterations']} |")
    return "\n".join(lines)


def sample_questions(golden: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Half lookups, half computed, drawn with a fixed seed so runs are comparable."""
    rng = random.Random(seed)
    picked = []
    for qtype in ("lookup", "computed"):
        pool = [g for g in golden if g["type"] == qtype]
        picked += rng.sample(pool, min(len(pool), n // 2 if qtype == "lookup" else n - n // 2))
    return picked


async def run(ids: List[str] = None, golden_path: Path = GOLDEN_PATH, sample: int = None,
              seed: int = 7, out_name: str = "results") -> None:
    from src.agents.graph import build_financial_agent_graph
    from src.agents.state import initial_agent_state

    golden = [json.loads(line) for line in Path(golden_path).open() if line.strip()]
    if ids:
        golden = [g for g in golden if g["id"] in ids]
    if sample:
        golden = sample_questions(golden, sample, seed)

    graph = build_financial_agent_graph()
    results = []
    for item in golden:
        start = time.perf_counter()
        try:
            state = await asyncio.wait_for(graph.ainvoke(initial_agent_state(item["question"])), QUESTION_TIMEOUT_SEC)
        except Exception as err:  # a crash or timeout is scored as a wrong, unaudited answer
            state = {"draft_report": f"ERROR: {err}", "audit_passed": False, "sources": [], "iteration_count": 0}
        report = state.get("draft_report", "")
        result = {
            **{k: item[k] for k in ("id", "type", "question", "answer", "unit")},
            "correct": is_correct(report, item["answer"], item["unit"]),
            "audit_passed": state.get("audit_passed", False),
            "sources": state.get("sources", []),
            "iterations": state.get("iteration_count", 0),
            "search_filters": state.get("search_filters", {}),
            "latency_sec": round(time.perf_counter() - start, 1),
            "report": report,
        }
        results.append(result)
        print(f"{item['id']}: {'correct' if result['correct'] else 'WRONG  '} | "
              f"audit {'pass' if result['audit_passed'] else 'fail'} | {result['latency_sec']}s", flush=True)
    write_results(results, out_name)


def rescore(golden_path: Path, out_name: str) -> None:
    """Re-check saved reports against the current golden answers, without re-running the agent
    (used after a golden answer is corrected)."""
    golden = {g["id"]: g for g in map(json.loads, Path(golden_path).open()) if g}
    results = json.loads((OUT_DIR / f"{out_name}.json").read_text())["results"]
    for result in results:
        result["answer"] = golden[result["id"]]["answer"]
        result["correct"] = is_correct(result["report"], result["answer"], result["unit"])
    write_results(results, out_name)


def write_results(results: List[Dict[str, Any]], out_name: str) -> None:
    summary = summarize(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{out_name}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    (OUT_DIR / f"{out_name}.md").write_text(to_markdown(summary, results) + "\n")
    print("\n" + to_markdown(summary, results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="Run only these question IDs")
    parser.add_argument("--golden", default=str(GOLDEN_PATH), help="Golden Q&A jsonl file")
    parser.add_argument("--sample", type=int, help="Evaluate a fixed random sample of this size")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results", help="Output name under data/eval/")
    parser.add_argument("--rescore", action="store_true", help="Re-score saved reports against the golden file")
    args = parser.parse_args()
    if args.rescore:
        rescore(Path(args.golden), args.out)
    else:
        asyncio.run(run(args.ids, Path(args.golden), args.sample, args.seed, args.out))
