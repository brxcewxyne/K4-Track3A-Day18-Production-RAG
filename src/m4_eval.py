from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _to_float(value, default: float = 0.0) -> float:
    """Ép NumPy/pandas scalar về float Python; NaN/không hợp lệ → default."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": [str(q) for q in questions],
            "answer": [str(a) for a in answers],
            "contexts": [list(c) for c in contexts],
            "ground_truth": [str(g) for g in ground_truths],
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=_to_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_to_float(row.get("answer_relevancy", 0.0)),
                context_precision=_to_float(row.get("context_precision", 0.0)),
                context_recall=_to_float(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]
        aggregate = {
            metric: _to_float(df[metric].mean()) if metric in df.columns else 0.0
            for metric in METRIC_NAMES
        }
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


DIAGNOSTIC_TREE = {
    "faithfulness": (
        "LLM answer is not sufficiently grounded in retrieved evidence / possible hallucination.",
        "Tighten the system/generation prompt and reduce temperature toward 0.",
    ),
    "context_recall": (
        "Retrieval is missing relevant evidence.",
        "Improve chunking and/or BM25/retrieval coverage.",
    ),
    "context_precision": (
        "Irrelevant context is ranked too highly.",
        "Improve Cross-Encoder reranking and/or metadata filtering.",
    ),
    "answer_relevancy": (
        "Generated answer is not focused on the user's question.",
        "Rewrite the generation prompt to require a more direct answer.",
    ),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    failures = []
    for result in eval_results:
        metrics = {
            "faithfulness": _to_float(result.faithfulness),
            "answer_relevancy": _to_float(result.answer_relevancy),
            "context_precision": _to_float(result.context_precision),
            "context_recall": _to_float(result.context_recall),
        }
        mean_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]
        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": mean_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    failures.sort(key=lambda f: f["score"])
    return failures[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
