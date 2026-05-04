"""
Evaluation metrics calculator for RAG + Agent chatbot.

Metrics computed:
  1. Token Overlap F1 – lexical overlap between expected and generated answers
  2. Answer Relevancy – embedding cosine similarity (question ↔ answer)
  3. Faithfulness – LLM-as-judge: "Does answer match context?" (1-5 scale → 0-1)
  4. Context Precision – fraction of retrieved chunks relevant to question
  5. Context Recall – fraction of expected answer's facts supported by context

Quantitative metrics evaluated in evaluation/metrics_report.json.
"""

import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def token_overlap_f1(reference: str, hypothesis: str) -> float:
    """Token-level F1 score (used by SQuAD evaluation)."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())
    common = ref_tokens & hyp_tokens

    if not common:
        return 0.0
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 1.0 if ref_tokens == hyp_tokens else 0.0

    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics_summary(
    eval_results: List[Dict],
) -> Dict:
    """
    Compute aggregate statistics over all evaluation results.

    Parameters
    ----------
    eval_results : list of dict
        Output from Evaluator.run_eval_dataset() – one dict per question
        with keys: id, question, answer_relevancy, faithfulness,
        context_precision, token_overlap_f1, sources_count, etc.

    Returns
    -------
    dict with keys:
        "total_items"       – int
        "successful_items"  – int (no errors)
        "answer_relevancy_mean"   – float [0, 1]
        "faithfulness_mean"       – float [0, 1]
        "context_precision_mean"  – float [0, 1]
        "token_overlap_f1_mean"   – float [0, 1]
        "sources_count_mean"      – float
        "errors"            – list of error messages
    """
    total = len(eval_results)
    successful = [r for r in eval_results if "error" not in r]
    num_success = len(successful)
    errors = [r.get("error", "") for r in eval_results if "error" in r]

    if num_success == 0:
        logger.warning("No successful evaluations; returning zeros")
        return {
            "total_items": total,
            "successful_items": 0,
            "answer_relevancy_mean": 0.0,
            "faithfulness_mean": 0.0,
            "context_precision_mean": 0.0,
            "token_overlap_f1_mean": 0.0,
            "sources_count_mean": 0.0,
            "errors": errors,
        }

    # Extract numeric scores (handle "N/A" tokens)
    ar_scores = [
        float(r["answer_relevancy"])
        for r in successful
        if "answer_relevancy" in r and r["answer_relevancy"] != "N/A"
    ]
    faith_scores = [
        float(r["faithfulness"])
        for r in successful
        if "faithfulness" in r and r["faithfulness"] != "N/A"
    ]
    cp_scores = [
        float(r["context_precision"])
        for r in successful
        if "context_precision" in r and r["context_precision"] != "N/A"
    ]
    to_scores = [
        float(r["token_overlap_f1"])
        for r in successful
        if "token_overlap_f1" in r and r["token_overlap_f1"] != "N/A"
    ]
    sources = [
        float(r["sources_count"])
        for r in successful
        if "sources_count" in r
    ]

    return {
        "total_items": total,
        "successful_items": num_success,
        "answer_relevancy_mean": round(sum(ar_scores) / len(ar_scores), 4) if ar_scores else 0.0,
        "faithfulness_mean": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else 0.0,
        "context_precision_mean": round(sum(cp_scores) / len(cp_scores), 4) if cp_scores else 0.0,
        "token_overlap_f1_mean": round(sum(to_scores) / len(to_scores), 4) if to_scores else 0.0,
        "sources_count_mean": round(sum(sources) / len(sources), 2) if sources else 0.0,
        "errors": errors,
    }


def results_to_dataframe(eval_results: List[Dict]) -> pd.DataFrame:
    """Convert eval results list to a pandas DataFrame for display."""
    df = pd.DataFrame(eval_results)

    # Clean column order
    key_cols = [
        "id", "question", "generated_answer", "answer_relevancy",
        "faithfulness", "context_precision", "token_overlap_f1", "sources_count"
    ]
    available_cols = [c for c in key_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in available_cols]

    return df[available_cols + other_cols] if available_cols else df


def save_metrics_report(
    eval_results: List[Dict],
    output_path: str = "evaluation/metrics_report.json"
) -> None:
    """
    Save aggregated metrics report + detailed results to JSON.

    Parameters
    ----------
    eval_results : list of dict
        Detailed per-question results from Evaluator.run_eval_dataset()
    output_path : str
        Path to write JSON report
    """
    summary = compute_metrics_summary(eval_results)

    report = {
        "metadata": {
            "version": "1.0",
            "timestamp": pd.Timestamp.now().isoformat(),
            "num_questions_evaluated": summary["total_items"],
        },
        "summary": summary,
        "detailed_results": eval_results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Metrics report saved to {output_path}")


if __name__ == "__main__":
    # Standalone test
    test_results = [
        {
            "id": "q1",
            "question": "What is X?",
            "generated_answer": "X is the 24th letter",
            "expected_answer": "X is the 24th letter of the alphabet",
            "answer_relevancy": 0.92,
            "faithfulness": 0.85,
            "context_precision": 0.88,
            "token_overlap_f1": 0.75,
            "sources_count": 3,
        },
        {
            "id": "q2",
            "question": "What is Y?",
            "generated_answer": "Y is unknown variable",
            "expected_answer": "Y is a variable in mathematics",
            "answer_relevancy": 0.78,
            "faithfulness": 0.70,
            "context_precision": 0.65,
            "token_overlap_f1": 0.50,
            "sources_count": 2,
        },
    ]

    summary = compute_metrics_summary(test_results)
    print("Metrics Summary:")
    print(json.dumps(summary, indent=2))
