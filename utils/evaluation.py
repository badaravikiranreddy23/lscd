"""
evaluation.py
Evaluation metrics: Spearman correlation, accuracy, F1.
"""

from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple
import numpy as np


def spearman_correlation(predicted: List[float], gold: List[float]) -> Tuple[float, float]:
    """
    Compute Spearman correlation between predicted graded scores and gold scores.
    Returns (rho, p_value).
    """
    rho, pval = spearmanr(predicted, gold)
    return float(rho), float(pval)


def binary_metrics(predicted_scores: List[float], gold_binary: List[int],
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Convert graded scores to binary using a threshold and compute accuracy + F1.
    """
    predicted_binary = [1 if s >= threshold else 0 for s in predicted_scores]
    acc = accuracy_score(gold_binary, predicted_binary)
    f1 = f1_score(gold_binary, predicted_binary, zero_division=0)
    return {"accuracy": acc, "f1": f1, "threshold": threshold}


def find_best_threshold(predicted_scores: List[float], gold_binary: List[int]) -> float:
    """Find threshold that maximizes F1."""
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.1, 1.0, 0.05):
        metrics = binary_metrics(predicted_scores, gold_binary, threshold=thresh)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = thresh
    return best_thresh


def print_results(word_scores: Dict[str, float], gold_graded: Dict[str, float],
                  gold_binary: Dict[str, int]) -> None:
    """Pretty print evaluation results."""
    words = [w for w in word_scores if w in gold_graded]
    predicted = [word_scores[w] for w in words]
    gold_g = [gold_graded[w] for w in words]
    gold_b = [gold_binary.get(w, 0) for w in words]

    rho, pval = spearman_correlation(predicted, gold_g)
    best_thresh = find_best_threshold(predicted, gold_b)
    bin_metrics = binary_metrics(predicted, gold_b, threshold=best_thresh)

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Subtask 2 (Graded) - Spearman rho: {rho:.4f}  (p={pval:.4f})")
    print(f"Subtask 1 (Binary) - Accuracy: {bin_metrics['accuracy']:.4f}  F1: {bin_metrics['f1']:.4f}")
    print(f"  Best threshold used: {best_thresh:.2f}")
    print("=" * 50)
    print("\nPer-word scores:")
    for w in words:
        print(f"  {w:20s}  predicted={word_scores[w]:.4f}  gold={gold_graded[w]:.4f}")
