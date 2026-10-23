"""Evaluation metrics for cross-species gene mapping benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List


def recall_at_k(ranks: List[int], k: int) -> float:
    """Fraction of items with rank <= k.

    Args:
        ranks: List of integer ranks (1-based) for ground-truth targets.
        k: Cutoff threshold.

    Returns:
        Fraction in [0, 1].
    """
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= k) / len(ranks)


def mrr(ranks: List[int]) -> float:
    """Mean Reciprocal Rank.

    Args:
        ranks: List of integer ranks (1-based) for ground-truth targets.

    Returns:
        MRR score in [0, 1].
    """
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks]))


def median_rank(ranks: List[int]) -> float:
    """Median rank across all ground-truth targets.

    Args:
        ranks: List of integer ranks (1-based).

    Returns:
        Median rank value.
    """
    if not ranks:
        return float('inf')
    return float(np.median(ranks))


def hits_at_k(ranks: List[int], k: int) -> int:
    """Count of items with rank <= k.

    Args:
        ranks: List of integer ranks (1-based).
        k: Cutoff threshold.

    Returns:
        Integer count.
    """
    return sum(1 for r in ranks if r <= k)


def mean_rank(ranks: List[int]) -> float:
    """Mean rank across all ground-truth targets.

    Args:
        ranks: List of integer ranks (1-based).

    Returns:
        Mean rank value.
    """
    if not ranks:
        return float('inf')
    return float(np.mean(ranks))


def compute_all_metrics(ranks: List[int], k_values: List[int] = None) -> dict:
    """Compute a full suite of metrics for a list of ranks.

    Args:
        ranks: List of integer ranks (1-based).
        k_values: List of k cutoffs for Recall@k and Hits@k.
                  Defaults to [5, 10, 20, 50, 100].

    Returns:
        Dictionary with all metric values.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100]

    results = {
        "n": len(ranks),
        "mrr": mrr(ranks),
        "mean_rank": mean_rank(ranks),
        "median_rank": median_rank(ranks),
    }
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(ranks, k)
        results[f"hits@{k}"] = hits_at_k(ranks, k)

    return results
