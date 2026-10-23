"""Evaluation metrics: P@k, R@k, nDCG@k, MRR, Jaccard."""

from typing import List, Set

import numpy as np


def precision_at_k(predicted: List[int], ground_truth: Set[int], k: int) -> float:
    if k == 0 or not predicted:
        return 0.0
    return sum(1 for p in predicted[:k] if p in ground_truth) / k


def recall_at_k(predicted: List[int], ground_truth: Set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    return sum(1 for p in predicted[:k] if p in ground_truth) / len(ground_truth)


def ndcg_at_k(predicted: List[int], ground_truth: Set[int], k: int) -> float:
    if k == 0 or not ground_truth:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(predicted[:k]) if p in ground_truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(predicted: List[int], ground_truth: Set[int]) -> float:
    for i, p in enumerate(predicted):
        if p in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0
