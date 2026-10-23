"""Baseline methods for comparison.

B1: Random mapping
B2: Sequence-based orthologs (from Ensembl Compara)
B3: Cosine similarity on raw sparse features (no HGT, no FGW)
B4: HGT embeddings + cosine similarity (no FGW)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def baseline_random(
    N_source: int,
    N_target: int,
    topk: int = 20,
    seed: int = 3407,
) -> Dict[int, List[Tuple[int, float]]]:
    """B1: Random assignment baseline."""
    rng = np.random.RandomState(seed)
    predictions = {}
    for i in range(N_source):
        idx = rng.choice(N_target, size=topk, replace=False)
        scores = np.ones(topk) / topk
        predictions[i] = list(zip(idx.tolist(), scores.tolist()))
    return predictions


def baseline_cosine_sparse(
    W_source: Dict[str, sp.csr_matrix],
    W_target: Dict[str, sp.csr_matrix],
    feature_weights: Dict[str, float],
    topk: int = 20,
) -> Dict[int, List[Tuple[int, float]]]:
    """B3: Cosine similarity on raw sparse features (no learning)."""
    N_s = list(W_source.values())[0].shape[0]
    predictions = {}

    for i in range(N_s):
        weighted_sim = np.zeros(list(W_target.values())[0].shape[0], dtype=np.float32)
        total_w = 0

        for key in ("bp", "mf", "disease"):
            w = feature_weights.get(key, 1.0)
            row_s = W_source[key][i].toarray().flatten()
            norm_s = np.linalg.norm(row_s)
            if norm_s == 0:
                continue

            # Compute cosine with all target rows
            X_t = W_target[key].toarray()
            norms_t = np.linalg.norm(X_t, axis=1)
            norms_t[norms_t == 0] = 1.0
            sim = (X_t @ row_s) / (norms_t * norm_s)
            weighted_sim += w * sim
            total_w += w

        if total_w > 0:
            weighted_sim /= total_w

        top_idx = np.argsort(-weighted_sim)[:topk]
        predictions[i] = [(int(j), float(weighted_sim[j])) for j in top_idx]

    return predictions


def baseline_cosine_hgt(
    E_source: Tensor,
    E_target: Tensor,
    topk: int = 20,
) -> Dict[int, List[Tuple[int, float]]]:
    """B4: HGT embeddings + cosine similarity (no FGW)."""
    E_s = F.normalize(E_source, p=2, dim=-1)
    E_t = F.normalize(E_target, p=2, dim=-1)

    predictions = {}
    N_s = E_s.shape[0]
    chunk = 500

    for start in range(0, N_s, chunk):
        end = min(start + chunk, N_s)
        sim = E_s[start:end] @ E_t.t()  # [chunk, N_t]
        topk_vals, topk_idx = torch.topk(sim, topk, dim=1)

        for i in range(end - start):
            src = start + i
            predictions[src] = [
                (int(topk_idx[i, j]), float(topk_vals[i, j]))
                for j in range(topk)
            ]

    return predictions
