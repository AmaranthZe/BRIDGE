"""
Cost Matrix Computation for FGW.

This module computes:
- Structural cost: pairwise distances from embeddings
- Feature cost: cross-domain similarity from sparse matrices
- Supervision cost: from orthologs (if available)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def cosine_distance_matrix(X: Tensor, Y: Optional[Tensor] = None) -> Tensor:
    """
    Compute pairwise cosine distance matrix.

    Args:
        X: First set of embeddings [N, D]
        Y: Second set of embeddings [M, D] (if None, use X)

    Returns:
        Distance matrix [N, M] where dist = 1 - cos_sim
    """
    X_norm = F.normalize(X, p=2, dim=1)

    if Y is None:
        Y_norm = X_norm
    else:
        Y_norm = F.normalize(Y, p=2, dim=1)

    # Cosine similarity
    sim = torch.matmul(X_norm, Y_norm.t())

    # Convert to distance
    dist = 1 - sim

    return dist


def euclidean_distance_matrix(X: Tensor, Y: Optional[Tensor] = None) -> Tensor:
    """
    Compute pairwise squared Euclidean distance matrix.

    Args:
        X: First set of embeddings [N, D]
        Y: Second set of embeddings [M, D]

    Returns:
        Distance matrix [N, M]
    """
    if Y is None:
        Y = X

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    X_sqnorm = (X ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    Y_sqnorm = (Y ** 2).sum(dim=1, keepdim=True)  # [M, 1]

    dist = X_sqnorm + Y_sqnorm.t() - 2 * torch.matmul(X, Y.t())
    dist = torch.clamp(dist, min=0)  # Numerical stability

    return dist


def structural_cost(
    E_source: Tensor,
    E_target: Tensor,
    metric: str = 'cosine',
) -> Tuple[Tensor, Tensor]:
    """
    Compute structural cost matrices for FGW.

    Args:
        E_source: Source embeddings [N_s, D]
        E_target: Target embeddings [N_t, D]
        metric: Distance metric ('cosine' or 'euclidean')

    Returns:
        Tuple of (C_source [N_s, N_s], C_target [N_t, N_t])
    """
    if metric == 'cosine':
        C_s = cosine_distance_matrix(E_source)
        C_t = cosine_distance_matrix(E_target)
    elif metric == 'euclidean':
        C_s = euclidean_distance_matrix(E_source)
        C_t = euclidean_distance_matrix(E_target)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return C_s, C_t


def feature_cost_from_sparse(
    W_source: Dict[str, sp.csr_matrix],
    W_target: Dict[str, sp.csr_matrix],
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    feature_weights: Dict[str, float],
    device: str = 'cpu',
) -> Tensor:
    """
    Compute feature cost from sparse matrices.

    Cost = 1 - weighted_cosine_similarity

    Args:
        W_source: Source sparse matrices {'bp': W_sB, 'mf': W_sF, 'disease': W_sD}
        W_target: Target sparse matrices
        source_indices: Source gene indices [N_s]
        target_indices: Target gene indices [N_t]
        feature_weights: Weights per feature type {'bp': 0.25, 'mf': 0.15, 'disease': 0.60}
        device: Device for output tensor

    Returns:
        Feature cost matrix [N_s, N_t]
    """
    N_s = len(source_indices)
    N_t = len(target_indices)

    # Initialize weighted similarity
    weighted_sim = np.zeros((N_s, N_t), dtype=np.float32)
    total_weight = 0

    for key in ['bp', 'mf', 'disease']:
        W_s = W_source[key]
        W_t = W_target[key]
        weight = feature_weights.get(key, 1.0)

        # Extract rows for indices
        X_s = W_s[source_indices].toarray()  # [N_s, N_terms]
        X_t = W_t[target_indices].toarray()  # [N_t, N_terms]

        # Compute cosine similarity
        # Normalize rows
        norm_s = np.linalg.norm(X_s, axis=1, keepdims=True)
        norm_t = np.linalg.norm(X_t, axis=1, keepdims=True)
        norm_s[norm_s == 0] = 1
        norm_t[norm_t == 0] = 1

        X_s_norm = X_s / norm_s
        X_t_norm = X_t / norm_t

        sim = X_s_norm @ X_t_norm.T  # [N_s, N_t]

        weighted_sim += weight * sim
        total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        weighted_sim /= total_weight

    # Convert to cost
    cost = 1 - weighted_sim

    return torch.tensor(cost, dtype=torch.float32, device=device)


def feature_cost_from_tensors(
    X_source: Tensor,
    X_target: Tensor,
) -> Tensor:
    """
    Compute feature cost from pre-computed feature tensors.

    Args:
        X_source: Source features [N_s, F]
        X_target: Target features [N_t, F]

    Returns:
        Feature cost matrix [N_s, N_t]
    """
    return cosine_distance_matrix(X_source, X_target)


def supervision_cost(
    source_indices: Tensor,
    target_indices: Tensor,
    orthologs: Optional[Dict[int, Dict[int, float]]] = None,
    device: str = 'cpu',
) -> Tensor:
    """
    Compute supervision cost from ortholog annotations.

    Low cost for known ortholog pairs, high cost otherwise.

    Args:
        source_indices: Source gene indices [N_s]
        target_indices: Target gene indices [N_t]
        orthologs: Dict mapping source_idx -> {target_idx: confidence}
        device: Device for output tensor

    Returns:
        Supervision cost matrix [N_s, N_t]
    """
    N_s = len(source_indices)
    N_t = len(target_indices)

    # Default: high cost (no known relationship)
    L = torch.ones(N_s, N_t, device=device)

    if orthologs is None:
        return L

    # Create target index lookup
    target_to_pos = {int(idx): pos for pos, idx in enumerate(target_indices.tolist())}

    for i, src_idx in enumerate(source_indices.tolist()):
        if src_idx in orthologs:
            for tgt_idx, confidence in orthologs[src_idx].items():
                if tgt_idx in target_to_pos:
                    j = target_to_pos[tgt_idx]
                    # Low cost for known orthologs
                    L[i, j] = 1 - confidence

    return L


class CostComputer:
    """
    Unified cost computation for FGW.
    """

    def __init__(
        self,
        W_source: Dict[str, sp.csr_matrix],
        W_target: Dict[str, sp.csr_matrix],
        feature_weights: Dict[str, float],
        structural_metric: str = 'cosine',
        orthologs: Optional[Dict] = None,
        device: str = 'cpu',
    ):
        self.W_source = W_source
        self.W_target = W_target
        self.feature_weights = feature_weights
        self.structural_metric = structural_metric
        self.orthologs = orthologs
        self.device = device

    def compute_all(
        self,
        E_source: Tensor,
        E_target: Tensor,
        source_indices: Tensor,
        target_indices: Tensor,
        soft_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute all cost matrices.

        Args:
            E_source: Source embeddings [N_s, D]
            E_target: Target embeddings [N_t, D]
            source_indices: Source gene indices
            target_indices: Target gene indices
            soft_mask: Optional soft mask [N_s, N_t]

        Returns:
            Dictionary with 'C_s', 'C_t', 'M', 'L'
        """
        # Structural costs
        C_s, C_t = structural_cost(E_source, E_target, self.structural_metric)

        # Feature cost
        M = feature_cost_from_sparse(
            self.W_source,
            self.W_target,
            source_indices.cpu().numpy(),
            target_indices.cpu().numpy(),
            self.feature_weights,
            self.device,
        )

        # Apply soft mask if provided
        if soft_mask is not None:
            M = M * soft_mask

        # Supervision cost
        L = supervision_cost(
            source_indices,
            target_indices,
            self.orthologs,
            self.device,
        )

        return {
            'C_s': C_s,
            'C_t': C_t,
            'M': M,
            'L': L,
        }
