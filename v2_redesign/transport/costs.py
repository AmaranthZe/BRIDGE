"""Cost matrix computation for FGW.

Critical fix from v1: Use pre-computed SVD features (Z_h, Z_m) instead of
calling .toarray() on sparse matrices, which was CPU-bound with GPU ping-ponging.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def cosine_distance_matrix(X: Tensor, Y: Optional[Tensor] = None) -> Tensor:
    """Pairwise cosine distance: 1 - cos_sim."""
    X_n = F.normalize(X, p=2, dim=1, eps=1e-8)
    Y_n = F.normalize(Y, p=2, dim=1, eps=1e-8) if Y is not None else X_n
    return 1 - X_n @ Y_n.t()


def sqeuclidean_distance_matrix(X: Tensor, Y: Optional[Tensor] = None) -> Tensor:
    """Pairwise squared Euclidean distance, normalized to [0, 1].

    Unlike cosine distance, this preserves magnitude variation —
    critical for breaking embedding isotropy.
    """
    Y = Y if Y is not None else X
    D = torch.cdist(X, Y, p=2).pow(2)
    d_max = D.max()
    if d_max > 1e-8:
        D = D / d_max
    return D


def structural_cost(E_source: Tensor, E_target: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute intra-species structural cost matrices.

    Uses squared Euclidean distance (not cosine) so that magnitude
    variation in centered embeddings contributes structural signal.
    """
    C_s = sqeuclidean_distance_matrix(E_source)
    C_t = sqeuclidean_distance_matrix(E_target)
    return C_s, C_t


def feature_cost_from_svd(
    Z_source: Tensor,
    Z_target: Tensor,
    source_indices: Tensor,
    target_indices: Tensor,
) -> Tensor:
    """
    Compute feature cost from pre-computed SVD features.

    FIX: v1 used .toarray() on sparse matrices, materializing huge dense arrays
    on CPU and ping-ponging between CPU/GPU. This version uses the pre-computed
    unified SVD embeddings (Z_h, Z_m) which are already dense, compact, and
    can live on GPU.

    Args:
        Z_source: Source SVD features [N_source, D]
        Z_target: Target SVD features [N_target, D]
        source_indices: Batch source indices [B]
        target_indices: Candidate target indices [J]

    Returns:
        Feature cost matrix [B, J]
    """
    Z_s = Z_source[source_indices]  # [B, D]
    Z_t = Z_target[target_indices]  # [J, D]
    return cosine_distance_matrix(Z_s, Z_t)


class CostComputer:
    """Unified cost computation using SVD features."""

    def __init__(
        self,
        Z_h: Tensor,
        Z_m: Tensor,
        device: str = "cpu",
    ):
        """
        Args:
            Z_h: Human SVD features [N_h, D] (torch tensor on device)
            Z_m: Mouse SVD features [N_m, D] (torch tensor on device)
            device: Compute device
        """
        self.Z_h = Z_h.to(device) if isinstance(Z_h, Tensor) else torch.tensor(Z_h, dtype=torch.float32, device=device)
        self.Z_m = Z_m.to(device) if isinstance(Z_m, Tensor) else torch.tensor(Z_m, dtype=torch.float32, device=device)
        self.device = device

    def compute(
        self,
        E_source: Tensor,
        E_target: Tensor,
        source_indices: Tensor,
        target_indices: Tensor,
        direction: str = "mouse_to_human",
    ) -> Dict[str, Tensor]:
        """
        Compute structural and feature costs.

        Args:
            E_source: Source HGT embeddings [B, D]
            E_target: Target HGT embeddings [J, D]
            source_indices: Source gene indices [B]
            target_indices: Target gene indices [J]
            direction: 'mouse_to_human' or 'human_to_mouse'

        Returns:
            Dict with 'C_s', 'C_t', 'M'
        """
        C_s, C_t = structural_cost(E_source, E_target)

        if direction == "mouse_to_human":
            Z_src, Z_tgt = self.Z_m, self.Z_h
        else:
            Z_src, Z_tgt = self.Z_h, self.Z_m

        M = feature_cost_from_svd(Z_src, Z_tgt, source_indices, target_indices)

        return {"C_s": C_s, "C_t": C_t, "M": M}
