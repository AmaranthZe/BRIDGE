"""
Soft Candidate Selection with Random Exploration and MMR.

This module implements:
- Co-attention scoring between mouse and human gene embeddings
- Soft mask with sigmoid falloff (instead of hard top-k truncation)
- Random exploration to avoid local minima
- Maximal Marginal Relevance (MMR) for diverse candidate selection (A-fix #3)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def maximal_marginal_relevance(
    query_vec: Tensor,
    candidate_vecs: Tensor,
    candidate_scores: Tensor,
    lambda_mmr: float = 0.25,
    top_k: Optional[int] = None,
) -> Tensor:
    """
    Maximal Marginal Relevance (MMR) for diverse candidate selection.

    Iteratively selects candidates that balance relevance to the query
    with diversity (dissimilarity to already-selected candidates).

    MMR(c) = lambda * rel(c) - (1-lambda) * max_{s in S} sim(c, s)

    Args:
        query_vec: Query vector [D] or [1, D]
        candidate_vecs: Candidate vectors [K, D]
        candidate_scores: Relevance scores [K]
        lambda_mmr: Trade-off (0=max diversity, 1=pure relevance)
        top_k: Number of candidates to select (default: all)

    Returns:
        Indices into candidate_vecs in MMR order [top_k]
    """
    K = candidate_vecs.shape[0]
    if top_k is None:
        top_k = K
    top_k = min(top_k, K)

    if K == 0:
        return torch.tensor([], dtype=torch.long, device=candidate_vecs.device)

    # Normalize for cosine similarity
    C = F.normalize(candidate_vecs, p=2, dim=-1)  # [K, D]

    # Precompute candidate-candidate similarity matrix
    cc_sim = C @ C.t()  # [K, K]

    # Normalize relevance scores to [0, 1]
    rel = candidate_scores.float().clone()
    if (rel.max() - rel.min()).item() > 1e-8:
        rel = (rel - rel.min()) / (rel.max() - rel.min())

    selected: List[int] = []
    remaining = list(range(K))

    for _ in range(top_k):
        if not remaining:
            break

        if len(selected) == 0:
            # First selection: pick highest relevance
            best_local = int(rel[remaining].argmax().item())
            best_idx = remaining[best_local]
        else:
            best_score = float('-inf')
            best_idx = remaining[0]
            selected_t = torch.tensor(selected, dtype=torch.long, device=cc_sim.device)

            for idx in remaining:
                max_sim_to_selected = cc_sim[idx, selected_t].max().item()
                mmr_score = lambda_mmr * rel[idx].item() - (1 - lambda_mmr) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return torch.tensor(selected, dtype=torch.long, device=candidate_vecs.device)


class SoftCandidateSelector(nn.Module):
    """
    Soft candidate selection with learned projections and random exploration.
    """

    def __init__(
        self,
        embed_dim: int,
        topc: int = 256,
        temperature: float = 0.1,
        explore_ratio: float = 0.05,
        T_init: float = 0.07,
        max_candidates: int = 1024,
        mmr_lambda: float = 0.25,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            topc: Top-c candidates for soft threshold
            temperature: Temperature for sigmoid soft mask
            explore_ratio: Fraction of random exploration candidates
            T_init: Temperature for prior coupling softmax
            max_candidates: Maximum number of candidates to return
            mmr_lambda: MMR diversity-relevance trade-off (0=max diversity, 1=pure relevance)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.topc = topc
        self.temperature = temperature
        self.explore_ratio = explore_ratio
        self.T_init = T_init
        self.max_candidates = max_candidates
        self.mmr_lambda = mmr_lambda

        # Learnable projections for co-attention
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)

    def forward(
        self,
        E_source: Tensor,  # Source embeddings [B, D] or [N_source, D]
        E_target: Tensor,  # Target embeddings [N_target, D]
        source_indices: Optional[Tensor] = None,  # If E_source is full matrix
        return_full_attention: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute soft candidates and prior coupling with per-query row mask.

        Args:
            E_source: Source gene embeddings (batch or full)
            E_target: Target gene embeddings (full)
            source_indices: Indices into E_source if it's the full matrix
            return_full_attention: Return full attention matrix

        Returns:
            Tuple of:
                - J_cand: Candidate target indices [num_candidates]
                - soft_mask: Soft mask over candidates [B, num_candidates]
                - Pi0: Prior coupling over candidates [B, num_candidates]
                - row_mask: Per-query boolean mask [B, num_candidates] (True = allowed)
                - A: Full attention matrix [B, N_target] (if return_full_attention)
        """
        # Get batch embeddings
        if source_indices is not None:
            E_batch = E_source[source_indices]  # [B, D]
        else:
            E_batch = E_source  # [B, D]

        B = E_batch.shape[0]
        N_target = E_target.shape[0]
        D = self.embed_dim

        # Compute co-attention scores
        Q = self.W_Q(E_batch)  # [B, D]
        K = self.W_K(E_target)  # [N_target, D]

        # L2 normalize to use cosine similarity (improves discriminative power)
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # Cosine similarity (no scaling needed after normalization)
        A = torch.matmul(Q, K.t())  # [B, N_target], values in [-1, 1]

        # Get top-c threshold per row
        topc = min(self.topc, N_target)
        topk_vals, topk_idx = torch.topk(A, topc, dim=1)  # [B, topc]
        tau = topk_vals[:, -1:].expand(-1, N_target)  # [B, N_target]

        # Soft mask with sigmoid falloff
        soft_mask = torch.sigmoid((A - tau) / self.temperature)  # [B, N_target]

        # Random exploration: add random candidates
        n_explore = max(1, int(self.explore_ratio * topc))
        if n_explore > 0 and self.training:
            # Clone to avoid inplace modification
            soft_mask = soft_mask.clone()
            for i in range(B):
                # Sample random indices not in top-c
                all_idx = torch.arange(N_target, device=E_target.device)
                top_mask = torch.zeros(N_target, dtype=torch.bool, device=E_target.device)
                top_mask[topk_idx[i]] = True

                # Get non-top indices
                non_top_idx = all_idx[~top_mask]
                if len(non_top_idx) > 0:
                    n_sample = min(n_explore, len(non_top_idx))
                    perm = torch.randperm(len(non_top_idx), device=E_target.device)[:n_sample]
                    explore_idx = non_top_idx[perm]
                    # Boost soft mask for exploration candidates
                    soft_mask[i, explore_idx] = torch.clamp(
                        soft_mask[i, explore_idx] + 0.5,
                        max=1.0
                    )

        # Prior coupling with temperature-scaled softmax
        Pi0_full = F.softmax(A / self.T_init, dim=1) * soft_mask  # [B, N_target]
        Pi0_full = Pi0_full / (Pi0_full.sum(dim=1, keepdim=True) + 1e-8)

        # Find candidate indices (any row has soft_mask > threshold)
        threshold = 0.1  # Stricter threshold
        cand_mask = (soft_mask > threshold).any(dim=0)  # [N_target]
        J_cand = cand_mask.nonzero(as_tuple=True)[0]  # [num_candidates]

        if len(J_cand) == 0:
            # Fallback: use top-c from first row
            J_cand = topk_idx[0]

        # Hard cap on maximum candidates to prevent OOM — use MMR for diversity (A-fix #3)
        if len(J_cand) > self.max_candidates:
            avg_scores = soft_mask[:, J_cand].mean(dim=0)  # [num_cand]
            cand_vecs = K[J_cand]  # [num_cand, D] — already L2 normalized
            mmr_order = maximal_marginal_relevance(
                query_vec=Q.mean(dim=0),   # average query
                candidate_vecs=cand_vecs,
                candidate_scores=avg_scores,
                lambda_mmr=self.mmr_lambda,
                top_k=self.max_candidates,
            )
            J_cand = J_cand[mmr_order]

        # Extract soft mask and prior for candidates only
        soft_mask_cand = soft_mask[:, J_cand]  # [B, num_candidates]
        Pi0_cand = Pi0_full[:, J_cand]  # [B, num_candidates]

        # Build per-query boolean row mask: True where this row's top-c is present
        # Each row's own candidates are those with soft_mask > threshold
        row_mask = soft_mask_cand > threshold  # [B, num_candidates]

        # Renormalize Pi0 over candidates
        Pi0_cand = Pi0_cand / (Pi0_cand.sum(dim=1, keepdim=True) + 1e-8)

        if return_full_attention:
            return J_cand, soft_mask_cand, Pi0_cand, row_mask, A
        else:
            return J_cand, soft_mask_cand, Pi0_cand, row_mask, None


def build_soft_candidates(
    E_m: Tensor,
    E_h: Tensor,
    I_batch: Tensor,
    W_Q: Tensor,
    W_K: Tensor,
    topc: int = 256,
    temperature: float = 0.1,
    explore_ratio: float = 0.05,
    T_init: float = 0.07,
    training: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Functional version of soft candidate selection.

    Args:
        E_m: Mouse embeddings [N_m, D]
        E_h: Human embeddings [N_h, D]
        I_batch: Batch mouse indices [B]
        W_Q: Query projection [D, D]
        W_K: Key projection [D, D]
        topc: Top-c candidates
        temperature: Soft mask temperature
        explore_ratio: Random exploration ratio
        T_init: Prior coupling temperature
        training: Whether in training mode

    Returns:
        Tuple of (J_cand, soft_mask, Pi0)
    """
    D = E_m.shape[1]
    B = len(I_batch)
    N_h = E_h.shape[0]

    # Get batch embeddings
    E_batch = E_m[I_batch]  # [B, D]

    # Compute attention scores
    Q = E_batch @ W_Q  # [B, D]
    K = E_h @ W_K  # [N_h, D]

    # L2 normalize for cosine similarity
    Q = F.normalize(Q, p=2, dim=-1)
    K = F.normalize(K, p=2, dim=-1)
    A = Q @ K.t()  # [B, N_h] - cosine similarity

    # Top-c threshold
    topc = min(topc, N_h)
    topk_vals, topk_idx = torch.topk(A, topc, dim=1)
    tau = topk_vals[:, -1:].expand(-1, N_h)

    # Soft mask
    soft_mask = torch.sigmoid((A - tau) / temperature)

    # Random exploration (only during training)
    if training:
        n_explore = max(1, int(explore_ratio * topc))
        for i in range(B):
            explore_idx = torch.randperm(N_h, device=E_h.device)[:n_explore]
            soft_mask[i, explore_idx] = torch.clamp(soft_mask[i, explore_idx] + 0.5, max=1.0)

    # Prior coupling
    Pi0 = F.softmax(A / T_init, dim=1) * soft_mask
    Pi0 = Pi0 / (Pi0.sum(dim=1, keepdim=True) + 1e-8)

    # Candidate indices
    J_cand = (soft_mask > 0.01).any(dim=0).nonzero(as_tuple=True)[0]

    if len(J_cand) == 0:
        J_cand = topk_idx[0]

    return J_cand, soft_mask[:, J_cand], Pi0[:, J_cand]
