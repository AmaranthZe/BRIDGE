"""Soft Candidate Selection with MMR diversity.

Performance-critical: All operations are vectorized on GPU.
No Python loops over candidates or .item() calls in hot paths.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def maximal_marginal_relevance(
    candidate_vecs: Tensor,
    candidate_scores: Tensor,
    lambda_mmr: float = 0.25,
    top_k: Optional[int] = None,
) -> Tensor:
    """MMR for diverse candidate selection — fully vectorized on GPU.

    Greedy selection: each step picks the candidate that maximizes
    lambda * relevance - (1-lambda) * max_similarity_to_already_selected.
    The loop over top_k steps is O(top_k) Python iterations but each step
    is a single GPU tensor operation (no .item() calls).
    """
    K = candidate_vecs.shape[0]
    top_k = min(top_k or K, K)
    if K == 0:
        return torch.tensor([], dtype=torch.long, device=candidate_vecs.device)

    device = candidate_vecs.device
    C = F.normalize(candidate_vecs, p=2, dim=-1, eps=1e-8)
    cc_sim = C @ C.t()  # [K, K]

    rel = candidate_scores.float().clone()
    if (rel.max() - rel.min()).item() > 1e-8:
        rel = (rel - rel.min()) / (rel.max() - rel.min())

    selected = torch.empty(top_k, dtype=torch.long, device=device)
    mask = torch.zeros(K, dtype=torch.bool, device=device)
    max_sim = torch.full((K,), float("-inf"), device=device)

    # First: pick highest relevance
    selected[0] = rel.argmax()
    mask[selected[0]] = True
    max_sim = cc_sim[:, selected[0]]

    for step in range(1, top_k):
        # MMR score: lambda * relevance - (1-lambda) * max_sim_to_selected
        mmr = lambda_mmr * rel - (1 - lambda_mmr) * max_sim
        mmr[mask] = float("-inf")

        best = mmr.argmax()
        selected[step] = best
        mask[best] = True

        # Update max similarity incrementally
        max_sim = torch.maximum(max_sim, cc_sim[:, best])

    return selected


class SoftCandidateSelector(nn.Module):
    """Soft candidate selection with co-attention, random exploration, and MMR."""

    def __init__(
        self,
        embed_dim: int,
        topc: int = 256,
        temperature: float = 0.1,
        explore_ratio: float = 0.05,
        T_init: float = 0.07,
        max_candidates: int = 2048,
        mmr_lambda: float = 0.25,
        svd_score_weight: float = 0.7,
    ):
        super().__init__()
        self.topc = topc
        self.temperature = temperature
        self.explore_ratio = explore_ratio
        self.T_init = T_init
        self.max_candidates = max_candidates
        self.mmr_lambda = mmr_lambda

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)

        self.svd_score_weight = svd_score_weight

        # Per-modality retrieval weights (BP, MF, Disease)
        self.modality_weights = {"bp": 0.25, "mf": 0.15, "disease": 0.60}

    def forward(
        self,
        E_source: Tensor,
        E_target: Tensor,
        source_indices: Optional[Tensor] = None,
        eligible_mask: Optional[Tensor] = None,
        Z_source: Optional[Tensor] = None,
        Z_target: Optional[Tensor] = None,
        Z_source_mods: Optional[dict] = None,
        Z_target_mods: Optional[dict] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            eligible_mask: Optional boolean mask [N_t] — True = eligible target gene.
                           Ineligible genes are excluded from candidate selection.
            Z_source: Optional unified SVD features [N_s, D] (fallback).
            Z_target: Optional unified SVD features [N_t, D] (fallback).
            Z_source_mods: Optional dict with per-modality SVD features for source.
                           Keys: 'bp', 'mf', 'disease'. Values: [N_s, D] tensors.
            Z_target_mods: Optional dict with per-modality SVD features for target.
                           Keys: 'bp', 'mf', 'disease'. Values: [N_t, D] tensors.
        Returns:
            J_cand: Candidate indices [num_cand]
            soft_mask: Soft mask [B, num_cand]
            Pi0: Prior coupling [B, num_cand]
            row_mask: Per-query boolean mask [B, num_cand]
        """
        E_batch = E_source[source_indices] if source_indices is not None else E_source
        B = E_batch.shape[0]
        N_t = E_target.shape[0]

        # Learned co-attention scores (used for soft mask + Pi0)
        Q = F.normalize(self.W_Q(E_batch), p=2, dim=-1)
        K = F.normalize(self.W_K(E_target), p=2, dim=-1)
        A_qk = Q @ K.t()  # [B, N_t]

        # Per-modality SVD retrieval (weighted ensemble of per-modality cosines)
        if Z_source_mods is not None and Z_target_mods is not None:
            A_svd = torch.zeros(B, N_t, device=E_batch.device)
            for mod, weight in self.modality_weights.items():
                Zs = Z_source_mods[mod]
                Zt = Z_target_mods[mod]
                Zs_batch = Zs[source_indices] if source_indices is not None else Zs
                Zs_n = F.normalize(Zs_batch, p=2, dim=-1)
                Zt_n = F.normalize(Zt, p=2, dim=-1)
                A_svd = A_svd + weight * (Zs_n @ Zt_n.t())
            A_retrieve = A_svd
            A_score = self.svd_score_weight * A_svd + (1 - self.svd_score_weight) * A_qk
        elif Z_source is not None and Z_target is not None:
            # Fallback: unified SVD
            Z_batch = Z_source[source_indices] if source_indices is not None else Z_source
            Z_batch_n = F.normalize(Z_batch, p=2, dim=-1)
            Z_target_n = F.normalize(Z_target, p=2, dim=-1)
            A_svd = Z_batch_n @ Z_target_n.t()  # [B, N_t]
            A_retrieve = A_svd
            A_score = self.svd_score_weight * A_svd + (1 - self.svd_score_weight) * A_qk
        else:
            A_retrieve = A_qk
            A_score = A_qk

        # Mask out ineligible target genes
        if eligible_mask is not None:
            A_retrieve[:, ~eligible_mask] = float("-inf")
            A_score[:, ~eligible_mask] = float("-inf")

        topc = min(self.topc, N_t)
        topk_vals, topk_idx = torch.topk(A_retrieve, topc, dim=1)
        tau = topk_vals[:, -1:].expand(-1, N_t)
        soft_mask = torch.sigmoid((A_score - tau) / self.temperature)

        # Random exploration during training — fully vectorized, no in-place ops
        if self.training:
            n_explore = max(1, int(self.explore_ratio * topc))
            explore_idx = torch.randint(0, N_t, (B, n_explore), device=soft_mask.device)
            # Build a delta tensor (new, no autograd history) and add out-of-place
            boost = torch.zeros_like(soft_mask)
            boost.scatter_(1, explore_idx, 0.5)  # OK: boost is a fresh tensor
            soft_mask = torch.clamp(soft_mask + boost, max=1.0)

        # Prior coupling
        Pi0_full = F.softmax(A_score / self.T_init, dim=1) * soft_mask
        Pi0_full = Pi0_full / (Pi0_full.sum(dim=1, keepdim=True) + 1e-8)

        # Gather candidate indices
        threshold = 0.1
        cand_mask = (soft_mask > threshold).any(dim=0)
        J_cand = cand_mask.nonzero(as_tuple=True)[0]

        if len(J_cand) == 0:
            J_cand = topk_idx[0]

        # MMR cap (vectorized — no Python loops over candidates)
        if len(J_cand) > self.max_candidates:
            avg_scores = soft_mask[:, J_cand].mean(dim=0)
            # Pre-filter to 2x max_candidates by score to limit similarity matrix size
            prefilter_k = min(len(J_cand), self.max_candidates * 2)
            if prefilter_k < len(J_cand):
                _, pre_order = torch.topk(avg_scores, prefilter_k)
                J_cand = J_cand[pre_order]
                avg_scores = avg_scores[pre_order]
            mmr_order = maximal_marginal_relevance(
                K[J_cand], avg_scores,
                lambda_mmr=self.mmr_lambda, top_k=self.max_candidates,
            )
            J_cand = J_cand[mmr_order]

        soft_mask_c = soft_mask[:, J_cand]
        Pi0_c = Pi0_full[:, J_cand]
        Pi0_c = Pi0_c / (Pi0_c.sum(dim=1, keepdim=True) + 1e-8)
        row_mask = soft_mask_c > threshold

        return J_cand, soft_mask_c, Pi0_c, row_mask
