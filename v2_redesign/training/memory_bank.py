"""EMA Memory Bank for batch coupling consistency.

FIX from v1: get_prior uses vectorized operations instead of slow Python loops.
"""

import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class MemoryBank:
    """Sparse EMA memory bank: stores top-k coupling values per source gene."""

    def __init__(
        self,
        N_source: int,
        N_target: int,
        momentum: float = 0.9,
        device: str = "cpu",
        max_entries_per_row: int = 256,
    ):
        self.N_source = N_source
        self.N_target = N_target
        self.momentum = momentum
        self.device = device
        self.max_k = max_entries_per_row

        # Padded storage: [N_source, max_k] for indices and values
        self.bank_idx = torch.full((N_source, max_entries_per_row), -1,
                                   dtype=torch.long, device=device)
        self.bank_val = torch.zeros(N_source, max_entries_per_row,
                                    dtype=torch.float32, device=device)
        self.counts = torch.zeros(N_source, dtype=torch.long, device=device)

    def update(self, I_batch: Tensor, J_cand: Tensor, T_batch: Tensor):
        """EMA update for batch coupling values."""
        T_batch = T_batch.detach()
        beta = self.momentum

        for i, src in enumerate(I_batch.tolist()):
            new_idx = J_cand
            new_val = T_batch[i]

            if self.counts[src] == 0:
                # First update
                k = min(len(new_idx), self.max_k)
                if len(new_idx) > self.max_k:
                    topk_val, topk_pos = torch.topk(new_val, k)
                    self.bank_idx[src, :k] = new_idx[topk_pos]
                    self.bank_val[src, :k] = topk_val
                else:
                    self.bank_idx[src, :k] = new_idx
                    self.bank_val[src, :k] = new_val
            else:
                # Merge old + new
                old_valid = self.bank_idx[src] >= 0
                old_idx = self.bank_idx[src, old_valid]
                old_val = self.bank_val[src, old_valid]

                all_idx = torch.cat([old_idx, new_idx])
                all_val = torch.cat([old_val * beta, new_val * (1 - beta)])

                unique_idx, inverse = torch.unique(all_idx, return_inverse=True)
                merged = torch.zeros(len(unique_idx), device=self.device)
                merged.scatter_add_(0, inverse, all_val)

                k = min(len(unique_idx), self.max_k)
                topk_val, topk_pos = torch.topk(merged, k)

                self.bank_idx[src] = -1
                self.bank_val[src] = 0
                self.bank_idx[src, :k] = unique_idx[topk_pos]
                self.bank_val[src, :k] = topk_val

            self.counts[src] += 1

    def get_prior(self, I_batch: Tensor, J_cand: Tensor) -> Tensor:
        """
        Get memory-based prior for consistency regularization.

        VECTORIZED: Uses tensor ops instead of Python loops over stored entries.
        """
        B = len(I_batch)
        num_cand = len(J_cand)
        prior = torch.zeros(B, num_cand, device=self.device)

        # Build a lookup from target index -> position in J_cand
        # Use a dense lookup (fast if N_target is manageable)
        cand_lookup = torch.full((self.N_target,), -1, dtype=torch.long, device=self.device)
        cand_lookup[J_cand] = torch.arange(num_cand, device=self.device)

        for i, src in enumerate(I_batch.tolist()):
            if self.counts[src] == 0:
                continue
            valid = self.bank_idx[src] >= 0
            stored_idx = self.bank_idx[src, valid]
            stored_val = self.bank_val[src, valid]

            # Map stored indices to candidate positions
            positions = cand_lookup[stored_idx]
            hit = positions >= 0
            if hit.any():
                prior[i].scatter_add_(0, positions[hit], stored_val[hit])

        # Normalize rows
        row_sum = prior.sum(dim=1, keepdim=True)
        prior = torch.where(
            row_sum > 0,
            prior / row_sum,
            torch.ones_like(prior) / num_cand,
        )
        return prior

    def global_refresh(self, model, E_source, E_target, chunk_size: int = 1000):
        """
        Periodic global refresh using co-attention scores.

        FIX from v1: Properly merges old and new indices instead of replacing.
        FIX: Handle device mismatch — model's W_Q/W_K may be on GPU while
        E_source/E_target are on CPU. Project on model's device, store on self.device.
        """
        N_s = E_source.shape[0]
        model_device = next(model.candidate_selector.parameters()).device

        with torch.no_grad():
            K = model.candidate_selector.W_K(E_target.to(model_device))

            for start in range(0, N_s, chunk_size):
                end = min(start + chunk_size, N_s)
                Q = model.candidate_selector.W_Q(E_source[start:end].to(model_device))
                A = Q @ K.t()
                T_chunk = torch.softmax(A, dim=1)

                topk_val, topk_idx = torch.topk(T_chunk, min(self.max_k, T_chunk.shape[1]), dim=1)
                topk_val = topk_val.to(self.device)
                topk_idx = topk_idx.to(self.device)

                for i in range(end - start):
                    src = start + i
                    new_idx = topk_idx[i]
                    new_val = topk_val[i]

                    if self.counts[src] == 0:
                        k = min(len(new_idx), self.max_k)
                        self.bank_idx[src, :k] = new_idx[:k]
                        self.bank_val[src, :k] = new_val[:k]
                    else:
                        # Merge with blending
                        old_valid = self.bank_idx[src] >= 0
                        old_idx = self.bank_idx[src, old_valid]
                        old_val = self.bank_val[src, old_valid]

                        all_idx = torch.cat([old_idx, new_idx])
                        all_val = torch.cat([old_val * 0.5, new_val * 0.5])

                        unique_idx, inverse = torch.unique(all_idx, return_inverse=True)
                        merged = torch.zeros(len(unique_idx), device=self.device)
                        merged.scatter_add_(0, inverse, all_val)

                        k = min(len(unique_idx), self.max_k)
                        tk_val, tk_pos = torch.topk(merged, k)

                        self.bank_idx[src] = -1
                        self.bank_val[src] = 0
                        self.bank_idx[src, :k] = unique_idx[tk_pos]
                        self.bank_val[src, :k] = tk_val

                    self.counts[src] = max(self.counts[src].item(), 1)

        logger.info("Global memory refresh complete")

    def get_stats(self) -> dict:
        updated = (self.counts > 0).sum().item()
        nnz = (self.bank_idx >= 0).sum().item()
        return {
            "updated_rows": updated,
            "total_rows": self.N_source,
            "avg_update_count": self.counts[self.counts > 0].float().mean().item() if updated else 0,
            "nnz": nnz,
            "density": nnz / (self.N_source * self.N_target),
        }
