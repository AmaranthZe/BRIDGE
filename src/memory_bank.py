"""
Memory Bank for Batch Consistency.

This module implements an EMA (Exponential Moving Average) memory bank
that maintains global coupling consistency across batches.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class MemoryBank:
    """
    EMA Memory Bank for coupling matrix consistency.

    Stores running average of coupling values to ensure batch-wise FGW
    produces globally consistent mappings.
    """

    def __init__(
        self,
        N_source: int,
        N_target: int,
        momentum: float = 0.9,
        device: str = 'cpu',
        sparse: bool = True,
        max_entries_per_row: int = 256,
    ):
        """
        Args:
            N_source: Number of source genes (e.g., mouse)
            N_target: Number of target genes (e.g., human)
            momentum: EMA momentum (higher = slower update)
            device: Device to store memory
            sparse: Whether to use sparse storage
            max_entries_per_row: Max non-zero entries per row for sparse
        """
        self.N_source = N_source
        self.N_target = N_target
        self.momentum = momentum
        self.device = device
        self.sparse = sparse
        self.max_entries_per_row = max_entries_per_row

        # Initialize storage
        if sparse:
            # Store as list of (indices, values) per source gene
            self.bank_indices = [None] * N_source
            self.bank_values = [None] * N_source
        else:
            # Dense storage (memory-intensive for large graphs)
            self.bank = torch.zeros(N_source, N_target, device=device)

        # Track update counts per source gene
        self.counts = torch.zeros(N_source, device=device)

    def update(
        self,
        I_batch: Tensor,
        J_cand: Tensor,
        T_batch: Tensor,
    ):
        """
        Update memory bank with batch coupling values using EMA.

        Args:
            I_batch: Source indices in batch [B]
            J_cand: Candidate target indices [num_candidates]
            T_batch: Coupling values [B, num_candidates]
        """
        beta = self.momentum
        T_batch = T_batch.detach()

        if self.sparse:
            self._update_sparse(I_batch, J_cand, T_batch, beta)
        else:
            self._update_dense(I_batch, J_cand, T_batch, beta)

    def _update_sparse(self, I_batch, J_cand, T_batch, beta):
        """Sparse update implementation."""
        for i, src_idx in enumerate(I_batch.tolist()):
            new_values = T_batch[i]  # [num_candidates]

            if self.counts[src_idx] == 0:
                # First update: just store
                self.bank_indices[src_idx] = J_cand.clone()
                self.bank_values[src_idx] = new_values.clone()
            else:
                # EMA update
                old_indices = self.bank_indices[src_idx]
                old_values = self.bank_values[src_idx]

                # Merge old and new indices
                all_indices = torch.cat([old_indices, J_cand])
                unique_indices, inverse = torch.unique(all_indices, return_inverse=True)

                # Create merged values
                merged_values = torch.zeros(len(unique_indices), device=self.device)

                # Add old values (weighted by beta)
                old_inverse = inverse[:len(old_indices)]
                merged_values.scatter_add_(
                    0, old_inverse,
                    old_values * beta
                )

                # Add new values (weighted by 1-beta)
                new_inverse = inverse[len(old_indices):]
                merged_values.scatter_add_(
                    0, new_inverse,
                    new_values * (1 - beta)
                )

                # Keep only top-k by value
                if len(unique_indices) > self.max_entries_per_row:
                    topk_vals, topk_idx = torch.topk(
                        merged_values,
                        self.max_entries_per_row
                    )
                    self.bank_indices[src_idx] = unique_indices[topk_idx]
                    self.bank_values[src_idx] = topk_vals
                else:
                    self.bank_indices[src_idx] = unique_indices
                    self.bank_values[src_idx] = merged_values

            self.counts[src_idx] += 1

    def _update_dense(self, I_batch, J_cand, T_batch, beta):
        """Dense update implementation."""
        for i, src_idx in enumerate(I_batch.tolist()):
            if self.counts[src_idx] == 0:
                self.bank[src_idx, J_cand] = T_batch[i]
            else:
                old = self.bank[src_idx, J_cand]
                new = T_batch[i]
                self.bank[src_idx, J_cand] = beta * old + (1 - beta) * new

            self.counts[src_idx] += 1

    def get_prior(
        self,
        I_batch: Tensor,
        J_cand: Tensor,
    ) -> Tensor:
        """
        Get memory-based prior for consistency regularization.

        Args:
            I_batch: Source indices [B]
            J_cand: Candidate target indices [num_candidates]

        Returns:
            Prior coupling [B, num_candidates]
        """
        B = len(I_batch)
        num_cand = len(J_cand)

        prior = torch.zeros(B, num_cand, device=self.device)

        if self.sparse:
            for i, src_idx in enumerate(I_batch.tolist()):
                if self.bank_indices[src_idx] is not None:
                    # Find intersection of stored indices and J_cand
                    stored_idx = self.bank_indices[src_idx]
                    stored_val = self.bank_values[src_idx]

                    # Create lookup
                    j_cand_set = set(J_cand.tolist())
                    for j, idx in enumerate(stored_idx.tolist()):
                        if idx in j_cand_set:
                            cand_pos = (J_cand == idx).nonzero(as_tuple=True)[0]
                            if len(cand_pos) > 0:
                                prior[i, cand_pos[0]] = stored_val[j]
        else:
            for i, src_idx in enumerate(I_batch.tolist()):
                prior[i] = self.bank[src_idx, J_cand]

        # Normalize
        prior_sum = prior.sum(dim=1, keepdim=True)
        prior = torch.where(
            prior_sum > 0,
            prior / prior_sum,
            torch.ones_like(prior) / num_cand
        )

        return prior

    def global_refresh(
        self,
        T_global: Tensor,
        weight: float = 0.5,
    ):
        """
        Periodic global refresh with full coupling matrix.

        Args:
            T_global: Global coupling matrix [N_source, N_target]
            weight: Weight for old values (1-weight for new)
        """
        if self.sparse:
            # For sparse storage, extract top-k per row from T_global
            for i in range(self.N_source):
                row = T_global[i]
                topk_vals, topk_idx = torch.topk(row, self.max_entries_per_row)

                if self.bank_indices[i] is None:
                    self.bank_indices[i] = topk_idx
                    self.bank_values[i] = topk_vals
                else:
                    # Blend old and new
                    old_idx = self.bank_indices[i]
                    old_val = self.bank_values[i]

                    # Merge
                    all_idx = torch.cat([old_idx, topk_idx])
                    unique_idx, inverse = torch.unique(all_idx, return_inverse=True)

                    merged = torch.zeros(len(unique_idx), device=self.device)
                    merged.scatter_add_(0, inverse[:len(old_idx)], old_val * weight)
                    merged.scatter_add_(0, inverse[len(old_idx):], topk_vals * (1 - weight))

                    # Keep top-k
                    if len(unique_idx) > self.max_entries_per_row:
                        tk_vals, tk_idx = torch.topk(merged, self.max_entries_per_row)
                        self.bank_indices[i] = unique_idx[tk_idx]
                        self.bank_values[i] = tk_vals
                    else:
                        self.bank_indices[i] = unique_idx
                        self.bank_values[i] = merged
        else:
            self.bank = weight * self.bank + (1 - weight) * T_global

        logger.info("Performed global memory refresh")

    def get_full_matrix(self) -> Tensor:
        """Convert sparse storage to dense matrix for analysis."""
        if not self.sparse:
            return self.bank

        matrix = torch.zeros(self.N_source, self.N_target, device=self.device)
        for i in range(self.N_source):
            if self.bank_indices[i] is not None:
                matrix[i, self.bank_indices[i]] = self.bank_values[i]
        return matrix

    def get_stats(self) -> dict:
        """Get memory bank statistics."""
        updated = (self.counts > 0).sum().item()
        avg_count = self.counts[self.counts > 0].mean().item() if updated > 0 else 0

        if self.sparse:
            nnz = sum(
                len(v) if v is not None else 0
                for v in self.bank_values
            )
            density = nnz / (self.N_source * self.N_target)
        else:
            nnz = (self.bank > 0).sum().item()
            density = nnz / (self.N_source * self.N_target)

        return {
            'updated_rows': updated,
            'total_rows': self.N_source,
            'avg_update_count': avg_count,
            'nnz': nnz,
            'density': density,
        }
