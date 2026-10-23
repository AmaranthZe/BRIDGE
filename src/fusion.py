"""
Module Fusion Attention.

This module combines embeddings from three functional modules (BP, MF, Disease)
into a single gene embedding using learned attention weights.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ModuleFusionAttention(nn.Module):
    """
    Fuse BP, MF, and Disease module embeddings via attention.

    For each gene, computes:
        beta_k = softmax(q^T tanh(W_k z^(k)))
        E = sum_k beta_k z^(k)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        entropy_reg: float = 0.001,
        num_modules: int = 3,  # BP, MF, Disease
    ):
        """
        Args:
            embed_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for attention (default: embed_dim)
            entropy_reg: Entropy regularization coefficient
            num_modules: Number of modules to fuse
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.entropy_reg = entropy_reg
        self.num_modules = num_modules

        # Projection layers for each module
        self.module_proj = nn.ModuleList([
            nn.Linear(embed_dim, self.hidden_dim)
            for _ in range(num_modules)
        ])

        # Query vector (learnable)
        self.query = nn.Parameter(torch.Tensor(self.hidden_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in self.module_proj:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.normal_(self.query, std=0.01)

    def forward(
        self,
        module_embeddings: Tuple[Tensor, Tensor, Tensor],
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Fuse module embeddings.

        Args:
            module_embeddings: Tuple of (z_bp, z_mf, z_disease), each [N, D]
            return_weights: Whether to return attention weights

        Returns:
            Tuple of:
                - Fused embedding [N, D]
                - Attention weights [N, 3] (if return_weights)
                - Entropy regularization loss (scalar)
        """
        z_list = list(module_embeddings)
        N = z_list[0].shape[0]

        # Stack module embeddings: [N, 3, D]
        z_stack = torch.stack(z_list, dim=1)

        # Compute attention scores
        attn_scores = []
        for k, proj in enumerate(self.module_proj):
            # [N, hidden_dim]
            h = torch.tanh(proj(z_list[k]))
            # [N]
            score = (h * self.query.unsqueeze(0)).sum(dim=-1)
            attn_scores.append(score)

        # [N, 3]
        attn_scores = torch.stack(attn_scores, dim=1)

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # [N, 3]

        # Weighted sum
        # [N, 3, 1] * [N, 3, D] -> [N, 3, D] -> sum -> [N, D]
        fused = (attn_weights.unsqueeze(-1) * z_stack).sum(dim=1)

        # Compute entropy regularization loss
        # Encourage diverse attention (high entropy)
        entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
        entropy_loss = -self.entropy_reg * entropy  # Negative because we want to maximize entropy

        if return_weights:
            return fused, attn_weights, entropy_loss
        else:
            return fused, None, entropy_loss


class GeneModuleFusion(nn.Module):
    """
    Extract per-module embeddings for genes and fuse them.

    This module handles the extraction of BP, MF, Disease contributions
    for each gene from the unified graph embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        entropy_reg: float = 0.001,
    ):
        super().__init__()

        self.fusion = ModuleFusionAttention(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            entropy_reg=entropy_reg,
        )

        # Projection to extract module-specific views
        self.bp_proj = nn.Linear(embed_dim, embed_dim)
        self.mf_proj = nn.Linear(embed_dim, embed_dim)
        self.disease_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        gene_embedding: Tensor,
        bp_embedding: Optional[Tensor] = None,
        mf_embedding: Optional[Tensor] = None,
        disease_embedding: Optional[Tensor] = None,
        edge_index_bp: Optional[Tensor] = None,
        edge_index_mf: Optional[Tensor] = None,
        edge_index_disease: Optional[Tensor] = None,
        edge_attr_bp: Optional[Tensor] = None,
        edge_attr_mf: Optional[Tensor] = None,
        edge_attr_disease: Optional[Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Extract and fuse module embeddings for genes.

        If module embeddings and edges are provided, aggregate from neighbors.
        Otherwise, use learned projections of gene embeddings.

        Args:
            gene_embedding: Gene embeddings [N_genes, D]
            bp_embedding: BP node embeddings [N_bp, D]
            mf_embedding: MF node embeddings [N_mf, D]
            disease_embedding: Disease node embeddings [N_disease, D]
            edge_index_*: Edge indices [2, E] from genes to modules
            edge_attr_*: Edge weights [E]
            return_weights: Whether to return attention weights

        Returns:
            Tuple of (fused_embedding, attention_weights, entropy_loss)
        """
        N = gene_embedding.shape[0]
        D = gene_embedding.shape[1]
        device = gene_embedding.device

        # Initialize module embeddings as zeros
        z_bp = torch.zeros(N, D, device=device)
        z_mf = torch.zeros(N, D, device=device)
        z_disease = torch.zeros(N, D, device=device)

        # Aggregate from BP neighbors
        if bp_embedding is not None and edge_index_bp is not None and edge_index_bp.shape[1] > 0:
            src_idx, dst_idx = edge_index_bp[0], edge_index_bp[1]
            msg = bp_embedding[dst_idx]
            if edge_attr_bp is not None:
                msg = msg * edge_attr_bp.unsqueeze(-1)
            z_bp.scatter_add_(0, src_idx.unsqueeze(-1).expand(-1, D), msg)
            # Normalize by degree
            deg = torch.zeros(N, device=device)
            deg.scatter_add_(0, src_idx, torch.ones_like(src_idx, dtype=torch.float))
            z_bp = z_bp / (deg.unsqueeze(-1) + 1e-8)

        # Aggregate from MF neighbors
        if mf_embedding is not None and edge_index_mf is not None and edge_index_mf.shape[1] > 0:
            src_idx, dst_idx = edge_index_mf[0], edge_index_mf[1]
            msg = mf_embedding[dst_idx]
            if edge_attr_mf is not None:
                msg = msg * edge_attr_mf.unsqueeze(-1)
            z_mf.scatter_add_(0, src_idx.unsqueeze(-1).expand(-1, D), msg)
            deg = torch.zeros(N, device=device)
            deg.scatter_add_(0, src_idx, torch.ones_like(src_idx, dtype=torch.float))
            z_mf = z_mf / (deg.unsqueeze(-1) + 1e-8)

        # Aggregate from Disease neighbors
        if disease_embedding is not None and edge_index_disease is not None and edge_index_disease.shape[1] > 0:
            src_idx, dst_idx = edge_index_disease[0], edge_index_disease[1]
            msg = disease_embedding[dst_idx]
            if edge_attr_disease is not None:
                msg = msg * edge_attr_disease.unsqueeze(-1)
            z_disease.scatter_add_(0, src_idx.unsqueeze(-1).expand(-1, D), msg)
            deg = torch.zeros(N, device=device)
            deg.scatter_add_(0, src_idx, torch.ones_like(src_idx, dtype=torch.float))
            z_disease = z_disease / (deg.unsqueeze(-1) + 1e-8)

        # Apply projections
        z_bp = self.bp_proj(z_bp + gene_embedding)
        z_mf = self.mf_proj(z_mf + gene_embedding)
        z_disease = self.disease_proj(z_disease + gene_embedding)

        # Fuse
        return self.fusion((z_bp, z_mf, z_disease), return_weights=return_weights)
