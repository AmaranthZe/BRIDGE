"""Module Fusion Attention.

Combines BP, MF, Disease module embeddings into a single gene embedding.

CRITICAL FIXES:
- v1 model.py line 191 passed (E_h, E_h, E_h) — same embedding 3x, making fusion useless.
- v2 uses GeneModuleFusion which aggregates from actual BP/MF/Disease neighbor nodes.
- v2.1 (Issue #1 fix): term embeddings now use per-modality SVD init (not unified),
  and fusion residual uses per-modality gene features instead of unified gene_emb.
  This makes z_bp, z_mf, z_dis genuinely different — critical for fusion attention
  to produce discriminative, modality-aware embeddings.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ModuleFusionAttention(nn.Module):
    """
    Fuse BP, MF, Disease embeddings via learned attention.

    beta_k = softmax(q^T tanh(W_k z^(k)))
    E = sum_k beta_k * z^(k)
    """

    def __init__(self, embed_dim: int, entropy_reg: float = 0.001, num_modules: int = 3):
        super().__init__()
        self.module_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modules)
        ])
        self.query = nn.Parameter(torch.randn(embed_dim) * 0.01)
        self.entropy_reg = entropy_reg

    def forward(
        self,
        module_embeddings: Tuple[Tensor, Tensor, Tensor],
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Args:
            module_embeddings: (z_bp, z_mf, z_disease), each [N, D]

        Returns:
            (fused [N, D], attn_weights [N, 3] or None, entropy_loss scalar)
        """
        z_list = list(module_embeddings)
        z_stack = torch.stack(z_list, dim=1)  # [N, 3, D]

        scores = torch.stack([
            (torch.tanh(proj(z)) * self.query.unsqueeze(0)).sum(-1)
            for proj, z in zip(self.module_proj, z_list)
        ], dim=1)  # [N, 3]

        weights = F.softmax(scores, dim=1)  # [N, 3]
        fused = (weights.unsqueeze(-1) * z_stack).sum(dim=1)  # [N, D]

        # Entropy reg: encourage diverse attention (maximize entropy)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
        entropy_loss = -self.entropy_reg * entropy

        return fused, (weights if return_weights else None), entropy_loss


class GeneModuleFusion(nn.Module):
    """
    Extract per-module embeddings for genes by aggregating from BP/MF/Disease
    neighbor nodes, then fuse via attention or concatenation.

    This is the CORRECT fusion that uses genuinely different per-module info.
    """

    def __init__(self, embed_dim: int, entropy_reg: float = 0.001, mode: str = "attention"):
        super().__init__()
        self.mode = mode
        self.fusion = ModuleFusionAttention(embed_dim, entropy_reg)
        self.bp_proj = nn.Linear(embed_dim, embed_dim)
        self.mf_proj = nn.Linear(embed_dim, embed_dim)
        self.disease_proj = nn.Linear(embed_dim, embed_dim)
        if mode == "concat":
            self.concat_proj = nn.Linear(embed_dim * 3, embed_dim)

    def _aggregate(
        self,
        gene_emb: Tensor,
        term_emb: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        N: int,
        D: int,
    ) -> Tensor:
        """Weighted mean aggregation from term neighbors."""
        z = torch.zeros(N, D, device=gene_emb.device)
        if edge_index is None or edge_index.shape[1] == 0:
            return z

        src, dst = edge_index[0], edge_index[1]
        msg = term_emb[dst]
        if edge_attr is not None:
            msg = msg * edge_attr.unsqueeze(-1)

        z.scatter_add_(0, src.unsqueeze(-1).expand(-1, D), msg)
        deg = torch.zeros(N, device=gene_emb.device)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        return z / (deg.unsqueeze(-1) + 1e-8)

    def forward(
        self,
        gene_emb: Tensor,
        bp_emb: Tensor,
        mf_emb: Tensor,
        disease_emb: Tensor,
        ei_bp: Optional[Tensor] = None,
        ei_mf: Optional[Tensor] = None,
        ei_disease: Optional[Tensor] = None,
        ea_bp: Optional[Tensor] = None,
        ea_mf: Optional[Tensor] = None,
        ea_disease: Optional[Tensor] = None,
        bp_gene_feat: Optional[Tensor] = None,
        mf_gene_feat: Optional[Tensor] = None,
        dis_gene_feat: Optional[Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        N, D = gene_emb.shape

        # Use per-modality gene features as residual when available.
        # This replaces the unified gene_emb that previously drowned out
        # module-specific signals (all three z_* collapsed to ~proj(gene_emb)).
        bp_res = bp_gene_feat if bp_gene_feat is not None else gene_emb
        mf_res = mf_gene_feat if mf_gene_feat is not None else gene_emb
        dis_res = dis_gene_feat if dis_gene_feat is not None else gene_emb

        z_bp = self.bp_proj(self._aggregate(gene_emb, bp_emb, ei_bp, ea_bp, N, D) + bp_res)
        z_mf = self.mf_proj(self._aggregate(gene_emb, mf_emb, ei_mf, ea_mf, N, D) + mf_res)
        z_dis = self.disease_proj(self._aggregate(gene_emb, disease_emb, ei_disease, ea_disease, N, D) + dis_res)

        if self.mode == "concat":
            fused = self.concat_proj(torch.cat([z_bp, z_mf, z_dis], dim=1))
            dummy_entropy = gene_emb.new_tensor(0.0)
            return fused, None, dummy_entropy

        return self.fusion((z_bp, z_mf, z_dis), return_weights=return_weights)
