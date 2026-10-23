"""Heterogeneous Graph Transformer using PyG's HGTConv."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


def compute_per_modality_gene_features(sparse_matrices, embed_dim):
    """Compute per-modality SVD features for genes.

    Instead of the unified SVD (which mixes BP/MF/Disease into one 128-dim space),
    this computes separate SVD for each modality. Each gene gets three independent
    feature vectors capturing its BP profile, MF profile, and Disease profile.

    Used for:
    - Term initialization: BP terms get BP-specific init, MF terms get MF-specific, etc.
    - Fusion residual: replaces the all-purpose gene_emb with modality-specific features.
    """
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD

    features = {}
    for mod, h_key, m_key in [
        ("BP", "W_hB", "W_mB"),
        ("MF", "W_hF", "W_mF"),
        ("Disease", "W_hD", "W_mD"),
    ]:
        W_h = sparse_matrices[h_key]
        W_m = sparse_matrices[m_key]
        X = sp.vstack([W_h, W_m], format="csr")
        actual_dim = min(embed_dim, X.shape[1] - 1, X.shape[0] - 1)
        svd = TruncatedSVD(n_components=actual_dim, random_state=3407)
        Z_all = svd.fit_transform(X).astype(np.float32)

        N_h = W_h.shape[0]
        Z_h_mod = Z_all[:N_h]
        Z_m_mod = Z_all[N_h:]

        # L2 normalize
        for Z in (Z_h_mod, Z_m_mod):
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            Z /= norms

        # Pad if modality has fewer terms than embed_dim
        if actual_dim < embed_dim:
            Z_h_mod = np.pad(Z_h_mod, ((0, 0), (0, embed_dim - actual_dim)))
            Z_m_mod = np.pad(Z_m_mod, ((0, 0), (0, embed_dim - actual_dim)))

        features[f"Z_h_{mod}"] = Z_h_mod
        features[f"Z_m_{mod}"] = Z_m_mod

        logger.info(
            f"  Per-modality SVD ({mod}): "
            f"explained_var={svd.explained_variance_ratio_.sum():.4f}, dim={actual_dim}"
        )

    return features


class HGT(nn.Module):
    """HGT with learnable node embeddings, skip gate, and gradient checkpointing.

    The skip gate blends initial SVD embeddings with HGT message-passing output,
    preventing over-smoothing on dense bipartite graphs. No out_proj is used —
    hidden_channels == out_channels, and a 128→128 linear layer was found to
    collapse embedding diversity (random cosine ~0.998 after training).

    Edge weight support (Issue #4 fix): PyG's HGTConv does not accept per-edge
    weights. We add a parallel weighted neighbor aggregation step in each layer,
    blended with HGTConv output via edge_weight_blend. This injects TF-IDF edge
    weight information that was previously ignored.
    """

    def __init__(
        self,
        node_counts: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        edge_weight_blend: float = 0.1,
    ):
        super().__init__()
        self.node_types = metadata[0]
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_weight_blend = edge_weight_blend

        # Learnable node embeddings
        self.embeddings = nn.ModuleDict({
            nt: nn.Embedding(node_counts[nt], hidden_channels)
            for nt in self.node_types
        })

        from torch_geometric.nn import HGTConv
        self.layers = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=heads)
            for _ in range(num_layers)
        ])

        # Per-layer linear transforms for weighted aggregation (Issue #4)
        # Each layer gets its own projection so weighted messages are transformed
        self.weighted_projs = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels, bias=False)
            for _ in range(num_layers)
        ])

        # NOTE: out_proj removed — it was a 128→128 linear layer that caused
        # embedding collapse by projecting diverse skip-gated embeddings into
        # a narrow subspace. Fusion module projections handle any needed transform.

        # Per-node-type residual scales: HGT output is an ADDITIVE refinement on SVD.
        # x_out = SVD + sigmoid(scale) * (HGT - SVD)
        # Initialized at -2.2 → sigmoid(-2.2) ≈ 0.10, so HGT starts as 10% refinement.
        # Model learns how much to trust HGT per node type.
        # This replaces the old skip_gate which used alpha*SVD + (1-alpha)*HGT,
        # where even 20% HGT could destroy SVD cross-species signal.
        self.residual_scales = nn.ParameterDict({
            nt: nn.Parameter(torch.tensor(-2.2))
            for nt in self.node_types
        })

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

    def init_from_svd(
        self,
        Z_h: "np.ndarray",
        Z_m: "np.ndarray",
        sparse_matrices: Dict,
    ):
        """Initialize node embeddings from SVD features.

        Gene nodes get their unified SVD embeddings directly.
        Term nodes get per-modality initialization: each term type (BP, MF, Disease)
        is initialized from its OWN modality's SVD features, not the unified SVD.
        This makes term embeddings carry modality-specific information, which is
        critical for the fusion module to produce genuinely different per-module signals.

        Also registers per-modality gene feature buffers for fusion.
        """
        d = self.hidden_channels

        # Gene embeddings from unified SVD (unchanged)
        with torch.no_grad():
            self.embeddings["Human_Gene"].weight.copy_(
                torch.tensor(Z_h, dtype=torch.float32))
            self.embeddings["Mouse_Gene"].weight.copy_(
                torch.tensor(Z_m, dtype=torch.float32))

        # Compute per-modality gene features
        logger.info("Computing per-modality SVD features...")
        mod_features = compute_per_modality_gene_features(sparse_matrices, d)

        # Initialize term embeddings from per-modality gene features (NOT unified SVD)
        for nt, W_h_key, W_m_key in [
            ("BP", "W_hB", "W_mB"),
            ("MF", "W_hF", "W_mF"),
            ("Disease", "W_hD", "W_mD"),
        ]:
            W_h = sparse_matrices[W_h_key]
            W_m = sparse_matrices[W_m_key]

            W_h_bin = (W_h > 0).astype(np.float32)
            W_m_bin = (W_m > 0).astype(np.float32)

            Z_h_mod = mod_features[f"Z_h_{nt}"]
            Z_m_mod = mod_features[f"Z_m_{nt}"]

            # Term embedding = mean of connected genes' PER-MODALITY features
            term_emb = W_h_bin.T @ Z_h_mod + W_m_bin.T @ Z_m_mod

            degrees = np.array(W_h_bin.sum(axis=0) + W_m_bin.sum(axis=0)).squeeze()
            degrees = np.maximum(degrees, 1)
            term_emb = term_emb / degrees[:, None]

            norms = np.linalg.norm(term_emb, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            term_emb = term_emb / norms

            gene_scale = np.linalg.norm(Z_h, axis=1).mean()
            term_emb = term_emb * gene_scale

            with torch.no_grad():
                self.embeddings[nt].weight.copy_(
                    torch.tensor(term_emb, dtype=torch.float32))

        # Store per-modality gene features as buffers (used by fusion module)
        for key, val in mod_features.items():
            self.register_buffer(key, torch.tensor(val, dtype=torch.float32))

        logger.info("Initialized HGT: gene=unified SVD, terms=per-modality SVD, "
                     "per-modality gene features stored as buffers")

    def forward(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Dict[str, Tensor]:
        x_dict = {nt: self.embeddings[nt].weight for nt in self.node_types}
        x0 = x_dict  # Save initial (SVD) embeddings — no clone needed, just ref

        for i, layer in enumerate(self.layers):
            w_proj = self.weighted_projs[i]
            if self.training:
                x_dict = checkpoint(
                    self._layer_forward, x_dict, edge_index_dict,
                    edge_attr_dict, layer, w_proj,
                    use_reentrant=False,
                )
            else:
                x_dict = self._layer_forward(
                    x_dict, edge_index_dict, edge_attr_dict, layer, w_proj)

        # Additive residual: SVD + scale * (HGT - SVD)
        # When scale→0: pure SVD. When scale→1: pure HGT.
        # Initialized near 0 so HGT must earn its contribution.
        x_out = {
            nt: x0[nt] + torch.sigmoid(self.residual_scales[nt]) * (x_dict[nt] - x0[nt])
            for nt in self.node_types
        }

        return x_out

    def _layer_forward(self, x_dict, edge_index_dict, edge_attr_dict, layer, w_proj):
        """Single HGTConv layer + edge-weight-aware aggregation.

        Issue #4 fix: PyG's HGTConv ignores per-edge weights. We add a parallel
        weighted neighbor aggregation and blend it with HGTConv output.
        This injects TF-IDF edge weight information into the message passing.
        """
        # Standard HGT (learned attention, no per-edge weights)
        x_hgt = layer(x_dict, edge_index_dict)
        x_hgt = {k: self.dropout(F.relu(v)) for k, v in x_hgt.items()}

        # Weighted neighbor aggregation using actual edge weights
        blend = self.edge_weight_blend
        if blend > 0 and edge_attr_dict:
            x_w = self._weighted_neighbor_agg(x_dict, edge_index_dict,
                                               edge_attr_dict, w_proj)
            x_hgt = {
                k: (1 - blend) * x_hgt[k] + blend * x_w.get(k, x_hgt[k])
                for k in x_hgt
            }

        return x_hgt

    def _weighted_neighbor_agg(self, x_dict, edge_index_dict, edge_attr_dict,
                                w_proj):
        """Weighted mean of neighbor features using TF-IDF edge weights.

        For each edge type, source node features are scaled by edge weight,
        aggregated to destination nodes, and normalized by weight sum.
        The result is transformed by a learnable projection.
        """
        D = self.hidden_channels
        # Accumulate weighted messages per node type
        agg = {nt: torch.zeros_like(x_dict[nt]) for nt in self.node_types}
        wdeg = {nt: torch.zeros(x_dict[nt].size(0), 1, device=x_dict[nt].device)
                for nt in self.node_types}

        for et, ei in edge_index_dict.items():
            ea = edge_attr_dict.get(et)
            if ea is None:
                continue
            src_type, _, dst_type = et
            src_idx, dst_idx = ei[0], ei[1]

            # Weighted message: source features * edge weight
            msg = x_dict[src_type][src_idx] * ea.unsqueeze(-1)
            agg[dst_type].scatter_add_(
                0, dst_idx.unsqueeze(-1).expand(-1, D), msg)
            wdeg[dst_type].scatter_add_(0, dst_idx.unsqueeze(-1), ea.unsqueeze(-1))

        # Normalize and transform
        out = {}
        for nt in self.node_types:
            normed = agg[nt] / (wdeg[nt] + 1e-8)
            out[nt] = F.relu(w_proj(normed))

        return out
