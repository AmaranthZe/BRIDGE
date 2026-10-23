"""
Complete Cross-Species Gene Mapping Model.

This module wraps all components into a single model:
- HGT on unified heterogeneous graph
- Module fusion attention
- Soft candidate selection
- FGW optimal transport
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hgt import HGT
from .fusion import ModuleFusionAttention
from .soft_candidates import SoftCandidateSelector
from .fgw import FGWSolver, compute_fgw_loss
from .costs import structural_cost

logger = logging.getLogger(__name__)


class CrossSpeciesModel(nn.Module):
    """
    Cross-Species Gene Mapping Model.

    Combines HGT, fusion, candidate selection, and FGW into end-to-end model.
    """

    def __init__(
        self,
        node_counts: Dict[str, int],
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
        edge_bias_kappa: float = 1.0,
        share_go_weights: bool = True,
        share_gene_weights: bool = False,
        fusion_entropy_reg: float = 0.001,
        topc: int = 256,
        soft_temperature: float = 0.1,
        explore_ratio: float = 0.05,
        T_init: float = 0.07,
        max_candidates: int = 1024,
        mmr_lambda: float = 0.25,
        fgw_alpha: float = 0.5,
        fgw_epsilon: float = 0.005,
        fgw_lambda_prior: float = 0.5,
        fgw_lambda_memory: float = 0.3,
        fgw_use_unbalanced: bool = True,
        fgw_tau: float = 0.05,
    ):
        """
        Args:
            node_counts: Dict of node counts per type
            hidden_dim: Hidden dimension for HGT
            out_dim: Output embedding dimension
            num_layers: Number of HGT layers
            heads: Number of attention heads
            dropout: Dropout probability
            edge_bias_kappa: Edge weight bias scale
            share_go_weights: Share weights for GO/Disease nodes
            share_gene_weights: Share weights for gene nodes
            fusion_entropy_reg: Entropy regularization for fusion
            topc: Top-c candidates
            soft_temperature: Soft mask temperature
            explore_ratio: Random exploration ratio
            T_init: Prior coupling temperature
            fgw_*: FGW solver parameters
        """
        super().__init__()

        self.node_counts = node_counts
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Define graph metadata
        self.node_types = ['Human_Gene', 'Mouse_Gene', 'BP', 'MF', 'Disease']
        self.edge_types = [
            ('Human_Gene', 'HG-BP', 'BP'),
            ('Human_Gene', 'HG-MF', 'MF'),
            ('Human_Gene', 'HG-D', 'Disease'),
            ('Mouse_Gene', 'MG-BP', 'BP'),
            ('Mouse_Gene', 'MG-MF', 'MF'),
            ('Mouse_Gene', 'MG-D', 'Disease'),
            # Reverse edges will be added
            ('BP', 'rev_HG-BP', 'Human_Gene'),
            ('MF', 'rev_HG-MF', 'Human_Gene'),
            ('Disease', 'rev_HG-D', 'Human_Gene'),
            ('BP', 'rev_MG-BP', 'Mouse_Gene'),
            ('MF', 'rev_MG-MF', 'Mouse_Gene'),
            ('Disease', 'rev_MG-D', 'Mouse_Gene'),
        ]
        self.metadata = (self.node_types, self.edge_types)

        # HGT backbone
        self.hgt = HGT(
            node_counts=node_counts,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            metadata=self.metadata,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_bias_kappa=edge_bias_kappa,
            share_go_weights=share_go_weights,
            share_gene_weights=share_gene_weights,
        )

        # Module fusion for human and mouse genes
        self.fusion_human = ModuleFusionAttention(
            embed_dim=out_dim,
            entropy_reg=fusion_entropy_reg,
        )
        self.fusion_mouse = ModuleFusionAttention(
            embed_dim=out_dim,
            entropy_reg=fusion_entropy_reg,
        )

        # Candidate selection with MMR diversity (A-fix #3)
        self.candidate_selector = SoftCandidateSelector(
            embed_dim=out_dim,
            topc=topc,
            temperature=soft_temperature,
            explore_ratio=explore_ratio,
            T_init=T_init,
            max_candidates=max_candidates,
            mmr_lambda=mmr_lambda,
        )

        # FGW solver
        self.fgw_solver = FGWSolver(
            alpha=fgw_alpha,
            epsilon=fgw_epsilon,
            lambda_prior=fgw_lambda_prior,
            lambda_memory=fgw_lambda_memory,
            use_unbalanced=fgw_use_unbalanced,
            tau_row=fgw_tau,
            tau_col=fgw_tau,
        )

        # Store parameters for later use
        self.fgw_alpha = fgw_alpha
        self.fgw_epsilon = fgw_epsilon

    def forward(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass to get all node embeddings.

        Args:
            edge_index_dict: Edge indices per type
            edge_attr_dict: Edge weights per type

        Returns:
            Node embeddings per type
        """
        return self.hgt(edge_index_dict, edge_attr_dict)

    def get_gene_embeddings(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get fused gene embeddings for human and mouse.

        Returns:
            (E_human [N_h, D], E_mouse [N_m, D])
        """
        # Get all embeddings
        emb_dict = self.forward(edge_index_dict, edge_attr_dict)

        E_h = emb_dict['Human_Gene']
        E_m = emb_dict['Mouse_Gene']
        E_bp = emb_dict['BP']
        E_mf = emb_dict['MF']
        E_disease = emb_dict['Disease']

        # Apply fusion (simplified - using direct embeddings without edge aggregation)
        E_h_fused, _, entropy_h = self.fusion_human((E_h, E_h, E_h))
        E_m_fused, _, entropy_m = self.fusion_mouse((E_m, E_m, E_m))

        # L2 normalize embeddings for better discriminative power
        E_h_fused = F.normalize(E_h_fused, p=2, dim=-1)
        E_m_fused = F.normalize(E_m_fused, p=2, dim=-1)

        return E_h_fused, E_m_fused, entropy_h + entropy_m

    def compute_batch_coupling(
        self,
        E_source: Tensor,
        E_target: Tensor,
        source_indices: Tensor,
        C_s: Tensor,
        C_t: Tensor,
        M: Tensor,
        L: Optional[Tensor] = None,
        memory_prior: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        """
        Compute coupling for a batch.

        Args:
            E_source: Source embeddings [N_s, D]
            E_target: Target embeddings [N_t, D]
            source_indices: Batch source indices [B]
            C_s: Source structural cost [B, B]
            C_t: Target structural cost (for candidates) [J, J]
            M: Feature cost [B, J]
            L: Supervision cost [B, J]
            memory_prior: Memory bank prior [B, J]

        Returns:
            (J_cand, soft_mask, T, info)
        """
        # Soft candidate selection (returns per-query row_mask)
        J_cand, soft_mask, Pi0, row_mask, _ = self.candidate_selector(
            E_source=E_source,
            E_target=E_target,
            source_indices=source_indices,
        )

        # Apply soft mask to feature cost
        M_masked = M[:, J_cand] * soft_mask

        # Get structural costs for candidates
        C_t_cand = C_t[J_cand][:, J_cand]

        # Get memory prior for candidates
        mem_prior_cand = None
        if memory_prior is not None:
            mem_prior_cand = memory_prior[:, J_cand]
            # Renormalize
            mem_prior_cand = mem_prior_cand / (mem_prior_cand.sum(dim=1, keepdim=True) + 1e-8)

        # Get supervision cost for candidates
        L_cand = None
        if L is not None:
            L_cand = L[:, J_cand]

        # Solve FGW with per-query row mask
        T, info = self.fgw_solver.solve(
            C_s=C_s,
            C_t=C_t_cand,
            M=M_masked,
            L=L_cand,
            Pi0=Pi0,
            memory_prior=mem_prior_cand,
            mask=row_mask,
        )

        return J_cand, soft_mask, T, info

    def compute_loss(
        self,
        C_s: Tensor,
        C_t: Tensor,
        M: Tensor,
        T: Tensor,
        Pi0: Optional[Tensor] = None,
        memory_prior: Optional[Tensor] = None,
        entropy_loss: Tensor = None,
    ) -> Dict[str, Tensor]:
        """
        Compute all loss components.

        Returns:
            Dictionary of loss components
        """
        losses = self.fgw_solver.compute_loss(
            C_s=C_s,
            C_t=C_t,
            M=M,
            T=T,
            Pi0=Pi0,
            memory_prior=memory_prior,
        )

        if entropy_loss is not None:
            losses['fusion_entropy'] = entropy_loss
            losses['total'] = losses['total'] + entropy_loss

        return losses


class ModelWrapper:
    """
    Wrapper for model with data loading and preprocessing.
    """

    def __init__(
        self,
        model: CrossSpeciesModel,
        graph_data,  # HeteroData
        indexers: Dict,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.graph_data = graph_data
        self.indexers = indexers
        self.device = device

        # Move graph data to device
        self._prepare_graph_data()

    def _prepare_graph_data(self):
        """Prepare graph data for model."""
        self.edge_index_dict = {}
        self.edge_attr_dict = {}

        for edge_type in self.graph_data.edge_types:
            edge_data = self.graph_data[edge_type]
            self.edge_index_dict[edge_type] = edge_data.edge_index.to(self.device)
            if hasattr(edge_data, 'edge_attr') and edge_data.edge_attr is not None:
                self.edge_attr_dict[edge_type] = edge_data.edge_attr.to(self.device)

    def get_embeddings(self) -> Tuple[Tensor, Tensor]:
        """Get gene embeddings."""
        self.model.eval()
        with torch.no_grad():
            E_h, E_m, _ = self.model.get_gene_embeddings(
                self.edge_index_dict,
                self.edge_attr_dict,
            )
        return E_h, E_m

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()


def create_model(config: Dict, node_counts: Dict[str, int]) -> CrossSpeciesModel:
    """
    Create model from configuration.

    Args:
        config: Configuration dictionary
        node_counts: Node counts per type

    Returns:
        CrossSpeciesModel instance
    """
    model_cfg = config['model']
    fgw_cfg = config['fgw']
    cand_cfg = config['candidates']

    return CrossSpeciesModel(
        node_counts=node_counts,
        hidden_dim=model_cfg['d'],
        out_dim=model_cfg['d'],
        num_layers=model_cfg['layers'],
        heads=model_cfg['heads'],
        dropout=model_cfg['dropout'],
        edge_bias_kappa=model_cfg['edge_bias_kappa'],
        share_go_weights=model_cfg['share_go_weights'],
        share_gene_weights=model_cfg['share_gene_weights'],
        fusion_entropy_reg=model_cfg['fusion_entropy_reg'],
        topc=cand_cfg['topc'],
        soft_temperature=cand_cfg['soft_temperature'],
        explore_ratio=cand_cfg['explore_ratio'],
        T_init=cand_cfg['T_init'],
        max_candidates=cand_cfg.get('max_candidates', 1024),
        mmr_lambda=cand_cfg.get('mmr_lambda', 0.25),
        fgw_alpha=fgw_cfg['alpha'],
        fgw_epsilon=fgw_cfg['epsilon'],
        fgw_lambda_prior=fgw_cfg['lambda_prior'],
        fgw_lambda_memory=fgw_cfg['lambda_memory'],
        fgw_use_unbalanced=fgw_cfg['use_unbalanced'],
        fgw_tau=fgw_cfg['tau_row'],
    )
