"""v3 CrossSpeciesModel with 18-edge-type metadata.

The only change from v2 is the METADATA constant: 6 new cross-species edge types.
HGTConv is instantiated with this metadata so it creates attention parameter groups
for the new edge types. Everything else (HGT internals, fusion, candidates) is
imported from v2 unchanged.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v2_redesign.model.hgt import HGT
from v2_redesign.model.fusion import GeneModuleFusion
from v2_redesign.model.candidates import SoftCandidateSelector

logger = logging.getLogger(__name__)

# v3 node types: same as v2
NODE_TYPES = ["Human_Gene", "Mouse_Gene", "BP", "MF", "Disease"]

# v3 edge types: v2's 12 + 6 new cross-species
EDGE_TYPES = [
    # v2 forward edges (gene -> term)
    ("Human_Gene", "HG-BP",  "BP"),
    ("Human_Gene", "HG-MF",  "MF"),
    ("Human_Gene", "HG-D",   "Disease"),
    ("Mouse_Gene", "MG-BP",  "BP"),
    ("Mouse_Gene", "MG-MF",  "MF"),
    ("Mouse_Gene", "MG-D",   "Disease"),
    # v2 reverse edges (term -> gene)
    ("BP",      "rev_HG-BP", "Human_Gene"),
    ("MF",      "rev_HG-MF", "Human_Gene"),
    ("Disease", "rev_HG-D",  "Human_Gene"),
    ("BP",      "rev_MG-BP", "Mouse_Gene"),
    ("MF",      "rev_MG-MF", "Mouse_Gene"),
    ("Disease", "rev_MG-D",  "Mouse_Gene"),
    # NEW v3: cross-species direct edges
    ("Human_Gene", "xBP", "Mouse_Gene"),
    ("Human_Gene", "xMF", "Mouse_Gene"),
    ("Human_Gene", "xD",  "Mouse_Gene"),
    # NEW v3: reverse cross-species
    ("Mouse_Gene", "rev_xBP", "Human_Gene"),
    ("Mouse_Gene", "rev_xMF", "Human_Gene"),
    ("Mouse_Gene", "rev_xD",  "Human_Gene"),
]

METADATA = (NODE_TYPES, EDGE_TYPES)


class CrossSpeciesModelV3(nn.Module):
    """v3 model: identical to v2 but HGT has 18-edge-type metadata.

    The HGT conv layers get 6 extra attention parameter groups (one per new
    cross-species edge type). All other components (fusion, candidates,
    residual_scales, embeddings) are identical to v2.
    """

    def __init__(self, node_counts: Dict[str, int], cfg):
        super().__init__()
        mc = cfg.model
        cc = cfg.candidates

        self.out_dim = mc.hidden_dim

        # HGT backbone with v3 METADATA (18 edge types)
        self.hgt = HGT(
            node_counts=node_counts,
            hidden_channels=mc.hidden_dim,
            out_channels=mc.hidden_dim,
            metadata=METADATA,          # KEY CHANGE: 18 edge types
            num_layers=mc.num_layers,
            heads=mc.heads,
            dropout=mc.dropout,
            edge_weight_blend=mc.edge_weight_blend,
        )

        # Fusion modules (identical to v2)
        fusion_mode = getattr(cfg.train, 'fusion_mode', 'attention')
        self.fusion_human = GeneModuleFusion(
            mc.hidden_dim, mc.fusion_entropy_reg, mode=fusion_mode
        )
        self.fusion_mouse = GeneModuleFusion(
            mc.hidden_dim, mc.fusion_entropy_reg, mode=fusion_mode
        )

        # Candidate selector (identical to v2)
        self.candidate_selector = SoftCandidateSelector(
            embed_dim=mc.hidden_dim,
            topc=cc.topc,
            temperature=cc.soft_temperature,
            explore_ratio=cc.explore_ratio,
            T_init=cc.T_init,
            max_candidates=cc.max_candidates,
            mmr_lambda=cc.mmr_lambda,
            svd_score_weight=getattr(cc, 'svd_score_weight', 0.7),
        )

    def forward(self, edge_index_dict, edge_attr_dict=None):
        """Get all node embeddings from HGT."""
        return self.hgt(edge_index_dict, edge_attr_dict)

    def get_gene_embeddings(self, edge_index_dict, edge_attr_dict=None):
        """Get fused gene embeddings for human and mouse.

        Returns: (E_human [N_h, D], E_mouse [N_m, D], entropy_loss scalar)
        """
        emb = self.forward(edge_index_dict, edge_attr_dict)

        E_h_raw = emb["Human_Gene"]
        E_m_raw = emb["Mouse_Gene"]
        E_bp = emb["BP"]
        E_mf = emb["MF"]
        E_dis = emb["Disease"]

        def _get_mod_feat(name):
            buf = getattr(self.hgt, name, None)
            return buf.to(E_h_raw.device) if buf is not None else None

        def _get_ei(key):
            return edge_index_dict.get(key), (edge_attr_dict or {}).get(key)

        # Human fusion
        ei_hb, ea_hb = _get_ei(("Human_Gene", "HG-BP", "BP"))
        ei_hf, ea_hf = _get_ei(("Human_Gene", "HG-MF", "MF"))
        ei_hd, ea_hd = _get_ei(("Human_Gene", "HG-D", "Disease"))

        E_h, _, ent_h = self.fusion_human(
            E_h_raw, E_bp, E_mf, E_dis,
            ei_bp=ei_hb, ei_mf=ei_hf, ei_disease=ei_hd,
            ea_bp=ea_hb, ea_mf=ea_hf, ea_disease=ea_hd,
            bp_gene_feat=_get_mod_feat('Z_h_BP'),
            mf_gene_feat=_get_mod_feat('Z_h_MF'),
            dis_gene_feat=_get_mod_feat('Z_h_Disease'),
        )

        # Mouse fusion
        ei_mb, ea_mb = _get_ei(("Mouse_Gene", "MG-BP", "BP"))
        ei_mf_edge, ea_mf_edge = _get_ei(("Mouse_Gene", "MG-MF", "MF"))
        ei_md, ea_md = _get_ei(("Mouse_Gene", "MG-D", "Disease"))

        E_m, _, ent_m = self.fusion_mouse(
            E_m_raw, E_bp, E_mf, E_dis,
            ei_bp=ei_mb, ei_mf=ei_mf_edge, ei_disease=ei_md,
            ea_bp=ea_mb, ea_mf=ea_mf_edge, ea_disease=ea_md,
            bp_gene_feat=_get_mod_feat('Z_m_BP'),
            mf_gene_feat=_get_mod_feat('Z_m_MF'),
            dis_gene_feat=_get_mod_feat('Z_m_Disease'),
        )

        # Center embeddings
        E_h = E_h - E_h.mean(dim=0, keepdim=True)
        E_m = E_m - E_m.mean(dim=0, keepdim=True)

        return E_h, E_m, ent_h + ent_m


def create_model_v3(cfg, node_counts: Dict[str, int]) -> CrossSpeciesModelV3:
    """Create v3 model from config."""
    return CrossSpeciesModelV3(node_counts=node_counts, cfg=cfg)
