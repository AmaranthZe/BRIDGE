"""Inference pipeline with FGW refinement.

CRITICAL FIX from v1:
- v1 infer.py lines 266-278 used ONLY cosine similarity for matching,
  completely bypassing FGW. The local_refine_iters config was read but never used.
- v2 runs full local FGW refinement per query gene for proper optimal transport
  matching with structure + feature costs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..model.model import CrossSpeciesModel
from ..transport.costs import CostComputer, structural_cost
from ..transport.fgw import FGWSolver
from .readout import ReadoutModule

logger = logging.getLogger(__name__)


def parse_query(query: str) -> Tuple[List[str], List[str]]:
    """Parse semicolon-separated gene IDs into (human_ids, mouse_ids)."""
    ids = [x.strip() for x in query.split(";") if x.strip()]
    human, mouse = [], []
    for gid in ids:
        if gid.startswith("ENSMUSG"):
            mouse.append(gid)
        elif gid.startswith("ENSG"):
            human.append(gid)
        else:
            logger.warning(f"Unknown gene ID format: {gid}")
    return human, mouse


class Inferencer:
    """Inference engine with FGW refinement."""

    def __init__(
        self,
        model: CrossSpeciesModel,
        graph_data,
        sparse_matrices: Dict,
        Z_h: np.ndarray,
        Z_m: np.ndarray,
        indexers: Dict,
        cfg,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.indexers = indexers
        self.cfg = cfg
        self.device = device

        ic = cfg.infer
        self.topk_genes = ic.topk_genes
        self.fgw_refine_iters = ic.fgw_refine_iters

        # FGW solver for local refinement
        fc = cfg.fgw
        self.fgw_solver = FGWSolver(
            alpha=fc.alpha, epsilon=fc.epsilon,
            lambda_prior=0, lambda_memory=0,
            use_unbalanced=fc.use_unbalanced,
            tau_row=fc.tau_row, tau_col=fc.tau_col,
            max_iter=self.fgw_refine_iters,
        )

        # Cost computer
        self.cost_computer = CostComputer(Z_h=Z_h, Z_m=Z_m, device=device)

        # SVD features as tensors on device (for candidate selector)
        self.Z_h_gpu = self.cost_computer.Z_h  # already on device
        self.Z_m_gpu = self.cost_computer.Z_m

        # Per-modality gene features from HGT buffers (loaded via checkpoint)
        self.mod_features = {}
        for key in ['Z_h_BP', 'Z_h_MF', 'Z_h_Disease', 'Z_m_BP', 'Z_m_MF', 'Z_m_Disease']:
            buf = getattr(self.model.hgt, key, None)
            if buf is not None:
                self.mod_features[key] = buf.to(device)
        if self.mod_features:
            logger.info(f"Loaded {len(self.mod_features)} per-modality gene feature tensors for inference")
        else:
            logger.warning("No per-modality gene features found — candidate selector will use unified SVD only")

        # Readout
        W_human = {"bp": sparse_matrices["W_hB"], "mf": sparse_matrices["W_hF"], "disease": sparse_matrices["W_hD"]}
        W_mouse = {"bp": sparse_matrices["W_mB"], "mf": sparse_matrices["W_mF"], "disease": sparse_matrices["W_mD"]}
        self.readout = ReadoutModule(
            W_human=W_human, W_mouse=W_mouse, indexers=indexers,
            topk={"bp": ic.topk_bp, "mf": ic.topk_mf, "disease": ic.topk_disease},
        )

        # Pre-compute graph data and embeddings
        self.edge_index_dict = {}
        self.edge_attr_dict = {}
        for et in graph_data.edge_types:
            ed = graph_data[et]
            self.edge_index_dict[et] = ed.edge_index.to(device)
            if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
                self.edge_attr_dict[et] = ed.edge_attr.to(device)

        with torch.no_grad():
            self.E_h, self.E_m, _ = self.model.get_gene_embeddings(
                self.edge_index_dict, self.edge_attr_dict)

    def infer(self, query: str, direction: str = "auto") -> Dict:
        """Run inference on semicolon-separated gene IDs."""
        human_ids, mouse_ids = parse_query(query)
        if not human_ids and not mouse_ids:
            return {"error": "No valid gene IDs found"}

        if direction == "auto":
            if human_ids and mouse_ids:
                direction = "bidirectional"
            elif human_ids:
                direction = "human_to_mouse"
            else:
                direction = "mouse_to_human"

        results = {
            "query": human_ids + mouse_ids,
            "direction": direction,
            "timestamp": datetime.now().isoformat(),
            "per_gene_results": [],
        }

        if direction in ("human_to_mouse", "bidirectional") and human_ids:
            results["per_gene_results"].extend(
                self._infer_direction(human_ids, "human"))
        if direction in ("mouse_to_human", "bidirectional") and mouse_ids:
            results["per_gene_results"].extend(
                self._infer_direction(mouse_ids, "mouse"))

        return results

    def _infer_direction(self, source_ids: List[str], source_species: str) -> List[Dict]:
        if source_species == "human":
            src_idx_map = self.indexers["human_genes"]
            tgt_inv = self.indexers["mouse_genes_inv"]
            E_src, E_tgt = self.E_h, self.E_m
            Z_src, Z_tgt = self.Z_h_gpu, self.Z_m_gpu
            tgt_species = "mouse"
            fgw_dir = "human_to_mouse"
            Z_src_mods = {
                'bp': self.mod_features.get('Z_h_BP'),
                'mf': self.mod_features.get('Z_h_MF'),
                'disease': self.mod_features.get('Z_h_Disease'),
            }
            Z_tgt_mods = {
                'bp': self.mod_features.get('Z_m_BP'),
                'mf': self.mod_features.get('Z_m_MF'),
                'disease': self.mod_features.get('Z_m_Disease'),
            }
        else:
            src_idx_map = self.indexers["mouse_genes"]
            tgt_inv = self.indexers["human_genes_inv"]
            E_src, E_tgt = self.E_m, self.E_h
            Z_src, Z_tgt = self.Z_m_gpu, self.Z_h_gpu
            tgt_species = "human"
            fgw_dir = "mouse_to_human"
            Z_src_mods = {
                'bp': self.mod_features.get('Z_m_BP'),
                'mf': self.mod_features.get('Z_m_MF'),
                'disease': self.mod_features.get('Z_m_Disease'),
            }
            Z_tgt_mods = {
                'bp': self.mod_features.get('Z_h_BP'),
                'mf': self.mod_features.get('Z_h_MF'),
                'disease': self.mod_features.get('Z_h_Disease'),
            }

        results = []
        for sid in source_ids:
            if sid not in src_idx_map:
                logger.warning(f"Gene not found: {sid}")
                continue

            src_i = src_idx_map[sid]
            I_batch = torch.tensor([src_i], dtype=torch.long, device=self.device)

            with torch.no_grad():
                # Step 1: Candidate selection (with SVD features for ensemble retrieval)
                J_cand, soft_mask, Pi0, row_mask = self.model.candidate_selector(
                    E_source=E_src, E_target=E_tgt, source_indices=I_batch,
                    Z_source=Z_src, Z_target=Z_tgt,
                    Z_source_mods=Z_src_mods, Z_target_mods=Z_tgt_mods)

                if len(J_cand) == 0:
                    continue

                E_src_single = E_src[src_i:src_i + 1]
                E_tgt_cand = E_tgt[J_cand]

                # Step 2: Local FGW refinement (THE FIX — v1 skipped this entirely)
                costs = self.cost_computer.compute(
                    E_source=E_src_single,
                    E_target=E_tgt_cand,
                    source_indices=I_batch,
                    target_indices=J_cand,
                    direction=fgw_dir,
                )

                M_masked = costs["M"] * soft_mask

                T_local, _ = self.fgw_solver.solve(
                    C_s=costs["C_s"], C_t=costs["C_t"],
                    M=M_masked, Pi0=Pi0, mask=row_mask)

                # Step 3: Extract top-k from FGW coupling
                t_row = T_local[0]  # [J]
                topk = min(self.topk_genes, len(t_row))
                topk_scores, topk_pos = torch.topk(t_row, topk)

                # Normalize scores
                topk_scores = topk_scores / (topk_scores.sum() + 1e-8)

            matched_idx = J_cand[topk_pos].cpu().numpy()
            matched_sc = topk_scores.cpu().numpy()

            matched_genes = []
            for rank, (idx, score) in enumerate(zip(matched_idx, matched_sc)):
                tgt_id = tgt_inv[int(idx)]
                symbol = self.readout.get_gene_symbol(tgt_species, tgt_id)
                shared = self.readout.compute_shared(source_species, src_i, int(idx))
                matched_genes.append({
                    "id": tgt_id, "symbol": symbol,
                    "score": float(score), "rank": rank + 1,
                    "shared_terms": shared,
                })

            related = self.readout.compute_for_matches(
                tgt_species, matched_idx.tolist(), matched_sc)

            results.append({
                "source": {
                    "id": sid,
                    "symbol": self.readout.get_gene_symbol(source_species, sid),
                    "species": source_species,
                },
                "matched_genes": matched_genes,
                "related_gobp": related["bp"],
                "related_gomf": related["mf"],
                "related_diseases": related["disease"],
            })

        return results
