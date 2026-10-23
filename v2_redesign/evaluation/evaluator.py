"""Full evaluation pipeline.

Evaluation methods:
1. Held-out functional term recovery
2. Ortholog recovery rate
3. Conserved pathway agreement
4. Cycle consistency (H→M→H)
5. PPI network conservation
6. Phenotype ontology agreement
7. Tissue expression correlation
8. Drug target consistency
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

from .metrics import (
    jaccard_similarity,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive evaluation for cross-species gene mapping."""

    def __init__(
        self,
        sparse_matrices: Dict[str, sp.csr_matrix],
        indexers: Dict,
        split_info: Optional[Dict] = None,
    ):
        self.sparse_matrices = sparse_matrices
        self.indexers = indexers
        self.split_info = split_info

    def evaluate_heldout_terms(
        self,
        predictions: Dict[int, List[Tuple[int, float]]],
        direction: str = "mouse_to_human",
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """
        Method 1: Held-out functional term recovery.

        For each source gene, check if the matched target genes share
        the held-out GO/Disease terms with the source.
        """
        if self.split_info is None:
            logger.warning("No split_info for held-out evaluation")
            return {}

        if direction == "mouse_to_human":
            W_src_bp, W_tgt_bp = self.sparse_matrices["W_mB"], self.sparse_matrices["W_hB"]
            W_src_mf, W_tgt_mf = self.sparse_matrices["W_mF"], self.sparse_matrices["W_hF"]
            W_src_dis, W_tgt_dis = self.sparse_matrices["W_mD"], self.sparse_matrices["W_hD"]
        else:
            W_src_bp, W_tgt_bp = self.sparse_matrices["W_hB"], self.sparse_matrices["W_mB"]
            W_src_mf, W_tgt_mf = self.sparse_matrices["W_hF"], self.sparse_matrices["W_mF"]
            W_src_dis, W_tgt_dis = self.sparse_matrices["W_hD"], self.sparse_matrices["W_mD"]

        bp_held = set(self.split_info.get("bp_held", []))
        mf_held = set(self.split_info.get("mf_held", []))
        dis_held = set(self.split_info.get("dis_held", []))

        jaccards = []
        for src_idx, pred_list in predictions.items():
            # Source gene's held-out terms
            src_bp = set(W_src_bp[src_idx].nonzero()[1]) & bp_held
            src_mf = set(W_src_mf[src_idx].nonzero()[1]) & mf_held
            src_dis = set(W_src_dis[src_idx].nonzero()[1]) & dis_held
            src_held = src_bp | src_mf | src_dis

            if not src_held:
                continue

            # Matched genes' terms in held-out columns
            tgt_held = set()
            for tgt_idx, score in pred_list[:k_values[-1]]:
                tgt_held |= set(W_tgt_bp[tgt_idx].nonzero()[1]) & bp_held
                tgt_held |= set(W_tgt_mf[tgt_idx].nonzero()[1]) & mf_held
                tgt_held |= set(W_tgt_dis[tgt_idx].nonzero()[1]) & dis_held

            jaccards.append(jaccard_similarity(src_held, tgt_held))

        return {"heldout_jaccard": float(np.mean(jaccards)) if jaccards else 0.0}

    def evaluate_go_agreement(
        self,
        predictions: Dict[int, List[Tuple[int, float]]],
        direction: str = "mouse_to_human",
    ) -> Dict[str, float]:
        """Method 3: GO term agreement (Jaccard) for matched pairs."""
        if direction == "mouse_to_human":
            W_src_bp = self.sparse_matrices["W_mB"]
            W_tgt_bp = self.sparse_matrices["W_hB"]
        else:
            W_src_bp = self.sparse_matrices["W_hB"]
            W_tgt_bp = self.sparse_matrices["W_mB"]

        jaccards = []
        for src_idx, pred_list in predictions.items():
            src_terms = set(W_src_bp[src_idx].nonzero()[1])
            if not src_terms:
                continue
            for tgt_idx, score in pred_list[:5]:  # Top-5
                tgt_terms = set(W_tgt_bp[tgt_idx].nonzero()[1])
                jaccards.append(jaccard_similarity(src_terms, tgt_terms))

        return {
            "go_bp_jaccard_mean": float(np.mean(jaccards)) if jaccards else 0.0,
        }

    def evaluate_cycle_consistency(
        self,
        pred_forward: Dict[int, List[Tuple[int, float]]],
        pred_backward: Dict[int, List[Tuple[int, float]]],
        topk: int = 10,
    ) -> Dict[str, float]:
        """Method 4: Cycle consistency — if A→B, does B→A?"""
        n_consistent = 0
        n_total = 0

        for src, fwd_list in pred_forward.items():
            fwd_targets = {t for t, _ in fwd_list[:topk]}
            for tgt in fwd_targets:
                if tgt in pred_backward:
                    bwd_sources = {s for s, _ in pred_backward[tgt][:topk]}
                    if src in bwd_sources:
                        n_consistent += 1
                    n_total += 1

        return {
            "cycle_consistency": n_consistent / max(n_total, 1),
            "cycle_total_pairs": n_total,
        }

    def evaluate_all(
        self,
        predictions_m2h: Dict[int, List[Tuple[int, float]]],
        predictions_h2m: Optional[Dict[int, List[Tuple[int, float]]]] = None,
    ) -> Dict[str, float]:
        """Run all applicable evaluation methods."""
        results = {}

        # Held-out terms
        results.update(self.evaluate_heldout_terms(predictions_m2h, "mouse_to_human"))

        # GO agreement
        results.update(self.evaluate_go_agreement(predictions_m2h, "mouse_to_human"))

        # Cycle consistency
        if predictions_h2m is not None:
            results.update(self.evaluate_cycle_consistency(predictions_m2h, predictions_h2m))

        return results


def assemble_P_rowwise(
    model,
    E_source: Tensor,
    E_target: Tensor,
    cost_computer,
    fgw_solver,
    direction: str = "mouse_to_human",
    source_indices: Optional[List[int]] = None,
    device: str = "cpu",
    threshold: float = 1e-6,
) -> sp.csr_matrix:
    """Row-wise P assembly via per-query local FGW."""
    N_t = E_target.shape[0]
    N_s = E_source.shape[0]

    if source_indices is None:
        source_indices = list(range(N_s))

    rows, cols, vals = [], [], []

    model.eval()
    with torch.no_grad():
        for row_idx, src_idx in enumerate(source_indices):
            I = torch.tensor([src_idx], dtype=torch.long, device=device)
            E_s1 = E_source[src_idx:src_idx + 1]

            J_cand, soft_mask, Pi0, row_mask = model.candidate_selector(
                E_source=E_source, E_target=E_target, source_indices=I)

            if len(J_cand) == 0:
                continue

            costs = cost_computer.compute(
                E_source=E_s1, E_target=E_target[J_cand],
                source_indices=I, target_indices=J_cand,
                direction=direction)

            T_local, _ = fgw_solver.solve(
                C_s=costs["C_s"], C_t=costs["C_t"],
                M=costs["M"] * soft_mask, Pi0=Pi0, mask=row_mask)

            t_row = T_local[0].cpu().numpy()
            j_np = J_cand.cpu().numpy()

            nz = t_row > threshold
            if nz.any():
                ci = j_np[nz]
                vi = t_row[nz]
                order = np.argsort(ci)
                rows.extend([row_idx] * len(ci))
                cols.extend(ci[order].tolist())
                vals.extend(vi[order].tolist())

    N_eval = len(source_indices)
    if rows:
        P = sp.csr_matrix(
            (np.array(vals, dtype=np.float32),
             (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
            shape=(N_eval, N_t))
    else:
        P = sp.csr_matrix((N_eval, N_t), dtype=np.float32)

    logger.info(f"Assembled P: shape={P.shape}, nnz={P.nnz}")
    return P
