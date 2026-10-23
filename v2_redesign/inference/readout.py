"""Readout module: compute shared and aggregated GO/Disease terms."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class ReadoutModule:
    """Compute related terms for matched gene pairs."""

    def __init__(
        self,
        W_human: Dict[str, sp.csr_matrix],
        W_mouse: Dict[str, sp.csr_matrix],
        indexers: Dict,
        topk: Dict[str, int],
    ):
        """
        Args:
            W_human: {'bp': W_hB, 'mf': W_hF, 'disease': W_hD}
            W_mouse: {'bp': W_mB, 'mf': W_mF, 'disease': W_mD}
            indexers: ID indexers with inverse mappings
            topk: {'bp': 20, 'mf': 10, 'disease': 20}
        """
        self.W_human = W_human
        self.W_mouse = W_mouse
        self.indexers = indexers
        self.topk = topk

        # Build inverse term indexers
        self.bp_inv = {i: t for t, i in indexers["bp_terms"].items()}
        self.mf_inv = {i: t for t, i in indexers["mf_terms"].items()}
        self.dis_inv = {i: t for t, i in indexers["diseases"].items()}

    def get_gene_symbol(self, species: str, gene_id: str) -> str:
        return self.indexers["gene_symbols"].get(gene_id, gene_id)

    def compute_shared(
        self,
        source_species: str,
        source_idx: int,
        target_idx: int,
    ) -> Dict[str, List[str]]:
        """Compute shared GO/Disease terms between a source-target gene pair."""
        if source_species == "human":
            W_src, W_tgt = self.W_human, self.W_mouse
        else:
            W_src, W_tgt = self.W_mouse, self.W_human

        shared = {}
        for key, inv in [("bp", self.bp_inv), ("mf", self.mf_inv), ("disease", self.dis_inv)]:
            src_terms = set(W_src[key][source_idx].nonzero()[1])
            tgt_terms = set(W_tgt[key][target_idx].nonzero()[1])
            common = src_terms & tgt_terms
            shared[key] = [inv[i] for i in sorted(common) if i in inv]

        return shared

    def compute_for_matches(
        self,
        target_species: str,
        matched_indices: List[int],
        matched_scores: np.ndarray,
    ) -> Dict[str, List[Dict]]:
        """
        Compute score-weighted aggregated terms from matched genes.

        Returns terms from the matched genes, ranked by weighted frequency.
        """
        if target_species == "human":
            W_tgt = self.W_human
        else:
            W_tgt = self.W_mouse

        results = {}
        for key, inv, k in [
            ("bp", self.bp_inv, self.topk["bp"]),
            ("mf", self.mf_inv, self.topk["mf"]),
            ("disease", self.dis_inv, self.topk["disease"]),
        ]:
            term_scores = {}
            for idx, score in zip(matched_indices, matched_scores):
                row = W_tgt[key][idx]
                for col_idx in row.nonzero()[1]:
                    weight = float(row[0, col_idx]) * float(score)
                    term_scores[col_idx] = term_scores.get(col_idx, 0) + weight

            # Sort by weighted score
            sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])[:k]
            results[key] = [
                {"term": inv.get(t, str(t)), "score": round(s, 4)}
                for t, s in sorted_terms
            ]

        return results
