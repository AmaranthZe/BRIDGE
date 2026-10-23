"""
Readout Module for Related Terms Computation.

This module computes:
- Related GOBP terms for matched genes
- Related GOMF terms for matched genes
- Related Disease terms for matched genes
- Shared terms between source and target genes (for explainability)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def compute_related_terms(
    matched_gene_ids: List[int],
    matched_scores: np.ndarray,
    W_bar: Dict[str, sp.csr_matrix],
    term_indexers: Dict[str, Dict[str, int]],
    topk: Dict[str, int] = None,
) -> Dict[str, List[Dict]]:
    """
    Compute related GOBP/GOMF/Disease terms for matched genes.

    Algorithm:
        score(term) = sum_{gene i in matched} coupling_score[i] * W_bar[i, term]

    Args:
        matched_gene_ids: List of matched gene indices
        matched_scores: Coupling scores for matched genes [num_matched]
        W_bar: Dict of normalized sparse matrices {'bp': W_B, 'mf': W_F, 'disease': W_D}
        term_indexers: Dict of term-to-index mappings
        topk: Dict of top-k limits per term type

    Returns:
        {
            'bp': [{'id': 'GO:xxx', 'score': 0.67, 'gene_count': 3}, ...],
            'mf': [...],
            'disease': [...]
        }
    """
    if topk is None:
        topk = {'bp': 20, 'mf': 10, 'disease': 20}

    matched_scores = np.asarray(matched_scores).flatten()
    results = {}

    for term_type in ['bp', 'mf', 'disease']:
        W = W_bar[term_type]
        k = topk.get(term_type, 20)

        if len(matched_gene_ids) == 0:
            results[term_type] = []
            continue

        # Extract rows for matched genes
        W_matched = W[matched_gene_ids]  # [num_matched, num_terms]

        # Weight by coupling scores
        # scores_col: [num_matched, 1]
        scores_col = matched_scores.reshape(-1, 1)

        # Element-wise multiply (sparse * dense -> sparse)
        weighted_W = W_matched.multiply(scores_col)

        # Sum across genes to get per-term scores
        term_scores = np.array(weighted_W.sum(axis=0)).flatten()

        # Count how many matched genes have each term
        gene_counts = np.array((W_matched > 0).sum(axis=0)).flatten()

        # Get top-K terms
        top_indices = np.argsort(term_scores)[::-1][:k]

        # Build result list
        term_indexer = term_indexers.get(term_type, term_indexers.get(f'{term_type}_terms', {}))
        idx_to_term = {v: k for k, v in term_indexer.items()}

        term_list = []
        for idx in top_indices:
            score = term_scores[idx]
            if score > 0:
                term_id = idx_to_term.get(idx, f"UNKNOWN_{idx}")
                term_list.append({
                    'id': term_id,
                    'score': float(score),
                    'gene_count': int(gene_counts[idx])
                })

        results[term_type] = term_list

    return results


def compute_shared_terms(
    source_gene_id: int,
    target_gene_id: int,
    W_source: Dict[str, sp.csr_matrix],
    W_target: Dict[str, sp.csr_matrix],
    term_indexers: Dict[str, Dict[str, int]],
) -> Dict[str, List[str]]:
    """
    Find GO/Disease terms shared between source and target gene.

    These are the "bridges" in the unified graph explaining the match.

    Args:
        source_gene_id: Source gene index
        target_gene_id: Target gene index
        W_source: Source species sparse matrices
        W_target: Target species sparse matrices
        term_indexers: Term-to-index mappings

    Returns:
        {
            'bp': ['GO:0006915', 'GO:0006979', ...],
            'mf': ['GO:0003700', ...],
            'disease': ['EFO_0000616', ...]
        }
    """
    shared = {}

    for term_type in ['bp', 'mf', 'disease']:
        W_s = W_source[term_type]
        W_t = W_target[term_type]

        # Get non-zero term indices for both genes
        source_terms = set(W_s[source_gene_id].nonzero()[1])
        target_terms = set(W_t[target_gene_id].nonzero()[1])

        # Intersection = shared terms
        common_indices = source_terms & target_terms

        # Convert back to term IDs
        term_indexer = term_indexers.get(term_type, term_indexers.get(f'{term_type}_terms', {}))
        idx_to_term = {v: k for k, v in term_indexer.items()}

        shared[term_type] = [idx_to_term.get(i, f"UNKNOWN_{i}") for i in sorted(common_indices)]

    return shared


def compute_aggregate_results(
    per_gene_results: List[Dict],
) -> Dict:
    """
    Compute aggregate results across all input genes.

    Args:
        per_gene_results: List of per-gene result dictionaries

    Returns:
        Aggregate results with combined top terms and genes
    """
    if not per_gene_results:
        return {}

    # Aggregate matched genes
    gene_scores = {}
    gene_symbols = {}
    gene_sources = {}

    for result in per_gene_results:
        source_id = result['source']['id']
        for match in result.get('matched_genes', []):
            gene_id = match['id']
            score = match['score']

            if gene_id not in gene_scores:
                gene_scores[gene_id] = []
                gene_symbols[gene_id] = match.get('symbol', gene_id)
                gene_sources[gene_id] = set()

            gene_scores[gene_id].append(score)
            gene_sources[gene_id].add(source_id)

    # Compute aggregate gene scores (mean of scores)
    agg_genes = []
    for gene_id in gene_scores:
        scores = gene_scores[gene_id]
        agg_genes.append({
            'id': gene_id,
            'symbol': gene_symbols[gene_id],
            'score': float(np.mean(scores)),
            'source_count': len(gene_sources[gene_id]),
        })

    agg_genes = sorted(agg_genes, key=lambda x: -x['score'])

    # Aggregate term scores
    agg_terms = {'bp': {}, 'mf': {}, 'disease': {}}

    for result in per_gene_results:
        for term_type in ['bp', 'mf', 'disease']:
            key = f'related_{term_type}' if f'related_{term_type}' in result else f'related_go{term_type}'
            terms = result.get(key, result.get(f'related_{term_type}', []))

            for term in terms:
                term_id = term['id']
                if term_id not in agg_terms[term_type]:
                    agg_terms[term_type][term_id] = {
                        'scores': [],
                        'gene_counts': [],
                    }
                agg_terms[term_type][term_id]['scores'].append(term['score'])
                agg_terms[term_type][term_id]['gene_counts'].append(term.get('gene_count', 1))

    # Build aggregate term lists
    def build_term_list(term_dict, max_count=20):
        items = []
        for term_id, data in term_dict.items():
            items.append({
                'id': term_id,
                'score': float(np.mean(data['scores'])),
                'coverage': len(data['scores']) / len(per_gene_results),
            })
        items = sorted(items, key=lambda x: -x['score'])[:max_count]
        return items

    return {
        'matched_genes': agg_genes[:20],
        'top_gobp': build_term_list(agg_terms['bp']),
        'top_gomf': build_term_list(agg_terms['mf']),
        'top_diseases': build_term_list(agg_terms['disease']),
    }


class ReadoutModule:
    """
    Readout module for computing related terms from coupling results.
    """

    def __init__(
        self,
        W_human: Dict[str, sp.csr_matrix],
        W_mouse: Dict[str, sp.csr_matrix],
        indexers: Dict,
        topk: Dict[str, int] = None,
    ):
        """
        Args:
            W_human: Human gene sparse matrices {'bp': W_hB, 'mf': W_hF, 'disease': W_hD}
            W_mouse: Mouse gene sparse matrices
            indexers: ID indexers
            topk: Top-k limits per term type
        """
        self.W_human = W_human
        self.W_mouse = W_mouse
        self.indexers = indexers
        self.topk = topk or {'bp': 20, 'mf': 10, 'disease': 20}

        # Build term indexers
        self.term_indexers = {
            'bp': indexers.get('bp_terms', {}),
            'mf': indexers.get('mf_terms', {}),
            'disease': indexers.get('diseases', {}),
        }

    def compute_for_matches(
        self,
        source_species: str,
        matched_indices: List[int],
        matched_scores: np.ndarray,
    ) -> Dict[str, List[Dict]]:
        """
        Compute related terms for matched genes.

        Args:
            source_species: 'human' or 'mouse' (determines which W matrices to use)
            matched_indices: Matched gene indices in target species
            matched_scores: Coupling scores

        Returns:
            Related terms dictionary
        """
        # Target species is opposite of source
        if source_species == 'human':
            W_bar = self.W_mouse
        else:
            W_bar = self.W_human

        return compute_related_terms(
            matched_gene_ids=matched_indices,
            matched_scores=matched_scores,
            W_bar=W_bar,
            term_indexers=self.term_indexers,
            topk=self.topk,
        )

    def compute_shared(
        self,
        source_species: str,
        source_gene_id: int,
        target_gene_id: int,
    ) -> Dict[str, List[str]]:
        """
        Compute shared terms between source and target gene.

        Args:
            source_species: 'human' or 'mouse'
            source_gene_id: Source gene index
            target_gene_id: Target gene index

        Returns:
            Shared terms dictionary
        """
        if source_species == 'human':
            W_source = self.W_human
            W_target = self.W_mouse
        else:
            W_source = self.W_mouse
            W_target = self.W_human

        return compute_shared_terms(
            source_gene_id=source_gene_id,
            target_gene_id=target_gene_id,
            W_source=W_source,
            W_target=W_target,
            term_indexers=self.term_indexers,
        )

    def get_gene_symbol(self, species: str, gene_id: str) -> str:
        """Get gene symbol for a gene ID."""
        if species == 'human':
            names = self.indexers.get('human_gene_names', {})
        else:
            names = self.indexers.get('mouse_gene_names', {})

        return names.get(gene_id, gene_id)
