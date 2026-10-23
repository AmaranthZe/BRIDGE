"""
Inference Pipeline for Cross-Species Gene Mapping.

This module implements:
- Query parsing and validation
- Gene mapping with local FGW refinement
- Related terms computation
- JSON output formatting
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import CrossSpeciesModel, create_model
from .memory_bank import MemoryBank
from .readout import ReadoutModule, compute_aggregate_results
from .fgw import solve_fgw
from .costs import structural_cost, feature_cost_from_sparse

logger = logging.getLogger(__name__)


def parse_query(query: str) -> Tuple[List[str], List[str]]:
    """
    Parse query string into human and mouse gene IDs.

    Args:
        query: Semicolon-separated gene IDs

    Returns:
        (human_ids, mouse_ids)
    """
    ids = [x.strip() for x in query.split(';') if x.strip()]

    human_ids = []
    mouse_ids = []

    for gene_id in ids:
        if gene_id.startswith('ENSMUSG'):
            mouse_ids.append(gene_id)
        elif gene_id.startswith('ENSG'):
            human_ids.append(gene_id)
        else:
            logger.warning(f"Unknown gene ID format: {gene_id}")

    return human_ids, mouse_ids


class Inferencer:
    """
    Inference engine for cross-species gene mapping.
    """

    def __init__(
        self,
        model: CrossSpeciesModel,
        graph_data,
        sparse_matrices: Dict,
        indexers: Dict,
        config: Dict,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.graph_data = graph_data
        self.sparse_matrices = sparse_matrices
        self.indexers = indexers
        self.config = config
        self.device = device

        # Inference config
        infer_cfg = config['infer']
        self.local_refine_iters = infer_cfg['local_refine_iters']
        self.topk_genes = infer_cfg['topk_genes']
        self.topk_bp = infer_cfg['topk_bp']
        self.topk_mf = infer_cfg['topk_mf']
        self.topk_disease = infer_cfg['topk_disease']

        # Initialize components
        self._prepare_graph_data()
        self._init_readout()
        self._compute_embeddings()

    def _prepare_graph_data(self):
        """Prepare graph data."""
        self.edge_index_dict = {}
        self.edge_attr_dict = {}

        for edge_type in self.graph_data.edge_types:
            edge_data = self.graph_data[edge_type]
            self.edge_index_dict[edge_type] = edge_data.edge_index.to(self.device)
            if hasattr(edge_data, 'edge_attr') and edge_data.edge_attr is not None:
                self.edge_attr_dict[edge_type] = edge_data.edge_attr.to(self.device)

    def _init_readout(self):
        """Initialize readout module."""
        W_human = {
            'bp': self.sparse_matrices['W_hB'],
            'mf': self.sparse_matrices['W_hF'],
            'disease': self.sparse_matrices['W_hD'],
        }
        W_mouse = {
            'bp': self.sparse_matrices['W_mB'],
            'mf': self.sparse_matrices['W_mF'],
            'disease': self.sparse_matrices['W_mD'],
        }

        self.readout = ReadoutModule(
            W_human=W_human,
            W_mouse=W_mouse,
            indexers=self.indexers,
            topk={
                'bp': self.topk_bp,
                'mf': self.topk_mf,
                'disease': self.topk_disease,
            },
        )

    def _compute_embeddings(self):
        """Pre-compute gene embeddings."""
        with torch.no_grad():
            self.E_h, self.E_m, _ = self.model.get_gene_embeddings(
                self.edge_index_dict,
                self.edge_attr_dict,
            )

    def infer(
        self,
        query: str,
        direction: str = 'auto',
    ) -> Dict:
        """
        Run inference on query.

        Args:
            query: Semicolon-separated gene IDs
            direction: 'human_to_mouse', 'mouse_to_human', or 'auto'

        Returns:
            Result dictionary with matched genes and related terms
        """
        human_ids, mouse_ids = parse_query(query)

        if not human_ids and not mouse_ids:
            return {'error': 'No valid gene IDs found in query'}

        results = {
            'query': human_ids + mouse_ids,
            'timestamp': datetime.now().isoformat(),
            'per_gene_results': [],
        }

        # Determine direction
        if direction == 'auto':
            if human_ids and mouse_ids:
                direction = 'bidirectional'
            elif human_ids:
                direction = 'human_to_mouse'
            else:
                direction = 'mouse_to_human'

        results['direction'] = direction

        # Process human → mouse
        if direction in ['human_to_mouse', 'bidirectional'] and human_ids:
            h_results = self._infer_direction(
                source_ids=human_ids,
                source_species='human',
            )
            results['per_gene_results'].extend(h_results)

        # Process mouse → human
        if direction in ['mouse_to_human', 'bidirectional'] and mouse_ids:
            m_results = self._infer_direction(
                source_ids=mouse_ids,
                source_species='mouse',
            )
            results['per_gene_results'].extend(m_results)

        # Compute aggregate results
        results['aggregate_results'] = compute_aggregate_results(
            results['per_gene_results']
        )

        # Add metadata
        results['meta'] = {
            'model_version': '1.0.0',
            'config': {
                'topk_genes': self.topk_genes,
                'topk_bp': self.topk_bp,
                'topk_mf': self.topk_mf,
                'topk_disease': self.topk_disease,
            },
        }

        return results

    def _infer_direction(
        self,
        source_ids: List[str],
        source_species: str,
    ) -> List[Dict]:
        """
        Infer matches for one direction.

        Args:
            source_ids: List of source gene IDs
            source_species: 'human' or 'mouse'

        Returns:
            List of per-gene results
        """
        if source_species == 'human':
            source_indexer = self.indexers['human_genes']
            target_indexer = self.indexers['mouse_genes']
            target_inv = self.indexers['mouse_genes_inv']
            E_source = self.E_h
            E_target = self.E_m
            W_source = {
                'bp': self.sparse_matrices['W_hB'],
                'mf': self.sparse_matrices['W_hF'],
                'disease': self.sparse_matrices['W_hD'],
            }
            W_target = {
                'bp': self.sparse_matrices['W_mB'],
                'mf': self.sparse_matrices['W_mF'],
                'disease': self.sparse_matrices['W_mD'],
            }
            target_species = 'mouse'
        else:
            source_indexer = self.indexers['mouse_genes']
            target_indexer = self.indexers['human_genes']
            target_inv = self.indexers['human_genes_inv']
            E_source = self.E_m
            E_target = self.E_h
            W_source = {
                'bp': self.sparse_matrices['W_mB'],
                'mf': self.sparse_matrices['W_mF'],
                'disease': self.sparse_matrices['W_mD'],
            }
            W_target = {
                'bp': self.sparse_matrices['W_hB'],
                'mf': self.sparse_matrices['W_hF'],
                'disease': self.sparse_matrices['W_hD'],
            }
            target_species = 'human'

        results = []

        for source_id in source_ids:
            if source_id not in source_indexer:
                logger.warning(f"Gene ID not found: {source_id}")
                continue

            source_idx = source_indexer[source_id]

            # Get source embedding
            e_source = E_source[source_idx:source_idx+1]

            # Compute attention scores to all targets
            with torch.no_grad():
                Q = self.model.candidate_selector.W_Q(e_source)
                K = self.model.candidate_selector.W_K(E_target)
                # L2 normalize for cosine similarity (matching training)
                Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
                K = torch.nn.functional.normalize(K, p=2, dim=-1)
                scores = (Q @ K.t()).squeeze(0)

                # Get top-k targets
                topk_scores, topk_idx = torch.topk(scores, self.topk_genes)
                topk_scores = torch.softmax(topk_scores, dim=0)

            # Convert to numpy
            matched_indices = topk_idx.cpu().numpy()
            matched_scores = topk_scores.cpu().numpy()

            # Get matched gene info
            matched_genes = []
            for i, (idx, score) in enumerate(zip(matched_indices, matched_scores)):
                target_id = target_inv[int(idx)]
                symbol = self.readout.get_gene_symbol(target_species, target_id)

                # Compute shared terms
                shared = self.readout.compute_shared(
                    source_species=source_species,
                    source_gene_id=source_idx,
                    target_gene_id=int(idx),
                )

                matched_genes.append({
                    'id': target_id,
                    'symbol': symbol,
                    'score': float(score),
                    'rank': i + 1,
                    'shared_terms': shared,
                })

            # Compute related terms
            related = self.readout.compute_for_matches(
                source_species=source_species,
                matched_indices=matched_indices.tolist(),
                matched_scores=matched_scores,
            )

            # Build result
            source_symbol = self.readout.get_gene_symbol(source_species, source_id)
            result = {
                'source': {
                    'id': source_id,
                    'symbol': source_symbol,
                    'species': source_species,
                },
                'matched_genes': matched_genes,
                'related_gobp': related['bp'],
                'related_gomf': related['mf'],
                'related_diseases': related['disease'],
            }

            results.append(result)

        return results


def load_model_for_inference(
    checkpoint_path: str,
    config: Dict,
    device: str = 'cuda',
) -> Tuple[CrossSpeciesModel, Dict, Dict]:
    """
    Load model for inference.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to use

    Returns:
        (model, sparse_matrices, indexers)
    """
    from .io_schema import load_indexers
    from .build_sparse import load_sparse_matrices
    from .unified_graph import load_unified_graph

    proc_dir = Path(config['paths']['proc_dir'])

    # Load preprocessed data
    indexers = load_indexers(proc_dir / 'indexers.json')
    sparse_matrices = load_sparse_matrices(proc_dir)
    graph_data = load_unified_graph(proc_dir / 'unified_graph.pt')

    # Create model
    node_counts = {
        'Human_Gene': len(indexers['human_genes']),
        'Mouse_Gene': len(indexers['mouse_genes']),
        'BP': len(indexers['bp_terms']),
        'MF': len(indexers['mf_terms']),
        'Disease': len(indexers['diseases']),
    }

    model = create_model(config, node_counts)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")

    return model, graph_data, sparse_matrices, indexers


def run_inference(
    query: str,
    checkpoint_path: str,
    config: Dict,
    direction: str = 'auto',
    device: str = 'cuda',
) -> Dict:
    """
    Run inference on a query.

    Args:
        query: Semicolon-separated gene IDs
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        direction: 'human_to_mouse', 'mouse_to_human', or 'auto'
        device: Device to use

    Returns:
        Result dictionary
    """
    model, graph_data, sparse_matrices, indexers = load_model_for_inference(
        checkpoint_path, config, device
    )

    inferencer = Inferencer(
        model=model,
        graph_data=graph_data,
        sparse_matrices=sparse_matrices,
        indexers=indexers,
        config=config,
        device=device,
    )

    return inferencer.infer(query, direction)
