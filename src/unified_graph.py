"""
Unified Heterogeneous Graph Construction.

This module builds a single PyG HeteroData graph where:
- Node types: Human_Gene, Mouse_Gene, BP, MF, Disease
- Edge types: HG-BP, HG-MF, HG-D, MG-BP, MG-MF, MG-D

GO/Disease nodes are SHARED across species - this is the key design feature
that enables natural embedding alignment.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def sparse_to_edge_index(W: sp.csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert CSR sparse matrix to PyG edge_index and edge_attr.

    Args:
        W: CSR sparse matrix

    Returns:
        Tuple of (edge_index [2, E], edge_attr [E])
    """
    coo = W.tocoo()
    edge_index = torch.tensor(
        np.vstack([coo.row, coo.col]),
        dtype=torch.long
    )
    edge_attr = torch.tensor(coo.data, dtype=torch.float)
    return edge_index, edge_attr


def build_unified_graph(
    matrices: Dict[str, sp.csr_matrix],
    indexers: Dict
) -> 'HeteroData':
    """
    Build unified heterogeneous graph with shared GO/Disease nodes.

    Args:
        matrices: Dictionary of sparse matrices (W_hB, W_hF, W_hD, W_mB, W_mF, W_mD)
        indexers: Dictionary of ID indexers

    Returns:
        PyG HeteroData object with:
        - 5 node types: Human_Gene, Mouse_Gene, BP, MF, Disease
        - 6 edge types (+ 6 reverse): HG-BP, HG-MF, HG-D, MG-BP, MG-MF, MG-D
    """
    # Import here to allow module to load without torch_geometric
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    data = HeteroData()

    # Node counts
    N_h = len(indexers['human_genes'])
    N_m = len(indexers['mouse_genes'])
    N_B = len(indexers['bp_terms'])
    N_F = len(indexers['mf_terms'])
    N_D = len(indexers['diseases'])

    logger.info(f"Building unified graph with:")
    logger.info(f"  Human genes: {N_h}")
    logger.info(f"  Mouse genes: {N_m}")
    logger.info(f"  BP terms (shared): {N_B}")
    logger.info(f"  MF terms (shared): {N_F}")
    logger.info(f"  Diseases (shared): {N_D}")

    # Set node counts (actual embeddings will be created by the model)
    data['Human_Gene'].num_nodes = N_h
    data['Mouse_Gene'].num_nodes = N_m
    data['BP'].num_nodes = N_B
    data['MF'].num_nodes = N_F
    data['Disease'].num_nodes = N_D

    # Human Gene → BP edges
    ei, ea = sparse_to_edge_index(matrices['W_hB'])
    data['Human_Gene', 'HG-BP', 'BP'].edge_index = ei
    data['Human_Gene', 'HG-BP', 'BP'].edge_attr = ea
    logger.info(f"  HG-BP edges: {ei.shape[1]}")

    # Human Gene → MF edges
    ei, ea = sparse_to_edge_index(matrices['W_hF'])
    data['Human_Gene', 'HG-MF', 'MF'].edge_index = ei
    data['Human_Gene', 'HG-MF', 'MF'].edge_attr = ea
    logger.info(f"  HG-MF edges: {ei.shape[1]}")

    # Human Gene → Disease edges
    ei, ea = sparse_to_edge_index(matrices['W_hD'])
    data['Human_Gene', 'HG-D', 'Disease'].edge_index = ei
    data['Human_Gene', 'HG-D', 'Disease'].edge_attr = ea
    logger.info(f"  HG-D edges: {ei.shape[1]}")

    # Mouse Gene → BP edges (connecting to SAME BP nodes!)
    ei, ea = sparse_to_edge_index(matrices['W_mB'])
    data['Mouse_Gene', 'MG-BP', 'BP'].edge_index = ei
    data['Mouse_Gene', 'MG-BP', 'BP'].edge_attr = ea
    logger.info(f"  MG-BP edges: {ei.shape[1]}")

    # Mouse Gene → MF edges (connecting to SAME MF nodes!)
    ei, ea = sparse_to_edge_index(matrices['W_mF'])
    data['Mouse_Gene', 'MG-MF', 'MF'].edge_index = ei
    data['Mouse_Gene', 'MG-MF', 'MF'].edge_attr = ea
    logger.info(f"  MG-MF edges: {ei.shape[1]}")

    # Mouse Gene → Disease edges (connecting to SAME Disease nodes!)
    ei, ea = sparse_to_edge_index(matrices['W_mD'])
    data['Mouse_Gene', 'MG-D', 'Disease'].edge_index = ei
    data['Mouse_Gene', 'MG-D', 'Disease'].edge_attr = ea
    logger.info(f"  MG-D edges: {ei.shape[1]}")

    # Add reverse edges for bidirectional message passing
    data = T.ToUndirected()(data)

    # Count total edges
    total_edges = sum(
        data[edge_type].edge_index.shape[1]
        for edge_type in data.edge_types
    )
    logger.info(f"Total edges (including reverse): {total_edges}")

    return data


def save_unified_graph(data: 'HeteroData', path: str):
    """Save unified graph to file."""
    torch.save(data, path)
    logger.info(f"Saved unified graph to {path}")


def load_unified_graph(path: str) -> 'HeteroData':
    """Load unified graph from file."""
    data = torch.load(path, weights_only=False)
    logger.info(f"Loaded unified graph from {path}")
    return data


def extract_subgraph(
    data: 'HeteroData',
    human_gene_indices: Optional[torch.Tensor] = None,
    mouse_gene_indices: Optional[torch.Tensor] = None,
    k_hops: int = 1
) -> 'HeteroData':
    """
    Extract k-hop subgraph around specified gene nodes.

    Args:
        data: Full unified graph
        human_gene_indices: Indices of human genes to include
        mouse_gene_indices: Indices of mouse genes to include
        k_hops: Number of hops to expand

    Returns:
        Subgraph HeteroData
    """
    from torch_geometric.data import HeteroData

    # Start with seed nodes
    node_mask = {
        'Human_Gene': torch.zeros(data['Human_Gene'].num_nodes, dtype=torch.bool),
        'Mouse_Gene': torch.zeros(data['Mouse_Gene'].num_nodes, dtype=torch.bool),
        'BP': torch.zeros(data['BP'].num_nodes, dtype=torch.bool),
        'MF': torch.zeros(data['MF'].num_nodes, dtype=torch.bool),
        'Disease': torch.zeros(data['Disease'].num_nodes, dtype=torch.bool),
    }

    if human_gene_indices is not None:
        node_mask['Human_Gene'][human_gene_indices] = True
    if mouse_gene_indices is not None:
        node_mask['Mouse_Gene'][mouse_gene_indices] = True

    # Expand k hops
    for _ in range(k_hops):
        new_mask = {k: v.clone() for k, v in node_mask.items()}

        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index

            # Find edges where source is in mask
            src_in_mask = node_mask[src_type][edge_index[0]]
            # Add destination nodes to mask
            new_mask[dst_type][edge_index[1, src_in_mask]] = True

            # Also check reverse direction
            dst_in_mask = node_mask[dst_type][edge_index[1]]
            new_mask[src_type][edge_index[0, dst_in_mask]] = True

        node_mask = new_mask

    # Build subgraph
    subgraph = HeteroData()

    # Node mapping (old index -> new index)
    node_mapping = {}
    for node_type in node_mask:
        mask = node_mask[node_type]
        old_indices = mask.nonzero(as_tuple=True)[0]
        node_mapping[node_type] = {
            'mask': mask,
            'old_to_new': {int(old): new for new, old in enumerate(old_indices.tolist())},
            'new_to_old': old_indices
        }
        subgraph[node_type].num_nodes = int(mask.sum())

    # Remap edges
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        edge_attr = data[edge_type].edge_attr if hasattr(data[edge_type], 'edge_attr') else None

        # Find edges where both endpoints are in subgraph
        src_mask = node_mask[src_type][edge_index[0]]
        dst_mask = node_mask[dst_type][edge_index[1]]
        edge_mask = src_mask & dst_mask

        if edge_mask.sum() > 0:
            # Remap indices
            src_old = edge_index[0, edge_mask]
            dst_old = edge_index[1, edge_mask]

            src_new = torch.tensor([
                node_mapping[src_type]['old_to_new'][int(i)]
                for i in src_old.tolist()
            ], dtype=torch.long)
            dst_new = torch.tensor([
                node_mapping[dst_type]['old_to_new'][int(i)]
                for i in dst_old.tolist()
            ], dtype=torch.long)

            subgraph[edge_type].edge_index = torch.stack([src_new, dst_new])

            if edge_attr is not None:
                subgraph[edge_type].edge_attr = edge_attr[edge_mask]

    # Store mapping for later use
    subgraph.node_mapping = node_mapping

    return subgraph


def get_shared_term_nodes(data: 'HeteroData') -> Dict[str, int]:
    """
    Get counts of shared term nodes (BP, MF, Disease).

    These are the nodes that bridge human and mouse genes.
    """
    return {
        'BP': data['BP'].num_nodes,
        'MF': data['MF'].num_nodes,
        'Disease': data['Disease'].num_nodes,
    }


def compute_node_degrees(data: 'HeteroData') -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute in-degree and out-degree for all node types.

    Returns:
        Dictionary mapping node_type -> {'in': degrees, 'out': degrees}
    """
    degrees = {}

    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        in_deg = torch.zeros(num_nodes, dtype=torch.long)
        out_deg = torch.zeros(num_nodes, dtype=torch.long)

        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index

            if dst_type == node_type:
                # Count incoming edges
                in_deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))

            if src_type == node_type:
                # Count outgoing edges
                out_deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))

        degrees[node_type] = {'in': in_deg, 'out': out_deg}

    return degrees
