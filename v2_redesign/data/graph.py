"""Unified Heterogeneous Graph Construction.

Node types: Human_Gene, Mouse_Gene, BP, MF, Disease
GO/Disease nodes are SHARED across species — this enables natural embedding alignment.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch

logger = logging.getLogger(__name__)


def sparse_to_edge_index(W: sp.csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert CSR sparse matrix to PyG edge_index and edge_attr."""
    coo = W.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    edge_attr = torch.tensor(coo.data, dtype=torch.float)
    return edge_index, edge_attr


def build_unified_graph(
    matrices: Dict[str, sp.csr_matrix],
    indexers: Dict,
) -> "HeteroData":
    """Build unified heterogeneous graph with shared GO/Disease nodes."""
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    data = HeteroData()

    # Node counts
    N_h = len(indexers["human_genes"])
    N_m = len(indexers["mouse_genes"])
    N_B = len(indexers["bp_terms"])
    N_F = len(indexers["mf_terms"])
    N_D = len(indexers["diseases"])

    data["Human_Gene"].num_nodes = N_h
    data["Mouse_Gene"].num_nodes = N_m
    data["BP"].num_nodes = N_B
    data["MF"].num_nodes = N_F
    data["Disease"].num_nodes = N_D

    edge_specs = [
        ("Human_Gene", "HG-BP", "BP", "W_hB"),
        ("Human_Gene", "HG-MF", "MF", "W_hF"),
        ("Human_Gene", "HG-D", "Disease", "W_hD"),
        ("Mouse_Gene", "MG-BP", "BP", "W_mB"),
        ("Mouse_Gene", "MG-MF", "MF", "W_mF"),
        ("Mouse_Gene", "MG-D", "Disease", "W_mD"),
    ]

    for src, rel, dst, mat_key in edge_specs:
        ei, ea = sparse_to_edge_index(matrices[mat_key])
        data[src, rel, dst].edge_index = ei
        data[src, rel, dst].edge_attr = ea
        logger.info(f"  {rel} edges: {ei.shape[1]}")

    # Add reverse edges for bidirectional message passing
    data = T.ToUndirected()(data)

    total = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    logger.info(
        f"Unified graph: {N_h} human, {N_m} mouse, "
        f"{N_B} BP, {N_F} MF, {N_D} disease nodes, {total} edges"
    )
    return data


def save_unified_graph(data: "HeteroData", path: str):
    torch.save(data, path)


def load_unified_graph(path: str) -> "HeteroData":
    return torch.load(path, weights_only=False)
