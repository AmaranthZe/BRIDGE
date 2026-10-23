"""Heterogeneous Graph with cross-species direct edges (v3_crosshop).

Extends v2's unified graph with 3 new forward edge types:
    (Human_Gene, "xBP",  Mouse_Gene) - BP annotation 2-hop similarity
    (Human_Gene, "xMF",  Mouse_Gene) - MF annotation 2-hop similarity
    (Human_Gene, "xD",   Mouse_Gene) - Disease annotation 2-hop similarity

Plus 3 reverse types added by ToUndirected():
    (Mouse_Gene, "rev_xBP", Human_Gene)
    (Mouse_Gene, "rev_xMF", Human_Gene)
    (Mouse_Gene, "rev_xD",  Human_Gene)

Edge weights = TF-IDF inner product (2-hop similarity score).
This allows HGT to pass messages directly between human and mouse genes,
bypassing the GO/Disease node bottleneck.
"""

import logging
from typing import Dict

import numpy as np
import scipy.sparse as sp
import torch

logger = logging.getLogger(__name__)

# Mapping from modality name to sparse matrix keys
_MODALITY_KEYS = {
    "BP":      ("W_hB", "W_mB"),
    "MF":      ("W_hF", "W_mF"),
    "Disease": ("W_hD", "W_mD"),
}

# Edge type names for each modality
_MODALITY_EDGE_NAMES = {
    "BP":      "xBP",
    "MF":      "xMF",
    "Disease": "xD",
}


def compute_cross_species_edges(
    matrices: Dict[str, sp.csr_matrix],
    modality: str,
    top_k: int = 32,
    weight_scale: float = 1.0,
    min_weight_threshold: float = 1e-6,
):
    """Compute top-k cross-species edges for one modality.

    For each human gene, compute its 2-hop similarity to all mouse genes
    via shared annotation terms: P[i,j] = W_h[i,:] @ W_m[j,:].T
    Then keep top-k mouse genes per human gene as direct edges.

    Args:
        matrices: Dict of 6 sparse matrices (W_hB, W_mB, ...).
        modality: "BP", "MF", or "Disease".
        top_k: Number of top mouse genes per human gene.
        weight_scale: Multiply edge weights by this.
        min_weight_threshold: Drop edges with weight < this.

    Returns:
        edge_index: LongTensor [2, E], row=human gene idx, col=mouse gene idx
        edge_attr:  FloatTensor [E], 2-hop similarity weights
    """
    h_key, m_key = _MODALITY_KEYS[modality]
    W_h = matrices[h_key]  # (N_h, N_terms)
    W_m = matrices[m_key]  # (N_m, N_terms)

    # Sparse matmul: P[i,j] = sum_k W_h[i,k] * W_m[j,k]
    # Result shape: (N_h, N_m), sparse CSR — no densification needed
    P = (W_h @ W_m.T).tocsr()

    N_h = P.shape[0]
    rows, cols, vals = [], [], []

    for i in range(N_h):
        start, end = P.indptr[i], P.indptr[i + 1]
        if start == end:
            # No connections for this human gene in this modality
            continue

        seg_cols = P.indices[start:end]
        seg_data = P.data[start:end]

        # Take top-k by value
        if len(seg_data) > top_k:
            topk_local = np.argpartition(seg_data, -top_k)[-top_k:]
            seg_cols = seg_cols[topk_local]
            seg_data = seg_data[topk_local]

        # Apply weight scale and threshold
        seg_data = seg_data * weight_scale
        keep = seg_data >= min_weight_threshold
        if not keep.any():
            continue

        seg_cols = seg_cols[keep]
        seg_data = seg_data[keep]

        rows.extend([i] * len(seg_cols))
        cols.extend(seg_cols.tolist())
        vals.extend(seg_data.tolist())

    if not rows:
        logger.warning(
            f"Modality {modality}: no cross-species edges found "
            f"(check sparse matrices)"
        )
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.float)
    else:
        rows_arr = np.array(rows, dtype=np.int64)
        cols_arr = np.array(cols, dtype=np.int64)
        vals_arr = np.array(vals, dtype=np.float32)
        edge_index = torch.tensor(
            np.vstack([rows_arr, cols_arr]), dtype=torch.long
        )
        edge_attr = torch.tensor(vals_arr, dtype=torch.float)

    n_genes_with_edges = len(set(rows)) if rows else 0
    avg_per_gene = len(rows) / max(n_genes_with_edges, 1)
    logger.info(
        f"  {modality} cross-species edges: {edge_index.shape[1]:,} "
        f"({n_genes_with_edges:,}/{N_h:,} human genes have edges, "
        f"avg {avg_per_gene:.1f} per gene)"
    )
    return edge_index, edge_attr


def build_crosshop_graph(
    matrices: Dict[str, sp.csr_matrix],
    indexers: Dict,
    crosshop_cfg,
):
    """Build unified heterogeneous graph with cross-species direct edges.

    Constructs the full graph from scratch (9 forward edge types):
    - 6 gene-term edges (same as v2)
    - 3 cross-species edges (new in v3)
    Then applies ToUndirected() once to get all 18 edge types.

    NOTE: We do NOT call v2's build_unified_graph() because it internally
    calls ToUndirected(). Calling ToUndirected() twice would create
    duplicate rev_rev_* edge types.

    Args:
        matrices: Dict of 6 sparse matrices from v2 preprocessing.
        indexers: ID mapping dictionaries from v2 preprocessing.
        crosshop_cfg: CrossHopConfig with cross_hop_k, modalities, etc.

    Returns:
        PyG HeteroData with 5 node types and 18 edge types.
    """
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    # Reuse v2's sparse_to_edge_index utility
    from v2_redesign.data.graph import sparse_to_edge_index

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

    # ── Standard gene-term edges (6 forward, same as v2) ──
    v2_edge_specs = [
        ("Human_Gene", "HG-BP", "BP", "W_hB"),
        ("Human_Gene", "HG-MF", "MF", "W_hF"),
        ("Human_Gene", "HG-D", "Disease", "W_hD"),
        ("Mouse_Gene", "MG-BP", "BP", "W_mB"),
        ("Mouse_Gene", "MG-MF", "MF", "W_mF"),
        ("Mouse_Gene", "MG-D", "Disease", "W_mD"),
    ]

    for src, rel, dst, mat_key in v2_edge_specs:
        ei, ea = sparse_to_edge_index(matrices[mat_key])
        data[src, rel, dst].edge_index = ei
        data[src, rel, dst].edge_attr = ea
        logger.info(f"  {rel} edges: {ei.shape[1]:,}")

    # ── Cross-species direct edges (3 new forward types) ──
    logger.info("Computing cross-species edges...")
    for mod in crosshop_cfg.modalities:
        edge_name = _MODALITY_EDGE_NAMES[mod]
        ei, ea = compute_cross_species_edges(
            matrices=matrices,
            modality=mod,
            top_k=crosshop_cfg.cross_hop_k,
            weight_scale=crosshop_cfg.weight_scale,
            min_weight_threshold=crosshop_cfg.min_weight_threshold,
        )
        data["Human_Gene", edge_name, "Mouse_Gene"].edge_index = ei
        data["Human_Gene", edge_name, "Mouse_Gene"].edge_attr = ea

    # ── Add reverse edges for ALL 9 forward types at once ──
    data = T.ToUndirected()(data)

    # Summary
    total = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    n_cross = sum(
        data[et].edge_index.shape[1]
        for et in data.edge_types
        if any(x in str(et) for x in ["xBP", "xMF", "xD"])
    )
    logger.info(
        f"v3 graph: {N_h} human, {N_m} mouse, "
        f"{N_B} BP, {N_F} MF, {N_D} disease nodes"
    )
    logger.info(
        f"Total edges: {total:,} "
        f"({n_cross:,} cross-species, {total - n_cross:,} gene-term)"
    )
    logger.info(f"Edge types ({len(data.edge_types)}): {data.edge_types}")

    return data


def save_crosshop_graph(data, path: str):
    """Save v3 graph to disk."""
    torch.save(data, path)
    logger.info(f"Saved v3 crosshop graph to {path}")


def load_crosshop_graph(path: str):
    """Load v3 graph from disk."""
    return torch.load(path, weights_only=False)
