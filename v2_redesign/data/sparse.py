"""Sparse matrix construction with TF-IDF weighting and L2 normalization."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def build_sparse_matrix(
    df,
    row_col: str,
    col_col: str,
    row_indexer: Dict[str, int],
    col_indexer: Dict[str, int],
    value_col: Optional[str] = None,
) -> sp.csr_matrix:
    """Build sparse CSR matrix from dataframe."""
    n_rows = len(row_indexer)
    n_cols = len(col_indexer)

    valid = df[row_col].isin(row_indexer) & df[col_col].isin(col_indexer)
    df_valid = df[valid]

    if len(df_valid) == 0:
        logger.warning("No valid entries after filtering with indexers")
        return sp.csr_matrix((n_rows, n_cols), dtype=np.float32)

    row_indices = df_valid[row_col].map(row_indexer).values
    col_indices = df_valid[col_col].map(col_indexer).values
    values = (
        df_valid[value_col].values.astype(np.float32)
        if value_col is not None
        else np.ones(len(df_valid), dtype=np.float32)
    )

    W = sp.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    ).tocsr()
    W.sum_duplicates()

    logger.info(f"Built sparse matrix: shape={W.shape}, nnz={W.nnz}")
    return W


def apply_tfidf(W: sp.csr_matrix) -> sp.csr_matrix:
    """Apply TF-IDF weighting: IDF = log(N / (1 + df))."""
    if W.nnz == 0:
        return W
    N = W.shape[0]
    df = np.array((W > 0).sum(axis=0)).flatten()
    idf = np.log(N / (1 + df))
    return W @ sp.diags(idf, format="csr")


def row_l2_normalize(W: sp.csr_matrix) -> sp.csr_matrix:
    """Row-wise L2 normalization."""
    if W.nnz == 0:
        return W
    row_norms = np.sqrt(np.array(W.power(2).sum(axis=1)).flatten())
    row_norms[row_norms == 0] = 1.0
    return sp.diags(1.0 / row_norms, format="csr") @ W


def winsorize_scores(W: sp.csr_matrix, q: float = 0.995) -> sp.csr_matrix:
    """Winsorize extreme values and normalize to [0, 1]."""
    if W.nnz == 0:
        return W
    data = W.data.copy()
    upper = np.percentile(data, q * 100)
    data = np.minimum(data, upper)
    if data.max() > 0:
        data = data / data.max()
    W_out = W.copy()
    W_out.data = data.astype(np.float32)
    return W_out


def drop_go_roots(
    W: sp.csr_matrix,
    col_indexer: Dict[str, int],
    roots: List[str],
) -> Tuple[sp.csr_matrix, Dict[str, int]]:
    """Remove GO root columns from sparse matrix."""
    root_indices = [col_indexer[r] for r in roots if r in col_indexer]
    if not root_indices:
        return W, col_indexer

    keep_mask = np.ones(W.shape[1], dtype=bool)
    keep_mask[root_indices] = False
    W_filtered = W[:, keep_mask]

    new_indexer = {}
    new_idx = 0
    for term, old_idx in sorted(col_indexer.items(), key=lambda x: x[1]):
        if old_idx not in root_indices:
            new_indexer[term] = new_idx
            new_idx += 1

    logger.info(f"Removed {len(root_indices)} GO root columns")
    return W_filtered.tocsr(), new_indexer


def build_all_sparse_matrices(
    dataframes: Dict,
    indexers: Dict,
    cfg,
) -> Dict[str, sp.csr_matrix]:
    """Build all 6 sparse matrices (W_hB, W_hF, W_hD, W_mB, W_mF, W_mD)."""
    apply_l2 = cfg.preproc.row_l2
    disease_q = cfg.preproc.disease_winsor_q

    specs = [
        ("W_hB", "human_gobp", "human_genes", "bp_terms", None),
        ("W_hF", "human_gomf", "human_genes", "mf_terms", None),
        ("W_hD", "human_disease", "human_genes", "diseases", "score"),
        ("W_mB", "mouse_gobp", "mouse_genes", "bp_terms", None),
        ("W_mF", "mouse_gomf", "mouse_genes", "mf_terms", None),
        ("W_mD", "mouse_disease", "mouse_genes", "diseases", "score"),
    ]

    matrices = {}
    for name, df_key, row_key, col_key, val_col in specs:
        logger.info(f"Building {name}")
        W = build_sparse_matrix(
            dataframes[df_key],
            row_col="gene_id",
            col_col="go_term" if "go" in df_key else "disease_id",
            row_indexer=indexers[row_key],
            col_indexer=indexers[col_key],
            value_col=val_col,
        )
        if val_col == "score":
            W = winsorize_scores(W, q=disease_q)
        W = apply_tfidf(W)
        if apply_l2:
            W = row_l2_normalize(W)
        matrices[name] = W

    for name, W in matrices.items():
        density = W.nnz / max(W.shape[0] * W.shape[1], 1)
        logger.info(f"{name}: shape={W.shape}, nnz={W.nnz}, density={density:.6f}")

    return matrices


def save_sparse_matrices(matrices: Dict[str, sp.csr_matrix], output_dir: str):
    """Save sparse matrices to NPZ files."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, W in matrices.items():
        sp.save_npz(out / f"{name}.npz", W)


def load_sparse_matrices(input_dir: str) -> Dict[str, sp.csr_matrix]:
    """Load sparse matrices from NPZ files."""
    from pathlib import Path
    d = Path(input_dir)
    return {
        name: sp.load_npz(d / f"{name}.npz")
        for name in ["W_hB", "W_hF", "W_hD", "W_mB", "W_mF", "W_mD"]
    }
