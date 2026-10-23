"""
Sparse matrix construction with TF-IDF weighting and L2 normalization.

This module handles:
- Building CSR sparse matrices from gene-term/disease pairs
- TF-IDF column weighting
- Row L2 normalization
- Disease score winsorization
- GO root node removal
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.stats import mstats

logger = logging.getLogger(__name__)


def build_sparse_matrix(
    df,
    row_col: str,
    col_col: str,
    row_indexer: Dict[str, int],
    col_indexer: Dict[str, int],
    value_col: Optional[str] = None
) -> sp.csr_matrix:
    """
    Build sparse CSR matrix from dataframe.

    Args:
        df: DataFrame with gene-term/disease pairs
        row_col: Column name for row entities (genes)
        col_col: Column name for column entities (terms/diseases)
        row_indexer: Mapping from row entity ID to index
        col_indexer: Mapping from column entity ID to index
        value_col: Optional column for values (default: binary 1.0)

    Returns:
        CSR sparse matrix
    """
    n_rows = len(row_indexer)
    n_cols = len(col_indexer)

    # Filter to only include entities in indexers
    valid_rows = df[row_col].isin(row_indexer)
    valid_cols = df[col_col].isin(col_indexer)
    df_valid = df[valid_rows & valid_cols].copy()

    if len(df_valid) == 0:
        logger.warning("No valid entries after filtering with indexers")
        return sp.csr_matrix((n_rows, n_cols), dtype=np.float32)

    # Map to indices
    row_indices = df_valid[row_col].map(row_indexer).values
    col_indices = df_valid[col_col].map(col_indexer).values

    # Get values
    if value_col is not None:
        values = df_valid[value_col].values.astype(np.float32)
    else:
        values = np.ones(len(df_valid), dtype=np.float32)

    # Build COO matrix and convert to CSR
    W = sp.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_rows, n_cols),
        dtype=np.float32
    ).tocsr()

    # Handle duplicate entries (sum them)
    W.sum_duplicates()

    logger.info(f"Built sparse matrix: shape={W.shape}, nnz={W.nnz}")
    return W


def apply_tfidf(W: sp.csr_matrix) -> sp.csr_matrix:
    """
    Apply TF-IDF weighting to sparse matrix.

    TF (term frequency) = raw count (already in matrix)
    IDF (inverse document frequency) = log(N / (1 + df)) where df = document frequency

    Args:
        W: Sparse matrix (genes x terms)

    Returns:
        TF-IDF weighted sparse matrix
    """
    if W.nnz == 0:
        return W

    N = W.shape[0]  # Number of genes (documents)

    # Compute document frequency for each term (column)
    # df[j] = number of genes that have term j
    df = np.array((W > 0).sum(axis=0)).flatten()

    # Compute IDF: log(N / (1 + df))
    idf = np.log(N / (1 + df))

    # Apply IDF weighting (multiply each column by its IDF)
    # Create diagonal matrix of IDF values
    idf_diag = sp.diags(idf, format='csr')

    # W_tfidf = W @ diag(idf)
    W_tfidf = W @ idf_diag

    logger.info(f"Applied TF-IDF weighting")
    return W_tfidf


def row_l2_normalize(W: sp.csr_matrix) -> sp.csr_matrix:
    """
    Apply row-wise L2 normalization.

    Args:
        W: Sparse matrix

    Returns:
        Row-normalized sparse matrix
    """
    if W.nnz == 0:
        return W

    # Compute row norms
    row_norms = np.sqrt(np.array(W.power(2).sum(axis=1)).flatten())

    # Avoid division by zero
    row_norms[row_norms == 0] = 1.0

    # Create diagonal matrix of inverse norms
    inv_norms = 1.0 / row_norms
    inv_norm_diag = sp.diags(inv_norms, format='csr')

    # W_normalized = diag(1/norm) @ W
    W_normalized = inv_norm_diag @ W

    logger.info(f"Applied row L2 normalization")
    return W_normalized


def drop_go_roots(
    W: sp.csr_matrix,
    col_indexer: Dict[str, int],
    roots: List[str]
) -> Tuple[sp.csr_matrix, Dict[str, int]]:
    """
    Remove GO root columns from sparse matrix.

    Note: The indexers should already exclude roots, so this is a safety check.

    Args:
        W: Sparse matrix
        col_indexer: Term to index mapping
        roots: List of root term IDs to remove

    Returns:
        Tuple of (filtered matrix, updated indexer)
    """
    root_indices = [col_indexer[r] for r in roots if r in col_indexer]

    if not root_indices:
        logger.info("No GO roots found in indexer (already removed)")
        return W, col_indexer

    # Create mask for columns to keep
    keep_mask = np.ones(W.shape[1], dtype=bool)
    keep_mask[root_indices] = False

    # Filter columns
    W_filtered = W[:, keep_mask]

    # Update indexer
    new_indexer = {}
    new_idx = 0
    for term, old_idx in sorted(col_indexer.items(), key=lambda x: x[1]):
        if old_idx not in root_indices:
            new_indexer[term] = new_idx
            new_idx += 1

    logger.info(f"Removed {len(root_indices)} GO root columns")
    return W_filtered.tocsr(), new_indexer


def winsorize_scores(W: sp.csr_matrix, q: float = 0.995) -> sp.csr_matrix:
    """
    Winsorize extreme values in sparse matrix.

    Args:
        W: Sparse matrix with score values
        q: Quantile for upper cap (default: 0.995)

    Returns:
        Winsorized sparse matrix
    """
    if W.nnz == 0:
        return W

    # Get non-zero values
    data = W.data.copy()

    # Compute upper threshold
    upper = np.percentile(data, q * 100)

    # Cap values
    data = np.minimum(data, upper)

    # Normalize to [0, 1]
    if data.max() > 0:
        data = data / data.max()

    # Create new matrix with winsorized values
    W_wins = W.copy()
    W_wins.data = data.astype(np.float32)

    logger.info(f"Winsorized scores at q={q}, upper={upper:.4f}")
    return W_wins


def build_all_sparse_matrices(
    dataframes: Dict,
    indexers: Dict,
    config: Dict
) -> Dict[str, sp.csr_matrix]:
    """
    Build all 6 sparse matrices from dataframes.

    Args:
        dataframes: Dictionary of loaded dataframes
        indexers: Dictionary of ID indexers
        config: Configuration dictionary

    Returns:
        Dictionary of sparse matrices:
        - W_hB: Human gene x BP
        - W_hF: Human gene x MF
        - W_hD: Human gene x Disease
        - W_mB: Mouse gene x BP
        - W_mF: Mouse gene x MF
        - W_mD: Mouse gene x Disease
    """
    preproc_cfg = config['preproc']
    go_roots = preproc_cfg.get('go_roots', [])
    disease_winsor_q = preproc_cfg.get('disease_winsor_q', 0.995)
    apply_l2 = preproc_cfg.get('row_l2', True)

    matrices = {}

    # Human GOBP
    logger.info("Building W_hB (Human gene x BP)")
    W_hB = build_sparse_matrix(
        dataframes['human_gobp'],
        row_col='gene_id',
        col_col='go_term',
        row_indexer=indexers['human_genes'],
        col_indexer=indexers['bp_terms']
    )
    W_hB = apply_tfidf(W_hB)
    if apply_l2:
        W_hB = row_l2_normalize(W_hB)
    matrices['W_hB'] = W_hB

    # Human GOMF
    logger.info("Building W_hF (Human gene x MF)")
    W_hF = build_sparse_matrix(
        dataframes['human_gomf'],
        row_col='gene_id',
        col_col='go_term',
        row_indexer=indexers['human_genes'],
        col_indexer=indexers['mf_terms']
    )
    W_hF = apply_tfidf(W_hF)
    if apply_l2:
        W_hF = row_l2_normalize(W_hF)
    matrices['W_hF'] = W_hF

    # Human Disease
    logger.info("Building W_hD (Human gene x Disease)")
    W_hD = build_sparse_matrix(
        dataframes['human_disease'],
        row_col='gene_id',
        col_col='disease_id',
        row_indexer=indexers['human_genes'],
        col_indexer=indexers['diseases'],
        value_col='score'
    )
    W_hD = winsorize_scores(W_hD, q=disease_winsor_q)
    W_hD = apply_tfidf(W_hD)
    if apply_l2:
        W_hD = row_l2_normalize(W_hD)
    matrices['W_hD'] = W_hD

    # Mouse GOBP
    logger.info("Building W_mB (Mouse gene x BP)")
    W_mB = build_sparse_matrix(
        dataframes['mouse_gobp'],
        row_col='gene_id',
        col_col='go_term',
        row_indexer=indexers['mouse_genes'],
        col_indexer=indexers['bp_terms']
    )
    W_mB = apply_tfidf(W_mB)
    if apply_l2:
        W_mB = row_l2_normalize(W_mB)
    matrices['W_mB'] = W_mB

    # Mouse GOMF
    logger.info("Building W_mF (Mouse gene x MF)")
    W_mF = build_sparse_matrix(
        dataframes['mouse_gomf'],
        row_col='gene_id',
        col_col='go_term',
        row_indexer=indexers['mouse_genes'],
        col_indexer=indexers['mf_terms']
    )
    W_mF = apply_tfidf(W_mF)
    if apply_l2:
        W_mF = row_l2_normalize(W_mF)
    matrices['W_mF'] = W_mF

    # Mouse Disease
    logger.info("Building W_mD (Mouse gene x Disease)")
    W_mD = build_sparse_matrix(
        dataframes['mouse_disease'],
        row_col='gene_id',
        col_col='disease_id',
        row_indexer=indexers['mouse_genes'],
        col_indexer=indexers['diseases'],
        value_col='score'
    )
    W_mD = winsorize_scores(W_mD, q=disease_winsor_q)
    W_mD = apply_tfidf(W_mD)
    if apply_l2:
        W_mD = row_l2_normalize(W_mD)
    matrices['W_mD'] = W_mD

    # Log statistics
    for name, W in matrices.items():
        logger.info(f"{name}: shape={W.shape}, nnz={W.nnz}, "
                    f"density={W.nnz / (W.shape[0] * W.shape[1]):.6f}")

    return matrices


def save_sparse_matrices(matrices: Dict[str, sp.csr_matrix], output_dir: str):
    """Save sparse matrices to NPZ files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, W in matrices.items():
        path = output_path / f"{name}.npz"
        sp.save_npz(path, W)
        logger.info(f"Saved {name} to {path}")


def load_sparse_matrices(input_dir: str) -> Dict[str, sp.csr_matrix]:
    """Load sparse matrices from NPZ files."""
    input_path = Path(input_dir)

    matrices = {}
    for name in ['W_hB', 'W_hF', 'W_hD', 'W_mB', 'W_mF', 'W_mD']:
        path = input_path / f"{name}.npz"
        if path.exists():
            matrices[name] = sp.load_npz(path)
            logger.info(f"Loaded {name} from {path}")
        else:
            raise FileNotFoundError(f"Matrix file not found: {path}")

    return matrices


# ── A-fix #4: Unified SVD across species and modules ──────────────────────


def split_terms(
    n_terms: int,
    heldout_ratio: float = 0.1,
    seed: int = 3407,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split term (column) indices into train and held-out sets.

    Args:
        n_terms: Total number of terms
        heldout_ratio: Fraction to hold out
        seed: Random seed

    Returns:
        (train_idx, heldout_idx) — sorted arrays of column indices
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_terms)
    n_held = max(1, int(n_terms * heldout_ratio))
    heldout = np.sort(perm[:n_held])
    train = np.sort(perm[n_held:])
    return train, heldout


def build_unified_svd(
    matrices: Dict[str, sp.csr_matrix],
    svd_dim: int = 128,
    heldout_ratio: float = 0.1,
    seed: int = 3407,
) -> Tuple[np.ndarray, np.ndarray, object, Dict]:
    """
    Build unified SVD basis across species and modules (A-fix #4).

    Steps:
        1. Split columns per modality into train/heldout.
        2. hstack [BP_train, MF_train, D_train] per species → X_h_train, X_m_train.
        3. vstack across species → X_all_train.
        4. Fit TruncatedSVD on X_all_train.
        5. Transform full (all columns) hstacked matrices → Z_h, Z_m.
        6. Row-L2 normalize Z_h, Z_m.

    Args:
        matrices: Dict with W_hB, W_hF, W_hD, W_mB, W_mF, W_mD
        svd_dim: Number of SVD components
        heldout_ratio: Column heldout fraction per modality
        seed: Random seed

    Returns:
        (Z_h, Z_m, svd_model, split_info)
        - Z_h: Human gene features [N_h, svd_dim] (numpy float32, row-L2)
        - Z_m: Mouse gene features [N_m, svd_dim] (numpy float32, row-L2)
        - svd_model: Fitted TruncatedSVD object
        - split_info: Dict with train/heldout indices per modality
    """
    from sklearn.decomposition import TruncatedSVD

    # Split columns per modality
    n_bp = matrices['W_hB'].shape[1]
    n_mf = matrices['W_hF'].shape[1]
    n_dis = matrices['W_hD'].shape[1]

    bp_train, bp_held = split_terms(n_bp, heldout_ratio, seed)
    mf_train, mf_held = split_terms(n_mf, heldout_ratio, seed + 1)
    dis_train, dis_held = split_terms(n_dis, heldout_ratio, seed + 2)

    split_info = {
        'bp_train': bp_train, 'bp_held': bp_held,
        'mf_train': mf_train, 'mf_held': mf_held,
        'dis_train': dis_train, 'dis_held': dis_held,
    }

    logger.info(f"Column splits — BP: {len(bp_train)}/{len(bp_held)}, "
                f"MF: {len(mf_train)}/{len(mf_held)}, "
                f"DIS: {len(dis_train)}/{len(dis_held)}")

    # hstack train columns per species
    X_h_train = sp.hstack([
        matrices['W_hB'][:, bp_train],
        matrices['W_hF'][:, mf_train],
        matrices['W_hD'][:, dis_train],
    ], format='csr')

    X_m_train = sp.hstack([
        matrices['W_mB'][:, bp_train],
        matrices['W_mF'][:, mf_train],
        matrices['W_mD'][:, dis_train],
    ], format='csr')

    # vstack across species for unified SVD
    X_all_train = sp.vstack([X_h_train, X_m_train], format='csr')
    logger.info(f"Unified train matrix shape: {X_all_train.shape}")

    # Fit SVD on train columns only
    actual_dim = min(svd_dim, X_all_train.shape[1] - 1, X_all_train.shape[0] - 1)
    svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
    svd.fit(X_all_train)
    logger.info(f"SVD explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

    # Transform using ALL columns (train+heldout) through the fitted SVD.
    # To do this correctly, we hstack ALL columns in the same order as training
    # and use svd.transform on the train-column subset (same feature space as fit).
    # This is the semantically correct approach: SVD basis is defined by train columns,
    # and we project all genes into that basis using the same train columns.
    X_h_for_transform = sp.hstack([
        matrices['W_hB'][:, bp_train],
        matrices['W_hF'][:, mf_train],
        matrices['W_hD'][:, dis_train],
    ], format='csr')
    X_m_for_transform = sp.hstack([
        matrices['W_mB'][:, bp_train],
        matrices['W_mF'][:, mf_train],
        matrices['W_mD'][:, dis_train],
    ], format='csr')

    Z_h = svd.transform(X_h_for_transform).astype(np.float32)
    Z_m = svd.transform(X_m_for_transform).astype(np.float32)

    # Row-L2 normalize
    norms_h = np.linalg.norm(Z_h, axis=1, keepdims=True)
    norms_h[norms_h == 0] = 1.0
    Z_h = Z_h / norms_h

    norms_m = np.linalg.norm(Z_m, axis=1, keepdims=True)
    norms_m[norms_m == 0] = 1.0
    Z_m = Z_m / norms_m

    logger.info(f"Z_h shape: {Z_h.shape}, Z_m shape: {Z_m.shape}")

    return Z_h, Z_m, svd, split_info


def save_svd_artifacts(
    Z_h: np.ndarray,
    Z_m: np.ndarray,
    svd_model: object,
    split_info: Dict,
    output_dir: str,
):
    """Save unified SVD artifacts."""
    import pickle
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / 'Z_h.npy', Z_h)
    np.save(output_path / 'Z_m.npy', Z_m)

    with open(output_path / 'svd_unified.pkl', 'wb') as f:
        pickle.dump(svd_model, f)

    with open(output_path / 'svd_split_info.pkl', 'wb') as f:
        pickle.dump(split_info, f)

    logger.info(f"Saved SVD artifacts to {output_path}")


def load_svd_artifacts(input_dir: str) -> Tuple[np.ndarray, np.ndarray, object, Dict]:
    """Load unified SVD artifacts."""
    import pickle
    input_path = Path(input_dir)

    Z_h = np.load(input_path / 'Z_h.npy')
    Z_m = np.load(input_path / 'Z_m.npy')

    with open(input_path / 'svd_unified.pkl', 'rb') as f:
        svd_model = pickle.load(f)

    split_info = {}
    split_path = input_path / 'svd_split_info.pkl'
    if split_path.exists():
        with open(split_path, 'rb') as f:
            split_info = pickle.load(f)

    logger.info(f"Loaded SVD artifacts: Z_h={Z_h.shape}, Z_m={Z_m.shape}")
    return Z_h, Z_m, svd_model, split_info
