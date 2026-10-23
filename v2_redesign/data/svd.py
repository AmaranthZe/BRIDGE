"""Unified SVD across species and modalities."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def split_terms(
    n_terms: int,
    heldout_ratio: float = 0.1,
    seed: int = 3407,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split term (column) indices into train and held-out sets."""
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
    Build unified SVD basis across species and modalities.

    Steps:
        1. Split columns per modality into train/heldout.
        2. hstack [BP_train, MF_train, D_train] per species.
        3. vstack across species → X_all_train.
        4. Fit TruncatedSVD on X_all_train.
        5. Transform using train columns → Z_h, Z_m.
        6. Row-L2 normalize Z_h, Z_m.

    Returns:
        (Z_h, Z_m, svd_model, split_info)
    """
    from sklearn.decomposition import TruncatedSVD

    n_bp = matrices["W_hB"].shape[1]
    n_mf = matrices["W_hF"].shape[1]
    n_dis = matrices["W_hD"].shape[1]

    bp_train, bp_held = split_terms(n_bp, heldout_ratio, seed)
    mf_train, mf_held = split_terms(n_mf, heldout_ratio, seed + 1)
    dis_train, dis_held = split_terms(n_dis, heldout_ratio, seed + 2)

    split_info = {
        "bp_train": bp_train, "bp_held": bp_held,
        "mf_train": mf_train, "mf_held": mf_held,
        "dis_train": dis_train, "dis_held": dis_held,
    }

    logger.info(
        f"Column splits — BP: {len(bp_train)}/{len(bp_held)}, "
        f"MF: {len(mf_train)}/{len(mf_held)}, "
        f"DIS: {len(dis_train)}/{len(dis_held)}"
    )

    def _hstack_train(prefix: str) -> sp.csr_matrix:
        return sp.hstack(
            [
                matrices[f"W_{prefix}B"][:, bp_train],
                matrices[f"W_{prefix}F"][:, mf_train],
                matrices[f"W_{prefix}D"][:, dis_train],
            ],
            format="csr",
        )

    X_h_train = _hstack_train("h")
    X_m_train = _hstack_train("m")
    X_all_train = sp.vstack([X_h_train, X_m_train], format="csr")

    logger.info(f"Unified train matrix shape: {X_all_train.shape}")

    actual_dim = min(svd_dim, X_all_train.shape[1] - 1, X_all_train.shape[0] - 1)
    svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
    svd.fit(X_all_train)
    logger.info(f"SVD explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

    # Transform using train columns (same feature space as fit)
    Z_h = svd.transform(X_h_train).astype(np.float32)
    Z_m = svd.transform(X_m_train).astype(np.float32)

    # Row-L2 normalize
    for Z in (Z_h, Z_m):
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Z /= norms

    logger.info(f"Z_h shape: {Z_h.shape}, Z_m shape: {Z_m.shape}")
    return Z_h, Z_m, svd, split_info


def save_svd_artifacts(Z_h, Z_m, svd_model, split_info, output_dir: str):
    """Save unified SVD artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "Z_h.npy", Z_h)
    np.save(out / "Z_m.npy", Z_m)
    with open(out / "svd_unified.pkl", "wb") as f:
        pickle.dump(svd_model, f)
    with open(out / "svd_split_info.pkl", "wb") as f:
        pickle.dump(split_info, f)


def load_svd_artifacts(input_dir: str) -> Tuple[np.ndarray, np.ndarray, object, Dict]:
    """Load unified SVD artifacts."""
    d = Path(input_dir)
    Z_h = np.load(d / "Z_h.npy")
    Z_m = np.load(d / "Z_m.npy")
    with open(d / "svd_unified.pkl", "rb") as f:
        svd_model = pickle.load(f)
    split_info = {}
    sp_path = d / "svd_split_info.pkl"
    if sp_path.exists():
        with open(sp_path, "rb") as f:
            split_info = pickle.load(f)
    return Z_h, Z_m, svd_model, split_info
