"""
Test A-fix #4: Six-matrix preprocessing naming & unified SVD.

Builds six toy matrices, splits columns, fits unified SVD on train columns,
and verifies Z_h / Z_m have correct shapes and are row-L2 normalized.
"""

import numpy as np
import scipy.sparse as sp
from src.build_sparse import build_unified_svd, split_terms


def _make_toy_matrices():
    """Create toy sparse matrices for testing."""
    N_h, N_m = 10, 8
    N_bp, N_mf, N_dis = 20, 15, 12

    rng = np.random.RandomState(42)

    def random_sparse(rows, cols, density=0.3):
        data = rng.rand(int(rows * cols * density)).astype(np.float32)
        row_idx = rng.randint(0, rows, len(data))
        col_idx = rng.randint(0, cols, len(data))
        return sp.csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols))

    return {
        'W_hB': random_sparse(N_h, N_bp),
        'W_hF': random_sparse(N_h, N_mf),
        'W_hD': random_sparse(N_h, N_dis),
        'W_mB': random_sparse(N_m, N_bp),
        'W_mF': random_sparse(N_m, N_mf),
        'W_mD': random_sparse(N_m, N_dis),
    }


def test_six_matrices_naming():
    """Verify all six matrices are present with correct names."""
    matrices = _make_toy_matrices()
    expected = {'W_hB', 'W_hF', 'W_hD', 'W_mB', 'W_mF', 'W_mD'}
    assert set(matrices.keys()) == expected


def test_split_terms():
    """Column split produces non-overlapping train/heldout."""
    n = 50
    train, held = split_terms(n, heldout_ratio=0.2, seed=42)

    assert len(train) + len(held) == n
    assert len(set(train) & set(held)) == 0, "Train and heldout overlap!"
    assert len(held) >= 1


def test_unified_svd_shapes():
    """Unified SVD produces Z_h, Z_m with correct shapes."""
    matrices = _make_toy_matrices()
    svd_dim = 8

    Z_h, Z_m, svd_model, split_info = build_unified_svd(
        matrices, svd_dim=svd_dim, heldout_ratio=0.1, seed=42
    )

    N_h = matrices['W_hB'].shape[0]
    N_m = matrices['W_mB'].shape[0]

    assert Z_h.shape[0] == N_h, f"Z_h rows: {Z_h.shape[0]} != {N_h}"
    assert Z_m.shape[0] == N_m, f"Z_m rows: {Z_m.shape[0]} != {N_m}"
    assert Z_h.shape[1] == Z_m.shape[1], "Z_h and Z_m must have same dim"
    assert Z_h.shape[1] <= svd_dim


def test_unified_svd_row_normalized():
    """Z_h and Z_m should be row-L2 normalized."""
    matrices = _make_toy_matrices()
    Z_h, Z_m, _, _ = build_unified_svd(matrices, svd_dim=8, seed=42)

    norms_h = np.linalg.norm(Z_h, axis=1)
    norms_m = np.linalg.norm(Z_m, axis=1)

    # Non-zero rows should have norm ~1
    for i, n in enumerate(norms_h):
        if n > 1e-8:
            assert abs(n - 1.0) < 1e-5, f"Z_h row {i} norm = {n}"

    for i, n in enumerate(norms_m):
        if n > 1e-8:
            assert abs(n - 1.0) < 1e-5, f"Z_m row {i} norm = {n}"


def test_unified_svd_dtype():
    """Z_h and Z_m should be float32."""
    matrices = _make_toy_matrices()
    Z_h, Z_m, _, _ = build_unified_svd(matrices, svd_dim=8, seed=42)

    assert Z_h.dtype == np.float32
    assert Z_m.dtype == np.float32


def test_split_info_keys():
    """split_info should contain train/held for each modality."""
    matrices = _make_toy_matrices()
    _, _, _, split_info = build_unified_svd(matrices, svd_dim=8, seed=42)

    expected_keys = {'bp_train', 'bp_held', 'mf_train', 'mf_held', 'dis_train', 'dis_held'}
    assert set(split_info.keys()) == expected_keys


if __name__ == '__main__':
    test_six_matrices_naming()
    print("PASS: test_six_matrices_naming")

    test_split_terms()
    print("PASS: test_split_terms")

    test_unified_svd_shapes()
    print("PASS: test_unified_svd_shapes")

    test_unified_svd_row_normalized()
    print("PASS: test_unified_svd_row_normalized")

    test_unified_svd_dtype()
    print("PASS: test_unified_svd_dtype")

    test_split_info_keys()
    print("PASS: test_split_info_keys")

    print("\nAll preprocessing/SVD tests passed!")
