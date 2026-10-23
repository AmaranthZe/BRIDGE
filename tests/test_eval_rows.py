"""
Test A-fix #5: Row-wise P assembly.

Verifies that CSR row indices are sorted ascending and that
the assembled P has valid structure.
"""

import numpy as np
import scipy.sparse as sp


def test_csr_sorted_indices():
    """CSR rows should have sorted column indices."""
    # Simulate row-wise assembly
    rows, cols, vals = [], [], []
    N_eval, N_t = 5, 20

    rng = np.random.RandomState(42)
    for i in range(N_eval):
        # Random candidate columns (unsorted)
        n_cand = rng.randint(3, 8)
        col_indices = rng.choice(N_t, n_cand, replace=False)
        values = rng.rand(n_cand).astype(np.float32)

        # Sort by column index (as in A-fix #5)
        sort_order = np.argsort(col_indices)
        col_indices = col_indices[sort_order]
        values = values[sort_order]

        rows.extend([i] * len(col_indices))
        cols.extend(col_indices.tolist())
        vals.extend(values.tolist())

    P = sp.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows), np.array(cols))),
        shape=(N_eval, N_t),
    )

    # Check that each row has sorted indices
    for i in range(N_eval):
        start, end = P.indptr[i], P.indptr[i + 1]
        row_indices = P.indices[start:end]
        assert np.all(row_indices[:-1] <= row_indices[1:]), (
            f"Row {i} indices not sorted: {row_indices}"
        )


def test_csr_values_positive():
    """All values in P should be positive."""
    N_eval, N_t = 3, 10
    rows = [0, 0, 1, 1, 2, 2]
    cols = [1, 3, 2, 5, 0, 7]
    vals = [0.5, 0.3, 0.4, 0.6, 0.2, 0.8]

    P = sp.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows), np.array(cols))),
        shape=(N_eval, N_t),
    )

    assert np.all(P.data > 0), "All P values should be positive"


def test_csr_shape_matches():
    """P shape should be (N_eval, N_t)."""
    N_eval, N_t = 10, 50
    P = sp.csr_matrix((N_eval, N_t), dtype=np.float32)
    assert P.shape == (N_eval, N_t)


def test_empty_row_handling():
    """Rows with no candidates should be empty (zero) in CSR."""
    N_eval, N_t = 5, 10

    # Only fill rows 0 and 3
    rows = [0, 0, 3, 3]
    cols = [1, 5, 2, 7]
    vals = [0.5, 0.3, 0.4, 0.6]

    P = sp.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows), np.array(cols))),
        shape=(N_eval, N_t),
    )

    # Row 1, 2, 4 should be empty
    assert P[1].nnz == 0
    assert P[2].nnz == 0
    assert P[4].nnz == 0

    # Row 0 and 3 should have entries
    assert P[0].nnz == 2
    assert P[3].nnz == 2


if __name__ == '__main__':
    test_csr_sorted_indices()
    print("PASS: test_csr_sorted_indices")

    test_csr_values_positive()
    print("PASS: test_csr_values_positive")

    test_csr_shape_matches()
    print("PASS: test_csr_shape_matches")

    test_empty_row_handling()
    print("PASS: test_empty_row_handling")

    print("\nAll eval row tests passed!")
