"""
Test A-fix #1: Per-query candidates + row mask in FGW/UOT.

Two queries with disjoint candidates; verify that P[~mask] < 1e-8.
"""

import torch
from src.fgw import solve_fgw, sinkhorn_unbalanced, _compute_big_m


def test_mask_enforcement_balanced():
    """Balanced Sinkhorn: forbidden entries should be near zero."""
    torch.manual_seed(42)

    B, J = 2, 6
    # Disjoint candidates: query 0 -> cols [0,1,2], query 1 -> cols [3,4,5]
    mask = torch.zeros(B, J, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, 3:] = True

    # Random cost matrix
    M = torch.rand(B, J)
    C_s = torch.eye(B)
    C_t = torch.eye(J)

    T, info = solve_fgw(
        C_s=C_s,
        C_t=C_t,
        M=M,
        alpha=0.5,
        epsilon=0.05,
        mask=mask,
        use_unbalanced=False,
    )

    # Forbidden entries should be near zero
    forbidden = T[~mask]
    assert forbidden.max().item() < 1e-6, (
        f"Forbidden entries too large: max={forbidden.max().item():.2e}"
    )
    # Allowed entries should have positive mass
    allowed = T[mask]
    assert allowed.sum().item() > 0.01, "Allowed entries have no mass"


def test_mask_enforcement_unbalanced():
    """Unbalanced Sinkhorn: forbidden entries should be near zero."""
    torch.manual_seed(42)

    B, J = 2, 6
    mask = torch.zeros(B, J, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, 3:] = True

    M = torch.rand(B, J)
    C_s = torch.eye(B)
    C_t = torch.eye(J)

    T, info = solve_fgw(
        C_s=C_s,
        C_t=C_t,
        M=M,
        alpha=0.5,
        epsilon=0.05,
        mask=mask,
        use_unbalanced=True,
        tau_row=0.05,
        tau_col=0.05,
    )

    forbidden = T[~mask]
    assert forbidden.max().item() < 1e-6, (
        f"Forbidden entries too large: max={forbidden.max().item():.2e}"
    )


def test_no_mask_backward_compat():
    """Without mask, solve_fgw should still work normally."""
    torch.manual_seed(42)

    N, M_dim = 4, 6
    M = torch.rand(N, M_dim)
    C_s = torch.eye(N)
    C_t = torch.eye(M_dim)

    T, info = solve_fgw(C_s=C_s, C_t=C_t, M=M, mask=None)

    assert T.shape == (N, M_dim)
    assert T.sum().item() > 0.5


def test_big_m_computation():
    """BIG_M should be larger than any allowed entry."""
    C = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, True, False], [False, True, True]])

    big_m = _compute_big_m(C, mask)
    allowed_max = C[mask].max().item()
    assert big_m.item() > allowed_max, "BIG_M should exceed max allowed value"


if __name__ == '__main__':
    test_mask_enforcement_balanced()
    print("PASS: test_mask_enforcement_balanced")

    test_mask_enforcement_unbalanced()
    print("PASS: test_mask_enforcement_unbalanced")

    test_no_mask_backward_compat()
    print("PASS: test_no_mask_backward_compat")

    test_big_m_computation()
    print("PASS: test_big_m_computation")

    print("\nAll mask tests passed!")
