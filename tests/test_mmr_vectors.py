"""
Test A-fix #3: MMR with candidate vectors.

Verifies that MMR promotes diversity when two top candidates are similar.
"""

import torch
from src.soft_candidates import maximal_marginal_relevance


def test_mmr_promotes_diversity():
    """
    Given: query q, candidate A (high score, similar to q),
           candidate B (high score, nearly identical to A),
           candidate C (moderate score, very different from A & B).

    MMR should pick A first, then prefer C over B for diversity.
    """
    D = 8
    torch.manual_seed(42)

    q = torch.randn(D)
    q = q / q.norm()

    # A: very similar to q
    A = q + 0.05 * torch.randn(D)
    A = A / A.norm()

    # B: nearly identical to A (similar to q too)
    B = A + 0.02 * torch.randn(D)
    B = B / B.norm()

    # C: quite different from both A and B
    C = torch.randn(D)
    C = C / C.norm()

    candidate_vecs = torch.stack([A, B, C])  # [3, D]
    # Scores: A=0.95, B=0.93, C=0.60
    candidate_scores = torch.tensor([0.95, 0.93, 0.60])

    # Pure relevance (lambda=1) should pick A, B, C
    order_pure = maximal_marginal_relevance(
        q, candidate_vecs, candidate_scores, lambda_mmr=1.0, top_k=3
    )
    assert order_pure[0].item() == 0, "First should be A (highest relevance)"
    assert order_pure[1].item() == 1, "Second should be B in pure relevance mode"

    # With diversity (lambda=0.25) should pick A first, then C (diverse), then B
    order_diverse = maximal_marginal_relevance(
        q, candidate_vecs, candidate_scores, lambda_mmr=0.25, top_k=3
    )
    assert order_diverse[0].item() == 0, "First should still be A"
    assert order_diverse[1].item() == 2, (
        f"Second should be C (diverse), got {order_diverse[1].item()}"
    )


def test_mmr_single_candidate():
    """MMR with a single candidate should just return it."""
    D = 4
    q = torch.randn(D)
    c = torch.randn(1, D)
    s = torch.tensor([1.0])

    order = maximal_marginal_relevance(q, c, s, lambda_mmr=0.5, top_k=1)
    assert len(order) == 1
    assert order[0].item() == 0


def test_mmr_empty():
    """MMR with no candidates should return empty."""
    D = 4
    q = torch.randn(D)
    c = torch.zeros(0, D)
    s = torch.tensor([])

    order = maximal_marginal_relevance(q, c, s, lambda_mmr=0.5, top_k=5)
    assert len(order) == 0


def test_mmr_top_k_less_than_candidates():
    """MMR should respect top_k."""
    D = 4
    torch.manual_seed(42)

    q = torch.randn(D)
    c = torch.randn(10, D)
    s = torch.rand(10)

    order = maximal_marginal_relevance(q, c, s, lambda_mmr=0.5, top_k=3)
    assert len(order) == 3
    # Should have unique indices
    assert len(set(order.tolist())) == 3


if __name__ == '__main__':
    test_mmr_promotes_diversity()
    print("PASS: test_mmr_promotes_diversity")

    test_mmr_single_candidate()
    print("PASS: test_mmr_single_candidate")

    test_mmr_empty()
    print("PASS: test_mmr_empty")

    test_mmr_top_k_less_than_candidates()
    print("PASS: test_mmr_top_k_less_than_candidates")

    print("\nAll MMR tests passed!")
