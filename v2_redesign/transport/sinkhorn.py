"""Log-domain Sinkhorn solvers (balanced and KL-UOT).

Critical fixes from v1:
- UOT exponent: exp = tau / (tau + epsilon), NOT epsilon / tau / (epsilon/tau + epsilon)
- No T = T / T.sum() — this breaks marginal constraints
- epsilon NOT silently clamped to 0.01
"""

import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _compute_big_m(C: Tensor, mask: Tensor) -> Tensor:
    """Compute BIG_M penalty for forbidden entries."""
    if mask.any():
        allowed = C[mask]
        p95 = torch.quantile(allowed, 0.95)
        q75 = torch.quantile(allowed, 0.75)
        q25 = torch.quantile(allowed, 0.25)
        iqr = q75 - q25
        BIG_M = torch.maximum(p95 + 5 * iqr, allowed.max() + C.new_tensor(10.0))
    else:
        BIG_M = C.max() + C.new_tensor(100.0)
    return BIG_M


def sinkhorn_log(
    M: Tensor,
    a: Tensor,
    b: Tensor,
    epsilon: float,
    max_iter: int = 100,
    threshold: float = 1e-6,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Balanced Sinkhorn in log domain.

    Args:
        M: Cost matrix [N, K]
        a: Source marginal [N]
        b: Target marginal [K]
        epsilon: Entropic regularization (used as-is, NOT clamped)
        max_iter: Maximum iterations
        threshold: Convergence threshold
        mask: Optional boolean mask [N, K] — True = allowed

    Returns:
        Transport plan [N, K]
    """
    N, K = M.shape

    if mask is not None:
        BIG_M = _compute_big_m(M, mask)
        M = M.masked_fill(~mask, BIG_M)

    # Log kernel
    log_K = -M / epsilon
    log_K = torch.clamp(log_K, min=-100, max=0)

    log_a = torch.log(torch.clamp(a, min=1e-30))
    log_b = torch.log(torch.clamp(b, min=1e-30))

    u = torch.zeros(N, device=M.device, dtype=M.dtype)
    v = torch.zeros(K, device=M.device, dtype=M.dtype)

    for i in range(max_iter):
        u_prev = u.clone()

        u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        u = torch.clamp(u, min=-100, max=100)

        v = log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
        v = torch.clamp(v, min=-100, max=100)

        if torch.isnan(u).any() or torch.isnan(v).any():
            logger.warning(f"NaN in Sinkhorn at iter {i}, using fallback")
            return a.unsqueeze(1) * b.unsqueeze(0)

        if torch.max(torch.abs(u - u_prev)) < threshold:
            break

    log_T = u.unsqueeze(1) + log_K + v.unsqueeze(0)
    T = torch.exp(torch.clamp(log_T, min=-100, max=0))
    T = torch.clamp(T, min=0)
    # NOTE: No T = T / T.sum() — marginals are already enforced by Sinkhorn
    return T


def sinkhorn_unbalanced(
    M: Tensor,
    a: Tensor,
    b: Tensor,
    epsilon: float,
    tau_row: float,
    tau_col: float,
    max_iter: int = 100,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Unbalanced Sinkhorn with KL marginal relaxation — LOG DOMAIN.

    Log-domain implementation avoids overflow/underflow that occurs in direct
    domain when tau >> epsilon (strong marginal constraints). The UOT proximal
    step becomes a simple scaling in log space:
        log_u = exp_row * (log_a - logsumexp(log_K + log_v, dim=1))

    Exponent: exp = tau / (tau + epsilon)

    Args:
        M: Cost matrix [N, K]
        a: Source marginal [N]
        b: Target marginal [K]
        epsilon: Entropic regularization
        tau_row: Row marginal relaxation strength
        tau_col: Column marginal relaxation strength
        max_iter: Maximum iterations
        mask: Optional boolean mask

    Returns:
        Transport plan [N, K]
    """
    N, K = M.shape

    if mask is not None:
        BIG_M = _compute_big_m(M, mask)
        M = M.masked_fill(~mask, BIG_M)

    # Log kernel — epsilon used as-is (NOT clamped to 0.01)
    log_K = -M / epsilon
    log_K = torch.clamp(log_K, min=-100, max=0)

    log_a = torch.log(torch.clamp(a, min=1e-30))
    log_b = torch.log(torch.clamp(b, min=1e-30))

    # UOT exponents: tau / (tau + epsilon)
    exp_row = tau_row / (tau_row + epsilon)
    exp_col = tau_col / (tau_col + epsilon)

    log_u = torch.zeros(N, device=M.device, dtype=M.dtype)
    log_v = torch.zeros(K, device=M.device, dtype=M.dtype)

    for it in range(max_iter):
        log_u_prev = log_u.clone()

        # Update log_v: log_v = exp_col * (log_b - logsumexp(log_K + log_u, dim=0))
        log_Ktu = torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)  # [K]
        log_v = exp_col * (log_b - log_Ktu)
        log_v = torch.clamp(log_v, min=-100, max=100)

        # Update log_u: log_u = exp_row * (log_a - logsumexp(log_K + log_v, dim=1))
        log_Kv = torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)  # [N]
        log_u = exp_row * (log_a - log_Kv)
        log_u = torch.clamp(log_u, min=-100, max=100)

        if torch.isnan(log_u).any() or torch.isnan(log_v).any():
            logger.warning(f"NaN in UOT Sinkhorn at iter {it}, using fallback")
            return a.unsqueeze(1) * b.unsqueeze(0)

        if torch.max(torch.abs(log_u - log_u_prev)) < 1e-6:
            break

    log_T = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    T = torch.exp(torch.clamp(log_T, min=-100, max=0))
    T = torch.clamp(T, min=0)
    # NOTE: No T = T / T.sum() — UOT does NOT preserve total mass by design
    return T
