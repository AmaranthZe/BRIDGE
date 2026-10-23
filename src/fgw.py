"""
Fused Gromov-Wasserstein Optimal Transport Solver.

This module implements FGW with:
- Structural + feature + supervision costs
- Prior coupling regularization
- Memory bank consistency
- Optional unbalanced transport
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


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
    Sinkhorn algorithm in log domain for numerical stability.

    Args:
        M: Cost matrix [N, M]
        a: Source marginal [N]
        b: Target marginal [M]
        epsilon: Regularization strength
        max_iter: Maximum iterations
        threshold: Convergence threshold
        mask: Optional boolean mask [N, M] — True = allowed

    Returns:
        Transport plan [N, M]
    """
    N, M_dim = M.shape

    # Enforce mask via BIG_M penalty on forbidden entries
    if mask is not None:
        BIG_M = _compute_big_m(M, mask)
        M = M.masked_fill(~mask, BIG_M)

    # Clamp cost matrix to prevent extreme values
    M = torch.clamp(M, min=0, max=100)

    # Initialize dual variables
    u = torch.zeros(N, device=M.device, dtype=M.dtype)
    v = torch.zeros(M_dim, device=M.device, dtype=M.dtype)

    # Log marginals with better clamping
    log_a = torch.log(torch.clamp(a, min=1e-10))
    log_b = torch.log(torch.clamp(b, min=1e-10))

    # Log kernel with clamping to prevent extreme values
    log_K = -M / max(epsilon, 0.01)  # Ensure minimum epsilon
    log_K = torch.clamp(log_K, min=-100, max=0)  # Prevent extreme values

    for i in range(max_iter):
        u_prev = u.clone()

        # Update u with numerical stability
        u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        u = torch.clamp(u, min=-100, max=100)

        # Update v with numerical stability
        v = log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
        v = torch.clamp(v, min=-100, max=100)

        # Check for NaN and break if found
        if torch.isnan(u).any() or torch.isnan(v).any():
            logger.warning(f"NaN detected in Sinkhorn iteration {i}, using fallback")
            # Fallback to uniform transport
            T = a.unsqueeze(1) * b.unsqueeze(0)
            return T

        # Check convergence
        if torch.max(torch.abs(u - u_prev)) < threshold:
            break

    # Compute transport plan with clamping
    log_T = u.unsqueeze(1) + log_K + v.unsqueeze(0)
    log_T = torch.clamp(log_T, min=-100, max=0)
    T = torch.exp(log_T)

    # Ensure valid transport plan
    T = torch.clamp(T, min=1e-10)
    T = T / T.sum()  # Normalize

    return T


def gw_cost(
    C_s: Tensor,
    C_t: Tensor,
    T: Tensor,
) -> Tensor:
    """
    Compute Gromov-Wasserstein cost term.

    L_GW = sum_{i,k,j,l} (C_s[i,k] - C_t[j,l])^2 * T[i,j] * T[k,l]

    Efficient computation using tensor products.

    Args:
        C_s: Source structure matrix [N_s, N_s]
        C_t: Target structure matrix [N_t, N_t]
        T: Transport plan [N_s, N_t]

    Returns:
        GW cost (scalar)
    """
    # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
    # L(C_s, C_t) = f1(C_s) @ T @ 1 + 1 @ T @ f2(C_t)^T - h1(C_s) @ T @ h2(C_t)^T

    N_s, N_t = T.shape

    # Compute (C_s^2) @ T @ 1_t = C_s^2 @ p
    p = T.sum(dim=1)  # [N_s]
    q = T.sum(dim=0)  # [N_t]

    term1 = (C_s ** 2) @ p  # [N_s]

    # Compute 1_s @ T @ (C_t^2)^T = q^T @ C_t^2
    term2 = q @ (C_t ** 2)  # [N_t]

    # Compute C_s @ T @ C_t
    term3 = C_s @ T @ C_t  # [N_s, N_t]

    # Full cost tensor: (term1 + term2 - 2*term3) * T
    cost_tensor = term1.unsqueeze(1) + term2.unsqueeze(0) - 2 * term3

    return (cost_tensor * T).sum()


def fgw_iteration(
    C_s: Tensor,
    C_t: Tensor,
    M: Tensor,
    T: Tensor,
    alpha: float,
    epsilon: float,
    a: Tensor,
    b: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    One iteration of FGW optimization.

    Args:
        C_s: Source structure matrix [N_s, N_s]
        C_t: Target structure matrix [N_t, N_t]
        M: Feature cost matrix [N_s, N_t]
        T: Current transport plan [N_s, N_t]
        alpha: Structure-feature tradeoff
        epsilon: Entropic regularization
        a: Source marginal
        b: Target marginal
        mask: Optional boolean mask [N_s, N_t] — True = allowed

    Returns:
        Updated transport plan
    """
    p = T.sum(dim=1, keepdim=True)  # [N_s, 1]
    q = T.sum(dim=0, keepdim=True)  # [1, N_t]

    grad_gw = (C_s ** 2) @ p + q @ (C_t ** 2) - 2 * C_s @ T @ C_t

    # Combined cost
    cost = (1 - alpha) * grad_gw + alpha * M

    # Sinkhorn step with mask
    T_new = sinkhorn_log(cost, a, b, epsilon, mask=mask)

    return T_new


def solve_fgw(
    C_s: Tensor,
    C_t: Tensor,
    M: Tensor,
    alpha: float = 0.5,
    epsilon: float = 0.005,
    max_iter: int = 50,
    threshold: float = 1e-5,
    L: Optional[Tensor] = None,
    lambda_label: float = 0.0,
    Pi0: Optional[Tensor] = None,
    lambda_prior: float = 0.0,
    memory_prior: Optional[Tensor] = None,
    lambda_memory: float = 0.0,
    a: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    use_unbalanced: bool = False,
    tau_row: float = 0.05,
    tau_col: float = 0.05,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict]:
    """
    Solve Fused Gromov-Wasserstein optimal transport.

    Args:
        C_s: Source structure matrix [N_s, N_s]
        C_t: Target structure matrix [N_t, N_t]
        M: Feature cost matrix [N_s, N_t]
        alpha: Structure-feature tradeoff (0 = pure GW, 1 = pure Wasserstein)
        epsilon: Entropic regularization
        max_iter: Maximum iterations
        threshold: Convergence threshold
        L: Supervision cost matrix [N_s, N_t]
        lambda_label: Weight for supervision cost
        Pi0: Prior coupling matrix [N_s, N_t]
        lambda_prior: Weight for prior KL divergence
        memory_prior: Memory bank prior [N_s, N_t]
        lambda_memory: Weight for memory consistency
        a: Source marginal (default: uniform)
        b: Target marginal (default: uniform)
        use_unbalanced: Whether to use unbalanced OT
        tau_row: Row marginal relaxation
        tau_col: Column marginal relaxation
        mask: Optional boolean mask [N_s, N_t] — True = allowed entry

    Returns:
        Tuple of (transport_plan, info_dict)
    """
    N_s, N_t = M.shape
    device = M.device

    # Default marginals (always on correct device)
    if a is None:
        a = torch.full((N_s,), 1.0 / N_s, device=device)
    if b is None:
        b = torch.full((N_t,), 1.0 / N_t, device=device)

    # Initialize transport plan
    if Pi0 is not None:
        T = Pi0.clone()
    else:
        T = a.unsqueeze(1) * b.unsqueeze(0)

    # Add supervision cost to feature cost
    M_total = M.clone()
    if L is not None and lambda_label > 0:
        M_total = M_total + lambda_label * L

    # FGW iterations
    losses = []
    for i in range(max_iter):
        T_prev = T.clone()

        # Compute GW gradient
        p = T.sum(dim=1, keepdim=True)
        q = T.sum(dim=0, keepdim=True)
        grad_gw = (C_s ** 2) @ p + q @ (C_t ** 2) - 2 * C_s @ T @ C_t

        # Combined cost
        cost = (1 - alpha) * grad_gw + alpha * M_total

        # Add prior regularization
        if Pi0 is not None and lambda_prior > 0:
            # KL divergence gradient: log(T/Pi0)
            log_ratio = torch.log(T / (Pi0 + 1e-10) + 1e-10)
            cost = cost + lambda_prior * log_ratio

        # Add memory consistency
        if memory_prior is not None and lambda_memory > 0:
            log_ratio_mem = torch.log(T / (memory_prior + 1e-10) + 1e-10)
            cost = cost + lambda_memory * log_ratio_mem

        # Sinkhorn step with mask threading
        if use_unbalanced:
            T = sinkhorn_unbalanced(cost, a, b, epsilon, tau_row, tau_col, mask=mask)
        else:
            T = sinkhorn_log(cost, a, b, epsilon, mask=mask)

        # Compute loss
        loss = compute_fgw_loss(C_s, C_t, M_total, T, alpha, epsilon, Pi0, lambda_prior)
        losses.append(loss.item())

        # Check convergence
        diff = torch.max(torch.abs(T - T_prev)).item()
        if diff < threshold:
            break

    info = {
        'iterations': i + 1,
        'final_diff': diff,
        'losses': losses,
    }

    return T, info


def _compute_big_m(C: Tensor, mask: Tensor) -> Tensor:
    """
    Compute BIG_M penalty for forbidden entries from allowed entries.

    Args:
        C: Cost matrix [N, M]
        mask: Boolean mask [N, M] — True = allowed

    Returns:
        BIG_M scalar tensor on same device as C
    """
    if (mask.sum() > 0).item():
        allowed = C[mask]
        p95 = torch.quantile(allowed, 0.95)
        q75 = torch.quantile(allowed, 0.75)
        q25 = torch.quantile(allowed, 0.25)
        iqr = q75 - q25
        batch_max = torch.max(allowed)
        BIG_M = torch.maximum(p95 + 5 * iqr, batch_max + C.new_tensor(10.0))
    else:
        BIG_M = C.max() + C.new_tensor(100.0)
    return BIG_M


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
    Unbalanced Sinkhorn with KL marginal relaxation.

    Args:
        M: Cost matrix [N, M]
        a: Source marginal [N]
        b: Target marginal [M]
        epsilon: Entropic regularization
        tau_row: Row marginal relaxation
        tau_col: Column marginal relaxation
        max_iter: Maximum iterations
        mask: Optional boolean mask [N, M] — True = allowed, False = forbidden

    Returns:
        Transport plan [N, M]
    """
    N, M_dim = M.shape

    # Enforce mask via BIG_M penalty on forbidden entries
    if mask is not None:
        BIG_M = _compute_big_m(M, mask)
        M = M.masked_fill(~mask, BIG_M)

    # Clamp cost matrix
    M = torch.clamp(M, min=0, max=100)
    epsilon = max(epsilon, 0.01)

    # Compute scaling factors with numerical stability
    rho_row = epsilon / max(tau_row, 0.01)
    rho_col = epsilon / max(tau_col, 0.01)

    # Initialize kernel with clamping
    log_K = -M / epsilon
    log_K = torch.clamp(log_K, min=-100, max=0)
    K = torch.exp(log_K)

    u = torch.ones(N, device=M.device, dtype=M.dtype)
    v = torch.ones(M_dim, device=M.device, dtype=M.dtype)

    for it in range(max_iter):
        # Update v
        Ktu = K.t() @ u
        Ktu = torch.clamp(Ktu, min=1e-10)
        exp_v = rho_col / (rho_col + epsilon)
        v = torch.pow(b / Ktu, exp_v)
        v = torch.clamp(v, min=1e-10, max=1e10)

        # Update u
        Kv = K @ v
        Kv = torch.clamp(Kv, min=1e-10)
        exp_u = rho_row / (rho_row + epsilon)
        u = torch.pow(a / Kv, exp_u)
        u = torch.clamp(u, min=1e-10, max=1e10)

        # Check for NaN
        if torch.isnan(u).any() or torch.isnan(v).any():
            logger.warning(f"NaN in unbalanced Sinkhorn at iteration {it}, using fallback")
            return a.unsqueeze(1) * b.unsqueeze(0)

    T = u.unsqueeze(1) * K * v.unsqueeze(0)
    T = torch.clamp(T, min=1e-10)
    T = T / (T.sum() + 1e-10)

    return T


def compute_fgw_loss(
    C_s: Tensor,
    C_t: Tensor,
    M: Tensor,
    T: Tensor,
    alpha: float,
    epsilon: float,
    Pi0: Optional[Tensor] = None,
    lambda_prior: float = 0.0,
) -> Tensor:
    """
    Compute full FGW loss.

    Args:
        C_s, C_t: Structure matrices
        M: Feature cost
        T: Transport plan
        alpha: Tradeoff parameter
        epsilon: Entropic regularization
        Pi0: Prior coupling
        lambda_prior: Prior weight

    Returns:
        Total loss
    """
    # Ensure T is valid
    T = torch.clamp(T, min=1e-10)

    # GW term
    loss_gw = gw_cost(C_s, C_t, T)

    # Wasserstein term
    loss_w = (M * T).sum()

    # Entropy term with numerical stability
    log_T = torch.log(torch.clamp(T, min=1e-10))
    loss_entropy = epsilon * (T * (log_T - 1)).sum()

    # Prior KL with numerical stability
    loss_prior = torch.tensor(0.0, device=T.device, dtype=T.dtype)
    if Pi0 is not None and lambda_prior > 0:
        Pi0_clamped = torch.clamp(Pi0, min=1e-10)
        log_ratio = torch.log(T / Pi0_clamped)
        log_ratio = torch.clamp(log_ratio, min=-100, max=100)
        loss_prior = lambda_prior * (T * log_ratio).sum()

    total = (1 - alpha) * loss_gw + alpha * loss_w + loss_entropy + loss_prior

    # Check for NaN and return a safe value
    if torch.isnan(total):
        logger.warning("NaN detected in FGW loss, returning 0")
        return torch.tensor(0.0, device=T.device, dtype=T.dtype, requires_grad=True)

    return total


class FGWSolver:
    """
    FGW solver wrapper with configuration.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        epsilon: float = 0.005,
        lambda_label: float = 0.0,
        lambda_prior: float = 0.5,
        lambda_memory: float = 0.3,
        use_unbalanced: bool = True,
        tau_row: float = 0.05,
        tau_col: float = 0.05,
        max_iter: int = 50,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambda_label = lambda_label
        self.lambda_prior = lambda_prior
        self.lambda_memory = lambda_memory
        self.use_unbalanced = use_unbalanced
        self.tau_row = tau_row
        self.tau_col = tau_col
        self.max_iter = max_iter

    def solve(
        self,
        C_s: Tensor,
        C_t: Tensor,
        M: Tensor,
        L: Optional[Tensor] = None,
        Pi0: Optional[Tensor] = None,
        memory_prior: Optional[Tensor] = None,
        a: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Solve FGW with configured parameters.

        Args:
            mask: Optional boolean mask [N_s, N_t] — True = allowed entry.
                  Used to enforce per-query candidate isolation.
        """
        return solve_fgw(
            C_s=C_s,
            C_t=C_t,
            M=M,
            alpha=self.alpha,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            L=L,
            lambda_label=self.lambda_label,
            Pi0=Pi0,
            lambda_prior=self.lambda_prior,
            memory_prior=memory_prior,
            lambda_memory=self.lambda_memory,
            a=a,
            b=b,
            use_unbalanced=self.use_unbalanced,
            tau_row=self.tau_row,
            tau_col=self.tau_col,
            mask=mask,
        )

    def compute_loss(
        self,
        C_s: Tensor,
        C_t: Tensor,
        M: Tensor,
        T: Tensor,
        Pi0: Optional[Tensor] = None,
        memory_prior: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute all loss components for logging.
        """
        losses = {}

        # Ensure T is valid
        T = torch.clamp(T, min=1e-10)
        log_T = torch.log(T)

        # GW loss
        losses['gw'] = gw_cost(C_s, C_t, T)

        # Wasserstein loss
        losses['wasserstein'] = (M * T).sum()

        # Entropy with numerical stability
        losses['entropy'] = (T * (log_T - 1)).sum()

        # Prior KL with numerical stability
        if Pi0 is not None:
            Pi0_clamped = torch.clamp(Pi0, min=1e-10)
            log_ratio = torch.clamp(log_T - torch.log(Pi0_clamped), min=-100, max=100)
            losses['prior_kl'] = (T * log_ratio).sum()
        else:
            losses['prior_kl'] = torch.tensor(0.0, device=T.device, dtype=T.dtype)

        # Memory KL with numerical stability
        if memory_prior is not None:
            mem_clamped = torch.clamp(memory_prior, min=1e-10)
            log_ratio_mem = torch.clamp(log_T - torch.log(mem_clamped), min=-100, max=100)
            losses['memory_kl'] = (T * log_ratio_mem).sum()
        else:
            losses['memory_kl'] = torch.tensor(0.0, device=T.device, dtype=T.dtype)

        # Total
        losses['total'] = (
            (1 - self.alpha) * losses['gw'] +
            self.alpha * losses['wasserstein'] +
            self.epsilon * losses['entropy'] +
            self.lambda_prior * losses['prior_kl'] +
            self.lambda_memory * losses['memory_kl']
        )

        # Check for NaN and replace with safe values
        for key in losses:
            if torch.isnan(losses[key]):
                logger.warning(f"NaN detected in loss component '{key}', replacing with 0")
                losses[key] = torch.tensor(0.0, device=T.device, dtype=T.dtype, requires_grad=True)

        return losses
