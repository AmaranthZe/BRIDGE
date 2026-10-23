"""Fused Gromov-Wasserstein Optimal Transport Solver."""

import logging
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .sinkhorn import sinkhorn_log, sinkhorn_unbalanced

logger = logging.getLogger(__name__)


def gw_cost(C_s: Tensor, C_t: Tensor, T: Tensor) -> Tensor:
    """
    Compute Gromov-Wasserstein cost: sum_{i,k,j,l} (C_s[i,k]-C_t[j,l])^2 T[i,j] T[k,l].

    Efficient O(n^2 m + n m^2) computation.
    """
    p = T.sum(dim=1)  # [N_s]
    q = T.sum(dim=0)  # [N_t]

    term1 = (C_s ** 2) @ p      # [N_s]
    term2 = q @ (C_t ** 2)      # [N_t]
    term3 = C_s @ T @ C_t       # [N_s, N_t]

    cost_tensor = term1.unsqueeze(1) + term2.unsqueeze(0) - 2 * term3
    return (cost_tensor * T).sum()


def gw_gradient(C_s: Tensor, C_t: Tensor, T: Tensor) -> Tensor:
    """Gradient of GW cost w.r.t. T."""
    p = T.sum(dim=1, keepdim=True)  # [N_s, 1]
    q = T.sum(dim=0, keepdim=True)  # [1, N_t]
    return (C_s ** 2) @ p + q @ (C_t ** 2) - 2 * C_s @ T @ C_t


def solve_fgw(
    C_s: Tensor,
    C_t: Tensor,
    M: Tensor,
    alpha: float = 0.5,
    epsilon: float = 0.05,
    max_iter: int = 50,
    threshold: float = 1e-5,
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

    Cost = (1-alpha) * GW_structural + alpha * Wasserstein_feature
    """
    N_s, N_t = M.shape
    device = M.device

    if a is None:
        a = torch.full((N_s,), 1.0 / N_s, device=device)
    if b is None:
        b = torch.full((N_t,), 1.0 / N_t, device=device)

    # Initialize T
    T = Pi0.clone() if Pi0 is not None else a.unsqueeze(1) * b.unsqueeze(0)

    # FGW iterations
    losses = []
    diff = float("inf")
    for i in range(max_iter):
        T_prev = T.clone()

        # GW gradient
        grad_gw = gw_gradient(C_s, C_t, T)

        # Combined cost
        cost = (1 - alpha) * grad_gw + alpha * M

        # Prior regularization: KL(T || Pi0)
        if Pi0 is not None and lambda_prior > 0:
            log_ratio = torch.log(T / (Pi0 + 1e-30) + 1e-30)
            cost = cost + lambda_prior * log_ratio

        # Memory consistency: KL(T || memory)
        if memory_prior is not None and lambda_memory > 0:
            log_ratio = torch.log(T / (memory_prior + 1e-30) + 1e-30)
            cost = cost + lambda_memory * log_ratio

        # Sinkhorn step
        if use_unbalanced:
            T = sinkhorn_unbalanced(cost, a, b, epsilon, tau_row, tau_col, mask=mask)
        else:
            T = sinkhorn_log(cost, a, b, epsilon, mask=mask)

        # Check convergence
        diff = torch.max(torch.abs(T - T_prev)).item()
        if diff < threshold:
            break

    info = {"iterations": i + 1, "final_diff": diff}
    return T, info


class FGWSolver:
    """FGW solver with configuration."""

    def __init__(
        self,
        alpha: float = 0.5,
        epsilon: float = 0.05,
        lambda_prior: float = 0.5,
        lambda_memory: float = 0.3,
        use_unbalanced: bool = True,
        tau_row: float = 0.05,
        tau_col: float = 0.05,
        max_iter: int = 50,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambda_prior = lambda_prior
        self.lambda_memory = lambda_memory
        self.use_unbalanced = use_unbalanced
        self.tau_row = tau_row
        self.tau_col = tau_col
        self.max_iter = max_iter

    def solve(self, C_s, C_t, M, Pi0=None, memory_prior=None,
              a=None, b=None, mask=None) -> Tuple[Tensor, Dict]:
        return solve_fgw(
            C_s=C_s, C_t=C_t, M=M,
            alpha=self.alpha, epsilon=self.epsilon,
            max_iter=self.max_iter,
            Pi0=Pi0, lambda_prior=self.lambda_prior,
            memory_prior=memory_prior, lambda_memory=self.lambda_memory,
            a=a, b=b,
            use_unbalanced=self.use_unbalanced,
            tau_row=self.tau_row, tau_col=self.tau_col,
            mask=mask,
        )

    def compute_loss(self, C_s, C_t, M, T, Pi0=None, memory_prior=None) -> Dict[str, Tensor]:
        """Compute all loss components for logging."""
        T_c = torch.clamp(T, min=1e-30)
        log_T = torch.log(T_c)

        losses = {
            "gw": gw_cost(C_s, C_t, T_c),
            "wasserstein": (M * T_c).sum(),
            "entropy": (T_c * (log_T - 1)).sum(),
        }

        if Pi0 is not None:
            Pi0_c = torch.clamp(Pi0, min=1e-30)
            losses["prior_kl"] = (T_c * torch.clamp(log_T - torch.log(Pi0_c), -100, 100)).sum()
        else:
            losses["prior_kl"] = T.new_tensor(0.0)

        if memory_prior is not None:
            mem_c = torch.clamp(memory_prior, min=1e-30)
            losses["memory_kl"] = (T_c * torch.clamp(log_T - torch.log(mem_c), -100, 100)).sum()
        else:
            losses["memory_kl"] = T.new_tensor(0.0)

        losses["total"] = (
            (1 - self.alpha) * losses["gw"]
            + self.alpha * losses["wasserstein"]
            + self.epsilon * losses["entropy"]
            + self.lambda_prior * losses["prior_kl"]
            + self.lambda_memory * losses["memory_kl"]
        )

        # NaN safety
        for key in losses:
            if torch.isnan(losses[key]):
                logger.warning(f"NaN in loss '{key}', replacing with 0")
                losses[key] = T.new_tensor(0.0, requires_grad=True)

        return losses
