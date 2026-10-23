"""Loss computation for training."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def column_entropy(T: Tensor) -> Tensor:
    """Compute entropy of the column-sum distribution of coupling matrix T.

    H = -sum(p_j * log(p_j)) where p_j = sum_i T[i,j] / sum T.
    Maximum when all columns have equal mass (no hubs).
    Returns 0 if T has no mass.
    """
    col_mass = T.sum(dim=0)  # [N_t]
    total = col_mass.sum()
    if total < 1e-30:
        return T.new_tensor(0.0)
    col_dist = col_mass / total
    # Avoid log(0)
    log_dist = torch.log(col_dist + 1e-30)
    return -(col_dist * log_dist).sum()


def covariance_regularization(E: Tensor) -> Tensor:
    """VICReg-style covariance loss — minimize off-diagonal covariance.

    Encourages each embedding dimension to encode independent information,
    breaking the high inter-dimension correlation that causes isotropy.
    Returns mean squared off-diagonal covariance entry.
    """
    E_c = E - E.mean(dim=0, keepdim=True)
    N, D = E_c.shape
    cov = (E_c.T @ E_c) / (N - 1)  # [D, D]
    # Only penalize off-diagonal (diagonal = variance, which we want to keep)
    off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
    return off_diag / D


def embedding_alignment_loss(
    T: Tensor, E_source: Tensor, E_target: Tensor,
) -> Tensor:
    """FGW-driven embedding alignment: pull high-coupling pairs closer.

    loss = sum_{i,j} T[i,j] * (1 - cos(E_src[i], E_tgt[j]))

    Uses T.detach() to avoid double backprop through FGW solver.
    Gradients flow only through E_source and E_target embeddings.
    """
    T_d = T.detach()
    E_src_n = F.normalize(E_source, dim=1)
    E_tgt_n = F.normalize(E_target, dim=1)
    cos_sim = E_src_n @ E_tgt_n.T  # [B, J]
    return (T_d * (1.0 - cos_sim)).sum()


def compute_total_loss(
    fgw_losses: Dict[str, Tensor],
    entropy_loss: Tensor,
    T: Optional[Tensor] = None,
    col_entropy_weight: float = 0.0,
    E_h: Optional[Tensor] = None,
    E_m: Optional[Tensor] = None,
    cov_reg_weight: float = 0.0,
    align_loss_weight: float = 0.0,
    E_source_batch: Optional[Tensor] = None,
    E_target_cand: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Combine FGW losses with fusion entropy loss and anti-hub regularization.

    When col_entropy_weight > 0 and T is provided, adds a term that maximizes
    the entropy of T's column sums — penalizing hub formation where a few
    target genes absorb mass from many sources.
    """
    losses = dict(fgw_losses)
    losses["fusion_entropy"] = entropy_loss
    losses["total"] = losses["total"] + entropy_loss

    # Anti-hub: maximize column entropy of coupling matrix
    if T is not None and col_entropy_weight > 0:
        col_ent = column_entropy(T)
        losses["col_entropy"] = col_ent
        # Maximize entropy → subtract from loss (we minimize loss)
        losses["total"] = losses["total"] - col_entropy_weight * col_ent
    else:
        losses["col_entropy"] = entropy_loss.new_tensor(0.0)

    # Anti-isotropy: minimize off-diagonal covariance of embeddings
    if cov_reg_weight > 0 and E_h is not None and E_m is not None:
        cov_loss = covariance_regularization(E_h) + covariance_regularization(E_m)
        losses["cov_reg"] = cov_loss
        losses["total"] = losses["total"] + cov_reg_weight * cov_loss
    else:
        losses["cov_reg"] = entropy_loss.new_tensor(0.0)

    # Embedding alignment: pull high-coupling pairs closer in embedding space
    if align_loss_weight > 0 and T is not None and E_source_batch is not None and E_target_cand is not None:
        align = embedding_alignment_loss(T, E_source_batch, E_target_cand)
        losses["align"] = align
        losses["total"] = losses["total"] + align_loss_weight * align
    else:
        losses["align"] = entropy_loss.new_tensor(0.0)

    return losses
