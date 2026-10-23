"""
Evaluation Metrics for Cross-Species Gene Mapping.

This module implements:
- Precision/Recall/F1 at k
- nDCG (normalized Discounted Cumulative Gain)
- GO term agreement metrics
- Embedding alignment quality (CKA)
- Row-wise P assembly via local FGW (A-fix #5)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def precision_at_k(predicted: List[int], ground_truth: set, k: int) -> float:
    """
    Compute Precision@k.

    Args:
        predicted: Ordered list of predicted indices
        ground_truth: Set of correct indices
        k: Number of top predictions to consider

    Returns:
        Precision@k score
    """
    if k == 0 or len(predicted) == 0:
        return 0.0

    top_k = predicted[:k]
    hits = sum(1 for p in top_k if p in ground_truth)
    return hits / k


def recall_at_k(predicted: List[int], ground_truth: set, k: int) -> float:
    """
    Compute Recall@k.

    Args:
        predicted: Ordered list of predicted indices
        ground_truth: Set of correct indices
        k: Number of top predictions to consider

    Returns:
        Recall@k score
    """
    if len(ground_truth) == 0:
        return 0.0

    top_k = predicted[:k]
    hits = sum(1 for p in top_k if p in ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(predicted: List[int], ground_truth: set, k: int) -> float:
    """
    Compute nDCG@k (normalized Discounted Cumulative Gain).

    Args:
        predicted: Ordered list of predicted indices
        ground_truth: Set of correct indices
        k: Number of top predictions to consider

    Returns:
        nDCG@k score
    """
    if k == 0 or len(ground_truth) == 0:
        return 0.0

    # Compute DCG
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

    # Compute ideal DCG
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_reciprocal_rank(predicted: List[int], ground_truth: set) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        predicted: Ordered list of predicted indices
        ground_truth: Set of correct indices

    Returns:
        Reciprocal rank of first correct prediction
    """
    for i, p in enumerate(predicted):
        if p in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def go_term_agreement(
    source_terms: set,
    target_terms: set,
    all_matches: List[Tuple[int, int, float]],
    W_source,
    W_target,
) -> Dict[str, float]:
    """
    Compute GO term agreement between matched gene pairs.

    Args:
        source_terms: Set of all source GO terms
        target_terms: Set of all target GO terms
        all_matches: List of (source_idx, target_idx, score) tuples
        W_source: Source gene-term sparse matrix
        W_target: Target gene-term sparse matrix

    Returns:
        Dictionary with agreement metrics
    """
    if len(all_matches) == 0:
        return {'mean_jaccard': 0.0, 'weighted_jaccard': 0.0}

    jaccards = []
    weights = []

    for src_idx, tgt_idx, score in all_matches:
        src_terms = set(W_source[src_idx].nonzero()[1])
        tgt_terms = set(W_target[tgt_idx].nonzero()[1])

        jacc = jaccard_similarity(src_terms, tgt_terms)
        jaccards.append(jacc)
        weights.append(score)

    mean_jaccard = np.mean(jaccards)
    weighted_jaccard = np.average(jaccards, weights=weights) if sum(weights) > 0 else 0.0

    return {
        'mean_jaccard': float(mean_jaccard),
        'weighted_jaccard': float(weighted_jaccard),
    }


def centered_kernel_alignment(X: Tensor, Y: Tensor) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two embedding spaces.

    CKA measures similarity between representations.
    Lower CKA after training indicates better cross-species alignment.

    Args:
        X: First embedding matrix [N, D]
        Y: Second embedding matrix [M, D]

    Returns:
        CKA score (0-1, higher = more similar)
    """
    def centering_matrix(n: int, device) -> Tensor:
        I = torch.eye(n, device=device)
        ones = torch.ones(n, n, device=device) / n
        return I - ones

    # Compute Gram matrices
    K = X @ X.t()
    L = Y @ Y.t()

    # Center the Gram matrices
    n = K.shape[0]
    m = L.shape[0]

    H_n = centering_matrix(n, K.device)
    H_m = centering_matrix(m, L.device)

    K_c = H_n @ K @ H_n
    L_c = H_m @ L @ H_m

    # For different sizes, we need to align
    if n != m:
        # Use random projection to align sizes
        min_size = min(n, m)
        if n > m:
            idx = torch.randperm(n)[:min_size]
            K_c = K_c[idx][:, idx]
        else:
            idx = torch.randperm(m)[:min_size]
            L_c = L_c[idx][:, idx]

    # Compute CKA
    hsic_kl = (K_c * L_c).sum()
    hsic_kk = (K_c * K_c).sum()
    hsic_ll = (L_c * L_c).sum()

    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-10)

    return float(cka)


def procrustes_distance(X: Tensor, Y: Tensor) -> float:
    """
    Compute Procrustes distance between two embedding spaces.

    Measures how well X can be aligned to Y via orthogonal transformation.

    Args:
        X: First embedding matrix [N, D]
        Y: Second embedding matrix [N, D] (same size as X)

    Returns:
        Procrustes distance (lower = more alignable)
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")

    # Center the matrices
    X_c = X - X.mean(dim=0)
    Y_c = Y - Y.mean(dim=0)

    # SVD of Y^T X
    U, S, Vh = torch.linalg.svd(Y_c.t() @ X_c)

    # Optimal rotation
    R = Vh.t() @ U.t()

    # Align X to Y
    X_aligned = X_c @ R

    # Compute distance
    dist = ((X_aligned - Y_c) ** 2).sum().sqrt()

    return float(dist)


class Evaluator:
    """
    Evaluator for cross-species gene mapping.
    """

    def __init__(
        self,
        sparse_matrices: Dict,
        indexers: Dict,
        orthologs: Optional[Dict] = None,
    ):
        """
        Args:
            sparse_matrices: Sparse matrices for GO/Disease
            indexers: ID indexers
            orthologs: Optional ortholog pairs for evaluation
        """
        self.sparse_matrices = sparse_matrices
        self.indexers = indexers
        self.orthologs = orthologs

    def evaluate_predictions(
        self,
        predictions: Dict[int, List[Tuple[int, float]]],
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """
        Evaluate prediction quality against orthologs.

        Args:
            predictions: Dict mapping source_idx -> [(target_idx, score), ...]
            k_values: List of k values for Precision/Recall/nDCG

        Returns:
            Dictionary of evaluation metrics
        """
        if self.orthologs is None:
            logger.warning("No orthologs available for evaluation")
            return {}

        metrics = {}

        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []
            mrrs = []

            for src_idx, pred_list in predictions.items():
                if src_idx not in self.orthologs:
                    continue

                ground_truth = set(self.orthologs[src_idx].keys())
                predicted = [p[0] for p in pred_list]

                precisions.append(precision_at_k(predicted, ground_truth, k))
                recalls.append(recall_at_k(predicted, ground_truth, k))
                ndcgs.append(ndcg_at_k(predicted, ground_truth, k))
                mrrs.append(mean_reciprocal_rank(predicted, ground_truth))

            if precisions:
                metrics[f'precision@{k}'] = float(np.mean(precisions))
                metrics[f'recall@{k}'] = float(np.mean(recalls))
                metrics[f'ndcg@{k}'] = float(np.mean(ndcgs))

            if mrrs and k == k_values[0]:
                metrics['mrr'] = float(np.mean(mrrs))

        return metrics

    def evaluate_go_agreement(
        self,
        matches: List[Tuple[int, int, float]],
        term_type: str = 'bp',
    ) -> Dict[str, float]:
        """
        Evaluate GO term agreement for matched pairs.

        Args:
            matches: List of (mouse_idx, human_idx, score) tuples
            term_type: 'bp', 'mf', or 'disease'

        Returns:
            Agreement metrics
        """
        key_m = f'W_m{term_type[0].upper()}'
        key_h = f'W_h{term_type[0].upper()}'

        if key_m not in self.sparse_matrices or key_h not in self.sparse_matrices:
            return {}

        W_m = self.sparse_matrices[key_m]
        W_h = self.sparse_matrices[key_h]

        return go_term_agreement(
            source_terms=set(),  # Not used
            target_terms=set(),  # Not used
            all_matches=matches,
            W_source=W_m,
            W_target=W_h,
        )

    def evaluate_embedding_alignment(
        self,
        E_human: Tensor,
        E_mouse: Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate embedding space alignment quality.

        Args:
            E_human: Human gene embeddings
            E_mouse: Mouse gene embeddings

        Returns:
            Alignment metrics
        """
        return {
            'cka': centered_kernel_alignment(E_human, E_mouse),
        }

    def assemble_P_rowwise(
        self,
        model,
        E_source: Tensor,
        E_target: Tensor,
        cost_computer,
        source_indices: Optional[List[int]] = None,
        device: str = 'cpu',
        threshold: float = 1e-6,
    ) -> sp.csr_matrix:
        """
        Row-wise P assembly via per-query local FGW (A-fix #5).

        For each query gene:
          1. Run candidate selection (per-query candidates).
          2. Build local mask and run local FGW.
          3. Write a single CSR row with sorted column indices.

        Args:
            model: CrossSpeciesModel (in eval mode)
            E_source: Source embeddings [N_s, D]
            E_target: Target embeddings [N_t, D]
            cost_computer: CostComputer instance
            source_indices: Specific source indices to evaluate (default: all)
            device: Device
            threshold: Min value to keep in sparse P

        Returns:
            Sparse coupling P [N_s_eval, N_t] in CSR format with sorted indices
        """
        from .costs import structural_cost as struct_cost

        N_t = E_target.shape[0]
        N_s = E_source.shape[0]

        if source_indices is None:
            source_indices = list(range(N_s))

        rows, cols, vals = [], [], []

        model.eval()
        with torch.no_grad():
            for row_idx, src_idx in enumerate(source_indices):
                # Single query
                I_batch = torch.tensor([src_idx], dtype=torch.long, device=device)
                E_src_single = E_source[src_idx:src_idx + 1]  # [1, D]

                # Per-query candidate selection
                J_cand, soft_mask, Pi0, row_mask, _ = model.candidate_selector(
                    E_source=E_source,
                    E_target=E_target,
                    source_indices=I_batch,
                )

                if len(J_cand) == 0:
                    continue

                E_tgt_cand = E_target[J_cand]

                # Local structural costs
                C_s, _ = struct_cost(E_src_single, E_src_single)  # [1, 1]
                _, C_t = struct_cost(E_tgt_cand, E_tgt_cand)      # [J, J]

                # Feature cost
                M = cost_computer.compute_all(
                    E_source=E_src_single,
                    E_target=E_tgt_cand,
                    source_indices=I_batch,
                    target_indices=J_cand,
                )['M']

                M_masked = M * soft_mask

                # Solve local FGW with row mask
                T_local, _ = model.fgw_solver.solve(
                    C_s=C_s,
                    C_t=C_t,
                    M=M_masked,
                    Pi0=Pi0,
                    mask=row_mask,
                )

                # Extract the single row and write CSR entries
                t_row = T_local[0].cpu().numpy()  # [J]
                j_cand_np = J_cand.cpu().numpy()

                # Filter by threshold and sort by column index
                nonzero = t_row > threshold
                if nonzero.any():
                    col_indices = j_cand_np[nonzero]
                    values = t_row[nonzero]

                    # Sort by column index (required for valid CSR)
                    sort_order = np.argsort(col_indices)
                    col_indices = col_indices[sort_order]
                    values = values[sort_order]

                    rows.extend([row_idx] * len(col_indices))
                    cols.extend(col_indices.tolist())
                    vals.extend(values.tolist())

        # Build CSR
        N_eval = len(source_indices)
        if len(rows) > 0:
            P = sp.csr_matrix(
                (np.array(vals, dtype=np.float32),
                 (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
                shape=(N_eval, N_t),
            )
        else:
            P = sp.csr_matrix((N_eval, N_t), dtype=np.float32)

        logger.info(f"Assembled P: shape={P.shape}, nnz={P.nnz}")
        return P
