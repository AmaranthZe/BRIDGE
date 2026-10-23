"""Comprehensive evaluation, inference, and baseline comparison script.

Loads trained best_model.pt and runs:
1. Held-out GO/Disease term evaluation
2. GO agreement evaluation
3. Cycle consistency evaluation
4. Baseline comparisons (B1: random, B3: cosine SVD, B4: cosine HGT)
5. Inference on known gene families (Trp53, Wnt, Notch, etc.)
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("v2_evaluate.log"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.data.graph import load_unified_graph
from v2_redesign.model.model import create_model
from v2_redesign.transport.costs import CostComputer, structural_cost
from v2_redesign.transport.fgw import FGWSolver
from v2_redesign.evaluation.evaluator import Evaluator, assemble_P_rowwise
from v2_redesign.evaluation.baselines import baseline_random, baseline_cosine_hgt
from v2_redesign.evaluation.metrics import (
    jaccard_similarity, precision_at_k, recall_at_k, ndcg_at_k, mean_reciprocal_rank,
)


def load_model_and_data(cfg, device):
    """Load all data and restore trained model."""
    proc = Path(cfg.paths.proc_dir)

    # Load data
    logger.info("Loading preprocessed data...")
    with open(proc / "indexers.pkl", "rb") as f:
        indexers = pickle.load(f)

    Z_h, Z_m, svd_model, split_info = load_svd_artifacts(str(proc))
    sparse_matrices = load_sparse_matrices(str(proc))
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    logger.info(f"Indexers: {', '.join(f'{k}: {len(v):,}' for k, v in indexers.items())}")
    logger.info(f"SVD: Z_h {Z_h.shape}, Z_m {Z_m.shape}")
    logger.info(f"Split info keys: {list(split_info.keys())}")

    # Create model
    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }
    model = create_model(cfg, node_counts)

    # Load checkpoint
    ckpt_path = proc / "runs" / "best_model.pt"
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")

    return model, graph, sparse_matrices, Z_h, Z_m, indexers, split_info


def compute_embeddings(model, graph, device):
    """Compute fused gene embeddings using the trained model.

    Strategy: Run HGT and fusion entirely on CPU first, then move
    only the final embeddings to GPU. This avoids GPU OOM from
    large edge index tensors (8.3M edges for disease).
    """
    logger.info("Running HGT + Fusion on CPU to avoid GPU OOM on edge indices...")
    model = model.to("cpu")
    model.eval()

    edge_index_dict = {}
    edge_attr_dict = {}
    for et in graph.edge_types:
        ed = graph[et]
        edge_index_dict[et] = ed.edge_index  # CPU
        if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
            edge_attr_dict[et] = ed.edge_attr  # CPU

    with torch.no_grad():
        E_h, E_m, entropy = model.get_gene_embeddings(edge_index_dict, edge_attr_dict)

    logger.info(f"Embeddings (CPU): E_h {E_h.shape}, E_m {E_m.shape}, entropy {entropy.item():.4f}")

    # Move final embeddings to GPU for fast candidate selection / FGW
    E_h = E_h.to(device)
    E_m = E_m.to(device)

    # Move candidate selector and fusion to GPU for inference
    model.candidate_selector = model.candidate_selector.to(device)
    # Keep HGT and fusion on CPU — we only need embeddings from now on

    logger.info(f"Embeddings on {device}: E_h {E_h.shape}, E_m {E_m.shape}")
    return E_h, E_m


def generate_predictions_fast(model, E_source, E_target, cost_computer, fgw_solver,
                               direction, n_sample=2000, topk=20, device="cuda",
                               Z_source=None, Z_target=None,
                               Z_source_mods=None, Z_target_mods=None):
    """Generate predictions via per-gene FGW for a sample of source genes.

    Full evaluation on all genes would be very slow. Sample n_sample genes.
    """
    N_s = E_source.shape[0]
    if n_sample is not None and n_sample < N_s:
        rng = np.random.RandomState(3407)
        sample_indices = sorted(rng.choice(N_s, n_sample, replace=False).tolist())
    else:
        sample_indices = list(range(N_s))

    logger.info(f"Generating predictions for {len(sample_indices)} source genes ({direction})...")
    predictions = {}
    model.eval()

    t0 = time.time()
    with torch.no_grad():
        for i, src_idx in enumerate(sample_indices):
            if i > 0 and i % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / i * (len(sample_indices) - i)
                logger.info(f"  {i}/{len(sample_indices)} done, ETA {eta:.0f}s")

            I = torch.tensor([src_idx], dtype=torch.long, device=device)

            # Candidate selection (with SVD features for ensemble retrieval)
            J_cand, soft_mask, Pi0, row_mask = model.candidate_selector(
                E_source=E_source, E_target=E_target, source_indices=I,
                Z_source=Z_source, Z_target=Z_target,
                Z_source_mods=Z_source_mods, Z_target_mods=Z_target_mods)

            if len(J_cand) == 0:
                continue

            E_src_single = E_source[src_idx:src_idx + 1]
            E_tgt_cand = E_target[J_cand]

            # Cost computation
            costs = cost_computer.compute(
                E_source=E_src_single, E_target=E_tgt_cand,
                source_indices=I, target_indices=J_cand,
                direction=direction)

            # FGW solve
            M_masked = costs["M"] * soft_mask
            T_local, _ = fgw_solver.solve(
                C_s=costs["C_s"], C_t=costs["C_t"],
                M=M_masked, Pi0=Pi0, mask=row_mask)

            # Extract top-k
            t_row = T_local[0]
            k = min(topk, len(t_row))
            topk_scores, topk_pos = torch.topk(t_row, k)

            cand_indices = J_cand[topk_pos].cpu().numpy()
            scores = topk_scores.cpu().numpy()

            predictions[src_idx] = [(int(idx), float(sc)) for idx, sc in zip(cand_indices, scores)]

    elapsed = time.time() - t0
    logger.info(f"Generated {len(predictions)} predictions in {elapsed:.1f}s")
    return predictions


def run_baselines(E_h, E_m, sparse_matrices, n_source_m2h, n_source_h2m, topk=20,
                   Z_h=None, Z_m=None):
    """Run baseline methods."""
    results = {}

    # B1: Random
    logger.info("Running B1: Random baseline...")
    t0 = time.time()
    pred_random_m2h = baseline_random(n_source_m2h, E_h.shape[0], topk=topk)
    pred_random_h2m = baseline_random(n_source_h2m, E_m.shape[0], topk=topk)
    results["B1_random"] = {"m2h": pred_random_m2h, "h2m": pred_random_h2m}
    logger.info(f"  B1 done in {time.time()-t0:.1f}s")

    # B3: Cosine SVD (using SVD features, NOT raw sparse -- too slow)
    # We'll compute cosine similarity on Z_h/Z_m directly as a fast proxy
    logger.info("Running B3: Cosine SVD baseline...")
    t0 = time.time()
    # This is effectively B3 but using SVD features instead of raw sparse
    # (which would OOM on disease matrices)
    pred_svd_m2h = baseline_cosine_hgt(
        torch.tensor(Z_m, dtype=torch.float32),
        torch.tensor(Z_h, dtype=torch.float32),
        topk=topk)
    pred_svd_h2m = baseline_cosine_hgt(
        torch.tensor(Z_h, dtype=torch.float32),
        torch.tensor(Z_m, dtype=torch.float32),
        topk=topk)
    results["B3_cosine_svd"] = {"m2h": pred_svd_m2h, "h2m": pred_svd_h2m}
    logger.info(f"  B3 done in {time.time()-t0:.1f}s")

    # B4: Cosine HGT
    logger.info("Running B4: Cosine HGT baseline...")
    t0 = time.time()
    pred_hgt_m2h = baseline_cosine_hgt(E_m.cpu(), E_h.cpu(), topk=topk)
    pred_hgt_h2m = baseline_cosine_hgt(E_h.cpu(), E_m.cpu(), topk=topk)
    results["B4_cosine_hgt"] = {"m2h": pred_hgt_m2h, "h2m": pred_hgt_h2m}
    logger.info(f"  B4 done in {time.time()-t0:.1f}s")

    return results


def evaluate_predictions(evaluator, predictions_m2h, predictions_h2m, label="Model"):
    """Run evaluation suite on a set of predictions."""
    results = {}

    # Held-out terms
    heldout = evaluator.evaluate_heldout_terms(predictions_m2h, "mouse_to_human")
    results.update({f"{label}_m2h_{k}": v for k, v in heldout.items()})

    heldout_h2m = evaluator.evaluate_heldout_terms(predictions_h2m, "human_to_mouse")
    results.update({f"{label}_h2m_{k}": v for k, v in heldout_h2m.items()})

    # GO agreement
    go = evaluator.evaluate_go_agreement(predictions_m2h, "mouse_to_human")
    results.update({f"{label}_m2h_{k}": v for k, v in go.items()})

    go_h2m = evaluator.evaluate_go_agreement(predictions_h2m, "human_to_mouse")
    results.update({f"{label}_h2m_{k}": v for k, v in go_h2m.items()})

    # Cycle consistency
    cc = evaluator.evaluate_cycle_consistency(predictions_m2h, predictions_h2m)
    results.update({f"{label}_{k}": v for k, v in cc.items()})

    return results


def run_inference_examples(model, E_h, E_m, cost_computer, fgw_solver,
                            indexers, sparse_matrices, device,
                            Z_h_gpu=None, Z_m_gpu=None, mod_features=None):
    """Run inference on well-known cross-species gene families."""

    # Known test cases
    test_genes = {
        # Mouse → Human
        "mouse_to_human": {
            "Trp53": "ENSMUSG00000059552",   # TP53/TP73/TP63 family
            "Braf": "ENSMUSG00000002413",     # BRAF
            "Ctnnb1": "ENSMUSG00000006932",   # CTNNB1 (beta-catenin, Wnt pathway)
            "Notch1": "ENSMUSG00000026923",   # NOTCH1
            "Jak2": "ENSMUSG00000024789",     # JAK2
        },
        # Human → Mouse
        "human_to_mouse": {
            "TP53": "ENSG00000141510",
            "BRCA1": "ENSG00000012048",
            "EGFR": "ENSG00000146648",
            "KRAS": "ENSG00000133703",
            "TNF": "ENSG00000232810",
        },
    }

    all_results = {}
    model.eval()

    if mod_features is None:
        mod_features = {}

    for direction, gene_dict in test_genes.items():
        if direction == "mouse_to_human":
            src_idx_map = indexers["mouse_genes"]
            tgt_inv = indexers["human_genes_inv"]
            E_src, E_tgt = E_m, E_h
            fgw_dir = "mouse_to_human"
            Z_src, Z_tgt = Z_m_gpu, Z_h_gpu
            Z_src_mods = {
                'bp': mod_features.get('Z_m_BP'),
                'mf': mod_features.get('Z_m_MF'),
                'disease': mod_features.get('Z_m_Disease'),
            }
            Z_tgt_mods = {
                'bp': mod_features.get('Z_h_BP'),
                'mf': mod_features.get('Z_h_MF'),
                'disease': mod_features.get('Z_h_Disease'),
            }
        else:
            src_idx_map = indexers["human_genes"]
            tgt_inv = indexers["mouse_genes_inv"]
            E_src, E_tgt = E_h, E_m
            fgw_dir = "human_to_mouse"
            Z_src, Z_tgt = Z_h_gpu, Z_m_gpu
            Z_src_mods = {
                'bp': mod_features.get('Z_h_BP'),
                'mf': mod_features.get('Z_h_MF'),
                'disease': mod_features.get('Z_h_Disease'),
            }
            Z_tgt_mods = {
                'bp': mod_features.get('Z_m_BP'),
                'mf': mod_features.get('Z_m_MF'),
                'disease': mod_features.get('Z_m_Disease'),
            }

        for symbol, gene_id in gene_dict.items():
            if gene_id not in src_idx_map:
                logger.warning(f"Gene {symbol} ({gene_id}) not found in indexer")
                continue

            src_i = src_idx_map[gene_id]
            I = torch.tensor([src_i], dtype=torch.long, device=device)

            with torch.no_grad():
                J_cand, soft_mask, Pi0, row_mask = model.candidate_selector(
                    E_source=E_src, E_target=E_tgt, source_indices=I,
                    Z_source=Z_src, Z_target=Z_tgt,
                    Z_source_mods=Z_src_mods, Z_target_mods=Z_tgt_mods)

                if len(J_cand) == 0:
                    logger.warning(f"No candidates for {symbol}")
                    continue

                costs = cost_computer.compute(
                    E_source=E_src[src_i:src_i + 1],
                    E_target=E_tgt[J_cand],
                    source_indices=I,
                    target_indices=J_cand,
                    direction=fgw_dir)

                M_masked = costs["M"] * soft_mask
                T_local, fgw_info = fgw_solver.solve(
                    C_s=costs["C_s"], C_t=costs["C_t"],
                    M=M_masked, Pi0=Pi0, mask=row_mask)

                t_row = T_local[0]
                topk = min(20, len(t_row))
                topk_scores, topk_pos = torch.topk(t_row, topk)
                topk_scores = topk_scores / (topk_scores.sum() + 1e-8)

            matched_idx = J_cand[topk_pos].cpu().numpy()
            matched_sc = topk_scores.cpu().numpy()

            # Get gene symbols
            matched_genes = []
            for rank, (idx, score) in enumerate(zip(matched_idx, matched_sc)):
                tgt_id = tgt_inv[int(idx)]
                tgt_symbol = indexers["gene_symbols"].get(tgt_id, tgt_id)
                matched_genes.append({
                    "rank": rank + 1,
                    "id": tgt_id,
                    "symbol": tgt_symbol,
                    "score": float(score),
                })

            result_key = f"{direction}/{symbol}({gene_id})"
            all_results[result_key] = {
                "num_candidates": int(len(J_cand)),
                "fgw_iters": fgw_info["iterations"],
                "top_matches": matched_genes[:10],  # Top 10 for display
            }

            # Pretty print top 10
            logger.info(f"\n{'='*60}")
            logger.info(f"Query: {symbol} ({gene_id}) [{direction}]")
            logger.info(f"Candidates: {len(J_cand)}, FGW iters: {fgw_info['iterations']}")
            logger.info(f"Top 10 matches:")
            for g in matched_genes[:10]:
                logger.info(f"  #{g['rank']:2d}  {g['symbol']:15s} ({g['id']})  score={g['score']:.4f}")

    return all_results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = Config()

    # ── Load model and data ──
    model, graph, sparse_matrices, Z_h, Z_m, indexers, split_info = load_model_and_data(cfg, device)

    # ── Compute embeddings ──
    logger.info("="*60)
    logger.info("Computing embeddings...")
    E_h, E_m = compute_embeddings(model, graph, device)

    # ── Prepare SVD features for candidate selection ──
    # Unified SVD features as tensors on device
    Z_h_gpu = torch.tensor(Z_h, dtype=torch.float32, device=device)
    Z_m_gpu = torch.tensor(Z_m, dtype=torch.float32, device=device)

    # Per-modality gene features from HGT buffers (move to device)
    mod_features = {}
    for key in ['Z_h_BP', 'Z_h_MF', 'Z_h_Disease', 'Z_m_BP', 'Z_m_MF', 'Z_m_Disease']:
        buf = getattr(model.hgt, key, None)
        if buf is not None:
            mod_features[key] = buf.to(device)
    if mod_features:
        logger.info(f"Loaded {len(mod_features)} per-modality gene feature tensors on {device}")
    else:
        logger.warning("No per-modality gene features found — candidate selection will use co-attention only")

    # ── Setup evaluation components ──
    cost_computer = CostComputer(Z_h=Z_h, Z_m=Z_m, device=device)
    fgw_solver = FGWSolver(
        alpha=cfg.fgw.alpha, epsilon=cfg.fgw.epsilon,
        lambda_prior=0, lambda_memory=0,  # No regularization for evaluation
        use_unbalanced=cfg.fgw.use_unbalanced,
        tau_row=cfg.fgw.tau_row, tau_col=cfg.fgw.tau_col,
        max_iter=cfg.infer.fgw_refine_iters,
    )

    evaluator = Evaluator(
        sparse_matrices=sparse_matrices,
        indexers=indexers,
        split_info=split_info,
    )

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: Generate model predictions (sampled)
    # ═══════════════════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("STAGE 1: Generating model predictions (FGW)")
    logger.info("="*60)

    N_SAMPLE = 2000  # Sample for speed (full would take hours)

    pred_m2h = generate_predictions_fast(
        model, E_m, E_h, cost_computer, fgw_solver,
        "mouse_to_human", n_sample=N_SAMPLE, device=device,
        Z_source=Z_m_gpu, Z_target=Z_h_gpu,
        Z_source_mods={
            'bp': mod_features.get('Z_m_BP'),
            'mf': mod_features.get('Z_m_MF'),
            'disease': mod_features.get('Z_m_Disease'),
        },
        Z_target_mods={
            'bp': mod_features.get('Z_h_BP'),
            'mf': mod_features.get('Z_h_MF'),
            'disease': mod_features.get('Z_h_Disease'),
        })

    pred_h2m = generate_predictions_fast(
        model, E_h, E_m, cost_computer, fgw_solver,
        "human_to_mouse", n_sample=N_SAMPLE, device=device,
        Z_source=Z_h_gpu, Z_target=Z_m_gpu,
        Z_source_mods={
            'bp': mod_features.get('Z_h_BP'),
            'mf': mod_features.get('Z_h_MF'),
            'disease': mod_features.get('Z_h_Disease'),
        },
        Z_target_mods={
            'bp': mod_features.get('Z_m_BP'),
            'mf': mod_features.get('Z_m_MF'),
            'disease': mod_features.get('Z_m_Disease'),
        })

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: Evaluate model
    # ═══════════════════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("STAGE 2: Evaluating model")
    logger.info("="*60)

    model_results = evaluate_predictions(evaluator, pred_m2h, pred_h2m, "FGW_Model")
    for k, v in sorted(model_results.items()):
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: Run baselines
    # ═══════════════════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("STAGE 3: Running baselines")
    logger.info("="*60)

    # Use same sample indices as model predictions for fair comparison
    m2h_sample = sorted(pred_m2h.keys())
    h2m_sample = sorted(pred_h2m.keys())

    baseline_preds = run_baselines(E_h, E_m, sparse_matrices,
                                    len(m2h_sample), len(h2m_sample),
                                    Z_h=Z_h, Z_m=Z_m)

    # Filter baselines to same sample indices
    all_eval_results = {"FGW_Model": model_results}

    for bname, bpred in baseline_preds.items():
        # Map baseline predictions (which are indexed 0..n_sample-1) to
        # actual gene indices used in model predictions
        mapped_m2h = {}
        for i, src_idx in enumerate(m2h_sample):
            if i in bpred["m2h"]:
                mapped_m2h[src_idx] = bpred["m2h"][i]

        mapped_h2m = {}
        for i, src_idx in enumerate(h2m_sample):
            if i in bpred["h2m"]:
                mapped_h2m[src_idx] = bpred["h2m"][i]

        b_results = evaluate_predictions(evaluator, mapped_m2h, mapped_h2m, bname)
        all_eval_results[bname] = b_results

        for k, v in sorted(b_results.items()):
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: Comparison summary
    # ═══════════════════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)

    metrics_to_compare = [
        "m2h_heldout_jaccard", "h2m_heldout_jaccard",
        "m2h_go_bp_jaccard_mean", "h2m_go_bp_jaccard_mean",
        "cycle_consistency",
    ]

    header = f"{'Metric':<35s}"
    for method in all_eval_results:
        header += f"  {method:>16s}"
    logger.info(header)
    logger.info("-" * len(header))

    for metric in metrics_to_compare:
        row = f"{metric:<35s}"
        for method, res in all_eval_results.items():
            key = f"{method}_{metric}"
            val = res.get(key, None)
            if val is not None:
                row += f"  {val:>16.4f}"
            else:
                row += f"  {'N/A':>16s}"
        logger.info(row)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 5: Inference on known genes
    # ═══════════════════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("STAGE 5: Inference on known gene families")
    logger.info("="*60)

    inference_results = run_inference_examples(
        model, E_h, E_m, cost_computer, fgw_solver,
        indexers, sparse_matrices, device,
        Z_h_gpu=Z_h_gpu, Z_m_gpu=Z_m_gpu, mod_features=mod_features)

    # ═══════════════════════════════════════════════════════════════
    # Save all results
    # ═══════════════════════════════════════════════════════════════
    output_dir = Path(cfg.paths.proc_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation metrics
    eval_output = {
        "model_checkpoint": "best_model.pt",
        "n_sample": N_SAMPLE,
        "evaluation_results": {},
        "inference_results": inference_results,
    }
    for method, res in all_eval_results.items():
        eval_output["evaluation_results"][method] = {
            k: v for k, v in res.items()
        }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(eval_output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_dir / 'evaluation_results.json'}")

    # Save predictions for downstream analysis
    torch.save({
        "pred_m2h": pred_m2h,
        "pred_h2m": pred_h2m,
        "E_h": E_h.cpu(),
        "E_m": E_m.cpu(),
    }, output_dir / "predictions.pt")
    logger.info(f"Predictions saved to {output_dir / 'predictions.pt'}")

    logger.info("="*60)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
