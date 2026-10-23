"""Evaluation + inference script for trained v2 model.

Runs:
1. Sample inference on well-known genes (Trp53, Brca1, Egfr, etc.)
2. Batch GO-BP agreement evaluation
3. Held-out term evaluation
4. Cosine baseline comparison
5. FGW-based ortholog recovery (Recall@K using FGW coupling scores)

Supports --checkpoint to evaluate specific checkpoints (default: best_model).
Filters unannotated genes from candidate pool using eligible_mask.
"""

import argparse
import logging
import sys
import pickle
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("v2_eval.log"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.data.graph import load_unified_graph
from v2_redesign.model.model import create_model
from v2_redesign.model.hgt import compute_per_modality_gene_features
from v2_redesign.transport.costs import CostComputer
from v2_redesign.transport.fgw import FGWSolver


def compute_eligible_masks(sparse_matrices, min_annotations=3, device="cpu"):
    """Compute boolean masks for genes with sufficient annotations.

    Genes with fewer than min_annotations total annotations (across BP+MF+Disease)
    are excluded from the candidate pool during inference. These poorly-annotated
    genes have near-zero SVD features and act as noise sinks in the embedding space.
    """
    W_hB = sparse_matrices["W_hB"]
    W_hF = sparse_matrices["W_hF"]
    W_hD = sparse_matrices["W_hD"]
    W_mB = sparse_matrices["W_mB"]
    W_mF = sparse_matrices["W_mF"]
    W_mD = sparse_matrices["W_mD"]

    h_counts = (
        np.array((W_hB > 0).sum(axis=1)).squeeze()
        + np.array((W_hF > 0).sum(axis=1)).squeeze()
        + np.array((W_hD > 0).sum(axis=1)).squeeze()
    )
    m_counts = (
        np.array((W_mB > 0).sum(axis=1)).squeeze()
        + np.array((W_mF > 0).sum(axis=1)).squeeze()
        + np.array((W_mD > 0).sum(axis=1)).squeeze()
    )

    h_mask = torch.tensor(h_counts >= min_annotations, dtype=torch.bool, device=device)
    m_mask = torch.tensor(m_counts >= min_annotations, dtype=torch.bool, device=device)

    logger.info(
        f"Eligible genes (>={min_annotations} annotations): "
        f"human {h_mask.sum().item():,}/{len(h_mask):,}, "
        f"mouse {m_mask.sum().item():,}/{len(m_mask):,}"
    )
    return h_mask, m_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename (e.g. checkpoint_epoch_40.pt). Default: best_model.pt")
    parser.add_argument("--min-annotations", type=int, default=3,
                        help="Minimum annotations to be eligible as candidate (default: 3)")
    parser.add_argument("--topc", type=int, default=None,
                        help="Override candidate pool size (default: use config default 128)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override FGW alpha (structure vs feature weight)")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override FGW epsilon (entropic regularization)")
    parser.add_argument("--fusion-mode", type=str, default=None,
                        help="Override fusion mode (attention or concat)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    cfg = Config()
    if args.topc is not None:
        cfg.candidates.topc = args.topc
        logger.info(f"Override topc={args.topc}")
    if args.alpha is not None:
        cfg.fgw.alpha = args.alpha
        logger.info(f"Override alpha={args.alpha}")
    if args.epsilon is not None:
        cfg.fgw.epsilon = args.epsilon
        logger.info(f"Override epsilon={args.epsilon}")
    if args.fusion_mode is not None:
        cfg.train.fusion_mode = args.fusion_mode
        logger.info(f"Override fusion_mode={args.fusion_mode}")
    proc = Path(cfg.paths.proc_dir)

    # ── Load data ──
    logger.info("Loading data...")
    with open(proc / "indexers.pkl", "rb") as f:
        indexers = pickle.load(f)

    sparse_matrices = load_sparse_matrices(str(proc))
    Z_h, Z_m, _, split_info = load_svd_artifacts(str(proc))
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    logger.info(f"Human genes: {len(indexers['human_genes']):,}")
    logger.info(f"Mouse genes: {len(indexers['mouse_genes']):,}")

    # ── Compute eligible masks ──
    h_eligible, m_eligible = compute_eligible_masks(
        sparse_matrices, min_annotations=args.min_annotations, device=device)

    # ── Load model ──
    ckpt_name = args.checkpoint or "best_model.pt"
    ckpt_path = proc / "runs" / ckpt_name
    logger.info(f"Loading model from {ckpt_path}...")

    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }
    model = create_model(cfg, node_counts)
    # Compute per-modality gene features (needed for fusion, also initializes term embeddings)
    model.hgt.init_from_svd(Z_h, Z_m, sparse_matrices)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # strict=False: old checkpoints lack per-modality buffers (Z_h_BP etc.)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    logger.info(f"Loaded checkpoint: epoch {ckpt['epoch']}")

    # CPU HGT + GPU fusion split (same as training — HGT graph too large for GPU)
    model.hgt.to("cpu")
    model.fusion_human.to(device)
    model.fusion_mouse.to(device)
    model.candidate_selector.to(device)
    model.eval()

    # ── Build edge indices: CPU for HGT, GPU for fusion ──
    logger.info("Computing embeddings (CPU HGT + GPU Fusion)...")
    cpu_edge_index_dict = {}
    cpu_edge_attr_dict = {}
    for et in graph.edge_types:
        ed = graph[et]
        cpu_edge_index_dict[et] = ed.edge_index
        if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
            cpu_edge_attr_dict[et] = ed.edge_attr

    def _get_gpu_ei(key):
        ei = cpu_edge_index_dict.get(key)
        ea = cpu_edge_attr_dict.get(key)
        return (ei.to(device) if ei is not None else None,
                ea.to(device) if ea is not None else None)

    ei_hb, ea_hb = _get_gpu_ei(("Human_Gene", "HG-BP", "BP"))
    ei_hf, ea_hf = _get_gpu_ei(("Human_Gene", "HG-MF", "MF"))
    ei_hd, ea_hd = _get_gpu_ei(("Human_Gene", "HG-D", "Disease"))
    ei_mb, ea_mb = _get_gpu_ei(("Mouse_Gene", "MG-BP", "BP"))
    ei_mf_e, ea_mf_e = _get_gpu_ei(("Mouse_Gene", "MG-MF", "MF"))
    ei_md, ea_md = _get_gpu_ei(("Mouse_Gene", "MG-D", "Disease"))

    # Per-modality gene features on GPU for fusion
    mod_feat = {}
    for key in ['Z_h_BP', 'Z_h_MF', 'Z_h_Disease', 'Z_m_BP', 'Z_m_MF', 'Z_m_Disease']:
        buf = getattr(model.hgt, key, None)
        if buf is not None:
            mod_feat[key] = buf.to(device)
    logger.info(f"Per-modality gene features: {len(mod_feat)} tensors on GPU")

    with torch.no_grad():
        emb_dict = model.hgt(cpu_edge_index_dict, cpu_edge_attr_dict)

        E_h_raw = emb_dict["Human_Gene"].to(device)
        E_m_raw = emb_dict["Mouse_Gene"].to(device)
        E_bp = emb_dict["BP"].to(device)
        E_mf = emb_dict["MF"].to(device)
        E_dis = emb_dict["Disease"].to(device)

        E_h, _, ent_h = model.fusion_human(
            E_h_raw, E_bp, E_mf, E_dis,
            ei_bp=ei_hb, ei_mf=ei_hf, ei_disease=ei_hd,
            ea_bp=ea_hb, ea_mf=ea_hf, ea_disease=ea_hd,
            bp_gene_feat=mod_feat.get('Z_h_BP'),
            mf_gene_feat=mod_feat.get('Z_h_MF'),
            dis_gene_feat=mod_feat.get('Z_h_Disease'))

        E_m, _, ent_m = model.fusion_mouse(
            E_m_raw, E_bp, E_mf, E_dis,
            ei_bp=ei_mb, ei_mf=ei_mf_e, ei_disease=ei_md,
            ea_bp=ea_mb, ea_mf=ea_mf_e, ea_disease=ea_md,
            bp_gene_feat=mod_feat.get('Z_m_BP'),
            mf_gene_feat=mod_feat.get('Z_m_MF'),
            dis_gene_feat=mod_feat.get('Z_m_Disease'))

        # Center embeddings (remove dominant direction to break isotropy)
        E_h = E_h - E_h.mean(dim=0, keepdim=True)
        E_m = E_m - E_m.mean(dim=0, keepdim=True)

    logger.info(f"Embeddings: E_h {E_h.shape}, E_m {E_m.shape}")

    # ── Embedding quality diagnostics ──
    logger.info("\n--- Embedding quality diagnostics ---")
    with torch.no_grad():
        # Check cosine similarity distribution for random pairs
        rng_d = np.random.RandomState(123)
        rand_h = rng_d.choice(E_h.shape[0], 1000, replace=False)
        rand_m = rng_d.choice(E_m.shape[0], 1000, replace=False)
        cos_hh = (E_h[rand_h[:500]] @ E_h[rand_h[500:]].t()).diagonal()
        cos_mm = (E_m[rand_m[:500]] @ E_m[rand_m[500:]].t()).diagonal()
        logger.info(f"  Random human-human cosine: mean={cos_hh.mean():.4f}, std={cos_hh.std():.4f}")
        logger.info(f"  Random mouse-mouse cosine: mean={cos_mm.mean():.4f}, std={cos_mm.std():.4f}")

        # Embedding-space ortholog check (cosine — quick diagnostic only)
        ortho_pairs_sample = [
            ("ENSMUSG00000059552", "ENSG00000141510", "Trp53", "TP53"),
            ("ENSMUSG00000017146", "ENSG00000012048", "Brca1", "BRCA1"),
            ("ENSMUSG00000020122", "ENSG00000146648", "Egfr", "EGFR"),
        ]
        E_h_n = torch.nn.functional.normalize(E_h, p=2, dim=-1)
        E_m_n = torch.nn.functional.normalize(E_m, p=2, dim=-1)
        for m_id, h_id, m_sym, h_sym in ortho_pairs_sample:
            mi = indexers["mouse_genes"][m_id]
            hi = indexers["human_genes"][h_id]
            cos = (E_m_n[mi] @ E_h_n[hi]).item()
            sims = (E_m_n[mi:mi+1] @ E_h_n.t()).squeeze()
            rank = (sims > sims[hi]).sum().item() + 1
            logger.info(f"  {m_sym}->{h_sym}: emb_cos={cos:.4f}, emb_rank={rank}")
        logger.info("  (Full FGW-based ortholog eval in Part 5 below)")

    # ── Setup FGW solver + cost computer ──
    fc = cfg.fgw
    fgw_solver = FGWSolver(
        alpha=fc.alpha, epsilon=fc.epsilon,
        lambda_prior=0, lambda_memory=0,
        use_unbalanced=fc.use_unbalanced,
        tau_row=fc.tau_row, tau_col=fc.tau_col,
        max_iter=cfg.infer.fgw_refine_iters,
    )
    cost_computer = CostComputer(Z_h=Z_h, Z_m=Z_m, device=device)

    # SVD features on GPU for candidate retrieval
    Z_h_gpu = cost_computer.Z_h  # already on device
    Z_m_gpu = cost_computer.Z_m

    # Per-modality retrieval dicts (for multi-source candidate selection)
    Z_h_mods = {
        'bp': mod_feat.get('Z_h_BP'),
        'mf': mod_feat.get('Z_h_MF'),
        'disease': mod_feat.get('Z_h_Disease'),
    }
    Z_m_mods = {
        'bp': mod_feat.get('Z_m_BP'),
        'mf': mod_feat.get('Z_m_MF'),
        'disease': mod_feat.get('Z_m_Disease'),
    }
    has_mod_retrieval = all(v is not None for v in Z_h_mods.values())
    logger.info(f"Per-modality candidate retrieval: {'enabled' if has_mod_retrieval else 'disabled (fallback to unified SVD)'}")

    # ══════════════════════════════════════════
    # Part 1: Sample inference on known genes
    # ══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Part 1: Sample inference on well-known genes")

    test_genes = {
        "mouse_to_human": [
            ("ENSMUSG00000059552", "Trp53"),
            ("ENSMUSG00000017146", "Brca1"),
            ("ENSMUSG00000020122", "Egfr"),
            ("ENSMUSG00000029026", "Trp73"),
            ("ENSMUSG00000026923", "Notch1"),
        ],
        "human_to_mouse": [
            ("ENSG00000141510", "TP53"),
            ("ENSG00000012048", "BRCA1"),
            ("ENSG00000146648", "EGFR"),
            ("ENSG00000078900", "TP73"),
            ("ENSG00000148400", "NOTCH1"),
        ],
    }

    all_results = {}

    for direction, gene_list in test_genes.items():
        logger.info(f"\n--- Direction: {direction} ---")
        if direction == "mouse_to_human":
            E_src, E_tgt = E_m, E_h
            Z_src, Z_tgt = Z_m_gpu, Z_h_gpu
            Z_src_mods, Z_tgt_mods = Z_m_mods, Z_h_mods
            src_map = indexers["mouse_genes"]
            tgt_inv = indexers["human_genes_inv"]
            tgt_eligible = h_eligible
            symbols = indexers["gene_symbols"]
        else:
            E_src, E_tgt = E_h, E_m
            Z_src, Z_tgt = Z_h_gpu, Z_m_gpu
            Z_src_mods, Z_tgt_mods = Z_h_mods, Z_m_mods
            src_map = indexers["human_genes"]
            tgt_inv = indexers["mouse_genes_inv"]
            tgt_eligible = m_eligible
            symbols = indexers["gene_symbols"]

        for gene_id, gene_name in gene_list:
            if gene_id not in src_map:
                logger.warning(f"  {gene_name} ({gene_id}) not found in indexers")
                continue

            src_idx = src_map[gene_id]
            I_batch = torch.tensor([src_idx], dtype=torch.long, device=device)

            with torch.no_grad():
                J_cand, soft_mask, Pi0, row_mask = model.candidate_selector(
                    E_source=E_src, E_target=E_tgt, source_indices=I_batch,
                    eligible_mask=tgt_eligible,
                    Z_source=Z_src, Z_target=Z_tgt,
                    Z_source_mods=Z_src_mods if has_mod_retrieval else None,
                    Z_target_mods=Z_tgt_mods if has_mod_retrieval else None)

                if len(J_cand) == 0:
                    logger.warning(f"  {gene_name}: No candidates found")
                    continue

                costs = cost_computer.compute(
                    E_source=E_src[src_idx:src_idx + 1], E_target=E_tgt[J_cand],
                    source_indices=I_batch, target_indices=J_cand,
                    direction=direction,
                )
                M_masked = costs["M"] * soft_mask
                T_local, _ = fgw_solver.solve(
                    C_s=costs["C_s"], C_t=costs["C_t"],
                    M=M_masked, Pi0=Pi0, mask=row_mask)

                t_row = T_local[0]
                topk = min(20, len(t_row))
                topk_scores, topk_pos = torch.topk(t_row, topk)
                topk_scores = topk_scores / (topk_scores.sum() + 1e-8)

            matched_idx = J_cand[topk_pos].cpu().numpy()
            matched_sc = topk_scores.cpu().numpy()

            logger.info(f"\n  {gene_name} ({gene_id}) → Top-10 matches:")
            gene_results = []
            for rank, (idx, score) in enumerate(zip(matched_idx[:10], matched_sc[:10])):
                tgt_id = tgt_inv[int(idx)]
                tgt_sym = symbols.get(tgt_id, tgt_id)
                logger.info(f"    #{rank+1}: {tgt_sym} ({tgt_id}) score={score:.4f}")
                gene_results.append({
                    "rank": rank + 1, "id": tgt_id,
                    "symbol": tgt_sym, "score": float(score),
                })

            all_results[f"{gene_name}_{direction}"] = {
                "query": {"id": gene_id, "symbol": gene_name},
                "direction": direction,
                "num_candidates": len(J_cand),
                "matches": gene_results,
            }

    # ── Hub frequency analysis ──
    from collections import Counter
    target_freq = Counter()
    for key, result in all_results.items():
        for match in result["matches"][:10]:
            target_freq[match["symbol"]] += 1

    logger.info("\n--- Hub analysis (target frequency across all queries' top-10) ---")
    n_queries = len(all_results)
    hubs_found = [(gene, count) for gene, count in target_freq.most_common(30) if count > 1]
    if hubs_found:
        for gene, count in hubs_found:
            logger.info(f"  {gene}: appears in {count}/{n_queries} queries")
        logger.info(f"  Total unique targets in top-10: {len(target_freq)}")
        logger.info(f"  Hub genes (>1 query): {len(hubs_found)}/{len(target_freq)}")
    else:
        logger.info("  No hub genes detected — each target appears in at most 1 query.")

    # ══════════════════════════════════════════
    # Part 2: Batch evaluation — GO agreement
    # ══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part 2: Batch evaluation — GO-BP agreement")

    n_eval = 500
    W_mB = sparse_matrices["W_mB"]
    W_hB = sparse_matrices["W_hB"]

    annotated_mouse = [i for i in range(W_mB.shape[0]) if W_mB[i].nnz > 0]
    rng = np.random.RandomState(42)
    eval_indices = rng.choice(annotated_mouse, min(n_eval, len(annotated_mouse)), replace=False)

    jaccards = []
    m2h_predictions = {}
    batch_target_freq = Counter()  # Track hub frequency across batch eval

    for count, src_idx in enumerate(eval_indices):
        I_batch = torch.tensor([src_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            J_cand, soft_mask, Pi0, row_mask = model.candidate_selector(
                E_source=E_m, E_target=E_h, source_indices=I_batch,
                eligible_mask=h_eligible,
                Z_source=Z_m_gpu, Z_target=Z_h_gpu,
                Z_source_mods=Z_m_mods if has_mod_retrieval else None,
                Z_target_mods=Z_h_mods if has_mod_retrieval else None)

            if len(J_cand) == 0:
                continue

            costs = cost_computer.compute(
                E_source=E_m[src_idx:src_idx + 1], E_target=E_h[J_cand],
                source_indices=I_batch, target_indices=J_cand,
                direction="mouse_to_human",
            )
            M_masked = costs["M"] * soft_mask
            T_local, _ = fgw_solver.solve(
                C_s=costs["C_s"], C_t=costs["C_t"],
                M=M_masked, Pi0=Pi0, mask=row_mask)

            t_row = T_local[0]
            topk = min(20, len(t_row))
            topk_scores, topk_pos = torch.topk(t_row, topk)

        matched_idx = J_cand[topk_pos].cpu().numpy()

        src_bp = set(W_mB[src_idx].nonzero()[1])
        if not src_bp:
            continue
        for tgt_idx in matched_idx[:5]:
            tgt_bp = set(W_hB[int(tgt_idx)].nonzero()[1])
            if src_bp or tgt_bp:
                union = len(src_bp | tgt_bp)
                inter = len(src_bp & tgt_bp)
                if union > 0:
                    jaccards.append(inter / union)

        m2h_predictions[src_idx] = [(int(idx), float(sc))
                                     for idx, sc in zip(matched_idx, topk_scores.cpu().numpy())]

        # Track target frequency for hub analysis
        for tgt_idx in matched_idx[:5]:
            batch_target_freq[int(tgt_idx)] += 1

        if (count + 1) % 100 == 0:
            logger.info(f"  Evaluated {count + 1}/{len(eval_indices)} genes...")

    go_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
    logger.info(f"\nGO-BP Jaccard (top-5, {len(eval_indices)} genes): {go_jaccard:.4f}")
    logger.info(f"  Total Jaccard pairs: {len(jaccards)}")

    # Batch hub analysis
    n_unique_targets = len(batch_target_freq)
    n_total_slots = len(eval_indices) * 5
    top_hubs = batch_target_freq.most_common(10)
    logger.info(f"\n--- Batch hub analysis (top-5, {len(eval_indices)} queries) ---")
    logger.info(f"  Unique targets selected: {n_unique_targets} / {n_total_slots} slots")
    logger.info(f"  Target reuse ratio: {n_total_slots / max(n_unique_targets, 1):.1f}x")
    for idx, count in top_hubs:
        tgt_id = indexers["human_genes_inv"][idx]
        tgt_sym = indexers["gene_symbols"].get(tgt_id, tgt_id)
        logger.info(f"  Hub: {tgt_sym} ({tgt_id}) — selected by {count}/{len(eval_indices)} queries")

    # ══════════════════════════════════════════
    # Part 3: Held-out term evaluation
    # ══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part 3: Held-out term evaluation")

    bp_held = set(split_info.get("bp_held", []))
    mf_held = set(split_info.get("mf_held", []))
    dis_held = set(split_info.get("dis_held", []))
    logger.info(f"Held-out terms: BP={len(bp_held)}, MF={len(mf_held)}, Disease={len(dis_held)}")

    W_mF = sparse_matrices["W_mF"]
    W_hF = sparse_matrices["W_hF"]
    W_mD = sparse_matrices["W_mD"]
    W_hD = sparse_matrices["W_hD"]

    heldout_jaccards = []
    for src_idx, pred_list in m2h_predictions.items():
        src_bp_held = set(W_mB[src_idx].nonzero()[1]) & bp_held
        src_mf_held = set(W_mF[src_idx].nonzero()[1]) & mf_held
        src_dis_held = set(W_mD[src_idx].nonzero()[1]) & dis_held
        src_held = src_bp_held | src_mf_held | src_dis_held

        if not src_held:
            continue

        tgt_held = set()
        for tgt_idx, score in pred_list[:20]:
            tgt_held |= set(W_hB[tgt_idx].nonzero()[1]) & bp_held
            tgt_held |= set(W_hF[tgt_idx].nonzero()[1]) & mf_held
            tgt_held |= set(W_hD[tgt_idx].nonzero()[1]) & dis_held

        if src_held or tgt_held:
            union = len(src_held | tgt_held)
            inter = len(src_held & tgt_held)
            if union > 0:
                heldout_jaccards.append(inter / union)

    heldout_jaccard = float(np.mean(heldout_jaccards)) if heldout_jaccards else 0.0
    logger.info(f"Held-out Jaccard@20: {heldout_jaccard:.4f} ({len(heldout_jaccards)} genes evaluated)")

    # ══════════════════════════════════════════
    # Part 4: Baseline comparison — cosine similarity (no FGW)
    # ══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part 4: Baseline — cosine nearest neighbor (no FGW)")

    baseline_jaccards = []
    with torch.no_grad():
        for src_idx in eval_indices[:200]:
            src_emb = E_m[src_idx:src_idx + 1]
            sims = (src_emb @ E_h.t()).squeeze(0)
            # Also apply eligible mask to cosine baseline
            sims[~h_eligible] = float("-inf")
            topk_scores_b, topk_idx_b = torch.topk(sims, 5)

            src_bp = set(W_mB[src_idx].nonzero()[1])
            if not src_bp:
                continue
            for tgt_idx in topk_idx_b.cpu().numpy():
                tgt_bp = set(W_hB[int(tgt_idx)].nonzero()[1])
                union = len(src_bp | tgt_bp)
                if union > 0:
                    baseline_jaccards.append(len(src_bp & tgt_bp) / union)

    baseline_jaccard = float(np.mean(baseline_jaccards)) if baseline_jaccards else 0.0
    logger.info(f"Baseline cosine GO-BP Jaccard (top-5, 200 genes): {baseline_jaccard:.4f}")
    logger.info(f"FGW model GO-BP Jaccard (top-5, {len(eval_indices)} genes): {go_jaccard:.4f}")

    # ══════════════════════════════════════════
    # Part 5: FGW-based ortholog recovery
    # ══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part 5: FGW-based ortholog recovery (Ensembl Compara)")

    ortho_path = Path("/Genomics/chanlab/Ze/MGPmodel/Porject2_Gene2Gene/data_cache/human_mouse_orthologs.tsv")
    if ortho_path.exists():
        import csv
        # Load 1:1 orthologs that exist in our gene sets
        ortho_1to1 = []
        with open(ortho_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                h_id = row["Gene stable ID"]
                m_id = row["Mouse gene stable ID"]
                htype = row["Mouse homology type"]
                if (htype == "ortholog_one2one"
                        and h_id in indexers["human_genes"]
                        and m_id in indexers["mouse_genes"]):
                    hi = indexers["human_genes"][h_id]
                    mi = indexers["mouse_genes"][m_id]
                    # Only use eligible genes
                    if h_eligible[hi] and m_eligible[mi]:
                        ortho_1to1.append((mi, hi, m_id, h_id))

        logger.info(f"  Loaded {len(ortho_1to1)} eligible 1:1 orthologs")

        # Sample for tractability (FGW per query is expensive)
        n_ortho_eval = min(500, len(ortho_1to1))
        rng_o = np.random.RandomState(7)
        ortho_sample = [ortho_1to1[i] for i in rng_o.choice(len(ortho_1to1), n_ortho_eval, replace=False)]

        # For each sampled ortholog pair, run FGW and check rank of true ortholog
        recall_at = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
        fgw_ranks = []
        in_candidate_count = 0

        for count_o, (mi, hi, m_id, h_id) in enumerate(ortho_sample):
            I_batch = torch.tensor([mi], dtype=torch.long, device=device)

            with torch.no_grad():
                J_cand, soft_mask, Pi0, row_mask = model.candidate_selector(
                    E_source=E_m, E_target=E_h, source_indices=I_batch,
                    eligible_mask=h_eligible,
                    Z_source=Z_m_gpu, Z_target=Z_h_gpu)

                if len(J_cand) == 0:
                    continue

                # Check if true ortholog is in candidate set
                cand_set = set(J_cand.cpu().numpy().tolist())
                ortho_in_cand = hi in cand_set

                if ortho_in_cand:
                    in_candidate_count += 1

                costs = cost_computer.compute(
                    E_source=E_m[mi:mi+1], E_target=E_h[J_cand],
                    source_indices=I_batch, target_indices=J_cand,
                    direction="mouse_to_human",
                )
                M_masked = costs["M"] * soft_mask
                T_local, _ = fgw_solver.solve(
                    C_s=costs["C_s"], C_t=costs["C_t"],
                    M=M_masked, Pi0=Pi0, mask=row_mask)

                t_row = T_local[0]
                # Sort by FGW score descending
                sorted_scores, sorted_pos = torch.sort(t_row, descending=True)
                sorted_cand_idx = J_cand[sorted_pos].cpu().numpy()

                # Find rank of true ortholog in FGW-sorted list
                if ortho_in_cand:
                    rank_positions = np.where(sorted_cand_idx == hi)[0]
                    if len(rank_positions) > 0:
                        fgw_rank = int(rank_positions[0]) + 1
                        fgw_ranks.append(fgw_rank)
                        for k in recall_at:
                            if fgw_rank <= k:
                                recall_at[k] += 1

            if (count_o + 1) % 100 == 0:
                logger.info(f"  Ortholog eval: {count_o + 1}/{n_ortho_eval}...")

        n_evaluated = len(fgw_ranks)
        logger.info(f"\n--- FGW Ortholog Recovery ({n_ortho_eval} sampled, {n_evaluated} ranked) ---")
        logger.info(f"  Orthologs in candidate set: {in_candidate_count}/{n_ortho_eval} "
                     f"({100 * in_candidate_count / max(n_ortho_eval, 1):.1f}%)")
        for k in sorted(recall_at.keys()):
            r = recall_at[k] / max(n_evaluated, 1)
            logger.info(f"  Recall@{k:>3d}: {recall_at[k]:>4d}/{n_evaluated} = {r:.4f}")
        if fgw_ranks:
            logger.info(f"  Median rank: {int(np.median(fgw_ranks))}")
            logger.info(f"  Mean rank:   {np.mean(fgw_ranks):.1f}")
            logger.info(f"  MRR:         {np.mean([1.0/r for r in fgw_ranks]):.4f}")

        # Re-run a few specific well-known orthologs to show their FGW ranks
        logger.info("\n  Well-known ortholog FGW ranks:")
        known_ortho = [
            ("ENSMUSG00000059552", "ENSG00000141510", "Trp53", "TP53"),
            ("ENSMUSG00000017146", "ENSG00000012048", "Brca1", "BRCA1"),
            ("ENSMUSG00000020122", "ENSG00000146648", "Egfr", "EGFR"),
            ("ENSMUSG00000029026", "ENSG00000078900", "Trp73", "TP73"),
            ("ENSMUSG00000026923", "ENSG00000148400", "Notch1", "NOTCH1"),
        ]
        for m_id, h_id, m_sym, h_sym in known_ortho:
            if m_id not in indexers["mouse_genes"] or h_id not in indexers["human_genes"]:
                continue
            mi = indexers["mouse_genes"][m_id]
            hi = indexers["human_genes"][h_id]
            I_b = torch.tensor([mi], dtype=torch.long, device=device)
            with torch.no_grad():
                J_c, sm, P0, rm = model.candidate_selector(
                    E_source=E_m, E_target=E_h, source_indices=I_b,
                    eligible_mask=h_eligible,
                    Z_source=Z_m_gpu, Z_target=Z_h_gpu,
                    Z_source_mods=Z_m_mods if has_mod_retrieval else None,
                    Z_target_mods=Z_h_mods if has_mod_retrieval else None)
                if len(J_c) == 0:
                    logger.info(f"    {m_sym}->{h_sym}: no candidates")
                    continue
                in_cand = hi in set(J_c.cpu().numpy().tolist())
                cs = cost_computer.compute(
                    E_source=E_m[mi:mi+1], E_target=E_h[J_c],
                    source_indices=I_b, target_indices=J_c,
                    direction="mouse_to_human")
                Mm = cs["M"] * sm
                Tl, _ = fgw_solver.solve(C_s=cs["C_s"], C_t=cs["C_t"], M=Mm, Pi0=P0, mask=rm)
                tr = Tl[0]
                _, sp = torch.sort(tr, descending=True)
                sc = J_c[sp].cpu().numpy()
                if in_cand:
                    rp = np.where(sc == hi)[0]
                    rk = int(rp[0]) + 1 if len(rp) > 0 else -1
                    score = tr[sp[rk-1]].item() if rk > 0 else 0
                    logger.info(f"    {m_sym}->{h_sym}: FGW rank={rk}/{len(J_c)}, score={score:.4f}")
                else:
                    logger.info(f"    {m_sym}->{h_sym}: not in candidates ({len(J_c)} candidates)")
    else:
        logger.warning(f"  Ortholog file not found: {ortho_path}")
        n_evaluated = 0
        recall_at = {}
        fgw_ranks = []
        in_candidate_count = 0
        n_ortho_eval = 0

    # ══════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model checkpoint: {ckpt_name} (epoch {ckpt['epoch']})")
    logger.info(f"Min annotations filter: {args.min_annotations}")
    logger.info(f"GO-BP Jaccard (FGW, top-5):       {go_jaccard:.4f}")
    logger.info(f"GO-BP Jaccard (cosine baseline):   {baseline_jaccard:.4f}")
    logger.info(f"Held-out Jaccard@20:               {heldout_jaccard:.4f}")
    logger.info(f"Unique targets / slots:            {n_unique_targets}/{n_total_slots} ({n_total_slots / max(n_unique_targets, 1):.1f}x reuse)")
    logger.info(f"Hub genes in sample inference:     {len(hubs_found)}")
    if fgw_ranks:
        for k in sorted(recall_at.keys()):
            r = recall_at[k] / max(n_evaluated, 1)
            logger.info(f"Ortholog Recall@{k:<3d} (FGW):        {r:.4f}")
        logger.info(f"Ortholog MRR (FGW):               {np.mean([1.0/r for r in fgw_ranks]):.4f}")
        logger.info(f"Ortholog in candidates:            {in_candidate_count}/{n_ortho_eval} ({100*in_candidate_count/max(n_ortho_eval,1):.1f}%)")

    # Save results
    suffix = ckpt_name.replace(".pt", "").replace("/", "_")
    summary = {
        "model_epoch": ckpt["epoch"],
        "checkpoint": ckpt_name,
        "min_annotations": args.min_annotations,
        "go_bp_jaccard_fgw": go_jaccard,
        "go_bp_jaccard_baseline": baseline_jaccard,
        "heldout_jaccard_at_20": heldout_jaccard,
        "n_eval_genes": len(eval_indices),
        "n_eligible_human": int(h_eligible.sum().item()),
        "n_eligible_mouse": int(m_eligible.sum().item()),
        "hub_stats": {
            "unique_targets": n_unique_targets,
            "total_slots": n_total_slots,
            "reuse_ratio": n_total_slots / max(n_unique_targets, 1),
            "sample_hub_count": len(hubs_found),
            "top_batch_hubs": [
                {"id": indexers["human_genes_inv"][idx],
                 "symbol": indexers["gene_symbols"].get(indexers["human_genes_inv"][idx], "?"),
                 "count": cnt}
                for idx, cnt in top_hubs
            ],
        },
        "sample_inferences": all_results,
        "ortholog_eval": {
            "n_sampled": n_ortho_eval,
            "n_evaluated": n_evaluated if fgw_ranks else 0,
            "in_candidate_pct": 100 * in_candidate_count / max(n_ortho_eval, 1),
            "recall_at": {str(k): recall_at[k] / max(n_evaluated, 1) for k in sorted(recall_at.keys())} if fgw_ranks else {},
            "mrr": float(np.mean([1.0/r for r in fgw_ranks])) if fgw_ranks else 0.0,
            "median_rank": int(np.median(fgw_ranks)) if fgw_ranks else 0,
            "mean_rank": float(np.mean(fgw_ranks)) if fgw_ranks else 0.0,
        },
    }
    out_path = proc / "runs" / f"eval_results_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
