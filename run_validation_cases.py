"""Run Tier A validation cases across ALL checkpoints.

6 cases x multiple checkpoints -> comprehensive validation table.
"""
import json
import logging
import pickle
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.data.graph import load_unified_graph
from v2_redesign.model.model import create_model
from v2_redesign.transport.costs import CostComputer
from v2_redesign.transport.fgw import FGWSolver


# ── Tier A Validation Cases ──
TIER_A_CASES = [
    {
        "name": "DMD (Duchenne MD)",
        "human_gene": "ENSG00000198947",  # DMD
        "human_symbol": "DMD",
        "expected_mouse": {
            "strict": ["ENSMUSG00000045103"],  # Dmd
            "paralog_set": ["ENSMUSG00000045103", "ENSMUSG00000019820"],  # Dmd, Utrn
        },
        "expected_symbols": {"ENSMUSG00000045103": "Dmd", "ENSMUSG00000019820": "Utrn"},
        "match_type": "paralog_compensation",
    },
    {
        "name": "ABCD1 (X-ALD)",
        "human_gene": "ENSG00000101986",  # ABCD1
        "human_symbol": "ABCD1",
        "expected_mouse": {
            "strict": ["ENSMUSG00000031378"],  # Abcd1
            "paralog_set": ["ENSMUSG00000031378", "ENSMUSG00000055782"],  # Abcd1, Abcd2
        },
        "expected_symbols": {"ENSMUSG00000031378": "Abcd1", "ENSMUSG00000055782": "Abcd2"},
        "match_type": "paralog_compensation",
    },
    {
        "name": "RB1 (Retinoblastoma)",
        "human_gene": "ENSG00000139687",  # RB1
        "human_symbol": "RB1",
        "expected_mouse": {
            "strict": ["ENSMUSG00000022105"],  # Rb1
            "paralog_set": ["ENSMUSG00000022105", "ENSMUSG00000027641", "ENSMUSG00000031666"],  # Rb1, Rbl1, Rbl2
        },
        "expected_symbols": {"ENSMUSG00000022105": "Rb1", "ENSMUSG00000027641": "Rbl1", "ENSMUSG00000031666": "Rbl2"},
        "match_type": "paralog_compensation",
    },
    {
        "name": "SHOX (Short stature)",
        "human_gene": "ENSG00000185960",  # SHOX
        "human_symbol": "SHOX",
        "expected_mouse": {
            "strict": ["ENSMUSG00000027833"],  # Shox2 (SHOX absent in mouse)
            "paralog_set": ["ENSMUSG00000027833"],
        },
        "expected_symbols": {"ENSMUSG00000027833": "Shox2"},
        "match_type": "non_ortholog_analog",
    },
    {
        "name": "SERPINA1 (A1AT deficiency)",
        "human_gene": "ENSG00000277377",  # SERPINA1
        "human_symbol": "SERPINA1",
        "expected_mouse": {
            "strict": ["ENSMUSG00000071177"],  # Serpina1a
            "paralog_set": ["ENSMUSG00000071177", "ENSMUSG00000071178",
                           "ENSMUSG00000079015", "ENSMUSG00000072849"],  # a, b, c, e
        },
        "expected_symbols": {
            "ENSMUSG00000071177": "Serpina1a", "ENSMUSG00000071178": "Serpina1b",
            "ENSMUSG00000079015": "Serpina1c", "ENSMUSG00000072849": "Serpina1e",
        },
        "match_type": "multi_paralog_set",
    },
    {
        "name": "CFTR (CF lung disease)",
        "human_gene": "ENSG00000001626",  # CFTR
        "human_symbol": "CFTR",
        "expected_mouse": {
            "strict": ["ENSMUSG00000041301"],  # Cftr
            "paralog_set": ["ENSMUSG00000041301", "ENSMUSG00000030873",
                           "ENSMUSG00000030340", "ENSMUSG00000000216"],  # Cftr, Scnn1b, Scnn1a, Scnn1g
        },
        "expected_symbols": {
            "ENSMUSG00000041301": "Cftr", "ENSMUSG00000030873": "Scnn1b",
            "ENSMUSG00000030340": "Scnn1a", "ENSMUSG00000000216": "Scnn1g",
        },
        "match_type": "pathway_module",
    },
]

# ── Tier B Validation Cases ──
TIER_B_CASES = [
    {
        "name": "Dyskeratosis congenita (DKC1/telomere)",
        "human_gene": "ENSG00000130826",  # DKC1
        "human_symbol": "DKC1",
        "expected_mouse": {
            "strict": ["ENSMUSG00000024174"],  # Pot1b
            "paralog_set": ["ENSMUSG00000024174", "ENSMUSG00000118406",
                           "ENSMUSG00000021611"],  # Pot1b, Terc, Tert
        },
        "expected_symbols": {
            "ENSMUSG00000024174": "Pot1b", "ENSMUSG00000118406": "Terc",
            "ENSMUSG00000021611": "Tert",
        },
        "match_type": "multi_gene_substitute",
    },
    {
        "name": "Dyskeratosis congenita (TERT)",
        "human_gene": "ENSG00000164362",  # TERT
        "human_symbol": "TERT",
        "expected_mouse": {
            "strict": ["ENSMUSG00000021611"],  # Tert
            "paralog_set": ["ENSMUSG00000021611", "ENSMUSG00000024174",
                           "ENSMUSG00000118406"],  # Tert, Pot1b, Terc
        },
        "expected_symbols": {
            "ENSMUSG00000021611": "Tert", "ENSMUSG00000024174": "Pot1b",
            "ENSMUSG00000118406": "Terc",
        },
        "match_type": "multi_gene_substitute",
    },
    {
        "name": "CXCL8/IL8 inflammation",
        "human_gene": "ENSG00000169429",  # CXCL8
        "human_symbol": "CXCL8",
        "expected_mouse": {
            "strict": ["ENSMUSG00000029380"],  # Cxcl1
            "paralog_set": ["ENSMUSG00000029380", "ENSMUSG00000058427",
                           "ENSMUSG00000029371"],  # Cxcl1, Cxcl2, Cxcl5
        },
        "expected_symbols": {
            "ENSMUSG00000029380": "Cxcl1", "ENSMUSG00000058427": "Cxcl2",
            "ENSMUSG00000029371": "Cxcl5",
        },
        "match_type": "missing_ortholog_module",
    },
    {
        "name": "TLR8 sensing",
        "human_gene": "ENSG00000101916",  # TLR8
        "human_symbol": "TLR8",
        "expected_mouse": {
            "strict": ["ENSMUSG00000040522"],  # Tlr8
            "paralog_set": ["ENSMUSG00000040522", "ENSMUSG00000033777"],  # Tlr8, Tlr13
        },
        "expected_symbols": {
            "ENSMUSG00000040522": "Tlr8", "ENSMUSG00000033777": "Tlr13",
        },
        "match_type": "non_ortholog_analog",
    },
    {
        "name": "MMP1 disease biology",
        "human_gene": "ENSG00000196611",  # MMP1
        "human_symbol": "MMP1",
        "expected_mouse": {
            "strict": ["ENSMUSG00000043089"],  # Mmp1a
            "paralog_set": ["ENSMUSG00000043089", "ENSMUSG00000041620",
                           "ENSMUSG00000050578"],  # Mmp1a, Mmp1b, Mmp13
        },
        "expected_symbols": {
            "ENSMUSG00000043089": "Mmp1a", "ENSMUSG00000041620": "Mmp1b",
            "ENSMUSG00000050578": "Mmp13",
        },
        "match_type": "paralog_compensation",
    },
    {
        "name": "ALOX15 (lipoxygenase)",
        "human_gene": "ENSG00000161905",  # ALOX15
        "human_symbol": "ALOX15",
        "expected_mouse": {
            "strict": ["ENSMUSG00000018924"],  # Alox15
            "paralog_set": ["ENSMUSG00000018924", "ENSMUSG00000000320",
                           "ENSMUSG00000025701"],  # Alox15, Alox12, Alox5
        },
        "expected_symbols": {
            "ENSMUSG00000018924": "Alox15", "ENSMUSG00000000320": "Alox12",
            "ENSMUSG00000025701": "Alox5",
        },
        "match_type": "paralog_compensation",
    },
    {
        "name": "ALOX5 (lipoxygenase 5)",
        "human_gene": "ENSG00000012779",  # ALOX5
        "human_symbol": "ALOX5",
        "expected_mouse": {
            "strict": ["ENSMUSG00000025701"],  # Alox5
            "paralog_set": ["ENSMUSG00000025701", "ENSMUSG00000018924",
                           "ENSMUSG00000000320"],
        },
        "expected_symbols": {
            "ENSMUSG00000025701": "Alox5", "ENSMUSG00000018924": "Alox15",
            "ENSMUSG00000000320": "Alox12",
        },
        "match_type": "paralog_compensation",
    },
    {
        "name": "ALOX12 (lipoxygenase 12)",
        "human_gene": "ENSG00000108839",  # ALOX12
        "human_symbol": "ALOX12",
        "expected_mouse": {
            "strict": ["ENSMUSG00000000320"],  # Alox12
            "paralog_set": ["ENSMUSG00000000320", "ENSMUSG00000018924",
                           "ENSMUSG00000025701"],
        },
        "expected_symbols": {
            "ENSMUSG00000000320": "Alox12", "ENSMUSG00000018924": "Alox15",
            "ENSMUSG00000025701": "Alox5",
        },
        "match_type": "paralog_compensation",
    },
]


def load_checkpoint_model(cfg, ckpt_path, node_counts, device="cpu"):
    """Load model from checkpoint, handling per-modality buffers."""
    model = create_model(cfg, node_counts)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = ckpt["model_state_dict"]

    # Register per-modality buffers from checkpoint
    for key in ["Z_h_BP", "Z_m_BP", "Z_h_MF", "Z_m_MF", "Z_h_Disease", "Z_m_Disease"]:
        full_key = f"hgt.{key}"
        if full_key in state and not hasattr(model.hgt, key):
            model.hgt.register_buffer(key, state[full_key])

    # Handle old checkpoints with skip_gate (convert to residual_scales)
    if "hgt.skip_gate" in state and "hgt.residual_scales.Human_Gene" not in state:
        old_alpha = torch.sigmoid(state.pop("hgt.skip_gate"))
        # old: alpha*SVD + (1-alpha)*HGT = SVD + (1-alpha)*(HGT-SVD)
        # new: SVD + sigmoid(scale)*(HGT-SVD), so sigmoid(scale) = 1-alpha
        import math
        new_scale = math.log((1 - old_alpha.item()) / max(old_alpha.item(), 1e-8))
        for nt in model.hgt.node_types:
            state[f"hgt.residual_scales.{nt}"] = torch.tensor(new_scale)
        logger.info(f"  Converted skip_gate(alpha={old_alpha.item():.3f}) -> residual_scale={new_scale:.3f}")

    model.load_state_dict(state, strict=False)
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", float("nan"))
    return model, epoch, loss


def compute_embeddings(model, graph):
    """Compute fused gene embeddings on CPU."""
    model = model.to("cpu")
    model.eval()
    edge_index_dict = {}
    edge_attr_dict = {}
    for et in graph.edge_types:
        ed = graph[et]
        edge_index_dict[et] = ed.edge_index
        if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
            edge_attr_dict[et] = ed.edge_attr
    with torch.no_grad():
        E_h, E_m, entropy = model.get_gene_embeddings(edge_index_dict, edge_attr_dict)
    return E_h, E_m


def run_single_inference(model, E_h, E_m, cost_computer, fgw_solver,
                          src_idx, direction, device,
                          Z_h_svd=None, Z_m_svd=None, sparse_matrices=None):
    """Run FGW inference for a single source gene.

    CRITICAL FIX: Pass SVD features to candidate selector, matching training pipeline.
    Without SVD features, candidate selection falls back to pure HGT co-attention
    which produces poor candidates (root cause of all-zero validation results).
    """
    if direction == "human_to_mouse":
        E_src, E_tgt = E_h, E_m
        Z_src, Z_tgt = Z_h_svd, Z_m_svd
    else:
        E_src, E_tgt = E_m, E_h
        Z_src, Z_tgt = Z_m_svd, Z_h_svd

    E_src_dev = E_src.to(device)
    E_tgt_dev = E_tgt.to(device)
    model.candidate_selector = model.candidate_selector.to(device)

    I = torch.tensor([src_idx], dtype=torch.long, device=device)

    # Prepare SVD features for candidate retrieval (matching training pipeline)
    Z_src_dev = torch.tensor(Z_src, dtype=torch.float32, device=device) if Z_src is not None else None
    Z_tgt_dev = torch.tensor(Z_tgt, dtype=torch.float32, device=device) if Z_tgt is not None else None

    # Per-modality SVD features from HGT buffers
    Z_src_mods = None
    Z_tgt_mods = None
    if direction == "human_to_mouse":
        if hasattr(model.hgt, 'Z_h_BP') and model.hgt.Z_h_BP is not None:
            Z_src_mods = {
                'bp': model.hgt.Z_h_BP.to(device),
                'mf': model.hgt.Z_h_MF.to(device),
                'disease': model.hgt.Z_h_Disease.to(device),
            }
            Z_tgt_mods = {
                'bp': model.hgt.Z_m_BP.to(device),
                'mf': model.hgt.Z_m_MF.to(device),
                'disease': model.hgt.Z_m_Disease.to(device),
            }
    else:
        if hasattr(model.hgt, 'Z_m_BP') and model.hgt.Z_m_BP is not None:
            Z_src_mods = {
                'bp': model.hgt.Z_m_BP.to(device),
                'mf': model.hgt.Z_m_MF.to(device),
                'disease': model.hgt.Z_m_Disease.to(device),
            }
            Z_tgt_mods = {
                'bp': model.hgt.Z_h_BP.to(device),
                'mf': model.hgt.Z_h_MF.to(device),
                'disease': model.hgt.Z_h_Disease.to(device),
            }

    with torch.no_grad():
        J_cand, soft_mask, Pi0, row_mask = model.candidate_selector(
            E_source=E_src_dev, E_target=E_tgt_dev, source_indices=I,
            Z_source=Z_src_dev, Z_target=Z_tgt_dev,
            Z_source_mods=Z_src_mods, Z_target_mods=Z_tgt_mods)

        if len(J_cand) == 0:
            return []

        costs = cost_computer.compute(
            E_source=E_src_dev[src_idx:src_idx+1], E_target=E_tgt_dev[J_cand],
            source_indices=I, target_indices=J_cand, direction=direction)

        M_masked = costs["M"] * soft_mask
        T_local, _ = fgw_solver.solve(
            C_s=costs["C_s"], C_t=costs["C_t"],
            M=M_masked, Pi0=Pi0, mask=row_mask)

        t_row = T_local[0]
        topk = min(100, len(t_row))  # Get top-100 for thorough analysis
        topk_scores, topk_pos = torch.topk(t_row, topk)

        cand_indices = J_cand[topk_pos].cpu().numpy()
        scores = topk_scores.cpu().numpy()

    return [(int(idx), float(sc)) for idx, sc in zip(cand_indices, scores)]


def evaluate_case(predictions, case, indexers):
    """Evaluate a single validation case at three levels."""
    pred_ids = [p[0] for p in predictions]  # These are integer indices
    mouse_genes = indexers["mouse_genes"]  # ensembl_id -> int_index
    mouse_inv = indexers["mouse_genes_inv"]
    gene_symbols = indexers.get("gene_symbols", {})

    # Convert expected Ensembl IDs to integer indices for comparison
    def eid_to_idx(eid):
        return mouse_genes.get(eid)

    result = {
        "case": case["name"],
        "match_type": case["match_type"],
        "n_predictions": len(predictions),
    }

    # Strict hit: expected gene in top-k
    for k in [5, 10, 20, 50, 100]:
        top_k_ids = set(pred_ids[:k])
        strict_hits = [eid for eid in case["expected_mouse"]["strict"]
                       if eid_to_idx(eid) in top_k_ids]
        result[f"strict_hit@{k}"] = len(strict_hits) > 0

    # Paralog-set hit: any expected gene in top-k
    for k in [5, 10, 20, 50, 100]:
        top_k_ids = set(pred_ids[:k])
        para_hits = [eid for eid in case["expected_mouse"]["paralog_set"]
                     if eid_to_idx(eid) in top_k_ids]
        result[f"paralog_hit@{k}"] = len(para_hits) > 0
        result[f"paralog_count@{k}"] = len(para_hits)

    # Find ranks of expected genes
    for eid, sym in case["expected_symbols"].items():
        target_idx = eid_to_idx(eid)
        if target_idx is not None and target_idx in pred_ids:
            rank = pred_ids.index(target_idx) + 1
            score = predictions[rank - 1][1]
            result[f"rank_{sym}"] = rank
            result[f"score_{sym}"] = score
        else:
            result[f"rank_{sym}"] = ">100"
            result[f"score_{sym}"] = 0.0

    # Top-10 predictions with symbols
    top10 = []
    for idx, sc in predictions[:10]:
        eid = mouse_inv.get(idx, f"IDX_{idx}")
        sym = gene_symbols.get(eid, eid)
        top10.append(f"#{len(top10)+1} {sym}({sc:.4f})")
    result["top10"] = " | ".join(top10)

    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    cfg = Config()
    proc = Path(cfg.paths.proc_dir)

    # Load data (once)
    logger.info("Loading preprocessed data...")
    with open(proc / "indexers.pkl", "rb") as f:
        indexers = pickle.load(f)

    Z_h, Z_m, _, _ = load_svd_artifacts(str(proc))
    sparse_matrices = load_sparse_matrices(str(proc))
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }

    # All checkpoints to evaluate
    checkpoints = [
        ("retrain_v6", proc / "runs" / "retrain_v6" / "best_model.pt"),
        ("retrain_v5", proc / "runs" / "retrain_v5" / "best_model.pt"),
        ("exp_b_best", proc / "runs" / "exp_b" / "best_model.pt"),
        ("exp_d_best", proc / "runs" / "exp_d" / "best_model.pt"),
        ("exp_a_best", proc / "runs" / "exp_a" / "best_model.pt"),
        ("exp_c_best", proc / "runs" / "exp_c" / "best_model.pt"),
        ("default_best", proc / "runs" / "best_model.pt"),
        ("fusionfix_run3", proc / "runs" / "best_model_fusionfix_run3.pt"),
    ]

    # Filter to existing checkpoints
    checkpoints = [(name, path) for name, path in checkpoints if path.exists()]
    logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")

    all_results = {}

    for ckpt_name, ckpt_path in checkpoints:
        logger.info(f"\n{'='*70}")
        logger.info(f"CHECKPOINT: {ckpt_name} ({ckpt_path.name})")
        logger.info(f"{'='*70}")

        model, epoch, loss = load_checkpoint_model(cfg, ckpt_path, node_counts)
        logger.info(f"  Loaded epoch={epoch}, loss={loss:.4f}")

        E_h, E_m = compute_embeddings(model, graph)
        logger.info(f"  Embeddings: E_h={E_h.shape}, E_m={E_m.shape}")

        cost_computer = CostComputer(Z_h=Z_h, Z_m=Z_m, device=device)
        fgw_solver = FGWSolver(
            alpha=cfg.fgw.alpha, epsilon=cfg.fgw.epsilon,
            lambda_prior=0, lambda_memory=0,
            use_unbalanced=cfg.fgw.use_unbalanced,
            tau_row=cfg.fgw.tau_row, tau_col=cfg.fgw.tau_col,
            max_iter=getattr(cfg.infer, 'fgw_refine_iters', 20),
        )

        E_h_dev = E_h.to(device)
        E_m_dev = E_m.to(device)

        all_cases = TIER_A_CASES + TIER_B_CASES
        ckpt_results = []
        for case in all_cases:
            human_eid = case["human_gene"]
            src_idx = indexers["human_genes"].get(human_eid)
            if src_idx is None:
                logger.warning(f"  Gene {case['human_symbol']} not found!")
                continue

            logger.info(f"\n  Case: {case['name']} ({case['human_symbol']})")
            predictions = run_single_inference(
                model, E_h_dev, E_m_dev, cost_computer, fgw_solver,
                src_idx, "human_to_mouse", device,
                Z_h_svd=Z_h, Z_m_svd=Z_m, sparse_matrices=sparse_matrices)

            result = evaluate_case(predictions, case, indexers)
            ckpt_results.append(result)

            # Print key results
            logger.info(f"    Strict@20: {result['strict_hit@20']}, Paralog@20: {result['paralog_hit@20']} ({result['paralog_count@20']} hits)")
            for eid, sym in case["expected_symbols"].items():
                logger.info(f"    {sym}: rank={result[f'rank_{sym}']}, score={result.get(f'score_{sym}', 0):.4f}")
            logger.info(f"    Top10: {result['top10']}")

        all_results[ckpt_name] = ckpt_results

    # ── Summary Table ──
    all_cases = TIER_A_CASES + TIER_B_CASES
    logger.info(f"\n\n{'='*100}")
    logger.info("COMPREHENSIVE VALIDATION SUMMARY (TIER A + TIER B)")
    logger.info(f"{'='*100}")

    for case_idx, case in enumerate(all_cases):
        tier = "A" if case_idx < len(TIER_A_CASES) else "B"
        logger.info(f"\n--- [{tier}] {case['name']} ({case['match_type']}) ---")
        header = f"{'Checkpoint':<20s}"
        for sym in case["expected_symbols"].values():
            header += f"  {'rank_'+sym:>14s}"
        header += f"  {'strict@20':>10s}  {'para@20':>10s}  {'para_cnt':>8s}"
        logger.info(header)

        for ckpt_name in all_results:
            if case_idx < len(all_results[ckpt_name]):
                r = all_results[ckpt_name][case_idx]
                row = f"{ckpt_name:<20s}"
                for sym in case["expected_symbols"].values():
                    rank = r.get(f"rank_{sym}", "?")
                    row += f"  {str(rank):>14s}"
                row += f"  {str(r['strict_hit@20']):>10s}  {str(r['paralog_hit@20']):>10s}  {str(r['paralog_count@20']):>8s}"
                logger.info(row)

    # Save results
    output_path = Path("validation_results_all_tiers.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
