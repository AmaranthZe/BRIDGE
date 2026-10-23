"""FGW grid search: 5x5 alpha x epsilon on Bench5.

Mirrors the FGW (S11) evaluation approach from run_validation_final_rrf.py
but loops over alpha/epsilon grid.
"""
import json, logging, os, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.data.graph import load_unified_graph
from v2_redesign.model.model import create_model
from v2_redesign.transport.costs import CostComputer
from v2_redesign.transport.fgw import FGWSolver

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
EPSILONS = [0.01, 0.03, 0.05, 0.10, 0.30]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    cfg = Config()
    proc = Path(cfg.paths.proc_dir)

    # ── Load data (same as run_validation_final_rrf.py) ──
    sparse_data = load_sparse_matrices(str(proc))
    Z_h_np, Z_m_np, _, _ = load_svd_artifacts(str(proc))
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    with open(proc / "indexers.json") as f:
        indexers = json.load(f)

    h_genes = indexers["human_genes"]
    m_genes = indexers["mouse_genes"]
    N_h, N_m = len(h_genes), len(m_genes)

    # SVD features
    Z_h = torch.from_numpy(Z_h_np).float()
    Z_m = torch.from_numpy(Z_m_np).float()

    # Symbol lookups
    h_name2ens = {v: k for k, v in indexers.get("human_gene_names", {}).items()}
    m_name2ens = {v: k for k, v in indexers.get("mouse_gene_names", {}).items()}
    h_sym2idx = {sym: h_genes[ens] for sym, ens in h_name2ens.items() if ens in h_genes}
    m_sym2idx = {sym: m_genes[ens] for sym, ens in m_name2ens.items() if ens in m_genes}

    # ── Load model (same pattern as run_validation_final_rrf.py) ──
    logger.info("Loading v6 model...")
    ckpt_path = proc / "runs" / "retrain_v6" / "best_model.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # Derive node counts from checkpoint embedding shapes to avoid size mismatch
    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }
    for nt in ["Human_Gene", "Mouse_Gene", "BP", "MF", "Disease"]:
        emb_key = f"hgt.embeddings.{nt}.weight"
        if emb_key in state:
            ckpt_size = state[emb_key].shape[0]
            if ckpt_size != node_counts[nt]:
                logger.warning(f"  {nt}: indexer={node_counts[nt]}, ckpt={ckpt_size} → using ckpt size")
                node_counts[nt] = ckpt_size

    model = create_model(cfg, node_counts)
    for key in ["Z_h_BP","Z_m_BP","Z_h_MF","Z_m_MF","Z_h_Disease","Z_m_Disease"]:
        fk = f"hgt.{key}"
        if fk in state and not hasattr(model.hgt, key):
            model.hgt.register_buffer(key, state[fk])
    model.load_state_dict(state, strict=False)
    model.eval()

    # Get HGT embeddings
    ei, ea = {}, {}
    for et in graph.edge_types:
        ed = graph[et]
        ei[et] = ed.edge_index
        if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
            ea[et] = ed.edge_attr

    with torch.no_grad():
        E_h, E_m, _ = model.get_gene_embeddings(ei, ea)
    E_h_dev, E_m_dev = E_h.to(device), E_m.to(device)
    logger.info(f"Embeddings: H={E_h.shape}, M={E_m.shape}")

    cost_computer = CostComputer(Z_h=Z_h, Z_m=Z_m, device=device)

    # ── Load Bench5 pairs ──
    bench5_path = BASE / "benchmarks" / "results" / "bench5_M4.json"
    with open(bench5_path) as f:
        pairs = json.load(f)["pairs"]
    logger.info(f"Bench5: {len(pairs)} pairs")

    # Pre-compute candidate sets per query (top-256 by HGT cosine)
    unique_h = {}
    for p in pairs:
        h = p["h_sym"]
        if h not in unique_h and h in h_sym2idx:
            unique_h[h] = h_sym2idx[h]

    cand_cache = {}
    with torch.no_grad():
        for h_sym, h_idx in unique_h.items():
            q = E_h_dev[h_idx:h_idx+1]
            cos = F.cosine_similarity(q, E_m_dev, dim=1)
            _, J = cos.topk(min(256, N_m))
            cand_cache[h_sym] = J
    logger.info(f"Cached candidates for {len(cand_cache)} queries")

    # ── Grid search ──
    all_results = {}
    ref_pair_ranks = None

    for alpha in ALPHAS:
        for eps in EPSILONS:
            t0 = time.time()
            key = f"a{alpha}_e{eps}"
            logger.info(f"--- {key} ---")

            fgw = FGWSolver(
                alpha=alpha, epsilon=eps,
                lambda_prior=0, lambda_memory=0,
                use_unbalanced=cfg.fgw.use_unbalanced,
                tau_row=cfg.fgw.tau_row, tau_col=cfg.fgw.tau_col,
                max_iter=20,
            )

            hits5 = 0
            total = 0
            pair_ranks = {}

            with torch.no_grad():
                for p in pairs:
                    h_sym, m_sym = p["h_sym"], p["m_sym"]
                    h_idx = h_sym2idx.get(h_sym)
                    m_idx = m_sym2idx.get(m_sym)
                    J = cand_cache.get(h_sym)
                    if h_idx is None or m_idx is None or J is None:
                        continue

                    total += 1
                    q_emb = E_h_dev[h_idx:h_idx+1]

                    try:
                        costs = cost_computer.compute(
                            E_source=q_emb,
                            E_target=E_m_dev[J],
                            source_indices=torch.tensor([h_idx], device=device),
                            target_indices=J,
                            direction="human_to_mouse",
                        )

                        C_s = costs["C_s"]
                        C_t = costs["C_t"]
                        M = costs["M"]
                        P0 = torch.ones(1, len(J), device=device) / len(J)
                        mask = torch.ones(1, len(J), device=device, dtype=torch.bool)

                        T, _ = fgw.solve(C_s=C_s, C_t=C_t, M=M, Pi0=P0, mask=mask)
                        mass = T.sum(dim=0)
                        sorted_idx = mass.argsort(descending=True)

                        rank = N_m
                        for r, si in enumerate(sorted_idx):
                            if J[si].item() == m_idx:
                                rank = r + 1
                                break

                        pair_ranks[f"{h_sym}->{m_sym}"] = rank
                        if rank <= 5:
                            hits5 += 1
                    except Exception as e:
                        pair_ranks[f"{h_sym}->{m_sym}"] = N_m

            r5 = hits5 / total if total > 0 else 0
            dt = time.time() - t0

            all_results[key] = {
                "alpha": alpha, "epsilon": eps,
                "recall_at_5": round(r5, 4),
                "hits5": hits5, "total": total,
                "time_sec": round(dt, 1),
            }
            logger.info(f"  R@5={r5:.4f} ({hits5}/{total}) [{dt:.1f}s]")

            if alpha == 0.5 and eps == 0.05:
                ref_pair_ranks = pair_ranks.copy()

    # ── Kendall tau vs chosen (alpha=0.5, eps=0.05) ──
    # Would need to store per-pair ranks for each config.
    # For now, just report the R@5 grid.

    # ── Print 5x5 grid ──
    print("\n=== FGW Grid: Bench5 Recall@5 ===")
    header = "eps\\alpha " + "  ".join(f"  {a}" for a in ALPHAS)
    print(header)
    for eps in EPSILONS:
        row = f"  {eps:.2f}  "
        for alpha in ALPHAS:
            r5 = all_results[f"a{alpha}_e{eps}"]["recall_at_5"]
            row += f" {r5:.3f}"
        print(row)

    # ── Save ──
    outpath = BASE / "manuscript" / "figures_precompute" / "out" / "fgw_grid_search.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {outpath}")

if __name__ == "__main__":
    main()
