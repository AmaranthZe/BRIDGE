"""FGW sensitivity analysis: 5x5 grid search over alpha x epsilon on Bench5.

Uses the same evaluation approach as run_validation_final_rrf.py (strategy S11),
but loops over alpha/epsilon grid. Only the FGW transport is re-run; embeddings
are computed once.
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
from v2_redesign.model.hgt import compute_per_modality_gene_features
from v2_redesign.transport.costs import CostComputer
from v2_redesign.transport.fgw import FGWSolver

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
EPSILONS = [0.01, 0.03, 0.05, 0.10, 0.30]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    cfg = Config()

    # ── Load all data ──
    logger.info("Loading data...")
    sparse_data = load_sparse_matrices(cfg.paths.proc_dir)
    svd_data = load_svd_artifacts(cfg.paths.proc_dir)
    graph_data = load_unified_graph(str(Path(cfg.paths.proc_dir) / "unified_graph.pt"))

    with open(os.path.join(cfg.paths.proc_dir, "indexers.json")) as f:
        indexers = json.load(f)
    idx_h = indexers["human_genes"]  # ensembl_id -> int index
    idx_m = indexers["mouse_genes"]
    N_m = len(idx_m)

    # Build symbol -> index lookup using gene_names
    h_name2ens = {v: k for k, v in indexers.get("human_gene_names", {}).items()}
    m_name2ens = {v: k for k, v in indexers.get("mouse_gene_names", {}).items()}
    h_sym2idx = {sym: idx_h[ens] for sym, ens in h_name2ens.items() if ens in idx_h}
    m_sym2idx = {sym: idx_m[ens] for sym, ens in m_name2ens.items() if ens in idx_m}

    # ── Load model and compute embeddings once ──
    logger.info("Loading model...")
    model = create_model(cfg, graph_data)
    ckpt_path = BASE / "data" / "processed" / "runs" / "retrain_v6" / "best_model.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    logger.info("Computing HGT embeddings (once)...")
    with torch.no_grad():
        x_dict = {nt: graph_data[nt].x for nt in graph_data.node_types}
        ei_dict = {et: graph_data[et].edge_index for et in graph_data.edge_types}
        hgt_out = model.hgt(x_dict, ei_dict)

    E_h = hgt_out["Human_Gene"].to(device)
    E_m = hgt_out["Mouse_Gene"].to(device)
    logger.info(f"Embeddings: H={E_h.shape}, M={E_m.shape}")

    # Per-modality features
    logger.info("Loading per-modality gene features...")
    pm_feats = compute_per_modality_gene_features(
        sparse_data, svd_data, list(idx_h.keys()), list(idx_m.keys()),
        indexers, device=device,
    )

    # ── Load Bench5 pairs ──
    bench5_path = BASE / "benchmarks" / "results" / "bench5_M4.json"
    with open(bench5_path) as f:
        bench5 = json.load(f)
    pairs = bench5["pairs"]
    logger.info(f"Bench5 pairs: {len(pairs)}")

    # Pre-compute candidate sets for each human query (top-256 by HGT cosine)
    logger.info("Pre-computing candidate sets...")
    unique_h = list(set(p["h_sym"] for p in pairs if p["h_sym"] in h_sym2idx))
    cand_cache = {}
    with torch.no_grad():
        for h_sym in unique_h:
            h_idx = h_sym2idx[h_sym]
            q = E_h[h_idx:h_idx+1]
            cos = F.cosine_similarity(q, E_m, dim=1)
            _, J = cos.topk(min(256, N_m))
            cand_cache[h_sym] = J

    # ── Grid search ──
    logger.info("Starting grid search...")
    results = {}
    ref_ranks = None

    for alpha in ALPHAS:
        for eps in EPSILONS:
            t0 = time.time()
            logger.info(f"alpha={alpha}, eps={eps}")

            fgw = FGWSolver(
                alpha=alpha, epsilon=eps,
                use_unbalanced=cfg.fgw.use_unbalanced,
                tau_row=cfg.fgw.tau_row,
                tau_col=0.15,
                max_iter=20,
            )

            # Get cost computer with current model fusion weights
            cost_computer = CostComputer(cfg, pm_feats, device=device)

            hits5 = 0
            total = 0
            pair_ranks = {}

            with torch.no_grad():
                for p in pairs:
                    h_sym, m_sym = p["h_sym"], p["m_sym"]
                    h_idx = h_sym2idx.get(h_sym)
                    m_idx = m_sym2idx.get(m_sym)
                    if h_idx is None or m_idx is None:
                        continue

                    J = cand_cache.get(h_sym)
                    if J is None:
                        continue

                    total += 1
                    q_emb = E_h[h_idx:h_idx+1]

                    try:
                        costs = cost_computer.compute(
                            E_source=q_emb,
                            E_target=E_m[J],
                            source_indices=torch.tensor([h_idx], device=device),
                            target_indices=J,
                        )

                        M = costs.get("M", costs.get("C_t", torch.zeros(1, len(J), device=device)))
                        C_s = costs.get("C_s", torch.zeros(1, 1, device=device))
                        C_t = costs.get("C_t", M)
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

            key = f"a{alpha}_e{eps}"
            results[key] = {
                "alpha": alpha, "epsilon": eps,
                "recall_at_5": round(r5, 4),
                "hits5": hits5, "total": total, "elapsed_sec": round(dt, 1),
            }
            logger.info(f"  R@5={r5:.4f} ({hits5}/{total}) in {dt:.1f}s")

            if alpha == 0.5 and eps == 0.05:
                ref_ranks = pair_ranks.copy()

    # ── Kendall tau vs chosen config ──
    if ref_ranks is not None:
        logger.info("Computing Kendall tau vs chosen (a=0.5, e=0.05)...")
        ref_vec = ref_ranks
        for key, meta in results.items():
            alpha, eps = meta["alpha"], meta["epsilon"]
            # We'd need to store per-pair ranks for each config to compute tau.
            # For now, mark as needing rerun with rank storage.
            pass

    # ── Print 5x5 grid ──
    print("\n=== FGW Grid: Bench5 Recall@5 ===")
    header = "eps\\alpha  " + "  ".join(f"  {a}" for a in ALPHAS)
    print(header)
    for eps in EPSILONS:
        row = f"  {eps:.2f}   "
        for alpha in ALPHAS:
            key = f"a{alpha}_e{eps}"
            r5 = results[key]["recall_at_5"]
            row += f" {r5:.3f}"
        print(row)

    # ── Save ──
    outpath = BASE / "manuscript" / "figures_precompute" / "out" / "fgw_grid_search.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
