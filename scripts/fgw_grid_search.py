"""FGW sensitivity analysis: 5x5 grid search over alpha x epsilon.

Evaluates FGW-only retrieval (strategy S11) on Bench5 pairs
with different alpha/epsilon values, keeping the trained model fixed.
Outputs a 5x5 Recall@5 heatmap for ED Figure 2.
"""
import json, logging, os, sys, pickle, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import scipy.sparse as sp
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

# Grid search parameters
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
EPSILONS = [0.01, 0.03, 0.05, 0.10, 0.30]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    cfg = Config()

    # ── Load data ──
    logger.info("Loading data...")
    sparse_data = load_sparse_matrices(cfg.paths.proc_dir)
    svd_data = load_svd_artifacts(cfg.paths.proc_dir)
    graph_data = load_unified_graph(cfg.paths.proc_dir)

    with open(os.path.join(cfg.paths.proc_dir, "indexers.json")) as f:
        indexers = json.load(f)
    idx_h = indexers["human_gene_to_idx"]
    idx_m = indexers["mouse_gene_to_idx"]
    sym_h = {v: k for k, v in indexers.get("human_symbol_to_ensembl", {}).items()}
    sym_m = {v: k for k, v in indexers.get("mouse_symbol_to_ensembl", {}).items()}
    N_m = len(idx_m)

    # ── Load best v2 model (used for FGW) ──
    logger.info("Loading model...")
    model = create_model(cfg, graph_data)
    ckpt_path = BASE / "data" / "processed" / "runs" / "retrain_v6" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ── Get HGT embeddings ──
    logger.info("Computing HGT embeddings...")
    with torch.no_grad():
        x_dict = {}
        for ntype in graph_data.node_types:
            x_dict[ntype] = graph_data[ntype].x
        edge_index_dict = {etype: graph_data[etype].edge_index for etype in graph_data.edge_types}
        hgt_out = model.hgt(x_dict, edge_index_dict)

    E_h = hgt_out["Human_Gene"].to(device)
    E_m = hgt_out["Mouse_Gene"].to(device)

    # Per-modality gene features for cost computation
    pm_feats = compute_per_modality_gene_features(
        sparse_data, svd_data, list(idx_h.keys()), list(idx_m.keys()),
        indexers, device="cpu"
    )

    cost_computer = CostComputer(cfg, pm_feats, device=device)

    # ── Load Bench5 pairs ──
    bench5_path = BASE / "benchmarks" / "results" / "bench5_M4.json"
    with open(bench5_path) as f:
        bench5 = json.load(f)
    pairs = bench5["pairs"]
    logger.info(f"Bench5 pairs: {len(pairs)}")

    # Map symbols to indices
    # First build symbol->ensemble lookup from indexers
    h_sym2idx = {}
    for ens, idx in idx_h.items():
        for sym, e in indexers.get("human_symbol_to_ensembl", {}).items():
            if e == ens:
                h_sym2idx[sym] = idx
    m_sym2idx = {}
    for ens, idx in idx_m.items():
        for sym, e in indexers.get("mouse_symbol_to_ensembl", {}).items():
            if e == ens:
                m_sym2idx[sym] = idx

    # ── Grid search ──
    results = {}
    ref_ranks = None  # For Kendall tau vs chosen config

    for alpha in ALPHAS:
        for eps in EPSILONS:
            logger.info(f"  Testing alpha={alpha}, epsilon={eps}")

            fgw_solver = FGWSolver(
                alpha=alpha, epsilon=eps,
                use_unbalanced=cfg.fgw.use_unbalanced,
                tau_row=cfg.fgw.tau_row,
                tau_col=0.15,  # inference tau_col
                max_iter=20,
            )

            hits = 0
            total = 0
            pair_ranks = {}

            for p in pairs:
                h_sym, m_sym = p["h_sym"], p["m_sym"]
                h_idx = h_sym2idx.get(h_sym)
                m_idx = m_sym2idx.get(m_sym)
                if h_idx is None or m_idx is None:
                    continue

                total += 1

                # Compute FGW for this query -> all mouse genes
                with torch.no_grad():
                    q_emb = E_h[h_idx:h_idx+1]  # [1, d]

                    # Cosine similarity for candidate selection
                    cos_sim = F.cosine_similarity(q_emb, E_m, dim=1)  # [N_m]
                    _, top_idx = cos_sim.topk(min(256, N_m))
                    J = top_idx

                    # Cost computation
                    try:
                        costs = cost_computer.compute(
                            E_source=q_emb,
                            E_target=E_m[J],
                            source_indices=torch.tensor([h_idx], device=device),
                            target_indices=J,
                        )

                        sm = model.fusion.get_softmax_weights() if hasattr(model, 'fusion') else None
                        if sm is None:
                            sm = torch.ones(1, device=device)

                        M = costs["M"] * sm if "M" in costs else costs.get("C_t", torch.zeros(1))
                        C_s = costs.get("C_s", torch.zeros(1, 1, device=device))
                        C_t = costs.get("C_t", M)

                        P0 = torch.ones(1, len(J), device=device) / len(J)
                        mask = torch.ones(1, len(J), device=device, dtype=torch.bool)

                        T, _ = fgw_solver.solve(C_s=C_s, C_t=C_t, M=M, Pi0=P0, mask=mask)

                        # Rank by transport mass
                        mass = T.sum(dim=0)  # [|J|]
                        sorted_idx = mass.argsort(descending=True)

                        rank = N_m  # default if not found
                        for r, si in enumerate(sorted_idx):
                            if J[si].item() == m_idx:
                                rank = r + 1
                                break

                        pair_ranks[f"{h_sym}->{m_sym}"] = rank
                        if rank <= 5:
                            hits += 1
                    except Exception as e:
                        pair_ranks[f"{h_sym}->{m_sym}"] = N_m

            r5 = hits / total if total > 0 else 0
            key = f"a{alpha}_e{eps}"
            results[key] = {
                "alpha": alpha,
                "epsilon": eps,
                "recall_at_5": round(r5, 4),
                "hits": hits,
                "total": total,
            }

            if alpha == 0.5 and eps == 0.05:
                ref_ranks = pair_ranks.copy()

            logger.info(f"    R@5 = {r5:.4f} ({hits}/{total})")

    # ── Kendall tau vs chosen config ──
    if ref_ranks:
        logger.info("Computing Kendall tau vs chosen config...")
        tau_results = {}
        for key, meta in results.items():
            alpha, eps = meta["alpha"], meta["epsilon"]
            if alpha == 0.5 and eps == 0.05:
                tau_results[key] = 1.0
                continue
            # Recompute this config's ranks (already stored if we saved them)
            # For now, compute tau from pair_ranks if available
            tau_results[key] = None  # placeholder
        results["kendall_tau_vs_chosen"] = tau_results

    # ── Save ──
    outpath = BASE / "manuscript" / "figures_precompute" / "out" / "fgw_grid_search.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    # Print 5x5 grid
    print("\n=== FGW Grid Search: Bench5 Recall@5 ===")
    header = "eps\\alpha  " + "  ".join(f"  {a:.1f}" for a in ALPHAS)
    print(header)
    for eps in EPSILONS:
        row = f"  {eps:.2f}   "
        for alpha in ALPHAS:
            key = f"a{alpha}_e{eps}"
            r5 = results[key]["recall_at_5"]
            row += f" {r5:.3f}"
        print(row)

    logger.info(f"Saved to {outpath}")

if __name__ == "__main__":
    main()
