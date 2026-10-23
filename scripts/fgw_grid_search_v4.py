"""FGW grid search v4: Full S11 pipeline (with candidate_selector) over alpha x epsilon.

Uses the exact same pipeline as S11 in run_validation_final_rrf.py:
- model.candidate_selector() for candidates, soft_mask, prior, mask
- CostComputer for structural + feature costs
- costs["M"] * sm (soft mask modulation)
- FGWSolver with varying alpha/epsilon

Reports both R@5 and MRR for the Bench5 benchmark.
"""
import json, logging, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import torch
import torch.nn.functional as F

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
    proc = Path(cfg.paths.proc_dir)

    # ── Load data ──
    sparse = load_sparse_matrices(str(proc))
    Z_h_np, Z_m_np, _, _ = load_svd_artifacts(str(proc))
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    with open(proc / "indexers.json") as f:
        indexers = json.load(f)

    h_genes = indexers["human_genes"]
    m_genes = indexers["mouse_genes"]
    N_h, N_m = len(h_genes), len(m_genes)

    # SVD features (unified)
    Z_h = torch.from_numpy(Z_h_np).float()
    Z_m = torch.from_numpy(Z_m_np).float()

    # Per-modality SVD features (same as run_validation_final_rrf.py)
    logger.info("Computing per-modality SVD features...")
    mf = compute_per_modality_gene_features(sparse, 128)
    Z_h_BP = F.normalize(torch.tensor(mf['Z_h_BP']), dim=-1)
    Z_m_BP = F.normalize(torch.tensor(mf['Z_m_BP']), dim=-1)
    Z_h_MF = F.normalize(torch.tensor(mf['Z_h_MF']), dim=-1)
    Z_m_MF = F.normalize(torch.tensor(mf['Z_m_MF']), dim=-1)
    Z_h_D  = F.normalize(torch.tensor(mf['Z_h_Disease']), dim=-1)
    Z_m_D  = F.normalize(torch.tensor(mf['Z_m_Disease']), dim=-1)

    # Symbol lookups
    h_name2ens = {v: k for k, v in indexers.get("human_gene_names", {}).items()}
    m_name2ens = {v: k for k, v in indexers.get("mouse_gene_names", {}).items()}
    h_sym2idx = {sym: h_genes[ens] for sym, ens in h_name2ens.items() if ens in h_genes}
    m_sym2idx = {sym: m_genes[ens] for sym, ens in m_name2ens.items() if ens in m_genes}

    # ── Load model ──
    logger.info("Loading v6 model...")
    ckpt_path = proc / "runs" / "retrain_v6" / "best_model.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    node_counts = {
        "Human_Gene": len(h_genes),
        "Mouse_Gene": len(m_genes),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }
    for nt in node_counts:
        emb_key = f"hgt.embeddings.{nt}.weight"
        if emb_key in state:
            ckpt_size = state[emb_key].shape[0]
            if ckpt_size != node_counts[nt]:
                logger.warning(f"  {nt}: indexer={node_counts[nt]}, ckpt={ckpt_size} -> using ckpt")
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

    # Move candidate_selector to GPU (same as run_validation_final_rrf.py)
    model.candidate_selector = model.candidate_selector.to(device)

    cost_computer = CostComputer(Z_h=Z_h, Z_m=Z_m, device=device)
    logger.info(f"Embeddings: H={E_h.shape}, M={E_m.shape}")

    # Per-modality features on device
    Z_src_d = torch.tensor(Z_h_np, dtype=torch.float32, device=device)
    Z_tgt_d = torch.tensor(Z_m_np, dtype=torch.float32, device=device)
    Zsm = {k: v.to(device) for k, v in [('bp', Z_h_BP), ('mf', Z_h_MF), ('disease', Z_h_D)]}
    Ztm = {k: v.to(device) for k, v in [('bp', Z_m_BP), ('mf', Z_m_MF), ('disease', Z_m_D)]}

    # ── Load Bench5 pairs ──
    bench5_path = BASE / "benchmarks" / "results" / "bench5_M4.json"
    with open(bench5_path) as f:
        pairs = json.load(f)["pairs"]
    logger.info(f"Bench5: {len(pairs)} pairs")

    # ── Grid search ──
    all_results = {}

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
            rr_sum = 0.0  # for MRR

            with torch.no_grad():
                for p in pairs:
                    h_sym, m_sym = p["h_sym"], p["m_sym"]
                    h_idx = h_sym2idx.get(h_sym)
                    m_idx = m_sym2idx.get(m_sym)
                    if h_idx is None or m_idx is None:
                        continue

                    total += 1
                    I = torch.tensor([h_idx], dtype=torch.long, device=device)

                    try:
                        # Full candidate_selector (same as S11 in RRF validation)
                        J, sm, P0, rm = model.candidate_selector(
                            E_source=E_h_dev, E_target=E_m_dev, source_indices=I,
                            Z_source=Z_src_d, Z_target=Z_tgt_d,
                            Z_source_mods=Zsm, Z_target_mods=Ztm,
                        )

                        if len(J) == 0:
                            rr_sum += 0.0
                            continue

                        costs = cost_computer.compute(
                            E_source=E_h_dev[h_idx:h_idx+1],
                            E_target=E_m_dev[J],
                            source_indices=I,
                            target_indices=J,
                            direction="human_to_mouse",
                        )

                        T, _ = fgw.solve(
                            C_s=costs["C_s"], C_t=costs["C_t"],
                            M=costs["M"] * sm,  # soft mask modulation!
                            Pi0=P0, mask=rm,
                        )

                        topk = min(100, len(T[0]))
                        ts, tp = torch.topk(T[0], topk)

                        rank = N_m
                        for r, pidx in enumerate(tp):
                            if J[pidx].item() == m_idx:
                                rank = r + 1
                                break

                        if rank <= 5:
                            hits5 += 1
                        rr_sum += 1.0 / rank if rank <= N_m else 0.0

                    except Exception as e:
                        rr_sum += 0.0

            r5 = hits5 / total if total > 0 else 0
            mrr = rr_sum / total if total > 0 else 0
            dt = time.time() - t0

            all_results[key] = {
                "alpha": alpha, "epsilon": eps,
                "recall_at_5": round(r5, 4),
                "mrr": round(mrr, 4),
                "hits5": hits5, "total": total,
                "time_sec": round(dt, 1),
            }
            logger.info(f"  R@5={r5:.4f} MRR={mrr:.4f} ({hits5}/{total}) [{dt:.1f}s]")

    # ── Print 5x5 grids ──
    print("\n=== FGW Grid: Bench5 Recall@5 ===")
    header = "eps\\alpha " + "  ".join(f"  {a}" for a in ALPHAS)
    print(header)
    for eps in EPSILONS:
        row = f"  {eps:.2f}  "
        for alpha in ALPHAS:
            r5 = all_results[f"a{alpha}_e{eps}"]["recall_at_5"]
            row += f" {r5:.3f}"
        print(row)

    print("\n=== FGW Grid: Bench5 MRR ===")
    print(header)
    for eps in EPSILONS:
        row = f"  {eps:.2f}  "
        for alpha in ALPHAS:
            mrr = all_results[f"a{alpha}_e{eps}"]["mrr"]
            row += f" {mrr:.3f}"
        print(row)

    # ── Save ──
    outpath = BASE / "manuscript" / "figures_precompute" / "out" / "fgw_grid_search_v4.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
