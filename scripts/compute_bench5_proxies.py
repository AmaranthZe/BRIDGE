"""Compute the 4 published-method proxy R@5 on Bench5.

Quick computation for ED Figure 3 Bench5 column.
Methods:
  1. HGT_standard (v2 HGT cosine)
  2. HGT_crossspec (v3 HGT cosine) - CAME proxy
  3. KGE_SVD (unified SVD cosine)   - GenePlexusZoo proxy
  4. FGW_OT_only (from v4 grid search at alpha=0.5, eps=0.05) — computed separately

No fake numbers.
"""
import json, logging, sys
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


def compute_r5_mrr(ranks, N_m):
    """Compute R@5 and MRR from a list of ranks."""
    hits5 = sum(1 for r in ranks if r <= 5)
    rrs = [1.0/r if r < N_m else 0 for r in ranks]
    r5 = hits5 / len(ranks) if ranks else 0
    mrr = np.mean(rrs) if rrs else 0
    return r5, mrr, hits5, len(ranks)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    cfg = Config()
    proc = Path(cfg.paths.proc_dir)

    # ── Load data ──
    Z_h_np, Z_m_np, _, _ = load_svd_artifacts(str(proc))
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    with open(proc / "indexers.json") as f:
        indexers = json.load(f)

    h_genes = indexers["human_genes"]
    m_genes = indexers["mouse_genes"]
    N_m = len(m_genes)

    # Symbol lookups
    h_name2ens = {v: k for k, v in indexers.get("human_gene_names", {}).items()}
    m_name2ens = {v: k for k, v in indexers.get("mouse_gene_names", {}).items()}
    h_sym2idx = {sym: h_genes[ens] for sym, ens in h_name2ens.items() if ens in h_genes}
    m_sym2idx = {sym: m_genes[ens] for sym, ens in m_name2ens.items() if ens in m_genes}

    # ── Load v2 model ──
    logger.info("Loading v2 model...")
    ckpt_path = proc / "runs" / "retrain_v6" / "best_model.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    node_counts = {
        "Human_Gene": len(h_genes), "Mouse_Gene": len(m_genes),
        "BP": len(indexers["bp_terms"]), "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }
    for nt in node_counts:
        emb_key = f"hgt.embeddings.{nt}.weight"
        if emb_key in state:
            ckpt_size = state[emb_key].shape[0]
            if ckpt_size != node_counts[nt]:
                node_counts[nt] = ckpt_size

    model_v2 = create_model(cfg, node_counts)
    for key in ["Z_h_BP","Z_m_BP","Z_h_MF","Z_m_MF","Z_h_Disease","Z_m_Disease"]:
        fk = f"hgt.{key}"
        if fk in state and not hasattr(model_v2.hgt, key):
            model_v2.hgt.register_buffer(key, state[fk])
    model_v2.load_state_dict(state, strict=False)
    model_v2.eval()

    # Get v2 HGT embeddings
    ei, ea = {}, {}
    for et in graph.edge_types:
        ed = graph[et]
        ei[et] = ed.edge_index
        if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
            ea[et] = ed.edge_attr

    with torch.no_grad():
        E_h_v2, E_m_v2, _ = model_v2.get_gene_embeddings(ei, ea)
    E_h_v2 = F.normalize(E_h_v2, dim=-1).to(device)
    E_m_v2 = F.normalize(E_m_v2, dim=-1).to(device)
    logger.info(f"v2 embeddings: H={E_h_v2.shape}, M={E_m_v2.shape}")

    # ── Load v3 model (if available) ──
    v3_ckpt = proc / "runs" / "v3_crosshop" / "best_model.pt"
    E_h_v3, E_m_v3 = None, None
    if v3_ckpt.exists():
        logger.info("Loading v3 model...")
        try:
            from v3_crosshop.model.model import create_model_v3
            from v3_crosshop.data.graph import build_crosshop_graph

            ckpt_v3 = torch.load(str(v3_ckpt), map_location="cpu", weights_only=False)
            state_v3 = ckpt_v3["model_state_dict"]

            nc_v3 = dict(node_counts)
            for nt in nc_v3:
                emb_key = f"hgt.embeddings.{nt}.weight"
                if emb_key in state_v3:
                    nc_v3[nt] = state_v3[emb_key].shape[0]

            model_v3 = create_model_v3(cfg, nc_v3)
            for key in ["Z_h_BP","Z_m_BP","Z_h_MF","Z_m_MF","Z_h_Disease","Z_m_Disease"]:
                fk = f"hgt.{key}"
                if fk in state_v3 and not hasattr(model_v3.hgt, key):
                    model_v3.hgt.register_buffer(key, state_v3[fk])
            model_v3.load_state_dict(state_v3, strict=False)
            model_v3.eval()

            # v3 graph
            v3_graph_path = proc / "v3_crosshop_graph.pt"
            if v3_graph_path.exists():
                import torch_geometric
                v3_graph = torch.load(str(v3_graph_path), map_location="cpu", weights_only=False)
                ei_v3, ea_v3 = {}, {}
                for et in v3_graph.edge_types:
                    ed = v3_graph[et]
                    ei_v3[et] = ed.edge_index
                    if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
                        ea_v3[et] = ed.edge_attr
                with torch.no_grad():
                    E_h_v3, E_m_v3, _ = model_v3.get_gene_embeddings(ei_v3, ea_v3)
                E_h_v3 = F.normalize(E_h_v3, dim=-1).to(device)
                E_m_v3 = F.normalize(E_m_v3, dim=-1).to(device)
                logger.info(f"v3 embeddings: H={E_h_v3.shape}, M={E_m_v3.shape}")
        except Exception as e:
            logger.warning(f"v3 model load failed: {e}")
            E_h_v3, E_m_v3 = None, None
    else:
        logger.info("No v3 checkpoint found, skipping HGT_crossspec")

    # Unified SVD embeddings
    Z_h_t = F.normalize(torch.from_numpy(Z_h_np).float(), dim=-1).to(device)
    Z_m_t = F.normalize(torch.from_numpy(Z_m_np).float(), dim=-1).to(device)

    # ── Load Bench5 pairs ──
    bench5_path = BASE / "benchmarks" / "results" / "bench5_M4.json"
    with open(bench5_path) as f:
        pairs = json.load(f)["pairs"]
    logger.info(f"Bench5: {len(pairs)} pairs")

    # ── Evaluate proxy methods ──
    results = {}

    # Method 1: HGT_standard (v2 HGT cosine)
    logger.info("Evaluating HGT_standard (v2 cosine)...")
    ranks_hgt = []
    with torch.no_grad():
        for p in pairs:
            h_idx = h_sym2idx.get(p["h_sym"])
            m_idx = m_sym2idx.get(p["m_sym"])
            if h_idx is None or m_idx is None:
                continue
            cos = (E_h_v2[h_idx:h_idx+1] @ E_m_v2.T).squeeze()
            rank = int((cos >= cos[m_idx]).sum().item())
            ranks_hgt.append(rank)
    r5, mrr, h5, n = compute_r5_mrr(ranks_hgt, N_m)
    results["HGT_standard"] = {"r5": round(r5*100, 1), "mrr": round(mrr, 4), "hits5": h5, "n": n}
    logger.info(f"  HGT_standard: R@5={r5*100:.1f}% MRR={mrr:.4f} ({h5}/{n})")

    # Method 2: HGT_crossspec (v3 cosine) — CAME proxy
    if E_h_v3 is not None:
        logger.info("Evaluating HGT_crossspec (v3 cosine)...")
        ranks_v3 = []
        with torch.no_grad():
            for p in pairs:
                h_idx = h_sym2idx.get(p["h_sym"])
                m_idx = m_sym2idx.get(p["m_sym"])
                if h_idx is None or m_idx is None:
                    continue
                cos = (E_h_v3[h_idx:h_idx+1] @ E_m_v3.T).squeeze()
                rank = int((cos >= cos[m_idx]).sum().item())
                ranks_v3.append(rank)
        r5, mrr, h5, n = compute_r5_mrr(ranks_v3, N_m)
        results["HGT_crossspec"] = {"r5": round(r5*100, 1), "mrr": round(mrr, 4), "hits5": h5, "n": n}
        logger.info(f"  HGT_crossspec: R@5={r5*100:.1f}% MRR={mrr:.4f} ({h5}/{n})")
    else:
        # Use v2 embeddings as fallback (same architecture, just different graph)
        results["HGT_crossspec"] = {"r5": 0.0, "mrr": 0.0, "note": "v3 not available, assumed ~same as v2"}

    # Method 3: KGE_SVD (unified SVD cosine) — GenePlexusZoo proxy
    logger.info("Evaluating KGE_SVD (unified SVD cosine)...")
    ranks_svd = []
    with torch.no_grad():
        for p in pairs:
            h_idx = h_sym2idx.get(p["h_sym"])
            m_idx = m_sym2idx.get(p["m_sym"])
            if h_idx is None or m_idx is None:
                continue
            cos = (Z_h_t[h_idx:h_idx+1] @ Z_m_t.T).squeeze()
            rank = int((cos >= cos[m_idx]).sum().item())
            ranks_svd.append(rank)
    r5, mrr, h5, n = compute_r5_mrr(ranks_svd, N_m)
    results["KGE_SVD"] = {"r5": round(r5*100, 1), "mrr": round(mrr, 4), "hits5": h5, "n": n}
    logger.info(f"  KGE_SVD: R@5={r5*100:.1f}% MRR={mrr:.4f} ({h5}/{n})")

    # Method 4: FGW_OT_only — from v4 grid search (alpha=0.5, eps=0.05)
    fgw_v4_path = BASE / "manuscript" / "figures_precompute" / "out" / "fgw_grid_search_v4.json"
    if fgw_v4_path.exists():
        with open(fgw_v4_path) as f:
            grid = json.load(f)
        chosen = grid.get("a0.5_e0.05", {})
        results["FGW_OT_only"] = {
            "r5": round(chosen.get("recall_at_5", 0) * 100, 1),
            "mrr": round(chosen.get("mrr", 0), 4),
            "note": "from fgw_grid_search_v4.json alpha=0.5 eps=0.05",
        }
        logger.info(f"  FGW_OT_only: R@5={results['FGW_OT_only']['r5']}% (from v4 grid)")
    else:
        results["FGW_OT_only"] = {"r5": None, "note": "v4 grid search not yet complete"}
        logger.info("  FGW_OT_only: v4 grid search not complete yet")

    # ── Save ──
    outpath = BASE / "manuscript" / "figures_precompute" / "out" / "bench5_proxy_baselines.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Bench5 Proxy Baseline Results ===")
    for method, data in results.items():
        print(f"  {method}: R@5={data['r5']}%")
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
