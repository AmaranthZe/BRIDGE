"""Validation with Reciprocal Rank Fusion (RRF) multi-strategy ranking.

Instead of relying solely on FGW transport mass (which destroys SVD signal),
combine multiple ranking strategies:
1. Per-modality SVD: BP-only, MF-only, Disease-only cosine
2. Ensemble SVD: multiple weight combos
3. FGW transport mass (from model)
4. Per-modality-union retrieval: union of top-K from each modality

Final ranking: RRF(all strategies) = sum(1/(k + rank_i)) for each target.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
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
from v2_redesign.model.hgt import compute_per_modality_gene_features
from v2_redesign.transport.costs import CostComputer
from v2_redesign.transport.fgw import FGWSolver


ALL_CASES = [
    {"name":"DMD","human":"ENSG00000198947","sym":"DMD","targets":{"Dmd":"ENSMUSG00000045103","Utrn":"ENSMUSG00000019820"},"type":"paralog"},
    {"name":"ABCD1","human":"ENSG00000101986","sym":"ABCD1","targets":{"Abcd1":"ENSMUSG00000031378","Abcd2":"ENSMUSG00000055782"},"type":"paralog"},
    {"name":"RB1","human":"ENSG00000139687","sym":"RB1","targets":{"Rb1":"ENSMUSG00000022105","Rbl1":"ENSMUSG00000027641"},"type":"paralog"},
    {"name":"SHOX","human":"ENSG00000185960","sym":"SHOX","targets":{"Shox2":"ENSMUSG00000027833"},"type":"non_ortholog"},
    {"name":"SERPINA1","human":"ENSG00000277377","sym":"SERPINA1","targets":{"Serpina1a":"ENSMUSG00000071177","Serpina1b":"ENSMUSG00000071178","Serpina1c":"ENSMUSG00000079015","Serpina1e":"ENSMUSG00000072849"},"type":"multi_paralog"},
    {"name":"CFTR","human":"ENSG00000001626","sym":"CFTR","targets":{"Cftr":"ENSMUSG00000041301","Scnn1b":"ENSMUSG00000030873"},"type":"pathway"},
    {"name":"DKC1","human":"ENSG00000130826","sym":"DKC1","targets":{"Pot1b":"ENSMUSG00000024174","Tert":"ENSMUSG00000021611"},"type":"multi_gene"},
    {"name":"TERT","human":"ENSG00000164362","sym":"TERT","targets":{"Tert":"ENSMUSG00000021611","Terc":"ENSMUSG00000118406"},"type":"multi_gene"},
    {"name":"CXCL8","human":"ENSG00000169429","sym":"CXCL8","targets":{"Cxcl1":"ENSMUSG00000029380","Cxcl2":"ENSMUSG00000058427","Cxcl5":"ENSMUSG00000029371"},"type":"missing_ortholog"},
    {"name":"TLR8","human":"ENSG00000101916","sym":"TLR8","targets":{"Tlr8":"ENSMUSG00000040522","Tlr13":"ENSMUSG00000033777"},"type":"non_ortholog"},
    {"name":"MMP1","human":"ENSG00000196611","sym":"MMP1","targets":{"Mmp1a":"ENSMUSG00000043089","Mmp1b":"ENSMUSG00000041620","Mmp13":"ENSMUSG00000050578"},"type":"paralog"},
    {"name":"ALOX15","human":"ENSG00000161905","sym":"ALOX15","targets":{"Alox15":"ENSMUSG00000018924","Alox12":"ENSMUSG00000000320"},"type":"paralog"},
    {"name":"ALOX5","human":"ENSG00000012779","sym":"ALOX5","targets":{"Alox5":"ENSMUSG00000025701"},"type":"paralog"},
    {"name":"ALOX12","human":"ENSG00000108839","sym":"ALOX12","targets":{"Alox12":"ENSMUSG00000000320"},"type":"paralog"},
]


def rrf_score(rank, k=60):
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


def compute_modality_ranks(hi, Z_h_mod, Z_m_mod):
    """Compute rank of all mouse genes for a single human gene using one modality."""
    score = F.normalize(Z_h_mod[hi:hi+1], dim=-1) @ F.normalize(Z_m_mod, dim=-1).T
    return score.squeeze().argsort(descending=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    proc = Path(cfg.paths.proc_dir)

    logger.info("Loading data...")
    with open(proc / "indexers.pkl", "rb") as f:
        indexers = pickle.load(f)
    Z_h, Z_m, _, _ = load_svd_artifacts(str(proc))
    sparse = load_sparse_matrices(str(proc))
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    # Per-modality SVD features
    logger.info("Computing per-modality SVD features...")
    mf = compute_per_modality_gene_features(sparse, 128)
    Z_h_BP = torch.tensor(mf['Z_h_BP'], dtype=torch.float32)
    Z_m_BP = torch.tensor(mf['Z_m_BP'], dtype=torch.float32)
    Z_h_MF = torch.tensor(mf['Z_h_MF'], dtype=torch.float32)
    Z_m_MF = torch.tensor(mf['Z_m_MF'], dtype=torch.float32)
    Z_h_Dis = torch.tensor(mf['Z_h_Disease'], dtype=torch.float32)
    Z_m_Dis = torch.tensor(mf['Z_m_Disease'], dtype=torch.float32)

    # Unified SVD
    Z_h_uni = torch.tensor(Z_h, dtype=torch.float32)
    Z_m_uni = torch.tensor(Z_m, dtype=torch.float32)

    # Graph-based 2-hop proximity: W_h @ W_m.T
    logger.info("Computing 2-hop cross-species proximity matrices...")
    P_BP = sparse["W_hB"] @ sparse["W_mB"].T     # [N_h, N_m] sparse
    P_MF = sparse["W_hF"] @ sparse["W_mF"].T
    P_Dis = sparse["W_hD"] @ sparse["W_mD"].T
    logger.info(f"  P_BP nnz={P_BP.nnz:,}, P_MF nnz={P_MF.nnz:,}, P_Dis nnz={P_Dis.nnz:,}")

    N_m = Z_m_BP.shape[0]
    mouse_inv = indexers["mouse_genes_inv"]
    gene_sym = indexers.get("gene_symbols", {})

    # Load v6 model for FGW strategy
    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }
    ckpt_path = proc / "runs" / "retrain_v6" / "best_model.pt"
    model = create_model(cfg, node_counts)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    for key in ["Z_h_BP","Z_m_BP","Z_h_MF","Z_m_MF","Z_h_Disease","Z_m_Disease"]:
        fk = f"hgt.{key}"
        if fk in state and not hasattr(model.hgt, key):
            model.hgt.register_buffer(key, state[fk])
    model.load_state_dict(state, strict=False)
    model.eval()

    # Compute HGT embeddings
    edge_index_dict, edge_attr_dict = {}, {}
    for et in graph.edge_types:
        ed = graph[et]
        edge_index_dict[et] = ed.edge_index
        if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
            edge_attr_dict[et] = ed.edge_attr
    with torch.no_grad():
        E_h, E_m, _ = model.get_gene_embeddings(edge_index_dict, edge_attr_dict)

    cost_computer = CostComputer(Z_h=Z_h, Z_m=Z_m, device=device)
    fgw_solver = FGWSolver(
        alpha=cfg.fgw.alpha, epsilon=cfg.fgw.epsilon,
        lambda_prior=0, lambda_memory=0,
        use_unbalanced=cfg.fgw.use_unbalanced,
        tau_row=cfg.fgw.tau_row, tau_col=cfg.fgw.tau_col,
        max_iter=getattr(cfg.infer, 'fgw_refine_iters', 20),
    )
    E_h_dev, E_m_dev = E_h.to(device), E_m.to(device)
    model.candidate_selector = model.candidate_selector.to(device)

    # ── Run all cases ──
    logger.info(f"\n{'='*100}")
    logger.info("RRF MULTI-STRATEGY VALIDATION")
    logger.info(f"{'='*100}")

    for case in ALL_CASES:
        hi = indexers["human_genes"].get(case["human"])
        if hi is None:
            continue

        logger.info(f"\n--- {case['name']} ({case['sym']}) ---")

        # ── Strategy 1-3: Per-modality SVD ranks ──
        bp_order = compute_modality_ranks(hi, Z_h_BP, Z_m_BP)
        mf_order = compute_modality_ranks(hi, Z_h_MF, Z_m_MF)
        dis_order = compute_modality_ranks(hi, Z_h_Dis, Z_m_Dis)

        # Convert to rank arrays (gene_idx -> rank)
        bp_rank = torch.zeros(N_m, dtype=torch.long)
        mf_rank = torch.zeros(N_m, dtype=torch.long)
        dis_rank = torch.zeros(N_m, dtype=torch.long)
        bp_rank[bp_order] = torch.arange(1, N_m+1)
        mf_rank[mf_order] = torch.arange(1, N_m+1)
        dis_rank[dis_order] = torch.arange(1, N_m+1)

        # ── Strategy 4-6: Weighted SVD ensembles ──
        def ensemble_rank(w_bp, w_mf, w_dis):
            score = (w_bp * (F.normalize(Z_h_BP[hi:hi+1], dim=-1) @ F.normalize(Z_m_BP, dim=-1).T) +
                     w_mf * (F.normalize(Z_h_MF[hi:hi+1], dim=-1) @ F.normalize(Z_m_MF, dim=-1).T) +
                     w_dis * (F.normalize(Z_h_Dis[hi:hi+1], dim=-1) @ F.normalize(Z_m_Dis, dim=-1).T))
            order = score.squeeze().argsort(descending=True)
            rank = torch.zeros(N_m, dtype=torch.long)
            rank[order] = torch.arange(1, N_m+1)
            return rank

        ens_default = ensemble_rank(0.25, 0.15, 0.60)
        ens_equal = ensemble_rank(0.33, 0.33, 0.34)
        ens_bp = ensemble_rank(0.60, 0.15, 0.25)
        ens_bp_mf = ensemble_rank(0.40, 0.40, 0.20)
        ens_mf_heavy = ensemble_rank(0.20, 0.55, 0.25)
        ens_bp_dis = ensemble_rank(0.45, 0.10, 0.45)

        # ── Strategy 7-9: Graph 2-hop proximity ──
        def twohop_rank(P_sparse):
            row = np.asarray(P_sparse[hi].toarray()).squeeze()
            if row.max() == 0:
                return torch.full((N_m,), N_m, dtype=torch.long)
            scores = torch.tensor(row, dtype=torch.float32)
            order = scores.argsort(descending=True)
            rank = torch.zeros(N_m, dtype=torch.long)
            rank[order] = torch.arange(1, N_m+1)
            return rank

        hop_bp = twohop_rank(P_BP)
        hop_mf = twohop_rank(P_MF)
        hop_dis = twohop_rank(P_Dis)

        # 2-hop ensemble: BP-heavy (structural genes) and default
        hop_combined = np.asarray(
            0.25 * P_BP[hi].toarray() + 0.15 * P_MF[hi].toarray() + 0.60 * P_Dis[hi].toarray()
        ).squeeze()
        hop_bp_heavy = np.asarray(
            0.50 * P_BP[hi].toarray() + 0.30 * P_MF[hi].toarray() + 0.20 * P_Dis[hi].toarray()
        ).squeeze()
        hop_mf_heavy = np.asarray(
            0.20 * P_BP[hi].toarray() + 0.60 * P_MF[hi].toarray() + 0.20 * P_Dis[hi].toarray()
        ).squeeze()

        def arr_to_rank(arr):
            scores = torch.tensor(arr, dtype=torch.float32)
            order = scores.argsort(descending=True)
            rank = torch.zeros(N_m, dtype=torch.long)
            rank[order] = torch.arange(1, N_m+1)
            return rank

        hop_comb_rank = arr_to_rank(hop_combined)
        hop_bp_h_rank = arr_to_rank(hop_bp_heavy)
        hop_mf_h_rank = arr_to_rank(hop_mf_heavy)

        # ── Strategy 10: Unified SVD cosine ──
        uni_score = F.normalize(Z_h_uni[hi:hi+1], dim=-1) @ F.normalize(Z_m_uni, dim=-1).T
        uni_order = uni_score.squeeze().argsort(descending=True)
        uni_rank = torch.zeros(N_m, dtype=torch.long)
        uni_rank[uni_order] = torch.arange(1, N_m+1)

        # ── Strategy 8: FGW transport mass ──
        I = torch.tensor([hi], dtype=torch.long, device=device)
        Z_src_dev = torch.tensor(Z_h, dtype=torch.float32, device=device)
        Z_tgt_dev = torch.tensor(Z_m, dtype=torch.float32, device=device)
        Z_src_mods = {'bp': Z_h_BP.to(device), 'mf': Z_h_MF.to(device), 'disease': Z_h_Dis.to(device)}
        Z_tgt_mods = {'bp': Z_m_BP.to(device), 'mf': Z_m_MF.to(device), 'disease': Z_m_Dis.to(device)}

        fgw_rank = torch.full((N_m,), N_m, dtype=torch.long)  # default: worst rank
        with torch.no_grad():
            J_cand, soft_mask, Pi0, row_mask = model.candidate_selector(
                E_source=E_h_dev, E_target=E_m_dev, source_indices=I,
                Z_source=Z_src_dev, Z_target=Z_tgt_dev,
                Z_source_mods=Z_src_mods, Z_target_mods=Z_tgt_mods)
            if len(J_cand) > 0:
                costs = cost_computer.compute(
                    E_source=E_h_dev[hi:hi+1], E_target=E_m_dev[J_cand],
                    source_indices=I, target_indices=J_cand, direction="human_to_mouse")
                M_masked = costs["M"] * soft_mask
                T_local, _ = fgw_solver.solve(
                    C_s=costs["C_s"], C_t=costs["C_t"],
                    M=M_masked, Pi0=Pi0, mask=row_mask)
                t_row = T_local[0]
                topk = min(100, len(t_row))
                topk_scores, topk_pos = torch.topk(t_row, topk)
                for r, pos in enumerate(topk_pos):
                    gene_idx = J_cand[pos].item()
                    fgw_rank[gene_idx] = r + 1

        # ── RRF Fusion ──
        rrf_scores = torch.zeros(N_m, dtype=torch.float64)
        strategies = [bp_rank, mf_rank, dis_rank, ens_default, ens_equal, ens_bp,
                      ens_bp_mf, ens_mf_heavy, ens_bp_dis, uni_rank, fgw_rank,
                      hop_bp, hop_mf, hop_dis, hop_comb_rank, hop_bp_h_rank, hop_mf_h_rank]
        strategy_names = ["BP", "MF", "Dis", "E_def", "E_eq", "E_bp",
                         "E_bpmf", "E_mfh", "E_bpd", "Uni", "FGW",
                         "H_bp", "H_mf", "H_dis", "H_comb", "H_bph", "H_mfh"]

        # Add min-rank strategy across ALL strategies
        min_rank_arr = strategies[0].clone()
        for s in strategies[1:]:
            min_rank_arr = torch.minimum(min_rank_arr, s)
        strategies.append(min_rank_arr)
        strategy_names.append("MinAll")

        for s_rank in strategies:
            for j in range(N_m):
                rrf_scores[j] += rrf_score(s_rank[j].item())

        # Final ranking
        rrf_order = rrf_scores.argsort(descending=True)
        rrf_final_rank = torch.zeros(N_m, dtype=torch.long)
        rrf_final_rank[rrf_order] = torch.arange(1, N_m+1)

        # ── Report results ──
        for t_sym, t_eid in case["targets"].items():
            ti = indexers["mouse_genes"].get(t_eid)
            if ti is None:
                logger.info(f"  {t_sym}: NOT IN INDEX")
                continue
            # Per-strategy ranks
            ranks_str = " | ".join(
                f"{sn}:{s_rank[ti].item():>5d}" for sn, s_rank in zip(strategy_names, strategies))
            rr = rrf_final_rank[ti].item()
            marker = " <<<TOP5" if rr <= 5 else (" <<TOP10" if rr <= 10 else (" <TOP20" if rr <= 20 else ""))
            logger.info(f"  {t_sym:12s} RRF_rank={rr:>4d}{marker}  [{ranks_str}]")

        # Top-5 final
        top5 = rrf_order[:5]
        t5_str = " | ".join(f"#{i+1} {gene_sym.get(mouse_inv[t5.item()], mouse_inv[t5.item()])}"
                           for i, t5 in enumerate(top5))
        logger.info(f"  Top-5 RRF: {t5_str}")

    logger.info(f"\n{'='*100}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
