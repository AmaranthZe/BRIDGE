"""Final RRF: All strategies combined for maximum non-ortholog recovery.

Strategies (22 total):
- S1-3: Per-modality SVD cosine (BP, MF, Disease)
- S4-9: SVD ensemble (6 weight combos)
- S10: Unified SVD cosine
- S11: FGW transport mass (v6 model)
- S12-14: 2-hop graph proximity (BP, MF, Disease)
- S15-17: 2-hop weighted ensembles (combined, BP-heavy, MF-heavy)
- S18: Reactome pathway similarity
- S19: PPI direct (human gene's STRING partners mapped to mouse)
- S20: PPI cosine (degree-normalized STRING 2-hop)
- S21: PPI combined (direct + cosine)
- S22: MinAll (best rank across all strategies)
"""

import json, logging, pickle, sys, os
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
import urllib.request
from sklearn.preprocessing import normalize as sk_normalize

from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.data.graph import load_unified_graph
from v2_redesign.model.model import create_model
from v2_redesign.model.hgt import compute_per_modality_gene_features
from v2_redesign.transport.costs import CostComputer
from v2_redesign.transport.fgw import FGWSolver


ALL_CASES = [
    {"name":"DMD→Utrn","h":"ENSG00000198947","sym":"DMD","t_sym":"Utrn","t":"ENSMUSG00000019820","type":"paralog"},
    {"name":"ABCD1→Abcd2","h":"ENSG00000101986","sym":"ABCD1","t_sym":"Abcd2","t":"ENSMUSG00000055782","type":"paralog"},
    {"name":"RB1→Rbl1","h":"ENSG00000139687","sym":"RB1","t_sym":"Rbl1","t":"ENSMUSG00000027641","type":"paralog"},
    {"name":"SHOX→Shox2","h":"ENSG00000185960","sym":"SHOX","t_sym":"Shox2","t":"ENSMUSG00000027833","type":"non_ortholog"},
    {"name":"SERPINA1→S1a","h":"ENSG00000277377","sym":"SERPINA1","t_sym":"Serpina1a","t":"ENSMUSG00000071177","type":"multi_paralog"},
    {"name":"CFTR→Scnn1b","h":"ENSG00000001626","sym":"CFTR","t_sym":"Scnn1b","t":"ENSMUSG00000030873","type":"pathway"},
    {"name":"DKC1→Pot1b","h":"ENSG00000130826","sym":"DKC1","t_sym":"Pot1b","t":"ENSMUSG00000024174","type":"multi_gene"},
    {"name":"TERT→Terc","h":"ENSG00000164362","sym":"TERT","t_sym":"Terc","t":"ENSMUSG00000118406","type":"multi_gene"},
    {"name":"CXCL8→Cxcl1","h":"ENSG00000169429","sym":"CXCL8","t_sym":"Cxcl1","t":"ENSMUSG00000029380","type":"missing_ortholog"},
    {"name":"TLR8→Tlr13","h":"ENSG00000101916","sym":"TLR8","t_sym":"Tlr13","t":"ENSMUSG00000033777","type":"non_ortholog"},
    {"name":"MMP1→Mmp1a","h":"ENSG00000196611","sym":"MMP1","t_sym":"Mmp1a","t":"ENSMUSG00000043089","type":"paralog"},
    {"name":"ALOX15→Alox15","h":"ENSG00000161905","sym":"ALOX15","t_sym":"Alox15","t":"ENSMUSG00000018924","type":"paralog"},
    {"name":"ALOX5→Alox5","h":"ENSG00000012779","sym":"ALOX5","t_sym":"Alox5","t":"ENSMUSG00000025701","type":"paralog"},
    {"name":"ALOX12→Alox12","h":"ENSG00000108839","sym":"ALOX12","t_sym":"Alox12","t":"ENSMUSG00000000320","type":"paralog"},
]


def rrf_score(rank, k=60):
    return 1.0 / (k + rank)


def arr_to_rank(arr, N):
    order = np.argsort(-arr)
    rank = np.zeros(N, dtype=np.int64)
    rank[order] = np.arange(1, N + 1)
    return rank


def sparse_row_to_rank(sparse_mat, hi, N):
    row = np.asarray(sparse_mat[hi].toarray()).squeeze()
    return arr_to_rank(row, N)


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

    h_genes = indexers["human_genes"]
    m_genes = indexers["mouse_genes"]
    m_inv = indexers["mouse_genes_inv"]
    gene_sym = indexers.get("gene_symbols", {})
    sym_to_ensmusg = {sym: eid for eid, sym in gene_sym.items() if eid.startswith("ENSMUSG")}
    N_m = len(m_genes)

    # ── Per-modality SVD features ──
    logger.info("Computing per-modality SVD features...")
    mf = compute_per_modality_gene_features(sparse, 128)
    Z_h_BP = F.normalize(torch.tensor(mf['Z_h_BP']), dim=-1)
    Z_m_BP = F.normalize(torch.tensor(mf['Z_m_BP']), dim=-1)
    Z_h_MF = F.normalize(torch.tensor(mf['Z_h_MF']), dim=-1)
    Z_m_MF = F.normalize(torch.tensor(mf['Z_m_MF']), dim=-1)
    Z_h_D = F.normalize(torch.tensor(mf['Z_h_Disease']), dim=-1)
    Z_m_D = F.normalize(torch.tensor(mf['Z_m_Disease']), dim=-1)
    Z_h_uni = F.normalize(torch.tensor(Z_h, dtype=torch.float32), dim=-1)
    Z_m_uni = F.normalize(torch.tensor(Z_m, dtype=torch.float32), dim=-1)

    # ── 2-hop graph proximity matrices ──
    logger.info("Computing 2-hop proximity matrices...")
    P_BP = sparse["W_hB"] @ sparse["W_mB"].T
    P_MF = sparse["W_hF"] @ sparse["W_mF"].T
    P_Dis = sparse["W_hD"] @ sparse["W_mD"].T

    # ── Reactome pathway matrix ──
    logger.info("Loading Reactome pathways...")
    pw_gene = defaultdict(set)
    reactome_path = "dataset/reactome_ensembl_pathway.tsv"
    if os.path.exists(reactome_path):
        with open(reactome_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    pw_gene[parts[0]].add(parts[3])
        all_pws = sorted(set().union(*pw_gene.values()))
        pw_idx = {pw: i for i, pw in enumerate(all_pws)}
        rh, rc = [], []
        for eid, gi in h_genes.items():
            for pw in pw_gene.get(eid, []):
                if pw in pw_idx: rh.append(gi); rc.append(pw_idx[pw])
        rm, rmc = [], []
        for eid, gi in m_genes.items():
            for pw in pw_gene.get(eid, []):
                if pw in pw_idx: rm.append(gi); rmc.append(pw_idx[pw])
        W_hR = sp.csr_matrix((np.ones(len(rh)), (rh, rc)), shape=(len(h_genes), len(all_pws)))
        W_mR = sp.csr_matrix((np.ones(len(rm)), (rm, rmc)), shape=(N_m, len(all_pws)))
        P_react = W_hR @ W_mR.T
        has_reactome = True
        logger.info(f"  Reactome: {len(all_pws)} pathways, P nnz={P_react.nnz:,}")
    else:
        has_reactome = False

    # ── STRING PPI matrix ──
    logger.info("Loading STRING PPI network...")
    ppi_path = "dataset/string_mouse.txt"
    alias_path = "dataset/string_mouse_aliases.txt"
    has_ppi = os.path.exists(ppi_path) and os.path.exists(alias_path)
    if has_ppi:
        ensp_to_ensg = {}
        with open(alias_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[2] == 'Ensembl_gene':
                    ensp_to_ensg[parts[0]] = parts[1]
        pr, pc, ps = [], [], []
        with open(ppi_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                score = int(parts[2])
                if score < 400: continue
                g1, g2 = ensp_to_ensg.get(parts[0]), ensp_to_ensg.get(parts[1])
                if g1 and g2 and g1 in m_genes and g2 in m_genes:
                    i1, i2 = m_genes[g1], m_genes[g2]
                    pr.extend([i1, i2]); pc.extend([i2, i1])
                    ps.extend([score/1000, score/1000])
        PPI = sp.csr_matrix((ps, (pr, pc)), shape=(N_m, N_m))
        PPI_norm = sk_normalize(PPI, norm='l2', axis=1)
        logger.info(f"  STRING PPI: nnz={PPI.nnz:,}")

    # ── Load v6 model for FGW ──
    logger.info("Loading v6 model for FGW strategy...")
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
    ei, ea = {}, {}
    for et in graph.edge_types:
        ed = graph[et]
        ei[et] = ed.edge_index
        if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
            ea[et] = ed.edge_attr
    with torch.no_grad():
        E_h, E_m, _ = model.get_gene_embeddings(ei, ea)
    cost_computer = CostComputer(Z_h=Z_h, Z_m=Z_m, device=device)
    fgw_solver = FGWSolver(alpha=cfg.fgw.alpha, epsilon=cfg.fgw.epsilon,
                           lambda_prior=0, lambda_memory=0,
                           use_unbalanced=cfg.fgw.use_unbalanced,
                           tau_row=cfg.fgw.tau_row, tau_col=cfg.fgw.tau_col,
                           max_iter=getattr(cfg.infer, 'fgw_refine_iters', 20))
    E_h_dev, E_m_dev = E_h.to(device), E_m.to(device)
    model.candidate_selector = model.candidate_selector.to(device)

    # ── PPI helper ──
    def get_ppi_seeds(h_sym):
        if not has_ppi: return {}
        url = f'https://string-db.org/api/json/interaction_partners?identifiers={h_sym}&species=9606&required_score=400&limit=200'
        try:
            data = json.loads(urllib.request.urlopen(url, timeout=20).read())
            seeds = {}
            for d in data:
                for tn in [d['preferredName_B'], d['preferredName_B'].capitalize()]:
                    me = sym_to_ensmusg.get(tn)
                    if me and me in m_genes:
                        seeds[m_genes[me]] = d['score']/1000.0; break
            return seeds
        except:
            return {}

    # ══════════════════════════════════════════════
    logger.info(f"\n{'='*110}")
    logger.info("FINAL RRF MULTI-STRATEGY VALIDATION (22 strategies)")
    logger.info(f"{'='*110}\n")

    results = []
    for case in ALL_CASES:
        hi = h_genes.get(case["h"])
        ti = m_genes.get(case["t"])
        if hi is None or ti is None:
            continue

        strategies = {}

        # S1-3: Per-modality SVD cosine
        for name, Zh, Zm in [("SVD_BP",Z_h_BP,Z_m_BP),("SVD_MF",Z_h_MF,Z_m_MF),("SVD_Dis",Z_h_D,Z_m_D)]:
            s = (Zh[hi:hi+1] @ Zm.T).squeeze().numpy()
            strategies[name] = arr_to_rank(s, N_m)

        # S4-9: SVD ensembles
        for name, wb, wm, wd in [("E_def",.25,.15,.60),("E_eq",.33,.33,.34),("E_bp",.60,.15,.25),
                                   ("E_bpmf",.40,.40,.20),("E_mfh",.20,.55,.25),("E_bpd",.45,.10,.45)]:
            s = (wb*(Z_h_BP[hi:hi+1]@Z_m_BP.T) + wm*(Z_h_MF[hi:hi+1]@Z_m_MF.T) + wd*(Z_h_D[hi:hi+1]@Z_m_D.T)).squeeze().numpy()
            strategies[name] = arr_to_rank(s, N_m)

        # S10: Unified SVD
        s = (Z_h_uni[hi:hi+1] @ Z_m_uni.T).squeeze().numpy()
        strategies["Uni"] = arr_to_rank(s, N_m)

        # S11: FGW transport mass
        fgw_rank = np.full(N_m, N_m, dtype=np.int64)
        I = torch.tensor([hi], dtype=torch.long, device=device)
        Z_src_d = torch.tensor(Z_h, dtype=torch.float32, device=device)
        Z_tgt_d = torch.tensor(Z_m, dtype=torch.float32, device=device)
        Zsm = {k: v.to(device) for k, v in [('bp',Z_h_BP),('mf',Z_h_MF),('disease',Z_h_D)]}
        Ztm = {k: v.to(device) for k, v in [('bp',Z_m_BP),('mf',Z_m_MF),('disease',Z_m_D)]}
        with torch.no_grad():
            J, sm, P0, rm = model.candidate_selector(E_source=E_h_dev,E_target=E_m_dev,source_indices=I,
                                                      Z_source=Z_src_d,Z_target=Z_tgt_d,Z_source_mods=Zsm,Z_target_mods=Ztm)
            if len(J) > 0:
                costs = cost_computer.compute(E_source=E_h_dev[hi:hi+1],E_target=E_m_dev[J],
                                              source_indices=I,target_indices=J,direction="human_to_mouse")
                T, _ = fgw_solver.solve(C_s=costs["C_s"],C_t=costs["C_t"],M=costs["M"]*sm,Pi0=P0,mask=rm)
                topk = min(100, len(T[0]))
                ts, tp = torch.topk(T[0], topk)
                for r, p in enumerate(tp):
                    fgw_rank[J[p].item()] = r + 1
        strategies["FGW"] = fgw_rank

        # S12-17: 2-hop graph proximity
        for name, P in [("H_bp",P_BP),("H_mf",P_MF),("H_dis",P_Dis)]:
            strategies[name] = sparse_row_to_rank(P, hi, N_m)
        for name, wb, wm, wd in [("H_comb",.25,.15,.60),("H_bph",.50,.30,.20),("H_mfh",.20,.60,.20)]:
            s = (wb*np.asarray(P_BP[hi].toarray()).squeeze() +
                 wm*np.asarray(P_MF[hi].toarray()).squeeze() +
                 wd*np.asarray(P_Dis[hi].toarray()).squeeze())
            strategies[name] = arr_to_rank(s, N_m)

        # S18: Reactome pathway
        if has_reactome:
            strategies["React"] = sparse_row_to_rank(P_react, hi, N_m)

        # S19-21: STRING PPI
        if has_ppi:
            seeds = get_ppi_seeds(case["sym"])
            if seeds:
                sv = np.zeros(N_m)
                for si, sw in seeds.items(): sv[si] = sw
                strategies["PPI_dir"] = arr_to_rank(sv, N_m)
                ppi_cos = PPI_norm.dot(sv / (np.linalg.norm(sv) + 1e-10))
                strategies["PPI_cos"] = arr_to_rank(ppi_cos, N_m)
                comb = sv / (sv.max() + 1e-10) + ppi_cos / (ppi_cos.max() + 1e-10)
                strategies["PPI_cmb"] = arr_to_rank(comb, N_m)

        # S22: MinAll
        all_ranks = list(strategies.values())
        min_rank = all_ranks[0].copy()
        for r in all_ranks[1:]:
            min_rank = np.minimum(min_rank, r)
        strategies["MinAll"] = min_rank

        # ── Weighted RRF Fusion ──
        # PPI, Reactome, and 2-hop strategies get higher weights because they capture
        # non-ortholog functional connections that SVD cosine misses.
        # MinAll gets highest weight to reward genes that are strong in ANY strategy.
        strategy_weights = {}
        for name in strategies:
            if name == "MinAll":
                strategy_weights[name] = 5.0   # Best-of-all gets highest weight
            elif name.startswith("PPI"):
                strategy_weights[name] = 3.0   # PPI captures protein complex membership
            elif name == "React":
                strategy_weights[name] = 3.0   # Reactome captures pathway co-membership
            elif name.startswith("H_"):
                strategy_weights[name] = 2.0   # 2-hop graph captures indirect connections
            elif name == "FGW":
                strategy_weights[name] = 1.5   # Model's own prediction
            else:
                strategy_weights[name] = 1.0   # SVD-based strategies (baseline)

        rrf = np.zeros(N_m, dtype=np.float64)
        for s_name, s_rank in strategies.items():
            w = strategy_weights.get(s_name, 1.0)
            for j in range(N_m):
                rrf[j] += w * rrf_score(s_rank[j])
        rrf_order = np.argsort(-rrf)
        rrf_final = np.zeros(N_m, dtype=np.int64)
        rrf_final[rrf_order] = np.arange(1, N_m + 1)

        final_rank = int(rrf_final[ti])
        marker = " TOP5" if final_rank <= 5 else (" TOP10" if final_rank <= 10 else (" TOP20" if final_rank <= 20 else ""))

        # Best 3 strategies for this target
        best3 = sorted(strategies.items(), key=lambda x: x[1][ti])[:3]
        best3_str = ", ".join(f"{n}:{v[ti]}" for n, v in best3)

        top5_syms = [gene_sym.get(m_inv[rrf_order[i]], m_inv[rrf_order[i]]) for i in range(5)]
        top5_str = " | ".join(f"#{i+1} {s}" for i, s in enumerate(top5_syms))

        logger.info(f"{case['name'].encode('ascii','replace').decode():20s}  RRF={final_rank:>4d}{marker}  best=[{best3_str}]")
        logger.info(f"  {'Top-5':>20s}: {top5_str}")

        results.append({
            "case": case["name"], "type": case["type"],
            "rrf_rank": final_rank, "top5": top5_syms,
            "best_strategies": {n: int(v[ti]) for n, v in best3},
        })

    # ── Summary ──
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")
    top5_count = sum(1 for r in results if r["rrf_rank"] <= 5)
    top10_count = sum(1 for r in results if r["rrf_rank"] <= 10)
    top20_count = sum(1 for r in results if r["rrf_rank"] <= 20)
    logger.info(f"Top-5:  {top5_count}/{len(results)}")
    logger.info(f"Top-10: {top10_count}/{len(results)}")
    logger.info(f"Top-20: {top20_count}/{len(results)}")

    with open("validation_final_rrf_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to validation_final_rrf_results.json")


if __name__ == "__main__":
    main()
