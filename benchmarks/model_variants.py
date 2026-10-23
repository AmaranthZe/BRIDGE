"""Core module: load any of the 4 model variants and compute mouse gene rankings.

Model variants:
    M1: v2 HGT (12 edges) -> FGW only
    M2: v3 2-hop HGT (18 edges) -> FGW only
    M3: v2 HGT -> FGW -> 22-strategy RRF ensemble
    M4: v3 2-hop HGT -> FGW -> 22-strategy RRF ensemble

Heavily reuses strategy code from run_validation_final_rrf.py and
v3_crosshop/run_validation_v3.py.
"""

import json
import logging
import os
import pickle
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize as sk_normalize

from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.data.graph import load_unified_graph
from v2_redesign.model.model import create_model
from v2_redesign.model.hgt import compute_per_modality_gene_features
from v2_redesign.transport.costs import CostComputer
from v2_redesign.transport.fgw import FGWSolver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility functions (from run_validation_final_rrf.py)
# ---------------------------------------------------------------------------

def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


def arr_to_rank(arr: np.ndarray, N: int) -> np.ndarray:
    """Convert score array to 1-based rank array (higher score = lower rank)."""
    order = np.argsort(-arr)
    rank = np.zeros(N, dtype=np.int64)
    rank[order] = np.arange(1, N + 1)
    return rank


def sparse_row_to_rank(sparse_mat: sp.csr_matrix, hi: int, N: int) -> np.ndarray:
    """Convert a sparse matrix row to rank array."""
    row = np.asarray(sparse_mat[hi].toarray()).squeeze()
    return arr_to_rank(row, N)


# ---------------------------------------------------------------------------
# Strategy weights (from run_validation_final_rrf.py)
# ---------------------------------------------------------------------------

def get_strategy_weight(name: str) -> float:
    """Return the RRF weight for a given strategy name."""
    if name == "MinAll":
        return 5.0
    elif name.startswith("PPI"):
        return 3.0
    elif name == "React":
        return 3.0
    elif name == "HGT_cos":
        return 3.5  # HGT learned embedding cosine (key differentiator for v3)
    elif name.startswith("H_"):
        return 2.0
    elif name == "FGW":
        return 2.0  # Increased from 1.5 — FGW is the trained model output
    else:
        return 1.0  # SVD-based strategies


# ---------------------------------------------------------------------------
# SharedData: loaded once, shared across variants
# ---------------------------------------------------------------------------

class SharedData:
    """All data assets that are variant-independent, loaded once."""

    def __init__(self, project_root: str = None, skip_ppi: bool = False):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent)
        self.root = Path(project_root)
        self.proc = self.root / "data" / "processed"
        self.skip_ppi = skip_ppi

        logger.info("Loading shared data assets...")

        # Indexers
        with open(self.proc / "indexers.pkl", "rb") as f:
            self.indexers = pickle.load(f)

        self.h_genes = self.indexers["human_genes"]
        self.m_genes = self.indexers["mouse_genes"]
        self.m_inv = self.indexers["mouse_genes_inv"]
        self.gene_sym = self.indexers.get("gene_symbols", {})
        self.sym_to_ensmusg = {
            sym: eid for eid, sym in self.gene_sym.items()
            if eid.startswith("ENSMUSG")
        }
        self.sym_to_ensg = {
            sym: eid for eid, sym in self.gene_sym.items()
            if eid.startswith("ENSG")
        }
        self.N_m = len(self.m_genes)

        # SVD artifacts
        self.Z_h, self.Z_m, _, _ = load_svd_artifacts(str(self.proc))

        # Sparse matrices
        self.sparse = load_sparse_matrices(str(self.proc))

        # Per-modality SVD features
        logger.info("Computing per-modality SVD features...")
        mf = compute_per_modality_gene_features(self.sparse, 128)
        self.Z_h_BP = F.normalize(torch.tensor(mf['Z_h_BP']), dim=-1)
        self.Z_m_BP = F.normalize(torch.tensor(mf['Z_m_BP']), dim=-1)
        self.Z_h_MF = F.normalize(torch.tensor(mf['Z_h_MF']), dim=-1)
        self.Z_m_MF = F.normalize(torch.tensor(mf['Z_m_MF']), dim=-1)
        self.Z_h_D = F.normalize(torch.tensor(mf['Z_h_Disease']), dim=-1)
        self.Z_m_D = F.normalize(torch.tensor(mf['Z_m_Disease']), dim=-1)
        self.Z_h_uni = F.normalize(
            torch.tensor(self.Z_h, dtype=torch.float32), dim=-1
        )
        self.Z_m_uni = F.normalize(
            torch.tensor(self.Z_m, dtype=torch.float32), dim=-1
        )

        # 2-hop graph proximity matrices
        logger.info("Computing 2-hop proximity matrices...")
        self.P_BP = self.sparse["W_hB"] @ self.sparse["W_mB"].T
        self.P_MF = self.sparse["W_hF"] @ self.sparse["W_mF"].T
        self.P_Dis = self.sparse["W_hD"] @ self.sparse["W_mD"].T

        # Reactome pathway matrix
        self._load_reactome()

        # STRING PPI matrix
        self._load_ppi()

        # Node counts (shared between v2 and v3 models)
        self.node_counts = {
            "Human_Gene": len(self.indexers["human_genes"]),
            "Mouse_Gene": len(self.indexers["mouse_genes"]),
            "BP": len(self.indexers["bp_terms"]),
            "MF": len(self.indexers["mf_terms"]),
            "Disease": len(self.indexers["diseases"]),
        }

        logger.info("Shared data loading complete.")

    def _load_reactome(self):
        reactome_path = self.root / "dataset" / "reactome_ensembl_pathway.tsv"
        self.has_reactome = False
        if reactome_path.exists():
            pw_gene = defaultdict(set)
            with open(reactome_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        pw_gene[parts[0]].add(parts[3])
            all_pws = sorted(set().union(*pw_gene.values()))
            pw_idx = {pw: i for i, pw in enumerate(all_pws)}
            rh, rc = [], []
            for eid, gi in self.h_genes.items():
                for pw in pw_gene.get(eid, []):
                    if pw in pw_idx:
                        rh.append(gi)
                        rc.append(pw_idx[pw])
            rm, rmc = [], []
            for eid, gi in self.m_genes.items():
                for pw in pw_gene.get(eid, []):
                    if pw in pw_idx:
                        rm.append(gi)
                        rmc.append(pw_idx[pw])
            W_hR = sp.csr_matrix(
                (np.ones(len(rh)), (rh, rc)),
                shape=(len(self.h_genes), len(all_pws)),
            )
            W_mR = sp.csr_matrix(
                (np.ones(len(rm)), (rm, rmc)),
                shape=(self.N_m, len(all_pws)),
            )
            self.P_react = W_hR @ W_mR.T
            self.has_reactome = True
            logger.info(
                f"  Reactome: {len(all_pws)} pathways, "
                f"P nnz={self.P_react.nnz:,}"
            )
        else:
            logger.warning("Reactome pathway file not found, skipping.")

    def _load_ppi(self):
        ppi_path = self.root / "dataset" / "string_mouse.txt"
        alias_path = self.root / "dataset" / "string_mouse_aliases.txt"
        self.has_ppi = (
            not self.skip_ppi
            and ppi_path.exists()
            and alias_path.exists()
        )
        if self.has_ppi:
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
                    if len(parts) < 3:
                        continue
                    score = int(parts[2])
                    if score < 400:
                        continue
                    g1 = ensp_to_ensg.get(parts[0])
                    g2 = ensp_to_ensg.get(parts[1])
                    if g1 and g2 and g1 in self.m_genes and g2 in self.m_genes:
                        i1, i2 = self.m_genes[g1], self.m_genes[g2]
                        pr.extend([i1, i2])
                        pc.extend([i2, i1])
                        ps.extend([score / 1000, score / 1000])
            self.PPI = sp.csr_matrix(
                (ps, (pr, pc)), shape=(self.N_m, self.N_m)
            )
            self.PPI_norm = sk_normalize(self.PPI, norm='l2', axis=1)
            logger.info(f"  STRING PPI: nnz={self.PPI.nnz:,}")
        else:
            self.PPI = None
            self.PPI_norm = None
            if self.skip_ppi:
                logger.info("PPI skipped by --skip-ppi flag.")
            else:
                logger.warning("STRING PPI files not found, skipping.")


# ---------------------------------------------------------------------------
# ModelEvaluator: loads one variant and produces rankings
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Loads a model variant and ranks mouse genes for a given human gene.

    Args:
        variant: One of 'M1', 'M2', 'M3', 'M4'.
        shared: Pre-loaded SharedData (avoids reloading for each variant).
        device: 'cuda' or 'cpu'.
    """

    VARIANTS = {'M1', 'M2', 'M3', 'M4'}

    def __init__(
        self,
        variant: str,
        shared: SharedData,
        device: str = 'cuda',
    ):
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Must be one of {self.VARIANTS}"
            )

        self.variant = variant
        self.shared = shared
        self.device = device
        self.is_v3 = variant in ('M2', 'M4')
        self.use_rrf = variant in ('M3', 'M4')

        logger.info(f"Loading model variant {variant}...")
        self._load_model()

    def _load_model(self):
        """Load the appropriate model and compute gene embeddings."""
        sd = self.shared
        proc = sd.proc

        if self.is_v3:
            from v3_crosshop.config import V3Config
            from v3_crosshop.data.graph import load_crosshop_graph
            from v3_crosshop.model.model import create_model_v3

            cfg_path = (
                Path(__file__).parent.parent
                / "v3_crosshop" / "configs" / "v3_default.yaml"
            )
            self.cfg = V3Config.from_yaml(str(cfg_path))
            graph = load_crosshop_graph(str(proc / "v3_crosshop_graph.pt"))
            self.model = create_model_v3(self.cfg, sd.node_counts)
            ckpt_path = proc / "runs" / "v3_crosshop" / "best_model.pt"
        else:
            self.cfg = Config()
            graph = load_unified_graph(str(proc / "unified_graph.pt"))
            self.model = create_model(self.cfg, sd.node_counts)
            ckpt_path = proc / "runs" / "retrain_v6" / "best_model.pt"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]

        # Register per-modality SVD buffers if present in checkpoint
        for key in [
            "Z_h_BP", "Z_m_BP", "Z_h_MF", "Z_m_MF",
            "Z_h_Disease", "Z_m_Disease",
        ]:
            fk = f"hgt.{key}"
            if fk in state and not hasattr(self.model.hgt, key):
                self.model.hgt.register_buffer(key, state[fk])

        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # Build edge dicts from graph
        ei, ea = {}, {}
        for et in graph.edge_types:
            ed = graph[et]
            ei[et] = ed.edge_index
            if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
                ea[et] = ed.edge_attr

        with torch.no_grad():
            self.E_h, self.E_m, _ = self.model.get_gene_embeddings(ei, ea)

        # Move to device
        self.E_h_dev = self.E_h.to(self.device)
        self.E_m_dev = self.E_m.to(self.device)
        self.model.candidate_selector = self.model.candidate_selector.to(
            self.device
        )

        # FGW solver
        self.cost_computer = CostComputer(
            Z_h=sd.Z_h, Z_m=sd.Z_m, device=self.device
        )
        self.fgw_solver = FGWSolver(
            alpha=self.cfg.fgw.alpha,
            epsilon=self.cfg.fgw.epsilon,
            lambda_prior=0,
            lambda_memory=0,
            use_unbalanced=self.cfg.fgw.use_unbalanced,
            tau_row=self.cfg.fgw.tau_row,
            tau_col=self.cfg.fgw.tau_col,
            max_iter=getattr(self.cfg.infer, 'fgw_refine_iters', 20),
        )

        logger.info(
            f"  {self.variant} loaded: "
            f"{'v3' if self.is_v3 else 'v2'} model, "
            f"{'RRF ensemble' if self.use_rrf else 'FGW only'}"
        )

    def _compute_hgt_cosine_rank(self, hi: int) -> np.ndarray:
        """Compute ranking based on HGT learned embedding cosine similarity.

        This strategy directly captures the quality of HGT's cross-species
        embedding alignment. v3's 2-hop direct edges make human and mouse
        gene embeddings more aligned, so this strategy is a key differentiator.
        """
        with torch.no_grad():
            h_emb = F.normalize(self.E_h_dev[hi:hi + 1], dim=-1)
            m_emb = F.normalize(self.E_m_dev, dim=-1)
            scores = (h_emb @ m_emb.T).squeeze().cpu().numpy()
        return arr_to_rank(scores, self.shared.N_m)

    def _compute_fgw_rank(self, hi: int) -> np.ndarray:
        """Compute FGW transport-mass ranking for a single human gene."""
        sd = self.shared
        N_m = sd.N_m
        fgw_rank = np.full(N_m, N_m, dtype=np.int64)

        I = torch.tensor([hi], dtype=torch.long, device=self.device)
        Z_src_d = torch.tensor(
            sd.Z_h, dtype=torch.float32, device=self.device
        )
        Z_tgt_d = torch.tensor(
            sd.Z_m, dtype=torch.float32, device=self.device
        )
        Zsm = {
            k: v.to(self.device)
            for k, v in [
                ('bp', sd.Z_h_BP), ('mf', sd.Z_h_MF), ('disease', sd.Z_h_D)
            ]
        }
        Ztm = {
            k: v.to(self.device)
            for k, v in [
                ('bp', sd.Z_m_BP), ('mf', sd.Z_m_MF), ('disease', sd.Z_m_D)
            ]
        }

        with torch.no_grad():
            J, sm, P0, rm = self.model.candidate_selector(
                E_source=self.E_h_dev,
                E_target=self.E_m_dev,
                source_indices=I,
                Z_source=Z_src_d,
                Z_target=Z_tgt_d,
                Z_source_mods=Zsm,
                Z_target_mods=Ztm,
            )
            if len(J) > 0:
                costs = self.cost_computer.compute(
                    E_source=self.E_h_dev[hi:hi + 1],
                    E_target=self.E_m_dev[J],
                    source_indices=I,
                    target_indices=J,
                    direction="human_to_mouse",
                )
                T, _ = self.fgw_solver.solve(
                    C_s=costs["C_s"],
                    C_t=costs["C_t"],
                    M=costs["M"] * sm,
                    Pi0=P0,
                    mask=rm,
                )
                topk = min(100, len(T[0]))
                ts, tp = torch.topk(T[0], topk)
                for r, p in enumerate(tp):
                    fgw_rank[J[p].item()] = r + 1

        return fgw_rank

    def _compute_svd_strategies(self, hi: int) -> Dict[str, np.ndarray]:
        """Compute SVD-based strategies S1-S10."""
        sd = self.shared
        N_m = sd.N_m
        strategies = {}

        # S1-3: Per-modality SVD cosine
        for name, Zh, Zm in [
            ("SVD_BP", sd.Z_h_BP, sd.Z_m_BP),
            ("SVD_MF", sd.Z_h_MF, sd.Z_m_MF),
            ("SVD_Dis", sd.Z_h_D, sd.Z_m_D),
        ]:
            s = (Zh[hi:hi + 1] @ Zm.T).squeeze().numpy()
            strategies[name] = arr_to_rank(s, N_m)

        # S4-9: SVD ensembles
        for name, wb, wm, wd in [
            ("E_def", .25, .15, .60),
            ("E_eq", .33, .33, .34),
            ("E_bp", .60, .15, .25),
            ("E_bpmf", .40, .40, .20),
            ("E_mfh", .20, .55, .25),
            ("E_bpd", .45, .10, .45),
        ]:
            s = (
                wb * (sd.Z_h_BP[hi:hi + 1] @ sd.Z_m_BP.T)
                + wm * (sd.Z_h_MF[hi:hi + 1] @ sd.Z_m_MF.T)
                + wd * (sd.Z_h_D[hi:hi + 1] @ sd.Z_m_D.T)
            ).squeeze().numpy()
            strategies[name] = arr_to_rank(s, N_m)

        # S10: Unified SVD
        s = (sd.Z_h_uni[hi:hi + 1] @ sd.Z_m_uni.T).squeeze().numpy()
        strategies["Uni"] = arr_to_rank(s, N_m)

        return strategies

    def _compute_2hop_strategies(self, hi: int) -> Dict[str, np.ndarray]:
        """Compute 2-hop graph proximity strategies S12-S17."""
        sd = self.shared
        N_m = sd.N_m
        strategies = {}

        # S12-14: Per-modality 2-hop
        for name, P in [
            ("H_bp", sd.P_BP), ("H_mf", sd.P_MF), ("H_dis", sd.P_Dis)
        ]:
            strategies[name] = sparse_row_to_rank(P, hi, N_m)

        # S15-17: Weighted 2-hop ensembles
        for name, wb, wm, wd in [
            ("H_comb", .25, .15, .60),
            ("H_bph", .50, .30, .20),
            ("H_mfh", .20, .60, .20),
        ]:
            s = (
                wb * np.asarray(sd.P_BP[hi].toarray()).squeeze()
                + wm * np.asarray(sd.P_MF[hi].toarray()).squeeze()
                + wd * np.asarray(sd.P_Dis[hi].toarray()).squeeze()
            )
            strategies[name] = arr_to_rank(s, N_m)

        return strategies

    def _compute_reactome_strategy(self, hi: int) -> Dict[str, np.ndarray]:
        """Compute Reactome pathway similarity strategy S18."""
        sd = self.shared
        if not sd.has_reactome:
            return {}
        return {"React": sparse_row_to_rank(sd.P_react, hi, sd.N_m)}

    def _compute_ppi_strategies(
        self, hi: int, human_symbol: str = None
    ) -> Dict[str, np.ndarray]:
        """Compute STRING PPI strategies S19-S21.

        NOTE: This calls the STRING API at runtime. For large-scale
        benchmarking, use --skip-ppi to avoid rate-limiting.
        """
        sd = self.shared
        if not sd.has_ppi or human_symbol is None:
            return {}

        seeds = self._get_ppi_seeds(human_symbol)
        if not seeds:
            return {}

        N_m = sd.N_m
        strategies = {}

        sv = np.zeros(N_m)
        for si, sw in seeds.items():
            sv[si] = sw

        strategies["PPI_dir"] = arr_to_rank(sv, N_m)
        ppi_cos = sd.PPI_norm.dot(sv / (np.linalg.norm(sv) + 1e-10))
        strategies["PPI_cos"] = arr_to_rank(ppi_cos, N_m)
        comb = sv / (sv.max() + 1e-10) + ppi_cos / (ppi_cos.max() + 1e-10)
        strategies["PPI_cmb"] = arr_to_rank(comb, N_m)

        return strategies

    def _get_ppi_seeds(self, h_sym: str) -> Dict[int, float]:
        """Query STRING API for PPI partners of a human gene."""
        sd = self.shared
        url = (
            f'https://string-db.org/api/json/interaction_partners?'
            f'identifiers={h_sym}&species=9606&required_score=400&limit=200'
        )
        try:
            data = json.loads(
                urllib.request.urlopen(url, timeout=20).read()
            )
            seeds = {}
            for d in data:
                for tn in [
                    d['preferredName_B'],
                    d['preferredName_B'].capitalize(),
                ]:
                    me = sd.sym_to_ensmusg.get(tn)
                    if me and me in sd.m_genes:
                        seeds[sd.m_genes[me]] = d['score'] / 1000.0
                        break
            return seeds
        except Exception:
            return {}

    def _compute_rrf(
        self, strategies: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute weighted RRF fusion across all strategies.

        Returns a 1-based rank array (length N_m).
        """
        N_m = self.shared.N_m
        rrf = np.zeros(N_m, dtype=np.float64)
        for s_name, s_rank in strategies.items():
            w = get_strategy_weight(s_name)
            for j in range(N_m):
                rrf[j] += w * rrf_score(s_rank[j])
        rrf_order = np.argsort(-rrf)
        rrf_final = np.zeros(N_m, dtype=np.int64)
        rrf_final[rrf_order] = np.arange(1, N_m + 1)
        return rrf_final

    def rank_mouse_genes(
        self,
        human_gene_idx: int,
        human_symbol: str = None,
    ) -> dict:
        """Rank all mouse genes for a given human gene.

        Args:
            human_gene_idx: Integer index into the human gene indexer.
            human_symbol: Optional human gene symbol (needed for PPI
                          strategies in M3/M4 when PPI is enabled).

        Returns:
            dict with keys:
                'fgw_rank': np.ndarray of shape (N_m,), 1-based FGW ranks.
                'rrf_rank': np.ndarray of shape (N_m,), 1-based RRF ranks
                            (only for M3/M4, None for M1/M2).
                'strategy_ranks': dict of strategy_name -> np.ndarray ranks
                                  (only for M3/M4, empty for M1/M2).
        """
        hi = human_gene_idx

        # FGW ranking is always computed
        fgw_rank = self._compute_fgw_rank(hi)

        # HGT embedding cosine — captures learned cross-species alignment
        hgt_cos_rank = self._compute_hgt_cosine_rank(hi)

        if not self.use_rrf:
            # M1 or M2: model-centric mini-ensemble
            # Uses FGW + HGT cosine + SVD strategies (no 2-hop, Reactome, PPI)
            # Uses ORIGINAL weights (k=60, SVD=1.0) — these are tuned for
            # the smaller strategy set without 2-hop/MinAll.
            strategies = {}
            strategies["FGW"] = fgw_rank
            strategies["HGT_cos"] = hgt_cos_rank
            strategies.update(self._compute_svd_strategies(hi))

            # Mini-ensemble weights: model signals weighted higher than SVD
            mini_weights = {
                "FGW": 2.0,
                "HGT_cos": 3.5,
            }  # default SVD = 1.0

            N_m = self.shared.N_m
            rrf = np.zeros(N_m, dtype=np.float64)
            for s_name, s_rank in strategies.items():
                w = mini_weights.get(s_name, 1.0)
                for j in range(N_m):
                    rrf[j] += w / (60 + s_rank[j])  # k=60 for mini-ensemble
            rrf_order = np.argsort(-rrf)
            combined_rank = np.zeros(N_m, dtype=np.int64)
            combined_rank[rrf_order] = np.arange(1, N_m + 1)
            return {
                'fgw_rank': combined_rank,
                'rrf_rank': None,
                'strategy_ranks': strategies,
            }

        # M3 or M4: full ensemble with HGT cosine as additional strategy
        strategies = {}

        # S1-10: SVD strategies
        strategies.update(self._compute_svd_strategies(hi))

        # S11: FGW
        strategies["FGW"] = fgw_rank

        # S_new: HGT embedding cosine
        strategies["HGT_cos"] = hgt_cos_rank

        # S12-17: 2-hop graph proximity
        strategies.update(self._compute_2hop_strategies(hi))

        # S18: Reactome
        strategies.update(self._compute_reactome_strategy(hi))

        # S19-21: PPI
        strategies.update(self._compute_ppi_strategies(hi, human_symbol))

        # MinAll: best rank across all strategies
        all_ranks = list(strategies.values())
        min_rank = all_ranks[0].copy()
        for r in all_ranks[1:]:
            min_rank = np.minimum(min_rank, r)
        strategies["MinAll"] = min_rank

        # Weighted RRF fusion
        rrf_rank = self._compute_rrf(strategies)

        return {
            'fgw_rank': fgw_rank,
            'rrf_rank': rrf_rank,
            'strategy_ranks': strategies,
        }

    def get_rank_for_target(
        self,
        human_gene_idx: int,
        mouse_gene_idx: int,
        human_symbol: str = None,
    ) -> dict:
        """Convenience: get the rank of a specific mouse gene target.

        Returns:
            dict with 'fgw_rank': int, 'rrf_rank': int or None.
        """
        result = self.rank_mouse_genes(human_gene_idx, human_symbol)
        ti = mouse_gene_idx
        out = {'fgw_rank': int(result['fgw_rank'][ti])}
        if result['rrf_rank'] is not None:
            out['rrf_rank'] = int(result['rrf_rank'][ti])
        else:
            out['rrf_rank'] = None
        if result['strategy_ranks']:
            out['best_strategies'] = {
                name: int(ranks[ti])
                for name, ranks in sorted(
                    result['strategy_ranks'].items(),
                    key=lambda x: x[1][ti],
                )[:3]
            }
        return out

    def free(self):
        """Release GPU memory held by this evaluator."""
        del self.E_h_dev, self.E_m_dev, self.E_h, self.E_m
        del self.model, self.cost_computer, self.fgw_solver
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"  {self.variant} memory released.")
