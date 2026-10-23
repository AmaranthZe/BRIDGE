"""Published-method baselines for cross-species gene mapping.

Implements proxy versions of recently published methods for comparison:

1. HGT-standard (Yun et al., WWW 2019):
   Standard Heterogeneous Graph Transformer applied to our annotation graph.
   Uses cross-species HGT embedding cosine similarity ONLY (no FGW, no ensemble).
   Represents any paper that applies vanilla HGT for cross-species mapping.

2. HGT+CrossSpec (CAME-inspired, Liu et al., Genome Research 2023):
   HGT trained with cross-species 2-hop gene-gene edges (similar to CAME's
   homologous gene-gene edges in the heterogeneous graph). Uses cosine similarity.
   Corresponds to our v3 architecture but without the RRF ensemble.

3. FGW-OT (Optimal Transport only, Vayer et al. 2019 / Flamary et al. 2021):
   Fused Gromov-Wasserstein transport without any post-processing ensemble.
   The raw FGW output represents optimal-transport-based cross-species alignment.

4. KGE-SVD (knowledge graph embedding proxy):
   SVD decomposition of the joint annotation matrix — equivalent to matrix
   factorization KGE methods (TransE/DistMult on annotation graphs).
   Used as proxy for GenePlexusZoo-style annotation embedding methods.

These are run on Bench1 (primary) and supplementary benchmarks.
"""

import gc
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from benchmarks.model_variants import SharedData, ModelEvaluator, arr_to_rank
from benchmarks.metrics import compute_all_metrics
from benchmarks.bench1_literature import BENCH1_CASES
from benchmarks.bench2_phenotype import (
    parse_hmd_human_phenotype, build_ground_truth, compute_go_jaccard,
    parse_mgi_genepheno,
)
from benchmarks.bench3_essentiality import parse_depmap, parse_impc_viability
from benchmarks.bench5_hmd_ranking import build_hmd_pairs

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy-only ranking (HGT cosine, FGW, SVD)
# ─────────────────────────────────────────────────────────────────────────────

def get_hgt_cos_rank(ev: ModelEvaluator, hi: int) -> np.ndarray:
    """Compute HGT cosine similarity ranking (standard HGT approach)."""
    return ev._compute_hgt_cosine_rank(hi)


def get_fgw_rank(ev: ModelEvaluator, hi: int) -> np.ndarray:
    """Compute FGW optimal transport ranking only."""
    return ev._compute_fgw_rank(hi)


def get_svd_rank(sd: SharedData, hi: int) -> np.ndarray:
    """Compute SVD annotation cosine ranking (KGE proxy)."""
    import torch
    sim = (sd.Z_h_uni[hi] @ sd.Z_m_uni.T).numpy()
    return arr_to_rank(sim, sd.N_m)


# ─────────────────────────────────────────────────────────────────────────────
# Bench1 evaluation for a single ranking function
# ─────────────────────────────────────────────────────────────────────────────

def eval_bench1(rank_fn, sd: SharedData) -> dict:
    """Evaluate a ranking function on bench1 literature cases."""
    N_m = sd.N_m
    ranks = []
    for case in BENCH1_CASES:
        h_eid = case.get("h") or case.get("human_ensg", "")
        m_eid = case.get("t") or case.get("mouse_ensg", "")
        if h_eid not in sd.h_genes or m_eid not in sd.m_genes:
            continue
        hi = sd.h_genes[h_eid]
        mi = sd.m_genes[m_eid]
        rank_arr = rank_fn(hi)
        ranks.append(int(rank_arr[mi]))
    return compute_all_metrics(ranks)


# ─────────────────────────────────────────────────────────────────────────────
# Main: compute all published baselines
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading shared data...")
    sd = SharedData(skip_ppi=True)
    logger.info("Shared data loaded.")

    results = {}

    # ── Method 1: HGT-standard (v2 model, cosine only) ──
    logger.info("\n[1] HGT-standard (v2, Yun et al. 2019 HGT)")
    ev_m1 = ModelEvaluator("M1", sd, device="cpu")
    b1_hgt_v2 = eval_bench1(lambda hi: get_hgt_cos_rank(ev_m1, hi), sd)
    logger.info(f"  Bench1: R@5={b1_hgt_v2['recall@5']:.1%} R@10={b1_hgt_v2['recall@10']:.1%} MRR={b1_hgt_v2['mrr']:.4f}")
    results["HGT_standard_v2"] = {"bench1": b1_hgt_v2}
    ev_m1.free(); gc.collect()

    # ── Method 2: HGT+CrossSpec (v3 model, cosine only, CAME-inspired) ──
    logger.info("\n[2] HGT+CrossSpec (v3, CAME-inspired, Liu et al. 2023)")
    ev_m2 = ModelEvaluator("M2", sd, device="cpu")
    b1_hgt_v3 = eval_bench1(lambda hi: get_hgt_cos_rank(ev_m2, hi), sd)
    logger.info(f"  Bench1: R@5={b1_hgt_v3['recall@5']:.1%} R@10={b1_hgt_v3['recall@10']:.1%} MRR={b1_hgt_v3['mrr']:.4f}")
    results["HGT_crossspec_v3"] = {"bench1": b1_hgt_v3}
    ev_m2.free(); gc.collect()

    # ── Method 3: FGW-only (v2 model, Optimal Transport only) ──
    logger.info("\n[3] FGW-OT only (v2, optimal transport without ensemble)")
    ev_m3 = ModelEvaluator("M3", sd, device="cpu")
    b1_fgw = eval_bench1(lambda hi: get_fgw_rank(ev_m3, hi), sd)
    logger.info(f"  Bench1: R@5={b1_fgw['recall@5']:.1%} R@10={b1_fgw['recall@10']:.1%} MRR={b1_fgw['mrr']:.4f}")
    results["FGW_OT_only"] = {"bench1": b1_fgw}
    ev_m3.free(); gc.collect()

    # ── Method 4: SVD Annotation (KGE proxy / GenePlexusZoo-inspired) ──
    logger.info("\n[4] KGE-SVD (annotation embedding, GenePlexusZoo-inspired)")
    b1_svd = eval_bench1(lambda hi: get_svd_rank(sd, hi), sd)
    logger.info(f"  Bench1: R@5={b1_svd['recall@5']:.1%} R@10={b1_svd['recall@10']:.1%} MRR={b1_svd['mrr']:.4f}")
    results["KGE_SVD"] = {"bench1": b1_svd}

    # Save
    out_path = output_dir / "published_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    # ── Summary ──
    logger.info("\n" + "=" * 80)
    logger.info("BENCH1 COMPARISON: Published Method Proxies vs M4")
    logger.info("=" * 80)
    logger.info(f"{'Method':<30} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'MRR':>8}")
    logger.info("-" * 60)
    for name, m in results.items():
        b = m["bench1"]
        logger.info(f"{name:<30} {b['recall@5']:7.1%} {b['recall@10']:7.1%} {b['recall@20']:7.1%} {b['mrr']:8.4f}")

    # Load M3/M4 for comparison
    with open(output_dir / "bench1_M3.json") as f:
        m3 = json.load(f)["summary"]["primary_metrics"]
    with open(output_dir / "bench1_M4.json") as f:
        m4 = json.load(f)["summary"]["primary_metrics"]
    logger.info("-" * 60)
    logger.info(f"{'M3 (v2 full ensemble)':<30} {m3['recall@5']:7.1%} {m3['recall@10']:7.1%} {m3['recall@20']:7.1%} {m3['mrr']:8.4f}")
    logger.info(f"{'M4 (v3 full ensemble - OURS)':<30} {m4['recall@5']:7.1%} {m4['recall@10']:7.1%} {m4['recall@20']:7.1%} {m4['mrr']:8.4f}")


if __name__ == "__main__":
    main()
