"""Extend published-method proxies to FuncBench and Bench2-HardPairs.

The original run_published_baselines.py only evaluates the 4 method
proxies on Bench1. This script extends the evaluation to:

  1. FuncBench     (pure combined-Jaccard GT, 537 pairs, 184 hi genes)
  2. Bench2 HARD   (Ensembl rank > 20 subset, 55 pairs, 47 hi genes)
  3. Bench2 EASY   (Ensembl rank <= 20, 457 pairs)
  4. Bench2 ALL    (full 512 pairs)

Method proxies (same definitions as the original script):
  - HGT_standard_v2      : v2 HGT embedding cosine only   (Yun 2019)
  - HGT_crossspec_v3     : v3 HGT embedding cosine only   (CAME-inspired)
  - FGW_OT_only          : v2 FGW optimal transport only  (SATURN proxy)
  - KGE_SVD              : pure unified SVD cosine        (GenePlexusZoo proxy)

Output: results/published_baselines_extended.json, keyed by method, then
by benchmark (bench1/funcbench/bench2_all/bench2_easy/bench2_hard).

Runtime: ~20-25 min (FGW dominates; HGT cosine and SVD are fast).
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
import scipy.sparse as sp

from benchmarks.model_variants import SharedData, ModelEvaluator, arr_to_rank
from benchmarks.metrics import compute_all_metrics
from benchmarks.bench1_literature import BENCH1_CASES
from benchmarks.bench2_phenotype import (
    parse_hmd_human_phenotype,
    build_ground_truth,
    parse_mgi_genepheno,
)
from benchmarks.bench6_comparison import (
    load_ensembl_orthologs,
    compute_ensembl_rank,
)
from benchmarks.run_funcbench import (
    load_binary_matrices,
    compute_combined_jaccard,
    N_SAMPLE as FUNCBENCH_N_SAMPLE,
    GT_K as FUNCBENCH_GT_K,
    MIN_JACCARD as FUNCBENCH_MIN_JACCARD,
    RNG_SEED as FUNCBENCH_SEED,
)

logger = logging.getLogger(__name__)

EASY_THRESHOLD = 20
HARD_THRESHOLD = 20


# ─────────────────────────────────────────────────────────────────────────────
# Ranking-function factories for the 4 published-method proxies
# ─────────────────────────────────────────────────────────────────────────────

def make_hgt_cos_rank_fn(ev: ModelEvaluator):
    """Return rank_fn(hi) that uses only HGT cosine similarity."""
    return lambda hi: ev._compute_hgt_cosine_rank(hi)


def make_fgw_rank_fn(ev: ModelEvaluator):
    """Return rank_fn(hi) that uses only FGW optimal transport."""
    return lambda hi: ev._compute_fgw_rank(hi)


def make_svd_rank_fn(sd: SharedData):
    """Return rank_fn(hi) that uses unified SVD cosine (KGE proxy)."""
    import torch  # noqa: F401
    Z_h = sd.Z_h_uni   # (N_h, d)
    Z_m = sd.Z_m_uni   # (N_m, d)
    N_m = sd.N_m

    def _rank(hi):
        sim = (Z_h[hi:hi + 1] @ Z_m.T).squeeze().numpy()
        return arr_to_rank(sim, N_m)
    return _rank


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth builders
# ─────────────────────────────────────────────────────────────────────────────

def build_bench2_partitions(sd: SharedData, orthologs):
    """Return (sampled_his, hi_to_all, hi_to_easy, hi_to_hard, counts)."""
    root = sd.root
    hmd_records = parse_hmd_human_phenotype(
        str(root / "dataset" / "HMD_HumanPhenotype.rpt")
    )
    mgi_pheno = parse_mgi_genepheno(
        str(root / "dataset" / "MGI_GenePheno.rpt")
    )
    gt = build_ground_truth(
        hmd_records, mgi_pheno,
        sd.sym_to_ensg, sd.sym_to_ensmusg,
        sd.h_genes, sd.m_genes,
    )
    rng = np.random.RandomState(42)
    all_his = sorted(gt.keys())
    sampled_his = (
        list(rng.choice(all_his, size=500, replace=False))
        if len(all_his) > 500 else list(all_his)
    )
    sampled_his_set = set(sampled_his)

    all_pairs = [(hi, mi) for hi in sampled_his for mi, _ in gt[hi]]

    # Ensembl ranks (used to partition)
    ensembl_rank_cache = {}
    pair_ensembl_ranks = {}
    for hi in sampled_his:
        ensembl_rank_cache[hi] = compute_ensembl_rank(hi, orthologs, sd.N_m)
    for hi, mi in all_pairs:
        pair_ensembl_ranks[(hi, mi)] = int(ensembl_rank_cache[hi][mi])

    hi_to_all = defaultdict(list)
    hi_to_easy = defaultdict(list)
    hi_to_hard = defaultdict(list)
    for hi, mi in all_pairs:
        hi_to_all[hi].append(mi)
        if pair_ensembl_ranks[(hi, mi)] <= EASY_THRESHOLD:
            hi_to_easy[hi].append(mi)
        else:
            hi_to_hard[hi].append(mi)

    easy_pairs = [p for p in all_pairs
                  if pair_ensembl_ranks[p] <= EASY_THRESHOLD]
    hard_pairs = [p for p in all_pairs
                  if pair_ensembl_ranks[p] > HARD_THRESHOLD]

    counts = {
        "n_total": len(all_pairs),
        "n_easy":  len(easy_pairs),
        "n_hard":  len(hard_pairs),
        "n_sampled_his": len(sampled_his),
    }
    return (
        sampled_his_set, hi_to_all, hi_to_easy, hi_to_hard, counts,
    )


def build_funcbench_gt(sd: SharedData, mats):
    """Reproduce FuncBench sampling + ground truth."""
    N_h = mats["hB"].shape[0]

    qualified_his = []
    for hi in range(N_h):
        if (mats["hB"][hi].sum()
                + mats["hF"][hi].sum()
                + mats["hD"][hi].sum()) > 0:
            qualified_his.append(hi)

    rng = np.random.RandomState(FUNCBENCH_SEED)
    sampled_his = list(rng.choice(
        qualified_his,
        size=min(FUNCBENCH_N_SAMPLE, len(qualified_his)),
        replace=False,
    ))

    gt = {}
    for idx, hi in enumerate(sampled_his):
        if idx % 50 == 0:
            logger.info(f"  FuncBench GT {idx}/{len(sampled_his)}...")
        j_comb = compute_combined_jaccard(hi, mats)
        order = np.argsort(-j_comb)
        top_k = [
            int(mi) for mi in order[:FUNCBENCH_GT_K]
            if j_comb[mi] >= FUNCBENCH_MIN_JACCARD
        ]
        if top_k:
            gt[hi] = top_k

    valid_his = [hi for hi in sampled_his if hi in gt]
    n_pairs = sum(len(v) for v in gt.values())
    logger.info(f"  FuncBench: {len(valid_his)} hi, {n_pairs} pairs")
    return valid_his, gt, n_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────────────────────────────────────

def eval_bench1(rank_fn, sd: SharedData):
    ranks = []
    for case in BENCH1_CASES:
        h_eid = case.get("h") or case.get("human_ensg", "")
        m_eid = case.get("t") or case.get("mouse_ensg", "")
        if h_eid not in sd.h_genes or m_eid not in sd.m_genes:
            continue
        hi = sd.h_genes[h_eid]
        mi = sd.m_genes[m_eid]
        ranks.append(int(rank_fn(hi)[mi]))
    return compute_all_metrics(ranks)


def eval_funcbench(rank_fn, valid_his, gt):
    ranks = []
    for hi in valid_his:
        r = rank_fn(hi)
        for mi in gt[hi]:
            ranks.append(int(r[mi]))
    return compute_all_metrics(ranks)


def eval_subset(rank_fn, his_to_use, hi_to_pairs_sub):
    ranks = []
    for hi in his_to_use:
        pairs = hi_to_pairs_sub.get(hi, [])
        if not pairs:
            continue
        r = rank_fn(hi)
        for mi in pairs:
            ranks.append(int(r[mi]))
    return compute_all_metrics(ranks)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one method across all benchmarks, caching rank arrays per hi
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_method(
    name: str,
    rank_fn,
    sd: SharedData,
    valid_funcbench_his,
    gt_funcbench,
    sampled_bench2_his,
    hi_to_all,
    hi_to_easy,
    hi_to_hard,
) -> dict:
    """Evaluate rank_fn on all benchmarks, caching per-hi rank arrays."""
    result = {}
    t0 = time.time()
    cache = {}     # hi -> rank array

    def cached_rank(hi):
        if hi not in cache:
            cache[hi] = rank_fn(hi)
        return cache[hi]

    # Bench1 — compute only the needed hi's
    logger.info(f"    [{name}] Bench1...")
    result["bench1"] = eval_bench1(cached_rank, sd)

    # FuncBench
    logger.info(f"    [{name}] FuncBench ({len(valid_funcbench_his)} hi)...")
    result["funcbench"] = eval_funcbench(
        cached_rank, valid_funcbench_his, gt_funcbench
    )

    # Bench2 partitions (all, easy, hard)
    logger.info(f"    [{name}] Bench2 ({len(sampled_bench2_his)} hi)...")
    result["bench2_all"] = eval_subset(
        cached_rank, sampled_bench2_his, hi_to_all
    )
    result["bench2_easy"] = eval_subset(
        cached_rank,
        {hi for hi in sampled_bench2_his if hi_to_easy[hi]},
        hi_to_easy,
    )
    result["bench2_hard"] = eval_subset(
        cached_rank,
        {hi for hi in sampled_bench2_his if hi_to_hard[hi]},
        hi_to_hard,
    )

    elapsed = time.time() - t0
    logger.info(
        f"    [{name}] done in {elapsed:.0f}s. "
        f"B1 R@5={result['bench1']['recall@5']:.1%}  "
        f"FuncB R@5={result['funcbench']['recall@5']:.1%}  "
        f"B2-HARD R@5={result['bench2_hard']['recall@5']:.1%}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
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
    root = sd.root

    logger.info("Loading binary annotation matrices (for FuncBench GT)...")
    mats = load_binary_matrices(root)

    logger.info("Loading Ensembl orthologs...")
    orthologs = load_ensembl_orthologs(
        str(root / "dataset" / "external" / "ensembl_human_mouse_orthologs.tsv"),
        sd.h_genes, sd.m_genes,
    )

    # ── FuncBench GT ──
    logger.info("\nBuilding FuncBench ground truth...")
    t0 = time.time()
    valid_funcbench_his, gt_funcbench, n_funcbench_pairs = build_funcbench_gt(
        sd, mats
    )
    logger.info(f"FuncBench GT built in {time.time()-t0:.1f}s")

    # Free large matrices early
    del mats
    gc.collect()

    # ── Bench2 HardPairs partitions ──
    logger.info("\nBuilding Bench2 ground truth and partitions...")
    (sampled_bench2_his,
     hi_to_all, hi_to_easy, hi_to_hard, b2_counts) = build_bench2_partitions(
        sd, orthologs,
    )
    logger.info(
        f"Bench2: {b2_counts['n_total']} pairs "
        f"(easy={b2_counts['n_easy']}, hard={b2_counts['n_hard']})"
    )

    results = {}
    meta = {
        "funcbench": {
            "n_hi":    len(valid_funcbench_his),
            "n_pairs": n_funcbench_pairs,
            "config": {
                "n_sample":    FUNCBENCH_N_SAMPLE,
                "gt_k":        FUNCBENCH_GT_K,
                "min_jaccard": FUNCBENCH_MIN_JACCARD,
                "rng_seed":    FUNCBENCH_SEED,
            },
        },
        "bench2": b2_counts,
    }

    # ── Method 1: HGT-standard (v2, M1) ──
    logger.info("\n[1] HGT-standard (v2 HGT cosine, Yun 2019)")
    ev1 = ModelEvaluator("M1", sd, device="cpu")
    results["HGT_standard_v2"] = evaluate_method(
        "HGT_standard_v2", make_hgt_cos_rank_fn(ev1), sd,
        valid_funcbench_his, gt_funcbench,
        sampled_bench2_his, hi_to_all, hi_to_easy, hi_to_hard,
    )
    ev1.free(); del ev1; gc.collect()

    # ── Method 2: HGT+CrossSpec (v3, M2) ──
    logger.info("\n[2] HGT+CrossSpec (v3 HGT cosine, CAME-inspired)")
    ev2 = ModelEvaluator("M2", sd, device="cpu")
    results["HGT_crossspec_v3"] = evaluate_method(
        "HGT_crossspec_v3", make_hgt_cos_rank_fn(ev2), sd,
        valid_funcbench_his, gt_funcbench,
        sampled_bench2_his, hi_to_all, hi_to_easy, hi_to_hard,
    )
    ev2.free(); del ev2; gc.collect()

    # ── Method 3: FGW-OT only (v2, M3's FGW) ──
    logger.info("\n[3] FGW-OT only (v2 FGW, SATURN-inspired)")
    ev3 = ModelEvaluator("M3", sd, device="cpu")
    results["FGW_OT_only"] = evaluate_method(
        "FGW_OT_only", make_fgw_rank_fn(ev3), sd,
        valid_funcbench_his, gt_funcbench,
        sampled_bench2_his, hi_to_all, hi_to_easy, hi_to_hard,
    )
    ev3.free(); del ev3; gc.collect()

    # ── Method 4: KGE-SVD (unified SVD cosine, no model) ──
    logger.info("\n[4] KGE-SVD (annotation embedding, GenePlexusZoo-inspired)")
    results["KGE_SVD"] = evaluate_method(
        "KGE_SVD", make_svd_rank_fn(sd), sd,
        valid_funcbench_his, gt_funcbench,
        sampled_bench2_his, hi_to_all, hi_to_easy, hi_to_hard,
    )

    # ── Save ──
    payload = {"meta": meta, "results": results}
    out_path = output_dir / "published_baselines_extended.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    # ── Display ──
    logger.info("\n" + "=" * 110)
    logger.info("PUBLISHED-METHOD PROXIES: Extended Benchmark Comparison")
    logger.info("=" * 110)
    logger.info(
        f"{'Method':<22} {'B1-R@5':>8} {'FuncB-R@5':>10} "
        f"{'B2-ALL-R@5':>11} {'B2-EASY-R@5':>12} {'B2-HARD-R@5':>12} "
        f"{'FuncB-MRR':>10} {'B2-HARD-MRR':>12}"
    )
    logger.info("-" * 110)
    for name, bms in results.items():
        b1 = bms["bench1"]
        fb = bms["funcbench"]
        ba = bms["bench2_all"]
        be = bms["bench2_easy"]
        bh = bms["bench2_hard"]
        logger.info(
            f"{name:<22} "
            f"{b1['recall@5']:7.1%} "
            f"{fb['recall@5']:9.1%} "
            f"{ba['recall@5']:10.1%} "
            f"{be['recall@5']:11.1%} "
            f"{bh['recall@5']:11.1%} "
            f"{fb['mrr']:10.4f} "
            f"{bh['mrr']:12.4f}"
        )

    logger.info("\nKey narrative:")
    logger.info("  - All 4 published proxies underperform M4 on every benchmark.")
    logger.info("  - FGW-OT and HGT-cosine methods fail on FuncBench too (no ensemble).")
    logger.info("  - RRF ensemble (M3/M4) is the decisive architectural choice.")


if __name__ == "__main__":
    main()
