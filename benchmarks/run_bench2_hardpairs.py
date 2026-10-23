"""Option A: Ensembl-Fail ("Hard Pairs") Analysis on Bench2.

Partitions bench2 pairs into:
  - Easy: Ensembl Compara rank <= THRESHOLD (sequence methods already solve it)
  - Hard: Ensembl Compara rank >  THRESHOLD (sequence methods fail)

On HARD pairs, evaluates all methods: Ensembl, SVD, 2-hop, OrthoFinder, M1-M4.
This demonstrates that when sequence orthology fails, our model is the best.

Key expected result: on HARD pairs, M4 >> Ensembl/OrthoFinder while SVD/2-hop
also improve relatively; M4 should rank first among annotation-only methods.
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
from benchmarks.bench6_comparison import (
    load_ensembl_orthologs,
    compute_ensembl_rank,
    compute_svd_best_rank,
    compute_2hop_best_rank,
    compute_orthofinder_proxy_rank,
)
from benchmarks.bench2_phenotype import (
    parse_hmd_human_phenotype,
    build_ground_truth,
    parse_mgi_genepheno,
)

logger = logging.getLogger(__name__)

EASY_THRESHOLD = 20   # Ensembl rank <= 20 → "easy"
HARD_THRESHOLD = 20   # Ensembl rank >  20 → "hard"


def run_bench2_hardpairs():
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

    # ── Ensembl orthologs ──
    logger.info("Loading Ensembl orthologs...")
    orthologs = load_ensembl_orthologs(
        str(root / "dataset" / "external" / "ensembl_human_mouse_orthologs.tsv"),
        sd.h_genes, sd.m_genes,
    )

    # ── Bench2 ground truth (same sampling as model runs) ──
    logger.info("Building bench2 ground truth...")
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

    # Reproduce same 500-gene sample
    rng = np.random.RandomState(42)
    all_his = sorted(gt.keys())
    if 500 < len(all_his):
        sampled_his = list(rng.choice(all_his, size=500, replace=False))
    else:
        sampled_his = list(all_his)
    sampled_his_set = set(sampled_his)

    # Flat pair list
    all_pairs = []
    for hi in sampled_his:
        for mi, _ in gt[hi]:
            all_pairs.append((hi, mi))

    logger.info(f"Total bench2 pairs: {len(all_pairs)} from {len(sampled_his)} hi genes")

    # ── Step 1: Compute Ensembl rank for every pair (FAST) ──
    logger.info("Computing Ensembl ranks for all pairs...")
    t0 = time.time()
    ensembl_rank_cache = {}  # hi -> rank_array
    pair_ensembl_ranks = {}  # (hi, mi) -> rank

    for hi in sampled_his:
        if hi not in ensembl_rank_cache:
            ensembl_rank_cache[hi] = compute_ensembl_rank(hi, orthologs, sd.N_m)

    for hi, mi in all_pairs:
        pair_ensembl_ranks[(hi, mi)] = int(ensembl_rank_cache[hi][mi])

    logger.info(f"Ensembl ranks computed in {time.time()-t0:.1f}s")

    # ── Step 2: Partition pairs ──
    easy_pairs = [(hi, mi) for (hi, mi), r in pair_ensembl_ranks.items() if r <= EASY_THRESHOLD]
    hard_pairs = [(hi, mi) for (hi, mi), r in pair_ensembl_ranks.items() if r >  HARD_THRESHOLD]

    logger.info(f"Easy pairs (Ensembl rank <= {EASY_THRESHOLD}): {len(easy_pairs)} "
                f"({100*len(easy_pairs)/len(all_pairs):.1f}%)")
    logger.info(f"Hard pairs (Ensembl rank >  {HARD_THRESHOLD}): {len(hard_pairs)} "
                f"({100*len(hard_pairs)/len(all_pairs):.1f}%)")

    # Group by hi
    hi_to_all   = defaultdict(list)
    hi_to_easy  = defaultdict(list)
    hi_to_hard  = defaultdict(list)
    for hi, mi in all_pairs:
        hi_to_all[hi].append(mi)
        if pair_ensembl_ranks[(hi, mi)] <= EASY_THRESHOLD:
            hi_to_easy[hi].append(mi)
        else:
            hi_to_hard[hi].append(mi)

    hard_his = set(hi for hi, _ in hard_pairs)
    logger.info(f"Distinct hi genes with hard pairs: {len(hard_his)}")

    # ── Evaluation function ──
    def eval_on_subset(rank_fn, his_to_use, hi_to_pairs_subset):
        """Evaluate rank_fn on a specific set of hi genes and their target mi."""
        ranks = []
        for hi in his_to_use:
            if not hi_to_pairs_subset[hi]:
                continue
            rank_arr = rank_fn(hi)
            for mi in hi_to_pairs_subset[hi]:
                ranks.append(int(rank_arr[mi]))
        return compute_all_metrics(ranks)

    results = {}

    # ── External baselines (fast, run on ALL his but measure hard subset) ──
    logger.info("\nComputing external baseline rankings (all his, measure on hard subset)...")
    t0 = time.time()

    # Pre-compute all baseline rank arrays (cache per hi)
    svd_cache = {}
    hop_cache = {}
    orth_cache = {}
    for hi in sampled_his:
        svd_cache[hi] = compute_svd_best_rank(hi, sd)
        hop_cache[hi] = compute_2hop_best_rank(hi, sd)
        orth_cache[hi] = compute_orthofinder_proxy_rank(hi, orthologs, svd_cache[hi], sd.N_m)

    logger.info(f"  Baseline caches built in {time.time()-t0:.1f}s")

    for subset_name, his_to_use, hi_to_pairs_sub in [
        ("all",  sampled_his_set, hi_to_all),
        ("easy", {hi for hi,_ in easy_pairs}, hi_to_easy),
        ("hard", hard_his, hi_to_hard),
    ]:
        for method_name, cache in [
            ("Ensembl_Compara",   ensembl_rank_cache),
            ("SVD_Annotation",    svd_cache),
            ("2hop_Overlap",      hop_cache),
            ("OrthoFinder_proxy", orth_cache),
        ]:
            key = f"{method_name}"
            if key not in results:
                results[key] = {}
            results[key][subset_name] = eval_on_subset(
                lambda hi, c=cache: c[hi], his_to_use, hi_to_pairs_sub
            )

    # ── M1-M4 on HARD pairs only (efficient: only process hard_his) ──
    for variant in ["M1", "M2", "M3", "M4"]:
        logger.info(f"\nEvaluating {variant} on hard pairs ({len(hard_his)} genes)...")
        t1 = time.time()
        ev = ModelEvaluator(variant, sd, device="cpu")

        model_rank_cache = {}
        for hi in hard_his:
            r = ev.rank_mouse_genes(hi)
            model_rank_cache[hi] = r["rrf_rank"] if r["rrf_rank"] is not None else r["fgw_rank"]

        if variant not in results:
            results[variant] = {}

        # Also evaluate on all/easy for completeness (use pre-cached if available)
        # For ALL and EASY pairs: compute for all sampled_his
        logger.info(f"  Hard-pair eval done. Now computing all/easy for {variant}...")
        for hi in sampled_his_set - hard_his:
            r = ev.rank_mouse_genes(hi)
            model_rank_cache[hi] = r["rrf_rank"] if r["rrf_rank"] is not None else r["fgw_rank"]

        for subset_name, his_to_use, hi_to_pairs_sub in [
            ("all",  sampled_his_set, hi_to_all),
            ("easy", {hi for hi,_ in easy_pairs}, hi_to_easy),
            ("hard", hard_his, hi_to_hard),
        ]:
            results[variant][subset_name] = eval_on_subset(
                lambda hi, c=model_rank_cache: c[hi], his_to_use, hi_to_pairs_sub
            )

        logger.info(f"  {variant} done in {time.time()-t1:.1f}s total")
        ev.free()
        gc.collect()

    # ── Display ──
    def fmt(m, key="recall@5"):
        if not m:
            return "  --  "
        return f"{100*m.get(key,0):5.1f}%"

    logger.info("\n" + "="*100)
    logger.info(f"BENCH2 HARD-PAIR ANALYSIS  (Hard = Ensembl rank > {HARD_THRESHOLD})")
    logger.info(f"  Total pairs: {len(all_pairs)}  |  Easy: {len(easy_pairs)}  |  Hard: {len(hard_pairs)}")
    logger.info("="*100)
    logger.info(f"  {'Method':<22} {'ALL R@5':>8} {'EASY R@5':>9} {'HARD R@5':>9} "
                f"{'HARD R@10':>10} {'HARD R@20':>10} {'HARD MRR':>10}")
    logger.info("-"*82)

    order = ["Ensembl_Compara","OrthoFinder_proxy","SVD_Annotation","2hop_Overlap",
             "M1","M2","M3","M4"]
    for name in order:
        m = results.get(name, {})
        star = ">>>" if name == "M4" else "   "
        hard = m.get("hard", {})
        logger.info(
            f"{star}{name:<19} "
            f"{fmt(m.get('all')):>8} "
            f"{fmt(m.get('easy')):>9} "
            f"{fmt(m.get('hard')):>9} "
            f"{fmt(hard,'recall@10'):>10} "
            f"{fmt(hard,'recall@20'):>10} "
            f"{hard.get('mrr',0):10.4f}"
        )

    # ── Save ──
    out_path = output_dir / "bench2_hardpairs_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "hard_threshold": HARD_THRESHOLD,
            "n_total": len(all_pairs),
            "n_easy": len(easy_pairs),
            "n_hard": len(hard_pairs),
            "n_hard_hi_genes": len(hard_his),
            "results": results,
        }, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_bench2_hardpairs()
