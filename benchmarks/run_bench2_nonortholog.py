"""Bench2 Non-Orthologous Subset Analysis.

Filters bench2 (MGI Phenotype) pairs to exclude known Ensembl orthologs,
then re-evaluates all methods on this truly non-orthologous subset.

This is the scientifically cleanest comparison:
- Known orthologs are excluded (sequence methods have no advantage)
- Remaining pairs represent non-trivial functional correspondences
- M4 should outperform all external baselines on this subset

Methods evaluated:
  External: Ensembl_Compara, OrthoFinder_proxy, SVD_Annotation, 2hop_Overlap
  Model variants: M1, M2, M3, M4 (using saved result files + re-evaluation)
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


def build_ensembl_ortholog_set(orthologs_dict: dict) -> set:
    """Build a set of known (hi, mi) ortholog pairs from Ensembl data."""
    orth_pairs = set()
    for hi, mi_list in orthologs_dict.items():
        for mi in mi_list:
            orth_pairs.add((hi, mi))
    return orth_pairs


def run_bench2_nonortholog():
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

    # Load Ensembl orthologs
    logger.info("Loading Ensembl orthologs...")
    orthologs = load_ensembl_orthologs(
        str(root / "dataset" / "external" / "ensembl_human_mouse_orthologs.tsv"),
        sd.h_genes, sd.m_genes,
    )
    orth_pairs = build_ensembl_ortholog_set(orthologs)
    logger.info(f"  Known Ensembl ortholog pairs: {len(orth_pairs):,}")

    # Build bench2 ground truth pairs (same sampling as model runs)
    logger.info("Building bench2 ground truth pairs...")
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

    # Sample same 500 human genes as model runs
    rng = np.random.RandomState(42)
    all_his = sorted(gt.keys())
    if 500 < len(all_his):
        sampled_his = set(rng.choice(all_his, size=500, replace=False))
    else:
        sampled_his = set(all_his)

    # Build flat pair list for sampled genes
    all_pairs = []
    for hi in sampled_his:
        for mi, mp_terms in gt[hi]:
            all_pairs.append((hi, mi))

    # Separate: orthologous vs non-orthologous
    orth_subset = [(hi, mi) for hi, mi in all_pairs if (hi, mi) in orth_pairs]
    nonorth_subset = [(hi, mi) for hi, mi in all_pairs if (hi, mi) not in orth_pairs]

    logger.info(f"  Total pairs: {len(all_pairs)}")
    logger.info(f"  Known Ensembl orthologs: {len(orth_subset)} ({100*len(orth_subset)/len(all_pairs):.1f}%)")
    logger.info(f"  Non-orthologous pairs:   {len(nonorth_subset)} ({100*len(nonorth_subset)/len(all_pairs):.1f}%)")

    # Group by hi for efficiency
    hi_to_pairs_all = defaultdict(list)
    for hi, mi in all_pairs:
        hi_to_pairs_all[hi].append(mi)

    hi_to_nonorth = defaultdict(list)
    for hi, mi in nonorth_subset:
        hi_to_nonorth[hi].append(mi)

    hi_to_orth = defaultdict(list)
    for hi, mi in orth_subset:
        hi_to_orth[hi].append(mi)

    # ─────────────────────────────────────────────────────────────────────
    # Evaluate each method
    # ─────────────────────────────────────────────────────────────────────

    def eval_subsets(rank_fn_or_arr, label=""):
        """Evaluate a ranking on all/orth/nonorth subsets."""
        all_r, orth_r, nonorth_r = [], [], []
        for hi in sampled_his:
            rank_arr = rank_fn_or_arr(hi)
            for mi in hi_to_pairs_all[hi]:
                all_r.append(int(rank_arr[mi]))
            for mi in hi_to_orth[hi]:
                orth_r.append(int(rank_arr[mi]))
            for mi in hi_to_nonorth[hi]:
                nonorth_r.append(int(rank_arr[mi]))
        return {
            "all": compute_all_metrics(all_r),
            "orthologous": compute_all_metrics(orth_r) if orth_r else {},
            "non_orthologous": compute_all_metrics(nonorth_r) if nonorth_r else {},
        }

    results = {}
    t0 = time.time()

    # External baselines
    logger.info("\nEvaluating external baselines...")

    def ens_fn(hi):
        return compute_ensembl_rank(hi, orthologs, sd.N_m)
    results["Ensembl_Compara"] = eval_subsets(ens_fn, "Ensembl_Compara")

    def svd_fn(hi):
        return compute_svd_best_rank(hi, sd)
    results["SVD_Annotation"] = eval_subsets(svd_fn, "SVD_Annotation")

    def hop_fn(hi):
        return compute_2hop_best_rank(hi, sd)
    results["2hop_Overlap"] = eval_subsets(hop_fn, "2hop_Overlap")

    def orth_fn(hi):
        svd = compute_svd_best_rank(hi, sd)
        return compute_orthofinder_proxy_rank(hi, orthologs, svd, sd.N_m)
    results["OrthoFinder_proxy"] = eval_subsets(orth_fn, "OrthoFinder_proxy")

    logger.info(f"  External baselines done in {time.time()-t0:.1f}s")

    # Model variants
    for variant in ["M1", "M2", "M3", "M4"]:
        logger.info(f"\nEvaluating {variant}...")
        ev = ModelEvaluator(variant, sd, device="cpu")

        def model_fn(hi, ev=ev, variant=variant):
            r = ev.rank_mouse_genes(hi)
            return r["rrf_rank"] if r["rrf_rank"] is not None else r["fgw_rank"]

        results[variant] = eval_subsets(model_fn, variant)
        ev.free()
        gc.collect()

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # Display results
    # ─────────────────────────────────────────────────────────────────────
    def fmt(m, key):
        if not m:
            return "   n/a"
        return f"{m.get(key, 0):6.1%}"

    logger.info("\n" + "=" * 100)
    logger.info("BENCH2 STRATIFIED BY ORTHOLOG STATUS")
    logger.info(f"  Total: {len(all_pairs)} pairs | "
                f"Orthologs: {len(orth_subset)} | Non-orthologs: {len(nonorth_subset)}")
    logger.info("=" * 100)

    header = f"{'Method':<25} {'ALL R@5':>8} {'ALL R@10':>9} {'ORTH R@5':>9} {'ORTH R@10':>10} {'NON-ORTH R@5':>13} {'NON-ORTH R@10':>14} {'NON-ORTH MRR':>13}"
    logger.info(header)
    logger.info("-" * 105)

    order = ["Ensembl_Compara", "OrthoFinder_proxy", "SVD_Annotation", "2hop_Overlap",
             "M1", "M2", "M3", "M4"]
    for name in order:
        m = results[name]
        ma = m["all"]
        mo = m["orthologous"]
        mn = m["non_orthologous"]
        prefix = ">>>" if name == "M4" else "   "
        logger.info(
            f"{prefix}{name:<22} "
            f"{fmt(ma,'recall@5'):>8} {fmt(ma,'recall@10'):>9} "
            f"{fmt(mo,'recall@5'):>9} {fmt(mo,'recall@10'):>10} "
            f"{fmt(mn,'recall@5'):>13} {fmt(mn,'recall@10'):>14} "
            f"{mn.get('mrr',0):13.4f}"
        )

    # Save
    out_path = output_dir / "bench2_nonortholog_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_total": len(all_pairs),
            "n_orthologous": len(orth_subset),
            "n_non_orthologous": len(nonorth_subset),
            "results": results,
        }, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    # Key summary
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDING: Non-Orthologous Pair Performance")
    logger.info("=" * 70)
    for name in order:
        mn = results[name]["non_orthologous"]
        if mn:
            logger.info(
                f"  {name:<25} NON-ORTH R@5={mn.get('recall@5',0):.1%} "
                f"R@10={mn.get('recall@10',0):.1%} "
                f"R@20={mn.get('recall@20',0):.1%} "
                f"MRR={mn.get('mrr',0):.4f}"
            )


if __name__ == "__main__":
    run_bench2_nonortholog()
