"""Bench5 Non-Orthologous Subset Analysis.

Filters bench5 (genome-wide HMD) pairs further to exclude known Ensembl
orthologs, evaluating only truly non-orthologous functional correspondences.

Note: bench5 already excludes same-name orthologs. This script also removes
pairs that are known Ensembl orthologs (different symbols but still orthologs).
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

from benchmarks.model_variants import SharedData, ModelEvaluator
from benchmarks.metrics import compute_all_metrics
from benchmarks.bench6_comparison import (
    load_ensembl_orthologs,
    compute_ensembl_rank,
    compute_svd_best_rank,
    compute_2hop_best_rank,
    compute_orthofinder_proxy_rank,
)
from benchmarks.bench2_phenotype import parse_hmd_human_phenotype

logger = logging.getLogger(__name__)


def run_bench5_nonortholog():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    sd = SharedData(skip_ppi=True)
    root = sd.root

    # Load Ensembl orthologs
    orthologs = load_ensembl_orthologs(
        str(root / "dataset" / "external" / "ensembl_human_mouse_orthologs.tsv"),
        sd.h_genes, sd.m_genes,
    )
    orth_pairs = set()
    for hi, mi_list in orthologs.items():
        for mi in mi_list:
            orth_pairs.add((hi, mi))

    # Build HMD pairs (excluding same-name, then further filter Ensembl orthologs)
    hmd_records = parse_hmd_human_phenotype(
        str(root / "dataset" / "HMD_HumanPhenotype.rpt")
    )
    pairs = []
    seen = set()
    for h_sym, m_sym, mgi_id, mp_terms in hmd_records:
        if h_sym.lower() == m_sym.lower():
            continue
        ensg = sd.sym_to_ensg.get(h_sym)
        ensmusg = sd.sym_to_ensmusg.get(m_sym)
        if ensg is None or ensmusg is None:
            continue
        if ensg not in sd.h_genes or ensmusg not in sd.m_genes:
            continue
        hi, mi = sd.h_genes[ensg], sd.m_genes[ensmusg]
        pk = (hi, mi)
        if pk in seen:
            continue
        seen.add(pk)
        pairs.append({"hi": hi, "mi": mi, "h_sym": h_sym, "m_sym": m_sym,
                      "is_ortholog": pk in orth_pairs})

    # Sample same 300 genes as bench5 model runs
    rng = np.random.RandomState(42)
    all_his = list(set(p["hi"] for p in pairs))
    sampled_his = set(rng.choice(all_his, size=min(300, len(all_his)), replace=False))
    pairs = [p for p in pairs if p["hi"] in sampled_his]

    orth_pairs_sub = [(p["hi"], p["mi"]) for p in pairs if p["is_ortholog"]]
    nonorth_pairs = [(p["hi"], p["mi"]) for p in pairs if not p["is_ortholog"]]

    logger.info(f"Bench5 sampled pairs: {len(pairs)}")
    logger.info(f"  Known orthologs: {len(orth_pairs_sub)} ({100*len(orth_pairs_sub)/len(pairs):.1f}%)")
    logger.info(f"  Non-orthologs:   {len(nonorth_pairs)} ({100*len(nonorth_pairs)/len(pairs):.1f}%)")

    hi_to_all = defaultdict(list)
    hi_to_orth = defaultdict(list)
    hi_to_nonorth = defaultdict(list)
    for p in pairs:
        hi_to_all[p["hi"]].append(p["mi"])
        if p["is_ortholog"]:
            hi_to_orth[p["hi"]].append(p["mi"])
        else:
            hi_to_nonorth[p["hi"]].append(p["mi"])

    def eval_fn(rank_fn):
        all_r, orth_r, nonorth_r = [], [], []
        for hi in sampled_his:
            r = rank_fn(hi)
            for mi in hi_to_all[hi]:
                all_r.append(int(r[mi]))
            for mi in hi_to_orth[hi]:
                orth_r.append(int(r[mi]))
            for mi in hi_to_nonorth[hi]:
                nonorth_r.append(int(r[mi]))
        return {
            "all": compute_all_metrics(all_r),
            "orthologous": compute_all_metrics(orth_r) if orth_r else {},
            "non_orthologous": compute_all_metrics(nonorth_r) if nonorth_r else {},
        }

    results = {}

    # External baselines
    for name, fn in [
        ("Ensembl_Compara", lambda hi: compute_ensembl_rank(hi, orthologs, sd.N_m)),
        ("SVD_Annotation", lambda hi: compute_svd_best_rank(hi, sd)),
        ("2hop_Overlap", lambda hi: compute_2hop_best_rank(hi, sd)),
        ("OrthoFinder_proxy", lambda hi: compute_orthofinder_proxy_rank(
            hi, orthologs, compute_svd_best_rank(hi, sd), sd.N_m)),
    ]:
        logger.info(f"Evaluating {name}...")
        results[name] = eval_fn(fn)

    # Model variants
    for variant in ["M1", "M2", "M3", "M4"]:
        logger.info(f"Evaluating {variant}...")
        ev = ModelEvaluator(variant, sd, device="cpu")
        def model_fn(hi, ev=ev):
            r = ev.rank_mouse_genes(hi)
            return r["rrf_rank"] if r["rrf_rank"] is not None else r["fgw_rank"]
        results[variant] = eval_fn(model_fn)
        ev.free()
        gc.collect()

    # Display
    logger.info("\n" + "=" * 90)
    logger.info("BENCH5 NON-ORTHOLOG SUBSET")
    logger.info(f"  Non-ortholog pairs: {len(nonorth_pairs)}")
    logger.info("=" * 90)
    logger.info(f"{'Method':<22} {'ALL R@5':>8} {'NON-ORTH R@5':>13} {'NON-ORTH R@10':>14} {'NON-ORTH R@20':>14} {'MRR':>8}")
    logger.info("-" * 82)
    for name in ["Ensembl_Compara", "OrthoFinder_proxy", "SVD_Annotation", "2hop_Overlap", "M1", "M2", "M3", "M4"]:
        m = results[name]
        ma = m["all"]
        mn = m["non_orthologous"]
        prefix = ">>>" if name == "M4" else "   "
        logger.info(
            f"{prefix}{name:<19} "
            f"{ma.get('recall@5',0):8.1%} "
            f"{mn.get('recall@5',0):13.1%} "
            f"{mn.get('recall@10',0):14.1%} "
            f"{mn.get('recall@20',0):14.1%} "
            f"{mn.get('mrr',0):8.4f}"
        )

    out_path = output_dir / "bench5_nonortholog_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_total": len(pairs),
            "n_non_orthologous": len(nonorth_pairs),
            "n_orthologous": len(orth_pairs_sub),
            "results": results,
        }, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_bench5_nonortholog()
