"""Run external baselines on ALL benchmarks (bench1-bench5).

External baselines:
1. Ensembl Compara (Sequence Orthology)
2. SVD Annotation Cosine (No Learning)
3. 2-hop Annotation Overlap (Graph Statistics)
4. OrthoFinder-proxy (Ortholog Grouping)

These are applied to bench2 (MGI Phenotype), bench3 (Essentiality),
and bench5 (Genome-wide HMD) in addition to bench1 (already done in bench6).
"""

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats

from benchmarks.metrics import compute_all_metrics
from benchmarks.model_variants import SharedData, arr_to_rank
from benchmarks.bench6_comparison import (
    load_ensembl_orthologs,
    compute_ensembl_rank,
    compute_svd_best_rank,
    compute_2hop_best_rank,
    compute_orthofinder_proxy_rank,
)
from benchmarks.bench2_phenotype import parse_hmd_human_phenotype, compute_go_jaccard
from benchmarks.bench3_essentiality import parse_depmap, parse_impc_viability

logger = logging.getLogger(__name__)

BASELINES = ["Ensembl_Compara", "SVD_Annotation", "2hop_Overlap", "OrthoFinder_proxy"]


def baseline_rank(name, hi, sd, orthologs, svd_cache=None):
    """Compute ranking for one baseline method."""
    N_m = sd.N_m
    if name == "Ensembl_Compara":
        return compute_ensembl_rank(hi, orthologs, N_m)
    elif name == "SVD_Annotation":
        return compute_svd_best_rank(hi, sd)
    elif name == "2hop_Overlap":
        return compute_2hop_best_rank(hi, sd)
    elif name == "OrthoFinder_proxy":
        svd_rank = compute_svd_best_rank(hi, sd) if svd_cache is None else svd_cache
        return compute_orthofinder_proxy_rank(hi, orthologs, svd_rank, N_m)
    else:
        raise ValueError(f"Unknown baseline: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Bench2 External Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_bench2_external(sd, orthologs, output_dir, sample=500):
    """Run external baselines on bench2 (MGI Phenotype) cases."""
    logger.info(f"\n{'=' * 80}")
    logger.info("BENCH2 EXTERNAL: MGI Phenotype Transfer")
    logger.info(f"{'=' * 80}\n")

    root = sd.root
    hmd_records = parse_hmd_human_phenotype(
        str(root / "dataset" / "HMD_HumanPhenotype.rpt")
    )

    # Build ground truth pairs (same logic as bench2)
    pairs = []
    seen = set()
    for h_sym, m_sym, mgi_id, mp_terms in hmd_records:
        ensg = sd.sym_to_ensg.get(h_sym)
        ensmusg = sd.sym_to_ensmusg.get(m_sym)
        if ensg is None or ensmusg is None:
            continue
        if ensg not in sd.h_genes or ensmusg not in sd.m_genes:
            continue
        hi = sd.h_genes[ensg]
        mi = sd.m_genes[ensmusg]
        pk = (hi, mi)
        if pk in seen:
            continue
        seen.add(pk)
        jac = compute_go_jaccard(hi, mi, sd.sparse)
        pairs.append({"h_sym": h_sym, "m_sym": m_sym, "hi": hi, "mi": mi, "jac": jac})

    # Sample by human gene
    if sample is not None:
        rng = np.random.RandomState(42)
        all_his = list(set(p["hi"] for p in pairs))
        if sample < len(all_his):
            sampled = set(rng.choice(all_his, size=sample, replace=False))
            pairs = [p for p in pairs if p["hi"] in sampled]

    logger.info(f"  Evaluating {len(pairs)} pairs...")

    # Group by human gene
    hi_to_pairs = defaultdict(list)
    for p in pairs:
        hi_to_pairs[p["hi"]].append(p)

    baseline_ranks = {b: [] for b in BASELINES}
    t0 = time.time()

    for idx, (hi, hi_pairs) in enumerate(hi_to_pairs.items()):
        svd_rank = compute_svd_best_rank(hi, sd)
        for bname in BASELINES:
            if bname == "OrthoFinder_proxy":
                rank = compute_orthofinder_proxy_rank(hi, orthologs, svd_rank, sd.N_m)
            elif bname == "SVD_Annotation":
                rank = svd_rank
            else:
                rank = baseline_rank(bname, hi, sd, orthologs)
            for p in hi_pairs:
                baseline_ranks[bname].append(int(rank[p["mi"]]))

        if (idx + 1) % 200 == 0 or (idx + 1) == len(hi_to_pairs):
            logger.info(f"  Progress: {idx + 1}/{len(hi_to_pairs)} genes")

    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s")

    # Compute metrics per baseline
    results = {}
    for bname in BASELINES:
        ranks = baseline_ranks[bname]
        if ranks:
            m = compute_all_metrics(ranks)
            results[bname] = m
            logger.info(
                f"  {bname:<25} R@5={m['recall@5']:.1%} R@10={m['recall@10']:.1%} "
                f"R@20={m['recall@20']:.1%} MRR={m['mrr']:.4f} MedR={m['median_rank']:.0f}"
            )

    # Save
    out_path = Path(output_dir) / "bench2_external.json"
    with open(out_path, "w") as f:
        json.dump({"n_pairs": len(pairs), "baselines": results, "elapsed": elapsed}, f, indent=2)
    logger.info(f"  Saved to {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Bench3 External Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_bench3_external(sd, orthologs, output_dir, sample=500):
    """Run external baselines on bench3 (Essentiality) cases."""
    logger.info(f"\n{'=' * 80}")
    logger.info("BENCH3 EXTERNAL: Essentiality Conservation")
    logger.info(f"{'=' * 80}\n")

    root = sd.root
    essential_threshold = -0.5

    # Parse external data
    depmap = parse_depmap(str(root / "dataset" / "external" / "depmap_gene_effect.csv"))
    impc = parse_impc_viability(str(root / "dataset" / "external" / "impc_viability_screen.csv"))

    # Map human essential genes
    h_essential = []
    h_nonessential = []
    for h_sym, score in depmap.items():
        ensg = sd.sym_to_ensg.get(h_sym)
        if ensg is None or ensg not in sd.h_genes:
            continue
        hi = sd.h_genes[ensg]
        if score < essential_threshold:
            h_essential.append((hi, h_sym, score))
        elif score > -0.2:
            h_nonessential.append((hi, h_sym, score))

    # Map mouse essential genes
    m_essential_set = set()
    m_screened_set = set()
    for m_sym, is_ess in impc.items():
        ensmusg = sd.sym_to_ensmusg.get(m_sym)
        if ensmusg is None or ensmusg not in sd.m_genes:
            continue
        mi = sd.m_genes[ensmusg]
        m_screened_set.add(mi)
        if is_ess:
            m_essential_set.add(mi)

    bg_rate = len(m_essential_set) / max(len(m_screened_set), 1)
    logger.info(f"  Human: {len(h_essential)} essential, {len(h_nonessential)} non-essential")
    logger.info(f"  Mouse: {len(m_essential_set)} essential / {len(m_screened_set)} screened (bg={bg_rate:.3f})")

    # Sample
    rng = np.random.RandomState(42)
    if sample is not None and sample < len(h_essential):
        indices = rng.choice(len(h_essential), size=sample, replace=False)
        h_essential = [h_essential[i] for i in sorted(indices)]
    n_ne = min(len(h_nonessential), len(h_essential))
    if n_ne > 0:
        ne_idx = rng.choice(len(h_nonessential), size=n_ne, replace=False)
        h_nonessential = [h_nonessential[i] for i in sorted(ne_idx)]

    logger.info(f"  Evaluating {len(h_essential)} essential genes...")

    K_values = [5, 10, 20, 50, 100]
    results = {}

    for bname in BASELINES:
        precision_at_k = {k: [] for k in K_values}
        hit_at_k = {k: 0 for k in K_values}
        best_ess_ranks = []

        for idx, (hi, h_sym, dep_score) in enumerate(h_essential):
            svd_rank = compute_svd_best_rank(hi, sd) if bname in ("SVD_Annotation", "OrthoFinder_proxy") else None
            if bname == "SVD_Annotation":
                rank_arr = svd_rank
            elif bname == "OrthoFinder_proxy":
                rank_arr = compute_orthofinder_proxy_rank(hi, orthologs, svd_rank, sd.N_m)
            else:
                rank_arr = baseline_rank(bname, hi, sd, orthologs)

            for k in K_values:
                top_k = np.argsort(rank_arr)[:k]
                n_ess = sum(1 for mi in top_k if mi in m_essential_set)
                precision_at_k[k].append(n_ess / k)
                if n_ess > 0:
                    hit_at_k[k] += 1

            ess_r = [int(rank_arr[mi]) for mi in m_essential_set]
            best_ess_ranks.append(min(ess_r) if ess_r else sd.N_m)

            if (idx + 1) % 200 == 0 or (idx + 1) == len(h_essential):
                logger.info(f"  {bname} progress: {idx + 1}/{len(h_essential)}")

        # Non-essential concordance
        ne_prec = {k: [] for k in K_values}
        for hi, h_sym, dep_score in h_nonessential:
            svd_rank = compute_svd_best_rank(hi, sd) if bname in ("SVD_Annotation", "OrthoFinder_proxy") else None
            if bname == "SVD_Annotation":
                rank_arr = svd_rank
            elif bname == "OrthoFinder_proxy":
                rank_arr = compute_orthofinder_proxy_rank(hi, orthologs, svd_rank, sd.N_m)
            else:
                rank_arr = baseline_rank(bname, hi, sd, orthologs)
            for k in K_values:
                top_k = np.argsort(rank_arr)[:k]
                n_ess = sum(1 for mi in top_k if mi in m_essential_set)
                ne_prec[k].append(n_ess / k)

        # Summary
        ep = {}
        for k in K_values:
            ep[f"ess_precision@{k}"] = float(np.mean(precision_at_k[k]))
            ep[f"ess_hitrate@{k}"] = hit_at_k[k] / len(h_essential)

        concordance = {}
        for k in K_values:
            ess_p = np.mean(precision_at_k[k])
            ne_p = np.mean(ne_prec[k]) if ne_prec[k] else bg_rate
            concordance[f"precision_gap@{k}"] = float(ess_p - ne_p)
            if ne_prec[k]:
                try:
                    _, p_val = stats.mannwhitneyu(
                        precision_at_k[k], ne_prec[k], alternative='greater'
                    )
                    concordance[f"mannwhitney_p@{k}"] = float(p_val)
                except Exception:
                    concordance[f"mannwhitney_p@{k}"] = 1.0

        ber = compute_all_metrics(best_ess_ranks)

        results[bname] = {
            "essential_precision": ep,
            "concordance": concordance,
            "best_essential_rank": ber,
        }

        logger.info(
            f"  {bname:<25} Prec@5={ep['ess_precision@5']:.3f} "
            f"HitRate@5={ep['ess_hitrate@5']:.3f} "
            f"BER_MRR={ber['mrr']:.4f} Gap@5={concordance['precision_gap@5']:.4f}"
        )

    # Save
    out_path = Path(output_dir) / "bench3_external.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_essential": len(h_essential),
            "n_nonessential": len(h_nonessential),
            "background_rate": bg_rate,
            "baselines": results,
        }, f, indent=2)
    logger.info(f"  Saved to {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Bench5 External Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_bench5_external(sd, orthologs, output_dir, sample=300):
    """Run external baselines on bench5 (Genome-wide HMD) cases."""
    logger.info(f"\n{'=' * 80}")
    logger.info("BENCH5 EXTERNAL: Genome-Wide HMD Ranking")
    logger.info(f"{'=' * 80}\n")

    root = sd.root
    hmd_records = parse_hmd_human_phenotype(
        str(root / "dataset" / "HMD_HumanPhenotype.rpt")
    )

    # Build pairs excluding same-name orthologs
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
        hi = sd.h_genes[ensg]
        mi = sd.m_genes[ensmusg]
        pk = (hi, mi)
        if pk in seen:
            continue
        seen.add(pk)
        pairs.append({"h_sym": h_sym, "m_sym": m_sym, "hi": hi, "mi": mi})

    # Sample by human gene
    if sample is not None:
        rng = np.random.RandomState(42)
        all_his = list(set(p["hi"] for p in pairs))
        if sample < len(all_his):
            sampled = set(rng.choice(all_his, size=sample, replace=False))
            pairs = [p for p in pairs if p["hi"] in sampled]

    logger.info(f"  Evaluating {len(pairs)} pairs...")

    hi_to_pairs = defaultdict(list)
    for p in pairs:
        hi_to_pairs[p["hi"]].append(p)

    baseline_ranks = {b: [] for b in BASELINES}
    t0 = time.time()

    for idx, (hi, hi_pairs) in enumerate(hi_to_pairs.items()):
        svd_rank = compute_svd_best_rank(hi, sd)
        for bname in BASELINES:
            if bname == "OrthoFinder_proxy":
                rank = compute_orthofinder_proxy_rank(hi, orthologs, svd_rank, sd.N_m)
            elif bname == "SVD_Annotation":
                rank = svd_rank
            else:
                rank = baseline_rank(bname, hi, sd, orthologs)
            for p in hi_pairs:
                baseline_ranks[bname].append(int(rank[p["mi"]]))

        if (idx + 1) % 100 == 0 or (idx + 1) == len(hi_to_pairs):
            logger.info(f"  Progress: {idx + 1}/{len(hi_to_pairs)} genes")

    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s")

    results = {}
    for bname in BASELINES:
        ranks = baseline_ranks[bname]
        if ranks:
            m = compute_all_metrics(ranks)
            results[bname] = m
            logger.info(
                f"  {bname:<25} R@5={m['recall@5']:.1%} R@10={m['recall@10']:.1%} "
                f"R@20={m['recall@20']:.1%} MRR={m['mrr']:.4f} MedR={m['median_rank']:.0f}"
            )

    out_path = Path(output_dir) / "bench5_external.json"
    with open(out_path, "w") as f:
        json.dump({"n_pairs": len(pairs), "baselines": results, "elapsed": elapsed}, f, indent=2)
    logger.info(f"  Saved to {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    output_dir = str(Path(__file__).parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading shared data...")
    t0 = time.time()
    sd = SharedData(skip_ppi=True)
    logger.info(f"Shared data loaded in {time.time() - t0:.1f}s")

    # Load Ensembl orthologs
    logger.info("Loading Ensembl orthologs...")
    orthologs = load_ensembl_orthologs(
        str(sd.root / "dataset" / "external" / "ensembl_human_mouse_orthologs.tsv"),
        sd.h_genes, sd.m_genes,
    )

    # Run bench2 external
    run_bench2_external(sd, orthologs, output_dir, sample=500)

    # Run bench3 external
    run_bench3_external(sd, orthologs, output_dir, sample=500)

    # Run bench5 external
    run_bench5_external(sd, orthologs, output_dir, sample=300)

    logger.info("\nAll external baselines complete!")


if __name__ == "__main__":
    main()
