"""Benchmark 5: Genome-wide HMD ranking (large-scale).

Evaluates model variants on ranking ALL cross-species gene pairs from the
HMD_HumanPhenotype.rpt file, filtering out trivial same-name orthologs.

Logic:
    1. Parse HMD_HumanPhenotype.rpt for ALL (human, mouse) pairs
    2. Map symbols to ENSG/ENSMUSG IDs
    3. Filter: remove pairs where human_symbol.lower() == mouse_symbol.lower()
    4. For remaining non-trivial pairs, evaluate ranking
    5. Stratify by GO overlap
    6. Support --sample N for quick testing
"""

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from benchmarks.metrics import compute_all_metrics
from benchmarks.bench2_phenotype import (
    parse_hmd_human_phenotype,
    compute_go_jaccard,
)
from benchmarks.model_variants import ModelEvaluator, SharedData

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth construction for HMD genome-wide
# ─────────────────────────────────────────────────────────────────────────────

def build_hmd_pairs(
    hmd_records: List[Tuple[str, str, str, Set[str]]],
    sym_to_ensg: Dict[str, str],
    sym_to_ensmusg: Dict[str, str],
    h_genes: Dict[str, int],
    m_genes: Dict[str, int],
    exclude_same_name: bool = True,
) -> List[Dict]:
    """Build (human, mouse) pairs from HMD file.

    Args:
        hmd_records: Parsed HMD records.
        sym_to_ensg: Symbol -> ENSG mapping.
        sym_to_ensmusg: Symbol -> ENSMUSG mapping.
        h_genes: ENSG -> index mapping.
        m_genes: ENSMUSG -> index mapping.
        exclude_same_name: If True, remove pairs where symbols match
                          (case-insensitive). These are trivial orthologs.

    Returns:
        List of dicts with keys: h_sym, m_sym, ensg, ensmusg, hi, mi
    """
    pairs = []
    seen = set()

    for h_sym, m_sym, mgi_id, mp_terms in hmd_records:
        # Exclude trivial same-name orthologs
        if exclude_same_name and h_sym.lower() == m_sym.lower():
            continue

        ensg = sym_to_ensg.get(h_sym)
        ensmusg = sym_to_ensmusg.get(m_sym)
        if ensg is None or ensmusg is None:
            continue
        if ensg not in h_genes or ensmusg not in m_genes:
            continue

        hi = h_genes[ensg]
        mi = m_genes[ensmusg]
        pair_key = (hi, mi)
        if pair_key in seen:
            continue
        seen.add(pair_key)

        pairs.append({
            "h_sym": h_sym,
            "m_sym": m_sym,
            "ensg": ensg,
            "ensmusg": ensmusg,
            "hi": hi,
            "mi": mi,
        })

    logger.info(
        f"  HMD pairs (non-trivial): {len(pairs)} "
        f"({len(set(p['hi'] for p in pairs))} unique human genes)"
    )
    return pairs


def stratify_hmd_pairs(
    pairs: List[Dict],
    sparse_matrices: dict,
) -> Dict[str, List[Dict]]:
    """Split HMD pairs into GO Jaccard strata.

    Strata:
        'high':   Jaccard >= 0.3
        'medium': 0.1 <= Jaccard < 0.3
        'low':    0.01 <= Jaccard < 0.1
        'none':   Jaccard < 0.01
    """
    strata = {'high': [], 'medium': [], 'low': [], 'none': []}

    for p in pairs:
        jac = compute_go_jaccard(p['hi'], p['mi'], sparse_matrices)
        p['go_jaccard'] = jac
        if jac >= 0.3:
            strata['high'].append(p)
        elif jac >= 0.1:
            strata['medium'].append(p)
        elif jac >= 0.01:
            strata['low'].append(p)
        else:
            strata['none'].append(p)

    for s, sp_list in strata.items():
        logger.info(f"  Stratum '{s}': {len(sp_list)} pairs")

    return strata


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark logic
# ─────────────────────────────────────────────────────────────────────────────

def run_bench5(
    evaluator: ModelEvaluator,
    output_dir: str = None,
    sample: int = None,
) -> dict:
    """Run Benchmark 5: Genome-wide HMD ranking.

    Args:
        evaluator: A loaded ModelEvaluator instance.
        output_dir: Directory to write JSON results.
        sample: If set, randomly sample N human genes to evaluate.

    Returns:
        Dictionary with per-stratum and overall results.
    """
    sd = evaluator.shared
    variant = evaluator.variant
    root = sd.root

    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCHMARK 5: Genome-Wide HMD Ranking [{variant}]")
    logger.info(f"{'=' * 80}\n")

    # 1. Parse HMD
    logger.info("Parsing HMD_HumanPhenotype.rpt...")
    hmd_records = parse_hmd_human_phenotype(
        str(root / "dataset" / "HMD_HumanPhenotype.rpt")
    )

    # 2. Build pairs (excluding same-name orthologs)
    logger.info("Building non-trivial HMD pairs...")
    pairs = build_hmd_pairs(
        hmd_records,
        sd.sym_to_ensg,
        sd.sym_to_ensmusg,
        sd.h_genes,
        sd.m_genes,
        exclude_same_name=True,
    )

    if not pairs:
        logger.error("No valid HMD pairs found. Aborting.")
        return {"error": "No pairs"}

    # 3. Stratify by GO overlap
    logger.info("Stratifying by GO Jaccard...")
    strata = stratify_hmd_pairs(pairs, sd.sparse)

    # 4. Optionally sample by human gene
    if sample is not None:
        all_his = list(set(p['hi'] for p in pairs))
        rng = np.random.RandomState(42)
        if sample < len(all_his):
            sampled_his = set(rng.choice(all_his, size=sample, replace=False))
        else:
            sampled_his = set(all_his)
        # Filter strata
        for s in strata:
            strata[s] = [p for p in strata[s] if p['hi'] in sampled_his]
        pairs = [p for p in pairs if p['hi'] in sampled_his]
        logger.info(
            f"Sampled {len(sampled_his)} human genes "
            f"({len(pairs)} pairs total)"
        )

    # 5. Evaluate
    logger.info("Evaluating rankings...")
    t0 = time.time()

    # Group by human gene to avoid duplicate model calls
    hi_to_pairs = defaultdict(list)
    for p in pairs:
        hi_to_pairs[p['hi']].append(p)

    gene_sym = sd.gene_sym
    h_inv = sd.indexers.get("human_genes_inv", {})

    stratum_ranks = {s: [] for s in strata}
    all_ranks = []
    all_results = []
    n_evaluated = 0

    for idx, (hi, hi_pairs) in enumerate(hi_to_pairs.items()):
        ensg = h_inv.get(hi, "")
        h_sym = gene_sym.get(ensg, "")

        # Rank all mouse genes once
        result = evaluator.rank_mouse_genes(hi, h_sym)
        rank_arr = (
            result['rrf_rank']
            if result['rrf_rank'] is not None
            else result['fgw_rank']
        )

        for p in hi_pairs:
            mi = p['mi']
            r = int(rank_arr[mi])
            all_ranks.append(r)

            # Determine stratum
            jac = p.get('go_jaccard', 0)
            if jac >= 0.3:
                stratum_ranks['high'].append(r)
            elif jac >= 0.1:
                stratum_ranks['medium'].append(r)
            elif jac >= 0.01:
                stratum_ranks['low'].append(r)
            else:
                stratum_ranks['none'].append(r)

            all_results.append({
                "h_sym": p['h_sym'],
                "m_sym": p['m_sym'],
                "rank": r,
                "go_jaccard": round(jac, 4),
            })

        n_evaluated += 1
        if (idx + 1) % 200 == 0 or (idx + 1) == len(hi_to_pairs):
            logger.info(
                f"  Progress: {idx + 1}/{len(hi_to_pairs)} human genes "
                f"({len(all_ranks)} pairs evaluated)"
            )

    elapsed = time.time() - t0

    # 6. Compute metrics
    rank_key = "rrf_rank" if evaluator.use_rrf else "fgw_rank"
    overall_metrics = compute_all_metrics(all_ranks)
    stratum_metrics = {}
    for s, ranks in stratum_ranks.items():
        if ranks:
            stratum_metrics[s] = compute_all_metrics(ranks)
        else:
            stratum_metrics[s] = {"n": 0}

    summary = {
        "variant": variant,
        "primary_metric": rank_key,
        "n_human_genes": len(hi_to_pairs),
        "n_pairs_total": len(all_ranks),
        "n_sampled": sample,
        "elapsed_sec": round(elapsed, 1),
        "overall": overall_metrics,
        "strata": stratum_metrics,
    }

    # Log summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCH5 SUMMARY [{variant}] ({rank_key})")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"  Human genes: {summary['n_human_genes']}, "
        f"Pairs: {summary['n_pairs_total']}"
    )
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"\n  OVERALL:")
    logger.info(f"    MRR:         {overall_metrics['mrr']:.4f}")
    logger.info(f"    Mean Rank:   {overall_metrics['mean_rank']:.1f}")
    logger.info(f"    Median Rank: {overall_metrics['median_rank']:.1f}")
    for k in [10, 50, 100]:
        logger.info(
            f"    Recall@{k:>3d}:  {overall_metrics[f'recall@{k}']:.4f} "
            f"({overall_metrics[f'hits@{k}']}/{overall_metrics['n']})"
        )

    for s in ['high', 'medium', 'low', 'none']:
        sm = stratum_metrics.get(s, {})
        if sm.get('n', 0) == 0:
            continue
        logger.info(f"\n  STRATUM '{s}' (n={sm['n']}):")
        logger.info(f"    MRR:         {sm['mrr']:.4f}")
        logger.info(f"    Median Rank: {sm['median_rank']:.1f}")
        for k in [10, 50, 100]:
            logger.info(
                f"    Recall@{k:>3d}:  {sm[f'recall@{k}']:.4f}"
            )

    # Save results
    out_path = Path(output_dir) / f"bench5_{variant}.json"
    output = {"summary": summary}
    # Include per-pair results only for small runs (< 5000 pairs)
    if len(all_results) < 5000:
        output["pairs"] = all_results
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return summary


def main():
    """Standalone entry point for bench5."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Bench5: Genome-wide HMD ranking"
    )
    parser.add_argument(
        "--variant", default="M1",
        choices=["M1", "M2", "M3", "M4"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-ppi", action="store_true")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N human genes for quick testing",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    shared = SharedData(skip_ppi=args.skip_ppi)
    evaluator = ModelEvaluator(args.variant, shared, device=device)

    run_bench5(evaluator, output_dir=args.output_dir, sample=args.sample)
    evaluator.free()


if __name__ == "__main__":
    import torch
    main()
