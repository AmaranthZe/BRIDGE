"""Benchmark 2: MGI Phenotype cross-species transfer.

Evaluates whether human genes can be mapped to mouse genes that share
mammalian phenotype (MP) terms, using the MGI gene-phenotype database.

Logic:
    1. Parse MGI_GenePheno.rpt: mouse MGI_ID -> set of MP terms
    2. Parse HMD_HumanPhenotype.rpt: (human_symbol, mouse_symbol, MP terms)
    3. Map symbols to ENSG/ENSMUSG IDs using indexers reverse mapping
    4. Build ground truth: human gene -> set of mouse genes sharing phenotypes
    5. Stratified evaluation: GO Jaccard strata
    6. Compute Recall@K, MRR per stratum and overall
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
from benchmarks.model_variants import ModelEvaluator, SharedData

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing functions
# ─────────────────────────────────────────────────────────────────────────────

def parse_mgi_genepheno(path: str) -> Dict[str, Set[str]]:
    """Parse MGI_GenePheno.rpt: extract mouse MGI_ID -> set of MP terms.

    File format: tab-separated, 8 columns.
    Column 4 (0-indexed) = MP term (single term per row).
    Column 6 (0-indexed) = Gene MGI_ID.
    """
    mgi_to_mp = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            mp_term = parts[4].strip()
            mgi_id = parts[6].strip()
            if mp_term.startswith("MP:") and mgi_id.startswith("MGI:"):
                mgi_to_mp[mgi_id].add(mp_term)

    logger.info(
        f"  MGI_GenePheno: {len(mgi_to_mp)} genes with MP annotations"
    )
    return dict(mgi_to_mp)


def parse_hmd_human_phenotype(
    path: str,
) -> List[Tuple[str, str, str, Set[str]]]:
    """Parse HMD_HumanPhenotype.rpt.

    File format: tab-separated columns:
        HumanSymbol, EntrezID, MouseSymbol, MGI_ID, MP_terms, (empty)

    MP_terms is comma-separated like "MP:0005367, MP:0005369, ..."

    Returns:
        List of (human_symbol, mouse_symbol, mgi_id, set_of_mp_terms)
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            h_sym = parts[0].strip()
            m_sym = parts[2].strip()
            mgi_id = parts[3].strip()
            mp_str = parts[4].strip() if len(parts) > 4 else ""

            mp_terms = set()
            if mp_str:
                for term in mp_str.split(","):
                    term = term.strip()
                    if term.startswith("MP:"):
                        mp_terms.add(term)

            records.append((h_sym, m_sym, mgi_id, mp_terms))

    logger.info(
        f"  HMD_HumanPhenotype: {len(records)} rows, "
        f"{len(set(r[0] for r in records))} unique human symbols"
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth construction
# ─────────────────────────────────────────────────────────────────────────────

def build_ground_truth(
    hmd_records: List[Tuple[str, str, str, Set[str]]],
    mgi_pheno: Dict[str, Set[str]],
    sym_to_ensg: Dict[str, str],
    sym_to_ensmusg: Dict[str, str],
    h_genes: Dict[str, int],
    m_genes: Dict[str, int],
) -> Dict[int, List[Tuple[int, Set[str]]]]:
    """Build ground truth mapping: human gene idx -> [(mouse gene idx, shared MP terms)].

    Only includes pairs where:
        - Both symbols map to valid ENSG/ENSMUSG IDs in our indexers
        - The pair shares at least 1 MP term (from HMD file or MGI file)
    """
    gt = defaultdict(list)
    seen_pairs = set()

    for h_sym, m_sym, mgi_id, hmd_mp_terms in hmd_records:
        ensg = sym_to_ensg.get(h_sym)
        ensmusg = sym_to_ensmusg.get(m_sym)

        if ensg is None or ensmusg is None:
            continue
        if ensg not in h_genes or ensmusg not in m_genes:
            continue

        hi = h_genes[ensg]
        mi = m_genes[ensmusg]
        pair_key = (hi, mi)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Combine MP terms from HMD file and MGI file
        mp_terms = set(hmd_mp_terms)
        mgi_terms = mgi_pheno.get(mgi_id, set())
        mp_terms |= mgi_terms

        if len(mp_terms) > 0:
            gt[hi].append((mi, mp_terms))

    logger.info(
        f"  Ground truth: {len(gt)} human genes with "
        f"{sum(len(v) for v in gt.values())} total pairs"
    )
    return dict(gt)


# ─────────────────────────────────────────────────────────────────────────────
# GO Jaccard stratification
# ─────────────────────────────────────────────────────────────────────────────

def compute_go_jaccard(
    hi: int, mi: int, sparse_matrices: dict
) -> float:
    """Compute GO Biological Process Jaccard similarity between a
    human gene and a mouse gene using sparse annotation matrices.

    Uses W_hB (human -> BP) and W_mB (mouse -> BP).
    """
    h_bp = (sparse_matrices['W_hB'][hi].toarray().squeeze() > 0)
    m_bp = (sparse_matrices['W_mB'][mi].toarray().squeeze() > 0)
    intersection = (h_bp & m_bp).sum()
    union = (h_bp | m_bp).sum()
    return intersection / max(union, 1)


def stratify_pairs(
    gt: Dict[int, List[Tuple[int, Set[str]]]],
    sparse_matrices: dict,
) -> Dict[str, List[Tuple[int, int, Set[str]]]]:
    """Split ground-truth pairs into GO Jaccard strata.

    Strata:
        'high':   Jaccard >= 0.3
        'medium': 0.1 <= Jaccard < 0.3
        'low':    0.01 <= Jaccard < 0.1
        'none':   Jaccard < 0.01
    """
    strata = {
        'high': [],
        'medium': [],
        'low': [],
        'none': [],
    }

    for hi, targets in gt.items():
        for mi, mp_terms in targets:
            jac = compute_go_jaccard(hi, mi, sparse_matrices)
            if jac >= 0.3:
                strata['high'].append((hi, mi, mp_terms))
            elif jac >= 0.1:
                strata['medium'].append((hi, mi, mp_terms))
            elif jac >= 0.01:
                strata['low'].append((hi, mi, mp_terms))
            else:
                strata['none'].append((hi, mi, mp_terms))

    for s, pairs in strata.items():
        logger.info(f"  Stratum '{s}': {len(pairs)} pairs")

    return strata


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark logic
# ─────────────────────────────────────────────────────────────────────────────

def run_bench2(
    evaluator: ModelEvaluator,
    output_dir: str = None,
    sample: int = None,
) -> dict:
    """Run Benchmark 2 with the given model evaluator.

    Args:
        evaluator: A loaded ModelEvaluator instance.
        output_dir: Directory to write JSON results.
        sample: If set, randomly sample N human genes to evaluate
                (for quick testing).

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
    logger.info(f"BENCHMARK 2: MGI Phenotype Transfer [{variant}]")
    logger.info(f"{'=' * 80}\n")

    # 1. Parse phenotype data
    logger.info("Parsing phenotype files...")
    mgi_pheno = parse_mgi_genepheno(
        str(root / "dataset" / "MGI_GenePheno.rpt")
    )
    hmd_records = parse_hmd_human_phenotype(
        str(root / "dataset" / "HMD_HumanPhenotype.rpt")
    )

    # 2. Build ground truth
    logger.info("Building ground truth...")
    gt = build_ground_truth(
        hmd_records, mgi_pheno,
        sd.sym_to_ensg, sd.sym_to_ensmusg,
        sd.h_genes, sd.m_genes,
    )

    # 3. Stratify by GO overlap
    logger.info("Stratifying by GO Jaccard...")
    strata = stratify_pairs(gt, sd.sparse)

    # 4. Optionally sample
    if sample is not None:
        # Sample human genes, not pairs
        all_his = list(gt.keys())
        rng = np.random.RandomState(42)
        if sample < len(all_his):
            sampled_his = set(rng.choice(all_his, size=sample, replace=False))
        else:
            sampled_his = set(all_his)
        # Filter strata
        for s in strata:
            strata[s] = [
                (hi, mi, mp) for hi, mi, mp in strata[s]
                if hi in sampled_his
            ]
        logger.info(
            f"Sampled {len(sampled_his)} human genes "
            f"({sum(len(v) for v in strata.values())} pairs)"
        )

    # 5. Evaluate
    logger.info("Evaluating rankings...")
    t0 = time.time()

    # Collect all unique human genes to evaluate (avoid duplicate FGW calls)
    hi_to_pairs = defaultdict(list)
    for stratum_name, pairs in strata.items():
        for hi, mi, mp_terms in pairs:
            hi_to_pairs[hi].append((mi, mp_terms, stratum_name))

    # Get human symbols for PPI
    gene_sym = sd.gene_sym
    h_inv = sd.indexers.get("human_genes_inv", {})

    stratum_ranks = {s: [] for s in strata}
    all_ranks = []
    n_evaluated = 0

    for idx, (hi, pair_list) in enumerate(hi_to_pairs.items()):
        # Resolve human symbol
        ensg = h_inv.get(hi, "")
        h_sym = gene_sym.get(ensg, "")

        # Rank all mouse genes once for this human gene
        result = evaluator.rank_mouse_genes(hi, h_sym)
        rank_arr = result['rrf_rank'] if result['rrf_rank'] is not None else result['fgw_rank']

        for mi, mp_terms, stratum_name in pair_list:
            r = int(rank_arr[mi])
            stratum_ranks[stratum_name].append(r)
            all_ranks.append(r)

        n_evaluated += 1
        if (idx + 1) % 100 == 0 or (idx + 1) == len(hi_to_pairs):
            logger.info(
                f"  Progress: {idx + 1}/{len(hi_to_pairs)} human genes evaluated"
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
        "elapsed_sec": round(elapsed, 1),
        "overall": overall_metrics,
        "strata": stratum_metrics,
    }

    # Log summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCH2 SUMMARY [{variant}] ({rank_key})")
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
    out_path = Path(output_dir) / f"bench2_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return summary


def main():
    """Standalone entry point for bench2."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Bench2: MGI Phenotype transfer"
    )
    parser.add_argument(
        "--variant", default="M1",
        choices=["M1", "M2", "M3", "M4"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-ppi", action="store_true")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N human genes for quick testing")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    shared = SharedData(skip_ppi=args.skip_ppi)
    evaluator = ModelEvaluator(args.variant, shared, device=device)

    run_bench2(evaluator, output_dir=args.output_dir, sample=args.sample)
    evaluator.free()


if __name__ == "__main__":
    import torch
    main()
