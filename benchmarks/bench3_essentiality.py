"""Benchmark 3: Cross-Species Essentiality Conservation.

Tests whether model-predicted cross-species gene mappings preserve gene
essentiality information — a validation signal completely independent of
GO annotations and disease associations used in training.

Data sources:
    - DepMap CRISPR gene effect: human gene essentiality scores
      (more negative = more essential; < -0.5 is commonly-essential)
    - IMPC viability screen: mouse gene knockout viability
      (significant=true → lethal or subviable)

Logic:
    1. Parse DepMap: human_symbol -> essentiality score
    2. Parse IMPC: mouse_symbol -> essential (bool)
    3. Map symbols to ENSG/ENSMUSG via indexers
    4. For each essential human gene, model predicts top-K mouse genes
    5. Metrics:
       - Essential Precision@K: fraction of top-K that are IMPC-essential
       - Essential Enrichment: precision vs background rate
       - Essentiality Concordance@K: across all human genes, correlation
         between human essentiality and predicted mouse essentiality
"""

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats

from benchmarks.metrics import compute_all_metrics
from benchmarks.model_variants import ModelEvaluator, SharedData

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing functions
# ─────────────────────────────────────────────────────────────────────────────

def parse_depmap(path: str) -> Dict[str, float]:
    """Parse DepMap gene effect file.

    Format: gene,mean_gene_effect,n_cell_lines
    Gene column: "SYMBOL (ENTREZ_ID)" e.g. "TP53 (7157)"

    Returns:
        Dict of human_symbol -> mean_gene_effect
    """
    gene_scores = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene_str = row["gene"]
            # Parse symbol from "SYMBOL (ENTREZ_ID)"
            paren_idx = gene_str.find(" (")
            if paren_idx > 0:
                symbol = gene_str[:paren_idx].strip()
            else:
                symbol = gene_str.strip()
            score = float(row["mean_gene_effect"])
            n_lines = int(row["n_cell_lines"])
            # Only use genes tested in >= 100 cell lines for reliability
            if n_lines >= 100:
                gene_scores[symbol] = score

    logger.info(
        f"  DepMap: {len(gene_scores)} genes "
        f"(reliable, >= 100 cell lines)"
    )
    return gene_scores


def parse_impc_viability(path: str) -> Dict[str, bool]:
    """Parse IMPC viability screen.

    Format: marker_symbol,marker_accession_id,mp_term_id_options,
            mp_term_name_options,p_value,significant,effect_size,
            zygosity,top_level_mp_term_name

    Returns:
        Dict of mouse_symbol -> is_essential (True if lethal/subviable)
    """
    mouse_essential = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["marker_symbol"].strip()
            sig = row["significant"].strip().lower() == "true"
            # A gene is essential if ANY screen shows significant lethality
            if symbol not in mouse_essential:
                mouse_essential[symbol] = sig
            elif sig:
                mouse_essential[symbol] = True

    n_essential = sum(v for v in mouse_essential.values())
    logger.info(
        f"  IMPC: {len(mouse_essential)} genes screened, "
        f"{n_essential} essential (lethal/subviable)"
    )
    return mouse_essential


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark logic
# ─────────────────────────────────────────────────────────────────────────────

def run_bench3(
    evaluator: ModelEvaluator,
    output_dir: str = None,
    sample: int = None,
    essential_threshold: float = -0.5,
) -> dict:
    """Run Benchmark 3: Cross-Species Essentiality Conservation.

    Args:
        evaluator: A loaded ModelEvaluator instance.
        output_dir: Directory to write JSON results.
        sample: If set, randomly sample N essential human genes.
        essential_threshold: DepMap score threshold for "essential"
                            (default -0.5).

    Returns:
        Dictionary with essentiality transfer metrics.
    """
    sd = evaluator.shared
    variant = evaluator.variant
    root = sd.root

    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCHMARK 3: Essentiality Conservation [{variant}]")
    logger.info(f"{'=' * 80}\n")

    # 1. Parse external data
    logger.info("Parsing essentiality data...")
    depmap = parse_depmap(
        str(root / "dataset" / "external" / "depmap_gene_effect.csv")
    )
    impc = parse_impc_viability(
        str(root / "dataset" / "external" / "impc_viability_screen.csv")
    )

    # 2. Map to our indexers
    logger.info("Mapping to gene indexers...")

    # Map human DepMap genes
    h_essential = []  # (hi, symbol, depmap_score)
    h_nonessential = []
    for h_sym, score in depmap.items():
        ensg = sd.sym_to_ensg.get(h_sym)
        if ensg is None or ensg not in sd.h_genes:
            continue
        hi = sd.h_genes[ensg]
        if score < essential_threshold:
            h_essential.append((hi, h_sym, score))
        elif score > -0.2:  # clearly non-essential
            h_nonessential.append((hi, h_sym, score))

    logger.info(
        f"  Human: {len(h_essential)} essential, "
        f"{len(h_nonessential)} non-essential "
        f"(mapped to indexer)"
    )

    # Map mouse IMPC genes
    m_essential_set = set()  # set of mouse gene indices
    m_screened_set = set()   # all screened mouse genes
    for m_sym, is_ess in impc.items():
        ensmusg = sd.sym_to_ensmusg.get(m_sym)
        if ensmusg is None or ensmusg not in sd.m_genes:
            continue
        mi = sd.m_genes[ensmusg]
        m_screened_set.add(mi)
        if is_ess:
            m_essential_set.add(mi)

    bg_rate = len(m_essential_set) / max(len(m_screened_set), 1)
    logger.info(
        f"  Mouse: {len(m_essential_set)} essential / "
        f"{len(m_screened_set)} screened "
        f"(background rate = {bg_rate:.3f})"
    )

    if not h_essential:
        logger.error("No essential human genes found. Aborting.")
        return {"error": "No essential genes"}

    # 3. Optionally sample
    rng = np.random.RandomState(42)
    if sample is not None and sample < len(h_essential):
        indices = rng.choice(len(h_essential), size=sample, replace=False)
        h_essential = [h_essential[i] for i in sorted(indices)]
        logger.info(f"  Sampled {sample} essential human genes")

    # Also sample non-essential for concordance analysis
    n_nonessential = min(len(h_nonessential), len(h_essential))
    if n_nonessential > 0:
        ne_indices = rng.choice(
            len(h_nonessential), size=n_nonessential, replace=False
        )
        h_nonessential = [h_nonessential[i] for i in sorted(ne_indices)]

    # 4. Evaluate
    logger.info(f"Evaluating {len(h_essential)} essential human genes...")
    t0 = time.time()

    K_values = [5, 10, 20, 50, 100]
    gene_sym = sd.gene_sym
    h_inv = sd.indexers.get("human_genes_inv", {})

    # Essential precision: for essential human genes, what fraction of
    # top-K mouse predictions are IMPC-essential?
    precision_at_k = {k: [] for k in K_values}
    # Essential hit: does ANY top-K contain an IMPC-essential gene?
    hit_at_k = {k: 0 for k in K_values}
    # Essential enrichment: precision / background rate
    # Concordance: rank of best IMPC-essential mouse gene
    best_essential_ranks = []

    for idx, (hi, h_sym, dep_score) in enumerate(h_essential):
        result = evaluator.rank_mouse_genes(hi, h_sym)
        rank_arr = (
            result['rrf_rank']
            if result['rrf_rank'] is not None
            else result['fgw_rank']
        )

        # Get the top-K mouse gene indices
        # rank_arr is 1-based; find positions with lowest rank
        for k in K_values:
            top_k_indices = np.argsort(rank_arr)[:k]
            n_essential_in_topk = sum(
                1 for mi in top_k_indices if mi in m_essential_set
            )
            precision_at_k[k].append(n_essential_in_topk / k)
            if n_essential_in_topk > 0:
                hit_at_k[k] += 1

        # Best rank of any IMPC-essential mouse gene
        essential_ranks = [
            int(rank_arr[mi]) for mi in m_essential_set
        ]
        if essential_ranks:
            best_essential_ranks.append(min(essential_ranks))
        else:
            best_essential_ranks.append(sd.N_m)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(h_essential):
            logger.info(
                f"  Progress: {idx + 1}/{len(h_essential)} essential genes"
            )

    # 5. Concordance analysis: essential vs non-essential human genes
    # For both groups, compute the model's average "essential score"
    # for top-K mouse predictions
    logger.info(
        f"Computing concordance ({len(h_nonessential)} non-essential)..."
    )
    ne_precision_at_k = {k: [] for k in K_values}
    for idx, (hi, h_sym, dep_score) in enumerate(h_nonessential):
        result = evaluator.rank_mouse_genes(hi, h_sym)
        rank_arr = (
            result['rrf_rank']
            if result['rrf_rank'] is not None
            else result['fgw_rank']
        )
        for k in K_values:
            top_k_indices = np.argsort(rank_arr)[:k]
            n_ess = sum(1 for mi in top_k_indices if mi in m_essential_set)
            ne_precision_at_k[k].append(n_ess / k)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(h_nonessential):
            logger.info(
                f"  NE Progress: {idx + 1}/{len(h_nonessential)}"
            )

    elapsed = time.time() - t0

    # 6. Compute summary metrics
    rank_key = "rrf_rank" if evaluator.use_rrf else "fgw_rank"

    # Essential precision
    ess_precision = {}
    for k in K_values:
        vals = precision_at_k[k]
        ess_precision[f"ess_precision@{k}"] = float(np.mean(vals))
        ess_precision[f"ess_hit@{k}"] = hit_at_k[k]
        ess_precision[f"ess_hitrate@{k}"] = (
            hit_at_k[k] / len(h_essential)
        )

    # Enrichment over background
    enrichment = {}
    for k in K_values:
        p = np.mean(precision_at_k[k])
        enrichment[f"enrichment@{k}"] = float(p / max(bg_rate, 1e-10))

    # Concordance: precision gap between essential and non-essential
    concordance = {}
    for k in K_values:
        ess_p = np.mean(precision_at_k[k])
        ne_p = np.mean(ne_precision_at_k[k]) if ne_precision_at_k[k] else bg_rate
        concordance[f"precision_gap@{k}"] = float(ess_p - ne_p)
        # One-sided t-test: essential > non-essential
        if ne_precision_at_k[k]:
            try:
                t_stat, p_val = stats.mannwhitneyu(
                    precision_at_k[k], ne_precision_at_k[k],
                    alternative='greater',
                )
                concordance[f"mannwhitney_p@{k}"] = float(p_val)
            except Exception:
                concordance[f"mannwhitney_p@{k}"] = 1.0

    # Best essential rank statistics
    best_rank_metrics = compute_all_metrics(best_essential_ranks)

    summary = {
        "variant": variant,
        "primary_metric": rank_key,
        "essential_threshold": essential_threshold,
        "n_essential_human": len(h_essential),
        "n_nonessential_human": len(h_nonessential),
        "n_essential_mouse": len(m_essential_set),
        "n_screened_mouse": len(m_screened_set),
        "background_rate": round(bg_rate, 4),
        "elapsed_sec": round(elapsed, 1),
        "essential_precision": ess_precision,
        "enrichment": enrichment,
        "concordance": concordance,
        "best_essential_rank": best_rank_metrics,
    }

    # Log summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCH3 SUMMARY [{variant}] ({rank_key})")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"  Essential human: {len(h_essential)}, "
        f"Non-essential: {len(h_nonessential)}"
    )
    logger.info(
        f"  Mouse essential: {len(m_essential_set)}/"
        f"{len(m_screened_set)} (bg={bg_rate:.3f})"
    )
    logger.info(f"  Time: {elapsed:.1f}s")

    logger.info(f"\n  ESSENTIAL PRECISION:")
    for k in K_values:
        ep = ess_precision[f"ess_precision@{k}"]
        hr = ess_precision[f"ess_hitrate@{k}"]
        en = enrichment[f"enrichment@{k}"]
        logger.info(
            f"    @{k:>3d}: precision={ep:.4f}, "
            f"hitrate={hr:.4f}, enrichment={en:.2f}x"
        )

    logger.info(f"\n  CONCORDANCE (essential vs non-essential):")
    for k in K_values:
        gap = concordance[f"precision_gap@{k}"]
        pval = concordance.get(f"mannwhitney_p@{k}", "N/A")
        logger.info(
            f"    @{k:>3d}: gap={gap:.4f}, p={pval:.2e}"
            if isinstance(pval, float)
            else f"    @{k:>3d}: gap={gap:.4f}, p={pval}"
        )

    logger.info(f"\n  BEST ESSENTIAL RANK (across {len(h_essential)} genes):")
    logger.info(f"    MRR:         {best_rank_metrics['mrr']:.4f}")
    logger.info(f"    Median Rank: {best_rank_metrics['median_rank']:.1f}")
    for k in [10, 50, 100]:
        logger.info(
            f"    Recall@{k:>3d}:  "
            f"{best_rank_metrics[f'recall@{k}']:.4f}"
        )

    # Save results
    out_path = Path(output_dir) / f"bench3_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return summary


def main():
    """Standalone entry point for bench3."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Bench3: Cross-Species Essentiality Conservation"
    )
    parser.add_argument(
        "--variant", default="M1",
        choices=["M1", "M2", "M3", "M4"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-ppi", action="store_true")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N essential human genes",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    shared = SharedData(skip_ppi=args.skip_ppi)
    evaluator = ModelEvaluator(args.variant, shared, device=device)

    run_bench3(evaluator, output_dir=args.output_dir, sample=args.sample)
    evaluator.free()


if __name__ == "__main__":
    main()
