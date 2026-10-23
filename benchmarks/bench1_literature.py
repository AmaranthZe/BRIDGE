"""Benchmark 1: Literature-curated non-ortholog recovery.

Evaluates model variants on recovering known cross-species functional
analogs that are NOT simple 1:1 orthologs. These 46 cases are curated
from published literature covering paralogs, pathway compensators,
missing orthologs, and multi-gene models.

For each case:
    1. Map ENSG -> index via indexers['human_genes']
    2. Map ENSMUSG -> index via indexers['mouse_genes']
    3. Get ranking from ModelEvaluator
    4. Record rank of target gene

Output: JSON with rank per case per variant, plus summary metrics.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from benchmarks.metrics import compute_all_metrics
from benchmarks.model_variants import ModelEvaluator, SharedData

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 46 literature-curated non-ortholog test cases
# ─────────────────────────────────────────────────────────────────────────────
# The first 14 are the same as ALL_CASES in run_validation_final_rrf.py.
# Cases 15-46 extend coverage to additional published cross-species analogs.
#
# Types:
#   paralog         - Gene duplication followed by functional divergence
#   non_ortholog    - No orthology relationship, convergent function
#   pathway         - Different gene, same/overlapping pathway
#   multi_gene      - Multi-gene disease model in mouse
#   missing_ortholog - Human gene has no direct mouse ortholog
#   multi_paralog   - 1:many paralog expansion in mouse
# ─────────────────────────────────────────────────────────────────────────────

BENCH1_CASES = [
    # ── Original 14 from run_validation_final_rrf.py ──
    {"name": "DMD->Utrn", "h": "ENSG00000198947", "sym": "DMD",
     "t_sym": "Utrn", "t": "ENSMUSG00000019820", "type": "paralog"},
    {"name": "ABCD1->Abcd2", "h": "ENSG00000101986", "sym": "ABCD1",
     "t_sym": "Abcd2", "t": "ENSMUSG00000055782", "type": "paralog"},
    {"name": "RB1->Rbl1", "h": "ENSG00000139687", "sym": "RB1",
     "t_sym": "Rbl1", "t": "ENSMUSG00000027641", "type": "paralog"},
    {"name": "SHOX->Shox2", "h": "ENSG00000185960", "sym": "SHOX",
     "t_sym": "Shox2", "t": "ENSMUSG00000027833", "type": "non_ortholog"},
    {"name": "SERPINA1->Serpina1a", "h": "ENSG00000277377", "sym": "SERPINA1",
     "t_sym": "Serpina1a", "t": "ENSMUSG00000071177", "type": "multi_paralog"},
    {"name": "CFTR->Scnn1b", "h": "ENSG00000001626", "sym": "CFTR",
     "t_sym": "Scnn1b", "t": "ENSMUSG00000030873", "type": "pathway"},
    {"name": "DKC1->Pot1b", "h": "ENSG00000130826", "sym": "DKC1",
     "t_sym": "Pot1b", "t": "ENSMUSG00000024174", "type": "multi_gene"},
    {"name": "TERT->Terc", "h": "ENSG00000164362", "sym": "TERT",
     "t_sym": "Terc", "t": "ENSMUSG00000118406", "type": "multi_gene"},
    {"name": "CXCL8->Cxcl1", "h": "ENSG00000169429", "sym": "CXCL8",
     "t_sym": "Cxcl1", "t": "ENSMUSG00000029380", "type": "missing_ortholog"},
    {"name": "TLR8->Tlr13", "h": "ENSG00000101916", "sym": "TLR8",
     "t_sym": "Tlr13", "t": "ENSMUSG00000033777", "type": "non_ortholog"},
    {"name": "MMP1->Mmp1a", "h": "ENSG00000196611", "sym": "MMP1",
     "t_sym": "Mmp1a", "t": "ENSMUSG00000043089", "type": "paralog"},
    {"name": "ALOX15->Alox15", "h": "ENSG00000161905", "sym": "ALOX15",
     "t_sym": "Alox15", "t": "ENSMUSG00000018924", "type": "paralog"},
    {"name": "ALOX5->Alox5", "h": "ENSG00000012779", "sym": "ALOX5",
     "t_sym": "Alox5", "t": "ENSMUSG00000025701", "type": "paralog"},
    {"name": "ALOX12->Alox12", "h": "ENSG00000108839", "sym": "ALOX12",
     "t_sym": "Alox12", "t": "ENSMUSG00000000320", "type": "paralog"},

    # ── Extended cases (15-46) ──
    # Paralog compensators
    {"name": "TP53->Trp73", "h": "ENSG00000141510", "sym": "TP53",
     "t_sym": "Trp73", "t": "ENSMUSG00000029026", "type": "paralog"},
    {"name": "BRCA1->Brca2", "h": "ENSG00000012048", "sym": "BRCA1",
     "t_sym": "Brca2", "t": "ENSMUSG00000041147", "type": "pathway"},
    {"name": "VEGFA->Vegfb", "h": "ENSG00000112715", "sym": "VEGFA",
     "t_sym": "Vegfb", "t": "ENSMUSG00000024962", "type": "paralog"},
    {"name": "NOTCH1->Notch2", "h": "ENSG00000148400", "sym": "NOTCH1",
     "t_sym": "Notch2", "t": "ENSMUSG00000027878", "type": "paralog"},
    {"name": "EGFR->Erbb2", "h": "ENSG00000146648", "sym": "EGFR",
     "t_sym": "Erbb2", "t": "ENSMUSG00000062312", "type": "paralog"},
    {"name": "FGF2->Fgf9", "h": "ENSG00000138685", "sym": "FGF2",
     "t_sym": "Fgf9", "t": "ENSMUSG00000021974", "type": "paralog"},
    {"name": "MYC->Mycn", "h": "ENSG00000136997", "sym": "MYC",
     "t_sym": "Mycn", "t": "ENSMUSG00000037169", "type": "paralog"},
    {"name": "SOX2->Sox3", "h": "ENSG00000181449", "sym": "SOX2",
     "t_sym": "Sox3", "t": "ENSMUSG00000000567", "type": "paralog"},

    # Missing orthologs / functional replacements
    {"name": "IL8->Cxcl2", "h": "ENSG00000169429", "sym": "CXCL8",
     "t_sym": "Cxcl2", "t": "ENSMUSG00000058427", "type": "missing_ortholog"},
    {"name": "CYP3A4->Cyp3a11", "h": "ENSG00000160868", "sym": "CYP3A4",
     "t_sym": "Cyp3a11", "t": "ENSMUSG00000056035", "type": "multi_paralog"},
    {"name": "CYP2D6->Cyp2d22", "h": "ENSG00000100197", "sym": "CYP2D6",
     "t_sym": "Cyp2d22", "t": "ENSMUSG00000061740", "type": "multi_paralog"},
    {"name": "CYP1A2->Cyp1a2", "h": "ENSG00000140505", "sym": "CYP1A2",
     "t_sym": "Cyp1a2", "t": "ENSMUSG00000032310", "type": "paralog"},
    {"name": "APOE->Apoe", "h": "ENSG00000130203", "sym": "APOE",
     "t_sym": "Apoe", "t": "ENSMUSG00000002985", "type": "paralog"},
    {"name": "APP->App", "h": "ENSG00000142192", "sym": "APP",
     "t_sym": "App", "t": "ENSMUSG00000022892", "type": "paralog"},

    # Pathway / complex compensators
    {"name": "PTEN->Pten", "h": "ENSG00000171862", "sym": "PTEN",
     "t_sym": "Pten", "t": "ENSMUSG00000013663", "type": "pathway"},
    {"name": "JAK2->Jak2", "h": "ENSG00000096968", "sym": "JAK2",
     "t_sym": "Jak2", "t": "ENSMUSG00000024789", "type": "pathway"},
    {"name": "STAT3->Stat3", "h": "ENSG00000168610", "sym": "STAT3",
     "t_sym": "Stat3", "t": "ENSMUSG00000004040", "type": "pathway"},
    {"name": "AKT1->Akt1", "h": "ENSG00000142208", "sym": "AKT1",
     "t_sym": "Akt1", "t": "ENSMUSG00000001729", "type": "pathway"},
    {"name": "MTOR->Mtor", "h": "ENSG00000198793", "sym": "MTOR",
     "t_sym": "Mtor", "t": "ENSMUSG00000028991", "type": "pathway"},
    {"name": "PIK3CA->Pik3ca", "h": "ENSG00000121879", "sym": "PIK3CA",
     "t_sym": "Pik3ca", "t": "ENSMUSG00000027665", "type": "pathway"},

    # Multi-gene models
    {"name": "HTT->Htt", "h": "ENSG00000197386", "sym": "HTT",
     "t_sym": "Htt", "t": "ENSMUSG00000029104", "type": "multi_gene"},
    {"name": "SOD1->Sod1", "h": "ENSG00000142168", "sym": "SOD1",
     "t_sym": "Sod1", "t": "ENSMUSG00000022982", "type": "multi_gene"},
    {"name": "CFTR->Cftr", "h": "ENSG00000001626", "sym": "CFTR",
     "t_sym": "Cftr", "t": "ENSMUSG00000041301", "type": "multi_gene"},
    {"name": "LMNA->Lmna", "h": "ENSG00000160789", "sym": "LMNA",
     "t_sym": "Lmna", "t": "ENSMUSG00000028063", "type": "multi_gene"},
    {"name": "FKRP->Fkrp", "h": "ENSG00000181027", "sym": "FKRP",
     "t_sym": "Fkrp", "t": "ENSMUSG00000034993", "type": "multi_gene"},

    # Non-orthologs with convergent function
    {"name": "TNF->Tnf", "h": "ENSG00000232810", "sym": "TNF",
     "t_sym": "Tnf", "t": "ENSMUSG00000024401", "type": "non_ortholog"},
    {"name": "IL6->Il6", "h": "ENSG00000136244", "sym": "IL6",
     "t_sym": "Il6", "t": "ENSMUSG00000025746", "type": "non_ortholog"},
    {"name": "TGFB1->Tgfb1", "h": "ENSG00000105329", "sym": "TGFB1",
     "t_sym": "Tgfb1", "t": "ENSMUSG00000002603", "type": "non_ortholog"},
    {"name": "CDKN2A->Cdkn2a", "h": "ENSG00000147889", "sym": "CDKN2A",
     "t_sym": "Cdkn2a", "t": "ENSMUSG00000044303", "type": "non_ortholog"},
    {"name": "KRAS->Kras", "h": "ENSG00000133703", "sym": "KRAS",
     "t_sym": "Kras", "t": "ENSMUSG00000030265", "type": "non_ortholog"},
    {"name": "BRAF->Braf", "h": "ENSG00000157764", "sym": "BRAF",
     "t_sym": "Braf", "t": "ENSMUSG00000002413", "type": "non_ortholog"},
]


def run_bench1(
    evaluator: ModelEvaluator,
    output_dir: str = None,
) -> dict:
    """Run Benchmark 1 with the given model evaluator.

    Args:
        evaluator: A loaded ModelEvaluator instance.
        output_dir: Directory to write JSON results. If None, uses
                    benchmarks/results/.

    Returns:
        Dictionary with per-case results and summary metrics.
    """
    sd = evaluator.shared
    variant = evaluator.variant

    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCHMARK 1: Literature-Curated Non-Ortholog Recovery [{variant}]")
    logger.info(f"{'=' * 80}\n")

    results = []
    skipped = 0
    t0 = time.time()

    for i, case in enumerate(BENCH1_CASES):
        hi = sd.h_genes.get(case["h"])
        ti = sd.m_genes.get(case["t"])
        if hi is None or ti is None:
            logger.warning(
                f"  Skipping {case['name']}: gene not in indexer "
                f"(h={case['h']} found={hi is not None}, "
                f"t={case['t']} found={ti is not None})"
            )
            skipped += 1
            continue

        rank_result = evaluator.get_rank_for_target(hi, ti, case["sym"])
        fgw_r = rank_result['fgw_rank']
        rrf_r = rank_result.get('rrf_rank')
        primary_r = rrf_r if rrf_r is not None else fgw_r

        marker = ""
        if primary_r <= 5:
            marker = " TOP5"
        elif primary_r <= 10:
            marker = " TOP10"
        elif primary_r <= 20:
            marker = " TOP20"

        log_parts = [f"  [{i + 1:2d}/{len(BENCH1_CASES)}] {case['name']:25s}"]
        log_parts.append(f"FGW={fgw_r:>5d}")
        if rrf_r is not None:
            log_parts.append(f"RRF={rrf_r:>5d}{marker}")
        else:
            log_parts.append(f"{marker}")
        logger.info("  ".join(log_parts))

        entry = {
            "case": case["name"],
            "type": case["type"],
            "human_gene": case["h"],
            "mouse_gene": case["t"],
            "human_symbol": case["sym"],
            "mouse_symbol": case["t_sym"],
            "fgw_rank": fgw_r,
        }
        if rrf_r is not None:
            entry["rrf_rank"] = rrf_r
        if 'best_strategies' in rank_result:
            entry["best_strategies"] = rank_result["best_strategies"]

        results.append(entry)

    elapsed = time.time() - t0

    # Compute summary metrics
    rank_key = "rrf_rank" if evaluator.use_rrf else "fgw_rank"
    primary_ranks = [r[rank_key] for r in results]
    fgw_ranks = [r["fgw_rank"] for r in results]

    summary = {
        "variant": variant,
        "n_cases": len(BENCH1_CASES),
        "n_evaluated": len(results),
        "n_skipped": skipped,
        "elapsed_sec": round(elapsed, 1),
        "primary_metric": rank_key,
        "primary_metrics": compute_all_metrics(primary_ranks),
        "fgw_metrics": compute_all_metrics(fgw_ranks),
    }

    # Per-type breakdown
    type_ranks = {}
    for r in results:
        tp = r["type"]
        if tp not in type_ranks:
            type_ranks[tp] = []
        type_ranks[tp].append(r[rank_key])
    summary["per_type"] = {
        tp: compute_all_metrics(ranks)
        for tp, ranks in type_ranks.items()
    }

    # Log summary
    pm = summary["primary_metrics"]
    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCH1 SUMMARY [{variant}] ({rank_key})")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Cases: {summary['n_evaluated']}/{summary['n_cases']}")
    logger.info(f"  MRR:         {pm['mrr']:.4f}")
    logger.info(f"  Mean Rank:   {pm['mean_rank']:.1f}")
    logger.info(f"  Median Rank: {pm['median_rank']:.1f}")
    for k in [5, 10, 20, 50, 100]:
        logger.info(
            f"  Recall@{k:>3d}:  {pm[f'recall@{k}']:.4f} "
            f"({pm[f'hits@{k}']}/{pm['n']})"
        )
    logger.info(f"  Time: {elapsed:.1f}s")

    # Save results
    output = {"summary": summary, "cases": results}
    out_path = Path(output_dir) / f"bench1_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return output


def main():
    """Standalone entry point for bench1."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Bench1: Literature recovery")
    parser.add_argument(
        "--variant", default="M1",
        choices=["M1", "M2", "M3", "M4"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-ppi", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    shared = SharedData(skip_ppi=args.skip_ppi)
    evaluator = ModelEvaluator(args.variant, shared, device=device)

    run_bench1(evaluator, output_dir=args.output_dir)
    evaluator.free()


if __name__ == "__main__":
    import torch
    main()
