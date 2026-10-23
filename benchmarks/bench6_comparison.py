"""Benchmark 6: External Method Comparison.

Compares our M1-M4 variants against external baselines on the same
benchmark cases. External baselines:

1. Ensembl Compara (Sequence Orthology):
   Known one-to-one, one-to-many, many-to-many orthologs from Ensembl.
   Assigns rank 1 to high-confidence orthologs, else N_m.

2. SVD Annotation Cosine (No Learning):
   Cross-species SVD cosine similarity from TF-IDF annotation matrices.
   Represents pure annotation-based similarity without any GNN.

3. 2-hop Annotation Overlap (Graph Statistics):
   Direct W_h @ W_m^T matrix product (annotation overlap count).
   Represents the simplest graph-based method.

4. OrthoFinder-proxy (Ortholog Grouping):
   Approximated using Ensembl ortholog families — all orthologs of
   a human gene receive rank proportional to their homology confidence.

Key paper argument: Sequence orthology methods can only find HOMOLOGS.
Our method (M4) discovers non-homologous functional correspondences
that sequence methods fundamentally cannot identify.
"""

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.sparse as sp

from benchmarks.metrics import compute_all_metrics
from benchmarks.model_variants import (
    ModelEvaluator, SharedData, arr_to_rank, sparse_row_to_rank,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Ensembl Ortholog Baseline
# ─────────────────────────────────────────────────────────────────────────────

def load_ensembl_orthologs(
    path: str,
    h_genes: Dict[str, int],
    m_genes: Dict[str, int],
) -> Dict[int, List[Tuple[int, str, int]]]:
    """Load Ensembl human-mouse ortholog mappings.

    Returns:
        Dict[human_idx] -> List of (mouse_idx, homology_type, confidence)
    """
    orthologs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ensg = row["Gene stable ID"].strip()
            ensmusg = row.get("Mouse gene stable ID", "").strip()
            htype = row.get("Mouse homology type", "").strip()
            conf_str = row.get(
                "Mouse orthology confidence [0 low, 1 high]", ""
            ).strip()

            if not ensmusg or not htype:
                continue
            if ensg not in h_genes or ensmusg not in m_genes:
                continue

            hi = h_genes[ensg]
            mi = m_genes[ensmusg]
            conf = int(conf_str) if conf_str in ("0", "1") else 0
            orthologs[hi].append((mi, htype, conf))

    logger.info(
        f"  Ensembl orthologs: {len(orthologs)} human genes with "
        f"{sum(len(v) for v in orthologs.values())} ortholog pairs"
    )
    return dict(orthologs)


def compute_ensembl_rank(
    hi: int,
    orthologs: Dict[int, List[Tuple[int, str, int]]],
    N_m: int,
) -> np.ndarray:
    """Compute ranking from Ensembl ortholog predictions.

    Orthologs are ranked by: one2one(conf=1) > one2one(conf=0) >
    one2many > many2many. Non-orthologs get rank N_m.
    """
    rank = np.full(N_m, N_m, dtype=np.int64)
    pairs = orthologs.get(hi, [])

    if not pairs:
        return rank

    # Score: higher = more confident
    scored = []
    for mi, htype, conf in pairs:
        if htype == "ortholog_one2one" and conf == 1:
            score = 3.0
        elif htype == "ortholog_one2one" and conf == 0:
            score = 2.0
        elif htype == "ortholog_one2many":
            score = 1.0
        else:  # many2many
            score = 0.5
        scored.append((mi, score))

    # Sort by score descending
    scored.sort(key=lambda x: -x[1])

    for r, (mi, _) in enumerate(scored):
        rank[mi] = r + 1

    return rank


# ─────────────────────────────────────────────────────────────────────────────
# SVD / 2-hop Annotation Baselines
# ─────────────────────────────────────────────────────────────────────────────

def compute_svd_best_rank(hi: int, sd: SharedData) -> np.ndarray:
    """Best SVD cosine ensemble (no learning, annotation-only).

    Uses the E_eq ensemble (equal BP+MF+Disease weighting) which is
    a reasonable default without any hyperparameter tuning.
    """
    N_m = sd.N_m
    s = (
        0.33 * (sd.Z_h_BP[hi:hi + 1] @ sd.Z_m_BP.T)
        + 0.33 * (sd.Z_h_MF[hi:hi + 1] @ sd.Z_m_MF.T)
        + 0.34 * (sd.Z_h_D[hi:hi + 1] @ sd.Z_m_D.T)
    ).squeeze().numpy()
    return arr_to_rank(s, N_m)


def compute_2hop_best_rank(hi: int, sd: SharedData) -> np.ndarray:
    """Best 2-hop annotation overlap (graph statistics, no learning).

    Uses weighted combination of BP, MF, Disease annotation overlaps.
    """
    N_m = sd.N_m
    s = (
        0.25 * np.asarray(sd.P_BP[hi].toarray()).squeeze()
        + 0.15 * np.asarray(sd.P_MF[hi].toarray()).squeeze()
        + 0.60 * np.asarray(sd.P_Dis[hi].toarray()).squeeze()
    )
    return arr_to_rank(s, N_m)


def compute_orthofinder_proxy_rank(
    hi: int,
    orthologs: Dict[int, List[Tuple[int, str, int]]],
    svd_rank: np.ndarray,
    N_m: int,
) -> np.ndarray:
    """OrthoFinder-proxy: ortholog grouping + annotation fallback.

    Simulates OrthoFinder behavior: assigns sequence-based orthologs
    first, then falls back to annotation similarity for non-orthologs.
    This represents the best a sequence+annotation hybrid could do.
    """
    ens_rank = compute_ensembl_rank(hi, orthologs, N_m)

    # Combine: orthologs get a boost, others use SVD
    scores = np.zeros(N_m, dtype=np.float64)
    for j in range(N_m):
        # Sequence ortholog bonus
        if ens_rank[j] < N_m:
            # Orthologs get a large bonus
            scores[j] += 100.0 / ens_rank[j]
        # SVD annotation score
        # SVD rank 1 = best → score = 1/1 = 1.0
        scores[j] += 1.0 / (60 + svd_rank[j])

    return arr_to_rank(scores, N_m)


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison logic
# ─────────────────────────────────────────────────────────────────────────────

def run_bench6(
    shared: SharedData,
    bench1_cases: list,
    model_results: Dict[str, dict] = None,
    output_dir: str = None,
) -> dict:
    """Run external baseline comparison on bench1 cases.

    Args:
        shared: Loaded SharedData.
        bench1_cases: List of bench1 case dicts with h_sym, m_sym, etc.
        model_results: Pre-computed model results {variant: summary_dict}
        output_dir: Where to save results.

    Returns:
        Comparison summary dict.
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sd = shared
    root = sd.root

    logger.info(f"\n{'=' * 80}")
    logger.info("BENCHMARK 6: External Method Comparison")
    logger.info(f"{'=' * 80}\n")

    # 1. Load Ensembl orthologs
    logger.info("Loading Ensembl orthologs...")
    orthologs = load_ensembl_orthologs(
        str(root / "dataset" / "external" / "ensembl_human_mouse_orthologs.tsv"),
        sd.h_genes, sd.m_genes,
    )

    # 2. Evaluate each baseline on bench1 cases
    baselines = {
        "Ensembl_Compara": [],
        "SVD_Annotation": [],
        "2hop_Overlap": [],
        "OrthoFinder_proxy": [],
    }
    case_details = []

    logger.info(f"Evaluating {len(bench1_cases)} bench1 cases...")
    for case in bench1_cases:
        # Support both formats: bench1 uses "sym"/"t_sym" or "h"/"t"
        h_sym = case.get("sym", case.get("human_symbol", ""))
        m_sym = case.get("t_sym", case.get("mouse_symbol", ""))

        ensg = case.get("h", sd.sym_to_ensg.get(h_sym))
        ensmusg = case.get("t", sd.sym_to_ensmusg.get(m_sym))
        if ensg is None or ensmusg is None:
            continue
        if ensg not in sd.h_genes or ensmusg not in sd.m_genes:
            continue

        hi = sd.h_genes[ensg]
        mi = sd.m_genes[ensmusg]
        N_m = sd.N_m

        # Ensembl Compara
        ens_rank = compute_ensembl_rank(hi, orthologs, N_m)
        baselines["Ensembl_Compara"].append(int(ens_rank[mi]))

        # SVD Annotation
        svd_rank = compute_svd_best_rank(hi, sd)
        baselines["SVD_Annotation"].append(int(svd_rank[mi]))

        # 2-hop Overlap
        hop_rank = compute_2hop_best_rank(hi, sd)
        baselines["2hop_Overlap"].append(int(hop_rank[mi]))

        # OrthoFinder proxy
        of_rank = compute_orthofinder_proxy_rank(
            hi, orthologs, svd_rank, N_m
        )
        baselines["OrthoFinder_proxy"].append(int(of_rank[mi]))

        case_details.append({
            "case": f"{h_sym}->{m_sym}",
            "type": case.get("type", "unknown"),
            "Ensembl_Compara": int(ens_rank[mi]),
            "SVD_Annotation": int(svd_rank[mi]),
            "2hop_Overlap": int(hop_rank[mi]),
            "OrthoFinder_proxy": int(of_rank[mi]),
        })

    # 3. Compute metrics for each baseline
    baseline_metrics = {}
    for name, ranks in baselines.items():
        if ranks:
            baseline_metrics[name] = compute_all_metrics(ranks)

    # 4. Build comparison table
    all_methods = {}
    for name, metrics in baseline_metrics.items():
        all_methods[name] = metrics

    # Add model results if provided
    if model_results:
        for variant, result in model_results.items():
            if isinstance(result, dict) and "primary_metrics" in result:
                all_methods[variant] = result["primary_metrics"]

    # 5. Log comparison table
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPARISON TABLE")
    logger.info(f"{'=' * 80}")
    header = (
        f"{'Method':<25} {'R@5':>6} {'R@10':>6} {'R@20':>6} "
        f"{'R@50':>6} {'R@100':>6} {'MRR':>8} {'MedR':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for name, m in all_methods.items():
        logger.info(
            f"{name:<25} "
            f"{m.get('recall@5', 0):.1%}  "
            f"{m.get('recall@10', 0):.1%}  "
            f"{m.get('recall@20', 0):.1%}  "
            f"{m.get('recall@50', 0):.1%}  "
            f"{m.get('recall@100', 0):.1%}  "
            f"{m.get('mrr', 0):.4f}  "
            f"{m.get('median_rank', 0):.0f}"
        )

    # 6. Per-type breakdown for key baselines
    types = defaultdict(lambda: defaultdict(list))
    for detail in case_details:
        t = detail["type"]
        for method in baselines:
            types[t][method].append(detail[method])

    logger.info(f"\n  PER-TYPE BREAKDOWN:")
    for t in sorted(types):
        logger.info(f"\n  Type: {t}")
        for method in baselines:
            ranks = types[t][method]
            if ranks:
                m = compute_all_metrics(ranks)
                logger.info(
                    f"    {method:<25} R@5={m['recall@5']:.1%} "
                    f"R@10={m['recall@10']:.1%} "
                    f"R@20={m['recall@20']:.1%} MRR={m['mrr']:.4f}"
                )

    # 7. Save results
    summary = {
        "n_cases": len(bench1_cases),
        "n_evaluated": len(case_details),
        "baseline_metrics": baseline_metrics,
        "comparison": {
            name: {
                k: v for k, v in m.items()
                if isinstance(v, (int, float))
            }
            for name, m in all_methods.items()
        },
        "case_details": case_details,
    }

    out_path = Path(output_dir) / "bench6_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return summary


def main():
    """Standalone entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Bench6: External Method Comparison"
    )
    parser.add_argument("--skip-ppi", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    from benchmarks.bench1_literature import BENCH1_CASES

    shared = SharedData(skip_ppi=args.skip_ppi)

    # Load existing model results if available
    results_dir = Path(__file__).parent / "results"
    model_results = {}
    for variant in ["M1", "M2", "M3", "M4"]:
        p = results_dir / f"bench1_{variant}.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            if "summary" in data:
                model_results[variant] = data["summary"]

    run_bench6(
        shared, BENCH1_CASES,
        model_results=model_results,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
