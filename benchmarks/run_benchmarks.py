"""Entry point for running benchmark evaluations.

Usage examples:
    python benchmarks/run_benchmarks.py --variant M1 --benchmark bench1
    python benchmarks/run_benchmarks.py --variant all --benchmark all
    python benchmarks/run_benchmarks.py --variant M4 --benchmark bench2 --sample 200
    python benchmarks/run_benchmarks.py --variant M3 --benchmark bench5 --sample 100 --skip-ppi
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from benchmarks.model_variants import SharedData, ModelEvaluator
from benchmarks.bench1_literature import run_bench1
from benchmarks.bench2_phenotype import run_bench2
from benchmarks.bench3_essentiality import run_bench3
from benchmarks.bench5_hmd_ranking import run_bench5

logger = logging.getLogger(__name__)

ALL_VARIANTS = ["M1", "M2", "M3", "M4"]
ALL_BENCHMARKS = ["bench1", "bench2", "bench3", "bench5"]


def run_single(
    variant: str,
    benchmark: str,
    shared: SharedData,
    device: str,
    sample: int = None,
    output_dir: str = None,
) -> dict:
    """Run a single (variant, benchmark) combination.

    Loads the model, runs the benchmark, frees memory, returns results.
    """
    logger.info(f"\n{'#' * 80}")
    logger.info(f"# Running: {benchmark} with variant {variant}")
    logger.info(f"{'#' * 80}\n")

    evaluator = ModelEvaluator(variant, shared, device=device)

    try:
        if benchmark == "bench1":
            result = run_bench1(evaluator, output_dir=output_dir)
        elif benchmark == "bench2":
            result = run_bench2(
                evaluator, output_dir=output_dir, sample=sample
            )
        elif benchmark == "bench3":
            result = run_bench3(
                evaluator, output_dir=output_dir, sample=sample
            )
        elif benchmark == "bench5":
            result = run_bench5(
                evaluator, output_dir=output_dir, sample=sample
            )
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    finally:
        evaluator.free()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Species Gene Mapping Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model variants:
    M1: v2 HGT (12 edges) -> FGW only
    M2: v3 2-hop HGT (18 edges) -> FGW only
    M3: v2 HGT -> FGW -> 22-strategy RRF ensemble
    M4: v3 2-hop HGT -> FGW -> 22-strategy RRF ensemble

Benchmarks:
    bench1: Literature-curated non-ortholog recovery (46 cases)
    bench2: MGI Phenotype cross-species transfer
    bench3: Cross-species essentiality conservation (DepMap + IMPC)
    bench5: Genome-wide HMD ranking (large-scale)

Examples:
    python benchmarks/run_benchmarks.py --variant M1 --benchmark bench1
    python benchmarks/run_benchmarks.py --variant all --benchmark all
    python benchmarks/run_benchmarks.py --variant M4 --benchmark bench2 --sample 200
    python benchmarks/run_benchmarks.py --variant M3 --benchmark bench5 --sample 100 --skip-ppi
        """,
    )
    parser.add_argument(
        "--variant",
        default="M1",
        help="Model variant: M1, M2, M3, M4, or 'all' (default: M1)",
    )
    parser.add_argument(
        "--benchmark",
        default="bench1",
        help="Benchmark to run: bench1, bench2, bench5, or 'all' (default: bench1)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--skip-ppi",
        action="store_true",
        help="Skip STRING PPI strategies (avoids API calls for large runs)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N human genes for bench2/bench5 quick testing",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results JSON (default: benchmarks/results/)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"

    # Resolve variants and benchmarks
    variants = ALL_VARIANTS if args.variant.lower() == "all" else [args.variant]
    benchmarks = ALL_BENCHMARKS if args.benchmark.lower() == "all" else [args.benchmark]

    # Validate
    for v in variants:
        if v not in ALL_VARIANTS:
            parser.error(f"Unknown variant: {v}")
    for b in benchmarks:
        if b not in ALL_BENCHMARKS:
            parser.error(f"Unknown benchmark: {b}")

    # Output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load shared data ONCE (variant-independent)
    logger.info("=" * 80)
    logger.info("Loading shared data assets (one-time)...")
    logger.info("=" * 80)
    t0 = time.time()
    shared = SharedData(skip_ppi=args.skip_ppi)
    logger.info(f"Shared data loaded in {time.time() - t0:.1f}s\n")

    # Run all requested combinations
    all_results = {}
    total_t0 = time.time()

    for variant in variants:
        for benchmark in benchmarks:
            key = f"{benchmark}_{variant}"
            try:
                result = run_single(
                    variant=variant,
                    benchmark=benchmark,
                    shared=shared,
                    device=device,
                    sample=args.sample,
                    output_dir=output_dir,
                )
                all_results[key] = {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"FAILED: {key}: {e}", exc_info=True)
                all_results[key] = {"status": "error", "error": str(e)}

    total_elapsed = time.time() - total_t0

    # Final summary
    logger.info(f"\n{'=' * 80}")
    logger.info("ALL BENCHMARKS COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"Results directory: {output_dir}")
    logger.info(f"\nCompleted runs:")
    for key, res in all_results.items():
        status = res['status']
        logger.info(f"  {key}: {status}")

    # Save combined summary
    summary_path = Path(output_dir) / "benchmark_summary.json"
    # Extract just the summary/metric portions (not full pair data)
    summary_data = {}
    for key, res in all_results.items():
        if res['status'] == 'success':
            result = res['result']
            if isinstance(result, dict):
                # Remove large pair lists from summary
                cleaned = {
                    k: v for k, v in result.items()
                    if k not in ('cases', 'pairs')
                }
                summary_data[key] = cleaned
            else:
                summary_data[key] = str(result)
        else:
            summary_data[key] = {"error": res['error']}

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Combined summary saved to {summary_path}")


if __name__ == "__main__":
    main()
