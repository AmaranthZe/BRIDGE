"""Run remaining benchmark evaluations sequentially (single SharedData load)."""
import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from benchmarks.model_variants import SharedData, ModelEvaluator
from benchmarks.bench3_essentiality import run_bench3
from benchmarks.bench5_hmd_ranking import run_bench5

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    output_dir = str(Path(__file__).parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Single SharedData load
    logger.info("Loading shared data...")
    t0 = time.time()
    shared = SharedData(skip_ppi=True)
    logger.info(f"Shared data loaded in {time.time() - t0:.1f}s")

    # Jobs to run: (variant, benchmark_name, run_func, kwargs)
    jobs = []

    # bench3 for M2, M3, M4 (M1 is running in background)
    for v in ["M2", "M3", "M4"]:
        jobs.append((v, "bench3", run_bench3, {"sample": 500}))

    # bench5 for all variants with sample=300
    for v in ["M1", "M2", "M3", "M4"]:
        jobs.append((v, "bench5", run_bench5, {"sample": 300}))

    for variant, bench_name, run_func, kwargs in jobs:
        logger.info(f"\n{'#' * 80}")
        logger.info(f"# Running {bench_name} with variant {variant}")
        logger.info(f"{'#' * 80}\n")

        evaluator = ModelEvaluator(variant, shared, device="cpu")
        try:
            result = run_func(evaluator, output_dir=output_dir, **kwargs)
            logger.info(f"COMPLETED: {bench_name}_{variant}")
        except Exception as e:
            logger.error(f"FAILED: {bench_name}_{variant}: {e}", exc_info=True)
        finally:
            evaluator.free()
            gc.collect()

    logger.info("\nAll remaining benchmarks complete!")


if __name__ == "__main__":
    main()
