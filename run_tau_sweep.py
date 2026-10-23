"""Train + eval with different train_tau_col values.

Sequentially runs training and evaluation for each tau_col value,
saving results to separate directories for comparison.
"""

import logging
import sys
import pickle
import json
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tau_sweep.log"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.data.graph import load_unified_graph
from v2_redesign.model.model import create_model
from v2_redesign.training.trainer import Trainer, set_seed


TAU_COL_VALUES = [0.3, 0.5]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    cfg = Config()
    proc = Path(cfg.paths.proc_dir)

    # Load data once (shared across runs)
    logger.info("Loading data (shared)...")
    with open(proc / "indexers.pkl", "rb") as f:
        indexers = pickle.load(f)
    Z_h, Z_m, _, split_info = load_svd_artifacts(str(proc))
    sparse_matrices = load_sparse_matrices(str(proc))
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }

    for tau_col in TAU_COL_VALUES:
        logger.info("=" * 70)
        logger.info(f"SWEEP: train_tau_col = {tau_col}")
        logger.info("=" * 70)

        # Override train_tau_col + reduce memory for constrained GPU
        cfg.train.train_tau_col = tau_col
        cfg.train.batch_size = 128
        set_seed(cfg.train.seed)

        # Create fresh model
        model = create_model(cfg, node_counts)
        model.hgt.init_from_svd(Z_h, Z_m, sparse_matrices)

        # Train
        trainer = Trainer(
            model=model, graph_data=graph,
            Z_h=Z_h, Z_m=Z_m, indexers=indexers,
            cfg=cfg, device=device,
        )
        trainer.train()

        # Save checkpoint to named directory
        run_dir = proc / "runs" / f"tau_col_{tau_col}"
        run_dir.mkdir(parents=True, exist_ok=True)
        src_best = proc / "runs" / "best_model.pt"
        if src_best.exists():
            shutil.copy2(src_best, run_dir / "best_model.pt")
            logger.info(f"Saved best model to {run_dir}")

        # Free GPU memory before next run
        del trainer, model
        torch.cuda.empty_cache()

        logger.info(f"SWEEP: train_tau_col={tau_col} complete\n")

    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE — run eval on each checkpoint:")
    for tau_col in TAU_COL_VALUES:
        logger.info(f"  python run_v2_eval.py --checkpoint ../tau_col_{tau_col}/best_model.pt")


if __name__ == "__main__":
    main()
