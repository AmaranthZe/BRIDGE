"""Full training pipeline for v2_redesign."""

import logging
import sys
import pickle
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("v2_train.log"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from v2_redesign.config import Config
from v2_redesign.data.io_schema import load_all_data
from v2_redesign.data.sparse import build_all_sparse_matrices, save_sparse_matrices
from v2_redesign.data.svd import build_unified_svd, save_svd_artifacts
from v2_redesign.data.graph import build_unified_graph, save_unified_graph
from v2_redesign.model.model import create_model
from v2_redesign.training.trainer import Trainer, set_seed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load config
    cfg = Config()  # Use defaults
    proc = Path(cfg.paths.proc_dir)
    proc.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.train.seed)

    # ── Step 1: Load raw data ──
    logger.info("=" * 60)
    logger.info("Step 1: Loading raw data...")
    dataframes, indexers = load_all_data(cfg)

    # Save indexers
    with open(proc / "indexers.pkl", "wb") as f:
        pickle.dump(indexers, f)
    logger.info(f"Saved indexers to {proc / 'indexers.pkl'}")

    # ── Step 2: Build sparse matrices ──
    logger.info("=" * 60)
    logger.info("Step 2: Building sparse matrices...")
    matrices = build_all_sparse_matrices(dataframes, indexers, cfg)
    save_sparse_matrices(matrices, str(proc))

    # ── Step 3: Unified SVD ──
    logger.info("=" * 60)
    logger.info("Step 3: Building unified SVD...")
    Z_h, Z_m, svd_model, split_info = build_unified_svd(
        matrices,
        svd_dim=cfg.preproc.svd_dim,
        heldout_ratio=cfg.preproc.heldout_ratio,
        seed=cfg.train.seed,
    )
    save_svd_artifacts(Z_h, Z_m, svd_model, split_info, str(proc))

    # ── Step 4: Build heterogeneous graph ──
    logger.info("=" * 60)
    logger.info("Step 4: Building unified heterogeneous graph...")
    graph = build_unified_graph(matrices, indexers)
    save_unified_graph(graph, str(proc / "unified_graph.pt"))

    # ── Step 5: Create model ──
    logger.info("=" * 60)
    logger.info("Step 5: Creating model...")
    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }
    model = create_model(cfg, node_counts)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    for name, count in node_counts.items():
        logger.info(f"  {name}: {count:,} nodes")

    # ── Step 6: Train ──
    logger.info("=" * 60)
    logger.info("Step 6: Starting training...")
    trainer = Trainer(
        model=model,
        graph_data=graph,
        Z_h=Z_h,
        Z_m=Z_m,
        indexers=indexers,
        cfg=cfg,
        device=device,
    )
    trainer.train()

    logger.info("=" * 60)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
