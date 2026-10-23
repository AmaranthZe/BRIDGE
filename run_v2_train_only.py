"""Training-only script — loads preprocessed data and trains.

Assumes data/processed/ already has: indexers.pkl, Z_h.npy, Z_m.npy,
unified_graph.pt (from a prior preprocessing run).
"""

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

import numpy as np
import torch

from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.data.graph import load_unified_graph
from v2_redesign.model.model import create_model
from v2_redesign.training.trainer import Trainer, set_seed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"GPU Memory: {mem / 1e9:.1f} GB")

    cfg = Config()
    proc = Path(cfg.paths.proc_dir)

    set_seed(cfg.train.seed)

    # ── Load preprocessed data ──
    logger.info("Loading preprocessed data...")

    with open(proc / "indexers.pkl", "rb") as f:
        indexers = pickle.load(f)
    logger.info(f"Indexers: {', '.join(f'{k}: {len(v):,}' for k, v in indexers.items())}")

    Z_h, Z_m, _, _ = load_svd_artifacts(str(proc))
    logger.info(f"SVD: Z_h {Z_h.shape}, Z_m {Z_m.shape}")

    sparse_matrices = load_sparse_matrices(str(proc))
    logger.info(f"Sparse matrices loaded: {list(sparse_matrices.keys())}")

    graph = load_unified_graph(str(proc / "unified_graph.pt"))
    total_edges = sum(graph[et].edge_index.shape[1] for et in graph.edge_types)
    logger.info(f"Graph: {len(graph.node_types)} node types, {len(graph.edge_types)} edge types, {total_edges:,} edges")

    # ── Create model ──
    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }
    model = create_model(cfg, node_counts)

    # Initialize HGT node embeddings from SVD features
    # This gives HGT a strong starting point: genes with similar functional
    # annotations start with similar embeddings, preventing collapse.
    model.hgt.init_from_svd(Z_h, Z_m, sparse_matrices)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ── Train ──
    logger.info("=" * 60)
    logger.info("Starting training...")
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
