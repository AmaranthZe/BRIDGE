"""Train with Issue #4 fix: edge weights in HGT message passing.

Based on Run 3 (best results) — HGT gradient steps disabled, per-modality fusion.
New: edge_weight_blend=0.1 injects TF-IDF weights into HGT layers.
"""

import gc
import logging
import sys
import pickle
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_edge_weight.log"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

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

    cfg = Config()
    # Same as Run 3 (best) — anti-hub + per-modality fusion
    cfg.train.train_tau_col = 0.5
    cfg.train.batch_size = 256
    cfg.train.hgt_grad_start_epoch = 999  # HGT grad disabled (same as Run 3)
    cfg.train.epochs = 40
    # Edge weight blend (Issue #4 fix) — already default 0.1 in config
    # cfg.model.edge_weight_blend = 0.1

    proc = Path(cfg.paths.proc_dir)

    logger.info("Loading data...")
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

    set_seed(cfg.train.seed)
    model = create_model(cfg, node_counts)
    model.hgt.init_from_svd(Z_h, Z_m, sparse_matrices)

    del sparse_matrices, split_info
    gc.collect()
    logger.info("Freed sparse matrices from memory")

    trainer = Trainer(
        model=model, graph_data=graph,
        Z_h=Z_h, Z_m=Z_m, indexers=indexers,
        cfg=cfg, device=device,
    )

    del Z_h, Z_m, graph
    gc.collect()
    logger.info("Freed numpy arrays and graph from memory")

    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
