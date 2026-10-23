"""Experiment B: Skip gate loosened + longer training.

skip_gate_init = 0.0 → sigmoid(0.0) = 0.5 (HGT contributes 50% vs 20%).
80 epochs with cosine LR schedule (already default in trainer).
Based on Run 3 config (the best model).
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
        logging.FileHandler("train_exp_b.log"),
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
    # ── Run 3 base config ──
    cfg.train.train_tau_col = 0.5
    cfg.train.batch_size = 256
    cfg.train.hgt_grad_start_epoch = 999

    # ── Experiment B: skip gate + longer training ──
    cfg.train.epochs = 80  # 2x longer

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

    # Override skip gate: sigmoid(0.0) = 0.5 → HGT contributes 50%
    model.hgt.skip_gate.data.fill_(0.0)
    logger.info(f"Skip gate set to 0.0 → sigmoid = 0.5 (50% HGT contribution)")

    del sparse_matrices, split_info
    gc.collect()

    trainer = Trainer(
        model=model, graph_data=graph,
        Z_h=Z_h, Z_m=Z_m, indexers=indexers,
        cfg=cfg, device=device,
    )

    # Save to experiment-specific directory
    trainer.log_dir = Path("data/processed/runs/exp_b")
    trainer.log_dir.mkdir(parents=True, exist_ok=True)

    del Z_h, Z_m, graph
    gc.collect()

    trainer.train()
    logger.info("Experiment B training complete!")


if __name__ == "__main__":
    main()
