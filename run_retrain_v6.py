"""Retrain v6: Unified model with architectural improvements.

Changes from v5:
- P1: Additive residual HGT (skip_gate replaced with per-node-type residual_scales)
- P2: Larger candidate pool (topc=256, max_candidates=2048)
- P3: SVD-weighted candidate scoring (0.7 SVD + 0.3 co-attention)
- P4: Moderate tau_col (train=0.3, inference=0.15)
- batch_size=192 (reduced for larger candidate pool VRAM)
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
        logging.FileHandler("train_retrain_v6.log"),
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
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = Config()
    # Config already has v6 defaults from updated config.py/default.yaml:
    # - candidates.topc=256, max_candidates=2048, svd_score_weight=0.7
    # - train.train_tau_col=0.3, residual_scale_init=-2.2
    # - fgw.tau_col=0.15

    cfg.train.epochs = 80
    cfg.train.batch_size = 192          # Reduced for larger candidate pool VRAM
    cfg.train.hgt_grad_start_epoch = 999  # Freeze HGT backbone
    cfg.train.save_every = 5

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

    # Log the new architecture
    for nt in model.hgt.node_types:
        scale = torch.sigmoid(model.hgt.residual_scales[nt]).item()
        logger.info(f"  {nt} residual_scale init: {scale:.4f}")
    logger.info(f"  Candidate topc={cfg.candidates.topc}, max={cfg.candidates.max_candidates}")
    logger.info(f"  SVD score weight={cfg.candidates.svd_score_weight}")
    logger.info(f"  train_tau_col={cfg.train.train_tau_col}, infer_tau_col={cfg.fgw.tau_col}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    del sparse_matrices, split_info
    gc.collect()

    trainer = Trainer(
        model=model, graph_data=graph,
        Z_h=Z_h, Z_m=Z_m, indexers=indexers,
        cfg=cfg, device=device,
    )

    trainer.log_dir = Path("data/processed/runs/retrain_v6")
    trainer.log_dir.mkdir(parents=True, exist_ok=True)

    del Z_h, Z_m, graph
    gc.collect()

    logger.info("Starting v6 training...")
    trainer.train()
    logger.info("Retrain v6 complete!")


if __name__ == "__main__":
    main()
