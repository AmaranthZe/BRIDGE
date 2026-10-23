"""Training entry point for v3_crosshop.

Usage:
    python v3_crosshop/run_train_v3.py
    python v3_crosshop/run_train_v3.py --resume-v2 data/processed/runs/retrain_v6/best_model.pt
    python v3_crosshop/run_train_v3.py --rebuild-graph

Reuses v2's preprocessed data (sparse matrices, SVD, indexers).
Builds the cross-hop graph on-the-fly and caches to data/processed/v3_crosshop_graph.pt.
"""

import argparse
import gc
import logging
import pickle
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_v3_crosshop.log"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from v3_crosshop.config import V3Config
from v3_crosshop.data.graph import (
    build_crosshop_graph, save_crosshop_graph, load_crosshop_graph,
)
from v3_crosshop.model.model import create_model_v3
from v3_crosshop.training.trainer import TrainerV3
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.training.trainer import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train v3_crosshop model")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "v3_default.yaml"),
        help="Path to v3 config YAML",
    )
    parser.add_argument(
        "--resume-v2", default=None,
        help="Path to v2 checkpoint for warm-start (e.g. retrain_v6/best_model.pt)",
    )
    parser.add_argument(
        "--resume-v3", default=None,
        help="Path to v3 checkpoint to resume training",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--rebuild-graph", action="store_true",
        help="Force rebuild of cross-hop graph (ignore cached)",
    )
    args = parser.parse_args()

    cfg = V3Config.from_yaml(args.config)
    # Freeze HGT backbone — gradient checkpointing is incompatible with
    # cross-species edge types that change HGTConv's internal tensor shapes
    cfg.train.hgt_grad_start_epoch = 999
    proc = Path(cfg.paths.proc_dir)
    set_seed(cfg.train.seed)

    device = args.device
    logger.info(f"Device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")

    # ── Load preprocessed v2 data (no re-preprocessing) ──
    logger.info("Loading preprocessed v2 data...")
    with open(proc / "indexers.pkl", "rb") as f:
        indexers = pickle.load(f)
    sparse_matrices = load_sparse_matrices(str(proc))
    Z_h, Z_m, _, split_info = load_svd_artifacts(str(proc))

    # ── Build or load cross-hop graph ──
    graph_path = proc / "v3_crosshop_graph.pt"
    if graph_path.exists() and not args.rebuild_graph:
        logger.info(f"Loading cached v3 graph from {graph_path}")
        graph = load_crosshop_graph(str(graph_path))
    else:
        logger.info("Building v3 cross-hop graph...")
        graph = build_crosshop_graph(sparse_matrices, indexers, cfg.crosshop)
        save_crosshop_graph(graph, str(graph_path))

    # ── Create v3 model ──
    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }

    model = create_model_v3(cfg, node_counts)
    model.hgt.init_from_svd(Z_h, Z_m, sparse_matrices)

    # Log architecture
    for nt in model.hgt.node_types:
        scale = torch.sigmoid(model.hgt.residual_scales[nt]).item()
        logger.info(f"  {nt} residual_scale init: {scale:.4f}")
    logger.info(f"  Candidate topc={cfg.candidates.topc}, max={cfg.candidates.max_candidates}")
    logger.info(f"  SVD score weight={cfg.candidates.svd_score_weight}")
    logger.info(f"  Cross-hop k={cfg.crosshop.cross_hop_k}, "
                f"modalities={cfg.crosshop.modalities}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"v3 model parameters: {n_params:,}")

    del split_info
    gc.collect()

    # ── Set up trainer ──
    trainer = TrainerV3(
        model=model, graph_data=graph,
        Z_h=Z_h, Z_m=Z_m, indexers=indexers,
        cfg=cfg, device=device,
    )

    trainer.log_dir = Path("data/processed/runs/v3_crosshop")
    trainer.log_dir.mkdir(parents=True, exist_ok=True)

    del Z_h, Z_m, graph, sparse_matrices
    gc.collect()

    # ── Optional warm-start from v2 checkpoint ──
    if args.resume_v2:
        logger.info(f"Warm-starting from v2 checkpoint: {args.resume_v2}")
        trainer.load_checkpoint_v2(args.resume_v2)

    # ── Or resume interrupted v3 training ──
    if args.resume_v3:
        logger.info(f"Resuming v3 training from: {args.resume_v3}")
        ckpt = torch.load(args.resume_v3, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        for key in ["Z_h_BP", "Z_m_BP", "Z_h_MF", "Z_m_MF",
                     "Z_h_Disease", "Z_m_Disease"]:
            fk = f"hgt.{key}"
            if fk in state and not hasattr(trainer.model.hgt, key):
                trainer.model.hgt.register_buffer(key, state[fk])
        trainer.model.load_state_dict(state, strict=False)
        resume_epoch = ckpt.get("epoch", 0)
        remaining = cfg.train.epochs - resume_epoch
        if remaining > 0:
            trainer.epochs = remaining
            logger.info(f"Resumed from epoch {resume_epoch}, will train {remaining} more epochs (to {cfg.train.epochs})")
        else:
            logger.info(f"Checkpoint epoch {resume_epoch} >= target {cfg.train.epochs}, nothing to train")
            return

    # ── Train ──
    logger.info("Starting v3_crosshop training...")
    trainer.train()
    logger.info("v3_crosshop training complete!")


if __name__ == "__main__":
    main()
