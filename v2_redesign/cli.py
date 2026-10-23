"""CLI for Cross-Species Functional Gene Equivalence Mapping v2."""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_preprocess(args):
    """Preprocess: load data, build sparse matrices, SVD, graph."""
    from .config import Config
    from .data.io_schema import load_all_data
    from .data.sparse import build_all_sparse_matrices, save_sparse_matrices
    from .data.svd import build_unified_svd, save_svd_artifacts
    from .data.graph import build_unified_graph, save_unified_graph

    cfg = Config.from_yaml(args.config)
    proc = Path(cfg.paths.proc_dir)
    proc.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data...")
    dataframes, indexers = load_all_data(cfg)

    logger.info("Building sparse matrices...")
    matrices = build_all_sparse_matrices(dataframes, indexers, cfg)
    save_sparse_matrices(matrices, proc)

    logger.info("Building unified SVD...")
    Z_h, Z_m, svd_model, split_info = build_unified_svd(
        matrices, svd_dim=cfg.preproc.svd_dim,
        heldout_ratio=cfg.preproc.heldout_ratio, seed=cfg.train.seed,
    )
    save_svd_artifacts(Z_h, Z_m, svd_model, split_info, proc)

    logger.info("Building unified graph...")
    graph = build_unified_graph(matrices, indexers)
    save_unified_graph(graph, str(proc / "unified_graph.pt"))

    # Save indexers
    import pickle
    with open(proc / "indexers.pkl", "wb") as f:
        pickle.dump(indexers, f)

    logger.info("Preprocessing complete.")


def cmd_train(args):
    """Train the model."""
    import pickle
    import numpy as np
    import torch

    from .config import Config
    from .data.sparse import load_sparse_matrices
    from .data.svd import load_svd_artifacts
    from .data.graph import load_unified_graph
    from .model.model import create_model
    from .training.trainer import Trainer, set_seed

    cfg = Config.from_yaml(args.config)
    proc = Path(cfg.paths.proc_dir)
    device = args.device

    set_seed(cfg.train.seed)

    # Load preprocessed data
    with open(proc / "indexers.pkl", "rb") as f:
        indexers = pickle.load(f)
    Z_h, Z_m, _, _ = load_svd_artifacts(proc)
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    # Create model
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

    # Train
    trainer = Trainer(
        model=model, graph_data=graph,
        Z_h=Z_h, Z_m=Z_m, indexers=indexers,
        cfg=cfg, device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


def cmd_infer(args):
    """Run inference."""
    import pickle
    import torch

    from .config import Config
    from .data.sparse import load_sparse_matrices
    from .data.svd import load_svd_artifacts
    from .data.graph import load_unified_graph
    from .model.model import create_model
    from .inference.inferencer import Inferencer

    cfg = Config.from_yaml(args.config)
    proc = Path(cfg.paths.proc_dir)
    device = args.device

    with open(proc / "indexers.pkl", "rb") as f:
        indexers = pickle.load(f)
    sparse_matrices = load_sparse_matrices(proc)
    Z_h, Z_m, _, _ = load_svd_artifacts(proc)
    graph = load_unified_graph(str(proc / "unified_graph.pt"))

    node_counts = {
        "Human_Gene": len(indexers["human_genes"]),
        "Mouse_Gene": len(indexers["mouse_genes"]),
        "BP": len(indexers["bp_terms"]),
        "MF": len(indexers["mf_terms"]),
        "Disease": len(indexers["diseases"]),
    }
    model = create_model(cfg, node_counts)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Register per-modality buffers from checkpoint before loading
    state = ckpt["model_state_dict"]
    for key in ["Z_h_BP", "Z_m_BP", "Z_h_MF", "Z_m_MF", "Z_h_Disease", "Z_m_Disease"]:
        full_key = f"hgt.{key}"
        if full_key in state and not hasattr(model.hgt, key):
            model.hgt.register_buffer(key, state[full_key])
    model.load_state_dict(state, strict=False)

    inferencer = Inferencer(
        model=model, graph_data=graph, sparse_matrices=sparse_matrices,
        Z_h=Z_h, Z_m=Z_m, indexers=indexers, cfg=cfg, device=device,
    )

    result = inferencer.infer(args.query, direction=args.direction)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Species Functional Gene Equivalence Mapping v2")
    parser.add_argument("--config", default="v2_redesign/configs/default.yaml")
    sub = parser.add_subparsers(dest="command")

    # Preprocess
    p_pre = sub.add_parser("preprocess", help="Preprocess raw data")

    # Train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--device", default="cuda")
    p_train.add_argument("--resume", default=None, help="Resume from checkpoint")

    # Infer
    p_infer = sub.add_parser("infer", help="Run inference")
    p_infer.add_argument("query", help="Semicolon-separated gene IDs")
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--direction", default="auto",
                         choices=["auto", "human_to_mouse", "mouse_to_human"])
    p_infer.add_argument("--device", default="cuda")
    p_infer.add_argument("--output", default=None, help="Output JSON path")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    {"preprocess": cmd_preprocess, "train": cmd_train, "infer": cmd_infer}[args.command](args)


if __name__ == "__main__":
    main()
