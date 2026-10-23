"""Retrain v5: Informed by Tier A+B validation analysis.

Key findings from validation:
- fusionfix_run3 excels at ABCD1/RB1/TERT (skip_gate default=1.39, shorter training)
- exp_b/a/c/default excel at SERPINA1/CXCL8/MMP1 (higher train_tau_col=0.5)
- No single checkpoint covers all cases

Strategy: Combine best configs:
- train_tau_col=0.5 (from exp_b, proven for anti-hub)
- 80 epochs (longer training helps)
- skip_gate_init=0.5 (compromise: sigmoid(0.5)=0.62, moderate HGT contribution)
- Save checkpoints every 2 epochs for fine-grained selection
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
        logging.FileHandler("train_retrain_v5.log"),
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
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    cfg = Config()

    # ── Optimized config based on validation analysis ──
    cfg.train.train_tau_col = 0.5       # Anti-hub (from exp_b)
    cfg.train.batch_size = 256          # Standard
    cfg.train.epochs = 80               # Longer training
    cfg.train.hgt_grad_start_epoch = 999  # Freeze HGT (saves memory)
    cfg.train.save_every = 2            # Fine-grained checkpoint selection
    cfg.fgw.tau_col = 0.05             # Inference tau_col (anti-hub at eval)

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

    # Skip gate: sigmoid(0.5) = 0.62 (moderate HGT contribution)
    model.hgt.skip_gate.data.fill_(0.5)
    logger.info(f"Skip gate set to 0.5 -> sigmoid = {torch.sigmoid(torch.tensor(0.5)).item():.3f}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    del sparse_matrices, split_info
    gc.collect()

    trainer = Trainer(
        model=model, graph_data=graph,
        Z_h=Z_h, Z_m=Z_m, indexers=indexers,
        cfg=cfg, device=device,
    )

    # Save to new directory
    trainer.log_dir = Path("data/processed/runs/retrain_v5")
    trainer.log_dir.mkdir(parents=True, exist_ok=True)

    del Z_h, Z_m, graph
    gc.collect()

    logger.info("Starting training...")
    trainer.train()
    logger.info("Retrain v5 complete!")


if __name__ == "__main__":
    main()
