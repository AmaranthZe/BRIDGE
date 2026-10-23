"""Train with larger candidate pool (topc=256) + FGW hyperparameter tuning.

Based on Run 3 (best results so far):
- Per-modality fusion, no HGT gradient steps
- train_tau_col=0.5, col_entropy_weight=0.1

Changes:
- topc: 128 -> 256 (double candidate pool, ortholog-in-candidates ~43% -> hopefully higher)
- alpha: 0.5 -> 0.7 (increase feature cost weight — structure cost GW is very low anyway)
- epsilon: 0.05 -> 0.02 (less entropic smoothing -> sharper transport plan)
- edge_weight_blend: 0.1 -> 0.0 (disable edge weight blend — Run 5 showed no benefit)
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
        logging.FileHandler("train_topc_fgw.log"),
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

    # Same as Run 3 (best)
    cfg.train.train_tau_col = 0.5
    cfg.train.batch_size = 256
    cfg.train.hgt_grad_start_epoch = 999  # HGT grad disabled
    cfg.train.epochs = 40

    # Candidate pool: 128 -> 256
    cfg.candidates.topc = 256
    logger.info(f"Candidate pool: topc={cfg.candidates.topc}")

    # FGW tuning
    cfg.fgw.alpha = 0.7       # more weight on feature cost (annotation overlap)
    cfg.fgw.epsilon = 0.02    # sharper transport plan
    logger.info(f"FGW: alpha={cfg.fgw.alpha}, epsilon={cfg.fgw.epsilon}")

    # Disable edge weight blend (no benefit in Run 5)
    cfg.model.edge_weight_blend = 0.0

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
