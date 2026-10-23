#!/usr/bin/env python3
"""Diagnose where embedding collapse happens in the pipeline."""
import sys, pickle, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F

from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.data.graph import load_unified_graph
from v2_redesign.model.model import create_model

cfg = Config()
proc = Path(cfg.paths.proc_dir)

with open(proc / "indexers.pkl", "rb") as f:
    indexers = pickle.load(f)
Z_h, Z_m, _, _ = load_svd_artifacts(str(proc))
sparse_matrices = load_sparse_matrices(str(proc))
graph = load_unified_graph(str(proc / "unified_graph.pt"))

node_counts = {
    "Human_Gene": len(indexers["human_genes"]),
    "Mouse_Gene": len(indexers["mouse_genes"]),
    "BP": len(indexers["bp_terms"]),
    "MF": len(indexers["mf_terms"]),
    "Disease": len(indexers["diseases"]),
}
model = create_model(cfg, node_counts)
ckpt = torch.load(proc / "runs" / "best_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

edge_index_dict = {}
for et in graph.edge_types:
    edge_index_dict[et] = graph[et].edge_index

def random_cos_stats(emb, n=500, label=""):
    """Check cosine distribution for random pairs."""
    rng = np.random.RandomState(42)
    idx = rng.choice(emb.shape[0], min(2*n, emb.shape[0]), replace=False)
    a, b = emb[idx[:n]], emb[idx[n:2*n]]
    a_n = F.normalize(a, p=2, dim=-1, eps=1e-8)
    b_n = F.normalize(b, p=2, dim=-1, eps=1e-8)
    cos = (a_n * b_n).sum(dim=-1)
    logger.info(f"  {label}: cos mean={cos.mean():.4f}, std={cos.std():.4f}, min={cos.min():.4f}, max={cos.max():.4f}")

# Check the learned skip gate value
alpha = torch.sigmoid(model.hgt.skip_gate).item()
logger.info(f"\n=== Skip gate alpha (after training): {alpha:.4f} ===")
logger.info(f"    (0.0 = pure HGT, 1.0 = pure SVD init)")

with torch.no_grad():
    # Stage 0: Initial SVD embeddings (raw)
    logger.info("\n--- Stage 0: Raw SVD embeddings (from numpy) ---")
    Z_h_t = torch.tensor(Z_h, dtype=torch.float32)
    Z_m_t = torch.tensor(Z_m, dtype=torch.float32)
    random_cos_stats(Z_h_t, label="Human SVD")
    random_cos_stats(Z_m_t, label="Mouse SVD")

    # Stage 1: Learned embeddings (x0 = embedding weights)
    logger.info("\n--- Stage 1: HGT learned embeddings (x0, before message passing) ---")
    x0_h = model.hgt.embeddings["Human_Gene"].weight
    x0_m = model.hgt.embeddings["Mouse_Gene"].weight
    random_cos_stats(x0_h, label="Human x0")
    random_cos_stats(x0_m, label="Mouse x0")

    # Stage 2: After HGT message passing (before skip gate)
    logger.info("\n--- Stage 2: After HGT layers (before skip gate) ---")
    x_dict = {nt: model.hgt.embeddings[nt].weight for nt in model.hgt.node_types}
    for layer in model.hgt.layers:
        x_dict = model.hgt._layer_forward(x_dict, edge_index_dict, layer)
    random_cos_stats(x_dict["Human_Gene"], label="Human post-HGT")
    random_cos_stats(x_dict["Mouse_Gene"], label="Mouse post-HGT")

    # Stage 3: After skip gate (= final HGT output, no out_proj)
    logger.info("\n--- Stage 3: After skip gate (final HGT output) ---")
    x_skip = {
        nt: alpha * model.hgt.embeddings[nt].weight + (1 - alpha) * x_dict[nt]
        for nt in model.hgt.node_types
    }
    random_cos_stats(x_skip["Human_Gene"], label="Human skip-gated")
    random_cos_stats(x_skip["Mouse_Gene"], label="Mouse skip-gated")

    # Ortholog check at each stage
    mi = indexers["mouse_genes"]["ENSMUSG00000059552"]  # Trp53
    hi = indexers["human_genes"]["ENSG00000141510"]     # TP53
    logger.info("\n--- Trp53→TP53 cosine at each stage ---")
    for label, h_emb, m_emb in [
        ("SVD raw", Z_h_t, Z_m_t),
        ("x0 (learned emb)", x0_h, x0_m),
        ("post-HGT", x_dict["Human_Gene"], x_dict["Mouse_Gene"]),
        ("skip-gated", x_skip["Human_Gene"], x_skip["Mouse_Gene"]),
    ]:
        h = F.normalize(h_emb[hi:hi+1], p=2, dim=-1, eps=1e-8)
        m = F.normalize(m_emb[mi:mi+1], p=2, dim=-1, eps=1e-8)
        cos = (h * m).sum().item()

        # Rank
        sims = (m @ F.normalize(h_emb, p=2, dim=-1, eps=1e-8).t()).squeeze()
        rank = (sims > sims[hi]).sum().item() + 1
        logger.info(f"  {label}: cos={cos:.4f}, rank={rank}/{h_emb.shape[0]}")
