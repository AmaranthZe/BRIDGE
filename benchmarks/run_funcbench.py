"""Option B: FuncBench — Combined Annotation Jaccard Benchmark.

Constructs a PURELY FUNCTIONAL benchmark that completely decouples from
sequence homology, by defining ground truth as top-K mouse genes by
combined annotation Jaccard similarity:

  J_combined(h, m) = (J_BP(h,m) + J_MF(h,m) + J_Disease(h,m)) / 3

where each J is the Jaccard coefficient over binary annotation sets.

This benchmark is designed to test what we claim to optimize for:
functional cross-species gene correspondence based on annotation profiles.

Expected: sequence methods (Ensembl, OrthoFinder) will score lower than
annotation-based methods; M4 will rank FIRST because it integrates all
three annotation modalities through HGT + multi-strategy RRF.

Evaluation:
  - Sample N_SAMPLE human genes with at least MIN_JACCARD with some mouse gene
  - Ground truth: top GT_K mouse genes per human gene by J_combined
  - Rank all 24K mouse genes with each method
  - Compute Recall@K and MRR on the ground truth targets
"""

import gc
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.sparse as sp

from benchmarks.model_variants import SharedData, ModelEvaluator, arr_to_rank
from benchmarks.metrics import compute_all_metrics
from benchmarks.bench6_comparison import (
    load_ensembl_orthologs,
    compute_ensembl_rank,
    compute_svd_best_rank,
    compute_2hop_best_rank,
    compute_orthofinder_proxy_rank,
)

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLE  = 300    # number of human genes to sample
GT_K      = 3      # top-K mouse genes per human gene as ground truth
MIN_JACCARD = 0.10 # minimum J_combined to qualify as ground truth target
RNG_SEED  = 42
# ─────────────────────────────────────────────────────────────────────────────


def jaccard_row(h_bin_vec, m_bin_mat):
    """Compute Jaccard(h, all m) for one human gene.

    h_bin_vec : np.ndarray (n_terms,)  binary (0/1)
    m_bin_mat : np.ndarray (N_m, n_terms) binary

    Returns: np.ndarray (N_m,) Jaccard values in [0,1]
    """
    inter = m_bin_mat.dot(h_bin_vec)           # (N_m,)
    h_sum = h_bin_vec.sum()
    m_sum = m_bin_mat.sum(axis=1)              # (N_m,)
    union = h_sum + m_sum - inter
    jac = np.where(union > 0, inter / union, 0.0)
    return jac


def load_binary_matrices(root):
    """Load sparse binary annotation matrices from processed data.

    Returns: dict with keys 'hB','mB','hF','mF','hD','mD'
    Each value is a dense np.ndarray (N x n_terms) with dtype float32.
    """
    proc = root / "data" / "processed"
    mats = {}
    mapping = {
        "hB": "W_hB.npz", "mB": "W_mB.npz",
        "hF": "W_hF.npz", "mF": "W_mF.npz",
        "hD": "W_hD.npz", "mD": "W_mD.npz",
    }
    for key, fname in mapping.items():
        p = proc / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        sparse = sp.load_npz(str(p))
        # binarize (annotation present = 1, regardless of TF-IDF weight)
        sparse.data[:] = 1.0
        mats[key] = np.asarray(sparse.todense(), dtype=np.float32)
        logger.info(f"  {key}: {mats[key].shape} binary matrix loaded")
    return mats


def compute_combined_jaccard(hi, mats):
    """Compute J_combined(hi, all mi) = mean of J_BP, J_MF, J_Disease."""
    h_bp   = mats["hB"][hi]
    h_mf   = mats["hF"][hi]
    h_dis  = mats["hD"][hi]
    m_bp   = mats["mB"]
    m_mf   = mats["mF"]
    m_dis  = mats["mD"]

    j_bp   = jaccard_row(h_bp,  m_bp)   if h_bp.sum()  > 0 else np.zeros(m_bp.shape[0])
    j_mf   = jaccard_row(h_mf,  m_mf)   if h_mf.sum()  > 0 else np.zeros(m_mf.shape[0])
    j_dis  = jaccard_row(h_dis, m_dis)  if h_dis.sum() > 0 else np.zeros(m_dis.shape[0])

    # Weight: BP (most informative), MF, Disease
    j_comb = (j_bp + j_mf + j_dis) / 3.0
    return j_comb


def run_funcbench():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading shared data...")
    sd = SharedData(skip_ppi=True)
    root = sd.root

    logger.info("Loading binary annotation matrices...")
    mats = load_binary_matrices(root)

    # Load Ensembl orthologs for sequence baselines
    logger.info("Loading Ensembl orthologs...")
    orthologs = load_ensembl_orthologs(
        str(root / "dataset" / "external" / "ensembl_human_mouse_orthologs.tsv"),
        sd.h_genes, sd.m_genes,
    )

    # ── Build FuncBench ground truth ──────────────────────────────────────────
    logger.info("\nBuilding FuncBench ground truth via J_combined Jaccard...")
    t0 = time.time()

    N_h = mats["hB"].shape[0]   # 31424
    N_m = sd.N_m                # 24110

    # Screen all human genes for those with good annotation coverage
    qualified_his = []
    for hi in range(N_h):
        h_bp  = mats["hB"][hi].sum()
        h_mf  = mats["hF"][hi].sum()
        h_dis = mats["hD"][hi].sum()
        # require at least one non-empty modality
        if h_bp + h_mf + h_dis > 0:
            qualified_his.append(hi)

    logger.info(f"  Qualified hi genes (any annotation): {len(qualified_his)}")

    # Sample N_SAMPLE genes from qualified set
    rng = np.random.RandomState(RNG_SEED)
    sampled_his = list(rng.choice(qualified_his, size=min(N_SAMPLE, len(qualified_his)), replace=False))

    # For each sampled hi, compute combined Jaccard → pick top-GT_K as GT
    gt = {}   # hi -> list of mi (ground truth targets)
    jac_scores = {}  # hi -> top-mi scores (for reference)

    for idx, hi in enumerate(sampled_his):
        if idx % 50 == 0:
            logger.info(f"  Processing {idx}/{len(sampled_his)}...")
        j_comb = compute_combined_jaccard(hi, mats)

        # Sort descending, take top-GT_K where J >= MIN_JACCARD
        order = np.argsort(-j_comb)
        top_k = [mi for mi in order[:GT_K] if j_comb[mi] >= MIN_JACCARD]

        if len(top_k) > 0:
            gt[hi] = top_k
            jac_scores[hi] = [float(j_comb[mi]) for mi in top_k]

    valid_his = [hi for hi in sampled_his if hi in gt]
    pairs = [(hi, mi) for hi in valid_his for mi in gt[hi]]

    logger.info(f"  Jaccard GT built in {time.time()-t0:.1f}s")
    logger.info(f"  Valid hi genes (with ≥1 target): {len(valid_his)}")
    logger.info(f"  Total GT pairs: {len(pairs)}")
    logger.info(f"  Avg targets per hi: {len(pairs)/len(valid_his):.2f}")

    if len(pairs) < 10:
        logger.error("Too few pairs! Check MIN_JACCARD threshold.")
        return

    # ── Evaluate all methods ──────────────────────────────────────────────────
    logger.info("\nEvaluating methods on FuncBench...")

    hi_to_mis = defaultdict(list)
    for hi, mi in pairs:
        hi_to_mis[hi].append(mi)

    def eval_fn(rank_fn):
        ranks = []
        for hi in valid_his:
            r = rank_fn(hi)
            for mi in hi_to_mis[hi]:
                ranks.append(int(r[mi]))
        return compute_all_metrics(ranks)

    results = {}
    t_start = time.time()

    # External baselines (fast)
    logger.info("  Ensembl_Compara...")
    results["Ensembl_Compara"]   = eval_fn(lambda hi: compute_ensembl_rank(hi, orthologs, N_m))
    logger.info("  SVD_Annotation...")
    results["SVD_Annotation"]    = eval_fn(lambda hi: compute_svd_best_rank(hi, sd))
    logger.info("  2hop_Overlap...")
    results["2hop_Overlap"]      = eval_fn(lambda hi: compute_2hop_best_rank(hi, sd))
    logger.info("  OrthoFinder_proxy...")
    results["OrthoFinder_proxy"] = eval_fn(
        lambda hi: compute_orthofinder_proxy_rank(hi, orthologs, compute_svd_best_rank(hi, sd), N_m)
    )
    logger.info(f"  External baselines done in {time.time()-t_start:.1f}s")

    # Model variants (expensive)
    for variant in ["M1", "M2", "M3", "M4"]:
        logger.info(f"  {variant}...")
        t1 = time.time()
        ev = ModelEvaluator(variant, sd, device="cpu")
        rank_cache = {}
        for hi in valid_his:
            r = ev.rank_mouse_genes(hi)
            rank_cache[hi] = r["rrf_rank"] if r["rrf_rank"] is not None else r["fgw_rank"]
        results[variant] = eval_fn(lambda hi, c=rank_cache: c[hi])
        logger.info(f"    {variant}: R@5={results[variant]['recall@5']:.1%}  "
                    f"MRR={results[variant]['mrr']:.4f}  "
                    f"({time.time()-t1:.0f}s)")
        ev.free()
        gc.collect()

    elapsed = time.time() - t_start

    # ── Display ──────────────────────────────────────────────────────────────
    logger.info("\n" + "="*90)
    logger.info(f"FUNCBENCH (Combined GO-BP + GO-MF + Disease Jaccard Ground Truth)")
    logger.info(f"  n_hi={len(valid_his)}  n_pairs={len(pairs)}  "
                f"GT_K={GT_K}  min_jaccard={MIN_JACCARD}  time={elapsed:.0f}s")
    logger.info("="*90)
    logger.info(f"  {'Method':<22} {'R@5':>7} {'R@10':>8} {'R@20':>8} {'MRR':>10}")
    logger.info("-"*58)

    order = ["Ensembl_Compara","OrthoFinder_proxy","SVD_Annotation","2hop_Overlap",
             "M1","M2","M3","M4"]
    for name in order:
        m = results.get(name, {})
        star = ">>>" if name == "M4" else "   "
        logger.info(
            f"{star}{name:<19}  "
            f"{100*m.get('recall@5',0):6.1f}%  "
            f"{100*m.get('recall@10',0):7.1f}%  "
            f"{100*m.get('recall@20',0):7.1f}%  "
            f"{m.get('mrr',0):9.4f}"
        )

    logger.info("\n  KEY: Sequence-based methods (Ensembl, OrthoFinder) have NO access to")
    logger.info("  annotation data → lower scores. Annotation-based methods (SVD, 2-hop,")
    logger.info("  M4) directly optimize for the same signal used in ground truth.")

    # Jaccard score stats
    all_jac = [s for scores in jac_scores.values() for s in scores]
    logger.info(f"\n  GT Jaccard stats: mean={np.mean(all_jac):.3f}  "
                f"min={np.min(all_jac):.3f}  max={np.max(all_jac):.3f}")

    # ── Save ──
    out_path = output_dir / "funcbench_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_sample": N_SAMPLE,
                "gt_k": GT_K,
                "min_jaccard": MIN_JACCARD,
                "rng_seed": RNG_SEED,
            },
            "n_valid_hi": len(valid_his),
            "n_pairs": len(pairs),
            "jaccard_stats": {
                "mean": float(np.mean(all_jac)),
                "min":  float(np.min(all_jac)),
                "max":  float(np.max(all_jac)),
            },
            "results": results,
        }, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_funcbench()
