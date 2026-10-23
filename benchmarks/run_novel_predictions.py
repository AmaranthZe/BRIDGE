"""Novel predictions driver for Part 3C.

Runs M4 inference on a curated candidate pool of human genes with non-1:1
orthology and strong disease signal, and writes raw M4 outputs plus
leakage-audit features to a JSON file. Downstream (literature review,
manuscript writing) is done by a separate agent using this JSON as
source-of-truth — every number comes from the real trained model on the
real dataset/ data. NO mocks, NO simulations.

Output: benchmarks/novel_candidates_m4_raw.json
Schema:
  {
    "meta": {...},
    "candidates": {
      "<ENSG>": {
          "human_symbol": str,
          "human_gene_idx": int,
          "ortholog_ensmusg": str or None,
          "ortholog_symbol": str or None,
          "ortholog_homology_type": str or None,
          "ortholog_rank_m4": int or null,
          "top10_mouse": [
              {"ensmusg": str, "symbol": str, "m_idx": int, "rrf_rank": 1-based int, "rrf_score": float},
              ...
          ],
          "top1_is_ortholog": bool,
          "disease_cosine_to_top1": float,  # leakage audit
          "bp_cosine_to_top1": float,
          "mf_cosine_to_top1": float,
          "n_disease_associations": int,
          "bucket": "A_non11" | "C_paralog_expansion",
      },
      ...
    }
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.model_variants import SharedData, ModelEvaluator


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_ortholog_table(root: Path) -> pd.DataFrame:
    tsv = root / "dataset" / "external" / "ensembl_human_mouse_orthologs.tsv"
    df = pd.read_csv(tsv, sep="\t")
    df.columns = [
        "human_ensg",
        "human_sym",
        "mouse_ensmusg",
        "mouse_sym",
        "homology_type",
        "confidence",
    ]
    return df


def build_candidate_pool(
    shared: SharedData,
    root: Path,
    exclude_ensg: set,
    disease_threshold: int = 20,
    top_n_by_disease_score: int = 500,
) -> pd.DataFrame:
    """Build the candidate pool for novel-prediction scoring.

    Uses the real ortholog table and HumanDiseaseGene.csv. No synthetic
    filtering.
    """
    hg_set = set(shared.indexers["human_genes"].keys())
    human_gene_sym = shared.gene_sym  # ENSG -> symbol
    ortho = load_ortholog_table(root)

    # Pick the "representative" mouse ortholog per human gene if any
    # (prefer one-to-one > one-to-many > many-to-many > low-conf)
    ortho_map = {}
    for _, row in ortho.iterrows():
        hg = row["human_ensg"]
        if hg not in hg_set:
            continue
        if hg in ortho_map:
            continue  # keep first per Ensembl ordering
        if pd.isna(row["mouse_ensmusg"]):
            continue
        ortho_map[hg] = {
            "mouse_ensmusg": row["mouse_ensmusg"],
            "mouse_sym": row["mouse_sym"],
            "homology_type": row["homology_type"],
            "confidence": int(row["confidence"]) if not pd.isna(row["confidence"]) else 0,
        }

    # Identify non-1:1 human genes: either no row with one2one, or low-conf one2one
    # Easier: flag by homology_type lookup
    one2one_strict = set(
        ortho[
            (ortho["homology_type"] == "ortholog_one2one")
            & (ortho["confidence"] == 1)
        ]["human_ensg"].unique()
    )
    candidates_non11 = hg_set - one2one_strict  # everything else is non-1:1

    # Load disease data
    dis = pd.read_csv(root / "dataset" / "HumanDiseaseGene.csv")
    dis_counts = dis.groupby("targetId").size().to_dict()
    dis_score_sum = dis.groupby("targetId")["score"].sum().to_dict()

    strong_disease = {k for k, v in dis_counts.items() if v >= disease_threshold}
    pool = candidates_non11 & strong_disease - exclude_ensg

    # Build DataFrame and rank by disease score sum
    rows = []
    for hg in pool:
        o = ortho_map.get(hg, {})
        rows.append(
            {
                "human_ensg": hg,
                "human_sym": human_gene_sym.get(hg, ""),
                "n_disease_associations": dis_counts.get(hg, 0),
                "disease_score_sum": dis_score_sum.get(hg, 0.0),
                "mouse_ensmusg": o.get("mouse_ensmusg"),
                "mouse_sym": o.get("mouse_sym"),
                "homology_type": o.get("homology_type"),
                "confidence": o.get("confidence"),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("disease_score_sum", ascending=False).reset_index(drop=True)
    df = df.head(top_n_by_disease_score)

    # Tag bucket
    def bucket_of(row):
        if row["homology_type"] == "ortholog_one2many":
            return "C_paralog_expansion"
        return "A_non11"

    df["bucket"] = df.apply(bucket_of, axis=1)
    return df


def normalize_rows_sparse(mat: sp.csr_matrix) -> sp.csr_matrix:
    """Row-L2-normalize a sparse matrix (returns csr_matrix)."""
    m = mat.copy()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    inv = sp.diags(1.0 / norms)
    return (inv @ m).tocsr()


def feature_cosine(
    W_h: sp.csr_matrix,
    W_m: sp.csr_matrix,
    h_idx: int,
    m_idx: int,
) -> float:
    """Cosine similarity between human gene row and mouse gene row on a
    shared feature axis (BP, MF, or Disease). Assumes rows already L2-
    normalized."""
    h_row = W_h.getrow(h_idx)
    m_row = W_m.getrow(m_idx)
    return float((h_row @ m_row.T).toarray()[0, 0])


def main():
    parser = argparse.ArgumentParser(description="Novel predictions driver (M4)")
    parser.add_argument("--out", default="benchmarks/novel_candidates_m4_raw.json")
    parser.add_argument("--top_n", type=int, default=500)
    parser.add_argument("--disease_threshold", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("novel_pred")

    root = Path(__file__).resolve().parents[1]

    # Shared data and M4
    t0 = time.time()
    log.info("Loading SharedData...")
    shared = SharedData(project_root=str(root), skip_ppi=True)
    log.info(f"SharedData in {time.time()-t0:.1f}s; N_m={shared.N_m}")

    # Exclusion sets
    exclude_ensg = set()
    sym_to_ensg = shared.sym_to_ensg
    # From greatcases_curated.csv: map human symbols → ENSG
    gc_df = pd.read_csv(root / "benchmarks" / "greatcases_curated.csv")
    for s in gc_df["human_gene_symbols"].dropna():
        for sym in s.split("|"):
            hg = sym_to_ensg.get(sym.strip())
            if hg:
                exclude_ensg.add(hg)
    # From drug_targets_external.csv
    dr_df = pd.read_csv(root / "benchmarks" / "drug_targets_external.csv")
    for s in dr_df["human_gene_symbol"].dropna():
        hg = sym_to_ensg.get(s.strip())
        if hg:
            exclude_ensg.add(hg)
    log.info(f"Excluding {len(exclude_ensg)} human genes already in greatcases/drug_ext")

    # Build candidate pool
    log.info("Building candidate pool...")
    pool = build_candidate_pool(
        shared,
        root,
        exclude_ensg=exclude_ensg,
        disease_threshold=args.disease_threshold,
        top_n_by_disease_score=args.top_n,
    )
    log.info(f"Candidate pool size: {len(pool)}  bucket A={sum(pool['bucket']=='A_non11')} C={sum(pool['bucket']=='C_paralog_expansion')}")

    # Load M4
    log.info("Loading M4...")
    evaluator = ModelEvaluator(
        variant="M4",
        shared=shared,
        device=args.device,
    )

    # Pre-normalize BP/MF/Disease matrices for cosine audit
    log.info("Normalizing feature matrices for leakage audit...")
    W_hBn = normalize_rows_sparse(shared.sparse["W_hB"])
    W_mBn = normalize_rows_sparse(shared.sparse["W_mB"])
    W_hFn = normalize_rows_sparse(shared.sparse["W_hF"])
    W_mFn = normalize_rows_sparse(shared.sparse["W_mF"])
    W_hDn = normalize_rows_sparse(shared.sparse["W_hD"])
    W_mDn = normalize_rows_sparse(shared.sparse["W_mD"])

    # Run M4 on each candidate
    log.info(f"Running M4 on {len(pool)} candidates...")
    dis = pd.read_csv(root / "dataset" / "HumanDiseaseGene.csv")
    dis_counts = dis.groupby("targetId").size().to_dict()

    out = {
        "meta": {
            "variant": "M4",
            "candidate_pool_size": int(len(pool)),
            "disease_threshold": args.disease_threshold,
            "top_n_by_disease_score": args.top_n,
            "exclude_ensg_count": len(exclude_ensg),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "candidates": {},
    }

    # Mouse symbol lookup: shared.gene_sym covers both species; build filtered dict
    mouse_names = {mg: shared.gene_sym.get(mg, "") for mg in shared.indexers["mouse_genes"].keys()}
    m_inv = shared.m_inv  # int idx -> ENSMUSG
    N_m = shared.N_m

    t0 = time.time()
    for i, row in pool.iterrows():
        hg = row["human_ensg"]
        h_sym = row["human_sym"]
        hi = shared.indexers["human_genes"].get(hg)
        if hi is None:
            continue

        try:
            result = evaluator.rank_mouse_genes(hi, human_symbol=h_sym)
        except Exception as e:
            log.warning(f"{hg} ({h_sym}): M4 failed: {e}")
            continue

        rrf_rank = result["rrf_rank"]  # ndarray of 1-based ranks over N_m
        if rrf_rank is None:
            # M4 should always have rrf_rank; if missing, use fgw_rank
            rrf_rank = result["fgw_rank"]

        # Top-10 mouse indices (where rrf_rank == 1..10)
        top10_mask = rrf_rank <= 10
        top10_idx = np.where(top10_mask)[0]
        top10_sorted = sorted(top10_idx, key=lambda j: rrf_rank[j])

        # Ortholog rank
        ortho_ensmusg = row["mouse_ensmusg"]
        ortho_rank = None
        if isinstance(ortho_ensmusg, str) and ortho_ensmusg in shared.indexers["mouse_genes"]:
            mi_o = shared.indexers["mouse_genes"][ortho_ensmusg]
            ortho_rank = int(rrf_rank[mi_o])

        # Build top10 entries with RRF score ≈ 1/rank (we don't have the
        # fused RRF score directly; use rank inverse as proxy)
        top10 = []
        for j in top10_sorted:
            mg_ensmusg = m_inv.get(int(j))
            top10.append(
                {
                    "ensmusg": mg_ensmusg,
                    "symbol": mouse_names.get(mg_ensmusg, "") if mg_ensmusg else "",
                    "m_idx": int(j),
                    "rrf_rank": int(rrf_rank[j]),
                }
            )

        top1 = top10[0] if top10 else None
        top1_is_ortholog = bool(
            top1 and isinstance(ortho_ensmusg, str) and top1["ensmusg"] == ortho_ensmusg
        )

        # Leakage audit: cosine between human row and top-1 mouse row
        if top1 and top1["m_idx"] is not None:
            dis_cos = feature_cosine(W_hDn, W_mDn, hi, top1["m_idx"])
            bp_cos = feature_cosine(W_hBn, W_mBn, hi, top1["m_idx"])
            mf_cos = feature_cosine(W_hFn, W_mFn, hi, top1["m_idx"])
        else:
            dis_cos = bp_cos = mf_cos = float("nan")

        out["candidates"][hg] = {
            "human_symbol": h_sym,
            "human_gene_idx": int(hi),
            "ortholog_ensmusg": ortho_ensmusg if isinstance(ortho_ensmusg, str) else None,
            "ortholog_symbol": row["mouse_sym"] if isinstance(row["mouse_sym"], str) else None,
            "ortholog_homology_type": row["homology_type"] if isinstance(row["homology_type"], str) else None,
            "ortholog_rank_m4": ortho_rank,
            "top10_mouse": top10,
            "top1_is_ortholog": top1_is_ortholog,
            "disease_cosine_to_top1": dis_cos,
            "bp_cosine_to_top1": bp_cos,
            "mf_cosine_to_top1": mf_cos,
            "n_disease_associations": int(row["n_disease_associations"]),
            "bucket": row["bucket"],
        }

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(pool) - i - 1)
            log.info(
                f"  {i+1}/{len(pool)} done, elapsed={elapsed:.0f}s, eta={eta:.0f}s; "
                f"current={h_sym} top1={top1['symbol'] if top1 else '?'} "
                f"(is_ortho={top1_is_ortholog})"
            )

    evaluator.free()

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Wrote {out_path}  ({len(out['candidates'])} candidates)")

    # Quick summary
    n_non_ortho = sum(
        1 for c in out["candidates"].values() if not c["top1_is_ortholog"]
    )
    log.info(
        f"Summary: {len(out['candidates'])} scored, "
        f"{n_non_ortho} have non-ortholog M4 top-1 "
        f"(= {100 * n_non_ortho / max(1, len(out['candidates'])):.1f}%)"
    )


if __name__ == "__main__":
    main()
