"""Generate paper-ready figures for Part 3 of the manuscript.

Figures produced:
  - fig_3A_ooc_by_method_subtype.png
      OOC accuracy by method × subtype (grouped bar chart).
      Data: benchmarks/results/ooc_results_summary.csv

  - fig_3B_drug_capability_map.png
      Drug-translation capability map: case count per species_issue_type,
      stacked by M4 OOC pass/fail.
      Data: benchmarks/drug_targets_external.csv
            + benchmarks/results/ooc_results_per_case.csv (M4 × DX* rows)

  - fig_3C_prediction_cards.png
      Prediction-card grid for the 6 final cards in §3C.2.
      Each subplot: top-3 mouse candidates × {BP, MF, Disease} cosine heatmap.
      Data: benchmarks/novel_candidates_m4_raw.json
            + data/processed/W_{h,m}{B,F,D}.npz (for off-top1 cosines)

  - fig_SX_blast_m4_union.png
      BLAST ∩/∪ M4 complementarity on Tier-A/B OOC-B cases.
      Data: benchmarks/results/ooc_results_per_case.csv (blast + M4 × A/B tier)

All figures save a companion CSV with the underlying numeric table.
No data is invented — every number is traceable to a dataset file.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

BENCH = Path(__file__).parent
RESULTS = BENCH / "results"
PROC = BENCH.parent / "data" / "processed"

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Fig 3A: OOC accuracy by method × subtype ─────────────────────────────────

METHOD_ORDER = [
    "HGT_standard", "CAME", "SATURN", "GPZ",
    "seq_identity", "blast",
    "M1", "M2", "M3", "M4",
]
METHOD_COLOR = {
    "HGT_standard":  "#b0b0b0",
    "CAME":          "#909090",
    "SATURN":        "#7f7f7f",
    "GPZ":           "#505050",
    "seq_identity":  "#f4a582",
    "blast":         "#d6604d",
    "M1":            "#92c5de",
    "M2":            "#4393c3",
    "M3":            "#2166ac",
    "M4":            "#053061",
}


def fig_3A():
    """Fig 3A: OOC accuracy by method × subtype — grouped bar chart."""
    df = pd.read_csv(RESULTS / "ooc_results_summary.csv")
    df = df.set_index("method").loc[METHOD_ORDER]

    subtype_cols = [
        ("ooc_overall_accuracy",    "OOC overall"),
        ("ooc_A_promote_accuracy",  "OOC-A promote"),
        ("ooc_B_substitute_accuracy","OOC-B substitute"),
        ("ooc_C_compress_accuracy", "OOC-C compress"),
    ]
    n_methods = len(METHOD_ORDER)
    n_sub = len(subtype_cols)
    bar_w = 0.18

    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(n_methods)
    subtype_colors = ["#2b2b2b", "#6baed6", "#fd8d3c", "#74c476"]

    for i, (col, label) in enumerate(subtype_cols):
        offsets = (i - (n_sub - 1) / 2) * bar_w
        vals = df[col].astype(float).values * 100
        bars = ax.bar(x + offsets, vals, bar_w,
                      label=label, color=subtype_colors[i],
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, vals):
            if v >= 1.0:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
    ax.set_ylabel("OOC accuracy (%)")
    ax.set_title("Fig 3A. OOC accuracy by method × subtype "
                 "(n=71 scored cases on greatcases+drug_ext panel)")
    ax.set_ylim(0, max(df["ooc_A_promote_accuracy"].max() * 100, 40) + 15)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", frameon=False, ncols=4)

    # Annotate benchmark subtype: OOC-A "75%" for CAME is retrieval-meaningless
    # (rank > 14000). Caveat mark:
    ax.annotate("*passes at rank >14 000 — retrieval-meaningless",
                xy=(1, 75), xytext=(3.5, 85),
                fontsize=7, color="#606060", style="italic",
                arrowprops=dict(arrowstyle="->", color="#606060", lw=0.5))

    out_png = RESULTS / "fig_3A_ooc_by_method_subtype.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    # Companion CSV
    (RESULTS / "fig_3A_ooc_by_method_subtype.csv").write_text(
        df[[c for c, _ in subtype_cols]].to_csv())
    print(f"  saved {out_png.name}")
    return out_png


# ── Fig 3B: Drug-translation capability map ──────────────────────────────────

def fig_3B():
    """Fig 3B: per species_issue_type, stacked by M4 best_gold_rank bin.

    The external drug panel has no ground-truth OOC label (ooc_pass is
    NaN for 24/25 cases because the suspected proxy is a hypothesis, not
    a gold partner). What we *can* score is: for each case, at what rank
    does M4 retrieve the literature-proposed `suspected_mouse_proxy`?
    That rank is already in ooc_results_per_case.csv as `best_gold_rank`.
    """
    drug = pd.read_csv(BENCH / "drug_targets_external.csv")
    per = pd.read_csv(RESULTS / "ooc_results_per_case.csv")

    per_dx = per[(per["source"] == "drug_ext") & (per["method"] == "M4")].copy()
    # case_id format in per-case CSV is "DX<id_ext>" (e.g. DX1001 for id_ext 1001)
    drug["case_id"] = drug["id_ext"].apply(lambda v: f"DX{int(v)}")
    merged = per_dx.merge(drug[["case_id", "species_issue_type",
                                 "human_gene_symbol"]],
                          on="case_id", how="left")
    assert not merged["species_issue_type"].isna().all(), (
        "species_issue_type merge failed — check case_id key format")

    def rank_bin(r):
        if pd.isna(r):
            return "Proxy undefined"
        r = float(r)
        if r == 1:
            return "Top-1 hit"
        if r <= 10:
            return "Top-2-10 hit"
        return "Rank >10 (miss)"

    merged["rank_bin"] = merged["best_gold_rank"].apply(rank_bin)
    order_issue = [
        "divergent_pharmacology",
        "expression_pattern",
        "different_ligand_set",
        "missing_mouse_ortholog",
    ]
    order_bin = ["Top-1 hit", "Top-2-10 hit", "Rank >10 (miss)", "Proxy undefined"]

    agg = (merged.groupby(["species_issue_type", "rank_bin"])
                 .size().unstack(fill_value=0)
                 .reindex(index=order_issue, columns=order_bin, fill_value=0))

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x = np.arange(len(agg.index))
    bottoms = np.zeros(len(x))
    colors = {
        "Top-1 hit":        "#1b7837",
        "Top-2-10 hit":     "#5aae61",
        "Rank >10 (miss)":  "#d6604d",
        "Proxy undefined":  "#b8b8b8",
    }
    for col in order_bin:
        vals = agg[col].values
        ax.bar(x, vals, bottom=bottoms, label=col,
               color=colors[col], edgecolor="white", linewidth=0.6)
        for xi, v in enumerate(vals):
            if v > 0:
                ax.text(xi, bottoms[xi] + v / 2, f"{int(v)}",
                        ha="center", va="center",
                        color="white" if col != "Proxy undefined" else "#404040",
                        fontsize=8, fontweight="bold")
        bottoms += vals

    # Total labels
    for xi, total in enumerate(bottoms):
        ax.text(xi, total + 0.15, f"n={int(total)}", ha="center",
                va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([
        "Divergent\npharmacology",
        "Expression\npattern",
        "Different\nligand set",
        "Missing mouse\northolog",
    ])
    ax.set_ylabel("# external drug-translation cases")
    ax.set_title("Fig 3B. Drug-translation capability map — "
                 "M4 rank of literature-proposed mouse proxy (n=25)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=False, ncols=2)

    # §3B.5 class annotation
    ax.axhline(0, color="black", lw=0.5)
    ax.text(0.01, -0.22,
            "Class 1 (structural non-orthology, M4-addressable): "
            "missing_mouse_ortholog. "
            "Class 2 (residue-level pharmacology, M4 ranks the ortholog correctly "
            "but cannot flag affinity/epitope drift): other three.",
            transform=ax.transAxes, fontsize=8, color="#404040", style="italic",
            wrap=True)

    out_png = RESULTS / "fig_3B_drug_capability_map.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    agg.to_csv(RESULTS / "fig_3B_drug_capability_map.csv")
    print(f"  saved {out_png.name}")
    return out_png


# ── Fig 3C: Prediction-card grid ──────────────────────────────────────────────

# Six cards from paper_part3_draft.md §3C.2 (post-review final set)
CARDS = [
    ("POT1",    "ENSG00000128513"),
    ("C1S",     "ENSG00000182326"),
    ("CYP2B6",  "ENSG00000197408"),
    ("REG1A",   "ENSG00000115386"),
    ("CFH",     "ENSG00000000971"),
    ("APOL1",   "ENSG00000100342"),
]


def _cosine_rows(mat_h, h_idx, mat_m, m_idxs):
    """Cosine for (one human row) vs (several mouse rows).

    Inputs are already L2-normalized per the preprocessing pipeline
    (configs/default.yaml: preproc.row_l2 = true), so dot-product = cosine.
    """
    h_row = mat_h[h_idx]
    out = []
    for mi in m_idxs:
        m_row = mat_m[mi]
        dot = float(h_row.multiply(m_row).sum())
        out.append(dot)
    return out


def fig_3C():
    with open(BENCH / "novel_candidates_m4_raw.json") as f:
        raw = json.load(f)
    cands_by_ensg = raw["candidates"]

    # Load normalized sparse matrices and indexers
    W_hB = load_npz(PROC / "W_hB.npz").tocsr()
    W_hF = load_npz(PROC / "W_hF.npz").tocsr()
    W_hD = load_npz(PROC / "W_hD.npz").tocsr()
    W_mB = load_npz(PROC / "W_mB.npz").tocsr()
    W_mF = load_npz(PROC / "W_mF.npz").tocsr()
    W_mD = load_npz(PROC / "W_mD.npz").tocsr()

    heat_rows: List[Tuple[str, str, List[Tuple[str, float, float, float]]]] = []
    for sym, ensg in CARDS:
        card = cands_by_ensg[ensg]
        h_idx = card["human_gene_idx"]
        top3 = card["top10_mouse"][:3]
        entries = []
        for cand in top3:
            mi = cand["m_idx"]
            bp = _cosine_rows(W_hB, h_idx, W_mB, [mi])[0]
            mf = _cosine_rows(W_hF, h_idx, W_mF, [mi])[0]
            ds = _cosine_rows(W_hD, h_idx, W_mD, [mi])[0]
            entries.append((cand["symbol"], bp, mf, ds))
        heat_rows.append((sym, ensg, entries))

    # Plot: 3×2 grid of heatmaps
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 9.2), squeeze=False)
    for i, (sym, ensg, entries) in enumerate(heat_rows):
        ax = axes[i // 2, i % 2]
        mat = np.array([[e[1], e[2], e[3]] for e in entries])
        labels_row = [e[0] for e in entries]

        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["BP", "MF", "Disease"])
        ax.set_yticks(range(len(labels_row)))
        ax.set_yticklabels(labels_row)
        ax.set_title(f"{sym}  →  {labels_row[0]}")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                ax.text(c, r, f"{v:.2f}",
                        ha="center", va="center",
                        color="white" if v > 0.55 else "#202020",
                        fontsize=9)
        # axis label on leftmost column
        if i % 2 == 0:
            ax.set_ylabel("top-3 mouse candidate")

    cbar = fig.colorbar(im, ax=axes, shrink=0.62, pad=0.02,
                        location="right")
    cbar.set_label("cosine (row-L2 normalized TF-IDF)")
    fig.suptitle("Fig 3C. Prediction-card cosine grid — "
                 "final 6 cards (§3C.2)",
                 y=0.995, fontsize=12)

    out_png = RESULTS / "fig_3C_prediction_cards.png"
    fig.savefig(out_png)
    plt.close(fig)

    # Companion CSV
    flat = []
    for sym, ensg, entries in heat_rows:
        for rank, (mouse, bp, mf, ds) in enumerate(entries, start=1):
            flat.append({
                "human_symbol": sym, "human_ensg": ensg,
                "mouse_symbol": mouse, "rrf_rank": rank,
                "bp_cosine": bp, "mf_cosine": mf, "disease_cosine": ds,
            })
    pd.DataFrame(flat).to_csv(
        RESULTS / "fig_3C_prediction_cards.csv", index=False)
    print(f"  saved {out_png.name}")
    return out_png


# ── Fig S-X: BLAST ∪/∩ M4 complementarity ────────────────────────────────────

def fig_SX():
    """BLAST vs M4 pass/fail contingency on Tier-A/B greatcases (OOC-B + OOC-C)."""
    df = pd.read_csv(RESULTS / "ooc_results_per_case.csv")
    # Restrict to greatcases Tier A or B (the 46-case benchmark panel)
    sub = df[(df["source"] == "greatcases")
             & (df["tier_or_priority"].isin(["A", "B"]))
             & (df["method"].isin(["blast", "M4"]))].copy()

    # Pivot to case × method
    pv = sub.pivot_table(index="case_id", columns="method",
                         values="ooc_pass", aggfunc="first")
    pv = pv.dropna(subset=["blast", "M4"])  # keep only cases scored for both

    # 2x2 contingency
    both   = int(((pv["blast"] == 1) & (pv["M4"] == 1)).sum())
    only_b = int(((pv["blast"] == 1) & (pv["M4"] == 0)).sum())
    only_m = int(((pv["blast"] == 0) & (pv["M4"] == 1)).sum())
    neither = int(((pv["blast"] == 0) & (pv["M4"] == 0)).sum())
    n_total = both + only_b + only_m + neither

    # Panel layout: left = stacked bar of coverage, right = 2x2 matrix
    fig = plt.figure(figsize=(11, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.35)

    # Panel A: coverage bars
    axL = fig.add_subplot(gs[0, 0])
    cov_labels = ["BLAST only", "M4 only", "Both (AND)", "Union (OR)"]
    cov_vals = [only_b, only_m, both, both + only_b + only_m]
    colors_cov = ["#d6604d", "#2166ac", "#5aae61", "#1b7837"]
    bars = axL.bar(cov_labels, cov_vals, color=colors_cov,
                   edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, cov_vals):
        axL.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axL.set_ylabel(f"# Tier-A/B cases passing OOC (n={n_total} scored)")
    axL.set_title("Fig S-X(a). BLAST vs M4 — case-level OOC coverage")
    axL.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel B: 2x2 contingency
    axR = fig.add_subplot(gs[0, 1])
    mat = np.array([[neither, only_b], [only_m, both]], dtype=float)
    im = axR.imshow(mat, cmap="Blues", aspect="equal")
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["BLAST fail", "BLAST pass"])
    axR.set_yticks([0, 1])
    axR.set_yticklabels(["M4 fail", "M4 pass"])
    for r in range(2):
        for c in range(2):
            axR.text(c, r, f"{int(mat[r, c])}",
                     ha="center", va="center",
                     color="white" if mat[r, c] > mat.max() * 0.55 else "#202020",
                     fontsize=13, fontweight="bold")
    axR.set_title("Fig S-X(b). 2×2 OOC-pass contingency")

    # Spell out the top cases per cell
    case_cells = {
        "BLAST_only": pv.index[(pv["blast"] == 1) & (pv["M4"] == 0)].tolist(),
        "M4_only":    pv.index[(pv["blast"] == 0) & (pv["M4"] == 1)].tolist(),
        "Both":       pv.index[(pv["blast"] == 1) & (pv["M4"] == 1)].tolist(),
        "Neither":    pv.index[(pv["blast"] == 0) & (pv["M4"] == 0)].tolist(),
    }

    fig.suptitle(
        f"Fig S-X. BLAST AND/OR M4 complementarity on Tier-A/B "
        f"(union = {both + only_b + only_m}/{n_total} "
        f"= {100 * (both + only_b + only_m) / n_total:.1f} %)",
        y=1.02, fontsize=11)

    out_png = RESULTS / "fig_SX_blast_m4_union.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    # Companion CSV: contingency + per-cell case lists
    summary = pd.DataFrame([
        {"cell": "BLAST_only",       "count": only_b,
         "case_ids": ";".join(case_cells["BLAST_only"])},
        {"cell": "M4_only",          "count": only_m,
         "case_ids": ";".join(case_cells["M4_only"])},
        {"cell": "Both_intersection","count": both,
         "case_ids": ";".join(case_cells["Both"])},
        {"cell": "Neither",          "count": neither,
         "case_ids": ";".join(case_cells["Neither"])},
        {"cell": "Union",            "count": both + only_b + only_m,
         "case_ids": ";".join(sorted(set(
             case_cells["BLAST_only"] + case_cells["M4_only"]
             + case_cells["Both"])))},
        {"cell": "Total_scored",     "count": n_total, "case_ids": ""},
    ])
    summary.to_csv(RESULTS / "fig_SX_blast_m4_union.csv", index=False)
    print(f"  saved {out_png.name}")
    return out_png, summary


# ── Fig 4: Bench3 essentiality concordance gap ───────────────────────────────

# Methods in bottom-to-top plot order (baselines first, then M1..M4).
FIG4_METHODS = [
    "Ensembl_Compara",
    "SVD_Annotation",
    "2hop_Overlap",
    "OrthoFinder_proxy",
    "M1",
    "M2",
    "M3",
    "M4",
]

# Colour scheme: light grey for non-neural baselines, medium blue for M1-M3,
# dark blue for M4.
FIG4_COLORS = {
    "Ensembl_Compara":   "#b8b8b8",
    "SVD_Annotation":    "#b8b8b8",
    "2hop_Overlap":      "#b8b8b8",
    "OrthoFinder_proxy": "#b8b8b8",
    "M1":                "#6baed6",
    "M2":                "#4292c6",
    "M3":                "#2171b5",
    "M4":                "#08306b",
}


def _fig4_load_bench3() -> pd.DataFrame:
    """Load Bench3 precision_gap@5 / mannwhitney_p@5 / ess_hitrate@5 for the
    eight methods that appear in fig_4.

    M1-M4 live in their own bench3_{Mx}.json files.
    The four baselines live under ``baselines/<name>`` in bench3_external.json.
    """
    rows: List[Dict[str, Any]] = []

    # Load model variants M1..M4
    mvals: Dict[str, Dict[str, Any]] = {}
    for v in ["M1", "M2", "M3", "M4"]:
        with open(RESULTS / f"bench3_{v}.json") as f:
            d = json.load(f)
        ep = d.get("essential_precision", {}) or {}
        cc = d.get("concordance", {}) or {}
        mvals[v] = {
            "precision_gap@5":   cc.get("precision_gap@5"),
            "mannwhitney_p@5":   cc.get("mannwhitney_p@5"),
            "ess_hitrate@5":     ep.get("ess_hitrate@5"),
        }

    # Load external baselines
    with open(RESULTS / "bench3_external.json") as f:
        ext = json.load(f)
    bvals: Dict[str, Dict[str, Any]] = {}
    for name, data in (ext.get("baselines") or {}).items():
        ep = data.get("essential_precision", {}) or {}
        cc = data.get("concordance", {}) or {}
        bvals[name] = {
            "precision_gap@5":   cc.get("precision_gap@5"),
            "mannwhitney_p@5":   cc.get("mannwhitney_p@5"),
            "ess_hitrate@5":     ep.get("ess_hitrate@5"),
        }

    for m in FIG4_METHODS:
        src = mvals.get(m) if m in mvals else bvals.get(m)
        if src is None:
            # Record a missing-data row so the CSV still has 8 rows;
            # the bar plot will skip methods whose gap is None.
            rows.append({
                "method": m,
                "precision_gap@5": None,
                "mannwhitney_p@5": None,
                "ess_hitrate@5": None,
                "missing": True,
            })
            continue
        rows.append({
            "method": m,
            "precision_gap@5": src["precision_gap@5"],
            "mannwhitney_p@5": src["mannwhitney_p@5"],
            "ess_hitrate@5": src["ess_hitrate@5"],
            "missing": (src["precision_gap@5"] is None),
        })

    return pd.DataFrame(rows)


def _fig4_format_p(p: Any) -> str:
    """Format a p-value for right-bar annotation.

    Mirrors the convention in ``generate_paper_tables.py`` (post-fix at L467):
      p < 1e-10  ->  ``p < 10^{-N}`` with ``N = round(-log10(p)) - 1``
      moderate p ->  ``p = X.XXe-N``
      None       ->  ``n/a``
    """
    if p is None:
        return "n/a"
    try:
        pf = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(pf) or pf <= 0:
        return "n/a"
    if pf < 1e-10:
        exponent = int(round(-math.log10(pf))) - 1
        return f"p < 10^-{exponent}"
    return f"p = {pf:.2e}"


def fig_4_bench3_essentiality_gap():
    """Fig 4. Bench3 essentiality concordance gap (precision_gap@5) across
    8 methods. Horizontal bar chart, bottom-to-top ordering defined by
    ``FIG4_METHODS``.
    """
    df = _fig4_load_bench3()

    # Plot values (percentage points)
    gap_pct = np.array([
        (float(g) * 100.0) if g is not None else np.nan
        for g in df["precision_gap@5"].values
    ])
    colors = [
        FIG4_COLORS.get(m, "#c0c0c0") if not miss else "#e8e8e8"
        for m, miss in zip(df["method"].values, df["missing"].values)
    ]

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    y = np.arange(len(df))
    bars = ax.barh(
        y, np.nan_to_num(gap_pct, nan=0.0),
        color=colors, edgecolor="white", linewidth=0.6, height=0.72,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["method"].values)
    ax.set_xlabel("Precision gap at rank-5 (percentage points)")
    ax.axvline(0.0, color="#444444", linewidth=1.0,
               linestyle="--", zorder=0)
    # "No concordance" label just above the axvline
    ymax_text = len(df) - 0.4
    ax.text(0.03, ymax_text, "No concordance (gap = 0)",
            fontsize=8, color="#444444",
            ha="left", va="top", style="italic")

    # Right-side x-limit: leave room for the p-value annotations
    xmax = float(np.nanmax(gap_pct)) if np.isfinite(np.nanmax(gap_pct)) else 10.0
    ax.set_xlim(left=min(0.0, float(np.nanmin(gap_pct)) - 0.2),
                right=xmax * 1.45 + 1.5)

    # Per-bar annotations: gap value + Mann-Whitney p-value
    for yi, (_, row) in zip(y, df.iterrows()):
        g = row["precision_gap@5"]
        p = row["mannwhitney_p@5"]
        if g is None:
            ax.text(0.1, yi, "  no data", va="center", ha="left",
                    fontsize=9, color="#a02020", style="italic")
            continue
        g_pct = float(g) * 100.0
        pstr = _fig4_format_p(p)
        ax.text(g_pct + 0.25, yi,
                f"+{g_pct:.2f} pp   ({pstr})",
                va="center", ha="left", fontsize=8.5,
                color="#202020")

    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)

    ax.set_title(
        "Bench3 essentiality concordance: precision gap at rank-5 across 8 methods",
        fontsize=11)
    fig.text(
        0.5, 0.93,
        "Human essential genes map to mouse essential genes; "
        "gap = Pr(mouse essential | top-5) for essential vs non-essential human queries.",
        ha="center", va="center", fontsize=8.5, color="#404040",
        style="italic")

    out_png = RESULTS / "fig_4_bench3_essentiality_gap.png"
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # Companion CSV
    csv_rows = []
    for _, row in df.iterrows():
        m = row["method"]
        g = row["precision_gap@5"]
        p = row["mannwhitney_p@5"]
        h = row["ess_hitrate@5"]
        bar_color = (FIG4_COLORS.get(m, "#c0c0c0")
                     if not row["missing"] else "#e8e8e8 (missing)")
        csv_rows.append({
            "method": m,
            "precision_gap_pct": (float(g) * 100.0) if g is not None else None,
            "mannwhitney_p":     p,
            "ess_hitrate_pct":   (float(h) * 100.0) if h is not None else None,
            "bar_color":         bar_color,
        })
    pd.DataFrame(csv_rows, columns=[
        "method", "precision_gap_pct", "mannwhitney_p",
        "ess_hitrate_pct", "bar_color",
    ]).to_csv(RESULTS / "fig_4_bench3_essentiality_gap.csv", index=False)

    print(f"  saved {out_png.name}")
    return out_png, df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating Part-3 paper figures...")
    fig_3A()
    fig_3B()
    fig_3C()
    out_png, sx = fig_SX()
    print("\nFig S-X summary (Tier-A/B greatcases only):")
    print(sx.to_string(index=False).encode("ascii", errors="replace").decode())
    fig_4_bench3_essentiality_gap()


if __name__ == "__main__":
    main()
