"""
ED Fig 8(c): Negative-control analysis for delta_cos.

Selects 25 'non-druggable' control human genes matched on BP annotation density
to the 25 drug-target genes, then computes the same delta_cos as script02.

delta_cos = cos_BP(h, M4_top1_mouse) - cos_BP(h, naive_ortholog_mouse)

For non-druggable genes, BRIDGE should not systematically find a better mouse
proxy than the naive ortholog, so delta_cos ≈ 0.

No fake numbers.
"""
import json, sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from scipy.stats import mannwhitneyu, ttest_1samp

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manuscript/figures_precompute/out/08c_negative_controls.csv"
OUT_META = ROOT / "manuscript/figures_precompute/out/08c_negative_controls_meta.json"


def normalize_rows(X):
    X = X.tocsr().astype(np.float64)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    inv = sp.diags(1.0 / norms)
    return (inv @ X).tocsr()


def main():
    # ── Load data ──
    W_hB = normalize_rows(sp.load_npz(ROOT / "data/processed/W_hB.npz"))
    W_mB = normalize_rows(sp.load_npz(ROOT / "data/processed/W_mB.npz"))
    idx = json.load(open(ROOT / "data/processed/indexers.json"))
    hg2i = idx["human_genes"]
    mg2i = idx["mouse_genes"]
    hn = idx["human_gene_names"]       # {ENSG: symbol}
    mn = idx["mouse_gene_names"]       # {ENSMUSG: symbol}
    mg_inv = idx["mouse_genes_inv"]    # {row_str: ENSMUSG}

    # Symbol -> row-index
    h_sym2i = {}
    h_ensg2sym = {}
    for ensg, sym in hn.items():
        if sym and ensg in hg2i:
            h_sym2i[str(sym).upper()] = hg2i[ensg]
            h_ensg2sym[ensg] = str(sym)
    m_sym2i = {}
    for ensg, sym in mn.items():
        if sym and ensg in mg2i:
            m_sym2i[str(sym)] = mg2i[ensg]
    row_to_msym = {int(k): mn.get(v, v) for k, v in mg_inv.items()}

    # ── Load orthologs ──
    ortho_path = ROOT / "dataset/external/ensembl_human_mouse_orthologs.tsv"
    orthos = pd.read_csv(ortho_path, sep="\t", dtype=str, keep_default_na=False)
    orthos = orthos.rename(columns={
        "Gene stable ID": "h_ensg",
        "Gene name": "h_sym",
        "Mouse gene stable ID": "m_ensg",
        "Mouse gene name": "m_sym",
        "Mouse homology type": "htype",
        "Mouse orthology confidence [0 low, 1 high]": "conf",
    })
    orthos = orthos[(orthos.m_ensg != "") & (orthos.h_sym != "")]
    orthos["conf"] = pd.to_numeric(orthos["conf"], errors="coerce").fillna(0)
    orthos = orthos.sort_values(["h_sym", "conf"], ascending=[True, False])
    naive_map = orthos.drop_duplicates("h_sym").set_index("h_sym")[
        ["m_sym", "htype", "conf"]
    ].to_dict(orient="index")

    # ── Identify drug targets to exclude ──
    drugs = pd.read_csv(ROOT / "benchmarks/drug_targets_external.csv", dtype=str)
    drugs.columns = [c.strip() for c in drugs.columns]
    drug_syms = set(drugs["human_gene_symbol"].str.strip().str.upper())

    # Also exclude Bench1 and Tier-S genes
    bench1_path = ROOT / "benchmarks/results/bench1_M4.json"
    exclude_syms = set(drug_syms)
    if bench1_path.exists():
        b1 = json.load(open(bench1_path))
        for case in b1.get("cases", b1.get("pairs", [])):
            if "h_sym" in case:
                exclude_syms.add(case["h_sym"].upper())

    # Tier-S anchors
    tier_s_genes = {"DMD", "ABCD1", "RB1", "SHOX", "SERPINA1", "CFTR", "TERC", "HSA21"}
    exclude_syms |= tier_s_genes

    print(f"Excluding {len(exclude_syms)} gene symbols (drugs + Bench1 + Tier-S)")

    # ── Compute annotation density for drug targets ──
    drug_nnz = []
    for sym in drug_syms:
        if sym in h_sym2i:
            row_i = h_sym2i[sym]
            nnz = int(W_hB.getrow(row_i).nnz)
            drug_nnz.append(nnz)
    drug_nnz = np.array(drug_nnz)
    print(f"Drug target annotation density: mean={drug_nnz.mean():.1f}, "
          f"median={np.median(drug_nnz):.1f}, range=[{drug_nnz.min()}, {drug_nnz.max()}]")

    # ── Select 25 matched control genes ──
    # Criteria:
    # 1. Has a one-to-one ortholog with high confidence
    # 2. Not in any exclude set
    # 3. Has BP annotations (nnz > 0)
    # 4. Matched on annotation density to drug targets (within ±50% of median)

    median_nnz = np.median(drug_nnz)
    low_nnz = max(1, int(median_nnz * 0.3))
    high_nnz = int(median_nnz * 3.0)

    candidates = []
    for h_sym_upper, row_i in h_sym2i.items():
        if h_sym_upper in exclude_syms:
            continue
        # Must have one-to-one ortholog
        # Look up by original case symbol
        orig_sym = None
        for ensg, sym in hn.items():
            if str(sym).upper() == h_sym_upper:
                orig_sym = str(sym)
                break
        if orig_sym is None:
            continue

        entry = naive_map.get(orig_sym)
        if entry is None:
            continue
        if entry["htype"] != "ortholog_one2one":
            continue
        if entry["conf"] != 1.0:
            continue
        m_naive_sym = entry["m_sym"]
        if m_naive_sym not in m_sym2i:
            continue

        nnz = int(W_hB.getrow(row_i).nnz)
        if nnz < low_nnz or nnz > high_nnz:
            continue

        candidates.append({
            "h_sym": orig_sym,
            "h_sym_upper": h_sym_upper,
            "row_i": row_i,
            "nnz": nnz,
            "m_naive_sym": m_naive_sym,
            "abs_diff_nnz": abs(nnz - median_nnz),
        })

    print(f"Found {len(candidates)} eligible control candidates")

    # Sort by closeness to median annotation density, pick top 25
    candidates.sort(key=lambda x: x["abs_diff_nnz"])
    controls = candidates[:25]

    # Shuffle to avoid alphabetical bias, then re-sort by nnz for display
    rng = np.random.RandomState(42)
    # Actually, pick 25 with best annotation match, but spread evenly
    # Take 25 closest to the drug target median
    controls.sort(key=lambda x: x["nnz"])

    print(f"\nSelected {len(controls)} control genes:")
    print(f"  NNZ range: [{controls[0]['nnz']}, {controls[-1]['nnz']}]")
    print(f"  Drug target median NNZ: {median_nnz:.1f}")

    # ── Compute delta_cos for controls ──
    rows_out = []
    for c in controls:
        h_sym = c["h_sym"]
        hi = c["row_i"]
        h_vec = W_hB.getrow(hi)

        cos_vec = np.asarray((h_vec @ W_mB.T).todense()).ravel()

        # M4 top-1 mouse (argmax cosine)
        m_top_idx = int(np.argmax(cos_vec))
        m_top_sym = row_to_msym.get(m_top_idx, f"row{m_top_idx}")
        cos_m4 = float(cos_vec[m_top_idx])

        # Naive ortholog
        m_naive_sym = c["m_naive_sym"]
        cos_naive = float(cos_vec[m_sym2i[m_naive_sym]])

        delta = cos_m4 - cos_naive

        rows_out.append({
            "human_gene": h_sym,
            "bp_annotation_count": c["nnz"],
            "m4_top1_mouse": m_top_sym,
            "cos_bp_m4_top1": cos_m4,
            "naive_ortholog_mouse": m_naive_sym,
            "cos_bp_naive": cos_naive,
            "delta_cos": delta,
            "m4_equals_naive": m_top_sym == m_naive_sym or abs(delta) < 1e-6,
        })

    df = pd.DataFrame(rows_out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # ── Statistics ──
    deltas = df["delta_cos"].values
    n_zero = int((np.abs(deltas) < 1e-6).sum())
    n_pos = int((deltas > 1e-6).sum())
    n_neg = int((deltas < -1e-6).sum())

    # One-sample t-test: is mean delta significantly different from 0?
    t_stat, t_pval = ttest_1samp(deltas, 0)

    # Also load drug forest for comparison U-test
    drug_forest = pd.read_csv(ROOT / "manuscript/figures_precompute/out/02_drug_forest.csv")
    drug_deltas = pd.to_numeric(
        drug_forest["delta_cos_m4_minus_naive"], errors="coerce"
    ).dropna().values

    # Mann-Whitney U: control deltas vs drug deltas
    if len(drug_deltas) > 0 and len(deltas) > 0:
        u_stat, u_pval = mannwhitneyu(deltas, drug_deltas, alternative="less")
    else:
        u_stat, u_pval = float("nan"), float("nan")

    meta = {
        "n_controls": len(df),
        "annotation_density_range": [int(df["bp_annotation_count"].min()),
                                      int(df["bp_annotation_count"].max())],
        "drug_target_median_nnz": float(median_nnz),
        "delta_cos_summary": {
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "std": float(np.std(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
        },
        "n_m4_equals_naive": n_zero,
        "n_m4_better": n_pos,
        "n_naive_better": n_neg,
        "ttest_vs_zero": {
            "t_statistic": float(t_stat),
            "p_value": float(t_pval),
        },
        "mannwhitney_control_vs_drug": {
            "U_statistic": float(u_stat),
            "p_value": float(u_pval),
            "alternative": "control < drug (one-sided)",
        },
        "control_genes": [r["human_gene"] for r in rows_out],
    }

    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"\n=== Negative Control Results ===")
    print(f"N controls: {len(df)}")
    print(f"Delta_cos: mean={np.mean(deltas):.4f}, median={np.median(deltas):.4f}, "
          f"std={np.std(deltas):.4f}")
    print(f"  M4 == naive: {n_zero}/25")
    print(f"  M4 > naive: {n_pos}/25")
    print(f"  M4 < naive: {n_neg}/25")
    print(f"One-sample t-test vs 0: t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"Mann-Whitney U (ctrl < drug): U={u_stat:.1f}, p={u_pval:.4f}")
    print(f"\nDrug targets delta_cos: mean={np.mean(drug_deltas):.4f}")
    print(f"Controls delta_cos: mean={np.mean(deltas):.4f}")
    print(f"\nSaved: {OUT}")
    print(f"Saved: {OUT_META}")


if __name__ == "__main__":
    main()
