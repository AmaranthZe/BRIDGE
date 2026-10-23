"""Generate all paper-ready comparison tables.

Reads existing JSON benchmark results and produces:
1. Table 1: Main comparison on Bench1 (primary benchmark)
2. Table 2: Ablation study M1->M4 across all benchmarks
3. Table 3: Multi-benchmark comparison (bench2, bench3, bench5)
4. Table 4: Published-method proxy comparison on Bench1
5. LaTeX source for all tables

Citation mapping used:
- Ensembl Compara: Yates et al., Genome Biol. 2022
- OrthoFinder: Emms & Kelly, Genome Biol. 2019
- HGT-standard: Yun et al., WWW 2019
- HGT+CrossSpec (CAME): Liu et al., Genome Res. 2023
- FGW-OT (SATURN): Rosen et al., Nat. Methods 2024
- KGE-SVD (GenePlexusZoo): Krishnan et al., PLoS Comput. Biol. 2024
- Bioteque KGE: Fernandez-Torras et al., Nat. Commun. 2022
"""

import csv
import json
import math
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

# ── Helpers ──────────────────────────────────────────────────────────────────

def r(v, pct=True):
    if v is None:
        return "  --  "
    if pct:
        return f"{100*v:5.1f}\\%"
    return f"{v:.4f}"

def rp(v):
    """Render as percentage string (no LaTeX)."""
    if v is None:
        return "   --"
    return f"{100*v:5.1f}%"

def load(name):
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ── Data loading ─────────────────────────────────────────────────────────────

def get_bench1():
    """Return dict[method] -> {r5,r10,r20,mrr}."""
    out = {}

    # Model variants (M1-M4) -- use primary_metrics
    for v in ["M1","M2","M3","M4"]:
        d = load(f"bench1_{v}")
        pm = d.get("summary",{}).get("primary_metrics",{})
        out[v] = {
            "r5":  pm.get("recall@5"),
            "r10": pm.get("recall@10"),
            "r20": pm.get("recall@20"),
            "mrr": pm.get("mrr"),
            "n":   pm.get("n"),
        }

    # External baselines (bench6_comparison)
    b6 = load("bench6_comparison")
    for name, data in b6.get("comparison", b6.get("baseline_metrics",{})).items():
        out[name] = {
            "r5":  data.get("recall@5"),
            "r10": data.get("recall@10"),
            "r20": data.get("recall@20"),
            "mrr": data.get("mrr"),
            "n":   data.get("n"),
        }

    # Published-method proxies (run_published_baselines)
    pb = load("published_baselines")
    for name, data in pb.items():
        b1 = data.get("bench1", {})
        out[f"PUB_{name}"] = {
            "r5":  b1.get("recall@5"),
            "r10": b1.get("recall@10"),
            "r20": b1.get("recall@20"),
            "mrr": b1.get("mrr"),
            "n":   b1.get("n"),
        }

    return out


def get_bench2():
    out = {}
    for v in ["M1","M2","M3","M4"]:
        d = load(f"bench2_{v}")
        ov = d.get("overall", {})
        out[v] = {
            "r5":  ov.get("recall@5"),
            "r10": ov.get("recall@10"),
            "r20": ov.get("recall@20"),
            "mrr": ov.get("mrr"),
            "n":   ov.get("n"),
        }
    ext = load("bench2_external")
    for name, data in ext.get("baselines",{}).items():
        out[name] = {
            "r5":  data.get("recall@5"),
            "r10": data.get("recall@10"),
            "r20": data.get("recall@20"),
            "mrr": data.get("mrr"),
            "n":   data.get("n"),
        }
    return out


def get_bench5():
    out = {}
    for v in ["M1","M2","M3","M4"]:
        d = load(f"bench5_{v}")
        # bench5 wraps in "summary"
        ov = d.get("summary",{}).get("overall", d.get("overall", {}))
        out[v] = {
            "r5":  ov.get("recall@5"),
            "r10": ov.get("recall@10"),
            "r20": ov.get("recall@20"),
            "mrr": ov.get("mrr"),
            "n":   ov.get("n"),
        }
    ext = load("bench5_external")
    for name, data in ext.get("baselines",{}).items():
        out[name] = {
            "r5":  data.get("recall@5"),
            "r10": data.get("recall@10"),
            "r20": data.get("recall@20"),
            "mrr": data.get("mrr"),
            "n":   data.get("n"),
        }
    return out


def get_funcbench():
    """FuncBench: pure combined-Jaccard GT (537 pairs).

    Returns dict[method] -> {r5,r10,r20,mrr,n}.
    """
    out = {}
    d = load("funcbench_results")
    for name, data in d.get("results", {}).items():
        out[name] = {
            "r5":  data.get("recall@5"),
            "r10": data.get("recall@10"),
            "r20": data.get("recall@20"),
            "mrr": data.get("mrr"),
            "n":   data.get("n"),
        }
    return out


def get_hardpairs():
    """Bench2 HardPairs: partitioned by Ensembl rank <=20 vs >20.

    Returns dict[method] -> {
        all_r5, easy_r5, hard_r5, hard_r10, hard_r20, hard_mrr
    }.
    """
    out = {}
    d = load("bench2_hardpairs_analysis")
    for name, data in d.get("results", {}).items():
        a = data.get("all",  {})
        e = data.get("easy", {})
        h = data.get("hard", {})
        out[name] = {
            "all_r5":   a.get("recall@5"),
            "all_mrr":  a.get("mrr"),
            "easy_r5":  e.get("recall@5"),
            "easy_mrr": e.get("mrr"),
            "hard_r5":  h.get("recall@5"),
            "hard_r10": h.get("recall@10"),
            "hard_r20": h.get("recall@20"),
            "hard_mrr": h.get("mrr"),
        }
    return out


def get_published_extended():
    """Load extended published-method proxies across all benchmarks.

    Returns: dict[method] -> dict[benchmark_key] -> {r5,r10,r20,mrr,n}
    where benchmark_key is one of bench1/funcbench/bench2_all/bench2_easy/bench2_hard.
    """
    out = {}
    d = load("published_baselines_extended")
    for name, bms in d.get("results", {}).items():
        out[name] = {}
        for bk, m in bms.items():
            out[name][bk] = {
                "r5":  m.get("recall@5"),
                "r10": m.get("recall@10"),
                "r20": m.get("recall@20"),
                "mrr": m.get("mrr"),
                "n":   m.get("n"),
            }
    return out


def get_bench3():
    """Key metric: concordance precision_gap@5 and ess_hitrate@5."""
    out = {}
    for v in ["M1","M2","M3","M4"]:
        d = load(f"bench3_{v}")
        ep = d.get("essential_precision", {})
        conc = d.get("concordance", {})
        ber = d.get("best_essential_rank", {})
        out[v] = {
            "ess_hit5":    ep.get("ess_hitrate@5"),
            "prec_gap5":   conc.get("precision_gap@5"),
            "mwu_p5":      conc.get("mannwhitney_p@5"),
            "ber_r5":      ber.get("recall@5"),
            "ber_mrr":     ber.get("mrr"),
        }
    ext = load("bench3_external")
    for name, data in ext.get("baselines",{}).items():
        ep = data.get("essential_precision",{})
        conc = data.get("concordance",{})
        ber = data.get("best_essential_rank",{})
        out[name] = {
            "ess_hit5":    ep.get("ess_hitrate@5"),
            "prec_gap5":   conc.get("precision_gap@5"),
            "mwu_p5":      conc.get("mannwhitney_p@5"),
            "ber_r5":      ber.get("recall@5"),
            "ber_mrr":     ber.get("mrr"),
        }
    return out


# ── Plain-text display ────────────────────────────────────────────────────────

SEP = "=" * 110
SEP2 = "-" * 110

def print_table(title, rows, headers, fmt_row, note=None):
    print(f"\n{SEP}")
    print(title)
    print(SEP)
    header_str = "  ".join(f"{h:<{w}}" for h, w in headers)
    print(f"  {header_str}")
    print(SEP2)
    for row in rows:
        print(f"  {fmt_row(row)}")
    if note:
        print(f"\n  Note: {note}")
    print(SEP)


# ── Table 1: Main comparison Bench1 ──────────────────────────────────────────

def table1_bench1_comparison(b1):
    print(f"\n{'='*110}")
    print("TABLE 1  -- Bench1 (45 literature non-ortholog cases) -- PRIMARY BENCHMARK")
    print(f"{'='*110}")
    print(f"  {'Method':<35} {'Reference':<35} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'MRR':>8}")
    print(f"  {'-'*95}")

    groups = [
        # (display_name, key, reference)
        ("--- Published GNN-based methods ---", None, None),
        ("HGT-standard",         "PUB_HGT_standard_v2",   "Yun et al., WWW 2019"),
        ("HGT+CrossSpec (CAME)", "PUB_HGT_crossspec_v3",  "Liu et al., Genome Res. 2023"),
        ("FGW-OT (SATURN-proxy)","PUB_FGW_OT_only",       "Rosen et al., Nat. Methods 2024"),
        ("KGE-SVD (GenePlexus)", "PUB_KGE_SVD",           "Krishnan et al., PLoS CB 2024"),
        ("--- Sequence-based baselines ---",   None, None),
        ("Ensembl Compara",      "Ensembl_Compara",        "Yates et al., Genome Biol. 2022"),
        ("OrthoFinder",          "OrthoFinder_proxy",      "Emms & Kelly, Genome Biol. 2019"),
        ("--- Annotation baselines ---",      None, None),
        ("SVD-Annotation",       "SVD_Annotation",         "Bioteque, Nat. Commun. 2022"),
        ("2-hop Overlap",        "2hop_Overlap",           "this work (non-parametric)"),
        ("--- Our model variants ---",        None, None),
        ("M1 (v2-HGT + mini-RRF)",    "M1", "--"),
        ("M2 (v3-HGT + mini-RRF)",    "M2", "--"),
        ("M3 (v2-HGT + 22-strat RRF)","M3", "--"),
        ("M4 (v3-HGT + 22-strat RRF)","M4", "-- (OURS)"),
    ]

    for name, key, ref in groups:
        if key is None:
            print(f"\n  {name}")
            continue
        m = b1.get(key, {})
        star = " *" if key == "M4" else "  "
        print(f"{star} {name:<33}  {ref:<33}  "
              f"{rp(m.get('r5')):>7}  {rp(m.get('r10')):>7}  "
              f"{rp(m.get('r20')):>7}  {m.get('mrr',0):.4f}")

    print(f"\n  [*] = proposed method  |  n=45 cases  |  background=24,110 mouse genes")
    print(f"  Note: HGT-standard and HGT+CrossSpec give 0% R@5 because HGT embeddings")
    print(f"  occupy separate human/mouse subspaces with no cross-species alignment.")


# ── Table 2: Ablation study ───────────────────────────────────────────────────

def table2_ablation(b1, b2, b3, b5):
    print(f"\n{'='*110}")
    print("TABLE 2  -- Ablation Study: M1 -> M2 -> M3 -> M4")
    print(f"{'='*110}")
    print(f"  {'Variant':<12} {'Architecture':<35} {'B1-R@5':>8} {'B1-R@10':>8} {'B1-MRR':>8} "
          f"{'B2-R@5':>8} {'B5-R@5':>8} {'B3-Gap@5':>10}")
    print(f"  {'-'*100}")

    variants = [
        ("M1", "v2-HGT + FGW mini-ensemble"),
        ("M2", "v3-HGT + FGW mini-ensemble (+xspec edges)"),
        ("M3", "v2-HGT + 22-strategy RRF (full ensemble)"),
        ("M4", "v3-HGT + 22-strategy RRF (full ensemble)"),
    ]

    for v, arch in variants:
        m1 = b1.get(v, {})
        m2 = b2.get(v, {})
        m3 = b3.get(v, {})
        m5 = b5.get(v, {})
        star = "*" if v == "M4" else " "
        gap = m3.get("prec_gap5")
        gap_str = f"{100*gap:+6.2f}%" if gap is not None else "   --"
        print(f"  {star}{v:<11}  {arch:<35}  "
              f"{rp(m1.get('r5')):>7}  {rp(m1.get('r10')):>7}  {m1.get('mrr',0):.4f}  "
              f"{rp(m2.get('r5')):>7}  {rp(m5.get('r5')):>7}  {gap_str:>10}")

    print(f"\n  B1=Bench1 (45 lit. cases), B2=Bench2 (phenotype, 509 pairs),")
    print(f"  B5=Bench5 (HMD genome-wide, 374 pairs), B3-Gap@5=essentiality concordance gap")


# ── Table 3: Multi-benchmark vs all methods ────────────────────────────────────

def table3_multibench(b1, b2, b3, b5):
    print(f"\n{'='*110}")
    print("TABLE 3  -- Multi-Benchmark Comparison: M4 vs All Baselines")
    print(f"{'='*110}")

    methods = [
        ("Ensembl Compara",   "Ensembl_Compara"),
        ("OrthoFinder",       "OrthoFinder_proxy"),
        ("SVD-Annotation",    "SVD_Annotation"),
        ("2-hop Overlap",     "2hop_Overlap"),
        ("M1",                "M1"),
        ("M2",                "M2"),
        ("M3",                "M3"),
        ("M4 (OURS)",         "M4"),
    ]

    print(f"  {'Method':<22}  {'B1-R@5':>7} {'B1-R@10':>8} {'B1-MRR':>8}  "
          f"{'B2-R@5':>7} {'B2-R@10':>8}  {'B5-R@5':>7} {'B5-R@10':>8}  {'B3-Gap@5':>10}")
    print(f"  {'-'*105}")

    for disp, key in methods:
        m1 = b1.get(key, {})
        m2 = b2.get(key, {})
        m3 = b3.get(key, {})
        m5 = b5.get(key, {})
        star = "* " if key == "M4" else "  "
        gap = m3.get("prec_gap5")
        gap_str = f"{100*gap:+6.2f}%" if gap is not None else "   --"
        print(f"{star}{disp:<22}  "
              f"{rp(m1.get('r5')):>7} {rp(m1.get('r10')):>8} {m1.get('mrr',0):.4f}  "
              f"{rp(m2.get('r5')):>7} {rp(m2.get('r10')):>8}  "
              f"{rp(m5.get('r5')):>7} {rp(m5.get('r10')):>8}  {gap_str:>10}")

    print(f"\n  B1=Bench1 (n=45), B2=Bench2 (n=509/512), B5=Bench5 (n=374), B3=essentiality gap")
    print(f"  Note: B2/B5 sequence methods exploit gene-family structure (many-to-many).")
    print(f"  Among annotation-only methods, M4 ranks best on B1 and B2.")


# ── LaTeX table 1: main comparison ────────────────────────────────────────────

def latex_table1(b1):
    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\caption{Bench1 comparison on 45 literature-curated cross-species gene pairs.}")
    lines.append(r"\label{tab:bench1_main}")
    lines.append(r"\begin{tabular}{llcccr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Reference & R@5 & R@10 & R@20 & MRR \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Published GNN-based methods (proxies for our task)}} \\")

    rows = [
        ("HGT-standard",          "PUB_HGT_standard_v2",  r"\citet{yun2019graph}"),
        ("HGT+CrossSpec (CAME)",  "PUB_HGT_crossspec_v3", r"\citet{liu2023came}"),
        ("FGW-OT (SATURN-proxy)", "PUB_FGW_OT_only",      r"\citet{rosen2024saturn}"),
        ("KGE-SVD (GenePlexus)",  "PUB_KGE_SVD",          r"\citet{krishnan2024geneplexus}"),
    ]
    for disp, key, cite in rows:
        m = b1.get(key, {})
        lines.append(f"\\quad {disp} & {cite} & "
                     f"{r(m.get('r5'))} & {r(m.get('r10'))} & {r(m.get('r20'))} & {m.get('mrr',0):.4f} \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Sequence-based baselines}} \\")
    for disp, key, cite in [
        ("Ensembl Compara", "Ensembl_Compara",  r"\citet{yates2022ensembl}"),
        ("OrthoFinder",     "OrthoFinder_proxy",r"\citet{emms2019orthofinder}"),
    ]:
        m = b1.get(key, {})
        lines.append(f"\\quad {disp} & {cite} & "
                     f"{r(m.get('r5'))} & {r(m.get('r10'))} & {r(m.get('r20'))} & {m.get('mrr',0):.4f} \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Annotation-based baselines}} \\")
    for disp, key, cite in [
        ("SVD-Annotation", "SVD_Annotation", r"\citet{fernandezbioteque2022}"),
        ("2-hop Overlap",  "2hop_Overlap",   r"this work"),
    ]:
        m = b1.get(key, {})
        lines.append(f"\\quad {disp} & {cite} & "
                     f"{r(m.get('r5'))} & {r(m.get('r10'))} & {r(m.get('r20'))} & {m.get('mrr',0):.4f} \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Our model variants (ablation)}} \\")
    for v, desc in [("M1","v2-HGT + mini-RRF"),
                    ("M2","v3-HGT + mini-RRF"),
                    ("M3","v2-HGT + full RRF"),
                    ("M4","v3-HGT + full RRF (ours)")]:
        m = b1.get(v, {})
        bold = r"\textbf" if v == "M4" else ""
        def bf(x):
            s = r(x)
            return f"\\textbf{{{s}}}" if v == "M4" else s
        mrr_str = f"{m.get('mrr',0):.4f}"
        if v == "M4":
            mrr_str = "\\textbf{" + mrr_str + "}"
        lines.append(f"\\quad {desc} & -- & "
                     f"{bf(m.get('r5'))} & {bf(m.get('r10'))} & {bf(m.get('r20'))} & "
                     f"{mrr_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── LaTeX table 2: ablation ────────────────────────────────────────────────────

def latex_table2(b1, b2, b3, b5):
    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study across all benchmarks.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & B1-R@5 & B1-R@10 & B1-MRR & B2-R@5 & B5-R@5 & B3-Gap@5 & B3-$p$ \\")
    lines.append(r"\midrule")

    for v in ["M1","M2","M3","M4"]:
        m1 = b1.get(v, {})
        m2 = b2.get(v, {})
        m3 = b3.get(v, {})
        m5 = b5.get(v, {})
        gap = m3.get("prec_gap5", 0)
        pval = m3.get("mwu_p5", 1)
        bold = v == "M4"
        def bf(s):
            return f"\\textbf{{{s}}}" if bold else s

        if pval and pval < 1e-10:
            exponent = int(round(-math.log10(pval))) - 1
            pstr = f"$<10^{{-{exponent}}}$"
        elif pval:
            pstr = f"{pval:.1e}"
        else:
            pstr = "--"

        mrr_s = f"{m1.get('mrr',0):.4f}"
        gap_s = f"{100*gap:+.2f}%"
        lines.append(
            f"{bf(v)} & {bf(rp(m1.get('r5')))} & {bf(rp(m1.get('r10')))} & "
            f"{bf(mrr_s)} & {bf(rp(m2.get('r5')))} & "
            f"{bf(rp(m5.get('r5')))} & {bf(gap_s)} & {pstr} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def table5_funcbench(fb):
    """FuncBench: pure annotation Jaccard GT — sequence methods collapse."""
    print(f"\n{'='*110}")
    print("TABLE 5  -- FuncBench (Pure combined-Jaccard GT, 537 pairs, 184 hi genes)")
    print(f"{'='*110}")
    print(f"  {'Method':<22} {'R@5':>7} {'R@10':>8} {'R@20':>8} {'MRR':>10}")
    print(f"  {'-'*60}")

    methods = [
        ("Ensembl Compara",   "Ensembl_Compara"),
        ("OrthoFinder",       "OrthoFinder_proxy"),
        ("SVD-Annotation",    "SVD_Annotation"),
        ("2-hop Overlap",     "2hop_Overlap"),
        ("M1 (v2+mini)",      "M1"),
        ("M2 (v3+mini)",      "M2"),
        ("M3 (v2+RRF)",       "M3"),
        ("M4 (v3+RRF) OURS",  "M4"),
    ]
    for disp, key in methods:
        m = fb.get(key, {})
        star = "* " if key == "M4" else "  "
        print(f"{star}{disp:<22} "
              f"{rp(m.get('r5')):>7} {rp(m.get('r10')):>8} "
              f"{rp(m.get('r20')):>8} {m.get('mrr',0):10.4f}")

    print(f"\n  Note: GT = top-3 mouse genes per human by "
          f"(J_BP + J_MF + J_Disease)/3 with min J>=0.10.")
    print(f"  Sequence methods drop dramatically here "
          f"(Ensembl 89.1%->20.1%, OrthoFinder 93.8%->46.6%)")
    print(f"  because GT is independent of sequence homology. "
          f"M3/M4 dominate at ~65% R@5.")


def table6_hardpairs(hp):
    """Hard Pairs: Ensembl rank>20 subset — sequence cliff exposed."""
    print(f"\n{'='*110}")
    print("TABLE 6  -- Bench2 HARD PAIRS (Ensembl rank>20, 55 pairs, 47 hi genes)")
    print(f"{'='*110}")
    print(f"  {'Method':<22} {'ALL R@5':>8} {'EASY R@5':>9} "
          f"{'HARD R@5':>9} {'HARD R@10':>10} {'HARD R@20':>10} {'HARD MRR':>10}")
    print(f"  {'-'*95}")

    methods = [
        ("Ensembl Compara",   "Ensembl_Compara"),
        ("OrthoFinder",       "OrthoFinder_proxy"),
        ("SVD-Annotation",    "SVD_Annotation"),
        ("2-hop Overlap",     "2hop_Overlap"),
        ("M1 (v2+mini)",      "M1"),
        ("M2 (v3+mini)",      "M2"),
        ("M3 (v2+RRF)",       "M3"),
        ("M4 (v3+RRF) OURS",  "M4"),
    ]
    for disp, key in methods:
        m = hp.get(key, {})
        star = "* " if key == "M4" else "  "
        print(f"{star}{disp:<22} "
              f"{rp(m.get('all_r5')):>8} {rp(m.get('easy_r5')):>9} "
              f"{rp(m.get('hard_r5')):>9} {rp(m.get('hard_r10')):>10} "
              f"{rp(m.get('hard_r20')):>10} {m.get('hard_mrr',0):10.4f}")

    print(f"\n  Easy = Ensembl rank <= 20 (457 pairs), Hard = rank > 20 (55 pairs, 10.7%).")
    print(f"  Ensembl HARD R@5 = 0.0% by construction - sequence cliff. "
          f"M4 HARD R@5 = 61.8% (best).")
    print(f"  M4 > M3 on HARD R@5 (+1.8pp) and HARD R@20 (+1.8pp): "
          f"v3 cross-species edges help most where it matters.")


def table7_master(b1, b2, b3, b5, fb, hp):
    """Master cross-benchmark table showing the full narrative."""
    print(f"\n{'='*130}")
    print("TABLE 7  -- MASTER CROSS-BENCHMARK SUMMARY")
    print(f"{'='*130}")
    print(f"  {'Method':<22}"
          f"{'B1':>7} {'B2-all':>8} {'B2-HARD':>9} {'B5':>7} "
          f"{'FuncB':>7} {'B3-gap':>9}")
    print(f"  {'='*75}  (all values = R@5 unless noted)")

    methods = [
        ("Ensembl Compara",   "Ensembl_Compara"),
        ("OrthoFinder",       "OrthoFinder_proxy"),
        ("SVD-Annotation",    "SVD_Annotation"),
        ("2-hop Overlap",     "2hop_Overlap"),
        ("M1 (v2+mini)",      "M1"),
        ("M2 (v3+mini)",      "M2"),
        ("M3 (v2+RRF)",       "M3"),
        ("M4 (v3+RRF) OURS",  "M4"),
    ]
    for disp, key in methods:
        m1 = b1.get(key, {})
        m2 = b2.get(key, {})
        m3 = b3.get(key, {})
        m5 = b5.get(key, {})
        mf = fb.get(key, {})
        mh = hp.get(key, {})
        gap = m3.get("prec_gap5")
        gap_str = f"{100*gap:+6.2f}%" if gap is not None else "   --"
        star = "* " if key == "M4" else "  "
        print(f"{star}{disp:<22}"
              f"{rp(m1.get('r5')):>7} {rp(m2.get('r5')):>8} "
              f"{rp(mh.get('hard_r5')):>9} {rp(m5.get('r5')):>7} "
              f"{rp(mf.get('r5')):>7} {gap_str:>9}")

    print(f"\n  B1=Bench1 lit. (n=45), B2-all=Bench2 phenotype (n=512), "
          f"B2-HARD=Ensembl rank>20 (n=55),")
    print(f"  B5=Bench5 HMD (n=374), FuncB=FuncBench pure Jaccard GT (n=537), "
          f"B3-gap=essentiality precision gap.")
    print(f"\n  Narrative:")
    print(f"  - Sequence methods (Ensembl/OrthoFinder) dominate B2/B5 because "
          f"GT derives from sequence homology,")
    print(f"    but collapse on B1 (non-ortho lit. cases), B2-HARD "
          f"(Ensembl cliff=0%), and FuncBench (pure function).")
    print(f"  - M4 is best or tied-best on every pure-function benchmark "
          f"(B1, B2-HARD, FuncBench) + B3 concordance.")


def table8_published_extended(pub_ext, fb, hp, b1):
    """Extended comparison of 4 published-method proxies across benchmarks."""
    print(f"\n{'='*120}")
    print("TABLE 8  -- Published-Method Proxies: Extended Cross-Benchmark Evaluation")
    print(f"{'='*120}")
    print(f"  {'Method (proxy)':<32} {'Reference':<35} "
          f"{'B1':>7} {'FuncB':>7} {'B2-ALL':>8} {'B2-EASY':>9} {'B2-HARD':>9}")
    print(f"  {'-'*105}")

    rows = [
        ("HGT-standard",          "HGT_standard_v2",   "Yun et al., WWW 2019"),
        ("HGT+CrossSpec (CAME)",  "HGT_crossspec_v3",  "Liu et al., Genome Res. 2023"),
        ("FGW-OT (SATURN-proxy)", "FGW_OT_only",       "Rosen et al., Nat. Methods 2024"),
        ("KGE-SVD (GenePlexus)",  "KGE_SVD",           "Krishnan et al., PLoS CB 2024"),
    ]

    def r5_of(method_key, bench_key):
        m = pub_ext.get(method_key, {}).get(bench_key, {})
        return m.get("r5")

    for disp, key, ref in rows:
        print(f"  {disp:<32} {ref:<35} "
              f"{rp(r5_of(key, 'bench1')):>7} "
              f"{rp(r5_of(key, 'funcbench')):>7} "
              f"{rp(r5_of(key, 'bench2_all')):>8} "
              f"{rp(r5_of(key, 'bench2_easy')):>9} "
              f"{rp(r5_of(key, 'bench2_hard')):>9}")

    # M4 row for comparison
    m4_b1 = b1.get("M4", {}).get("r5")
    m4_fb = fb.get("M4", {}).get("r5")
    m4_ba = hp.get("M4", {}).get("all_r5")
    m4_be = hp.get("M4", {}).get("easy_r5")
    m4_bh = hp.get("M4", {}).get("hard_r5")
    print(f"  {'-'*105}")
    print(f"* {'M4 (v3+RRF) OURS':<32} {'this work':<35} "
          f"{rp(m4_b1):>7} {rp(m4_fb):>7} {rp(m4_ba):>8} "
          f"{rp(m4_be):>9} {rp(m4_bh):>9}")

    # Gap analysis
    print(f"\n  Gap analysis (M4 minus best published proxy):")
    for bk, label in [
        ("bench1",      "Bench1"),
        ("funcbench",   "FuncBench"),
        ("bench2_all",  "Bench2-ALL"),
        ("bench2_easy", "Bench2-EASY"),
        ("bench2_hard", "Bench2-HARD"),
    ]:
        vals = [r5_of(k, bk) or 0 for _, k, _ in rows]
        best_pub = max(vals) if vals else 0
        m4 = (b1.get("M4", {}).get("r5") if bk == "bench1" else
              fb.get("M4", {}).get("r5") if bk == "funcbench" else
              hp.get("M4", {}).get({"bench2_all": "all_r5",
                                     "bench2_easy": "easy_r5",
                                     "bench2_hard": "hard_r5"}[bk])) or 0
        gap_pp = 100 * (m4 - best_pub)
        print(f"    {label:<12} best published proxy = {100*best_pub:5.1f}%  "
              f"M4 = {100*m4:5.1f}%  gap = {gap_pp:+6.1f}pp")


def latex_table_published(pub_ext, fb, hp, b1):
    """LaTeX: Extended published-method proxy comparison."""
    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Extended comparison against recently published "
                 r"cross-species methods (R@5).} We implemented proxy versions of "
                 r"four GNN-based methods and evaluated them on all six benchmarks. "
                 r"Standard HGT variants achieve 0\% on every benchmark, confirming "
                 r"that learned heterogeneous graph embeddings do not yield "
                 r"cross-species alignment without explicit optimal transport. "
                 r"Our full M4 pipeline outperforms every proxy by $\geq$38 pp on "
                 r"non-homology benchmarks (Bench1, FuncBench, Bench2-HARD).}")
    lines.append(r"\label{tab:published_extended}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Reference & Bench1 & FuncBench & "
                 r"B2-ALL & B2-EASY & B2-HARD \\")
    lines.append(r"\midrule")

    rows = [
        ("HGT-standard",          "HGT_standard_v2",   r"\citet{yun2019graph}"),
        ("HGT+CrossSpec (CAME)",  "HGT_crossspec_v3",  r"\citet{liu2023came}"),
        ("FGW-OT (SATURN-proxy)", "FGW_OT_only",       r"\citet{rosen2024saturn}"),
        ("KGE-SVD (GenePlexus)",  "KGE_SVD",           r"\citet{krishnan2024geneplexus}"),
    ]
    for disp, key, cite in rows:
        d = pub_ext.get(key, {})
        lines.append(
            f"{disp} & {cite} & "
            f"{r(d.get('bench1', {}).get('r5'))} & "
            f"{r(d.get('funcbench', {}).get('r5'))} & "
            f"{r(d.get('bench2_all', {}).get('r5'))} & "
            f"{r(d.get('bench2_easy', {}).get('r5'))} & "
            f"{r(d.get('bench2_hard', {}).get('r5'))} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        f"\\textbf{{M4 (ours)}} & this work & "
        f"\\textbf{{{r(b1.get('M4', {}).get('r5'))}}} & "
        f"\\textbf{{{r(fb.get('M4', {}).get('r5'))}}} & "
        f"\\textbf{{{r(hp.get('M4', {}).get('all_r5'))}}} & "
        f"\\textbf{{{r(hp.get('M4', {}).get('easy_r5'))}}} & "
        f"\\textbf{{{r(hp.get('M4', {}).get('hard_r5'))}}} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_table1_benchmarks():
    """LaTeX: Table 1 — benchmark-design overview (static, documentary).

    Hand-coded per the §3.1 description in manuscript/section_3_results.md.
    Rows: the six benchmarks. Columns: n, GT source, sequence dependence.
    Numbers are not derived from JSON; they describe the evaluation panel
    design itself and match the §3.1 paragraph verbatim.
    """
    rows = [
        ("Bench1",       "45",
         "Literature-curated non-ortholog case studies",
         "Low"),
        ("Bench2-ALL",   "512",
         "MGI HMD phenotype pairs",
         "High"),
        ("Bench2-EASY",  "457",
         r"HMD pairs with Ensembl rank $\leq$ 20",
         "Very high"),
        ("Bench2-HARD",  "55",
         r"HMD pairs with Ensembl rank $>$ 20",
         "Zero (by construction)"),
        ("Bench5",       "374",
         "MGI HMD genome-wide ortholog pairs",
         "Very high"),
        ("FuncBench",    "537",
         "Top-3 by combined Jaccard of GO-BP/GO-MF/disease",
         "Zero"),
    ]

    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Benchmark-design overview.} "
                 r"Six complementary benchmarks are used throughout this work, "
                 r"chosen to disentangle functional correspondence from sequence "
                 r"homology. Bench2-HARD and FuncBench are designed specifically "
                 r"to eliminate sequence leakage from the ground truth and are "
                 r"treated as the primary evidence for biological (as opposed to "
                 r"phylogenetic) correspondence.}")
    lines.append(r"\label{tab:benchmarks}")
    lines.append(r"\begin{tabular}{lrll}")
    lines.append(r"\toprule")
    lines.append(r"Benchmark & $n$ & Ground-truth source & Sequence dependence \\")
    lines.append(r"\midrule")
    for name, n, gt, dep in rows:
        lines.append(f"{name} & {n} & {gt} & {dep} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_table7_master(b1, b2, b3, b5, fb, hp):
    """LaTeX: Table 7 — master cross-benchmark Recall@5 summary.

    Ports the plain-text ``table7_master`` to LaTeX. Rows are methods,
    columns are benchmarks (all values R@5 except the final B3-gap column,
    which is the essentiality precision gap@5).
    """
    methods = [
        ("Ensembl Compara",   "Ensembl_Compara"),
        ("OrthoFinder",       "OrthoFinder_proxy"),
        ("SVD-Annotation",    "SVD_Annotation"),
        ("2-hop Overlap",     "2hop_Overlap"),
        ("M1 (v2+mini)",      "M1"),
        ("M2 (v3+mini)",      "M2"),
        ("M3 (v2+RRF)",       "M3"),
        ("M4 (v3+RRF, ours)", "M4"),
    ]

    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Master cross-benchmark Recall@5 summary.} "
                 r"All values are Recall@5 (\%) except B3-gap, which is the "
                 r"essentiality precision gap@5 (percentage-point difference in "
                 r"mean DepMap essentiality of top-5 mouse predictions between "
                 r"essential and non-essential human query genes; two-sided "
                 r"Mann-Whitney U). B1 = Bench1 literature non-ortholog cases "
                 r"($n=45$); B2-ALL = Bench2 HMD phenotype pairs ($n=512$); "
                 r"B2-HARD = Bench2 pairs with Ensembl rank $>$ 20 ($n=55$); "
                 r"B5 = Bench5 HMD genome-wide ortholog pairs ($n=374$); "
                 r"FuncB = FuncBench pure combined-Jaccard ground truth "
                 r"($n=537$). M4 is best or tied-best on every sequence-"
                 r"independent benchmark (B1, B2-HARD, FuncB) and on B3 "
                 r"concordance.}")
    lines.append(r"\label{tab:master}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & B1 & B2-ALL & B2-HARD & B5 & FuncB & B3-gap \\")
    lines.append(r"\midrule")

    for disp, key in methods:
        m1 = b1.get(key, {})
        m2 = b2.get(key, {})
        m3 = b3.get(key, {})
        m5 = b5.get(key, {})
        mf = fb.get(key, {})
        mh = hp.get(key, {})
        gap = m3.get("prec_gap5")
        gap_str = f"{100*gap:+.2f}\\%" if gap is not None else "--"
        bold = key == "M4"
        def bf(s):
            return f"\\textbf{{{s}}}" if bold else s

        lines.append(
            f"{bf(disp)} & {bf(r(m1.get('r5')))} & {bf(r(m2.get('r5')))} & "
            f"{bf(r(mh.get('hard_r5')))} & {bf(r(m5.get('r5')))} & "
            f"{bf(r(mf.get('r5')))} & {bf(gap_str)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_table_cliff(fb, hp):
    """LaTeX: the sequence-cliff summary table (paper's key deliverable)."""
    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Sequence cliff:} "
                 r"sequence-based methods collapse on benchmarks where the "
                 r"ground truth is independent of sequence homology "
                 r"(FuncBench) or explicitly excludes orthologs (Bench2-HARD), "
                 r"while our method maintains high performance.}")
    lines.append(r"\label{tab:seqcliff}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & \multicolumn{2}{c}{FuncBench (n=537)} & "
                 r"\multicolumn{2}{c}{Bench2-HARD (n=55)} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r"       & R@5 & MRR & R@5 & R@20 \\")
    lines.append(r"\midrule")
    rows = [
        ("Ensembl Compara",   "Ensembl_Compara"),
        ("OrthoFinder",       "OrthoFinder_proxy"),
        ("SVD-Annotation",    "SVD_Annotation"),
        ("2-hop Overlap",     "2hop_Overlap"),
    ]
    for disp, key in rows:
        mf = fb.get(key, {})
        mh = hp.get(key, {})
        lines.append(
            f"{disp} & {r(mf.get('r5'))} & {mf.get('mrr',0):.4f} & "
            f"{r(mh.get('hard_r5'))} & {r(mh.get('hard_r20'))} \\\\"
        )
    lines.append(r"\midrule")
    for v, desc in [("M1","v2-HGT + mini-RRF"),
                    ("M2","v3-HGT + mini-RRF"),
                    ("M3","v2-HGT + full RRF"),
                    ("M4","v3-HGT + full RRF (\\textbf{OURS})")]:
        mf = fb.get(v, {})
        mh = hp.get(v, {})
        bold = v == "M4"
        def bf(x):
            s = r(x) if not isinstance(x, str) else x
            return f"\\textbf{{{s}}}" if bold else s
        mrr_s = f"{mf.get('mrr',0):.4f}"
        if bold:
            mrr_s = "\\textbf{" + mrr_s + "}"
        lines.append(
            f"{desc} & {bf(mf.get('r5'))} & {mrr_s} & "
            f"{bf(mh.get('hard_r5'))} & {bf(mh.get('hard_r20'))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── BLAST vs M4 union/intersection (Fig S-X companion) ──────────────────────

def _load_blast_m4_union_csv():
    """Load benchmarks/results/fig_SX_blast_m4_union.csv into a dict.

    Returns dict[cell_name] -> {"count": int, "case_ids": [str, ...]}
    where cell_name is one of:
        BLAST_only, M4_only, Both_intersection, Neither, Union, Total_scored.
    Returns {} if the CSV does not exist.
    """
    path = RESULTS / "fig_SX_blast_m4_union.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell = row["cell"]
            raw_ids = row.get("case_ids", "") or ""
            ids = [x for x in raw_ids.split(";") if x]
            try:
                cnt = int(row["count"])
            except (KeyError, ValueError):
                cnt = 0
            out[cell] = {"count": cnt, "case_ids": ids}
    return out


def table_blast_m4_union():
    """Print the BLAST vs M4 contingency summary (Fig S-X companion).

    Reads fig_SX_blast_m4_union.csv and echoes a plain-text summary of the
    2x2 contingency (BLAST_only / M4_only / Both / Neither), plus the three
    headline coverage rates (BLAST alone, M4 alone, Union) and the stacked
    gain figures that underpin the Sec. 3A.5 recommendation in Part 3.
    """
    data = _load_blast_m4_union_csv()
    if not data:
        print("\n[table_blast_m4_union] fig_SX_blast_m4_union.csv not found -- "
              "run benchmarks/generate_paper_figures.py first.")
        return

    total = data.get("Total_scored", {}).get("count", 0)
    blast_only = data.get("BLAST_only", {}).get("count", 0)
    m4_only = data.get("M4_only", {}).get("count", 0)
    both = data.get("Both_intersection", {}).get("count", 0)
    neither = data.get("Neither", {}).get("count", 0)
    union = data.get("Union", {}).get("count", 0)

    blast_cov = (blast_only + both) / total if total else 0.0
    m4_cov = (m4_only + both) / total if total else 0.0
    union_cov = union / total if total else 0.0
    intersect_cov = both / total if total else 0.0
    stacked_vs_blast = (union - (blast_only + both)) / total if total else 0.0
    stacked_vs_m4 = (union - (m4_only + both)) / total if total else 0.0

    print(f"\n{'='*110}")
    print("TABLE S-X  -- BLAST vs M4 union/intersection on Tier-A/B greatcases")
    print(f"{'='*110}")
    print(f"  Total scored by both methods: n = {total}")
    print()
    print(f"  {'Cell':<22} {'n':>4}  {'% of n':>8}  Case IDs")
    print(f"  {'-'*102}")

    def _fmt_row(label, cell_key):
        d = data.get(cell_key, {})
        n = d.get("count", 0)
        ids = d.get("case_ids", [])
        pct = 100.0 * n / total if total else 0.0
        ids_str = ";".join(ids) if ids else "--"
        if len(ids_str) > 72:
            ids_str = ids_str[:69] + "..."
        print(f"  {label:<22} {n:>4}  {pct:>7.1f}%  {ids_str}")

    _fmt_row("BLAST only",          "BLAST_only")
    _fmt_row("M4 only",             "M4_only")
    _fmt_row("Both (AND)",          "Both_intersection")
    _fmt_row("Neither",             "Neither")
    print(f"  {'-'*102}")
    _fmt_row("Union (OR)",          "Union")

    print()
    print(f"  Coverage (rank-1 correct on Tier-A/B panel):")
    print(f"    BLAST alone      = {100*blast_cov:5.1f}%  ({blast_only + both}/{total})")
    print(f"    M4 alone         = {100*m4_cov:5.1f}%  ({m4_only + both}/{total})")
    print(f"    Intersection     = {100*intersect_cov:5.1f}%  ({both}/{total})")
    print(f"    Union (stacked)  = {100*union_cov:5.1f}%  ({union}/{total})")
    print()
    print(f"  Stacked-deployment gain:")
    print(f"    Union - BLAST    = {100*stacked_vs_blast:+5.1f}pp  "
          f"(M4 uniquely rescues {m4_only}/{total} cases BLAST misses)")
    print(f"    Union - M4       = {100*stacked_vs_m4:+5.1f}pp  "
          f"(BLAST uniquely rescues {blast_only}/{total} cases M4 misses)")


def latex_table_blast_m4_union():
    """LaTeX: Fig S-X companion table -- BLAST vs M4 contingency.

    Renders the 2x2 contingency with per-cell case-ID lists and three
    coverage summary rows (BLAST alone, M4 alone, Union). Used to
    quantitatively substantiate the Sec. 3A.5 stacked-deployment
    recommendation (BLAST-first-then-M4-tie-break).
    """
    data = _load_blast_m4_union_csv()
    if not data:
        return ""

    total = data.get("Total_scored", {}).get("count", 0)
    blast_only = data.get("BLAST_only", {}).get("count", 0)
    m4_only = data.get("M4_only", {}).get("count", 0)
    both = data.get("Both_intersection", {}).get("count", 0)
    neither = data.get("Neither", {}).get("count", 0)
    union = data.get("Union", {}).get("count", 0)

    blast_cov = (blast_only + both) / total if total else 0.0
    m4_cov = (m4_only + both) / total if total else 0.0
    union_cov = union / total if total else 0.0
    intersect_cov = both / total if total else 0.0

    def _ids(cell_key, max_len=60):
        ids = data.get(cell_key, {}).get("case_ids", [])
        if not ids:
            return "--"
        s = ", ".join(ids)
        # LaTeX-safe: case IDs are alphanumeric (GC###) so no escaping needed.
        # Keep table width manageable by truncating overlong lists.
        if len(s) > max_len:
            head = ", ".join(ids[:6])
            return f"{head}, \\ldots ({len(ids)} total)"
        return s

    def _pct(num, den):
        if not den:
            return "--"
        return f"{100*num/den:.1f}\\%"

    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{BLAST vs M4 stacked-deployment contingency "
                 r"on the Tier-A/B greatcases panel (Fig.~S-X companion).} "
                 r"Cells report how many cases each method places the correct "
                 r"mouse target at rank-1. The union row $\{$BLAST $\cup$ M4$\}$ "
                 f"covers {union}/{total} cases ({100*union_cov:.1f}\\%), "
                 f"versus {blast_only+both}/{total} ({100*blast_cov:.1f}\\%) "
                 f"for BLAST alone and {m4_only+both}/{total} "
                 f"({100*m4_cov:.1f}\\%) for M4 alone, yielding "
                 f"{100*(union_cov-blast_cov):+.1f}pp incremental coverage "
                 r"over BLAST. This empirically supports the Sec.~3A.5 "
                 r"recommendation to deploy BLAST first and M4 as a "
                 r"complementary tie-breaker on non-ortholog hard cases. "
                 f"Panel: {total} of the 46 Tier-A/B greatcases had "
                 r"rank-1 outputs from both BLAST and M4 (the remaining "
                 r"cases had at least one method return NaN due to "
                 r"missing naive-ortholog anchor or unscored input) and "
                 r"are included here.}")
    lines.append(r"\label{tab:blast_m4_union}")
    lines.append(r"\begin{tabular}{lrrl}")
    lines.append(r"\toprule")
    lines.append(r"Cell & n & \% of n & Case IDs \\")
    lines.append(r"\midrule")
    # 2x2 contingency
    lines.append(
        f"BLAST only           & {blast_only} & {_pct(blast_only, total)} & "
        f"{_ids('BLAST_only')} \\\\"
    )
    lines.append(
        f"M4 only              & {m4_only}    & {_pct(m4_only, total)}    & "
        f"{_ids('M4_only')} \\\\"
    )
    lines.append(
        f"Both (BLAST $\\cap$ M4) & {both} & {_pct(both, total)} & "
        f"{_ids('Both_intersection')} \\\\"
    )
    lines.append(
        f"Neither              & {neither}    & {_pct(neither, total)}    & "
        f"{_ids('Neither')} \\\\"
    )
    lines.append(r"\midrule")
    # Coverage summary rows
    lines.append(
        f"BLAST alone          & {blast_only + both} & "
        f"{_pct(blast_only + both, total)} & (BLAST\\_only $\\cup$ Both) \\\\"
    )
    lines.append(
        f"M4 alone             & {m4_only + both}    & "
        f"{_pct(m4_only + both, total)} & (M4\\_only $\\cup$ Both) \\\\"
    )
    lines.append(
        f"Union (BLAST $\\cup$ M4) & \\textbf{{{union}}} & "
        f"\\textbf{{{_pct(union, total)}}} & "
        f"(stacked deployment) \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_ooc_master():
    """Render a Markdown section for the OOC evaluation master table.

    Reads benchmarks/results/ooc_results_summary.csv (produced by
    ooc_evaluation.py) and prints a human-readable Markdown block.
    """
    path = RESULTS / "ooc_results_summary.csv"
    if not path.exists():
        print("\n[table_ooc_master] ooc_results_summary.csv not found — "
              "run benchmarks/ooc_evaluation.py first.")
        return ""

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def fmt(v):
        if v in ("", None):
            return "--"
        try:
            return f"{100*float(v):.1f}%"
        except (TypeError, ValueError):
            return str(v)

    md_lines = []
    md_lines.append("## Ortholog-Override Criterion (OOC) results\n")
    md_lines.append(
        "All numbers are accuracies (%) on the 57 greatcases + drug_ext panel.\n"
    )
    md_lines.append(
        "| Method | n | Strict@10 | Paralog@10 | Module@20 | "
        "OOC overall | OOC-A | OOC-B | OOC-C |"
    )
    md_lines.append(
        "|--------|---|-----------|-----------|-----------|"
        "-------------|-------|-------|-------|"
    )
    for r in rows:
        name = r["method"]
        bold = name == "M4"
        disp = f"**{name}**" if bold else name
        md_lines.append(
            f"| {disp} | {r['n_cases_scored']} | "
            f"{fmt(r['strict_accuracy_K10'])} | "
            f"{fmt(r['paralog_accuracy_K10'])} | "
            f"{fmt(r['module_accuracy_K20'])} | "
            f"{fmt(r['ooc_overall_accuracy'])} | "
            f"{fmt(r['ooc_A_promote_accuracy'])} | "
            f"{fmt(r['ooc_B_substitute_accuracy'])} | "
            f"{fmt(r['ooc_C_compress_accuracy'])} |"
        )
    md = "\n".join(md_lines) + "\n"
    print(f"\n{md}")
    return md


def main():
    print("Loading results...")
    b1 = get_bench1()
    b2 = get_bench2()
    b3 = get_bench3()
    b5 = get_bench5()
    fb = get_funcbench()
    hp = get_hardpairs()
    pub_ext = get_published_extended()

    print(f"  Bench1 methods loaded: {sorted(b1.keys())}")
    print(f"  Bench2 methods loaded: {sorted(b2.keys())}")
    print(f"  Bench3 methods loaded: {sorted(b3.keys())}")
    print(f"  Bench5 methods loaded: {sorted(b5.keys())}")
    print(f"  FuncBench methods loaded: {sorted(fb.keys())}")
    print(f"  HardPairs methods loaded: {sorted(hp.keys())}")
    print(f"  Published extended methods: {sorted(pub_ext.keys())}")

    # ── Plain-text tables ──
    table1_bench1_comparison(b1)
    table2_ablation(b1, b2, b3, b5)
    table3_multibench(b1, b2, b3, b5)

    # ── Bench3 detail ──
    print(f"\n{'='*110}")
    print("TABLE 4  -- Bench3 Essentiality Concordance Details")
    print(f"{'='*110}")
    print(f"  {'Method':<25}  {'Ess-Hitrate@5':>14} {'Gap@5':>8} {'Gap@10':>8} {'MWU p@5':>12}")
    print(f"  {'-'*75}")
    order = ["Ensembl_Compara","SVD_Annotation","2hop_Overlap","OrthoFinder_proxy","M1","M2","M3","M4"]
    for key in order:
        m = b3.get(key, {})
        gap5  = m.get("prec_gap5")
        pval  = m.get("mwu_p5")
        hit5  = m.get("ess_hit5")
        star = "* " if key == "M4" else "  "
        if pval and pval < 1e-10:
            exponent = int(round(-math.log10(pval))) - 1
            pstr = f"<1e-{exponent}"
        elif pval:
            pstr = f"{pval:.2e}"
        else:
            pstr = "   --"
        print(f"{star}{key:<23}  "
              f"{rp(hit5):>14}  "
              f"{f'{100*gap5:+.2f}%' if gap5 is not None else '   --':>8}  "
              f"{pstr:>12}")
    print(f"\n  Background rate (random): {33.41:.2f}%")
    print(f"  Gap@5 = (ess_precision@5 for essential genes) - (ess_precision@5 for non-essential)")
    print(f"  MWU = Mann-Whitney U test (two-sided) on rank distributions")

    # ── Bench1 published methods summary ──
    print(f"\n{'='*110}")
    print("KEY FINDING: Published GNN Method Proxies on Bench1")
    print(f"{'='*110}")
    pub_methods = [
        ("HGT-standard (CAME base)",   "PUB_HGT_standard_v2"),
        ("HGT+CrossSpec (CAME-insp.)", "PUB_HGT_crossspec_v3"),
        ("FGW-OT (SATURN-inspired)",   "PUB_FGW_OT_only"),
        ("KGE-SVD (GenePlexus-insp.)", "PUB_KGE_SVD"),
        ("SVD-Annotation (2-hop)",     "SVD_Annotation"),
        ("2-hop Overlap",              "2hop_Overlap"),
        ("OrthoFinder",                "OrthoFinder_proxy"),
        ("Ensembl Compara",            "Ensembl_Compara"),
        ("M4 (OURS)",                  "M4"),
    ]
    print(f"  {'Method':<35} {'R@5':>7} {'R@10':>8} {'R@20':>8} {'MRR':>10}")
    print(f"  {'-'*75}")
    for disp, key in pub_methods:
        m = b1.get(key, {})
        star = ">>>" if key == "M4" else "   "
        print(f"{star}{disp:<35} "
              f"{rp(m.get('r5')):>7} {rp(m.get('r10')):>8} {rp(m.get('r20')):>8} "
              f"{m.get('mrr',0):10.4f}")

    # ── NEW tables: FuncBench, HardPairs, Master summary, Published ext ──
    table5_funcbench(fb)
    table6_hardpairs(hp)
    table7_master(b1, b2, b3, b5, fb, hp)
    table8_published_extended(pub_ext, fb, hp, b1)

    # ── OOC master table (Part 2/3 of the paper) ──
    ooc_md = table_ooc_master()
    if ooc_md:
        (RESULTS / "ooc_master_table.md").write_text(ooc_md, encoding="utf-8")

    # ── BLAST vs M4 stacked-deployment contingency (Fig S-X companion, Part 3) ──
    table_blast_m4_union()

    # ── Save LaTeX ──
    latex1 = latex_table1(b1)
    latex2 = latex_table2(b1, b2, b3, b5)
    latex_cliff = latex_table_cliff(fb, hp)
    latex_pub = latex_table_published(pub_ext, fb, hp, b1)
    latex_bm4 = latex_table_blast_m4_union()
    latex_benchmarks = latex_table1_benchmarks()
    latex_master = latex_table7_master(b1, b2, b3, b5, fb, hp)

    out_dir = RESULTS
    (out_dir / "latex_table1_bench1.tex").write_text(latex1)
    (out_dir / "latex_table2_ablation.tex").write_text(latex2)
    (out_dir / "latex_table_seqcliff.tex").write_text(latex_cliff)
    (out_dir / "latex_table_published_extended.tex").write_text(latex_pub)
    (out_dir / "latex_table1_benchmarks.tex").write_text(
        latex_benchmarks, encoding="utf-8"
    )
    (out_dir / "latex_table7_master.tex").write_text(
        latex_master, encoding="utf-8"
    )
    if latex_bm4:
        (out_dir / "latex_table_blast_m4_union.tex").write_text(
            latex_bm4, encoding="utf-8"
        )

    # ── Save comprehensive JSON ──
    summary = {
        "bench1":             b1,
        "bench2":             b2,
        "bench3":             b3,
        "bench5":             b5,
        "funcbench":          fb,
        "hardpairs":          hp,
        "published_extended": pub_ext,
    }
    (out_dir / "paper_comprehensive_results.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: None)
    )

    print(f"\n\nSaved to {out_dir}/:")
    print(f"  latex_table1_benchmarks.tex")
    print(f"  latex_table1_bench1.tex")
    print(f"  latex_table2_ablation.tex")
    print(f"  latex_table7_master.tex")
    print(f"  latex_table_seqcliff.tex")
    print(f"  latex_table_published_extended.tex")
    if latex_bm4:
        print(f"  latex_table_blast_m4_union.tex")
    print(f"  paper_comprehensive_results.json")


if __name__ == "__main__":
    main()
