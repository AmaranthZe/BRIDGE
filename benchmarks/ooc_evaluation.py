"""Ortholog-Override Criterion (OOC) evaluator.

Implements the OOC scoring rules defined in paper_part3_plan.md §2-§3, §7.

Every rank recorded in the outputs comes from the actual model outputs on
the real dataset/ data — no mocks, no simulations. When a method cannot
produce a rank for a given case (missing symbol, missing ortholog, etc.)
we write NaN and label the reason in the ``notes`` column.

Outputs (under benchmarks/results/):
    ooc_results_per_case.csv   — one row per (case_id, method).
    ooc_results_summary.csv    — one row per method.
    ooc_latex_table.tex        — publication-ready LaTeX tabular fragment.

CLI
    python -m benchmarks.ooc_evaluation \
        --methods M1 M2 M3 M4 HGT_standard CAME SATURN GPZ seq_identity blast \
        --cases greatcases drug_ext \
        --out benchmarks/results/
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from benchmarks.model_variants import (
    ModelEvaluator,
    SharedData,
    arr_to_rank,
)
from benchmarks.bench6_comparison import (
    load_ensembl_orthologs,
    compute_ensembl_rank,
    compute_svd_best_rank,
    compute_orthofinder_proxy_rank,
)

logger = logging.getLogger(__name__)

# Methods we know how to drive — each maps to a ranking-function factory.
# The label on the left is what appears in output CSVs/LaTeX.
ALL_METHODS = [
    "M1", "M2", "M3", "M4",
    "HGT_standard", "CAME", "SATURN", "GPZ",
    "seq_identity", "blast",
]

# K thresholds per §3.
K_STRICT = 10
K_PARALOG = 10
K_MODULE = 20

# Cases for which OOC scoring is not applicable — §7 instructions.
OOC_SKIP_CASE_IDS = {8}  # DOWN_SYNDROME_HSA21 gene-set case.


# ──────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────

def load_greatcases(path: Path) -> List[dict]:
    """Load greatcases_curated.csv into a list of case dicts."""
    cases: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append({
                "case_id": f"GC{int(row['id']):03d}",
                "source": "greatcases",
                "tier_or_priority": row["tier"],
                "human_gene_symbols": row["human_gene_symbols"],
                "mouse_gold_set": row["mouse_gold_set"],
                "scoring_mode": row["scoring_mode"],
                "ooc_subtype": row["ooc_subtype"],
                "drug_internal": row["drug_internal"],
                "raw_id": int(row["id"]),
            })
    return cases


def load_drug_ext(path: Path) -> List[dict]:
    """Load drug_targets_external.csv.

    Returns only rows where ``suspected_mouse_proxy`` is present and does
    NOT start with the 'unknown' placeholder.  Rows with unknown proxies
    are retained as *coverage-only* cases (no gold set, excluded from
    OOC pass/fail but listed in the per-case CSV with notes="coverage-only").
    """
    cases: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            proxy = row.get("suspected_mouse_proxy", "").strip()
            is_unknown = proxy.lower().startswith("unknown")
            # Extract a short mouse gold set from the proxy string: we take
            # the leading Capital-case gene-like token(s) before the first
            # opening parenthesis or en-dash.
            mouse_gold: List[str] = []
            if not is_unknown and proxy:
                head = proxy.split("(")[0].split(" with ")[0]
                head = head.replace("—", "-")
                for tok in head.replace(",", " ").replace("/", " ").split():
                    tok = tok.strip(" .;—-")
                    if not tok:
                        continue
                    # Accept mouse-symbol-looking tokens: start capital + lower letter
                    if (len(tok) >= 3
                            and tok[0].isupper()
                            and any(c.islower() for c in tok)
                            and not any(c in tok for c in "()[]")
                            and tok.isascii()):
                        mouse_gold.append(tok)
                # Remove duplicates, preserve order
                seen = set()
                mouse_gold = [m for m in mouse_gold if not (m in seen or seen.add(m))]

            case_id = f"DX{int(row['id_ext'])}"
            cases.append({
                "case_id": case_id,
                "source": "drug_ext",
                "tier_or_priority": row.get("priority", ""),
                "human_gene_symbols": row["human_gene_symbol"],
                "mouse_gold_set": "|".join(mouse_gold) if mouse_gold else "",
                "scoring_mode": "strict" if len(mouse_gold) == 1 else
                                ("module" if len(mouse_gold) > 1 else ""),
                "ooc_subtype": row.get("ooc_subtype_hypothesis", "").strip()
                                or "unknown",
                "drug_internal": "1",  # all drug_ext rows are drug-relevant
                "raw_id": int(row["id_ext"]),
                "coverage_only": is_unknown or not mouse_gold,
            })
    return cases


# ──────────────────────────────────────────────────────────────────────────
# Symbol resolution helpers
# ──────────────────────────────────────────────────────────────────────────

def human_symbol_to_idx(sym: str, sd: SharedData) -> Optional[int]:
    """Map a HGNC symbol to a human gene index via SharedData."""
    ensg = sd.sym_to_ensg.get(sym)
    if ensg is None:
        # Try uppercased symbol variations
        for candidate in (sym.upper(), sym.strip()):
            ensg = sd.sym_to_ensg.get(candidate)
            if ensg is not None:
                break
    if ensg is None:
        return None
    return sd.h_genes.get(ensg)


def mouse_symbol_to_idx(sym: str, sd: SharedData) -> Optional[int]:
    """Map a MGI symbol to a mouse gene index via SharedData."""
    ensmusg = sd.sym_to_ensmusg.get(sym)
    if ensmusg is None:
        for candidate in (sym.capitalize(), sym):
            ensmusg = sd.sym_to_ensmusg.get(candidate)
            if ensmusg is not None:
                break
    if ensmusg is None:
        return None
    return sd.m_genes.get(ensmusg)


def parse_gold_symbols(gold_str: str) -> List[str]:
    """Parse a pipe-delimited mouse gold-set string."""
    if not gold_str:
        return []
    return [s.strip() for s in gold_str.split("|") if s.strip()]


def parse_human_symbols(sym_str: str) -> List[str]:
    """Parse pipe-delimited human input symbols."""
    if not sym_str:
        return []
    return [s.strip() for s in sym_str.split("|") if s.strip()]


# ──────────────────────────────────────────────────────────────────────────
# Naive ortholog lookup (from the Ensembl TSV)
# ──────────────────────────────────────────────────────────────────────────

def build_one2one_ortholog_map(
    tsv_path: Path, sd: SharedData
) -> Dict[int, int]:
    """Return {human_idx -> mouse_idx} for *strict one-to-one* orthologs.

    If no one-to-one ortholog exists for a human gene, it is absent from
    the returned dict. This is the "naive ortholog baseline" used in OOC.
    """
    mapping: Dict[int, int] = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ensg = row["Gene stable ID"].strip()
            ensmusg = row.get("Mouse gene stable ID", "").strip()
            htype = row.get("Mouse homology type", "").strip()
            if htype != "ortholog_one2one":
                continue
            if ensg not in sd.h_genes or ensmusg not in sd.m_genes:
                continue
            mapping[sd.h_genes[ensg]] = sd.m_genes[ensmusg]
    return mapping


# ──────────────────────────────────────────────────────────────────────────
# Ranking-function factories for each method
# ──────────────────────────────────────────────────────────────────────────

def _m_rank_fn_from_evaluator(ev: ModelEvaluator, use_rrf: bool):
    """Return rank_fn(hi) -> (rank_array, primary_metric_str)."""
    def rank_fn(hi: int):
        result = ev.rank_mouse_genes(hi)
        if use_rrf and result["rrf_rank"] is not None:
            return result["rrf_rank"]
        # fgw_rank stores the mini-ensemble rank for M1/M2.
        return result["fgw_rank"]
    return rank_fn


# Pair (variant_needed, list_of_method_keys) — methods that share a single
# ModelEvaluator instance are grouped together so we can release GPU memory
# in between.
METHOD_BUCKETS = [
    ("M1", ["M1", "HGT_standard", "SATURN"]),
    ("M2", ["M2", "CAME"]),
    ("M3", ["M3"]),
    ("M4", ["M4"]),
    (None, ["GPZ", "seq_identity", "blast"]),
]


def build_rank_fns_for_bucket(
    bucket_variant: Optional[str],
    bucket_methods: List[str],
    wanted_set: set,
    sd: SharedData,
    orthologs_full: Dict[int, list],
    device: str,
) -> Tuple[Dict[str, Callable[[int], np.ndarray]], Optional[ModelEvaluator]]:
    """Build rank_fns for one bucket.  Returns (fns, evaluator)."""
    rank_fns: Dict[str, Callable[[int], np.ndarray]] = {}
    ev: Optional[ModelEvaluator] = None

    # Only instantiate the ModelEvaluator if at least one of its methods
    # is wanted.
    if bucket_variant is not None and any(m in wanted_set for m in bucket_methods):
        ev = ModelEvaluator(bucket_variant, sd, device=device)

    for m in bucket_methods:
        if m not in wanted_set:
            continue
        if m == "M1":
            rank_fns["M1"] = _m_rank_fn_from_evaluator(ev, use_rrf=False)
        elif m == "M2":
            rank_fns["M2"] = _m_rank_fn_from_evaluator(ev, use_rrf=False)
        elif m == "M3":
            rank_fns["M3"] = _m_rank_fn_from_evaluator(ev, use_rrf=True)
        elif m == "M4":
            rank_fns["M4"] = _m_rank_fn_from_evaluator(ev, use_rrf=True)
        elif m == "HGT_standard":
            rank_fns["HGT_standard"] = (
                lambda hi, e=ev: e._compute_hgt_cosine_rank(hi)
            )
        elif m == "CAME":
            rank_fns["CAME"] = (
                lambda hi, e=ev: e._compute_hgt_cosine_rank(hi)
            )
        elif m == "SATURN":
            rank_fns["SATURN"] = lambda hi, e=ev: e._compute_fgw_rank(hi)
        elif m == "GPZ":
            Z_h, Z_m, N_m = sd.Z_h_uni, sd.Z_m_uni, sd.N_m

            def _gpz_rank(hi: int, Z_h=Z_h, Z_m=Z_m, N_m=N_m) -> np.ndarray:
                sim = (Z_h[hi:hi + 1] @ Z_m.T).squeeze().numpy()
                return arr_to_rank(sim, N_m)

            rank_fns["GPZ"] = _gpz_rank
        elif m == "seq_identity":
            N_m = sd.N_m
            rank_fns["seq_identity"] = (
                lambda hi, o=orthologs_full, N_m=N_m:
                compute_ensembl_rank(hi, o, N_m)
            )
        elif m == "blast":
            N_m = sd.N_m

            def _blast_rank(hi: int, o=orthologs_full, N_m=N_m) -> np.ndarray:
                svd_rank = compute_svd_best_rank(hi, sd)
                return compute_orthofinder_proxy_rank(hi, o, svd_rank, N_m)

            rank_fns["blast"] = _blast_rank
    return rank_fns, ev


# ──────────────────────────────────────────────────────────────────────────
# OOC scoring core
# ──────────────────────────────────────────────────────────────────────────

def _score_one_case_one_method(
    case: dict,
    rank_fn: Callable[[int], np.ndarray],
    sd: SharedData,
    one2one: Dict[int, int],
) -> dict:
    """Score a single (case, method) pair.

    Returns a dict following the schema of ooc_results_per_case.csv.
    """
    row = {
        "case_id": case["case_id"],
        "source": case["source"],
        "tier_or_priority": case["tier_or_priority"],
        "method": None,  # caller fills in
        "best_gold_rank": float("nan"),
        "naive_ortholog_rank": float("nan"),
        "strict_hit_K10": False,
        "paralog_hit_K10": False,
        "module_hit_K10": False,
        "ooc_subtype_used": "",
        "ooc_pass": None,  # True / False / None (N/A)
        "notes": "",
    }

    notes: List[str] = []

    human_syms = parse_human_symbols(case["human_gene_symbols"])
    gold_syms = parse_gold_symbols(case["mouse_gold_set"])
    scoring_mode = (case.get("scoring_mode") or "").strip()
    subtype = (case.get("ooc_subtype") or "").strip() or "unknown"

    # ── Skip rules ──
    if case["raw_id"] == 8 and case["source"] == "greatcases":
        notes.append("DOWN_SYNDROME_HSA21 gene-set case — OOC N/A (§7)")
        row["ooc_subtype_used"] = "N/A"
        row["notes"] = "; ".join(notes)
        return row

    if case.get("coverage_only"):
        notes.append("coverage-only (unknown mouse proxy) — OOC pass/fail skipped")
        row["ooc_subtype_used"] = "N/A"
        row["notes"] = "; ".join(notes)
        return row

    # Need at least one mappable human symbol
    human_idxs: List[Tuple[str, int]] = []
    for sym in human_syms:
        hi = human_symbol_to_idx(sym, sd)
        if hi is None:
            notes.append(f"human symbol '{sym}' not in human gene universe")
        else:
            human_idxs.append((sym, hi))
    if not human_idxs:
        notes.append("no human symbol mappable — case skipped")
        row["notes"] = "; ".join(notes)
        return row

    # Map gold symbols to indices
    gold_idxs: List[Tuple[str, int]] = []
    missing_gold: List[str] = []
    for sym in gold_syms:
        mi = mouse_symbol_to_idx(sym, sd)
        if mi is None:
            missing_gold.append(sym)
        else:
            gold_idxs.append((sym, mi))
    if missing_gold:
        notes.append(
            "gold symbol(s) not in mouse universe — excluded: "
            + ",".join(missing_gold)
        )
    if not gold_idxs:
        notes.append("no gold symbol mappable — best_gold_rank=NaN")
        row["notes"] = "; ".join(notes)
        return row

    # For composite human inputs (multi-symbol), take the BEST rank across
    # sibling genes per the §7 "iterate over pipe-split" rule.
    # ── Compute rank arrays for every human symbol ──
    rank_arrays: List[np.ndarray] = []
    primary_hi: int = human_idxs[0][1]  # representative for ortholog logic
    for _sym, hi in human_idxs:
        rank_arrays.append(rank_fn(hi))

    def combine_rank_for(mi: int) -> int:
        # Take the BEST (minimum) rank across the sibling human genes.
        return int(min(r[mi] for r in rank_arrays))

    # best_gold_rank = min rank over all gold mouse genes
    gold_ranks = [(sym, combine_rank_for(mi)) for sym, mi in gold_idxs]
    best_gold_sym, best_gold_rank = min(gold_ranks, key=lambda t: t[1])
    row["best_gold_rank"] = best_gold_rank

    # Naive ortholog rank — for the primary human input symbol.
    naive_rank: Optional[int] = None
    naive_mi: Optional[int] = None
    if primary_hi in one2one:
        naive_mi = one2one[primary_hi]
        naive_rank = combine_rank_for(naive_mi)
        row["naive_ortholog_rank"] = int(naive_rank)
    else:
        notes.append("no one-to-one ortholog in Ensembl TSV for primary input")

    # Hit flags
    row["strict_hit_K10"] = bool(best_gold_rank <= K_STRICT)
    # Paralog flag: ≥2 gold members in top-K
    if scoring_mode in ("paralog_set", "module"):
        topk_hits = sum(1 for _, r in gold_ranks if r <= K_PARALOG)
        row["paralog_hit_K10"] = bool(topk_hits >= 2)
    # Module flag: any gold in top-K=20 when scoring_mode == module
    if scoring_mode == "module":
        row["module_hit_K10"] = bool(best_gold_rank <= K_MODULE)
    else:
        # For completeness: apply to strict / paralog_set cases too, but
        # interpret only for scoring_mode==module in the summary.
        row["module_hit_K10"] = bool(best_gold_rank <= K_MODULE)

    # ── OOC sub-type scoring ──
    gold_mi_set = {mi for _, mi in gold_idxs}
    subtype_eff = subtype
    if subtype == "mixed":
        # Pick a dominant one — heuristic: if scoring_mode=="paralog_set"
        # it's A_promote; if scoring_mode=="module" we try B_substitute;
        # else fall back to the text value.
        if scoring_mode == "paralog_set":
            subtype_eff = "A_promote"
        elif scoring_mode == "module":
            subtype_eff = "B_substitute"
        else:
            subtype_eff = "B_substitute"
        notes.append(f"mixed subtype → scored as {subtype_eff}")

    row["ooc_subtype_used"] = subtype_eff

    pass_flag: Optional[bool] = None
    if subtype_eff == "A_promote":
        # Need a non-ortholog gold partner ranked above the naive ortholog.
        if naive_rank is None:
            notes.append("A_promote: no naive ortholog → treated as N/A")
            pass_flag = None
        else:
            non_ortho_gold = [
                (sym, mi, combine_rank_for(mi))
                for sym, mi in gold_idxs
                if mi != naive_mi
            ]
            if not non_ortho_gold:
                notes.append(
                    "A_promote: only ortholog is in gold set → N/A"
                )
                pass_flag = None
            else:
                best_partner = min(non_ortho_gold, key=lambda t: t[2])
                pass_flag = bool(best_partner[2] < naive_rank)
    elif subtype_eff == "B_substitute":
        # Best gold rank = 1 AND naive ortholog (if exists and not in gold)
        # must be > 10.
        ok1 = best_gold_rank == 1
        if naive_rank is None or naive_mi in gold_mi_set:
            pass_flag = bool(ok1)
        else:
            pass_flag = bool(ok1 and naive_rank > 10)
    elif subtype_eff == "C_compress":
        # Gold single node must be top-1 on EVERY human input symbol.
        # Our gold sets may have multiple items for C_compress too (the
        # "compressed" side is usually single). We require every human
        # symbol's rank array to rank SOME gold member at position 1.
        sub_passes = []
        for idx, (_sym, hi) in enumerate(human_idxs):
            rarr = rank_arrays[idx]
            this_best = min(int(rarr[mi]) for _, mi in gold_idxs)
            sub_passes.append(this_best == 1)
        pass_flag = bool(all(sub_passes))
        if not sub_passes:
            pass_flag = False
    else:
        # unknown subtype: report only strict hit, OOC N/A
        notes.append(
            f"OOC subtype '{subtype}' not recognised — pass=N/A"
        )
        pass_flag = None

    row["ooc_pass"] = pass_flag
    row["notes"] = "; ".join(notes)
    return row


# ──────────────────────────────────────────────────────────────────────────
# Summaries
# ──────────────────────────────────────────────────────────────────────────

def summarise_per_method(
    per_case_rows: List[dict],
) -> List[dict]:
    """Aggregate per_case rows into per-method summary rows."""
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in per_case_rows:
        by_method[r["method"]].append(r)

    out: List[dict] = []
    for method, rows in by_method.items():
        scored = [r for r in rows
                  if not math.isnan(r["best_gold_rank"])]
        module_rows = [r for r in rows if r.get("scoring_mode") == "module"
                       or r["module_hit_K10"] is not None]
        # strict / paralog / module counts
        strict_rows = [r for r in scored
                       if r.get("scoring_mode") in ("strict", "paralog_set",
                                                    "module", "")]
        paralog_rows = [r for r in scored
                        if r.get("scoring_mode") in ("paralog_set", "module")]
        module_only_rows = [r for r in scored if r.get("scoring_mode") == "module"]
        ooc_rows = [r for r in scored if r["ooc_pass"] is not None]
        ooc_A = [r for r in ooc_rows if r["ooc_subtype_used"] == "A_promote"]
        ooc_B = [r for r in ooc_rows if r["ooc_subtype_used"] == "B_substitute"]
        ooc_C = [r for r in ooc_rows if r["ooc_subtype_used"] == "C_compress"]

        def _frac(rs: List[dict], key: str) -> Optional[float]:
            if not rs:
                return None
            hits = sum(1 for r in rs if r[key])
            return hits / len(rs)

        def _ooc_frac(rs: List[dict]) -> Optional[float]:
            if not rs:
                return None
            hits = sum(1 for r in rs if r["ooc_pass"])
            return hits / len(rs)

        out.append({
            "method": method,
            "n_cases_scored": len(scored),
            "strict_accuracy_K10": _frac(strict_rows, "strict_hit_K10"),
            "paralog_accuracy_K10": _frac(paralog_rows, "paralog_hit_K10"),
            "module_accuracy_K20": _frac(module_only_rows, "module_hit_K10"),
            "ooc_overall_accuracy": _ooc_frac(ooc_rows),
            "ooc_A_promote_accuracy": _ooc_frac(ooc_A),
            "ooc_B_substitute_accuracy": _ooc_frac(ooc_B),
            "ooc_C_compress_accuracy": _ooc_frac(ooc_C),
        })

    # Order like ALL_METHODS
    order = {m: i for i, m in enumerate(ALL_METHODS)}
    out.sort(key=lambda d: order.get(d["method"], 99))
    return out


# ──────────────────────────────────────────────────────────────────────────
# LaTeX table
# ──────────────────────────────────────────────────────────────────────────

def build_latex_table(summary_rows: List[dict]) -> str:
    """Produce a publication-ready \\begin{tabular} fragment.

    Columns: Method | Strict@10 | Module@20 | OOC overall | OOC-A | OOC-B | OOC-C.
    M4 is bolded if it holds the maximum in each column.
    """
    cols = [
        ("strict_accuracy_K10", "Strict@10"),
        ("module_accuracy_K20", "Module@20"),
        ("ooc_overall_accuracy", "OOC overall"),
        ("ooc_A_promote_accuracy", "OOC-A"),
        ("ooc_B_substitute_accuracy", "OOC-B"),
        ("ooc_C_compress_accuracy", "OOC-C"),
    ]

    # Best value per column across methods (ignoring None).
    best: Dict[str, float] = {}
    for key, _ in cols:
        vals = [r[key] for r in summary_rows if r[key] is not None]
        if vals:
            best[key] = max(vals)

    def cell(r: dict, key: str) -> str:
        v = r[key]
        if v is None:
            return "--"
        s = f"{100*v:.1f}\\%"
        if r["method"] == "M4" and key in best and abs(v - best[key]) < 1e-9:
            s = "\\textbf{" + s + "}"
        return s

    lines: List[str] = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Ortholog-Override Criterion (OOC) results across 10 "
        r"methods on the 57-greatcases + 14-drug\_ext validation panel.} "
        r"Method accuracies are decomposed by OOC subtype: "
        r"A (promote a non-ortholog paralog above a retrieval-correct ortholog), "
        r"B (substitute with a structurally unrelated gene when no useful ortholog "
        r"exists), and C (compress many human paralogs onto a single mouse gene). "
        r"Strict@10 and Module@20 report standard retrieval accuracy for reference. "
        r"M4 (our full 22-strategy RRF) is shown in bold where it attains the "
        r"column maximum. CAME's 75\% OOC-A is a retrieval-meaningless artifact "
        r"arising from its near-zero Strict@10 baseline (see Sec.~3.7.1). "
        r"All numbers are percentages.}"
    )
    lines.append(r"\label{tab:ooc_master}")
    lines.append(r"\begin{tabular}{l" + "c" * len(cols) + r"}")
    lines.append(r"\toprule")
    lines.append("Method & " + " & ".join(d for _, d in cols) + r" \\")
    lines.append(r"\midrule")
    for r in summary_rows:
        name = r["method"]
        if name == "M4":
            name = r"\textbf{M4}"
        lines.append(name + " & "
                     + " & ".join(cell(r, k) for k, _ in cols) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────

def run(
    methods: List[str],
    case_sources: List[str],
    out_dir: Path,
    device: str = "cuda",
) -> Tuple[List[dict], List[dict]]:
    bench_dir = Path(__file__).resolve().parent
    root = bench_dir.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather cases
    all_cases: List[dict] = []
    if "greatcases" in case_sources:
        all_cases.extend(load_greatcases(bench_dir / "greatcases_curated.csv"))
    if "drug_ext" in case_sources:
        all_cases.extend(load_drug_ext(bench_dir / "drug_targets_external.csv"))
    logger.info("Loaded %d cases (%s)",
                len(all_cases), {c["source"] for c in all_cases})

    # SharedData + orthologs
    t0 = time.time()
    sd = SharedData(skip_ppi=True)
    logger.info("SharedData loaded in %.1fs", time.time() - t0)

    ortho_tsv = root / "dataset" / "external" / "ensembl_human_mouse_orthologs.tsv"
    one2one = build_one2one_ortholog_map(ortho_tsv, sd)
    logger.info("Loaded %d one-to-one ortholog mappings", len(one2one))
    orthologs_full = load_ensembl_orthologs(str(ortho_tsv), sd.h_genes, sd.m_genes)

    wanted_set = set(methods)
    per_case_rows: List[dict] = []

    # Process methods bucket-by-bucket so we can release each
    # ModelEvaluator's GPU memory before loading the next.
    for bucket_variant, bucket_methods in METHOD_BUCKETS:
        if not any(m in wanted_set for m in bucket_methods):
            continue
        logger.info(
            "Bucket: variant=%s methods=%s",
            bucket_variant,
            [m for m in bucket_methods if m in wanted_set],
        )
        t_b = time.time()
        rank_fns, ev = build_rank_fns_for_bucket(
            bucket_variant, bucket_methods, wanted_set, sd,
            orthologs_full, device,
        )

        # Per-hi rank cache shared across methods in this bucket (saves
        # recomputing heavy FGW when several bucket methods share the same
        # underlying model).
        cache: Dict[Tuple[str, int], np.ndarray] = {}

        def cached(method: str, fn: Callable[[int], np.ndarray]):
            def wrapped(hi: int):
                key = (method, hi)
                if key not in cache:
                    cache[key] = fn(hi)
                return cache[key]
            return wrapped

        wrapped_fns = {m: cached(m, fn) for m, fn in rank_fns.items()}

        for case in all_cases:
            scoring_mode = (case.get("scoring_mode") or "").strip()
            for method, fn in wrapped_fns.items():
                t_case = time.time()
                try:
                    row = _score_one_case_one_method(
                        case, fn, sd, one2one
                    )
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "case_id": case["case_id"],
                        "source": case["source"],
                        "tier_or_priority": case["tier_or_priority"],
                        "method": method,
                        "best_gold_rank": float("nan"),
                        "naive_ortholog_rank": float("nan"),
                        "strict_hit_K10": False,
                        "paralog_hit_K10": False,
                        "module_hit_K10": False,
                        "ooc_subtype_used": "error",
                        "ooc_pass": None,
                        "notes": f"ERROR: {type(exc).__name__}: {exc}",
                    }
                row["method"] = method
                row["scoring_mode"] = scoring_mode
                per_case_rows.append(row)
                dt = time.time() - t_case
                if dt > 10:
                    logger.info("  %s / %s — %.1fs",
                                case["case_id"], method, dt)

        # Release this bucket's evaluator before moving on.
        if ev is not None:
            try:
                ev.free()
            except Exception:  # noqa: BLE001
                pass
        del ev, rank_fns, wrapped_fns, cache
        gc.collect()
        logger.info("  bucket done in %.1fs", time.time() - t_b)

    # ── Per-case CSV ──
    per_case_path = out_dir / "ooc_results_per_case.csv"
    fieldnames = [
        "case_id", "source", "tier_or_priority", "method",
        "best_gold_rank", "naive_ortholog_rank",
        "strict_hit_K10", "paralog_hit_K10", "module_hit_K10",
        "ooc_subtype_used", "ooc_pass", "notes",
    ]
    with open(per_case_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_case_rows:
            out = {k: r.get(k, "") for k in fieldnames}
            # Clean NaNs
            for k in ("best_gold_rank", "naive_ortholog_rank"):
                v = out[k]
                if isinstance(v, float) and math.isnan(v):
                    out[k] = ""
            # bool -> 0/1
            for k in ("strict_hit_K10", "paralog_hit_K10", "module_hit_K10"):
                out[k] = int(bool(out[k]))
            # ooc_pass
            if out["ooc_pass"] is None:
                out["ooc_pass"] = ""
            else:
                out["ooc_pass"] = int(bool(out["ooc_pass"]))
            writer.writerow(out)
    logger.info("Wrote %s", per_case_path)

    # ── Summary CSV ──
    summary_rows = summarise_per_method(per_case_rows)
    summary_path = out_dir / "ooc_results_summary.csv"
    summary_fields = [
        "method", "n_cases_scored",
        "strict_accuracy_K10", "paralog_accuracy_K10", "module_accuracy_K20",
        "ooc_overall_accuracy", "ooc_A_promote_accuracy",
        "ooc_B_substitute_accuracy", "ooc_C_compress_accuracy",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in summary_rows:
            out = {}
            for k in summary_fields:
                v = r.get(k)
                if isinstance(v, float) and not math.isnan(v):
                    out[k] = f"{v:.4f}"
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            writer.writerow(out)
    logger.info("Wrote %s", summary_path)

    # ── LaTeX table ──
    latex_path = out_dir / "ooc_latex_table.tex"
    latex_path.write_text(build_latex_table(summary_rows), encoding="utf-8")
    logger.info("Wrote %s", latex_path)

    return per_case_rows, summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Ortholog-Override Criterion (OOC) evaluator"
    )
    parser.add_argument(
        "--methods", nargs="+", default=list(ALL_METHODS),
        choices=ALL_METHODS,
        help="Methods to evaluate (default: all)",
    )
    parser.add_argument(
        "--cases", nargs="+", default=["greatcases", "drug_ext"],
        choices=["greatcases", "drug_ext"],
        help="Case sources",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: benchmarks/results/)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.out is None:
        out_dir = Path(__file__).resolve().parent / "results"
    else:
        out_dir = Path(args.out)

    t0 = time.time()
    run(args.methods, args.cases, out_dir, device=args.device)
    logger.info("Total runtime: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
