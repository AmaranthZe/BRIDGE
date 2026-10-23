"""Compute Kendall tau between successive model stages for Fig 5(c)."""
import json, os, sys
from scipy.stats import kendalltau
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "benchmarks", "results")

def load_ranks(bench, model):
    path = os.path.join(RESULTS, f"{bench}_{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    # bench1: 'cases' with 'fgw_rank'
    if "cases" in d and d["cases"]:
        return {c["case"]: c["fgw_rank"] for c in d["cases"] if "fgw_rank" in c}
    # bench5: 'pairs' with 'rank'
    if "pairs" in d and d["pairs"]:
        return {f"{p['h_sym']}->{p['m_sym']}": p["rank"] for p in d["pairs"] if "rank" in p}
    return None

def compute_tau(bench, ma, mb):
    ra = load_ranks(bench, ma)
    rb = load_ranks(bench, mb)
    if ra is None or rb is None:
        return None, None, 0
    common = sorted(set(ra.keys()) & set(rb.keys()))
    if len(common) < 3:
        return None, None, len(common)
    va = [ra[c] for c in common]
    vb = [rb[c] for c in common]
    tau, pval = kendalltau(va, vb)
    return round(tau, 3), round(pval, 6), len(common)

models = ["M1", "M2", "M3", "M4"]
benchmarks = ["bench1", "bench5"]

print("=== Kendall Tau Heatmap Data (Fig 5c) ===")
print()
print(f"{'Transition':<12} {'Bench1 (n=45)':<20} {'Bench5 (n=374)':<20}")
print("-" * 55)

results = {}
for i in range(len(models) - 1):
    ma, mb = models[i], models[i + 1]
    label = f"{ma}->{mb}"
    row = {}
    vals = []
    for bench in benchmarks:
        tau, pval, n = compute_tau(bench, ma, mb)
        row[bench] = {"tau": tau, "p": pval, "n": n}
        if tau is not None:
            vals.append(f"tau={tau:.3f} (p={pval:.4f})")
        else:
            vals.append("N/A")
    results[label] = row
    print(f"{label:<12} {vals[0]:<20} {vals[1]:<20}")

# Also compute direct M1->M4
tau14, p14, n14 = compute_tau("bench1", "M1", "M4")
tau54, p54, n54 = compute_tau("bench5", "M1", "M4")
print()
print(f"M1->M4 direct: Bench1 tau={tau14:.3f}, Bench5 tau={tau54:.3f}")

# Save results
output = {
    "kendall_tau": results,
    "direct_M1_M4": {
        "bench1": {"tau": tau14, "p": p14, "n": n14},
        "bench5": {"tau": tau54, "p": p54, "n": n54},
    },
}
outpath = os.path.join(BASE, "manuscript", "figures_precompute", "out", "kendall_tau_stages.json")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {outpath}")
