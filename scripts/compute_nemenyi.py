"""Compute Friedman test and Nemenyi CD for Fig 2d.

Uses per-benchmark R@5 across all methods, treating benchmarks as blocks.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import friedmanchisquare, rankdata

ROOT = Path(r"C:/Users/Username/Desktop/Project1_DiseaseGo2Gene/Project1_DiseaseGo2Gene")

with open(ROOT / "benchmarks/results/paper_comprehensive_results.json") as f:
    data = json.load(f)

# Methods that have data across all usable benchmarks
methods = ["M1", "M2", "M3", "M4",
           "Ensembl_Compara", "SVD_Annotation", "2hop_Overlap", "OrthoFinder_proxy"]

# Benchmarks where all 8 methods have R@5 data
# bench2 is stored at top-level "bench2" key (= bench2-all, n=509/512)
benchmarks = ["bench1", "bench2", "bench5", "funcbench"]

# Build matrix: rows=benchmarks, cols=methods, values=R@5
matrix = []
for bench in benchmarks:
    row = []
    for method in methods:
        if bench in data and method in data[bench]:
            row.append(data[bench][method]["r5"])
        else:
            row.append(np.nan)
    matrix.append(row)

matrix = np.array(matrix)
print("R@5 matrix (benchmarks x methods):")
print(f"{'':20s}", end="")
for m in methods:
    print(f"{m[:8]:>10s}", end="")
print()
for i, bench in enumerate(benchmarks):
    print(f"{bench:20s}", end="")
    for j in range(len(methods)):
        if np.isnan(matrix[i, j]):
            print(f"{'NaN':>10s}", end="")
        else:
            print(f"{matrix[i, j]:10.3f}", end="")
    print()

# Handle NaN: for methods not evaluated on certain benchmarks, use median
# Actually, check which benchmarks have all methods
valid_benchmarks = []
for i, bench in enumerate(benchmarks):
    if not np.any(np.isnan(matrix[i, :])):
        valid_benchmarks.append(i)
    else:
        # Check which methods have NaN
        nan_methods = [methods[j] for j in range(len(methods)) if np.isnan(matrix[i, j])]
        print(f"\n{bench}: missing methods: {nan_methods}")

# Use only benchmarks where all methods have values
if len(valid_benchmarks) >= 3:
    mat = matrix[valid_benchmarks, :]
    bench_names = [benchmarks[i] for i in valid_benchmarks]
    print(f"\nUsing {len(valid_benchmarks)} benchmarks: {bench_names}")
else:
    # Impute NaN with 0 for methods not evaluated
    mat = np.nan_to_num(matrix, nan=0.0)
    bench_names = benchmarks
    print(f"\nUsing all {len(benchmarks)} benchmarks (NaN -> 0)")

# Rank within each benchmark (higher R@5 = lower rank = better)
ranks = np.zeros_like(mat)
for i in range(mat.shape[0]):
    ranks[i] = rankdata(-mat[i])  # negative because higher is better

print(f"\nAverage ranks across {len(bench_names)} benchmarks:")
avg_ranks = ranks.mean(axis=0)
for j, method in enumerate(methods):
    print(f"  {method:25s}: {avg_ranks[j]:.2f}")

# Sort by average rank
sorted_idx = np.argsort(avg_ranks)
print("\nSorted (best to worst):")
for j in sorted_idx:
    print(f"  {methods[j]:25s}: {avg_ranks[j]:.2f}")

# Friedman test
try:
    stat, p = friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    print(f"\nFriedman chi-squared({len(methods)-1}) = {stat:.1f}, p = {p:.2e}")
except Exception as e:
    print(f"\nFriedman test error: {e}")
    stat, p = np.nan, np.nan

# Nemenyi critical difference
k = len(methods)
N = len(bench_names)
# q_alpha values for alpha=0.05 (Demsar 2006, Table 5)
# These are q/sqrt(2) from the Studentized Range distribution
q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
q_alpha = q_alpha_table.get(k, 3.031)  # default to k=8
CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
print(f"\nNemenyi CD (k={k}, N={N}, alpha=0.05): {CD:.2f}")
print(f"  q_alpha = {q_alpha} (for k={k} groups)")

# Save results
out = {
    "friedman_chi2": float(stat) if not np.isnan(stat) else None,
    "friedman_df": k - 1,
    "friedman_p": float(p) if not np.isnan(p) else None,
    "nemenyi_cd": float(CD),
    "n_benchmarks": N,
    "n_methods": k,
    "average_ranks": {methods[j]: round(float(avg_ranks[j]), 2) for j in sorted_idx},
    "benchmarks_used": bench_names,
}
outpath = ROOT / "manuscript/figures_precompute/out/nemenyi_cd.json"
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outpath}")
