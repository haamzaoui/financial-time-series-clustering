"""
CROSS-ALGORITHM CONSISTENCY
=============================
Compares clustering results pairwise across all algorithms using:

  ARI (Adjusted Rand Index)
    Measures label agreement between two clusterings, adjusted for
    chance. A score of 1 means identical assignments, 0 means no
    better than random, negative means worse than random.
    → higher is better  (range: -1 to 1)

  NMI (Normalized Mutual Information)
    Measures the shared information between two label sets,
    normalised to [0, 1]. Less sensitive to cluster size imbalance
    than ARI, making it a useful complement.
    → higher is better  (range: 0 to 1)

NOTE ON DBSCAN:
  DBSCAN produced 1 cluster + noise (label = -1). It is included
  in the comparison — a 1-cluster result with noise is a valid
  label set. ARI and NMI treat noise as its own group.

Input:
  results/kmeans_labels.npy
  results/hierarchical_labels.npy
  results/dbscan_labels.npy
Output:
  results/eval_consistency_matrix.png
  results/eval_consistency_report.txt
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

ROOT        = Path(__file__).resolve().parents[2]
SRC         = ROOT / "src"
RESULTS_DIR = Path(__file__).parent / "results"

print("\n" + "=" * 70)
print("CROSS-ALGORITHM CONSISTENCY  (ARI & NMI)")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_H5   = ROOT / "data/processed/sample_50k.h5"

LABEL_FILES = {
    "K-Means"      : SRC / "clustering/kmeans/results/kmeans_labels.npy",
    "Hierarchical" : SRC / "clustering/hierarchical/results/hierarchical_labels.npy",
    "DBSCAN"       : SRC / "clustering/dbscan/results/dbscan_labels.npy",
}


# ============================================================
# STEP 1: LOAD LABELS
# ============================================================

print("\n[1/4] Loading labels...")
print("-" * 70)

# Load N for noise % reporting
with h5py.File(SAMPLE_H5, "r") as f:
    N = f["segments"].shape[0]

all_labels = {}
for name, path in LABEL_FILES.items():
    try:
        lbl = np.load(path)
        all_labels[name] = lbl
        n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_noise    = int((lbl == -1).sum())
        print(f"✓ Loaded {name:<15}: {n_clusters} clusters, "
              f"{n_noise:,} noise points ({n_noise/N*100:.1f}%)")
    except FileNotFoundError:
        print(f"✗ {name:<15}: file not found — {path}")

if len(all_labels) < 2:
    raise RuntimeError("Need at least 2 label files to compare. "
                       "Run clustering scripts first.")

algo_names = list(all_labels.keys())
n_algos    = len(algo_names)


# ============================================================
# STEP 2: COMPUTE ARI AND NMI MATRICES
# ============================================================

print(f"\n[2/4] Computing ARI and NMI matrices ({n_algos}×{n_algos})...")
print("-" * 70)

ari_matrix = np.zeros((n_algos, n_algos))
nmi_matrix = np.zeros((n_algos, n_algos))

for i, name_i in enumerate(algo_names):
    for j, name_j in enumerate(algo_names):
        if i == j:
            ari_matrix[i, j] = 1.0
            nmi_matrix[i, j] = 1.0
        else:
            ari_matrix[i, j] = adjusted_rand_score(
                all_labels[name_i], all_labels[name_j]
            )
            nmi_matrix[i, j] = normalized_mutual_info_score(
                all_labels[name_i], all_labels[name_j]
            )

# Print ARI
print(f"\n  ARI matrix  (1 = identical, 0 = random):")
print(f"  {'':16}" + "".join(f"  {n:>14}" for n in algo_names))
print("  " + "-" * (18 + 16 * n_algos))
for i, name_i in enumerate(algo_names):
    row = f"  {name_i:<16}"
    for j in range(n_algos):
        row += f"  {ari_matrix[i, j]:>14.4f}"
    print(row)

# Print NMI
print(f"\n  NMI matrix  (1 = identical, 0 = no shared info):")
print(f"  {'':16}" + "".join(f"  {n:>14}" for n in algo_names))
print("  " + "-" * (18 + 16 * n_algos))
for i, name_i in enumerate(algo_names):
    row = f"  {name_i:<16}"
    for j in range(n_algos):
        row += f"  {nmi_matrix[i, j]:>14.4f}"
    print(row)


# ============================================================
# STEP 3: VISUALISATION — heatmaps
# ============================================================

print(f"\n[3/4] Creating heatmap visualisation...")
print("-" * 70)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fig, (ax_ari, ax_nmi) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cross-Algorithm Consistency", fontsize=13, fontweight="bold")

for ax, matrix, title in [
    (ax_ari, ari_matrix, "ARI  (Adjusted Rand Index)"),
    (ax_nmi, nmi_matrix, "NMI  (Normalized Mutual Information)"),
]:
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_algos))
    ax.set_yticks(range(n_algos))
    ax.set_xticklabels(algo_names, fontsize=11, fontweight="bold")
    ax.set_yticklabels(algo_names, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    for i in range(n_algos):
        for j in range(n_algos):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold",
                    color="black" if val > 0.3 else "white")

plt.tight_layout()
fig.savefig(RESULTS_DIR / "eval_consistency_matrix.png",
            dpi=300, bbox_inches="tight")
print(f"✓ Saved: results/eval_consistency_matrix.png")
plt.show()


# ============================================================
# STEP 4: REPORT
# ============================================================

print(f"\n[4/4] Saving report...")
print("-" * 70)

def interpret_ari(a):
    if a > 0.80: return "Strong agreement"
    if a > 0.60: return "Moderate agreement"
    if a > 0.40: return "Weak agreement"
    return              "Little to no agreement"

report_lines = [
    "=" * 70,
    "CROSS-ALGORITHM CONSISTENCY — REPORT",
    "=" * 70,
    f"Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "CONFIGURATION",
    "─" * 70,
    f"  Algorithms compared : {', '.join(algo_names)}",
    "  DBSCAN noise points (label = -1) treated as a separate group.",
    "",
    "ARI MATRIX  (Adjusted Rand Index)",
    "─" * 70,
    "  1 = identical assignments | 0 = random | negative = worse than random",
    "",
    f"  {'':16}" + "".join(f"  {n:>14}" for n in algo_names),
    "  " + "-" * (18 + 16 * n_algos),
]

for i, name_i in enumerate(algo_names):
    row = f"  {name_i:<16}"
    for j in range(n_algos):
        row += f"  {ari_matrix[i, j]:>14.4f}"
    report_lines.append(row)

report_lines += [
    "",
    "NMI MATRIX  (Normalized Mutual Information)",
    "─" * 70,
    "  1 = identical | 0 = no shared information",
    "",
    f"  {'':16}" + "".join(f"  {n:>14}" for n in algo_names),
    "  " + "-" * (18 + 16 * n_algos),
]

for i, name_i in enumerate(algo_names):
    row = f"  {name_i:<16}"
    for j in range(n_algos):
        row += f"  {nmi_matrix[i, j]:>14.4f}"
    report_lines.append(row)

report_lines += ["", "PAIRWISE INTERPRETATION", "─" * 70]

for i in range(n_algos):
    for j in range(i + 1, n_algos):
        ari_val = ari_matrix[i, j]
        nmi_val = nmi_matrix[i, j]
        report_lines.append(
            f"  {algo_names[i]} vs {algo_names[j]:<16}:"
            f"  ARI = {ari_val:.4f}  |  NMI = {nmi_val:.4f}"
            f"  →  {interpret_ari(ari_val)}"
        )

report_lines += ["", "=" * 70]

report_text = "\n".join(report_lines)
(RESULTS_DIR / "eval_consistency_report.txt").write_text(report_text)
print(report_text)
print(f"\n✓ Saved: results/eval_consistency_report.txt")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✓ CONSISTENCY EVALUATION COMPLETE")
print("=" * 70)
print(f"""
FILES SAVED (results/):
───────────────────────
✓ eval_consistency_matrix.png
✓ eval_consistency_report.txt

STATUS: ✓ READY
""")
print("=" * 70 + "\n")