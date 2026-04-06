"""
HIERARCHICAL CLUSTERING (Ward Linkage)
=======================================
Agglomerative Hierarchical Clustering on normalized stock price segments.
Uses Ward linkage + Euclidean distance.

k=4 is fixed to match K-Means for direct comparison.

Because Ward linkage has O(n²) memory complexity, a pre-built shared
sample of 50,000 segments is loaded from sample_50k.h5.

Input:
  data/processed/sample_50k.h5
  data/processed/sample_50k_metadata.csv
Output:
  - hierarchical_labels.npy   (50k sample labels, cut at k=4)
  - hierarchical_centers.npy  (k=4 cluster centers)
  - hierarchical_results.csv  (metadata + cluster column, 50k rows)
  - hierarchical_report.txt   (cluster distribution)
  - hierarchical_dendrogram.png
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / "results"

print("\n" + "=" * 70)
print("HIERARCHICAL CLUSTERING  (Ward Linkage + Euclidean Distance)")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_H5   = ROOT / "data/processed/sample_50k.h5"
SAMPLE_META = ROOT / "data/processed/sample_50k_metadata.csv"
N_CLUSTERS  = 4     # fixed — matches K-Means for fair comparison


# ============================================================
# STEP 1: LOAD SHARED SAMPLE
# ============================================================

print("\n[1/4] Loading data...")
print("-" * 70)

with h5py.File(SAMPLE_H5, "r") as f:
    X_sample   = f["segments"][:]
    sample_idx = f["indices"][:]

metadata_sample = pd.read_csv(SAMPLE_META)
SAMPLE_SIZE = len(X_sample)

print(f"✓ Loaded sample   : {X_sample.shape}")
print(f"✓ Loaded metadata : {metadata_sample.shape}")


# ============================================================
# STEP 2: WARD LINKAGE
# ============================================================

print(f"\n[2/4] Computing Ward linkage (n={SAMPLE_SIZE:,})...")
print("-" * 70)
print("  Method  : ward")
print("  Metric  : euclidean\n")

t0 = time.time()
Z = linkage(X_sample, method="ward", metric="euclidean")
elapsed = time.time() - t0

print(f"✓ Linkage complete : {elapsed:.1f} s")
print(f"  Matrix Z shape   : {Z.shape}   (n-1 merges × 4 columns)")


# ============================================================
# STEP 3: CUT DENDROGRAM AT k=4 → LABELS
# ============================================================

print(f"\n[3/4] Cutting dendrogram at k={N_CLUSTERS}...")
print("-" * 70)

sample_labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust") - 1   # 0-indexed

centers = np.array([
    X_sample[sample_labels == cid].mean(axis=0)
    for cid in range(N_CLUSTERS)
])

unique, counts = np.unique(sample_labels, return_counts=True)
print(f"✓ Cluster distribution (k={N_CLUSTERS}, n={SAMPLE_SIZE:,}):")
for cid, cnt in zip(unique, counts):
    print(f"    Cluster {cid}: {cnt:7,}  ({cnt/SAMPLE_SIZE*100:.1f}%)")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

labels_file  = RESULTS_DIR / "hierarchical_labels.npy"
centers_file = RESULTS_DIR / "hierarchical_centers.npy"
results_csv  = RESULTS_DIR / "hierarchical_results.csv"

np.save(labels_file,  sample_labels)
np.save(centers_file, centers)
print(f"\n✓ Saved: {labels_file}")
print(f"✓ Saved: {centers_file}")

results_df            = metadata_sample.copy()
results_df["cluster"] = sample_labels
results_df.to_csv(results_csv, index=False)
print(f"✓ Saved: {results_csv}  ({len(results_df):,} rows)")


# ============================================================
# STEP 4: DENDROGRAM + REPORT
# ============================================================

print(f"\n[4/4] Plotting dendrogram and saving report...")
print("-" * 70)

cut_height = Z[-(N_CLUSTERS - 1), 2]

fig, ax = plt.subplots(figsize=(14, 6))

dendrogram(
    Z,
    ax=ax,
    truncate_mode="lastp",
    p=50,
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True,
    color_threshold=cut_height,
)

ax.axhline(
    y=cut_height,
    color="red", linestyle="--", linewidth=2,
    label=f"Cut → k={N_CLUSTERS}  (height = {cut_height:.2f})"
)

ax.set_xlabel("Segment index (merged nodes)", fontsize=12, fontweight="bold")
ax.set_ylabel("Ward Distance (merge cost)",   fontsize=12, fontweight="bold")
ax.set_title(
    f"Dendrogram — Ward Linkage  |  sample n={SAMPLE_SIZE:,}  |  cut at k={N_CLUSTERS}",
    fontsize=13, fontweight="bold"
)
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plot_file = RESULTS_DIR / "hierarchical_dendrogram.png"
fig.savefig(plot_file, dpi=300, bbox_inches="tight")
print(f"✓ Saved: {plot_file}")
plt.show()

# ── Report ────────────────────────────────────────────────
report_lines = [
    "=" * 70,
    "HIERARCHICAL CLUSTERING — RESULTS REPORT",
    "=" * 70,
    f"Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "CONFIGURATION",
    "─" * 70,
    f"  Linkage method   : Ward",
    f"  Distance metric  : Euclidean",
    f"  Sample size      : {SAMPLE_SIZE:,}",
    f"  k (fixed)        : {N_CLUSTERS}  (matches K-Means for comparison)",
    f"  Linkage time     : {elapsed:.1f} s",
    "",
    "CLUSTER DISTRIBUTION (k=4)",
    "─" * 70,
    f"  {'Cluster':<10} {'Segments':>10} {'Percentage':>12}",
    "  " + "-" * 34,
]

for cid, cnt in zip(unique, counts):
    report_lines.append(f"  {cid:<10} {cnt:>10,} {cnt/SAMPLE_SIZE*100:>11.1f}%")

report_lines += ["", "=" * 70]

report_text = "\n".join(report_lines)
report_file = RESULTS_DIR / "hierarchical_report.txt"
report_file.write_text(report_text)
print(report_text)
print(f"✓ Saved: {report_file}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✓ HIERARCHICAL CLUSTERING COMPLETE")
print("=" * 70)
print(f"""
SUMMARY:
────────
Linkage          : Ward (Euclidean)
Sample size      : {SAMPLE_SIZE:,}
Dendrogram cut   : k = {N_CLUSTERS}
Linkage time     : {elapsed:.1f} s

CLUSTER DISTRIBUTION (k={N_CLUSTERS}, n={SAMPLE_SIZE:,} segments):
────────────────────────────────────────────""")

for cid, cnt in zip(unique, counts):
    print(f"  Cluster {cid}: {cnt:7,}  ({cnt/SAMPLE_SIZE*100:.1f}%)")

print(f"""
FILES SAVED (results/):
───────────────────────
✓ hierarchical_labels.npy       ({SAMPLE_SIZE:,} labels, k={N_CLUSTERS})
✓ hierarchical_centers.npy      (shape {centers.shape})
✓ hierarchical_results.csv      ({len(results_df):,} rows)
✓ hierarchical_report.txt
✓ hierarchical_dendrogram.png
""")
print("=" * 70 + "\n")