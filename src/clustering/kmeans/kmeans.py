"""
K-MEANS CLUSTERING
==================
Runs K-Means on the 50k sample using the optimal k
determined by the elbow method.

Input:
  data/processed/sample_50k.h5
  data/processed/sample_50k_metadata.csv
Output:
  results/kmeans_labels.npy
  results/kmeans_centers.npy
  results/kmeans_results.csv
  results/kmeans_cluster_centers.png
  results/kmeans_distribution.png
  results/kmeans_report.txt
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
from pathlib import Path
from sklearn.cluster import KMeans

ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / "results"

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_H5   = ROOT / "data/processed/sample_50k.h5"
SAMPLE_META = ROOT / "data/processed/sample_50k_metadata.csv"
N_CLUSTERS  = 4     # determined by elbow method 
N_INIT      = 10    # number of random initialisations
RANDOM_STATE = 42


# ============================================================
# STEP 1: LOAD SHARED SAMPLE
# ============================================================

print("\n[1/4] Loading data...")
print("-" * 70)

with h5py.File(SAMPLE_H5, "r") as f:
    X          = f["segments"][:]
    sample_idx = f["indices"][:]

metadata = pd.read_csv(SAMPLE_META)

print(f"✓ Loaded segments : {X.shape}")
print(f"✓ Loaded metadata : {metadata.shape}")


# ============================================================
# STEP 2: RUN K-MEANS
# ============================================================

print(f"\n[2/4] Running K-Means (k={N_CLUSTERS}, n_init={N_INIT})...")
print("-" * 70)

t0     = time.time()
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=RANDOM_STATE)

labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

elapsed = time.time() - t0

unique, counts = np.unique(labels, return_counts=True)

print(f"✓ K-Means complete")
print(f"  Time       : {elapsed:.1f} s")
print(f"  Inertia    : {kmeans.inertia_:.2f}")
print(f"  Iterations : {kmeans.n_iter_}")


# ============================================================
# STEP 3: SAVE RESULTS
# ============================================================

print(f"\n[3/4] Saving results...")
print("-" * 70)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Labels
labels_file = RESULTS_DIR / "kmeans_labels.npy"
np.save(labels_file, labels)
print(f"✓ Saved: {labels_file}")

# Centers
centers_file = RESULTS_DIR / "kmeans_centers.npy"
np.save(centers_file, centers)
print(f"✓ Saved: {centers_file}")

# Results CSV — metadata + cluster column
results_df            = metadata.copy()
results_df["cluster"] = labels
results_csv           = RESULTS_DIR / "kmeans_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"✓ Saved: {results_csv}  ({len(results_df):,} rows)")


# ============================================================
# STEP 4: VISUALISATIONS + REPORT
# ============================================================

print(f"\n[4/4] Creating visualisations and report...")
print("-" * 70)

COLORS = plt.cm.Set3(np.linspace(0, 1, N_CLUSTERS))

# ── Plot 1: cluster centers ───────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 5))

for cid in range(N_CLUSTERS):
    ax1.plot(centers[cid], marker="o", linewidth=2, markersize=4,
             color=COLORS[cid], label=f"Cluster {cid}  (n={counts[cid]:,})")

ax1.set_xlabel("Day in segment (1–50)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Normalised price", fontsize=12, fontweight="bold")
ax1.set_title(
    f"K-Means cluster centers  (k={N_CLUSTERS})  —  typical price pattern per cluster",
    fontsize=13, fontweight="bold"
)
ax1.legend(fontsize=10, loc="best")
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.set_xlim([0, X.shape[1] - 1])
plt.tight_layout()
fig1.savefig(RESULTS_DIR / "kmeans_cluster_centers.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: results/kmeans_cluster_centers.png")
plt.show()

# ── Plot 2: distribution bar + pie ───────────────────────
fig2, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(f"K-Means cluster distribution  (k={N_CLUSTERS})",
              fontsize=13, fontweight="bold")

bars = ax_bar.bar(unique, counts, color=COLORS, edgecolor="black", linewidth=1.2)
ax_bar.set_xlabel("Cluster ID", fontsize=11, fontweight="bold")
ax_bar.set_ylabel("Number of segments", fontsize=11, fontweight="bold")
ax_bar.set_title("Segment count per cluster", fontsize=12, fontweight="bold")
ax_bar.set_xticks(unique)
ax_bar.grid(True, axis="y", alpha=0.3)
for bar in bars:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h,
                f"{int(h):,}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

pie_labels = [f"Cluster {c}\n({cnt:,})" for c, cnt in zip(unique, counts)]
ax_pie.pie(counts, labels=pie_labels, colors=COLORS,
           autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 9})
ax_pie.set_title("Percentage distribution", fontsize=12, fontweight="bold")

plt.tight_layout()
fig2.savefig(RESULTS_DIR / "kmeans_distribution.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: results/kmeans_distribution.png")
plt.show()

# ── Text report ───────────────────────────────────────────
report_lines = [
    "=" * 70,
    "K-MEANS CLUSTERING — RESULTS REPORT",
    "=" * 70,
    f"Generated  : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "CONFIGURATION",
    "─" * 70,
    f"  Sample size  : {len(X):,}",
    f"  Dimensions   : {X.shape[1]}",
    f"  k            : {N_CLUSTERS}  (elbow method)",
    f"  n_init       : {N_INIT}",
    f"  Random state : {RANDOM_STATE}",
    "",
    "RESULTS",
    "─" * 70,
    f"  Inertia    : {kmeans.inertia_:.2f}",
    f"  Iterations : {kmeans.n_iter_}",
    f"  Runtime    : {elapsed:.1f} s",
    "",
    "CLUSTER DISTRIBUTION",
    "─" * 70,
    f"  {'Cluster':<10} {'Segments':>10} {'Percentage':>12}",
    "  " + "-" * 34,
]

for cid, cnt in zip(unique, counts):
    report_lines.append(f"  {cid:<10} {cnt:>10,} {cnt/len(X)*100:>11.1f}%")

report_lines += ["", "=" * 70]

report_text = "\n".join(report_lines)
(RESULTS_DIR / "kmeans_report.txt").write_text(report_text)
print(report_text)
print(f"✓ Saved: results/kmeans_report.txt")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✓ K-MEANS CLUSTERING COMPLETE")
print("=" * 70)
print(f"""
SUMMARY:
────────
k          : {N_CLUSTERS}
Inertia    : {kmeans.inertia_:.2f}
Iterations : {kmeans.n_iter_}
Runtime    : {elapsed:.1f} s

CLUSTER DISTRIBUTION:
─────────────────────""")

for cid, cnt in zip(unique, counts):
    print(f"  Cluster {cid}: {cnt:7,}  ({cnt/len(X)*100:.1f}%)")

print(f"""
FILES SAVED (results/):
───────────────────────
✓ kmeans_labels.npy
✓ kmeans_centers.npy
✓ kmeans_results.csv
✓ kmeans_cluster_centers.png
✓ kmeans_distribution.png
✓ kmeans_report.txt

""")
print("=" * 70 + "\n")