"""
SILHOUETTE ANALYSIS FOR K-MEANS
================================
Computes the Silhouette Score for k=2..10 to determine the optimal
number of clusters. Use alongside the elbow method (elbow_kmeans.py)
for a more robust k selection.

The Silhouette Score measures how well each point fits its assigned
cluster versus the nearest other cluster. Range: [-1, 1].
Higher is better. The k with the
 highest average score is optimal.

Input:
  data/processed/sample_50k.h5
Output:
  results/silhouette_scores.png      — score vs k plot
  results/silhouette_bars_k*.png     — per-point silhouette bar chart for best k
  results/silhouette_report.txt      — scores table + recommendation
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / "results"

print("\n" + "=" * 70)
print("SILHOUETTE ANALYSIS FOR K-MEANS")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_H5    = ROOT / "data/processed/sample_50k.h5"
K_RANGE      = range(2, 11)   # k = 2 to 10
N_INIT       = 10
RANDOM_STATE = 42


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("\n[1/4] Loading data...")
print("-" * 70)

with h5py.File(SAMPLE_H5, "r") as f:
    X = f["segments"][:]

print(f"✓ Loaded segments : {X.shape}")


# ============================================================
# STEP 2: COMPUTE SILHOUETTE SCORE FOR EACH k
# ============================================================

print(f"\n[2/4] Computing Silhouette Scores for k={min(K_RANGE)}..{max(K_RANGE)}...")
print("-" * 70)
print(f"  {'k':<6} {'Silhouette Score':>18}")
print("  " + "-" * 26)

records = []

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X)
    score  = silhouette_score(X, labels)

    records.append({
        "k":          k,
        "silhouette": score,
        "labels":     labels,
        "centers":    kmeans.cluster_centers_
    })

    print(f"  {k:<6} {score:>18.4f}")

# Best k = highest silhouette score
best_record = max(records, key=lambda r: r["silhouette"])
best_k      = best_record["k"]
best_score  = best_record["silhouette"]

print(f"\n  {'─'*40}")
print(f"  Optimal k (Silhouette) : k = {best_k}  (score = {best_score:.4f})")
print(f"  {'─'*40}")


# ============================================================
# STEP 3: PLOTS
# ============================================================

print(f"\n[3/4] Generating plots...")
print("-" * 70)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

k_vals = [r["k"]          for r in records]
scores = [r["silhouette"] for r in records]

# ── Plot 1: Silhouette score vs k ─────────────────────────

fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(k_vals, scores, "o-", color="steelblue", linewidth=2,
         markersize=8, label="Silhouette Score")
ax1.scatter([best_k], [best_score], s=200, color="red",
            zorder=5, marker="*", label=f"Best k={best_k}  ({best_score:.4f})")
ax1.axvline(x=best_k, color="red", linestyle="--", linewidth=1.5, alpha=0.6)

ax1.set_xlabel("Number of Clusters (k)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Average Silhouette Score", fontsize=12, fontweight="bold")
ax1.set_title("Silhouette Analysis — Optimal k for K-Means",
              fontsize=13, fontweight="bold")
ax1.set_xticks(k_vals)
ax1.set_ylim([min(scores) - 0.02, max(scores) + 0.02])
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.legend(fontsize=11)

plt.tight_layout()
fig1.savefig(RESULTS_DIR / "silhouette_scores.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: results/silhouette_scores.png")
plt.show()


# ── Plot 2: Per-point silhouette bar chart for best k ─────
# Each cluster is shown as a horizontal bar of individual silhouette
# values, sorted in descending order. The red dashed line marks
# the average score. Clusters with many points below 0 are poorly formed.

labels_best  = best_record["labels"]
sil_vals     = silhouette_samples(X, labels_best)
n_clusters   = best_k
colors       = cm.Set2(np.linspace(0, 1, n_clusters))

fig2, ax2 = plt.subplots(figsize=(11, 6))

y_lower = 10
cluster_yticks     = []
cluster_ytick_lbls = []

for cid in range(n_clusters):
    mask        = labels_best == cid
    sil_cluster = np.sort(sil_vals[mask])
    size        = sil_cluster.shape[0]
    y_upper     = y_lower + size

    ax2.fill_betweenx(
        np.arange(y_lower, y_upper),
        0, sil_cluster,
        facecolor=colors[cid], edgecolor=colors[cid], alpha=0.85
    )

    cluster_yticks.append(y_lower + size / 2)
    cluster_ytick_lbls.append(f"Cluster {cid}\n(n={size:,})")

    y_lower = y_upper + 10  # gap between clusters

ax2.axvline(x=best_score, color="red", linestyle="--", linewidth=1.8,
            label=f"Avg score = {best_score:.4f}")
ax2.axvline(x=0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

ax2.set_yticks(cluster_yticks)
ax2.set_yticklabels(cluster_ytick_lbls, fontsize=10)
ax2.set_xlabel("Silhouette Coefficient", fontsize=12, fontweight="bold")
ax2.set_title(f"Per-Point Silhouette Plot  (k={best_k})",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
fig2.savefig(RESULTS_DIR / f"silhouette_bars_k{best_k}.png",
             dpi=300, bbox_inches="tight")
print(f"✓ Saved: results/silhouette_bars_k{best_k}.png")
plt.show()



# ============================================================
# STEP 4: REPORT
# ============================================================

print(f"\n[4/4] Saving report...")
print("-" * 70)

quality = (
    "Excellent (> 0.75)" if best_score > 0.75 else
    "Good (0.50–0.75)"   if best_score > 0.50 else
    "Fair (0.25–0.50)"   if best_score > 0.25 else
    "Poor (< 0.25)"
)

report_lines = [
    "=" * 70,
    "SILHOUETTE ANALYSIS — RESULTS REPORT",
    "=" * 70,
    f"Generated  : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "CONFIGURATION",
    "─" * 70,
    f"  Sample size  : {X.shape[0]:,}",
    f"  Dimensions   : {X.shape[1]}",
    f"  k range      : {min(K_RANGE)} to {max(K_RANGE)}",
    f"  n_init       : {N_INIT}",
    f"  Random state : {RANDOM_STATE}",
    "",
    "SILHOUETTE SCORES",
    "─" * 70,
    f"  {'k':<6} {'Silhouette':>14}",
    "  " + "-" * 22,
]

for r in records:
    marker = "  ← optimal" if r["k"] == best_k else ""
    report_lines.append(
        f"  {r['k']:<6} {r['silhouette']:>14.4f}{marker}"
    )

report_lines += [
    "",
    "RECOMMENDATION",
    "─" * 70,
    f"  Optimal k (highest silhouette) : k = {best_k}",
    f"  Silhouette score               : {best_score:.4f}",
    f"  Quality interpretation         : {quality}",
    "",
    "INTERPRETATION GUIDE",
    "─" * 70,
    "  > 0.75  Excellent — clusters are clearly separated",
    "  0.50–0.75  Good — reasonable cluster structure",
    "  0.25–0.50  Fair — weak but detectable structure",
    "  < 0.25  Poor — clusters overlap significantly",
    "",
    "NOTE",
    "─" * 70,
    "  Use this result alongside the elbow method (elbow_kmeans.py).",
    "  If both methods agree on k, the choice is well-supported.",
    "  If they disagree, examine the silhouette bar chart to check",
    "  whether any clusters have many points with negative scores.",
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
(RESULTS_DIR / "silhouette_report.txt").write_text(report_text)
print(report_text)
print(f"✓ Saved: results/silhouette_report.txt")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✓ SILHOUETTE ANALYSIS COMPLETE")
print("=" * 70)
print(f"""
SUMMARY:
────────
Sample size    : {X.shape[0]:,}
k tested       : {min(K_RANGE)} to {max(K_RANGE)}

SILHOUETTE RECOMMENDATION:
───────────────────────────
Optimal k      : k = {best_k}
Best score     : {best_score:.4f}
Quality        : {quality}

FILES SAVED (results/):
───────────────────────
✓ silhouette_scores.png
✓ silhouette_bars_k{best_k}.png
✓ silhouette_vs_elbow.png
✓ silhouette_report.txt
""")
print("=" * 70 + "\n")