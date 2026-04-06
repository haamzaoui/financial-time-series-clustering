"""
INTERNAL VALIDATION METRICS
=============================
Evaluates the quality of each clustering algorithm independently
using three complementary internal metrics — no ground truth needed.

METRICS:
  Silhouette Score       : how well each point fits its own cluster
                           vs the nearest neighbouring cluster
                           → higher is better  (range: -1 to 1)

  Calinski-Harabasz (CH) : ratio of between-cluster to within-cluster
                           variance, weighted by cluster sizes
                           → higher is better  (no fixed range)

  Davies-Bouldin (DB)    : average similarity between each cluster
                           and its most similar neighbour
                           → lower is better   (range: 0 to inf)

NOTE ON DBSCAN:
  DBSCAN produced 1 cluster + noise. All three metrics require >= 2
  clusters, so DBSCAN is skipped with an explanatory message.
  Noise points (label = -1) are excluded before scoring any algorithm.

Input:
  data/processed/sample_50k.h5
  results/kmeans_labels.npy
  results/hierarchical_labels.npy
  results/dbscan_labels.npy
  results/kmeans_centers.npy        (optional — for center comparison plot)
  results/hierarchical_centers.npy  (optional — for center comparison plot)
Output:
  results/eval_internal_metrics.png
  results/eval_cluster_centers_comparison.png
  results/eval_centers_kmeans.png
  results/eval_centers_hierarchical.png
  results/eval_internal_report.txt
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

ROOT        = Path(__file__).resolve().parents[2]
SRC         = ROOT / "src"
RESULTS_DIR = Path(__file__).parent / "results"

print("\n" + "=" * 70)
print("INTERNAL VALIDATION METRICS")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_H5   = ROOT / "data/processed/sample_50k.h5"
SAMPLE_META = ROOT / "data/processed/sample_50k_metadata.csv"

LABEL_FILES = {
    "K-Means"      : SRC / "clustering/kmeans/results/kmeans_labels.npy",
    "Hierarchical" : SRC / "clustering/hierarchical/results/hierarchical_labels.npy",
    "DBSCAN"       : SRC / "clustering/dbscan/results/dbscan_labels.npy",
}

CENTER_FILES = {
    "K-Means"      : SRC / "clustering/kmeans/results/kmeans_centers.npy",
    "Hierarchical" : SRC / "clustering/hierarchical/results/hierarchical_centers.npy",
}

RANDOM_STATE = 42
ALGO_COLORS       = {
    "K-Means"      : "steelblue",
    "Hierarchical" : "darkorange",
    "DBSCAN"       : "seagreen",
}


# ============================================================
# STEP 1: LOAD DATA AND LABELS
# ============================================================

print("\n[1/4] Loading data and labels...")
print("-" * 70)

with h5py.File(SAMPLE_H5, "r") as f:
    X = f["segments"][:]

metadata = pd.read_csv(SAMPLE_META)
N, D     = X.shape

print(f"✓ Loaded segments : {X.shape}")

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

if len(all_labels) == 0:
    raise RuntimeError("No label files found. Run clustering scripts first.")


# ============================================================
# STEP 2: COMPUTE METRICS
# ============================================================

print(f"\n[2/4] Computing internal metrics...")
print("-" * 70)
print(f"  Silhouette computed on all {N:,} points\n")

print(f"  {'Algorithm':<16} {'Clusters':>9} {'Silhouette':>12} "
      f"{'Calinski-H':>13} {'Davies-B':>11}")
print("  " + "-" * 63)

records = []

for name, lbl in all_labels.items():
    # Exclude noise points (DBSCAN label = -1)
    core_mask  = lbl >= 0
    lbl_clean  = lbl[core_mask]
    X_clean    = X[core_mask]
    n_clusters = len(set(lbl_clean))

    if n_clusters < 2:
        print(f"  {name:<16} {n_clusters:>9}  "
              f"{'N/A — fewer than 2 clusters':>37}")
        records.append({
            "algorithm": name, "n_clusters": n_clusters,
            "silhouette": np.nan, "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan
        })
        continue

    sil = silhouette_score(X_clean, lbl_clean, random_state=RANDOM_STATE)
    ch  = calinski_harabasz_score(X_clean, lbl_clean)
    db  = davies_bouldin_score(X_clean, lbl_clean)

    print(f"  {name:<16} {n_clusters:>9} {sil:>12.4f} {ch:>13.1f} {db:>11.4f}")
    records.append({
        "algorithm": name, "n_clusters": n_clusters,
        "silhouette": sil, "calinski_harabasz": ch,
        "davies_bouldin": db
    })

results_df = pd.DataFrame(records)


# ============================================================
# STEP 3: VISUALISATIONS
# ============================================================

print(f"\n[3/4] Creating visualisations...")
print("-" * 70)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot 1: metric comparison bar charts ─────────────────
valid_df = results_df.dropna(subset=["silhouette"])

if len(valid_df) > 0:
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle("Internal Validation Metrics", fontsize=13, fontweight="bold")

    metric_cfg = [
        ("silhouette",        "Silhouette score",        "↑ higher is better", True),
        ("calinski_harabasz", "Calinski-Harabasz score", "↑ higher is better", True),
        ("davies_bouldin",    "Davies-Bouldin index",    "↓ lower is better",  False),
    ]

    for ax, (col, ylabel, note, higher_better) in zip(axes1, metric_cfg):
        algos  = valid_df["algorithm"].tolist()
        values = valid_df[col].tolist()
        colors = [ALGO_COLORS.get(a, "gray") for a in algos]

        bars = ax.bar(algos, values, color=colors,
                      edgecolor="black", linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(note, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        # Highlight the best bar with a red border
        best_idx = int(np.argmax(values) if higher_better else np.argmin(values))
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / "eval_internal_metrics.png",
                 dpi=300, bbox_inches="tight")
    print(f"✓ Saved: results/eval_internal_metrics.png")
    plt.show()

# ── Plot 2: cluster centers side by side ─────────────────
available_centers = {
    name: np.load(path)
    for name, path in CENTER_FILES.items()
    if Path(path).exists()
}

if len(available_centers) > 0:
    n_plots = len(available_centers)
    fig2, axes2 = plt.subplots(1, n_plots,
                                figsize=(7 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes2 = [axes2]

    fig2.suptitle("Cluster Centers — Typical Price Pattern per Cluster",
                  fontsize=13, fontweight="bold")

    for ax, (name, centers) in zip(axes2, available_centers.items()):
        lbl    = all_labels[name]
        colors = plt.cm.Set3(np.linspace(0, 1, len(centers)))
        for cid, center in enumerate(centers):
            cnt = int((lbl == cid).sum())
            ax.plot(center, linewidth=2, marker="o", markersize=3,
                    color=colors[cid], label=f"Cluster {cid}  (n={cnt:,})")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Day in segment (1–50)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalised price", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim([0, D - 1])

    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "eval_cluster_centers_comparison.png",
                 dpi=300, bbox_inches="tight")
    print(f"✓ Saved: results/eval_cluster_centers_comparison.png")
    plt.show()

# ── Plot 3: individual center plot per algorithm ──────────
CLUSTER_COLORS = [
    "#E63946",  # vivid red
    "#457B9D",  # steel blue
    "#2A9D8F",  # teal
    "#E9A820",  # amber
    "#6A0572",  # purple
    "#F4A261",  # orange
    "#264653",  # dark teal
    "#A8DADC",  # light blue
]

for name, centers in available_centers.items():
    lbl    = all_labels[name]
    n_cl   = len(centers)
    fig3, ax3 = plt.subplots(figsize=(12, 5))

    for cid, center in enumerate(centers):
        cnt   = int((lbl == cid).sum())
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax3.plot(center, linewidth=2.5, marker="o", markersize=4,
                 color=color, label=f"Cluster {cid}  (n={cnt:,})")

    ax3.set_xlabel("Day in segment (1–50)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Normalised price", fontsize=12, fontweight="bold")
    ax3.set_title(
        f"{name} — Cluster Centers  (k={n_cl})",
        fontsize=13, fontweight="bold"
    )
    ax3.legend(fontsize=10, loc="best")
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.set_xlim([0, D - 1])

    plt.tight_layout()
    fname = f"eval_centers_{name.lower().replace('-', '').replace(' ', '_')}.png"
    fig3.savefig(RESULTS_DIR / fname, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: results/{fname}")
    plt.show()


# ============================================================
# STEP 4: REPORT
# ============================================================

print(f"\n[4/4] Saving report...")
print("-" * 70)

def interpret_silhouette(s):
    if np.isnan(s): return "N/A"
    if s > 0.75:    return "Excellent (> 0.75)"
    if s > 0.50:    return "Good (0.50–0.75)"
    if s > 0.25:    return "Fair (0.25–0.50)"
    return              "Poor (< 0.25)"

report_lines = [
    "=" * 70,
    "INTERNAL VALIDATION METRICS — REPORT",
    "=" * 70,
    f"Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "CONFIGURATION",
    "─" * 70,
    f"  Sample size          : {N:,}",
    f"  Dimensions           : {D}",
    f"  Silhouette subsample : all {N:,} points",
    f"  Algorithms           : {', '.join(all_labels.keys())}",
    "",
    "RESULTS",
    "─" * 70,
    "  Noise points excluded before scoring (DBSCAN label = -1).",
    "  N/A = fewer than 2 clusters — metric cannot be computed.",
    "",
    f"  {'Algorithm':<16} {'Clusters':>9} {'Silhouette':>12} "
    f"{'Calinski-H':>13} {'Davies-B':>11}  Quality",
    "  " + "-" * 78,
]

for _, row in results_df.iterrows():
    sil_str = f"{row['silhouette']:.4f}"          if not np.isnan(row['silhouette'])          else "   N/A"
    ch_str  = f"{row['calinski_harabasz']:.1f}"   if not np.isnan(row['calinski_harabasz'])   else "      N/A"
    db_str  = f"{row['davies_bouldin']:.4f}"      if not np.isnan(row['davies_bouldin'])      else "   N/A"
    quality = interpret_silhouette(row['silhouette'])
    report_lines.append(
        f"  {row['algorithm']:<16} {int(row['n_clusters']):>9} "
        f"{sil_str:>12} {ch_str:>13} {db_str:>11}  {quality}"
    )

# Best per metric
valid = results_df.dropna(subset=["silhouette"])
if len(valid) > 0:
    best_sil  = valid.loc[valid["silhouette"].idxmax()]
    best_ch   = valid.loc[valid["calinski_harabasz"].idxmax()]
    best_db   = valid.loc[valid["davies_bouldin"].idxmin()]
    report_lines += [
        "",
        "BEST PER METRIC",
        "─" * 70,
        f"  Silhouette (↑)       : {best_sil['algorithm']}  ({best_sil['silhouette']:.4f})",
        f"  Calinski-Harabasz (↑): {best_ch['algorithm']}  ({best_ch['calinski_harabasz']:.1f})",
        f"  Davies-Bouldin (↓)   : {best_db['algorithm']}  ({best_db['davies_bouldin']:.4f})",
    ]

report_lines += ["", "=" * 70]

report_text = "\n".join(report_lines)
(RESULTS_DIR / "eval_internal_report.txt").write_text(report_text)
print(report_text)
print(f"\n✓ Saved: results/eval_internal_report.txt")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✓ INTERNAL VALIDATION COMPLETE")
print("=" * 70)
print(f"""
FILES SAVED (results/):
───────────────────────
✓ eval_internal_metrics.png
✓ eval_cluster_centers_comparison.png
✓ eval_centers_kmeans.png
✓ eval_centers_hierarchical.png
✓ eval_internal_report.txt

STATUS: ✓ READY
""")
print("=" * 70 + "\n")