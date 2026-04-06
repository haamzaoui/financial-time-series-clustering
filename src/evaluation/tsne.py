"""
t-SNE VISUALISATION — ALL ALGORITHMS
======================================
Projects the 50k sample into 2D using t-SNE, then colours
the same 2D embedding with each algorithm's cluster labels.

Input:
  data/processed/sample_50k.h5
  data/processed/sample_50k_metadata.csv
  results/kmeans_labels.npy
  results/hierarchical_labels.npy
  results/dbscan_labels.npy
Output:
  results/tsne_coordinates.npy       (2D coords — reusable)
  results/tsne_kmeans.png
  results/tsne_hierarchical.png
  results/tsne_dbscan.png
  results/tsne_comparison.png        (all three side by side)
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
from pathlib import Path
from sklearn.manifold import TSNE

ROOT        = Path(__file__).resolve().parents[2]
SRC         = ROOT / "src"
RESULTS_DIR = Path(__file__).parent / "results"

print("\n" + "=" * 70)
print("t-SNE VISUALISATION  —  ALL ALGORITHMS")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_H5    = ROOT / "data/processed/sample_50k.h5"
SAMPLE_META  = ROOT / "data/processed/sample_50k_metadata.csv"
COORDS_FILE  = RESULTS_DIR / "tsne_coordinates.npy"

LABEL_FILES = {
    "K-Means"      : SRC / "clustering/kmeans/results/kmeans_labels.npy",
    "Hierarchical" : SRC / "clustering/hierarchical/results/hierarchical_labels.npy",
    "DBSCAN"       : SRC / "clustering/dbscan/results/dbscan_labels.npy",
}
# t-SNE parameters
PERPLEXITY   = 30
MAX_ITER     = 1000
RANDOM_STATE = 42

# Visual
POINT_SIZE = 8
ALPHA      = 0.5

# High-contrast color palette — up to 10 clusters
CLUSTER_COLORS = [
    "#E63946",   # vivid red
    "#457B9D",   # steel blue
    "#2A9D8F",   # teal
    "#E9A820",   # amber
    "#6A0572",   # purple
    "#F4A261",   # orange
    "#264653",   # dark teal
    "#A8DADC",   # light blue
    "#81B29A",   # sage green
    "#F2CC8F",   # sand
]


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
print(f"✓ Loaded metadata : {metadata.shape}")

# Load all available label files
all_labels = {}
for name, path in LABEL_FILES.items():
    if path.exists():
        lbl        = np.load(path)
        n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_noise    = int((lbl == -1).sum())
        all_labels[name] = lbl
        print(f"✓ Loaded {name:<15}: {n_clusters} clusters, "
              f"{n_noise:,} noise ({n_noise/N*100:.1f}%)")
    else:
        print(f"✗ {name:<15}: not found — {path}  (skipping)")

if len(all_labels) == 0:
    raise RuntimeError("No label files found. Run clustering scripts first.")


# ============================================================
# STEP 2: t-SNE  (compute once or load from cache)
# ============================================================

print(f"\n[2/4] t-SNE embedding...")
print("-" * 70)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if COORDS_FILE.exists():
    # Load cached coordinates — skip recomputation
    X_tsne = np.load(COORDS_FILE)
    print(f"✓ Loaded cached coordinates : {COORDS_FILE}")
    print(f"  Shape                     : {X_tsne.shape}")
    print(f"  (Delete {COORDS_FILE} to force recomputation)")
else:
    print(f"  Computing t-SNE on {N:,} × {D}D segments...")
    print(f"  perplexity={PERPLEXITY}, max_iter={MAX_ITER}, seed={RANDOM_STATE}")
    print(f"  This typically takes 5–15 minutes — please wait...\n")

    t0    = time.time()
    tsne  = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        verbose=1
    )
    X_tsne = tsne.fit_transform(X)
    elapsed = time.time() - t0

    np.save(COORDS_FILE, X_tsne)
    print(f"\n✓ t-SNE complete  : {elapsed/60:.1f} min")
    print(f"✓ Saved coords    : {COORDS_FILE}")

print(f"  Coordinate range  X: [{X_tsne[:,0].min():.1f}, {X_tsne[:,0].max():.1f}]")
print(f"  Coordinate range  Y: [{X_tsne[:,1].min():.1f}, {X_tsne[:,1].max():.1f}]")


# ============================================================
# STEP 3: PER-ALGORITHM PLOTS
# ============================================================

print(f"\n[3/4] Creating per-algorithm plots...")
print("-" * 70)

def plot_tsne(ax, X_tsne, labels, algo_name, show_noise=True):
    """
    Plot t-SNE scatter coloured by cluster labels.
    Noise points (label = -1) are plotted in light gray underneath.
    Each cluster gets a distinct colour from ALGO_COLORS.
    Cluster centroids in t-SNE space are marked with a black cross.
    """
    unique_labels = sorted(set(labels))
    cluster_ids   = [l for l in unique_labels if l >= 0]

    # Noise points first (rendered below clusters)
    if show_noise and -1 in unique_labels:
        mask = labels == -1
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c="lightgray", s=POINT_SIZE, alpha=0.3,
                   linewidths=0, label=f"Noise  (n={mask.sum():,})")

    # Cluster points — one distinct color per cluster with black edge
    for i, cid in enumerate(cluster_ids):
        mask  = labels == cid
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=color, s=POINT_SIZE, alpha=ALPHA,
                   edgecolors="black", linewidths=0.2,
                   label=f"Cluster {cid}  (n={mask.sum():,})")

        # Mark centroid in t-SNE space
        cx, cy = X_tsne[mask, 0].mean(), X_tsne[mask, 1].mean()
        ax.plot(cx, cy, 'k+', markersize=14, markeredgewidth=2, zorder=5)

    n_clusters = len(cluster_ids)
    ax.set_title(algo_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE component 1", fontsize=10)
    ax.set_ylabel("t-SNE component 2", fontsize=10)
    ax.legend(fontsize=8, loc="best", framealpha=0.85,
              markerscale=2, handlelength=1)
    ax.grid(True, alpha=0.2, linestyle="--")
    return n_clusters


# Individual full-size plot per algorithm
for algo_name, labels in all_labels.items():
    fig, ax = plt.subplots(figsize=(10, 8))
    n_cl = plot_tsne(ax, X_tsne, labels, algo_name)
    fig.suptitle(
        f"t-SNE embedding — {algo_name}  "
        f"({n_cl} cluster{'s' if n_cl != 1 else ''}, n={N:,})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fname = f"tsne_{algo_name.lower().replace(' ', '_').replace('-', '')}.png"
    fig.savefig(RESULTS_DIR / fname, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: results/{fname}")
    plt.show()


# ============================================================
# STEP 4: COMPARISON PLOT  (all algorithms side by side)
# ============================================================

print(f"\n[4/4] Creating comparison plot...")
print("-" * 70)

n_algos = len(all_labels)
fig, axes = plt.subplots(1, n_algos,
                          figsize=(9 * n_algos, 7),
                          sharex=True, sharey=True)

if n_algos == 1:
    axes = [axes]

fig.suptitle(
    f"t-SNE Embedding — Algorithm Comparison  (n={N:,} segments, {D}D → 2D)",
    fontsize=14, fontweight="bold"
)

for ax, (algo_name, labels) in zip(axes, all_labels.items()):
    plot_tsne(ax, X_tsne, labels, algo_name)

# Shared axis label
for ax in axes[1:]:
    ax.set_ylabel("")   # remove repeated y labels

plt.tight_layout()
fig.savefig(RESULTS_DIR / "tsne_comparison.png",
            dpi=300, bbox_inches="tight")
print(f"✓ Saved: results/tsne_comparison.png")
plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✓ t-SNE VISUALISATION COMPLETE")
print("=" * 70)
print(f"""
SUMMARY:
────────
Segments      : {N:,}
Dimensions    : {D}D → 2D
t-SNE coords  : {'loaded from cache' if COORDS_FILE.exists() else 'computed fresh'}
Algorithms    : {', '.join(all_labels.keys())}

FILES SAVED (results/):
───────────────────────
✓ tsne_coordinates.npy   (reusable — delete to recompute)""")

for algo_name in all_labels:
    fname = f"tsne_{algo_name.lower().replace(' ', '_').replace('-', '')}.png"
    print(f"✓ {fname}")

print(f"""✓ tsne_comparison.png

NOTE:
─────
All plots use the same t-SNE coordinates. Visual differences between
algorithm plots reflect only label differences, not coordinate changes.
The black + marker on each plot shows the cluster centroid in 2D t-SNE space.
""")
print("=" * 70 + "\n")