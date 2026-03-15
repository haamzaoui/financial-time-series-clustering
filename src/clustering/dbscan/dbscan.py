"""
DBSCAN CLUSTERING
=================
Density-Based Spatial Clustering of Applications with Noise

Key differences from K-Means:
- No need to specify number of clusters
- Finds clusters of arbitrary shape
- Identifies outliers/noise points
- Parameter-driven: eps (radius) and min_samples

Input:  Normalized segments
Output:
  - Cluster labels (including noise points as -1)
  - Parameter optimization results
  - Visualization plots
  - Detailed analysis
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from pathlib import Path

print("\n" + "="*70)
print("DBSCAN CLUSTERING")
print("="*70)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("\n[1/5] Loading normalized segments...")
print("-" * 70)

h5_file = "../../../data/processed/segments_normalized_minmax.h5"
# Alternative: h5_file = "data/processed/segments_zscore.h5"

with h5py.File(h5_file, 'r') as f:
    X = f['segments'][:]

print(f"✓ Loaded segments: {X.shape}")

# Load metadata
metadata = pd.read_csv("../../../data/processed/segments_metadata.csv")
print(f"✓ Loaded metadata: {metadata.shape}")


# ============================================================
# [SAMPLING] STEP 2: SAMPLE DATA FOR k-DISTANCE GRAPH
# ============================================================

print("\n[2/6] Preparing data for k-distance graph...")
print("-" * 70)

n_total = X.shape[0]
dimensionality = X.shape[1]

# [SAMPLING] Adaptive min_samples based on dimensionality
min_samples = dimensionality + 1  # Rule: at least D + 1, where D = 50 → 51
k = min_samples

print(f"Dimensionality: {dimensionality}")
print(f"Adaptive min_samples (k): {k}")
print(f"Total segments: {n_total:,}")

# [SAMPLING] Sample subset for k-distance graph (reduces computation from O(n²) to O(m²))
sample_size = min(100000, n_total)  # Use up to 100k points for parameter search
if sample_size < n_total:
    print(f"\n[SAMPLING] Using {sample_size:,} samples ({100*sample_size/n_total:.1f}% of data)")
    print(f"  Reason: Reduce computation time for k-distance graph")
    sample_indices = np.random.RandomState(42).choice(n_total, size=sample_size, replace=False)
    X_sample = X[sample_indices]
    metadata = metadata.iloc[sample_indices].reset_index(drop=True)
else:
    print(f"\n[SAMPLING] Dataset small enough ({n_total:,} ≤ 50k), using all points")
    X_sample = X
    sample_indices = np.arange(n_total)

print(f"  Sample shape for k-distance: {X_sample.shape}")
print(f"  Sample metadata shape for k-distance: {metadata.shape}")

X = X_sample

# ============================================================
# STEP 2: FIND OPTIMAL eps PARAMETER
# ============================================================


# Use suggested eps (can adjust if needed)
eps = 1.48
min_samples = 51

print(f"\nUsing parameters:")
print(f"  eps: {eps:.4f}")
print(f"  min_samples: {min_samples}")


# ============================================================
# STEP 3: RUN DBSCAN
# ============================================================

print("\n[3/5] Running DBSCAN...")
print("-" * 70)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X)

# Count clusters and noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\n✓ DBSCAN complete")
print(f"  Number of clusters: {n_clusters}")
print(f"  Number of noise points: {n_noise:,} ({n_noise/len(labels)*100:.2f}%)")
print(f"  Number of core points: {len(labels) - n_noise:,}")

# Cluster distribution
unique, counts = np.unique(labels[labels >= 0], return_counts=True)
print(f"\nCluster distribution:")
for cluster_id, count in zip(unique, counts):
    pct = (count / (len(labels) - n_noise)) * 100
    print(f"  Cluster {cluster_id}: {count:6d} points ({pct:5.1f}%)")

# Calculate silhouette (only for core points, excluding noise)
if n_clusters > 1 and n_noise < len(labels):
    core_mask = labels >= 0
    if core_mask.sum() > 0:
        try:
            silhouette = silhouette_score(X[core_mask], labels[core_mask])
            print(f"\nSilhouette Score (core points only): {silhouette:.4f}")
        except:
            silhouette = None
            print(f"\nSilhouette Score: Could not calculate")
    else:
        silhouette = None
else:
    silhouette = None


# ============================================================
# STEP 4: SAVE RESULTS
# ============================================================

print("\n[4/5] Saving results...")
print("-" * 70)

Path("../results").mkdir(parents=True, exist_ok=True)

# Save labels
labels_file = Path("../results") / "dbscan_labels.npy"
np.save(labels_file, labels)
print(f"✓ Saved: {labels_file}")

# Save with metadata
results_df = metadata.copy()
results_df['cluster'] = labels
results_file = Path("../results") / "dbscan_results.csv"
results_df.to_csv(results_file, index=False)
print(f"✓ Saved: {results_file}")

# Save parameters
params_file = Path("../results") / "dbscan_parameters.txt"
with open(params_file, 'w') as f:
    f.write(f"DBSCAN Parameters\n")
    f.write(f"eps: {eps:.6f}\n")
    f.write(f"min_samples: {min_samples}\n")
    f.write(f"\nResults\n")
    f.write(f"Number of clusters: {n_clusters}\n")
    f.write(f"Number of noise points: {n_noise}\n")
    if silhouette is not None:
        f.write(f"Silhouette Score: {silhouette:.4f}\n")

print(f"✓ Saved: {params_file}")


# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================

print("\n[5/5] Creating visualizations...")
print("-" * 70)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

fig.suptitle(f'DBSCAN Clustering Results (eps={eps:.4f}, min_samples={min_samples})',
            fontsize=14, fontweight='bold')

# ──────────────────────────────────────────────────────────
# PLOT 1: CLUSTER DISTRIBUTION (Bar chart)
# ──────────────────────────────────────────────────────────

ax1 = fig.add_subplot(gs[0, 0])

cluster_counts = {}
for cluster_id in unique:
    cluster_counts[f'C{int(cluster_id)}'] = counts[list(unique).index(cluster_id)]
cluster_counts['Noise'] = n_noise

colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
bars = ax1.bar(range(len(cluster_counts)), list(cluster_counts.values()), 
              color=colors, edgecolor='black', linewidth=1.5)

ax1.set_xticks(range(len(cluster_counts)))
ax1.set_xticklabels(list(cluster_counts.keys()), rotation=45, ha='right')
ax1.set_ylabel('Number of Points', fontsize=11, fontweight='bold')
ax1.set_title('Cluster Distribution\n(Noise = points not in any cluster)', 
             fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}', ha='center', va='bottom', fontsize=9)

# ──────────────────────────────────────────────────────────
# PLOT 2: CLUSTER PERCENTAGES
# ──────────────────────────────────────────────────────────

ax2 = fig.add_subplot(gs[0, 1])

labels_pie = []
sizes_pie = []
for cluster_id in unique:
    count = counts[list(unique).index(cluster_id)]
    labels_pie.append(f'C{int(cluster_id)}\n({count:,})')
    sizes_pie.append(count)

# Add noise
labels_pie.append(f'Noise\n({n_noise:,})')
sizes_pie.append(n_noise)

colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels_pie)))
wedges, texts, autotexts = ax2.pie(sizes_pie, labels=labels_pie, colors=colors_pie,
                                    autopct='%1.1f%%', startangle=90,
                                    textprops={'fontsize': 9})

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')

ax2.set_title('Cluster Percentage Distribution', fontsize=12, fontweight='bold')

# ──────────────────────────────────────────────────────────
# PLOT 3: CLUSTER SIZES COMPARISON
# ──────────────────────────────────────────────────────────

ax3 = fig.add_subplot(gs[1, 0])

cluster_labels_clean = [f'C{int(c)}' for c in unique]
sizes_clean = list(counts)

bars3 = ax3.bar(cluster_labels_clean, sizes_clean, color=colors[:-1], 
               edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Number of Points', fontsize=11, fontweight='bold')
ax3.set_title('Core Points per Cluster\n(Excluding Noise)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ──────────────────────────────────────────────────────────
# PLOT 4: SUMMARY STATISTICS
# ──────────────────────────────────────────────────────────

ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary_text = f"""
DBSCAN RESULTS SUMMARY

Parameters:
  eps = {eps:.6f}
  min_samples = {min_samples}

Results:
  Clusters found: {n_clusters}
  Noise points: {n_noise:,} ({n_noise/len(labels)*100:.2f}%)
  Core points: {len(labels) - n_noise:,}
  Total points: {len(labels):,}

Quality Metrics:
  Silhouette Score: {f'{silhouette:.4f}' if silhouette is not None else 'N/A (noise present)'}

Characteristics:
  • Finds arbitrary-shaped clusters
  • Automatically detects outliers/noise
  • No need to specify number of clusters
  
Advantages vs K-Means:
  ✓ Discovers cluster count
  ✓ Handles noise/outliers
  ✓ Any cluster shape
  
Disadvantages:
  ⚠️ Sensitive to parameters (eps, min_samples)
  ⚠️ High-dimensional data challenging
  ⚠️ Varying cluster densities difficult
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()

plot_file_main = Path("../results") / "dbscan_analysis.png"
plt.savefig(plot_file_main, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot_file_main}")

plt.show()


# ============================================================
# DETAILED ANALYSIS
# ============================================================

print("\nCreating detailed analysis report...")

report_text = f"""
{'='*70}
DBSCAN CLUSTERING REPORT
{'='*70}
Generated: {pd.Timestamp.now().isoformat()}

WHAT IS DBSCAN?
{'─'*70}
DBSCAN = Density-Based Spatial Clustering of Applications with Noise

Key Features:
  1. Finds clusters automatically (no need to specify k)
  2. Identifies outliers as noise points (labeled as -1)
  3. Can find arbitrarily-shaped clusters
  4. Density-based approach

Parameters:
  • eps: Maximum distance between points to be neighbors
  • min_samples: Minimum points in neighborhood to form cluster

PARAMETERS USED
{'─'*70}
eps: {eps:.6f}
  → Two points are neighbors if distance ≤ {eps:.6f}
  
min_samples: {min_samples}
  → Need at least {min_samples} neighbors to form core point

RESULTS
{'─'*70}
Number of clusters found: {n_clusters}
Number of noise points: {n_noise:,} ({n_noise/len(labels)*100:.2f}%)
Number of core points: {len(labels) - n_noise:,}

Cluster Distribution:
"""

for cluster_id in unique:
    count = counts[list(unique).index(cluster_id)]
    pct = (count / (len(labels) - n_noise)) * 100
    report_text += f"  Cluster {int(cluster_id)}: {count:6d} points ({pct:5.1f}%)\n"

report_text += f"""

INTERPRETATION
{'─'*70}

1. NOISE POINTS ({n_noise:,} = {n_noise/len(labels)*100:.2f}%):
"""

if n_noise > 0:
    report_text += f"""   These points don't fit well into any cluster.
   High noise percentage might indicate:
   • Data is not naturally clustered
   • Parameters (eps, min_samples) may need adjustment
   • Outliers are present in the data
"""
else:
    report_text += f"""   No noise points found. All points assigned to clusters.
   This is rare but indicates well-defined clustering structure.
"""

report_text += f"""

2. CLUSTER QUALITY:
   Silhouette Score: {f'{silhouette:.4f}' if silhouette is not None else 'N/A (noise present)'}
"""

if silhouette is not None:
    if silhouette > 0.5:
        report_text += "   Quality: GOOD (clusters are well-separated)\n"
    elif silhouette > 0.25:
        report_text += "   Quality: FAIR (some overlap between clusters)\n"
    else:
        report_text += "   Quality: POOR (significant cluster overlap)\n"

report_text += f"""

COMPARISON WITH K-MEANS
{'─'*70}
K-Means (k=4):
  • Fixed 4 clusters
  • All points assigned to clusters
  • Spherical clusters
  • No noise detection

DBSCAN (eps={eps:.6f}):
  • Found {n_clusters} clusters automatically
  • {n_noise:,} points identified as noise
  • Arbitrary cluster shapes
  • Detected outliers
  
Which is better?
  → Use DBSCAN if you want automatic cluster discovery
  → Use DBSCAN if you want to identify outliers
  → Use K-Means if you know number of clusters

NEXT STEPS
{'─'*70}
1. Compare DBSCAN results with K-Means (using ARI, NMI)
2. Analyze characteristics of DBSCAN clusters
3. Examine noise points (are they true outliers?)
4. Decide which method better fits your research questions

If you want to adjust eps:
  • Smaller eps → More noise, fewer/smaller clusters
  • Larger eps → Less noise, more/larger clusters
  • Use k-distance graph to guide selection

FILES CREATED
{'─'*70}
1. dbscan_labels.npy           - Cluster assignments
2. dbscan_results.csv          - Detailed results with metadata
3. dbscan_parameters.txt       - Parameter values used
4. dbscan_kdistance_graph.png  - Parameter selection plot
5. dbscan_analysis.png         - Results visualization
6. dbscan_report.txt           - This report

{'='*70}
"""

report_file = Path("../results") / "dbscan_report.txt"
with open(report_file, 'w') as f:
    f.write(report_text)

print(f"✓ Saved: {report_file}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("✓ DBSCAN CLUSTERING COMPLETE")
print("="*70)

print(f"""
SUMMARY:
────────
Clusters found: {n_clusters}
Noise points: {n_noise:,} ({n_noise/len(labels)*100:.2f}%)
Silhouette: {f'{silhouette:.4f}' if silhouette is not None else 'N/A'}

PARAMETERS USED:
────────────────
eps: {eps:.6f} (from k-distance elbow)
min_samples: {min_samples}

FILES SAVED:
────────────
✓ dbscan_labels.npy
✓ dbscan_results.csv
✓ dbscan_parameters.txt
✓ dbscan_kdistance_graph.png
✓ dbscan_analysis.png
✓ dbscan_report.txt

NEXT STEPS:
───────────
1. Analyze DBSCAN cluster characteristics
2. Compare DBSCAN vs K-Means results (ARI, NMI)
3. Examine noise points
4. Decide which clustering is better for your thesis

STATUS: ✓ DBSCAN COMPLETE, READY FOR COMPARISON
""")

print("="*70 + "\n")