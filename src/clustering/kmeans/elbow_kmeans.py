import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / "results"

# ==============================
# CONFIG
# ==============================
h5_file = ROOT / "data/processed/sample_50k.h5"
normalization_method = "Min-Max"

k_values = list(range(2, 11))  # Test k=2..10

# ==============================
# STEP 1: LOAD DATA
# ==============================

print(f"Loading normalized segments from: {h5_file}")
with h5py.File(h5_file, 'r') as f:
    X = f['segments'][:]

print(f"Data shape: {X.shape}, segments: {len(X):,}\n")

# ==============================
# STEP 2: COMPUTE INERTIA FOR DIFFERENT k
# ==============================

inertias = []

print(f"{'k':<5} {'Inertia':<20}")
print("-"*30)
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    inertia = kmeans.inertia_
    inertias.append(inertia)
    print(f"{k:<5} {inertia:<20.0f}")

# ==============================
# STEP 3: FIND ELBOW
# ==============================
inertia_array = np.array(inertias)
first_derivative = np.diff(inertia_array)
second_derivative = np.diff(first_derivative)
elbow_idx = np.argmax(second_derivative)
optimal_k = k_values[elbow_idx + 1]  # +1 because of diff
print(f"\nOptimal k (Elbow Method) = {optimal_k}")

# ==============================
# STEP 4: PLOT ELBOW
# ==============================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8, label='Inertia')
ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
           label=f'Elbow at k={optimal_k}')
ax.scatter([optimal_k], [inertias[optimal_k-2]], 
           s=200, c='red', marker='*', edgecolors='darkred', linewidth=2)

ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Inertia (Sum of Squared Distances)', fontsize=12, fontweight='bold')
ax.set_title(f'Elbow Method - {normalization_method} Normalization', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xticks(k_values)
ax.legend(fontsize=11)
plt.tight_layout()

# Save plot
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plot_file = RESULTS_DIR / "elbow_analysis.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Saved elbow plot: {plot_file}")

plt.show()