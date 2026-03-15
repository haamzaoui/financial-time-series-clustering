from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd

# Schritt 1: Optimales epsilon finden
def find_optimal_epsilon(data, k=4):
    """
    k-distance graph to find epsilon
    """
    # Berechne k-nearest neighbors distances
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(data)
    distances, indices = neighbors.kneighbors(data)
    
    # Sortiere distances
    distances = np.sort(distances[:, k-1], axis=0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k}-Nearest Neighbor Distance')
    plt.title('k-Distance Graph for epsilon selection')
    plt.grid(True)
    plt.show()
    
    return distances


# Load normalized segments
h5_file = "../../../data/processed/segments_normalized_minmax.h5"
# Alternative: h5_file = "../../data/processed/segments_zscore.h5"

with h5py.File(h5_file, 'r') as f:
    segments_normalized = f['segments'][:]

print(f"✓ Loaded segments: {segments_normalized.shape}")

# Load metadata
metadata = pd.read_csv("../../../data/processed/segments_metadata.csv")
print(f"✓ Loaded metadata: {metadata.shape}")
# Epsilon finden
distances = find_optimal_epsilon(segments_normalized, k=4)

# Schritt 2: DBSCAN anwenden
epsilon = 0.5  # Basierend auf k-distance graph
min_samples = 50  # Mindestens 50 Segmente pro Cluster

dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(segments_normalized)

# Analyse
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Cluster sizes: {np.bincount(dbscan_labels[dbscan_labels >= 0])}")