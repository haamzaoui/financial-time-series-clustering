"""
For inspecting .npy files, which are NumPy array files."""

import numpy as np

data = np.load("clustering/kmeans/results/kmeans_centers.npy")
print(type(data))
print(data.shape)
print(data.dtype)
print("*"*70)
print(data)