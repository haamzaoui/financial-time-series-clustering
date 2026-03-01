import numpy as np
import h5py
from pathlib import Path

print("\n" + "="*70)
print("DUAL NORMALIZATION (Min-Max & Z-Score)")
print("="*70)


# ============================================================
# PART 1: LOAD RAW SEGMENTS
# ============================================================

print("\n[1/5] LOADING RAW SEGMENTS...")
print("-" * 70)

h5_file = "../../data/processed/segments.h5"
npy_file = "../../data/processed/segments.npy"

try:
    print(f"Loading from HDF5: {h5_file}")
    with h5py.File(h5_file, 'r') as f:
        X_raw = f['segments'][:]
    print(f"✓ Loaded from HDF5")
except FileNotFoundError:
    print(f"HDF5 not found, loading from NPY: {npy_file}")
    X_raw = np.load(npy_file)
    print(f"✓ Loaded from NPY")

print(f"  Shape: {X_raw.shape}")
print(f"  Data type: {X_raw.dtype}")
print(f"  Range: [${X_raw.min():.2f}, ${X_raw.max():.2f}]")
print(f"  Segments: {len(X_raw):,}")


# ============================================================
# PART 2: MIN-MAX NORMALIZATION [0, 1]
# ============================================================

print("\n[2/5] MIN-MAX NORMALIZATION...")
print("-" * 70)

X_minmax = np.zeros_like(X_raw, dtype=np.float32)
flat_minmax = 0             # count the flat segments (Values are the same)
flat_segments = []
for i in range(len(X_raw)):
    segment = X_raw[i]
    x_min = segment.min()
    x_max = segment.max()
    
    if x_max > x_min:
        X_minmax[i] = (segment - x_min) / (x_max - x_min)
    else:
        X_minmax[i] = 0.5  # Flat segment
        flat_minmax += 1
        flat_segments.append(i)

print(f"\n✓ Min-Max normalization complete")
print(f"  Range: [{X_minmax.min():.6f}, {X_minmax.max():.6f}]")
print(f"  Mean: {X_minmax.mean():.6f}")
print(f"  Std: {X_minmax.std():.6f}")
print(f"  Flat segments: {flat_minmax:,}")
print(f"Flat segments : {flat_segments}")


# ============================================================
# PART 3: Z-SCORE NORMALIZATION
# ============================================================

print("\n[3/5] Z-SCORE NORMALIZATION...")
print("-" * 70)

X_zscore = np.zeros_like(X_raw, dtype=np.float32)
zero_std = 0

for i in range(len(X_raw)):
    segment = X_raw[i]
    x_mean = segment.mean()
    x_std = segment.std()
    
    if x_std > 0:
        X_zscore[i] = (segment - x_mean) / x_std
    else:
        X_zscore[i] = 0  # Flat segment (std=0)
        zero_std += 1

print(f"\n✓ Z-Score normalization complete")
print(f"  Range: [{X_zscore.min():.6f}, {X_zscore.max():.6f}]")
print(f"  Mean: {X_zscore.mean():.6f}")
print(f"  Std: {X_zscore.std():.6f}")
print(f"  Zero std segments: {zero_std:,}")


# ============================================================
# PART 4: SAVE BOTH NORMALIZED VERSIONS
# ============================================================

print("\n[4/5] SAVING NORMALIZED SEGMENTS...")
print("-" * 70)

output_dir = Path("../../data/processed")

# Save Min-Max as HDF5
h5_minmax = output_dir / "segments_normalized_minmax.h5"
with h5py.File(str(h5_minmax), 'w') as f:
    f.create_dataset('segments', data=X_minmax, 
                     compression='gzip', compression_opts=4)
    f.attrs['method'] = 'min-max'
    f.attrs['range'] = '[0, 1]'
    f.attrs['num_segments'] = len(X_minmax)

print(f"✓ Saved Min-Max HDF5: {h5_minmax}")
minmax_size = h5_minmax.stat().st_size / (1024**2)
print(f"  Size: {minmax_size:.1f} MB")

# Save Z-Score as HDF5
h5_zscore = output_dir / "segments_normalized_zscore.h5"
with h5py.File(str(h5_zscore), 'w') as f:
    f.create_dataset('segments', data=X_zscore, 
                     compression='gzip', compression_opts=4)
    f.attrs['method'] = 'z-score'
    f.attrs['range'] = '[-inf, inf]'
    f.attrs['num_segments'] = len(X_zscore)

print(f"✓ Saved Z-Score HDF5: {h5_zscore}")
zscore_size = h5_zscore.stat().st_size / (1024**2)
print(f"  Size: {zscore_size:.1f} MB")

# Also save as NPY for compatibility
npy_minmax = output_dir / "segments_normalized_minmax.npy"
np.save(str(npy_minmax), X_minmax)
print(f"✓ Saved Min-Max NPY: {npy_minmax}")

npy_zscore = output_dir / "segments_normalized_zscore.npy"
np.save(str(npy_zscore), X_zscore)
print(f"✓ Saved Z-Score NPY: {npy_zscore}")
