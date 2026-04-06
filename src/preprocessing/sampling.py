import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]

H5_FILE      = ROOT / "data/processed/segments_normalized_minmax.h5"
METADATA_CSV = ROOT / "data/processed/segments_metadata.csv"
OUTPUT_H5    = ROOT / "data/processed/sample_50k.h5"
OUTPUT_META  = ROOT / "data/processed/sample_50k_metadata.csv"
SAMPLE_SIZE  = 50000
RANDOM_STATE = 42


# ============================================================
# STEP 1: LOAD
# ============================================================

print("\n[1/3] Loading data...")
print("-" * 70)

with h5py.File(H5_FILE, "r") as f:
    X = f["segments"][:]

metadata = pd.read_csv(METADATA_CSV)

print(f"✓ Loaded segments : {X.shape}")
print(f"✓ Loaded metadata : {metadata.shape}")

# ============================================================
# STEP 2: SAMPLE
# ============================================================

print(f"\n[2/3] Drawing random sample (n={SAMPLE_SIZE:,}, seed={RANDOM_STATE})...")
print("-" * 70)

t0  = time.time()
rng = np.random.default_rng(RANDOM_STATE)

sample_idx = rng.choice(len(X), size=SAMPLE_SIZE, replace=False)
sample_idx.sort()                                   # preserve chronological order

X_sample        = X[sample_idx]
metadata_sample = metadata.iloc[sample_idx]

print(f"✓ Sample shape    : {X_sample.shape} ")


# ============================================================
# STEP 3: SAVE
# ============================================================

print(f"\n[3/3] Saving outputs...")
print("-" * 70)

# HDF5 — segments + indices
with h5py.File(OUTPUT_H5, "w") as f:
    f.create_dataset("segments", data=X_sample,  compression="gzip", compression_opts=4)
    f.create_dataset("indices",  data=sample_idx)
    f.attrs["sample_size"]    = SAMPLE_SIZE
    f.attrs["random_state"]   = RANDOM_STATE
    f.attrs["segment_length"] = X_sample.shape[1]
    f.attrs["source"]         = H5_FILE

size_mb = Path(OUTPUT_H5).stat().st_size / 1e6
print(f"✓ Saved HDF5     : {OUTPUT_H5}  ({size_mb:.1f} MB)")

# CSV — metadata
metadata_sample.to_csv(OUTPUT_META, index=False)
print(f"✓ Saved metadata : {OUTPUT_META}  ({len(metadata_sample):,} rows)")


