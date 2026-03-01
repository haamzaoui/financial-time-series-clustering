import pandas as pd
import numpy as np
from pathlib import Path
import h5py  


# LOAD DATA

print("\n[1/5] LOADING DATA...")
print("-" * 70)

# Read the CSV file you downloaded
prices = pd.read_csv("../../data/raw/sp500_prices.csv", index_col=0, parse_dates=True)

print(f"✓ Loaded successfully")
print(f"  Rows (trading days): {prices.shape[0]:,}")
print(f"  Columns (stocks): {prices.shape[1]}")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"  Total price points: {prices.shape[0] * prices.shape[1]:,}")


# CREATE SEGMENTS & METADATA

print("\n[2/5] CREATING SEGMENTS...")
print("-" * 70)

SEGMENT_LENGTH = 50      # Each segment = 50 trading days
STRIDE = 5               # Step forward by 5 days (90% overlap)

print(f"Segment length: {SEGMENT_LENGTH} days")
print(f"Stride: {STRIDE} days")
print(f"Overlap: {100 * (1 - STRIDE/SEGMENT_LENGTH):.0f}%")

# Lists to store segments and metadata
segments = []
metadata = []

# Loop through each stock
for ticker_idx, ticker in enumerate(prices.columns):

    # ============================================================
    # Get prices for this specific stock
    # ============================================================
    # prices[ticker] gives us a column (all prices for one stock)
    # .dropna().values removes NaN values and converts to numpy array
    prices_array = prices[ticker].dropna().values
    
    # Skip stocks with insufficient data
    if len(prices_array) < SEGMENT_LENGTH:
        print(f"  [SKIP] {ticker}: Only {len(prices_array)} days (need {SEGMENT_LENGTH})")
        continue
    
    for i in range(0, len(prices_array) - SEGMENT_LENGTH, STRIDE):
        # Extract 50 consecutive prices starting at position i
        segment = prices_array[i:i+SEGMENT_LENGTH]
        
        # Store the segment
        segments.append(segment)
        
        # Store metadata (info about this segment)
        metadata.append({
            'segment_idx': len(segments) - 1,  # Unique ID for this segment
            'ticker': ticker,                   # Which stock
            'start_day': i,                    # Position in stock's time series
        })

# Convert segments list to numpy array (more efficient)
X = np.array(segments, dtype=np.float32)

print(f"\n✓ Segmentation complete")
print(f"  Total segments created: {len(segments):,}")
print(f"  Array shape: {X.shape}")
print(f"  Per stock (average): {len(segments) / 498:.0f} segments")


# ============================================================
# PART 3: SAVE RAW SEGMENTS (HDF5 Format)
# ============================================================

print("\n[3/5] SAVING RAW SEGMENTS...")
print("-" * 70)

# HDF5 is best for large arrays because:
# - Compressed (saves space)
# - Fast access
# - Can store metadata
# - Professional format

h5_file = "../../data/processed/segments.h5"
with h5py.File(h5_file, 'w') as f:
    # Create dataset for segments
    f.create_dataset(
        'segments',
        data=X,
        compression='gzip',  # Compress to save space
        compression_opts=4   # Compression level (1-9)
    )
    
    # Store metadata about the array
    f.attrs['num_segments'] = len(segments)
    f.attrs['segment_length'] = SEGMENT_LENGTH
    f.attrs['stride'] = STRIDE
    f.attrs['num_stocks'] = 498

print(f"✓ Saved HDF5 format: {h5_file}")
print(f"  Shape: {X.shape}")
print(f"  Data type: {X.dtype}")
print(f"  Range: [${X.min():.2f}, ${X.max():.2f}]")

# ALSO save as .npy (numpy format) for compatibility
npy_file = "../../data/processed/segments.npy"
np.save(npy_file, X)

print(f"✓ Also saved NPY format: {npy_file}")

#save as csv 
#df = pd.DataFrame(X)
#df.to_csv("../../data/processed/segments.csv")
#print(f"✓ Also saved csv format")

# SAVE METADATA
#########################################################################

print("\n[4/5] SAVING METADATA...")
print("-" * 70)

# Convert metadata list to pandas DataFrame
metadata_df = pd.DataFrame(metadata)

# Save as CSV for easy inspection
csv_file = "../../data/processed/segments_metadata.csv"
metadata_df.to_csv(csv_file, index=False)

print(f"✓ Saved metadata: {csv_file}")
print(f"  Rows: {len(metadata_df):,}")
print(f"  Columns: {list(metadata_df.columns)}")

# Display sample
print(f"\nSample metadata (first 10 rows):")
print(metadata_df.head(10).to_string())


# ============================================================
# PART 5: SUMMARY AND VERIFICATION
# ============================================================

print("\n[5/5] VERIFICATION...")
print("-" * 70)

# Verify files were saved
files = {
    'HDF5 segments': h5_file,
    'NPY segments': npy_file,
    'Metadata CSV': csv_file
}

for name, path in files.items():
    if Path(path).exists():
        size = Path(path).stat().st_size / (1024**2)
        print(f"✓ {name}: {size:.1f} MB")
    else:
        print(f"✗ {name}: NOT FOUND")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("✓ SEGMENTATION COMPLETE")
print("="*70)

print(f"""
SUMMARY:
────────
Segments created: {len(segments):,}
Metadata rows: {len(metadata_df):,}
Array shape: {X.shape}
Per stock: {len(segments) / 498:.0f} segments (average)

FILES SAVED:
────────────
1. data/processed/segments_raw.h5        (HDF5 - main file)
2. data/processed/segments_raw.npy       (NumPy - backup)
3. data/processed/segment_metadata.csv   (Metadata)

""")

print("="*70 + "\n")