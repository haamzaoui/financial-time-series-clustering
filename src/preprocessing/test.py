
import h5py

# Open the file
with h5py.File('../../data/processed/segments_normalized_minmax.h5', 'r') as f:
    # List all datasets
    print("Datasets:", list(f.keys()))
    
    # Assuming your dataset is named 'data'
    dataset = f['segments']
    
    # Print first 10 rows
    print(dataset[:2])
print("*" * 70)

with h5py.File('../../data/processed/segments_normalized_zscore.h5', 'r') as f:
    # List all datasets
    print("Datasets:", list(f.keys()))
    
    # Assuming your dataset is named 'data'
    dataset = f['segments']
    
    # Print first 10 rows
    print(dataset[:2])
