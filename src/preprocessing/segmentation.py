import pandas as pd
import numpy as np

# Load
prices = pd.read_csv("../../data/raw/sp500_prices.csv", index_col=0, parse_dates=True)

# Create segments
segments = []
for ticker in prices.columns:
    prices_arr = prices[ticker].dropna().values
    
    # Stride = 25 (50% overlap)
    for i in range(0, len(prices_arr) - 50, 10):
        segment = prices_arr[i:i+50]
        segments.append(segment)

# Convert to array
X = np.array(segments)
print(f"Created: {X.shape}")  # (125000+, 50)

# Save
np.save("../../data/processed/segments.npy", X)