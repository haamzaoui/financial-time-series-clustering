# Technical Documentation

## Project Structure

```
financial-time-series-clustering/
├── data/
│   ├── raw/
│   │   ├── sp500_constituents.csv        # S&P 500 ticker list (input)
│   │   └── sp500_prices.parquet          # downloaded adjusted close prices
│   └── processed/
│       ├── segments.h5                   # all raw segments (570k×50)
│       ├── segments_metadata.csv         # ticker/position metadata per segment
│       ├── segments_normalized_minmax.h5 # min-max normalized segments
│       ├── segments_normalized_zscore.h5 # z-score normalized segments
│       ├── sample_50k.h5                 # 50k random sample (segments + indices)
│       └── sample_50k_metadata.csv       # metadata for the 50k sample
├── src/
│   ├── download/
│   │   ├── download_data.py
│   │   └── check_quality.py
│   ├── preprocessing/
│   │   ├── segmentation.py
│   │   ├── normalization.py
│   │   └── sampling.py
│   ├── clustering/
│   │   ├── kmeans/
│   │   │   ├── elbow_kmeans.py
│   │   │   ├── silhouette_kmeans.py
│   │   │   ├── kmeans.py
│   │   │   └── results/
│   │   ├── hierarchical/
│   │   │   ├── agglomerative.py
│   │   │   └── results/
│   │   └── dbscan/
│   │       ├── dbscan.py
│   │       ├── v2_dbscan.py
│   │       └── results/
│   └── evaluation/
│       ├── cluster_quality.py
│       ├── consistency.py
│       ├── tsne.py
│       ├── stats.py
│       └── results/
├── requirements.txt
└── DOCUMENTATION.md
```

---

## Dependencies

```
yfinance
pandas
lxml
fastparquet
numpy
matplotlib
scikit-learn
scipy
h5py
```

Install with: `pip install -r requirements.txt`

---


## Execution Order

Run scripts in the following sequence. Each step depends on outputs from previous steps.

```bash
# 1. Download price data
python src/download/download_data.py

# 2. (Optional) Validate data quality
python src/download/check_quality.py

# 3. Segment the time series
python src/preprocessing/segmentation.py

# 4. Normalize segments
python src/preprocessing/normalization.py

# 5. Draw the 50k random sample
python src/preprocessing/sampling.py

# 6. K-Means: determine k
python src/clustering/kmeans/elbow_kmeans.py
python src/clustering/kmeans/silhouette_kmeans.py

# 7. Run all three clustering algorithms
python src/clustering/kmeans/kmeans.py
python src/clustering/hierarchical/agglomerative.py
python src/clustering/dbscan/dbscan.py
#    If dbscan.py produces 1 cluster, run instead:
#    python src/clustering/dbscan/v2_dbscan.py

# 8. Evaluate results (steps 8a–8d can run in any order after step 7)
python src/evaluation/cluster_quality.py
python src/evaluation/consistency.py
python src/evaluation/tsne.py
python src/evaluation/stats.py
```

**Note on paths:** All scripts resolve paths relative to the project root using `Path(__file__).resolve().parents[N]`. Run each script from any working directory; absolute paths are constructed automatically.

---

## Data Flow Summary

```
sp500_constituents.csv
        │
        ▼
download_data.py  ──────────►  sp500_prices.parquet / .csv
        │
        ▼
segmentation.py  ───────────►  segments.h5  +  segments_metadata.csv
        │
        ▼
normalization.py  ──────────►  segments_normalized_minmax.h5  (+ zscore)
        │
        ▼
sampling.py  ───────────────►  sample_50k.h5  +  sample_50k_metadata.csv
        │
        ├──► elbow_kmeans.py      ──►  elbow_analysis.png
        ├──► silhouette_kmeans.py ──►  silhouette_scores.png + report
        │
        ├──► kmeans.py            ──►  kmeans_labels.npy  +  kmeans_centers.npy
        ├──► agglomerative.py     ──►  hierarchical_labels.npy  +  centers.npy
        └──► dbscan.py            ──►  dbscan_labels.npy
                │
                ├──► cluster_quality.py  ──►  internal metrics (Silhouette/CH/DB)
                ├──► consistency.py      ──►  ARI / NMI matrices
                ├──► tsne.py             ──►  2D visualizations
                └──► stats.py            ──►  per-cluster statistics
```

-