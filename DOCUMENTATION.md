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

## Module Reference

### 1. `src/download/download_data.py`

Downloads adjusted closing prices for all S&P 500 tickers from Yahoo Finance.

**Configuration constants:**

| Variable | Value | Description |
|---|---|---|
| `CSV_FILE` | `data/raw/sp500_constituents.csv` | Ticker source CSV (column: `Symbol`) |
| `START_DATE` | `2000-01-01` | Start of download window |
| `END_DATE` | `2026-01-01` | End of download window |
| `DELAY_BETWEEN_REQUESTS` | `0.05` | Throttle delay (seconds) |

**Behavior:**
- Converts ticker symbols from `.` to `-` (e.g. `BRK.B` → `BRK-B`) for Yahoo Finance compatibility.
- Downloads all tickers in a single batched call with `threads=True`.
- Extracts the `Close` column per ticker; skips any ticker with no data.
- Drops columns that are entirely NaN (fully missing tickers).
- Prints a quality summary: stocks downloaded, trading days, date range, completeness %.

**Outputs:**

| File | Format | Description |
|---|---|---|
| `data/raw/sp500_prices.parquet` | Parquet (snappy) | Wide-format price matrix (dates × tickers) |
| `data/raw/sp500_prices.csv` | CSV | Same data as CSV backup |

---

### 2. `src/download/check_quality.py`

Validates the downloaded price data against data corruption.

**Checks performed:**
1. Invalid prices: any value ≤ 0 across the full matrix.
2. Duplicate dates: duplicate row indices in the date index.
3. Complete history: count of tickers with no missing values over the full period.
4. Sample display: prints first 5 rows × 5 columns for visual inspection.

**Note:** Missing values due to IPO dates or delisting are expected and not flagged as errors. The script distinguishes between structural gaps (normal) and zero/negative prices (data corruption).

**Input:** `data/raw/sp500_prices.csv`
**Output:** Console only.

---

### 3. `src/preprocessing/segmentation.py`

Applies a sliding window to each stock's price series to create fixed-length segments.

**Configuration constants:**

| Variable | Value | Description |
|---|---|---|
| `SEGMENT_LENGTH` | `50` | Number of trading days per segment |
| `STRIDE` | `5` | Step size between segment starts (90% overlap) |

**Behavior:**
- Iterates over each ticker column in the price matrix.
- Skips tickers with fewer than 50 non-NaN trading days.
- For each valid position `i` in `range(0, len(prices) - 50, 5)`, extracts `prices[i:i+50]`.
- Stores the raw price values (not normalized) as `float32`.

**Outputs:**

| File | Description |
|---|---|
| `data/processed/segments.h5` | HDF5 dataset `segments`, shape `(N, 50)`, gzip compressed, level 4 |
| `data/processed/segments_metadata.csv` | Columns: `segment_idx`, `ticker`, `start_day` |

HDF5 attributes stored: `num_segments`, `segment_length`, `stride`, `num_stocks`.

---

### 4. `src/preprocessing/normalization.py`

Applies per-segment Min-Max and Z-Score normalization to the raw segments.

**Min-Max normalization:**

```
x_norm[i] = (x[i] - min(x)) / (max(x) - min(x))
```

Flat segments (where `max == min`) are assigned the constant `0.5`.

**Z-Score normalization:**

```
x_norm[i] = (x[i] - mean(x)) / std(x)
```

Flat segments (where `std == 0`) are assigned `0`.

**Inputs:** `data/processed/segments.h5` (falls back to `segments.npy` if HDF5 not found)

**Outputs:**

| File | HDF5 attrs |
|---|---|
| `data/processed/segments_normalized_minmax.h5` | `method='min-max'`, `range='[0, 1]'`, `num_segments` |
| `data/processed/segments_normalized_zscore.h5` | `method='z-score'`, `range='[-inf, inf]'`, `num_segments` |

Both datasets use gzip compression level 4, `float32` dtype.

---

### 5. `src/preprocessing/sampling.py`

Draws a reproducible random sample of 50,000 segments from the full normalized dataset.

**Configuration constants:**

| Variable | Value |
|---|---|
| `SAMPLE_SIZE` | `50000` |
| `RANDOM_STATE` | `42` |

**Behavior:**
- Uses `numpy.random.default_rng(42).choice(N, size=50000, replace=False)`.
- Sorts sampled indices to preserve chronological ordering.
- Saves both the segment matrix and the original indices (for traceability back to the full dataset).

**Inputs:**
- `data/processed/segments_normalized_minmax.h5`
- `data/processed/segments_metadata.csv`

**Outputs:**

| File | Contents |
|---|---|
| `data/processed/sample_50k.h5` | Datasets: `segments` (50000×50), `indices` (50000,); attrs: `sample_size`, `random_state`, `segment_length`, `source` |
| `data/processed/sample_50k_metadata.csv` | Subset of metadata rows corresponding to sampled indices |

---

### 6. `src/clustering/kmeans/elbow_kmeans.py`

Computes K-Means inertia for k = 2 to 10 and identifies the elbow point.

**Configuration:**

| Variable | Value |
|---|---|
| `k_values` | `range(2, 11)` |
| `n_init` | `10` |
| `random_state` | `42` |

**Elbow detection:** Computes the first and second discrete derivatives of the inertia array. The elbow is the index of the maximum value in the second derivative (`np.argmax(np.diff(np.diff(inertias)))`).

**Input:** `data/processed/sample_50k.h5`

**Outputs:**

| File | Description |
|---|---|
| `src/clustering/kmeans/results/elbow_analysis.png` | Inertia vs k with red vertical line at optimal k |

---

### 7. `src/clustering/kmeans/silhouette_kmeans.py`

Computes Silhouette Score for k = 2 to 10 to independently corroborate the elbow result.

**Configuration:**

| Variable | Value |
|---|---|
| `K_RANGE` | `range(2, 11)` |
| `N_INIT` | `10` |
| `RANDOM_STATE` | `42` |

**Behavior:** For each k, fits K-Means and computes `silhouette_score(X, labels)` on all 50,000 points. Selects the k with the highest average score as optimal.

**Input:** `data/processed/sample_50k.h5`

**Outputs:**

| File | Description |
|---|---|
| `results/silhouette_scores.png` | Average silhouette score vs k with starred optimal k |
| `results/silhouette_bars_k{best_k}.png` | Per-point silhouette bar chart grouped by cluster |
| `results/silhouette_report.txt` | Score table, recommendation, quality interpretation |

Silhouette quality thresholds: `> 0.75` Excellent, `0.50–0.75` Good, `0.25–0.50` Fair, `< 0.25` Poor.

---

### 8. `src/clustering/kmeans/kmeans.py`

Runs K-Means clustering with fixed k = 4 on the 50k sample.

**Configuration:**

| Variable | Value |
|---|---|
| `N_CLUSTERS` | `4` |
| `N_INIT` | `10` |
| `RANDOM_STATE` | `42` |

**Behavior:** Uses `sklearn.cluster.KMeans` with `k-means++` initialization (scikit-learn default). Saves cluster labels, cluster center vectors, and a CSV combining metadata with labels.

**Inputs:**
- `data/processed/sample_50k.h5`
- `data/processed/sample_50k_metadata.csv`

**Outputs:**

| File | Description |
|---|---|
| `results/kmeans_labels.npy` | Integer array of shape (50000,), values in {0,1,2,3} |
| `results/kmeans_centers.npy` | Float array of shape (4, 50), one row per cluster center |
| `results/kmeans_results.csv` | Metadata + `cluster` column |
| `results/kmeans_cluster_centers.png` | Line plot of 4 cluster center patterns |
| `results/kmeans_distribution.png` | Bar chart + pie chart of segment counts per cluster |
| `results/kmeans_report.txt` | Config, inertia, iterations, runtime, cluster distribution |

---

### 9. `src/clustering/hierarchical/agglomerative.py`

Runs Ward agglomerative hierarchical clustering on the 50k sample, fixed at k = 4.

**Configuration:**

| Variable | Value |
|---|---|
| `N_CLUSTERS` | `4` (fixed, matches K-Means for comparison) |
| Linkage method | `ward` |
| Distance metric | `euclidean` |

**Behavior:**
- Computes the full linkage matrix `Z` using `scipy.cluster.hierarchy.linkage(X, method='ward')`.
- Cuts the dendrogram at k = 4 using `fcluster(Z, t=4, criterion='maxclust')`.
- Labels are converted from 1-indexed (scipy convention) to 0-indexed.
- Cluster centers are computed as the mean of all segments in each cluster.
- Dendrogram is plotted with `truncate_mode='lastp', p=50` (shows last 50 merges) and a red dashed line at the cut height.

**Inputs:**
- `data/processed/sample_50k.h5`
- `data/processed/sample_50k_metadata.csv`

**Outputs:**

| File | Description |
|---|---|
| `results/hierarchical_labels.npy` | Integer array (50000,), values in {0,1,2,3} |
| `results/hierarchical_centers.npy` | Float array (4, 50) |
| `results/hierarchical_results.csv` | Metadata + `cluster` column |
| `results/hierarchical_dendrogram.png` | Truncated dendrogram with cut line |
| `results/hierarchical_report.txt` | Config, linkage time, cluster distribution |

**Runtime note:** Ward linkage on 50,000 × 50 points requires O(n²) memory and takes approximately 450–500 seconds.

---

### 10. `src/clustering/dbscan/dbscan.py`

Runs DBSCAN with automated parameter selection via k-distance graph and grid search.

**Configuration:**

| Variable | Value | Description |
|---|---|---|
| `N_EPS_CANDIDATES` | `12` | Number of eps values in grid search |
| `NOISE_MIN_PCT` | `5.0` | Minimum acceptable noise fraction (%) |
| `NOISE_MAX_PCT` | `40.0` | Maximum acceptable noise fraction (%) |
| `RANDOM_STATE` | `42` | Seed for silhouette subsampling |

**Parameter selection pipeline:**

**Step 1 — Fix min_samples:**
`min_samples = 2 × D = 2 × 50 = 100` (standard rule for high-dimensional data).

**Step 2 — k-distance graph:**
Fits `NearestNeighbors(n_neighbors=100, algorithm='ball_tree')` on all 50k points.
Sorts the 100th nearest neighbor distance for each point (ascending).
Detects the elbow via maximum second derivative over a 1000-point downsampled curve.
The elbow distance becomes `eps_elbow`.

**Step 3 — Grid search:**
Tests `np.linspace(0.4 × eps_elbow, 3.0 × eps_elbow, 12)` for two min_samples values: `[50, 100]`.
For each combination records: number of clusters, noise %, and silhouette score on core points (subsampled to 5000 if needed).
A combination is marked `OK` if `5% ≤ noise ≤ 40%` and `2 ≤ clusters ≤ 10`.
Selection priority: (1) OK combinations sorted by silhouette; (2) if no silhouette, OK row with noise closest to 20%; (3) if no OK row, fallback to noise closest to 20% across all combinations.

**Step 4 — Final DBSCAN:**
Runs `sklearn.cluster.DBSCAN(eps=chosen_eps, min_samples=chosen_min_s, n_jobs=-1)`.
Noise points receive label `-1`.

**Inputs:**
- `data/processed/sample_50k.h5`
- `data/processed/sample_50k_metadata.csv`

**Outputs:**

| File | Description |
|---|---|
| `results/dbscan_kdistance.png` | Sorted k-distance curve with elbow marker |
| `results/dbscan_param_search.csv` | All grid search combinations and metrics |
| `results/dbscan_param_search.png` | Three-panel plot: clusters / noise% / silhouette vs eps |
| `results/dbscan_labels.npy` | Integer array (50000,); `-1` = noise |
| `results/dbscan_results.csv` | Metadata + `cluster` column |
| `results/dbscan_clusters.png` | Bar + pie chart including noise count |
| `results/dbscan_centers.png` | Mean pattern per cluster (excluding noise) |
| `results/dbscan_report.txt` | Full parameter selection trace and results |

---

### 11. `src/clustering/dbscan/v2_dbscan.py`

Extended DBSCAN variant with relaxed parameter search. Used when the standard script produces a single cluster.

**Key differences from `dbscan.py`:**

| Parameter | `dbscan.py` | `v2_dbscan.py` |
|---|---|---|
| `min_samples` candidates | `[50, 100]` | `[5, 10, 20, 50, 100]` |
| k for distance graph | `100` | `10` |
| eps range | `0.4–3.0 × elbow` | `0.3–1.0 × elbow` (low zone) + `1.0–5.0 × elbow` (high zone) |
| Noise acceptance | `5–40%` | `2–60%` |
| Cluster count target | `2–10` | `2–15` |
| Fallback logic | Noise closest to 20% | Most clusters with lowest noise, then noise closest to 20% |

The extended fallback in v2: if no `OK` combination exists, it first selects the row with `n_clusters >= 2` and the lowest noise; if all runs produced exactly 1 cluster, it falls back to the row with noise closest to 20%.

**Outputs:** Same file names as `dbscan.py` (overwrites if run from the same directory).

---

### 12. `src/evaluation/cluster_quality.py`

Computes three internal validation metrics for each algorithm.

**Metrics:**
- **Silhouette Score** — `sklearn.metrics.silhouette_score(X, labels)` on all non-noise points. Range: [-1, 1], higher is better.
- **Calinski-Harabasz (CH) Index** — `calinski_harabasz_score(X, labels)`. Ratio of between-cluster to within-cluster variance. Higher is better.
- **Davies-Bouldin (DB) Index** — `davies_bouldin_score(X, labels)`. Average ratio of within-cluster scatter to between-cluster distance. Lower is better.

**DBSCAN handling:** If DBSCAN produced fewer than 2 clusters, all three metrics are recorded as `NaN` and a message is printed. Noise points (label = -1) are always excluded before computing any metric.

**Inputs:**
- `data/processed/sample_50k.h5`
- `src/clustering/kmeans/results/kmeans_labels.npy`
- `src/clustering/hierarchical/results/hierarchical_labels.npy`
- `src/clustering/dbscan/results/dbscan_labels.npy`
- `src/clustering/kmeans/results/kmeans_centers.npy` (optional)
- `src/clustering/hierarchical/results/hierarchical_centers.npy` (optional)

**Outputs:**

| File | Description |
|---|---|
| `results/eval_internal_metrics.png` | Three-bar chart (one panel per metric); best bar highlighted with red border |
| `results/eval_cluster_centers_comparison.png` | Side-by-side cluster center line plots for all algorithms with centers |
| `results/eval_centers_kmeans.png` | Individual center plot for K-Means using high-contrast palette |
| `results/eval_centers_hierarchical.png` | Individual center plot for Hierarchical using high-contrast palette |
| `results/eval_internal_report.txt` | Table of all metrics + best-per-metric summary |

**Color palette** (used in individual center plots):
`#E63946` (red), `#457B9D` (steel blue), `#2A9D8F` (teal), `#E9A820` (amber)

---

### 13. `src/evaluation/consistency.py`

Computes pairwise cross-algorithm label agreement using ARI and NMI.

**Metrics:**
- **ARI (Adjusted Rand Index)** — `sklearn.metrics.adjusted_rand_score`. Adjusted for chance. Range: [-1, 1]. Score of 1 = identical; 0 = no better than random.
- **NMI (Normalized Mutual Information)** — `normalized_mutual_info_score`. Range: [0, 1]. Normalized by the average entropy of both label sets.

**DBSCAN handling:** DBSCAN is included in comparison. Its noise label `-1` is treated as an additional group by both metrics.

**Outputs:**

| File | Description |
|---|---|
| `results/eval_consistency_matrix.png` | Two heatmaps (ARI and NMI) with color scale [0, 1] |
| `results/eval_consistency_report.txt` | Full ARI and NMI matrices + pairwise interpretation |

**ARI interpretation thresholds:** `> 0.80` Strong, `0.60–0.80` Moderate, `0.40–0.60` Weak, `< 0.40` Little to none.

---

### 14. `src/evaluation/tsne.py`

Projects the 50k × 50D sample to 2D via t-SNE, then colors the embedding with each algorithm's labels.

**Configuration:**

| Variable | Value |
|---|---|
| `PERPLEXITY` | `30` |
| `MAX_ITER` | `1000` |
| `RANDOM_STATE` | `42` |
| `POINT_SIZE` | `8` |
| `ALPHA` | `0.5` |

**Caching:** On first run, t-SNE coordinates are saved to `results/tsne_coordinates.npy`. Subsequent runs load from cache (skipping the 5–15 min computation). Delete the file to force recomputation.

**Plot function `plot_tsne(ax, X_tsne, labels, algo_name)`:**
- Noise points (label = -1) drawn first in light gray with `alpha=0.3`.
- Cluster points drawn with high-contrast colors and `edgecolors='black', linewidths=0.2`.
- Cluster centroid in t-SNE space marked with a black `+` marker.

**Color palette** (10 colors):
`#E63946`, `#457B9D`, `#2A9D8F`, `#E9A820`, `#6A0572`, `#F4A261`, `#264653`, `#A8DADC`, `#81B29A`, `#F2CC8F`

**Inputs:**
- `data/processed/sample_50k.h5`
- Label files for all three algorithms

**Outputs:**

| File | Description |
|---|---|
| `results/tsne_coordinates.npy` | Cached (50000, 2) float array |
| `results/tsne_kmeans.png` | 10×8 inch scatter, colored by K-Means labels |
| `results/tsne_hierarchical.png` | Same embedding, colored by Hierarchical labels |
| `results/tsne_dbscan.png` | Same embedding, colored by DBSCAN labels |
| `results/tsne_comparison.png` | All three side by side, shared axes |

---

### 15. `src/evaluation/stats.py`

Computes descriptive statistics per cluster per algorithm over the 50-dimensional normalized values.

**Statistics computed per cluster:**
- `n` — number of segments
- `Pct (%)` — percentage of total sample
- `Max`, `Min` — global max/min across all 50×n values in the cluster
- `Variance` — variance of the flattened cluster values
- `Mean` — mean of the flattened cluster values

DBSCAN noise points (label = -1) are included as a "Noise" row.

**Inputs:**
- `data/processed/sample_50k.h5`
- All three label files

**Outputs:**

| File | Description |
|---|---|
| `results/statistics/cluster_statistics.csv` | One row per cluster per algorithm |
| `results/statistics/cluster_statistics_report.txt` | Formatted tables for all three algorithms |

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

---

## Key Parameters Summary

| Parameter | Value | Set in |
|---|---|---|
| Date range | 2000-01-01 – 2026-01-01 | `download_data.py` |
| Segment length | 50 trading days | `segmentation.py` |
| Stride | 5 days (90% overlap) | `segmentation.py` |
| Normalization | Min-Max per segment | `normalization.py` |
| Sample size | 50,000 segments | `sampling.py` |
| Random seed | 42 (all scripts) | all scripts |
| K-Means k | 4 | `kmeans.py` |
| K-Means n_init | 10 | `kmeans.py`, `elbow_kmeans.py`, `silhouette_kmeans.py` |
| Hierarchical linkage | Ward + Euclidean | `agglomerative.py` |
| Hierarchical k | 4 (fixed) | `agglomerative.py` |
| DBSCAN min_samples | 100 (= 2 × 50 dims) | `dbscan.py` |
| DBSCAN eps | Data-driven (k-distance elbow + grid search) | `dbscan.py` |
| t-SNE perplexity | 30 | `tsne.py` |
| t-SNE iterations | 1000 | `tsne.py` |
