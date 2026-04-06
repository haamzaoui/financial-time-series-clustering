"""
DBSCAN CLUSTERING
=================
Density-Based Spatial Clustering of Applications with Noise.
Runs directly on the 50-dimensional normalised segments.

PARAMETER SELECTION STRATEGY:
  1. Fix min_samples = 2 × dimensions = 2 × 50 = 100
     (standard rule-of-thumb for high-dimensional data)
  2. k-distance graph → find natural eps from the elbow
  3. Grid search      → test eps values around the elbow,
                        evaluate by noise %, cluster count, silhouette
  4. Final DBSCAN     → run with chosen parameters

Input:
  data/processed/sample_50k.h5
  data/processed/sample_50k_metadata.csv
Output:
  results/dbscan_kdistance.png
  results/dbscan_param_search.png
  results/dbscan_param_search.csv
  results/dbscan_labels.npy
  results/dbscan_results.csv
  results/dbscan_clusters.png
  results/dbscan_report.txt
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / "results"

print("\n" + "=" * 70)
print("DBSCAN CLUSTERING")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_H5        = ROOT / "data/processed/sample_50k.h5"
SAMPLE_META      = ROOT / "data/processed/sample_50k_metadata.csv"
N_EPS_CANDIDATES = 12      # number of eps values to test in grid search
NOISE_MIN_PCT    = 5.0     # acceptable noise floor (%)
NOISE_MAX_PCT    = 40.0    # acceptable noise ceiling (%)
RANDOM_STATE     = 42


# ============================================================
# STEP 1: LOAD SHARED SAMPLE
# ============================================================

print("\n[1/5] Loading shared 50k sample...")
print("-" * 70)

with h5py.File(SAMPLE_H5, "r") as f:
    X          = f["segments"][:]
    sample_idx = f["indices"][:]

metadata = pd.read_csv(SAMPLE_META)
N, D     = X.shape

# min_samples: 2 × dimensions (rule-of-thumb for high-dimensional data)
MIN_SAMPLES_FIXED = 2 * D

print(f"✓ Loaded sample   : {X.shape}")
print(f"✓ Loaded metadata : {metadata.shape}")
print(f"  Dimensions      : {D}")
print(f"  min_samples     : 2 × {D} = {MIN_SAMPLES_FIXED}")


# ============================================================
# STEP 2: K-DISTANCE GRAPH  →  find natural eps
# ============================================================
# For each point compute its k-th nearest neighbour distance
# (k = MIN_SAMPLES_FIXED). Sort ascending and plot.
# The elbow = the point where the curve bends sharply upward.
# Below the elbow: dense regions (potential clusters).
# Above the elbow: sparse regions (noise).
# The elbow value is the natural eps.
# ============================================================

print(f"\n[2/5] k-distance graph (k={MIN_SAMPLES_FIXED})...")
print("-" * 70)
print("  Fitting NearestNeighbors — may take 2-4 min on 50k × 50D...")

t0   = time.time()
nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES_FIXED,
                        algorithm="ball_tree", n_jobs=-1)
nbrs.fit(X)
distances, _ = nbrs.kneighbors(X)
elapsed_nn   = time.time() - t0

# k-th neighbour distance = last column, sorted ascending
k_distances = np.sort(distances[:, -1])

print(f"✓ NearestNeighbors done : {elapsed_nn:.1f} s")
print(f"  k-distance range      : [{k_distances.min():.4f}, {k_distances.max():.4f}]")
print(f"  Median k-distance     : {np.median(k_distances):.4f}")
print(f"  75th percentile       : {np.percentile(k_distances, 75):.4f}")
print(f"  90th percentile       : {np.percentile(k_distances, 90):.4f}")

# Automatic elbow via maximum second derivative
step      = max(1, len(k_distances) // 1000)
curve     = k_distances[::step]
d2        = np.diff(np.diff(curve))
elbow_idx = int(np.argmax(d2) * step)
eps_elbow = float(k_distances[elbow_idx])

print(f"\n  Elbow at index  : {elbow_idx:,} / {N:,}")
print(f"  Suggested eps   : {eps_elbow:.4f}")

# ── k-distance plot ───────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fig_kd, ax_kd = plt.subplots(figsize=(12, 5))
ax_kd.plot(k_distances, linewidth=1.2, color="steelblue", label="k-distance")
ax_kd.axvline(x=elbow_idx, color="red", linestyle="--", linewidth=1.8,
              label=f"Elbow → eps ≈ {eps_elbow:.4f}")
ax_kd.axhline(y=eps_elbow, color="red", linestyle=":", linewidth=1.2, alpha=0.6)
ax_kd.scatter([elbow_idx], [eps_elbow], s=120, color="red", zorder=5)
ax_kd.set_xlabel("Points sorted by k-distance", fontsize=12, fontweight="bold")
ax_kd.set_ylabel(f"{MIN_SAMPLES_FIXED}-th nearest neighbour distance",
                 fontsize=12, fontweight="bold")
ax_kd.set_title(
    f"k-Distance Graph  (k={MIN_SAMPLES_FIXED})  —  elbow = eps ≈ {eps_elbow:.4f}",
    fontsize=13, fontweight="bold"
)
ax_kd.legend(fontsize=11)
ax_kd.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
fig_kd.savefig(RESULTS_DIR / "dbscan_kdistance.png",
               dpi=300, bbox_inches="tight")
print(f"\n✓ Saved: results/dbscan_kdistance.png")
plt.show()


# ============================================================
# STEP 3: PARAMETER GRID SEARCH
# ============================================================
# Test eps values around the elbow with two min_samples values.
# For each combination record:
#   - number of clusters found
#   - noise percentage
#   - silhouette score on core points
# Pick best: acceptable noise + cluster count, then highest silhouette.
# ============================================================

print(f"\n[3/5] Parameter grid search...")
print("-" * 70)

eps_low        = eps_elbow * 0.4
eps_high       = eps_elbow * 3.0
EPS_CANDIDATES = np.linspace(eps_low, eps_high, N_EPS_CANDIDATES)

# Test rule-of-thumb and a looser alternative (half)
MIN_S_CANDIDATES = [max(5, MIN_SAMPLES_FIXED // 2), MIN_SAMPLES_FIXED]

print(f"  eps range     : [{eps_low:.4f}, {eps_high:.4f}]  ({N_EPS_CANDIDATES} values)")
print(f"  min_samples   : {MIN_S_CANDIDATES}")
print(f"  Total runs    : {N_EPS_CANDIDATES * len(MIN_S_CANDIDATES)}\n")

print(f"  {'eps':>8}  {'min_s':>6}  {'clusters':>9}  {'noise%':>7}  "
      f"{'silhouette':>11}  {'verdict':>8}")
print("  " + "-" * 60)

search_records = []

for min_s in MIN_S_CANDIDATES:
    for eps_val in EPS_CANDIDATES:

        db     = DBSCAN(eps=eps_val, min_samples=min_s, n_jobs=-1)
        lbl    = db.fit_predict(X)
        n_cl   = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_ns   = int((lbl == -1).sum())
        ns_pct = n_ns / N * 100
        core_m = lbl >= 0

        sil = np.nan
        if n_cl >= 2 and core_m.sum() > 100:
            try:
                sil = silhouette_score(
                    X[core_m], lbl[core_m],
                    sample_size=min(5000, core_m.sum()),
                    random_state=RANDOM_STATE
                )
            except Exception:
                pass

        ok      = (NOISE_MIN_PCT <= ns_pct <= NOISE_MAX_PCT) and (2 <= n_cl <= 10)
        verdict = "OK" if ok else "--"

        search_records.append({
            "eps": eps_val, "min_samples": min_s,
            "n_clusters": n_cl, "noise_pct": ns_pct,
            "silhouette": sil, "verdict": verdict
        })

        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "   N/A"
        print(f"  {eps_val:>8.4f}  {min_s:>6}  {n_cl:>9}  "
              f"{ns_pct:>6.1f}%  {sil_str:>11}  {verdict:>8}")

search_df = pd.DataFrame(search_records)
search_df.to_csv(RESULTS_DIR / "dbscan_param_search.csv", index=False)
print(f"\n✓ Saved: results/dbscan_param_search.csv")

# ── Decision logic ────────────────────────────────────────
ok_rows = search_df[search_df["verdict"] == "OK"]

if len(ok_rows) > 0:
    ok_with_sil = ok_rows.dropna(subset=["silhouette"])
    if len(ok_with_sil) > 0:
        best_row        = ok_with_sil.loc[ok_with_sil["silhouette"].idxmax()]
        decision_reason = "highest silhouette among acceptable combinations"
    else:
        best_row        = ok_rows.loc[(ok_rows["noise_pct"] - 20).abs().idxmin()]
        decision_reason = "noise % closest to 20% (silhouette unavailable)"
else:
    best_row        = search_df.loc[(search_df["noise_pct"] - 20).abs().idxmin()]
    decision_reason = "no combination met criteria — fallback to noise % closest to 20%"

chosen_eps   = float(best_row["eps"])
chosen_min_s = int(best_row["min_samples"])

print(f"\n  {'─'*60}")
print(f"  Chosen eps         : {chosen_eps:.4f}")
print(f"  Chosen min_samples : {chosen_min_s}")
print(f"  Decision reason    : {decision_reason}")
print(f"  {'─'*60}")

# ── Parameter search plot ─────────────────────────────────
fig_ps, axes_ps = plt.subplots(1, 3, figsize=(16, 5))
fig_ps.suptitle("DBSCAN Parameter Grid Search", fontsize=13, fontweight="bold")

for min_s, color, marker in zip(MIN_S_CANDIDATES,
                                 ["steelblue", "darkorange"], ["o", "s"]):
    sub = search_df[search_df["min_samples"] == min_s]
    axes_ps[0].plot(sub["eps"], sub["n_clusters"], f"{marker}-",
                    color=color, linewidth=1.8, markersize=6,
                    label=f"min_s={min_s}")
    axes_ps[1].plot(sub["eps"], sub["noise_pct"], f"{marker}-",
                    color=color, linewidth=1.8, markersize=6,
                    label=f"min_s={min_s}")
    valid = sub.dropna(subset=["silhouette"])
    if len(valid):
        axes_ps[2].plot(valid["eps"], valid["silhouette"], f"{marker}-",
                        color=color, linewidth=1.8, markersize=6,
                        label=f"min_s={min_s}")

for ax in axes_ps:
    ax.axvline(x=chosen_eps, color="red", linestyle="--", linewidth=1.5,
               label=f"Chosen eps={chosen_eps:.3f}")
    ax.set_xlabel("eps", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)

axes_ps[0].set_title("Clusters found\n(target: 2–10)",
                     fontsize=11, fontweight="bold")
axes_ps[0].set_ylabel("Number of clusters", fontsize=11, fontweight="bold")
axes_ps[0].axhline(y=2,  color="green", linestyle=":", alpha=0.5)
axes_ps[0].axhline(y=10, color="green", linestyle=":", alpha=0.5)

axes_ps[1].set_title("Noise %\n(target: 5–40%)", fontsize=11, fontweight="bold")
axes_ps[1].set_ylabel("Noise %", fontsize=11, fontweight="bold")
axes_ps[1].axhline(y=NOISE_MIN_PCT, color="green", linestyle=":", alpha=0.5)
axes_ps[1].axhline(y=NOISE_MAX_PCT, color="green", linestyle=":", alpha=0.5)

axes_ps[2].set_title("Silhouette score\n(higher is better)",
                     fontsize=11, fontweight="bold")
axes_ps[2].set_ylabel("Silhouette", fontsize=11, fontweight="bold")

plt.tight_layout()
fig_ps.savefig(RESULTS_DIR / "dbscan_param_search.png",
               dpi=300, bbox_inches="tight")
print(f"✓ Saved: results/dbscan_param_search.png")
plt.show()


# ============================================================
# STEP 4: FINAL DBSCAN RUN
# ============================================================

print(f"\n[4/5] Final DBSCAN  "
      f"(eps={chosen_eps:.4f}, min_samples={chosen_min_s})...")
print("-" * 70)

t0     = time.time()
db     = DBSCAN(eps=chosen_eps, min_samples=chosen_min_s, n_jobs=-1)
labels = db.fit_predict(X)
elapsed_db = time.time() - t0

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = int((labels == -1).sum())
noise_pct  = n_noise / N * 100
core_mask  = labels >= 0

print(f"✓ DBSCAN done         : {elapsed_db:.1f} s")
print(f"  Clusters found      : {n_clusters}")
print(f"  Noise points        : {n_noise:,}  ({noise_pct:.1f}%)")
print(f"  Core points         : {core_mask.sum():,}  ({core_mask.sum()/N*100:.1f}%)")

unique_c, counts_c = np.unique(labels[core_mask], return_counts=True)
print(f"\n  Cluster distribution:")
for cid, cnt in zip(unique_c, counts_c):
    print(f"    Cluster {cid}: {cnt:7,}  ({cnt/N*100:.1f}% of total, "
          f"{cnt/core_mask.sum()*100:.1f}% of core)")

silhouette = np.nan
if n_clusters >= 2 and core_mask.sum() > 100:
    try:
        silhouette = silhouette_score(
            X[core_mask], labels[core_mask],
            sample_size=min(5000, core_mask.sum()),
            random_state=RANDOM_STATE
        )
        print(f"\n  Silhouette (core)   : {silhouette:.4f}")
    except Exception as e:
        print(f"\n  Silhouette          : could not compute ({e})")

# Save
np.save(RESULTS_DIR / "dbscan_labels.npy", labels)
print(f"\n✓ Saved: results/dbscan_labels.npy")

results_df            = metadata.copy()
results_df["cluster"] = labels
results_df.to_csv(RESULTS_DIR / "dbscan_results.csv", index=False)
print(f"✓ Saved: results/dbscan_results.csv  ({len(results_df):,} rows)")


# ============================================================
# STEP 5: VISUALISATIONS + REPORT
# ============================================================

print(f"\n[5/5] Visualisations and report...")
print("-" * 70)

COLORS = plt.cm.Set3(np.linspace(0, 1, max(n_clusters, 1) + 1))

# ── Bar + pie ─────────────────────────────────────────────
fig_cl, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 5))
fig_cl.suptitle(
    f"DBSCAN  (eps={chosen_eps:.4f}, min_samples={chosen_min_s})",
    fontsize=13, fontweight="bold"
)

bar_labels = [f"Cluster {c}" for c in unique_c] + ["Noise"]
bar_values = list(counts_c) + [n_noise]
bar_colors = [COLORS[i] for i in range(len(unique_c))] + ["#cccccc"]

bars = ax_bar.bar(bar_labels, bar_values, color=bar_colors,
                  edgecolor="black", linewidth=1.2)
ax_bar.set_ylabel("Number of segments", fontsize=11, fontweight="bold")
ax_bar.set_title("Segment count per cluster", fontsize=12, fontweight="bold")
ax_bar.grid(True, axis="y", alpha=0.3)
for bar in bars:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h,
                f"{int(h):,}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

ax_pie.pie(bar_values,
           labels=[f"{l}\n({v:,})" for l, v in zip(bar_labels, bar_values)],
           colors=bar_colors, autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 9})
ax_pie.set_title("Percentage distribution", fontsize=12, fontweight="bold")

plt.tight_layout()
fig_cl.savefig(RESULTS_DIR / "dbscan_clusters.png",
               dpi=300, bbox_inches="tight")
print(f"✓ Saved: results/dbscan_clusters.png")
plt.show()

# ── Cluster center patterns ───────────────────────────────
if n_clusters >= 1:
    fig_cp, ax_cp = plt.subplots(figsize=(12, 5))
    for i, (cid, cnt) in enumerate(zip(unique_c, counts_c)):
        center = X[labels == cid].mean(axis=0)
        ax_cp.plot(center, linewidth=2, markersize=3, marker="o",
                   color=COLORS[i],
                   label=f"Cluster {cid}  (n={cnt:,})")
    ax_cp.set_xlabel("Day in segment (1–50)", fontsize=12, fontweight="bold")
    ax_cp.set_ylabel("Normalised price", fontsize=12, fontweight="bold")
    ax_cp.set_title("Cluster centers — typical price pattern per cluster",
                    fontsize=13, fontweight="bold")
    ax_cp.legend(fontsize=10)
    ax_cp.grid(True, alpha=0.3, linestyle="--")
    ax_cp.set_xlim([0, D - 1])
    plt.tight_layout()
    fig_cp.savefig(RESULTS_DIR / "dbscan_centers.png",
                   dpi=300, bbox_inches="tight")
    print(f"✓ Saved: results/dbscan_centers.png")
    plt.show()

# ── Text report ───────────────────────────────────────────
sil_str = f"{silhouette:.4f}" if not np.isnan(silhouette) else "N/A"
sil_quality = (
    "Excellent (> 0.75)" if not np.isnan(silhouette) and silhouette > 0.75 else
    "Good (0.50–0.75)"   if not np.isnan(silhouette) and silhouette > 0.50 else
    "Fair (0.25–0.50)"   if not np.isnan(silhouette) and silhouette > 0.25 else
    "Poor (< 0.25)"      if not np.isnan(silhouette) else "N/A"
)

report_lines = [
    "=" * 70,
    "DBSCAN CLUSTERING — RESULTS REPORT",
    "=" * 70,
    f"Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "METHODOLOGY",
    "─" * 70,
    "  DBSCAN run directly on the 50-dimensional normalised segments.",
    "  No PCA applied — segments are Min-Max normalised per-segment,",
    "  preserving shape information. High autocorrelation between",
    "  adjacent days means Euclidean distances remain meaningful in",
    "  50D and the k-distance elbow is identifiable without reduction.",
    "",
    "CONFIGURATION",
    "─" * 70,
    f"  Sample size    : {N:,}",
    f"  Dimensions     : {D}",
    f"  min_samples    : {chosen_min_s}  (rule-of-thumb: 2 × {D} = {MIN_SAMPLES_FIXED})",
    f"  eps            : {chosen_eps:.4f}  (k-distance elbow + grid search)",
    f"  Decision basis : {decision_reason}",
    f"  Runtime        : {elapsed_db:.1f} s",
    "",
    "PARAMETER SELECTION",
    "─" * 70,
    f"  k-distance elbow eps   : {eps_elbow:.4f}",
    f"  Grid search range      : [{eps_low:.4f}, {eps_high:.4f}]",
    f"  Chosen eps             : {chosen_eps:.4f}",
    f"  Chosen min_samples     : {chosen_min_s}",
    "",
    "RESULTS",
    "─" * 70,
    f"  Clusters found         : {n_clusters}",
    f"  Noise points           : {n_noise:,}  ({noise_pct:.1f}%)",
    f"  Core points            : {core_mask.sum():,}  ({core_mask.sum()/N*100:.1f}%)",
    f"  Silhouette (core)      : {sil_str}  →  {sil_quality}",
    "",
    "CLUSTER DISTRIBUTION",
    "─" * 70,
    f"  {'Cluster':<10} {'Segments':>10} {'% of total':>12} {'% of core':>12}",
    "  " + "-" * 46,
]

for cid, cnt in zip(unique_c, counts_c):
    report_lines.append(
        f"  {cid:<10} {cnt:>10,} {cnt/N*100:>11.1f}%"
        f" {cnt/core_mask.sum()*100:>11.1f}%"
    )
report_lines += [
    f"  {'Noise':<10} {n_noise:>10,} {noise_pct:>11.1f}%            —",
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
(RESULTS_DIR / "dbscan_report.txt").write_text(report_text)
print(report_text)
print(f"✓ Saved: results/dbscan_report.txt")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✓ DBSCAN CLUSTERING COMPLETE")
print("=" * 70)
print(f"""
SUMMARY:
────────
eps            : {chosen_eps:.4f}
min_samples    : {chosen_min_s}
Clusters found : {n_clusters}
Noise          : {n_noise:,}  ({noise_pct:.1f}%)
Silhouette     : {sil_str}

FILES SAVED (results/):
───────────────────────
✓ dbscan_labels.npy
✓ dbscan_results.csv
✓ dbscan_kdistance.png
✓ dbscan_param_search.png
✓ dbscan_param_search.csv
✓ dbscan_clusters.png
✓ dbscan_centers.png
✓ dbscan_report.txt

STATUS: ✓ READY FOR ANALYSIS
""")
print("=" * 70 + "\n")