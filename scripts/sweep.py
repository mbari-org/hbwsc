"""Sweep UMAP dimensionality and HDBSCAN min_cluster_size, reporting clustering metrics.

Loads AVES embeddings from a cache file (no model inference) and evaluates every
combination of UMAP cluster dimensions and min_cluster_size.

Usage:
    uv run python scripts/sweep.py <embeddings.npy> [options]

Options:
    --dims     Comma-separated UMAP cluster dimensions to try  (default: 2,5,10,15,20,30)
    --mcs      Comma-separated min_cluster_size values to try  (default: 50,100,200)
    --neighbors  UMAP n_neighbors                              (default: 15)
    --out      Path to save results table as CSV               (optional)

Metrics reported per configuration:
    n_clusters   Number of clusters found (excluding noise label -1)
    noise_pct    Percentage of windows labelled noise
    dbcv         HDBSCAN relative validity index (DBCV); higher is better, range [-1, 1]
    median_sz    Median cluster size
    min_sz       Smallest cluster

Example (4.5-hour song session, 36127 windows at 0.5 s / 0.25 s hop):
    uv run python scripts/sweep.py \\
        output/embeddings_song_w0.5_h0.25.npy \\
        --dims 2,5,10,15,20,30 \\
        --mcs 50,100,200 \\
        --out output/sweep_results.csv

Results from that run:
    umap_dims  mcs  n_clusters  noise_pct     dbcv  median_sz
            2   50          42       30.6  -0.0052        241
            2  100          24       19.4  -0.2683        337
            2  200          14       28.6  -0.0640        907
            5   50          26       22.6  -0.0679        189
            5  100          10       14.5   0.0328        895
            5  200           7       18.5   0.0715       2184
           10   50          42       38.0  -0.0048        167
           10  100           9       14.2   0.2017       1225
           10  200           6       13.1   0.3075       4296
           15   50          32       29.6   0.0861        189
           15  100          10       14.4  -0.1037       1584
           15  200           8       18.1  -0.0270       2614
           20   50          34       33.1  -0.0378        202
           20  100           9       11.9   0.1630       1320
           20  200           3        1.4  -0.6831       9735
           30   50          29       24.1  -0.0078        215
           30  100          10       11.9   0.2541        973
           30  200           2        0.0  -0.8820      18063

    Interpretation: mcs=50 is consistently over-fragmented (negative DBCV).
    mcs=200 at 20-30D collapses to 2-3 mega-clusters.
    10-D mcs=100 (DBCV=0.20, 9 clusters) and 10-D mcs=200 (DBCV=0.31, 6 clusters)
    are the best-supported configurations for this session.
    max_sz       Largest cluster
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import hdbscan
import hdbscan.validity
import numpy as np
import umap as umap_lib


def run_one(embeddings: np.ndarray, umap_dims: int, mcs: int, n_neighbors: int) -> dict:
    t0 = time.perf_counter()

    reducer = umap_lib.UMAP(
        n_components=umap_dims,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    t_umap = time.perf_counter() - t0

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=mcs,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    clusterer.fit(reduced)
    t_hdbscan = time.perf_counter() - t0 - t_umap

    labels = clusterer.labels_
    n = len(labels)
    unique = [l for l in np.unique(labels) if l >= 0]
    n_clusters = len(unique)
    noise_pct = 100.0 * (labels == -1).sum() / n

    if n_clusters > 0:
        sizes = [int((labels == l).sum()) for l in unique]
        median_sz = int(np.median(sizes))
        min_sz = min(sizes)
        max_sz = max(sizes)
        dbcv = float(hdbscan.validity.validity_index(reduced.astype(np.float64), labels))
    else:
        median_sz = min_sz = max_sz = 0
        dbcv = float("nan")

    return {
        "umap_dims": umap_dims,
        "mcs": mcs,
        "n_clusters": n_clusters,
        "noise_pct": round(noise_pct, 1),
        "dbcv": round(dbcv, 4),
        "median_sz": median_sz,
        "min_sz": min_sz,
        "max_sz": max_sz,
        "t_umap_s": round(t_umap, 1),
        "t_hdbscan_s": round(t_hdbscan, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("embeddings", type=Path, help="Path to a cached embeddings .npy file")
    parser.add_argument("--dims", default="2,5,10,15,20,30", help="Comma-separated UMAP dimensions")
    parser.add_argument("--mcs", default="50,100,200", help="Comma-separated min_cluster_size values")
    parser.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--out", type=Path, default=None, help="Save results as CSV")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    print(f"Embeddings: {embeddings.shape}  ({args.embeddings})")

    dims_list = [int(x) for x in args.dims.split(",")]
    mcs_list = [int(x) for x in args.mcs.split(",")]
    n_neighbors = args.neighbors

    print(f"UMAP dims:         {dims_list}")
    print(f"min_cluster_size:  {mcs_list}")
    print(f"UMAP n_neighbors:  {n_neighbors}")
    print(f"Total configs:     {len(dims_list) * len(mcs_list)}\n")

    header = ["umap_dims", "mcs", "n_clusters", "noise_pct", "dbcv", "median_sz", "min_sz", "max_sz", "t_umap_s", "t_hdbscan_s"]
    col_w =  [9,           5,     10,            10,          8,      10,          7,        7,        10,          12]

    def fmt_row(row: dict) -> str:
        vals = [str(row[k]) for k in header]
        return "  ".join(v.rjust(w) for v, w in zip(vals, col_w))

    print("  ".join(h.rjust(w) for h, w in zip(header, col_w)))
    print("  ".join("-" * w for w in col_w))

    rows = []
    for dims in dims_list:
        for mcs in mcs_list:
            row = run_one(embeddings, dims, mcs, n_neighbors)
            rows.append(row)
            print(fmt_row(row), flush=True)

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
