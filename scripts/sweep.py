"""Sweep UMAP dimensionality and HDBSCAN min_cluster_size, reporting clustering metrics.

Loads AVES embeddings from a cache file (no model inference) and evaluates every
combination of UMAP cluster dimensions and min_cluster_size.

UMAP fits run serially (one dims value at a time) to cap memory usage, while
HDBSCAN parameter combos run in parallel for each UMAP reduction (CPU only;
GPU runs serially since cuML HDBSCAN is already fast).

Usage:
    uv run python scripts/sweep.py <embeddings.npy> [options]

Options:
    --dims     Comma-separated UMAP cluster dimensions to try  (default: 2,5,10,15,20,30)
    --mcs      Comma-separated min_cluster_size values to try  (default: 50,100,200)
    --neighbors  UMAP n_neighbors                              (default: 15)
    --out      Path to save results table as CSV               (optional)
    --workers  Number of parallel HDBSCAN workers              (default: 2, ignored on GPU)

Metrics reported per configuration:
    n_clusters   Number of clusters found (excluding noise label -1)
    noise_pct    Percentage of windows labelled noise
    dbcv         HDBSCAN relative validity index (DBCV); higher is better, range [-1, 1]
    median_sz    Median cluster size
    min_sz       Smallest cluster
    max_sz       Largest cluster

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
"""

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hbws_clustering.evaluation import load_raven_labels, map_labels_to_windows, compute_metrics

# ---------------------------------------------------------------------------
# Backend detection: prefer GPU (cuML) when available
# ---------------------------------------------------------------------------
# DOING THIS ONLY FOR UMAP NOW BECAUSE HDBSCAN ON cuML DOESNT HAVE DBCV
try:
    from cuml.manifold import UMAP as _UMAP
    # from cuml.cluster import HDBSCAN as _HDBSCAN
    import rmm
    rmm.reinitialize(pool_allocator=True)
    _BACKEND = "gpu"
except ImportError:
    import umap as _umap_lib
    _UMAP = _umap_lib.UMAP
    _BACKEND = "cpu"
import hdbscan as _hdbscan_lib
_HDBSCAN = _hdbscan_lib.HDBSCAN


# Removed global init_worker logic because we pass shm info directly to the worker now.

# ---------------------------------------------------------------------------
# HDBSCAN runner
# ---------------------------------------------------------------------------

def _run_single_hdbscan(reduced: np.ndarray, umap_dims: int, mcs: int, eps: float,
                         alpha: float, t_umap: float, manual_window=None) -> dict:
    """Run one HDBSCAN config on the given reduced array."""
    t0 = time.perf_counter()

    hdbscan_kwargs = dict(
        min_cluster_size=mcs,
        min_samples=mcs,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=eps,
        alpha=alpha,
        gen_min_span_tree=True,
    )
    # Stop HDBSCAN from spawning its own nested joblib workers inside our ProcessPoolExecutor
    if getattr(_HDBSCAN, "__module__", "").startswith("hdbscan"):
        hdbscan_kwargs["core_dist_n_jobs"] = 1

    clusterer = _HDBSCAN(**hdbscan_kwargs)
    clusterer.fit(reduced)
    t_hdbscan = time.perf_counter() - t0

    labels = clusterer.labels_
    # cuML may return cupy/cudf — ensure numpy
    if hasattr(labels, "get"):
        labels = labels.get()
    labels = np.asarray(labels)

    n = len(labels)
    unique = [lbl for lbl in np.unique(labels) if lbl >= 0]
    n_clusters = len(unique)
    noise_pct = 100.0 * (labels == -1).sum() / n

    if n_clusters > 0:
        sizes = [int((labels == lbl).sum()) for lbl in unique]
        median_sz = int(np.median(sizes))
        min_sz = min(sizes)
        max_sz = max(sizes)
        dbcv = float(getattr(clusterer, "relative_validity_", float("nan")))
    else:
        median_sz = min_sz = max_sz = 0
        dbcv = float("nan")

    # Similarity metrics
    t1 = time.perf_counter()
    if manual_window is not None:
        metrics = compute_metrics(labels, manual_window)
        jaccard = metrics["DetSim"]
        nmi = metrics["NMI"]
        ari = metrics["ARI"]
        v_hom = metrics["Homogeneity"]
    else:
        jaccard = float("nan")
        nmi = float("nan")
        ari = float("nan")
        v_hom = float("nan")
    t_metrics = time.perf_counter() - t1

    ret = {
        "umap_dims": umap_dims,
        "mcs": mcs,
        "eps": eps,
        "n_clusters": n_clusters,
        "noise_pct": round(noise_pct, 1),
        "dbcv": round(dbcv, 4),
        "median_sz": median_sz,
        "min_sz": min_sz,
        "max_sz": max_sz,
        "t_umap_s": round(t_umap, 1),
        "t_hdbscan_s": round(t_hdbscan, 1),
        "t_metrics_s": round(t_metrics, 2),
    }

    if manual_window is not None:
        ret["detsim"] = round(jaccard, 4)
        ret["nmi"] = round(nmi, 4)
        ret["ari"] = round(ari, 4)
        ret["homog"] = round(v_hom, 4)

    return ret


def _run_hdbscan_from_shm(shm_name: str, shape: tuple, dtype_str: str,
                           umap_dims: int, mcs: int, eps: float, alpha: float,
                           t_umap: float, manual_window=None) -> dict:
    """CPU multiprocessing wrapper: reads reduced data from shared memory."""
    shm = SharedMemory(name=shm_name)
    reduced = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    result = _run_single_hdbscan(reduced, umap_dims, mcs, eps, alpha, t_umap, manual_window)
    shm.close()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("embeddings", type=Path, help="Path to a cached embeddings .npy file")
    parser.add_argument("--dims", default="2,5,10,15,20,30", help="Comma-separated UMAP dimensions")
    parser.add_argument("--mcs", default="50,100,200", help="Comma-separated min_cluster_size values")
    parser.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--out", type=Path, default=None, help="Save results as CSV")
    parser.add_argument("--epsilon", type=float, default=0.0, help="HDBSCAN cluster_selection_epsilon (single-run default; sweep uses --epsilons)")
    parser.add_argument("--alpha", type=float, default=1.0, help="HDBSCAN alpha")
    parser.add_argument("--manual-labels", type=Path, default=None, help="Path to manual labels (Raven format) for supervised metrics")
    parser.add_argument("--npz", type=Path, default=None, help="Path to results.npz to read start_secs from (required if --manual-labels is used)")
    parser.add_argument("--window-sec", type=float, default=None, help="Analysis window size in seconds")
    parser.add_argument(
        "--workers", type=int, default=2, help="Parallel HDBSCAN workers (CPU only; ignored on GPU; default: 2)"
    )
    parser.add_argument("--epsilons", type=str, default="0.0", help="Comma-separated HDBSCAN epsilons")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    print(f"Embeddings: {embeddings.shape}  ({args.embeddings})")
    print(f"Backend:    {_BACKEND}")

    dims_list = [int(x) for x in args.dims.split(",")]
    mcs_list = [int(x) for x in args.mcs.split(",")]
    eps_list = [float(x) for x in args.epsilons.split(",")]
    n_neighbors = args.neighbors
    n_hdbscan_combos = len(mcs_list) * len(eps_list)
    n_total = len(dims_list) * n_hdbscan_combos

    n_workers = min(args.workers, n_hdbscan_combos)

    print(f"UMAP dims:         {dims_list}")
    print(f"min_cluster_size:  {mcs_list}")
    print(f"epsilons:          {eps_list}")
    print(f"UMAP n_neighbors:  {n_neighbors}")
    print(f"Total configs:     {n_total}  ({len(dims_list)} UMAP fits × {n_hdbscan_combos} HDBSCAN combos)")
    print(f"Workers:           {n_workers}  (HDBSCAN only; UMAP runs serially)\n")

    manual_window = None
    if args.manual_labels and args.npz:
        print(f"Loading manual labels from {args.manual_labels}")
        r = np.load(args.npz, allow_pickle=False)
        start_secs = r["start_secs"]
        hop_sec = float(np.median(np.diff(start_secs)))
        window_sec = args.window_sec if args.window_sec else 2 * hop_sec
        end_secs = start_secs + window_sec

        manual = load_raven_labels(args.manual_labels)
        manual_window, _ = map_labels_to_windows(manual, start_secs, end_secs)

    header = [
        "umap_dims",
        "mcs",
        "eps",
        "n_clusters",
        "noise_pct",
        "dbcv",
        "median_sz",
        "min_sz",
        "max_sz",
        "t_umap_s",
        "t_hdbscan_s",
        "t_metrics_s",
    ]
    col_w = [9, 5, 5, 10, 10, 8, 10, 7, 7, 10, 12, 12]

    if manual_window is not None:
        header.extend(["detsim", "nmi", "ari", "homog"])
        col_w.extend([8, 8, 8, 8])

    def fmt_row(row: dict) -> str:
        vals = [str(row[k]) for k in header]
        return "  ".join(v.rjust(w) for v, w in zip(vals, col_w))

    print("  ".join(h.rjust(w) for h, w in zip(header, col_w)))
    print("  ".join("-" * w for w in col_w))

    rows = []
    
    t_sweep_start = time.perf_counter()

    # Reuse the same ProcessPoolExecutor for the entire script to avoid joblib hangs
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for umap_dims in dims_list:
            # Serial UMAP fit
            print(
                f"\n  [UMAP] Fitting {umap_dims}-D reduction on "
                f"{embeddings.shape[0]} embeddings ...",
                end="", flush=True,
            )
            t0 = time.perf_counter()
            reducer = _UMAP(
                n_components=umap_dims,
                n_neighbors=n_neighbors,
                metric="cosine",
                random_state=42,
            )
            reduced = reducer.fit_transform(embeddings)
            # cuML may return cupy array
            if hasattr(reduced, "get"):
                reduced = reduced.get()
            reduced = np.ascontiguousarray(reduced, dtype=np.float32)
            t_umap = time.perf_counter() - t0
            print(f" done ({t_umap:.1f}s)")

            # Parallel HDBSCAN via shared memory
            shm = SharedMemory(create=True, size=reduced.nbytes)
            shm_arr = np.ndarray(reduced.shape, dtype=reduced.dtype, buffer=shm.buf)
            shm_arr[:] = reduced
            red_shape = reduced.shape
            red_dtype = reduced.dtype
            del shm_arr

            try:
                futures = {
                    executor.submit(
                        _run_hdbscan_from_shm, shm.name, red_shape, red_dtype.str,
                        umap_dims, mcs, eps, args.alpha, t_umap, manual_window,
                    ): (umap_dims, mcs, eps)
                    for mcs, eps in product(mcs_list, eps_list)
                }
                for f in as_completed(futures):
                    row = f.result()
                    rows.append(row)
                    print(fmt_row(row), flush=True)
            finally:
                shm.close()
                shm.unlink()

    rows.sort(key=lambda r: (r["umap_dims"], r["mcs"]))

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {args.out}")

    t_sweep_total = time.perf_counter() - t_sweep_start
    print(f"\nTotal sweep time: {t_sweep_total:.1f}s")


if __name__ == "__main__":
    main()
