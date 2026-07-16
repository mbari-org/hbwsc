"""Compare HDBSCAN cluster assignments against manual Raven labels.

Maps each analysis window onto the manual label grid, then computes:

  1. Detection similarity (Jaccard): overlap between clustered windows and
     manually labelled windows, regardless of which cluster/label.

  2. Contingency table: for windows in the intersection, how many windows
     of each cluster fall into each manual label type.

  3. Per-cluster summary: dominant manual label and purity (fraction of
     cluster windows that carry that label).

  4. Per-label summary: dominant cluster and recall (fraction of label
     windows covered by that cluster).

  5. NMI and V-measure (sklearn) over the intersection windows.

Usage:
    uv run python scripts/compare_labels.py <npz> <manual_labels> [--window-sec N]

Arguments:
    npz             Path to results .npz produced by hbws-cluster.
    manual_labels   Raven selection table (tab-separated) with manual labels.
    --window-sec    Analysis window size in seconds (default: inferred as
                    2 × median hop from start_secs).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, v_measure_score, adjusted_rand_score

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("npz", type=Path)
parser.add_argument("manual_labels", type=Path)
parser.add_argument("--window-sec", type=float, default=None)
parser.add_argument("--out", type=Path, default=None, help="Save output to this file in addition to stdout")
args = parser.parse_args()

import sys

class Tee:
    """Write to both stdout and a file."""
    def __init__(self, path):
        self._file = open(path, "w")
        self._stdout = sys.stdout
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()

tee = None
if args.out:
    tee = Tee(args.out)
    sys.stdout = tee

# ---------------------------------------------------------------------------
# Load npz
# ---------------------------------------------------------------------------

r = np.load(args.npz, allow_pickle=False)
labels = r["labels"]          # int array, -1 = noise
start_secs = r["start_secs"]  # float array

hop_sec = float(np.median(np.diff(start_secs)))
window_sec = args.window_sec if args.window_sec else 2 * hop_sec
end_secs = start_secs + window_sec
n_windows = len(labels)

print(f"NPZ:        {args.npz}")
print(f"Windows:    {n_windows}  (hop={hop_sec:.3f}s  window={window_sec:.3f}s)")

# ---------------------------------------------------------------------------
# Load manual labels
# ---------------------------------------------------------------------------

# Each entry: (begin_sec, end_sec, type_str)
manual = []
with open(args.manual_labels) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        manual.append((float(row["Begin Time (s)"]), float(row["End Time (s)"]), row["Type"].strip()))

unique_types = sorted(set(t for _, _, t in manual))
type_to_idx = {t: i for i, t in enumerate(unique_types)}
print(f"Manual:     {len(manual)} selections, {len(unique_types)} types: {unique_types}\n")

# ---------------------------------------------------------------------------
# Map manual labels onto windows
# (assign the type with greatest overlap; None if no overlap)
# ---------------------------------------------------------------------------

manual_window = np.full(n_windows, -1, dtype=int)  # -1 = no manual label

for i in range(n_windows):
    ws, we = start_secs[i], end_secs[i]
    best_overlap = 0.0
    best_type = -1
    for begin, end, ltype in manual:
        overlap = max(0.0, min(we, end) - max(ws, begin))
        if overlap > best_overlap:
            best_overlap = overlap
            best_type = type_to_idx[ltype]
    manual_window[i] = best_type

# ---------------------------------------------------------------------------
# Detection similarity (Jaccard)
# ---------------------------------------------------------------------------

is_clustered = labels >= 0          # not noise
is_labelled  = manual_window >= 0   # has a manual label

intersection = (is_clustered & is_labelled).sum()
union        = (is_clustered | is_labelled).sum()
jaccard      = intersection / union if union > 0 else 0.0

print("=== Detection similarity ===")
print(f"  Clustered windows (non-noise):  {is_clustered.sum():6d} / {n_windows}")
print(f"  Manually labelled windows:      {is_labelled.sum():6d} / {n_windows}")
print(f"  Intersection (both):            {intersection:6d}")
print(f"  Union (either):                 {union:6d}")
print(f"  Jaccard similarity:             {jaccard:.4f}\n")

# ---------------------------------------------------------------------------
# Contingency table (intersection only)
# ---------------------------------------------------------------------------

in_both = is_clustered & is_labelled
c_in  = labels[in_both]
m_in  = manual_window[in_both]

unique_clusters = sorted(np.unique(c_in).tolist())
n_c = len(unique_clusters)
n_m = len(unique_types)
cluster_to_row = {c: i for i, c in enumerate(unique_clusters)}

contingency = np.zeros((n_c, n_m), dtype=int)
for c, m in zip(c_in, m_in):
    contingency[cluster_to_row[c], m] += 1

print("=== Contingency table (intersection windows) ===")
col_w = 8
header = " " * 12 + "".join(t.rjust(col_w) for t in unique_types) + "   total"
print(header)
print("-" * len(header))
for ci, c in enumerate(unique_clusters):
    row = contingency[ci]
    print(f"  cluster {c:>3d}  " + "".join(str(v).rjust(col_w) for v in row) + f"  {row.sum():6d}")
print("-" * len(header))
col_totals = contingency.sum(axis=0)
print("  total      " + "".join(str(v).rjust(col_w) for v in col_totals))
print()

# ---------------------------------------------------------------------------
# Per-cluster summary
# ---------------------------------------------------------------------------

print("=== Per-cluster dominant label & purity ===")
print(f"  {'cluster':>8}  {'dominant_label':>15}  {'purity':>7}  {'n_windows':>10}")
print("  " + "-" * 48)
for ci, c in enumerate(unique_clusters):
    row = contingency[ci]
    dom_idx = row.argmax()
    purity = row[dom_idx] / row.sum() if row.sum() > 0 else 0.0
    print(f"  {c:>8}  {unique_types[dom_idx]:>15}  {purity:>7.3f}  {row.sum():>10}")
print()

# ---------------------------------------------------------------------------
# Per-label summary
# ---------------------------------------------------------------------------

print("=== Per-label dominant cluster & recall ===")
print(f"  {'label':>8}  {'dominant_cluster':>16}  {'recall':>7}  {'n_windows':>10}")
print("  " + "-" * 48)
for mi, t in enumerate(unique_types):
    col = contingency[:, mi]
    dom_idx = col.argmax()
    recall = col[dom_idx] / col.sum() if col.sum() > 0 else 0.0
    print(f"  {t:>8}  {unique_clusters[dom_idx]:>16}  {recall:>7.3f}  {col.sum():>10}")
print()

# ---------------------------------------------------------------------------
# NMI, ARI, and V-measure
# ---------------------------------------------------------------------------

from sklearn.metrics import homogeneity_completeness_v_measure
nmi = normalized_mutual_info_score(m_in, c_in)
ari = adjusted_rand_score(m_in, c_in)
v_hom, v_comp, v_meas = homogeneity_completeness_v_measure(m_in, c_in)

print("=== Information-theoretic metrics (intersection windows) ===")
print(f"  NMI (normalized mutual info):  {nmi:.4f}")
print(f"  ARI (adjusted rand index):     {ari:.4f}")
print(f"  Homogeneity  (cluster -> label): {v_hom:.4f}")
print(f"  Completeness (label -> cluster): {v_comp:.4f}")
print(f"  V-measure:                       {v_meas:.4f}")

if tee:
    sys.stdout = tee._stdout
    tee.close()
    print(f"\nOutput saved to {args.out}")
