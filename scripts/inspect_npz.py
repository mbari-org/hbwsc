"""Print a per-cluster summary of a results.npz file produced by hbws-cluster."""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results.npz"
r = np.load(path)
labels, reduced = r["labels"], r["reduced"]

unique, counts = np.unique(labels, return_counts=True)
total = len(labels)

print(f"\n{path}  —  {total} windows total\n")
print(f"  {'label':<10} {'count':>6}  {'%':>6}  xy centroid")
print(f"  {'-' * 45}")
for label, count in zip(unique, counts):
    mask = labels == label
    cx, cy = reduced[mask].mean(axis=0)
    tag = "noise" if label == -1 else f"cluster {label}"
    print(f"  {tag:<10} {count:>6}  {100 * count / total:>5.1f}%  ({cx:+.3f}, {cy:+.3f})")
print()
