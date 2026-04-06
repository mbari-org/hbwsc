"""Plot the temporal sequence of cluster assignments.

Each window is shown as a dot at its start time, coloured by cluster label.
Noise (-1) is shown in light grey. Cluster rows are separated on the y-axis
so co-occurring or rapidly alternating clusters are easy to see.

Usage:
    uv run python scripts/plot_timeline.py <npz> [out_png]

Arguments:
    npz      Path to a results .npz file produced by hbws-cluster.
    out_png  Output image path (default: <npz-stem>_timeline.png).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- args ---------------------------------------------------------------------

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

npz_path = Path(sys.argv[1])
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else npz_path.with_name(npz_path.stem + "_timeline.png")

# --- load ---------------------------------------------------------------------

r = np.load(npz_path, allow_pickle=False)
labels = r["labels"]
start_secs = r["start_secs"]

unique_labels = sorted(np.unique(labels).tolist())
cluster_labels = [lbl for lbl in unique_labels if lbl >= 0]
n_clusters = len(cluster_labels)

# Colour map for clusters; noise always grey.
cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
colours = {lbl: cmap(i / max(n_clusters - 1, 1)) for i, lbl in enumerate(cluster_labels)}
colours[-1] = (0.75, 0.75, 0.75, 0.4)  # grey, semi-transparent

# x axis in hours for long recordings
x = start_secs / 3600.0
xlabel = "Time (hours)"

# --- plot ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, max(3, n_clusters * 0.55 + 1.5)))

for lbl in unique_labels:
    mask = labels == lbl
    tag = "noise" if lbl == -1 else f"cluster {lbl}"
    ax.scatter(
        x[mask],
        np.full(mask.sum(), lbl),
        c=[colours[lbl]],
        s=2,
        linewidths=0,
        label=f"{tag} (n={mask.sum()})",
        zorder=2 if lbl >= 0 else 1,
    )

ax.set_xlabel(xlabel)
ax.set_ylabel("Cluster")
ax.set_yticks(unique_labels)
ax.set_yticklabels(["noise" if lbl == -1 else str(lbl) for lbl in unique_labels])
ax.set_title(f"Cluster timeline — {npz_path.name}")
ax.legend(loc="upper right", markerscale=5, fontsize=8, framealpha=0.8)
ax.grid(axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig(out_png, dpi=150)
print(f"Saved {out_png}")
