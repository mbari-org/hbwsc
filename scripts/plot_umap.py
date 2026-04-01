"""Plot the UMAP projection from a results.npz file produced by hbws-cluster."""

import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results.npz")
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else npz_path.with_suffix(".png")

r = np.load(npz_path)
xy = r["reduced"]  # (N, 2)
labels = r["labels"]  # (N,)

cluster_ids = sorted(k for k in np.unique(labels) if k >= 0)
noise_mask = labels == -1

fig, ax = plt.subplots(figsize=(8, 6))

# Noise points: small grey dots in the background
if noise_mask.any():
    ax.scatter(
        xy[noise_mask, 0],
        xy[noise_mask, 1],
        c="lightgrey",
        s=8,
        linewidths=0,
        alpha=0.6,
        label="noise",
        zorder=1,
    )

# Cluster points: one colour per cluster
palette = cm.tab10.colors
for cid in cluster_ids:
    mask = labels == cid
    color = palette[cid % len(palette)]
    ax.scatter(
        xy[mask, 0],
        xy[mask, 1],
        c=[color],
        s=18,
        linewidths=0,
        alpha=0.85,
        label=f"cluster {cid} (n={mask.sum()})",
        zorder=2,
    )
    # Mark centroid
    cx, cy = xy[mask].mean(axis=0)
    ax.scatter(cx, cy, c=[color], s=120, marker="*", edgecolors="white", linewidths=0.6, zorder=3)

ax.set_title(f"UMAP projection — {npz_path.name}")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(loc="best", fontsize=9, framealpha=0.7)
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()

fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
