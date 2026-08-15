"""Plot the UMAP projection from a results.npz file produced by hbws-cluster."""

import argparse
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hbws_clustering.colors import get_2d_colors, get_default_colors, get_3d_colors

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("npz", type=Path, nargs="?", default=Path("results.npz"), help="Path to results.npz")
parser.add_argument("out", type=Path, nargs="?", default=None, help="Output PNG path")
parser.add_argument("--color", type=str, default="2D", choices=["2D", "3D", "default"], help="Color mapping mode")
args = parser.parse_args()

npz_path = args.npz
out_path = args.out if args.out is not None else npz_path.with_suffix(".png")

r = np.load(npz_path)
xy = r["reduced"]  # (N, 2)
labels = r["labels"]  # (N,)
probabilities = r.get("probabilities")

cluster_ids = sorted(k for k in np.unique(labels) if k >= 0)
noise_mask = labels == -1

reduced_3d = r["reduced_3d"] if "reduced_3d" in r else None

if args.color == "2D" and probabilities is not None:
    colors = get_2d_colors(labels, xy, probabilities)
elif args.color == "3D" and reduced_3d is not None:
    colors = get_3d_colors(labels, reduced_3d, probabilities)
elif args.color == "3D" and reduced_3d is None:
    print("no 3D map found.")
    colors = get_2d_colors(labels, xy, probabilities) if probabilities is not None else get_default_colors(labels)
else:
    colors = get_default_colors(labels)

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

# Cluster points: one color per cluster
for cid in cluster_ids:
    mask = labels == cid
    color = colors[cid]
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
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.7)
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout(rect=[0, 0, 0.85, 1])

fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved {out_path}")
