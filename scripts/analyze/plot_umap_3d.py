"""Plot the 3D UMAP projection from a results.npz file produced by hbws-cluster."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hbws_clustering.colors import get_3d_colors, get_default_colors

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("npz", type=Path, nargs="?", default=Path("results.npz"), help="Path to results.npz")
parser.add_argument("out", type=Path, nargs="?", default=None, help="Output PNG path")
parser.add_argument("--default-colors", action="store_true", help="Use default tab20/turbo palette instead of spatial colors")
args = parser.parse_args()

npz_path = args.npz
out_path = args.out if args.out is not None else npz_path.with_suffix(".png").with_name(npz_path.stem + "_3d.png")

r = dict(np.load(npz_path))
labels = r["labels"]  # (N,)
probabilities = r.get("probabilities")

# Check for 3D UMAP
if "reduced_3d" in r:
    xyz = r["reduced_3d"]
else:
    if "embeddings" not in r:
        print("ERROR: npz file does not contain 'embeddings', cannot compute 3D UMAP.")
        sys.exit(1)
    embeddings = r["embeddings"]
    # UMAP parameters matching the 2D version typically used
    reducer = umap.UMAP(n_components=3, random_state=42)
    xyz = reducer.fit_transform(embeddings)
    
    # Save back to npz so it doesn't have to be computed again
    r["reduced_3d"] = xyz
    np.savez_compressed(npz_path, **r)

cluster_ids = sorted(k for k in np.unique(labels) if k >= 0)
noise_mask = labels == -1

if probabilities is not None and not args.default_colors:
    colors = get_3d_colors(labels, xyz, probabilities)
else:
    colors = get_default_colors(labels)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Noise points: small grey dots in the background
if noise_mask.any():
    ax.scatter(
        xyz[noise_mask, 0],
        xyz[noise_mask, 1],
        xyz[noise_mask, 2],
        c="lightgrey",
        s=8,
        linewidths=0,
        alpha=0.1,
        label="noise",
        zorder=1,
    )

# Cluster points: one color per cluster
for cid in cluster_ids:
    mask = labels == cid
    color = colors[cid]
    ax.scatter(
        xyz[mask, 0],
        xyz[mask, 1],
        xyz[mask, 2],
        c=[color],
        s=18,
        linewidths=0,
        alpha=0.85,
        label=f"cluster {cid} (n={mask.sum()})",
        zorder=2,
    )
    # Mark centroid
    cx, cy, cz = xyz[mask].mean(axis=0)
    ax.scatter(cx, cy, cz, c=[color], s=120, marker="*", edgecolors="white", linewidths=0.6, zorder=3)

ax.set_title(f"3D UMAP projection — {npz_path.name}")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")
ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=9, framealpha=0.7)
fig.tight_layout(rect=[0, 0, 0.8, 1])

fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved {out_path}")

print("\n--- Opening interactive 3D window ---")
print("You can rotate the plot in this window to explore the 3D clusters.")
plt.show()
