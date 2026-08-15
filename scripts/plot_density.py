"""Plot the proportional density of clusters over time as a stacked area chart.

The timeline is divided into bins (default 5 seconds). For each bin, the proportion
of windows belonging to each cluster is calculated and plotted.

By default a single plot covers the full recording. Use --segment-minutes to
split into multiple plots of that duration (e.g. 2 for 2-minute segments).

Usage:
    uv run python scripts/plot_density.py <npz> [out_png] [--window-sec S] [--segment-minutes N]

Arguments:
    npz                Path to a results .npz file produced by hbws-cluster.
    out_png            Output image path (default: <npz-stem>_density.png).
                       For segmented output: <stem>_density_seg0000.png, etc.
    --window-sec       Size of the bin in seconds for calculating density (default: 5.0).
    --segment-minutes  Minutes per segment (default: full recording in one plot).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hbws_clustering.colors import extract_colors_from_npz

# --- args ---------------------------------------------------------------------

args = sys.argv[1:]
if not args:
    print(__doc__)
    sys.exit(1)

window_sec = 60.0
if "--window-sec" in args:
    idx = args.index("--window-sec")
    window_sec = float(args[idx + 1])
    args = args[:idx] + args[idx + 2 :]

segment_minutes = None
if "--segment-minutes" in args:
    idx = args.index("--segment-minutes")
    segment_minutes = float(args[idx + 1])
    args = args[:idx] + args[idx + 2 :]
elif "--segment-hours" in args:  # legacy support from session.py
    idx = args.index("--segment-hours")
    segment_minutes = float(args[idx + 1]) * 60
    args = args[:idx] + args[idx + 2 :]

npz_path = Path(args[0])
out_png = Path(args[1]) if len(args) > 1 else npz_path.with_name(npz_path.stem + "_density.png")

# --- load ---------------------------------------------------------------------

r = np.load(npz_path, allow_pickle=False)
labels = r["labels"]
start_secs = r["start_secs"]

unique_labels = sorted(np.unique(labels).tolist())
cluster_labels = [lbl for lbl in unique_labels if lbl >= 0]
n_clusters = len(cluster_labels)

colours = extract_colors_from_npz(dict(r), color_mode="3D")

x_minutes = start_secs / 60.0

# --- density calculation ------------------------------------------------------

# Define bins in minutes
t_min_global = x_minutes.min()
t_max_global = x_minutes.max()
window_min = window_sec / 60.0

bins = np.arange(t_min_global, t_max_global + window_min, window_min)

# Compute counts for each label in each bin
counts = {lbl: np.zeros(len(bins) - 1) for lbl in unique_labels}
indices = np.digitize(x_minutes, bins) - 1

# Accumulate counts
for i, lbl in zip(indices, labels):
    if 0 <= i < len(bins) - 1:
        counts[lbl][i] += 1

# Convert counts to proportional density
total_counts = np.sum([counts[lbl] for lbl in unique_labels], axis=0)
# Avoid division by zero
total_counts[total_counts == 0] = 1

density = {lbl: counts[lbl] / total_counts for lbl in unique_labels}

# Order for stackplot: we typically want noise (-1) at the bottom
stack_labels = sorted(unique_labels)
stack_data = [density[lbl] for lbl in stack_labels]
stack_colors = [colours[lbl] for lbl in stack_labels]

x_step = np.repeat(bins, 2)[1:-1]
y_step = [np.repeat(d, 2) for d in stack_data]

# --- plot helper --------------------------------------------------------------

def plot_segment(x_min, x_max, out_path):
    mask_seg = (x_step >= x_min - window_min) & (x_step <= x_max + window_min)
    if not mask_seg.any():
        return

    seg_x = x_step[mask_seg]
    seg_data = [d[mask_seg] for d in y_step]

    fig, ax = plt.subplots(figsize=(24, 1.8))
    
    # stackplot
    ax.stackplot(seg_x, *seg_data, colors=stack_colors, linewidth=0)

    def fmt_hm(minutes, _pos=None):
        h, m = divmod(int(minutes), 60)
        return f"{h}:{m:02d}"

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_hm))
    ax.set_xlabel("Time (h:mm)")
    ax.set_ylabel("Density")
    ax.set_yticks([0, 0.5, 1.0])
    
    t_start = fmt_hm(x_min)
    t_end = fmt_hm(x_max)
    ax.set_title(f"{npz_path.parent.name} Density  [{t_start}–{t_end}]")

    # Legend
    legend_handles = []
    for lbl in stack_labels:
        tag = "noise" if lbl == -1 else f"cluster {lbl}"
        patch = plt.Rectangle((0,0),1,1, color=colours[lbl])
        legend_handles.append((patch, tag))
    
    patches, lbls = zip(*legend_handles)
    ax.legend(
        patches,
        lbls,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=min(n_clusters + 1, 10),
        fontsize=7,
        framealpha=0.8,
        borderaxespad=0,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- generate -----------------------------------------------------------------

if segment_minutes is None:
    plot_segment(t_min_global, t_max_global, out_png)
else:
    stem = out_png.with_suffix("").name
    seg_start = t_min_global
    seg_idx = 0
    while seg_start < t_max_global:
        seg_end = min(seg_start + segment_minutes, t_max_global)
        seg_path = out_png.parent / f"{stem}_seg{seg_idx:04d}.png"
        plot_segment(seg_start, seg_end, seg_path)
        seg_start = seg_end
        seg_idx += 1
