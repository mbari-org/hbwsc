"""Plot the temporal sequence of cluster assignments as a color strip.

Each window is shown as a coloured tick on a single horizontal line.
Noise (-1) is shown in light grey. Colour indicates cluster.

By default a single plot covers the full recording. Use --segment-minutes to
split into multiple plots of that duration (e.g. 2 for 2-minute segments).

When the npz contains multiple audio files, their timelines are placed
sequentially (file 2 starts where file 1 ends) so they don't overlap.

Usage:
    uv run python scripts/plot_timeline.py <npz> [out_png] [--segment-minutes N]

Arguments:
    npz                Path to a results .npz file produced by hbws-cluster.
    out_png            Output image path (default: <npz-stem>_timeline.png).
                       For segmented output: <stem>_seg0000.png, etc.
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
out_png = Path(args[1]) if len(args) > 1 else npz_path.with_name(npz_path.stem + "_timeline.png")

# --- load ---------------------------------------------------------------------

r = np.load(npz_path, allow_pickle=False)
labels = r["labels"]
start_secs = r["start_secs"].copy()

# Handling multiple source files
if "source_files" in r:
    source_files = r["source_files"].astype(str)
    seen = {}
    for sf in source_files:
        if sf not in seen:
            seen[sf] = len(seen)
    unique_sources = list(seen.keys())

    if len(unique_sources) > 1:
        gap_sec = 60  # 1-minute visual gap between files
        offset = 0.0
        for sf in unique_sources:
            mask = source_files == sf
            file_secs = start_secs[mask]
            start_secs[mask] = file_secs - file_secs.min() + offset
            offset = start_secs[mask].max() + gap_sec
        print(f"Multi-file timeline: {len(unique_sources)} files placed sequentially")
        for i, sf in enumerate(unique_sources):
            print(f"  [{i}] {Path(sf).name}")

unique_labels = sorted(np.unique(labels).tolist())
cluster_labels = [lbl for lbl in unique_labels if lbl >= 0]
n_clusters = len(cluster_labels)

colours = extract_colors_from_npz(dict(r), color_mode="3D")

x_minutes = start_secs / 60.0
hop_minutes = float(np.median(np.diff(start_secs))) / 60.0

# --- plot helper --------------------------------------------------------------


def plot_segment(x_all, labels_all, x_min, x_max, out_path):
    mask_seg = (x_all >= x_min) & (x_all < x_max)
    fig, ax = plt.subplots(figsize=(24, 1.8))
    for lbl in unique_labels:
        mask = mask_seg & (labels_all == lbl)
        if not mask.any():
            continue
        tag = "noise" if lbl == -1 else f"cluster {lbl}"
        ax.bar(
            x_all[mask],
            height=1.0,
            width=hop_minutes,
            bottom=-0.5,
            color=colours[lbl],
            linewidth=0,
            label=tag,
            zorder=2 if lbl >= 0 else 1,
            align="edge",
        )

    def fmt_hm(minutes, _pos=None):
        h, m = divmod(int(minutes), 60)
        return f"{h}:{m:02d}"

    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_hm))
    ax.set_xlabel("Time (h:mm)")
    ax.set_yticks([])
    t_start = fmt_hm(x_min)
    t_end = fmt_hm(x_max)
    ax.set_title(f"{npz_path.parent.name}  [{t_start}–{t_end}]")
    ax.legend(
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

x_min_global = x_minutes.min()
x_max_global = x_minutes.max()

if segment_minutes is None:
    plot_segment(x_minutes, labels, x_min_global, x_max_global, out_png)
else:
    stem = out_png.with_suffix("").name
    seg_start = x_min_global
    seg_idx = 0
    while seg_start < x_max_global:
        seg_end = min(seg_start + segment_minutes, x_max_global)
        seg_path = out_png.parent / f"{stem}_seg{seg_idx:04d}.png"
        plot_segment(x_minutes, labels, seg_start, seg_end, seg_path)
        seg_start = seg_end
        seg_idx += 1
