"""Interactive timeline browser: spectrogram + cluster color strip.

Displays a figure with panels for the current time segment:
  - Top:    spectrogram of the audio
  - Middle: cluster colour strip (HDBSCAN output)
  - Bottom: manual label strip (optional, Raven format)

Use the Prev / Next buttons (or left/right arrow keys) to navigate segments.
Spacebar or the Play button toggles play/pause.

Usage:
    uv run python scripts/browse_timeline.py <npz> [options]

Arguments:
    npz                Path to results .npz produced by hbws-cluster.
    --segment-minutes  Segment duration in minutes (default: 2).
    --audio            WAV file (default: first file in npz source_files).
    --manual-labels    Raven selection table with manual labels (optional).
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.widgets as mwidgets
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import spectrogram as scipy_spectrogram

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("npz", type=Path)
parser.add_argument("--segment-minutes", type=float, default=2.0)
parser.add_argument("--audio", type=Path, default=None)
parser.add_argument("--manual-labels", type=Path, default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load npz
# ---------------------------------------------------------------------------

r = np.load(args.npz, allow_pickle=False)
labels = r["labels"]
start_secs = r["start_secs"]

unique_labels = sorted(np.unique(labels).tolist())
cluster_labels = [lbl for lbl in unique_labels if lbl >= 0]
n_clusters = len(cluster_labels)

cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
colours = {lbl: cmap(i / max(n_clusters - 1, 1)) for i, lbl in enumerate(cluster_labels)}
colours[-1] = (0.75, 0.75, 0.75, 0.4)

hop_minutes = float(np.median(np.diff(start_secs))) / 60.0
x_minutes = start_secs / 60.0

# ---------------------------------------------------------------------------
# Load manual labels (Raven format)
# ---------------------------------------------------------------------------

manual_labels = None  # list of (begin_min, end_min, type_str)
manual_colours = {}

if args.manual_labels:
    manual_labels = []
    with open(args.manual_labels) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            begin_min = float(row["Begin Time (s)"]) / 60.0
            end_min = float(row["End Time (s)"]) / 60.0
            label_type = row["Type"].strip()
            manual_labels.append((begin_min, end_min, label_type))

    unique_types = sorted(set(t for _, _, t in manual_labels))
    n_types = len(unique_types)
    mcmap = plt.get_cmap("tab10" if n_types <= 10 else "tab20")
    manual_colours = {t: mcmap(i / max(n_types - 1, 1)) for i, t in enumerate(unique_types)}
    print(f"Manual labels: {len(manual_labels)} selections, {n_types} types: {unique_types}")

# ---------------------------------------------------------------------------
# Load audio
# ---------------------------------------------------------------------------

if args.audio:
    audio_path = args.audio
else:
    source_files = r["source_files"].astype(str).tolist()
    audio_path = Path(source_files[0])
    if not audio_path.exists():
        audio_path = args.npz.parent.parent.parent / audio_path.name
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {source_files[0]}")
        print("Use --audio to specify the WAV file.")
        sys.exit(1)

print(f"Loading audio: {audio_path} ...")
audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
print(f"Audio: {len(audio) / sample_rate:.1f}s  @{sample_rate}Hz")

# ---------------------------------------------------------------------------
# Segment state
# ---------------------------------------------------------------------------

seg_minutes = args.segment_minutes
total_minutes = x_minutes.max()
n_segments = int(np.ceil(total_minutes / seg_minutes))
seg_idx = [0]
playing = [False]

# Precompute per-segment local Jaccard for "best alignment" button
is_clustered = labels >= 0
is_labelled = np.zeros(len(labels), dtype=bool)
if manual_labels:
    end_secs = r["start_secs"] + 0.5  # window_sec
    for begin_min, end_min, _ in manual_labels:
        b, e = begin_min * 60, end_min * 60
        is_labelled |= (end_secs > b) & (r["start_secs"] < e)

seg_jaccard = []
for i in range(n_segments):
    t_min = i * seg_minutes
    t_max = min(t_min + seg_minutes, total_minutes)
    mask = (x_minutes >= t_min) & (x_minutes < t_max)
    c = is_clustered[mask]
    m = is_labelled[mask]
    inter = (c & m).sum()
    union = (c | m).sum()
    seg_jaccard.append(inter / union if union > 0 else 0.0)
best_seg = int(np.argmax(seg_jaccard))

# ---------------------------------------------------------------------------
# Figure layout  (add manual strip if labels provided)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(20, 9 if manual_labels else 7))

if manual_labels:
    ax_spec = fig.add_axes([0.05, 0.34, 0.93, 0.55])
    ax_time = fig.add_axes([0.05, 0.24, 0.93, 0.08])
    ax_manu = fig.add_axes([0.05, 0.14, 0.93, 0.08])
else:
    ax_spec = fig.add_axes([0.05, 0.32, 0.93, 0.60])
    ax_time = fig.add_axes([0.05, 0.20, 0.93, 0.10])
    ax_manu = None

bw = 0.08  # button width
ax_first = fig.add_axes([0.14, 0.02, bw, 0.07])
ax_prev  = fig.add_axes([0.23, 0.02, bw, 0.07])
ax_info  = fig.add_axes([0.32, 0.02, bw, 0.07])
ax_next  = fig.add_axes([0.41, 0.02, bw, 0.07])
ax_last  = fig.add_axes([0.50, 0.02, bw, 0.07])
ax_play  = fig.add_axes([0.61, 0.02, bw, 0.07])
ax_best  = fig.add_axes([0.71, 0.02, bw, 0.07])

btn_first = mwidgets.Button(ax_first, "|◀ First")
btn_prev  = mwidgets.Button(ax_prev,  "◀  Prev")
btn_next  = mwidgets.Button(ax_next,  "Next  ▶")
btn_last  = mwidgets.Button(ax_last,  "Last ▶|")
btn_play  = mwidgets.Button(ax_play,  "▶  Play")
btn_best  = mwidgets.Button(ax_best,  "★ Best")
ax_info.axis("off")
info_text = ax_info.text(0.5, 0.5, "", ha="center", va="center", fontsize=9, transform=ax_info.transAxes)

# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------

def fmt_hm(minutes):
    h, m = divmod(int(minutes), 60)
    s = int((minutes - int(minutes)) * 60)
    return f"{h}:{m:02d}:{s:02d}"


def draw(idx):
    t_min = idx * seg_minutes
    t_max = min(t_min + seg_minutes, total_minutes)

    # --- spectrogram ----------------------------------------------------------
    s0 = int(t_min * 60 * sample_rate)
    s1 = int(t_max * 60 * sample_rate)
    chunk = audio[s0:s1]

    nperseg = min(512, len(chunk))
    freqs, times, Sxx = scipy_spectrogram(chunk, fs=sample_rate, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    ax_spec.cla()
    ax_spec.pcolormesh(times / 60 + t_min, freqs, Sxx_db, shading="auto",
                       cmap="Blues_r", rasterized=True)
    ax_spec.set_xlim(t_min, t_max)
    ax_spec.set_ylabel("Freq (Hz)")
    ax_spec.set_title(f"{args.npz.parent.name}  [{fmt_hm(t_min)} – {fmt_hm(t_max)}]  (seg {idx + 1}/{n_segments})")
    ax_spec.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_hm(v)))
    ax_spec.tick_params(labelbottom=False)

    # --- cluster strip --------------------------------------------------------
    ax_time.cla()
    mask_seg = (x_minutes >= t_min) & (x_minutes < t_max)
    for lbl in unique_labels:
        mask = mask_seg & (labels == lbl)
        if not mask.any():
            continue
        tag = "noise" if lbl == -1 else f"cluster {lbl}"
        ax_time.bar(
            x_minutes[mask], height=1.0, width=hop_minutes, bottom=-0.5,
            color=colours[lbl], linewidth=0, label=tag, align="edge",
        )
    ax_time.set_xlim(t_min, t_max)
    ax_time.set_ylim(-0.5, 0.5)
    ax_time.set_yticks([])
    ax_time.set_ylabel("clusters", fontsize=7)
    ax_time.tick_params(labelbottom=False)

    handles, lbls = ax_time.get_legend_handles_labels()
    ax_time.legend(handles, lbls, loc="lower left", bbox_to_anchor=(0, 1.01),
                   ncol=min(n_clusters + 1, 12), fontsize=7, framealpha=0.8, borderaxespad=0)

    # --- manual labels strip --------------------------------------------------
    if ax_manu is not None:
        ax_manu.cla()
        seen = set()
        for begin_min, end_min, ltype in manual_labels:
            if end_min < t_min or begin_min > t_max:
                continue
            lbl_kw = dict(label=ltype) if ltype not in seen else {}
            seen.add(ltype)
            ax_manu.barh(
                0,
                width=end_min - begin_min,
                left=begin_min,
                height=1.0,
                color=manual_colours[ltype],
                linewidth=0,
                **lbl_kw,
            )
            # label text inside bar if wide enough
            mid = (begin_min + end_min) / 2
            if t_min <= mid <= t_max:
                ax_manu.text(mid, 0, ltype, ha="center", va="center", fontsize=6, clip_on=True)
        ax_manu.set_xlim(t_min, t_max)
        ax_manu.set_ylim(-0.5, 0.5)
        ax_manu.set_yticks([])
        ax_manu.set_ylabel("manual", fontsize=7)
        ax_manu.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_hm(v)))
        ax_manu.set_xlabel("Time (h:mm:ss)")

        mhandles, mlbls = ax_manu.get_legend_handles_labels()
        if mhandles:
            ax_manu.legend(mhandles, mlbls, loc="lower left", bbox_to_anchor=(0, 1.01),
                           ncol=min(len(manual_colours), 15), fontsize=7, framealpha=0.8, borderaxespad=0)
    else:
        ax_time.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_hm(v)))
        ax_time.set_xlabel("Time (h:mm:ss)")

    info_text.set_text(f"{idx + 1} / {n_segments}")
    fig.canvas.draw_idle()


def stop_playback():
    sd.stop()
    playing[0] = False
    btn_play.label.set_text("▶  Play")


def on_first(_event):
    stop_playback()
    seg_idx[0] = 0
    draw(seg_idx[0])


def on_prev(_event):
    if seg_idx[0] > 0:
        stop_playback()
        seg_idx[0] -= 1
        draw(seg_idx[0])


def on_next(_event):
    if seg_idx[0] < n_segments - 1:
        stop_playback()
        seg_idx[0] += 1
        draw(seg_idx[0])


def on_last(_event):
    stop_playback()
    seg_idx[0] = n_segments - 1
    draw(seg_idx[0])


def on_best(_event):
    stop_playback()
    seg_idx[0] = best_seg
    draw(seg_idx[0])


def on_play(_event):
    if playing[0]:
        sd.stop()
        playing[0] = False
        btn_play.label.set_text("▶  Play")
    else:
        idx = seg_idx[0]
        t_min = idx * seg_minutes
        t_max = min(t_min + seg_minutes, total_minutes)
        s0 = int(t_min * 60 * sample_rate)
        s1 = int(t_max * 60 * sample_rate)
        sd.play(audio[s0:s1], samplerate=sample_rate)
        playing[0] = True
        btn_play.label.set_text("⏸  Pause")
    fig.canvas.draw_idle()


def on_key(event):
    if event.key == "right":
        on_next(None)
    elif event.key == "left":
        on_prev(None)
    elif event.key == " ":
        on_play(None)
    elif event.key == "home":
        on_first(None)
    elif event.key == "end":
        on_last(None)
    elif event.key == "b":
        on_best(None)


btn_first.on_clicked(on_first)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)
btn_last.on_clicked(on_last)
btn_play.on_clicked(on_play)
btn_best.on_clicked(on_best)
fig.canvas.mpl_connect("key_press_event", on_key)

draw(0)
plt.show()
