"""Interactive timeline browser: spectrogram + cluster color strip.

Displays a figure with two panels for the current time segment:
  - Top:    spectrogram of the audio
  - Bottom: cluster color strip (same style as plot_timeline.py)

Use the Prev / Next buttons (or left/right arrow keys) to navigate segments.

Usage:
    uv run python scripts/browse_timeline.py <npz> [--segment-minutes N] [--audio FILE]

Arguments:
    npz                Path to results .npz produced by hbws-cluster.
    --segment-minutes  Segment duration in minutes (default: 2).
    --audio            WAV file to use (default: first file in npz source_files).
"""

import argparse
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
# Load audio
# ---------------------------------------------------------------------------

if args.audio:
    audio_path = args.audio
else:
    source_files = r["source_files"].astype(str).tolist()
    audio_path = Path(source_files[0])
    if not audio_path.exists():
        # try relative to npz location
        audio_path = args.npz.parent.parent.parent / audio_path.name
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {source_files[0]}")
        print("Use --audio to specify the WAV file.")
        sys.exit(1)

print(f"Loading audio: {audio_path} ...")
audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # mix to mono
print(f"Audio: {len(audio) / sample_rate:.1f}s  @{sample_rate}Hz")

# ---------------------------------------------------------------------------
# Segment state
# ---------------------------------------------------------------------------

seg_minutes = args.segment_minutes
total_minutes = x_minutes.max()
n_segments = int(np.ceil(total_minutes / seg_minutes))
seg_idx = [0]  # mutable for closure
playing = [False]

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(20, 5))
fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.22, hspace=0.08)

ax_spec = fig.add_axes([0.05, 0.32, 0.93, 0.54])
ax_time = fig.add_axes([0.05, 0.20, 0.93, 0.10])
ax_prev = fig.add_axes([0.33, 0.04, 0.10, 0.08])
ax_info = fig.add_axes([0.43, 0.04, 0.10, 0.08])
ax_next = fig.add_axes([0.53, 0.04, 0.10, 0.08])
ax_play = fig.add_axes([0.64, 0.04, 0.10, 0.08])

btn_prev = mwidgets.Button(ax_prev, "◀  Prev")
btn_next = mwidgets.Button(ax_next, "Next  ▶")
btn_play = mwidgets.Button(ax_play, "▶  Play")
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
    ax_spec.pcolormesh(times / 60 + t_min, freqs, Sxx_db, shading="auto", cmap="magma", rasterized=True)
    ax_spec.set_xlim(t_min, t_max)
    ax_spec.set_ylabel("Freq (Hz)")
    ax_spec.set_title(f"{args.npz.parent.name}  [{fmt_hm(t_min)} – {fmt_hm(t_max)}]  (seg {idx + 1}/{n_segments})")
    ax_spec.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_hm(v)))
    ax_spec.tick_params(labelbottom=False)

    # --- timeline strip -------------------------------------------------------
    ax_time.cla()
    mask_seg = (x_minutes >= t_min) & (x_minutes < t_max)
    for lbl in unique_labels:
        mask = mask_seg & (labels == lbl)
        if not mask.any():
            continue
        tag = "noise" if lbl == -1 else f"cluster {lbl}"
        ax_time.bar(
            x_minutes[mask],
            height=1.0,
            width=hop_minutes,
            bottom=-0.5,
            color=colours[lbl],
            linewidth=0,
            label=tag,
            align="edge",
        )
    ax_time.set_xlim(t_min, t_max)
    ax_time.set_ylim(-0.5, 0.5)
    ax_time.set_yticks([])
    ax_time.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_hm(v)))
    ax_time.set_xlabel("Time (h:mm:ss)")

    # legend above spectrogram
    handles, lbls = ax_time.get_legend_handles_labels()
    ax_spec.legend(
        handles,
        lbls,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=min(n_clusters + 1, 12),
        fontsize=7,
        framealpha=0.8,
        borderaxespad=0,
    )

    info_text.set_text(f"{idx + 1} / {n_segments}")
    fig.canvas.draw_idle()


def stop_playback():
    sd.stop()
    playing[0] = False
    btn_play.label.set_text("▶  Play")


def on_next(_event):
    if seg_idx[0] < n_segments - 1:
        stop_playback()
        seg_idx[0] += 1
        draw(seg_idx[0])


def on_prev(_event):
    if seg_idx[0] > 0:
        stop_playback()
        seg_idx[0] -= 1
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


btn_next.on_clicked(on_next)
btn_prev.on_clicked(on_prev)
btn_play.on_clicked(on_play)
fig.canvas.mpl_connect("key_press_event", on_key)

draw(0)
plt.show()
