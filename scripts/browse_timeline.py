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
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.widgets as mwidgets
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram as scipy_spectrogram
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score

_IS_MACOS = platform.system() == "Darwin"
if not _IS_MACOS:
    import sounddevice as sd

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("npz", type=Path)
parser.add_argument("--segment-minutes", type=float, default=2.0)
parser.add_argument("--audio", type=Path, default=None)
parser.add_argument("--manual-labels", type=Path, default=None)
parser.add_argument("--spectrogram-type", type=str, default="auto", choices=["auto", "default", "perch"], help="Spectrogram style")
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

cmap = plt.get_cmap("tab20")
colours = {lbl: cmap.colors[lbl % len(cmap.colors)] for lbl in cluster_labels}
colours[-1] = (0.75, 0.75, 0.75, 0.4)

hop_minutes = float(np.median(np.diff(start_secs))) / 60.0
x_minutes = start_secs / 60.0

# --- Density data (5-second bins) ------------------------------------------
window_min = 5.0 / 60.0
dens_bins = np.arange(x_minutes.min(), x_minutes.max() + window_min, window_min)
dens_centers = dens_bins[:-1] + window_min / 2.0
dens_counts = {lbl: np.zeros(len(dens_bins) - 1) for lbl in unique_labels}
dens_indices = np.digitize(x_minutes, dens_bins) - 1
for i, lbl in zip(dens_indices, labels):
    if 0 <= i < len(dens_bins) - 1:
        dens_counts[lbl][i] += 1
dens_total = np.sum([dens_counts[lbl] for lbl in unique_labels], axis=0)
dens_total[dens_total == 0] = 1
dens_data = [dens_counts[lbl] / dens_total for lbl in unique_labels]
dens_colors = [colours[lbl] for lbl in unique_labels]

# Create step-function arrays for stackplot to draw flat-topped bins
dens_x_step = np.repeat(dens_bins, 2)[1:-1]
dens_y_step = [np.repeat(d, 2) for d in dens_data]

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
play_start_wall   = [0.0]   # wall-clock time when playback began (adjusted for pause)
play_start_min    = [0.0]   # timeline position (minutes) at play start
pause_offset_sec  = [0.0]   # elapsed seconds at last pause
vline = [None]              # vertical playhead Line2D on ax_spec (animated)
_bg   = [None]              # blitting background snapshot
_audio_proc = [None]        # afplay subprocess (macOS only)
_tmp_wav    = [None]        # path to current temp WAV (macOS only)

# Precompute per-segment local Jaccard for "best alignment" button.
# When manual labels are present, also build a per-window class-index array
# (manual_window) so NMI/homogeneity can be computed per displayed segment.
is_clustered = labels >= 0
is_labelled = np.zeros(len(labels), dtype=bool)
manual_window = np.full(len(labels), -1, dtype=int)
unique_types_idx: dict[str, int] = {}
if manual_labels:
    unique_types_idx = {t: i for i, t in enumerate(sorted(set(t for _, _, t in manual_labels)))}
    end_secs = r["start_secs"] + 0.5  # window_sec
    best_overlap = np.zeros(len(labels), dtype=float)
    for begin_min, end_min, ltype in manual_labels:
        b, e = begin_min * 60, end_min * 60
        overlap = np.minimum(end_secs, e) - np.maximum(r["start_secs"], b)
        np.clip(overlap, 0.0, None, out=overlap)
        better = overlap > best_overlap
        best_overlap = np.where(better, overlap, best_overlap)
        manual_window = np.where(better, unique_types_idx[ltype], manual_window)
    is_labelled = manual_window >= 0

def _segment_metrics(t_min: float, t_max: float) -> dict | None:
    """DetSim (Jaccard), NMI, Homogeneity for windows in [t_min, t_max) minutes.

    Returns None when no manual labels are loaded.
    """
    if not manual_labels:
        return None
    mask = (x_minutes >= t_min) & (x_minutes < t_max)
    c = is_clustered[mask]
    m = is_labelled[mask]
    inter = c & m
    union = c | m
    detsim = inter.sum() / union.sum() if union.any() else 0.0
    if inter.sum() < 2:
        return {"detsim": detsim, "nmi": float("nan"), "homog": float("nan")}
    seg_labels = labels[mask][inter]
    seg_manual = manual_window[mask][inter]
    nmi = normalized_mutual_info_score(seg_manual, seg_labels)
    homog = homogeneity_score(seg_manual, seg_labels)
    return {"detsim": detsim, "nmi": nmi, "homog": homog}


# Precompute per-segment metrics for the "best" buttons
seg_detsim = np.zeros(n_segments)
seg_nmi    = np.zeros(n_segments)
seg_homog  = np.zeros(n_segments)
for i in range(n_segments):
    t_min = i * seg_minutes
    t_max = min(t_min + seg_minutes, total_minutes)
    metr = _segment_metrics(t_min, t_max)
    if metr is None:
        # No manual labels — fall back to fraction of clustered windows so the
        # "Best" button still has something useful to jump to.
        mask = (x_minutes >= t_min) & (x_minutes < t_max)
        seg_detsim[i] = is_clustered[mask].mean() if mask.any() else 0.0
    else:
        seg_detsim[i] = metr["detsim"]
        seg_nmi[i]    = 0.0 if np.isnan(metr["nmi"])   else metr["nmi"]
        seg_homog[i]  = 0.0 if np.isnan(metr["homog"]) else metr["homog"]
best_seg_detsim = int(np.argmax(seg_detsim))
best_seg_nmi    = int(np.argmax(seg_nmi))
best_seg_homog  = int(np.argmax(seg_homog))

# ---------------------------------------------------------------------------
# Figure layout  (add manual strip if labels provided)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(20, 10 if manual_labels else 8))

if manual_labels:
    ax_spec = fig.add_axes([0.05, 0.44, 0.93, 0.45])
    ax_time = fig.add_axes([0.05, 0.34, 0.93, 0.08])
    ax_manu = fig.add_axes([0.05, 0.24, 0.93, 0.08])
    ax_dens = fig.add_axes([0.05, 0.14, 0.93, 0.08])
else:
    ax_spec = fig.add_axes([0.05, 0.38, 0.93, 0.54])
    ax_time = fig.add_axes([0.05, 0.26, 0.93, 0.10])
    ax_dens = fig.add_axes([0.05, 0.14, 0.93, 0.10])
    ax_manu = None

bw = 0.05  # button width
ax_first  = fig.add_axes([0.06, 0.02, bw, 0.07])
ax_prev   = fig.add_axes([0.12, 0.02, bw, 0.07])
ax_next   = fig.add_axes([0.18, 0.02, bw, 0.07])
ax_last   = fig.add_axes([0.24, 0.02, bw, 0.07])
ax_play   = fig.add_axes([0.40, 0.02, bw, 0.07])
ax_rewind = fig.add_axes([0.46, 0.02, bw, 0.07])
ax_best_d = fig.add_axes([0.55, 0.02, bw, 0.07])

btn_first  = mwidgets.Button(ax_first,  "|◀ First")
btn_prev   = mwidgets.Button(ax_prev,   "◀  Prev")
btn_next   = mwidgets.Button(ax_next,   "Next  ▶")
btn_last   = mwidgets.Button(ax_last,   "Last ▶|")
btn_play   = mwidgets.Button(ax_play,   "▶  Play")
btn_rewind = mwidgets.Button(ax_rewind, "⏮ Rewind")
btn_best_d = mwidgets.Button(ax_best_d, "★ DetSim")

# NMI / Homogeneity best-segment buttons only when manual labels are loaded.
btn_best_n = btn_best_h = None
if manual_labels:
    ax_best_n = fig.add_axes([0.61, 0.02, bw, 0.07])
    ax_best_h = fig.add_axes([0.67, 0.02, bw, 0.07])
    btn_best_n = mwidgets.Button(ax_best_n, "★ NMI")
    btn_best_h = mwidgets.Button(ax_best_h, "★ Homog")

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

    spec_type = args.spectrogram_type
    if spec_type == "auto":
        spec_type = "perch" if "perch" in str(args.npz).lower() else "default"

    if spec_type == "perch":
        import librosa
        target_sr = 32000
        if sample_rate != target_sr:
            spec_chunk = librosa.resample(chunk, orig_sr=sample_rate, target_sr=target_sr)
        else:
            spec_chunk = chunk
            
        n_fft = 1024
        hop_length = 320
        win_length = 640
        
        D = librosa.stft(
            spec_chunk,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            window='hann'
        )
        mag = np.abs(D)
        
        window = librosa.filters.get_window('hann', win_length)
        scale = 1.0 / window.sum()
        mag = mag * scale
        
        n_mels = 160
        mel_basis = librosa.filters.mel(sr=target_sr, n_fft=n_fft, n_mels=n_mels, htk=True)
        mel_spec = np.dot(mel_basis, mag)
        
        log_mel = np.log(np.maximum(mel_spec, 1e-5))
        Sxx_db = log_mel * 0.1
        
        mel_freqs_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=target_sr / 2, htk=True)
        mel_bins = np.arange(n_mels)
        times = librosa.frames_to_time(np.arange(Sxx_db.shape[1]), sr=target_sr, hop_length=hop_length)
        
        ax_spec.cla()
        ax_spec.pcolormesh(times / 60 + t_min, mel_bins, Sxx_db, shading="auto",
                           cmap="Blues_r", rasterized=True)
        # Label y-axis ticks with actual Hz values at evenly-spaced mel bin positions
        tick_bins = np.linspace(0, n_mels - 1, 8, dtype=int)
        ax_spec.set_yticks(tick_bins)
        ax_spec.set_yticklabels([f"{int(mel_freqs_hz[b])}" for b in tick_bins])
        ax_spec.set_ylabel("Freq (Hz, Mel scale)")
    else:
        nperseg = min(512, len(chunk))
        freqs, times, Sxx = scipy_spectrogram(chunk, fs=sample_rate, nperseg=nperseg, noverlap=nperseg * 3 // 4)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
        ax_spec.cla()
        ax_spec.pcolormesh(times / 60 + t_min, freqs, Sxx_db, shading="auto",
                           cmap="Blues_r", rasterized=True)
        ax_spec.set_ylabel("Freq (Hz)")

    ax_spec.set_xlim(t_min, t_max)
    metrics = _segment_metrics(t_min, t_max)
    
    session_dir = args.npz.parent.parent
    params = {}
    try:
        with open(session_dir / "parameters.yml") as f:
            params = yaml.safe_load(f)
    except Exception:
        pass
        
    model = params.get("embedder_type", "unknown")
    w_sec = params.get("window_sec", "?")
    h_sec = params.get("hop_sec", "?")
    eps = params.get("hdbscan_epsilon", 0.0)

    title_prefix = f"{audio_path.name} | {model} | w:{w_sec}s h:{h_sec}s | eps:{eps} | {args.npz.parent.name}"
    title = f"{title_prefix}  [{fmt_hm(t_min)} - {fmt_hm(t_max)}]  (seg {idx + 1}/{n_segments})"

    if metrics is not None:
        nmi = "—" if np.isnan(metrics["nmi"]) else f"{metrics['nmi']:.2f}"
        hom = "—" if np.isnan(metrics["homog"]) else f"{metrics['homog']:.2f}"
        title += f"   DetSim: {metrics['detsim']:.2f}   NMI: {nmi}   Homog: {hom}"
    ax_spec.set_title(title)
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

    # --- density strip --------------------------------------------------------
    ax_dens.cla()
    mask_seg_dens = (dens_x_step >= t_min - window_min) & (dens_x_step <= t_max + window_min)
    if mask_seg_dens.any():
        seg_x = dens_x_step[mask_seg_dens]
        seg_data = [d[mask_seg_dens] for d in dens_y_step]
        ax_dens.stackplot(seg_x, *seg_data, colors=dens_colors, linewidth=0)
    ax_dens.set_xlim(t_min, t_max)
    ax_dens.set_ylim(0, 1.0)
    ax_dens.set_yticks([])
    ax_dens.set_ylabel("density", fontsize=7)
    
    ax_dens.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_hm(v)))
    ax_dens.tick_params(labelbottom=True)
    ax_dens.set_xlabel("Time (h:mm:ss)")

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
        ax_manu.tick_params(labelbottom=False)

        mhandles, mlbls = ax_manu.get_legend_handles_labels()
        if mhandles:
            ax_manu.legend(mhandles, mlbls, loc="lower left", bbox_to_anchor=(0, 1.01),
                           ncol=min(len(manual_colours), 15), fontsize=7, framealpha=0.8, borderaxespad=0)

    # Full synchronous draw, then snapshot background (vline excluded via animated=True)
    fig.canvas.draw()
    _bg[0] = fig.canvas.copy_from_bbox(fig.bbox)

    # Recreate animated playhead (not drawn during normal renders)
    vline[0] = ax_spec.axvline(x=t_min, color="white", linewidth=1.2,
                                alpha=0.85, animated=True, zorder=10)


def _kill_afplay():
    if _audio_proc[0] is not None:
        _audio_proc[0].terminate()
        _audio_proc[0] = None
    if _tmp_wav[0] is not None:
        try:
            Path(_tmp_wav[0]).unlink()
        except OSError:
            pass
        _tmp_wav[0] = None


def _play_audio(chunk):
    if _IS_MACOS:
        _kill_afplay()
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tf.name, chunk, sample_rate)
        tf.close()
        _tmp_wav[0] = tf.name
        _audio_proc[0] = subprocess.Popen(["afplay", tf.name])
    else:
        sd.play(chunk, samplerate=sample_rate)


def _stop_audio():
    if _IS_MACOS:
        _kill_afplay()
    else:
        sd.stop()


def stop_playback():
    _stop_audio()
    playing[0] = False
    pause_offset_sec[0] = 0.0
    btn_play.label.set_text("▶  Play")
    if _bg[0] is not None:
        fig.canvas.restore_region(_bg[0])
        fig.canvas.blit(fig.bbox)


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


def on_best_detsim(_event):
    stop_playback()
    seg_idx[0] = best_seg_detsim
    draw(seg_idx[0])


def on_best_nmi(_event):
    stop_playback()
    seg_idx[0] = best_seg_nmi
    draw(seg_idx[0])


def on_best_homog(_event):
    stop_playback()
    seg_idx[0] = best_seg_homog
    draw(seg_idx[0])


def on_rewind(_event):
    """Stop playback and reset position to the beginning of the current segment."""
    was_playing = playing[0]
    _stop_audio()
    playing[0] = False
    pause_offset_sec[0] = 0.0
    btn_play.label.set_text("▶  Play")
    if vline[0] is not None:
        t_min = seg_idx[0] * seg_minutes
        vline[0].set_xdata([t_min, t_min])
    fig.canvas.draw()
    _bg[0] = fig.canvas.copy_from_bbox(fig.bbox)
    if vline[0] is not None:
        ax_spec.draw_artist(vline[0])
        fig.canvas.blit(fig.bbox)


def on_play(_event):
    if playing[0]:
        # Pause: record how far we got and pin vline to that position
        pause_offset_sec[0] = time.time() - play_start_wall[0]
        _stop_audio()
        playing[0] = False
        btn_play.label.set_text("▶  Play")
        if vline[0] is not None:
            vline[0].set_xdata([play_start_min[0] + pause_offset_sec[0] / 60.0] * 2)
    else:
        idx = seg_idx[0]
        t_min = idx * seg_minutes
        t_max = min(t_min + seg_minutes, total_minutes)
        s0 = int(t_min * 60 * sample_rate)
        s1 = int(t_max * 60 * sample_rate)
        # Resume from pause offset (0 if starting fresh)
        offset_samples = int(pause_offset_sec[0] * sample_rate)
        _play_audio(audio[s0 + offset_samples:s1])
        play_start_wall[0] = time.time() - pause_offset_sec[0]
        play_start_min[0] = t_min
        playing[0] = True
        btn_play.label.set_text("⏸  Pause")
    # Redraw and refresh background so the timer doesn't restore the old label
    fig.canvas.draw()
    _bg[0] = fig.canvas.copy_from_bbox(fig.bbox)
    # Keep vline visible while paused
    if not playing[0] and vline[0] is not None:
        ax_spec.draw_artist(vline[0])
        fig.canvas.blit(fig.bbox)


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
        on_best_detsim(None)
    elif event.key == "n" and manual_labels:
        on_best_nmi(None)
    elif event.key == "m" and manual_labels:
        on_best_homog(None)
    elif event.key == "r":
        on_rewind(None)


btn_first.on_clicked(on_first)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)
btn_last.on_clicked(on_last)
btn_play.on_clicked(on_play)
btn_rewind.on_clicked(on_rewind)
btn_best_d.on_clicked(on_best_detsim)
if btn_best_n is not None:
    btn_best_n.on_clicked(on_best_nmi)
if btn_best_h is not None:
    btn_best_h.on_clicked(on_best_homog)
fig.canvas.mpl_connect("key_press_event", on_key)


def _tick_playhead(_):
    """Move the vertical playhead — uses blitting to avoid full redraws."""
    if not playing[0] or vline[0] is None or _bg[0] is None:
        return
    elapsed = time.time() - play_start_wall[0]
    pos = play_start_min[0] + elapsed / 60.0
    t_min = seg_idx[0] * seg_minutes
    t_max = min(t_min + seg_minutes, total_minutes)
    if pos >= t_max:
        stop_playback()
        return
    fig.canvas.restore_region(_bg[0])
    vline[0].set_xdata([pos, pos])
    ax_spec.draw_artist(vline[0])
    fig.canvas.blit(fig.bbox)


_timer = fig.canvas.new_timer(interval=300)
_timer.add_callback(_tick_playhead, None)
_timer.start()


def on_spectrogram_click(event):
    """Seek to clicked position in the spectrogram."""
    if event.inaxes is not ax_spec or event.xdata is None:
        return
    t_min = seg_idx[0] * seg_minutes
    t_max = min(t_min + seg_minutes, total_minutes)
    clicked_min = max(t_min, min(event.xdata, t_max))
    new_offset_sec = (clicked_min - t_min) * 60.0

    was_playing = playing[0]
    if was_playing:
        _stop_audio()
        playing[0] = False

    pause_offset_sec[0] = new_offset_sec
    if vline[0] is not None:
        vline[0].set_xdata([clicked_min, clicked_min])

    if was_playing:
        s0 = int(t_min * 60 * sample_rate)
        s1 = int(t_max * 60 * sample_rate)
        _play_audio(audio[s0 + int(new_offset_sec * sample_rate):s1])
        play_start_wall[0] = time.time() - new_offset_sec
        play_start_min[0] = t_min
        playing[0] = True
        btn_play.label.set_text("⏸  Pause")

    fig.canvas.draw()
    _bg[0] = fig.canvas.copy_from_bbox(fig.bbox)
    if vline[0] is not None:
        ax_spec.draw_artist(vline[0])
        fig.canvas.blit(fig.bbox)


fig.canvas.mpl_connect("button_press_event", on_spectrogram_click)
fig.canvas.mpl_connect("close_event", lambda _: _stop_audio())

draw(0)
plt.show()
