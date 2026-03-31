"""Export audio snippets for windows belonging to a given cluster.

Usage:
    python scripts/export_cluster.py <npz> <cluster_label> [out_dir] [n_samples] [window_sec]

Arguments:
    npz            Path to a results .npz file produced by hbws-cluster.
    cluster_label  Integer cluster label to export (-1 for noise).
    out_dir        Directory to write WAV files to.
                   Defaults to output/cluster_<label>/.
    n_samples      Maximum number of windows to export (default: 10).
                   Pass 0 to export all.
    window_sec     Duration of each exported snippet in seconds (default: 2.0).
                   Must match the value used when the pipeline was run.

Notes:
    Hydrophone recordings have very low signal levels relative to digital full
    scale. Each exported clip is peak-normalised to -3 dBFS so it is audible in
    standard audio players. This is purely for listening — it does not affect
    the clustering or any quantitative analysis.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# --- parse args ---------------------------------------------------------------

npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/results_from_scored.npz")
cluster_label = int(sys.argv[2]) if len(sys.argv) > 2 else 0
out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"output/cluster_{cluster_label}")
n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 10
window_sec = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0

TARGET_PEAK = 10 ** (-3 / 20)  # -3 dBFS


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak < 1e-9:
        return audio  # silence — don't amplify noise floor
    return (audio / peak * TARGET_PEAK).astype(np.float32)


# --- load npz -----------------------------------------------------------------

r = np.load(npz_path, allow_pickle=False)

if "start_secs" not in r or "source_files" not in r:
    print(
        "ERROR: npz is missing timestamp data.\n"
        "Re-run the pipeline with the current version to generate a new .npz."
    )
    sys.exit(1)

labels = r["labels"]
start_secs = r["start_secs"]
source_files = r["source_files"].astype(str)  # handles both str and bytes arrays

mask = labels == cluster_label
total = mask.sum()

if total == 0:
    print(f"No windows with label {cluster_label} found in {npz_path}.")
    sys.exit(0)

tag = "noise" if cluster_label == -1 else f"cluster {cluster_label}"
print(f"\n{tag}: {total} windows in {npz_path}")

# --- sample -------------------------------------------------------------------

indices = np.where(mask)[0]
if n_samples > 0 and total > n_samples:
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(indices, size=n_samples, replace=False))
    print(f"Exporting {n_samples} randomly sampled windows (seed=0 for reproducibility).")
else:
    print(f"Exporting all {total} windows.")

print("Clips are peak-normalised to -3 dBFS for listening.")

# --- export -------------------------------------------------------------------

out_dir.mkdir(parents=True, exist_ok=True)

for idx in indices:
    start_sec = float(start_secs[idx])
    source_file = source_files[idx]

    with sf.SoundFile(source_file) as f:
        sr = f.samplerate
        start_frame = int(start_sec * sr)
        n_frames = int(window_sec * sr)
        f.seek(start_frame)
        audio = f.read(n_frames, dtype="float32", always_2d=False)

    audio = peak_normalize(audio)

    label_tag = "noise" if cluster_label == -1 else f"c{cluster_label}"
    out_path = out_dir / f"{label_tag}_{start_sec:.2f}s.wav"
    sf.write(out_path, audio, sr)
    print(f"  {out_path}")

print(f"\nDone — {len(indices)} files written to {out_dir}/")
