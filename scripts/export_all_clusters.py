"""Export audio snippets for every cluster in an npz file.

Creates one subdirectory per cluster (including noise) under out_dir, each
containing n_samples randomly sampled WAV files peak-normalised to -3 dBFS.

Usage:
    uv run python scripts/export_all_clusters.py <npz> <window_sec> [n_samples] [out_dir]

Arguments:
    npz          Path to a results .npz file produced by hbws-cluster.
    window_sec   Window duration in seconds — must match the pipeline run.
    n_samples    Max samples per cluster (default: 10). Pass 0 for all.
    out_dir      Root output directory (default: output/clusters_<npz-stem>/).
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# --- args ---------------------------------------------------------------------

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

npz_path = Path(sys.argv[1])
window_sec = float(sys.argv[2])
n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(f"output/clusters_{npz_path.stem}")

TARGET_PEAK = 10 ** (-3 / 20)  # -3 dBFS


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak < 1e-9:
        return audio
    return (audio / peak * TARGET_PEAK).astype(np.float32)


# --- load npz -----------------------------------------------------------------

r = np.load(npz_path, allow_pickle=False)

if "start_secs" not in r or "source_files" not in r:
    print("ERROR: npz is missing timestamp data. Re-run the pipeline to generate a new .npz.")
    sys.exit(1)

labels = r["labels"]
start_secs = r["start_secs"]
source_files = r["source_files"].astype(str)

unique_labels = sorted(np.unique(labels).tolist())
rng = np.random.default_rng(0)

print(f"\n{npz_path.name}: {len(labels)} windows, {len(unique_labels)} labels")
print(f"Exporting up to {n_samples} samples per cluster to {out_dir}/\n")

# --- export -------------------------------------------------------------------

for label in unique_labels:
    tag = "noise" if label == -1 else f"cluster_{label}"
    mask = labels == label
    total = mask.sum()
    indices = np.where(mask)[0]

    if n_samples > 0 and total > n_samples:
        indices = np.sort(rng.choice(indices, size=n_samples, replace=False))
        note = f"{n_samples} sampled"
    else:
        note = f"all {total}"

    cluster_dir = out_dir / tag
    cluster_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {tag}: {total} windows — exporting {note}")

    for idx in indices:
        start_sec = float(start_secs[idx])
        source_file = source_files[idx]

        with sf.SoundFile(source_file) as f:
            sr = f.samplerate
            f.seek(int(start_sec * sr))
            audio = f.read(int(window_sec * sr), dtype="float32", always_2d=False)

        audio = peak_normalize(audio)
        out_path = cluster_dir / f"{tag}_{start_sec:.2f}s.wav"
        sf.write(out_path, audio, sr)

print(f"\nDone — {out_dir}/")
