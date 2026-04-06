"""Export a Raven Pro selection table from clustering results.

Writes a tab-separated .txt file that Raven Pro can open directly as an
annotation layer on top of the original WAV file. Each window becomes one
selection, annotated with its cluster label.

Usage:
    uv run python scripts/export_raven_table.py <npz> <window_sec> [out_txt]

Arguments:
    npz          Path to a results .npz file produced by hbws-cluster.
    window_sec   Window duration in seconds — must match the pipeline run.
    out_txt      Output path (default: <npz-stem>_raven.txt).
"""

import sys
from pathlib import Path

import numpy as np

# --- args ---------------------------------------------------------------------

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

npz_path = Path(sys.argv[1])
window_sec = float(sys.argv[2])
out_txt = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else npz_path.with_name(npz_path.stem + "_raven.txt")

# --- load ---------------------------------------------------------------------

r = np.load(npz_path, allow_pickle=False)

if "start_secs" not in r or "source_files" not in r:
    print("ERROR: npz is missing timestamp data. Re-run the pipeline to generate a new .npz.")
    sys.exit(1)

labels = r["labels"]
start_secs = r["start_secs"]
probabilities = r["probabilities"] if "probabilities" in r else np.ones(len(labels))

# --- write --------------------------------------------------------------------

columns = [
    "Selection",
    "View",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Cluster",
    "Label",
    "Score",
]

order = np.argsort(start_secs)

with open(out_txt, "w") as f:
    f.write("\t".join(columns) + "\n")
    for i, idx in enumerate(order, start=1):
        lbl = int(labels[idx])
        start = float(start_secs[idx])
        prob = float(probabilities[idx])
        tag = "noise" if lbl == -1 else str(lbl)
        row = [
            str(i),
            "Spectrogram 1",
            "1",
            f"{start:.6f}",
            f"{start + window_sec:.6f}",
            "0",
            "0",
            tag,
            f"cluster_{tag}",
            f"{prob:.4f}",
        ]
        f.write("\t".join(row) + "\n")

n_clusters = len(set(labels) - {-1})
print(f"Written {len(labels)} selections ({n_clusters} clusters + noise) to {out_txt}")
print("Open in Raven Pro: File → Open Sound → then File → Open Selection Table")
