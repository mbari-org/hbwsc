"""Takes a new session dirctory and runs a specified trained classifier (pkl file). Gets embeddings from audio file, and outputs predictions in a new npz file.

Usage:
    uv run python scripts/predict.py <parameters.yml> <pkl> [out.npz]
    [FUTURE] add model parameter (logreg vs kmeans etc )

Arguments:
    parameters.yml     Target session's parameters
    pkl                Path to a model.pkl file produced by train_classifier.
    out.npz            Output model path (default: <target_session>/predictions/<pkl-stem>_predictions.npz).
    [FUTURE] --embedding  type of embedding (Perch vs AVES).
"""
import sys
import joblib
import yaml
from pathlib import Path
import numpy as np

from hbws_clustering.windowing import AudioWindower
from hbws_clustering.embedding import PerchEmbedder, AvesEmbedder

# --- args ---------------------------------------------------------------------

args = sys.argv[1:]
if not args:
    print(__doc__)
    sys.exit(1)

parameters_path = Path(args[0])

pkl_path = Path(args[1])

npz_out = Path(args[2]) if len(args) > 2 and args[2] else parameters_path.parent / "predictions" / f"{pkl_path.parts[-4]}_{pkl_path.parts[-3]}.npz"

if npz_out.exists():
    print(f"Labels {npz_out} already exist. Skipping inference.")
    sys.exit(0)

# --- load ---------------------------------------------------------------------

model = joblib.load(pkl_path)

with open(parameters_path) as f:
    params = yaml.safe_load(f)

# Resolve relative paths against the parameters.yml location
audio_files = [str((parameters_path.parent / p).resolve()) for p in params['audio_files']]

windower = AudioWindower(
        window_sec=params['window_sec'], 
        hop_sec=params['hop_sec']
    )

windows = windower.window_file(audio_files[0]) # Just do first file for now, I'll deal with multi file later
if not windows:
        print("No windows extracted from audio file.")
        sys.exit(1)

# --- embed and classify ---------------------------------------------------------------------

embedder = PerchEmbedder(batch_size=64)
embeddings = embedder.embed_windows(windows)
labels = model.predict(embeddings)

# --- save output ---------------------------------------------------------------------


start_secs = np.array([w.start_sec for w in windows])
    
npz_out.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    npz_out,
    labels=labels,
    embeddings=embeddings,
    start_secs=start_secs,
    source_files=np.array([audio_files[0]] * len(windows))
)

print(f"Classifier output saved to {npz_out}.")