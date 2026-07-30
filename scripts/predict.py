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
from hbws_clustering.clustering import HdbscanClusterer  # Forces CUDA libraries to load for TensorFlow

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

windows = []
for af in audio_files:
    windows.extend(windower.window_file(af))

if not windows:
    print("No windows extracted from audio files.")
    sys.exit(1)

# --- embed and classify ---------------------------------------------------------------------

embeddings_file = parameters_path.parent / "embeddings.npy"
embeddings = None
if embeddings_file.exists():
    print(f"Loading existing embeddings from {embeddings_file}...")
    loaded_emb = np.load(embeddings_file)
    if len(loaded_emb) == len(windows):
        embeddings = loaded_emb
    else:
        print(f"WARNING: existing embeddings shape {loaded_emb.shape} does not match extracted windows ({len(windows)}). Re-embedding...")

if embeddings is None:
    print("Computing embeddings...")
    embedder_type = params.get('embedder_type', 'aves')
    if embedder_type == 'perch':
        embedder = PerchEmbedder(batch_size=params.get('batch_size', 64), padding=params.get('perch_padding', 'repeat'))
    else:
        embedder = AvesEmbedder(batch_size=params.get('batch_size', 16))
    embeddings = embedder.embed_windows(windows)
    
    print(f"Saving computed embeddings to {embeddings_file}...")
    np.save(embeddings_file, embeddings)

labels = model.predict(embeddings)

# --- save output ---------------------------------------------------------------------


start_secs = np.array([w.start_sec for w in windows])
    
npz_out.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    npz_out,
    labels=labels,
    embeddings=embeddings,
    start_secs=start_secs,
    source_files=np.array([w.source_file.name for w in windows])
)

print(f"Classifier output saved to {npz_out}.")