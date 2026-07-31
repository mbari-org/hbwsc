"""Takes a new session directory and runs a specified trained classifier (pkl file). Gets embeddings from audio file, and outputs predictions in a new npz file.

Usage:
    uv run python scripts/predict.py <parameters.yml> <pkl> [out.npz]
    [FUTURE] add model parameter (logreg vs kmeans etc )

Arguments:
    parameters.yml     Target session's parameters
    pkl                Path to a model.pkl file produced by train_classifier.
    out.npz            Output model path (default: <target_session>/predictions/<pkl-stem>_predictions.npz).
    --embeddings-cache-dir  Directory to look up / save embeddings by config key
                            (e.g. PERCH_SWEEP/sweep_embed/).  Keyed by
                            win{w}_hop{h}_{embedder}_{padding}/embeddings.npy.
    [FUTURE] --embedding  type of embedding (Perch vs AVES).
"""
import argparse
import sys
import joblib
import yaml
from pathlib import Path
import numpy as np

from hbws_clustering.windowing import AudioWindower
from hbws_clustering.embedding import PerchEmbedder, AvesEmbedder
from hbws_clustering.clustering import HdbscanClusterer  # Forces CUDA libraries to load for TensorFlow

# --- args ---------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("parameters", type=Path, help="Path to a parameters.yml file")
parser.add_argument("pkl", type=Path, help="Path to a trained model .pkl file")
parser.add_argument("out_npz", type=Path, nargs="?", default=None, help="Output predictions .npz path")
parser.add_argument("--embeddings-cache-dir", type=Path, default=None,
                    help="Directory of pre-computed embeddings keyed by win/hop/embedder/padding subdirs")
args = parser.parse_args()

parameters_path = args.parameters
pkl_path = args.pkl
npz_out = args.out_npz if args.out_npz else parameters_path.parent / "predictions" / f"{pkl_path.parts[-4]}_{pkl_path.parts[-3]}.npz"

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

def _cache_key(params: dict) -> str:
    """Build the sweep_embed naming convention key from parameters."""
    w = params.get("window_sec", 0.5)
    h = params.get("hop_sec", 0.25)
    emb = params.get("embedder_type", "aves")
    pad = params.get("perch_padding", "repeat")
    return f"win{w}_hop{h}_{emb}_{pad}"


def _try_load(path: Path, n_windows: int) -> np.ndarray | None:
    """Load embeddings from path if it exists and shape matches."""
    if not path.exists():
        return None
    loaded = np.load(path)
    if len(loaded) == n_windows:
        print(f"  Loaded embeddings from {path}")
        return loaded
    print(f"  WARNING: {path} has {loaded.shape[0]} rows but expected {n_windows}. Skipping.")
    return None


embeddings = None

# Look for local embeddings
if embeddings is None:
    local_path = parameters_path.parent / "embeddings.npy"
    embeddings = _try_load(local_path, len(windows))

# Try the shared embeddings cache dir (e.g. PERCH_SWEEP/sweep_embed/)
if args.embeddings_cache_dir is not None:
    cache_path = args.embeddings_cache_dir / _cache_key(params) / "embeddings.npy"
    embeddings = _try_load(cache_path, len(windows))

# If no embeddings found, compute embeddings
if embeddings is None:
    print("Computing embeddings...")
    embedder_type = params.get('embedder_type', 'aves')
    if embedder_type == 'perch':
        embedder = PerchEmbedder(batch_size=params.get('batch_size', 64), padding=params.get('perch_padding', 'repeat'))
    else:
        embedder = AvesEmbedder(batch_size=params.get('batch_size', 16))
    embeddings = embedder.embed_windows(windows)
    
    # Save to local path
    local_path = parameters_path.parent / "embeddings.npy"
    print(f"Saving computed embeddings to {local_path}...")
    np.save(local_path, embeddings)
    
    # Also save to cache dir for future reuse
    if args.embeddings_cache_dir is not None:
        cache_path = args.embeddings_cache_dir / _cache_key(params) / "embeddings.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"Saved to embeddings cache: {cache_path}")

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