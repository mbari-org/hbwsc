"""Train a classifier over previous HDBSCAN cluster assignments for inference on other audio files.

Takes an existing successfully clustered results.npz and creates a pickled trained model file 

Usage:
    uv run python scripts/train_classifier.py <npz> [out.pkl]
    [FUTURE] add model parameter (logreg vs kmeans etc )

Arguments:
    npz                Path to a results .npz file produced by hbws-cluster.
    out_pkl            Output model path (default: models/<npz-stem>_model.pkl).
    [FUTURE] --classifier  type of classifier (logreg, kmeans, random forest ...).
"""

import argparse
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# --- args ---------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("npz", type=Path, help="Path to a results .npz file produced by hbws-cluster")
parser.add_argument("out_pkl", type=Path, nargs="?", default=None, help="Output model path")
parser.add_argument("--drop-noise", action="store_true", help="Drop labels with -1 (noise) before training")
args = parser.parse_args()

npz_path = args.npz
model_out = args.out_pkl if args.out_pkl else npz_path.parent / "models" / f"{npz_path.stem}_model.pkl"

if model_out.exists():
    print(f"Model {model_out} already exists. Skipping training.")
    sys.exit(0)

# --- load ---------------------------------------------------------------------

r = np.load(npz_path, allow_pickle=False)
X_train = r["embeddings"]
y_train = r["labels"]

if args.drop_noise:
    valid_idx = y_train != -1
    X_train = X_train[valid_idx]
    y_train = y_train[valid_idx]
    print(f"Dropped noise: keeping {len(y_train)} / {len(r['labels'])} windows.")

# --- model training ------------------------------------------------------

# Logreg right now, as it works best on Perch, I assume will do poorly on AVES's embedding space
logreg = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logreg.fit(X_train, y_train)

model_out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(logreg, model_out)