import argparse
import numpy as np
from pathlib import Path
import sys

def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        print(f"Error: {path} not found.")
        sys.exit(1)
        
    print(f"Loading {path.name}...")
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        r = np.load(path, allow_pickle=False)
        if "embeddings" not in r:
            print(f"Error: No 'embeddings' array found in {path}")
            sys.exit(1)
        return r["embeddings"]
    else:
        print(f"Error: Unsupported file format {path.suffix}. Expected .npy or .npz")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Compare two embedding files (CPU vs GPU)")
    parser.add_argument("file1", type=Path, help="Path to first .npy or .npz file")
    parser.add_argument("file2", type=Path, help="Path to second .npy or .npz file")
    args = parser.parse_args()

    emb1 = load_embeddings(args.file1)
    emb2 = load_embeddings(args.file2)

    # Quick and dirty fix on now including the last small embedding
    if emb1.shape != emb2.shape:
        min_len = min(emb1.shape[0], emb2.shape[0])
        print(f"\nShape mismatch! Truncating to length: {min_len}")
        print(f"File 1 original shape: {emb1.shape}")
        print(f"File 2 original shape: {emb2.shape}")
        emb1 = emb1[:min_len]
        emb2 = emb2[:min_len]

    print(f"\nComparing arrays of shape {emb1.shape}...")

    # Calculate differences
    abs_diff = np.abs(emb1 - emb2)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    # Calculate how many elements are exactly identical
    exact_matches = np.sum(emb1 == emb2)
    total_elements = emb1.size
    match_pct = (exact_matches / total_elements) * 100

    print("\n--- Difference Statistics ---")
    print(f"Maximum absolute difference:  {max_diff:.8e}")
    print(f"Mean absolute difference:     {mean_diff:.8e}")
    print(f"Exact floating-point matches: {exact_matches}/{total_elements} ({match_pct:.2f}%)")

if __name__ == "__main__":
    main()
