# GPU Acceleration (Linux Only)

The clustering pipeline supports GPU-accelerated **UMAP** and **HDBSCAN** via [cuML](https://github.com/rapidsai/cuml) (part of NVIDIA RAPIDS). If cuML is installed, the pipeline automatically detects and uses it. If not, it falls back to the CPU versions (`umap-learn` and `hdbscan`) and prints a warning.

## Installation

Because cuML relies on a massive ~3GB tree of NVIDIA CUDA C++ libraries, it is placed in an optional `gpu` dependency group in `pyproject.toml`. This creates a dual-environment setup:

* **On Mac (CPU-only):** Run `uv sync`. This ignores the RAPIDS packages entirely, giving you a fast, lightweight environment.
* **On Linux (GPU-accelerated):** Run `uv sync --group gpu`. This tells `uv` to include the extra GPU dependencies and pulls `cuml-cu12` (and its required NVIDIA libraries) from the `https://pypi.nvidia.com` index.

## Verification

When you run the pipeline on a machine with cuML installed, you should see the backend confirmed in the standard output:

```text
Step 3/4: UMAP dimensionality reduction (10-D for clustering, 2-D for visualization)...
  UMAP backend: cuml
  ...
Step 4/4: HDBSCAN clustering...
  HDBSCAN backend: cuml
```
