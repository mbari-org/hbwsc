# GPU Acceleration (Linux Only)

The clustering pipeline supports GPU-accelerated **UMAP** and **HDBSCAN** via [cuML](https://github.com/rapidsai/cuml) (part of NVIDIA RAPIDS). If cuML is installed, the pipeline automatically detects and uses it. If not, it falls back to the CPU versions (`umap-learn` and `hdbscan`) and prints a warning.

## Installation

Because cuML relies on heavy CUDA C++ libraries and has strict dependency requirements (like `pandas < 2.4`), it is placed in an optional `gpu` dependency group in `pyproject.toml`.

To use GPU acceleration on a Linux machine, simply install the project with the `gpu` group enabled:

```bash
uv sync --group gpu
```

This will automatically pull `cuml-cu12` and its required NVIDIA libraries from the `https://pypi.nvidia.com` index.

## Verification

When you run the pipeline on a machine with cuML installed, you should see the backend confirmed in the standard output:

```text
Step 3/4: UMAP dimensionality reduction (10-D for clustering, 2-D for visualization)...
  UMAP backend: cuml
  ...
Step 4/4: HDBSCAN clustering...
  HDBSCAN backend: cuml
```
