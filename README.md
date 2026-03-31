# hbws-clustering

Humpback whale vocalization clustering pipeline using [AVES](https://github.com/earthspecies/aves) bioacoustic embeddings,
UMAP dimensionality reduction, and HDBSCAN clustering.

**Pipeline:** audio files → fixed-length windows → AVES embeddings → UMAP projection → HDBSCAN clusters

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mbari-org/hbws_clustering
cd hbws_clustering
uv sync
```

> **First run:** `AvesEmbedder` downloads the AVES checkpoint (~360 MB) from Google Cloud Storage
> on first use and caches it under `$(python -c "import torch; print(torch.hub.get_dir())")/aves/`.
> Subsequent runs are instant.

## CLI usage

```bash
# Single file, all defaults (2 s windows, no overlap, AVES-base-bio, 2-D UMAP)
uv run hbws-cluster song.wav

# Multiple files, 50% overlap, custom cluster size, save results
uv run hbws-cluster dive1.wav dive2.wav dive3.wav \
  --window-sec 2.0 \
  --hop-sec 1.0 \
  --min-cluster-size 10 \
  --output results.npz
```

Inspect saved results:

```python
import numpy as np
r = np.load("results.npz")
print(r["labels"])    # cluster ID per window (-1 = noise)
print(r["reduced"])   # (N, 2) UMAP coordinates
```

All CLI options:

| Option | Default | Description |
|---|---|---|
| `--window-sec` | `2.0` | Window duration in seconds |
| `--hop-sec` | `window-sec` | Step between windows (overlap when < window-sec) |
| `--sample-rate` | `16000` | Target sample rate for resampling |
| `--model` | `AVES_BASE_BIO` (GCS URL) | URL to a TorchAudio AVES checkpoint (`.pt`) |
| `--pooling` | `mean` | Embedding pooling: `mean` or `max` |
| `--umap-components` | `2` | UMAP output dimensions |
| `--umap-neighbors` | `15` | UMAP `n_neighbors` |
| `--min-cluster-size` | `5` | HDBSCAN `min_cluster_size` |
| `--output` / `-o` | — | Save results as `.npz` file |

## Python API

```python
from hbws_clustering import (
    ClusteringPipeline, AudioWindower, AvesEmbedder, UmapReducer, HdbscanClusterer
)
from hbws_clustering.embedding import AVES_BASE_BIO

# Fully custom config
pipe = ClusteringPipeline(
    windower=AudioWindower(window_sec=2.0, hop_sec=1.0),
    embedder=AvesEmbedder(model_url=AVES_BASE_BIO, pooling="mean", batch_size=32),
    reducer=UmapReducer(n_components=2, n_neighbors=30, min_dist=0.05, metric="cosine"),
    clusterer=HdbscanClusterer(min_cluster_size=15, min_samples=5),
)

result = pipe.run(["dive1.wav", "dive2.wav", "dive3.wav"])

print(result.cluster_summary())
# e.g. {-1: 12, 0: 47, 1: 38, 2: 21}  → 3 clusters + 12 noise windows

# Inspect windows belonging to a specific cluster
windows_c0 = [w for w, l in zip(result.windows, result.labels) if l == 0]
print(windows_c0[0].source_file, windows_c0[0].start_sec)
```

### Plot the UMAP projection

```python
import matplotlib.pyplot as plt

xy = result.reduced    # (N, 2)
sc = plt.scatter(xy[:, 0], xy[:, 1], c=result.labels, cmap="tab10", s=5, alpha=0.7)
plt.colorbar(sc, label="cluster")
plt.title("AVES embeddings — UMAP projection")
plt.savefig("umap.png", dpi=150)
```

### Window a numpy array directly

```python
import numpy as np
from hbws_clustering import AudioWindower

audio = np.random.randn(160_000).astype("float32")  # 10 s @ 16 kHz
windows = AudioWindower(window_sec=2.0, hop_sec=0.5, target_sr=None).window_array(audio, 16_000)
print(len(windows))                          # 17 windows
print(windows[0].start_sec, windows[0].end_sec)  # 0.0  2.0
```

## Project structure

```
hbws_clustering/
├── pyproject.toml
├── src/hbws_clustering/
│   ├── __init__.py
│   ├── __main__.py     # hbws-cluster CLI (typer)
│   ├── windowing.py    # AudioWindower — load + slice audio into windows
│   ├── embedding.py    # AvesEmbedder — HuggingFace AVES inference
│   ├── reduction.py    # UmapReducer — UMAP projection
│   ├── clustering.py   # HdbscanClusterer — HDBSCAN fit + labels
│   └── pipeline.py     # ClusteringPipeline — orchestrates all 4 steps
└── tests/
    ├── test_windowing.py
    └── test_reduction_clustering.py
```

## Design notes

- **AVES model:** checkpoints are hosted by the [Earth Species Project](https://github.com/earthspecies/aves) on Google Cloud Storage.
  Two variants are available via module-level constants:
    - `AVES_BASE_BIO` — pretrained on bioacoustic audio (default, recommended for wildlife)
    - `AVES_BASE_ALL` — pretrained on a broader audio corpus
- **Pooling:** mean-pools AVES frame embeddings → one fixed-size vector per window. `max` pooling also supported.
- **Windowing:** configurable `window_sec` / `hop_sec`, resamples to 16 kHz, zero-pads the final partial window when it meets `min_window_sec`.
- **Output:** `--output results.npz` saves embeddings, UMAP coordinates, cluster labels, and soft membership probabilities.

## Development

```bash
uv run pytest -v                        # all tests
uv run pytest tests/test_windowing.py  # windowing only (fast, no GPU/network)
uv run ruff check src/                  # lint
```
