# hbwsc

Humpback whale vocalization clustering pipeline using [AVES](https://github.com/earthspecies/avex)
bioacoustic embeddings, UMAP dimensionality reduction, and HDBSCAN clustering.

Optionally integrates with the
[NOAA/Google Humpback Whale Song Detector](https://www.kaggle.com/models/google/humpback-whale/tensorFlow2/humpback-whale/1),
as exercised via <https://github.com/mbari-org/humpback-whale-song-detection>,
to restrict analysis to high-confidence whale song regions.

**Pipeline:** audio files → (score-guided) windows → AVES embeddings → UMAP projection → HDBSCAN clusters

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

### Plain windowing

Windows every second of every file — useful for short recordings or initial exploration.

```bash
# Single file, all defaults (2 s windows, no overlap, AVES-base-bio, 2-D UMAP)
uv run hbws-cluster MARS-20260330T000000Z-16kHz.wav

# Multiple files, 50% overlap, custom cluster size, save results
uv run hbws-cluster \
  MARS-20260330T000000Z-16kHz.wav \
  MARS-20260331T000000Z-16kHz.wav \
  --window-sec 2.0 \
  --hop-sec 1.0 \
  --min-cluster-size 10 \
  --output results.npz
```

### Score-guided windowing

Pass `--score-dir` to use pre-computed HWSD detection scores. The score file for each WAV is
located automatically from the filename date (`MARS-YYYYMMDDTHHMMSSZ-16kHz.wav` →
`{score-dir}/YYYY/MM/Scores-YYYYMMDD.npy`). Only seconds where the detector score exceeds
`--score-threshold` (default `0.7`) are extracted.

```bash
uv run hbws-cluster \
  --score-dir /mnt/PAM_Analysis/GoogleHumpbackModel/Scores \
  --score-threshold 0.7 \
  MARS-20260330T000000Z-16kHz.wav \
  MARS-20260331T000000Z-16kHz.wav \
  --window-sec 2.0 \
  --hop-sec 1.0 \
  --min-cluster-size 10 \
  --output results.npz
```

The 4+ GB day files are never loaded into memory in full — only the relevant slices are read.

### Inspect and plot results

```bash
just inspect-npz results.npz   # per-cluster count table with UMAP centroids
just plot-umap results.npz     # saves results.png
```

Or directly:

```python
import numpy as np

r = np.load("output/results.npz")
print(r["labels"])  # cluster ID per window (-1 = noise)
print(r["reduced"])  # (N, 2) UMAP coordinates
```

### All CLI options

| Option | Default | Description |
|---|---|---|
| `--score-dir` | — | Base directory for HWSD score `.npy` files; enables score-guided windowing |
| `--score-threshold` | `0.7` | Minimum HWSD score to include a second |
| `--window-sec` | `2.0` | Window duration in seconds |
| `--hop-sec` | `window-sec` | Step between windows (overlap when < `window-sec`) |
| `--sample-rate` | `16000` | Target sample rate for resampling |
| `--model` | `AVES_BASE_BIO` (GCS URL) | URL to a TorchAudio AVES checkpoint (`.pt`) |
| `--pooling` | `mean` | Embedding pooling: `mean` or `max` |
| `--batch-size` | `16` | AVES inference batch size — increase to 64–128 on GPU |
| `--umap-components` | `2` | UMAP output dimensions |
| `--umap-neighbors` | `15` | UMAP `n_neighbors` |
| `--min-cluster-size` | `5` | HDBSCAN `min_cluster_size` |
| `--embeddings-cache` / `-e` | — | Cache AVES embeddings to `.npy`; reloaded on subsequent runs |
| `--output` / `-o` | — | Save results as `.npz` file |

## Parameter guide

### Step 1 — Score-guided windowing

**`--score-threshold`** (default `0.7`)
The HWSD scores are one value per second in [0, 1]. Only seconds at or above the threshold are
kept. Raising it gives fewer, higher-confidence windows; lowering it includes more borderline
material.

**`--window-sec`** (default `2.0`)
The fundamental unit of analysis. Every window is exactly this many seconds of audio and produces
exactly one embedding vector. This choice involves a tradeoff:
- Too short: may clip individual calls; AVES has less acoustic context to work with
- Too long: may smear different call types together in one window; embeddings become less specific
- The right value depends on the temporal scale of the structure you want to cluster
  (individual units vs. phrases) and should be informed by domain knowledge and listening to the data

**`--hop-sec`** (default `window-sec`, i.e. no overlap)
Step between successive window starts. At `hop-sec=1.0` with `window-sec=2.0` windows overlap by
50%, so each point in the audio appears in roughly two windows. More overlap gives finer temporal
coverage but also more redundancy — adjacent windows will be very similar, which can inflate
cluster sizes without adding new information.

### Step 2 — AVES embeddings

Each window → one 768-dimensional vector. AVES was pretrained on bioacoustic audio so it captures
acoustic structure meaningful to animal vocalizations. Nothing to tune here except `--batch-size`
for throughput (default 16 on CPU; 64–128 on GPU).

### Step 3 — UMAP

**`--umap-components`** (default `2`)
Projection target dimensionality. 2 is for visualization. If you want clustering to operate in a
richer space you could use 5–10 and skip the 2-D scatter plot, or run two passes: 2-D for
visualization and higher-D to feed HDBSCAN.

**`--umap-neighbors`** (default `15`)
Controls local vs. global structure. Small values (5–10) preserve fine local clusters; large
values (30–50) pull the global shape together but may merge distinct call types (per the
[UMAP documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html)). The default
of 15 is a reasonable starting point; tuning is worthwhile once you have a baseline.

### Step 4 — HDBSCAN

**`--min-cluster-size`** (default `5`)
The minimum number of windows to form a cluster — anything smaller is labeled noise (`-1`). The
right value depends on the dataset size and how rare the call types you care about are. At the
default of 5 with tens of thousands of windows the algorithm will be permissive; raising it will
produce fewer, larger clusters. What constitutes a meaningful minimum is a domain question.

### The interaction that matters most

`--window-sec` and `--hop-sec` determine what AVES sees, which determines what UMAP has to work
with, which determines what HDBSCAN can separate. If two call types are acoustically similar
within 2 s but distinguishable over longer context, no amount of UMAP/HDBSCAN tuning will
separate them — you would need a longer window. Conversely, if you are trying to distinguish short
stereotyped units (< 1 s), a 2-second window dilutes the signal with surrounding context.

Once you have results, the key question to ask is: **do the clusters make acoustic sense?** —
which means listening to a few windows from each cluster.

## GPU support

The embedder auto-detects CUDA (`"cuda" if torch.cuda.is_available() else "cpu"`), so no code
changes are needed on a GPU machine. The only requirement is installing the CUDA-enabled PyTorch
build instead of the default CPU-only one.

Add the following to `pyproject.toml` (replace `cu124` with the CUDA version reported by
`nvidia-smi` on the target machine — common values: `cu118`, `cu121`, `cu124`):

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchaudio = { index = "pytorch-cu124" }
```

Then run `uv sync` to reinstall with the CUDA build. After that, increase the batch size for
better GPU utilisation:

```bash
uv run hbws-cluster --batch-size 64 --embeddings-cache output/embeddings.npy ...
```

> **Note:** Because the index config is hardware-specific, it is not committed to the repo.
> CPU builds work on all machines; apply the override locally on GPU hosts.

## Python API

### Plain windowing

```python
from hbws_clustering import ClusteringPipeline, AudioWindower, AvesEmbedder, UmapReducer, HdbscanClusterer
from hbws_clustering.embedding import AVES_BASE_BIO

pipe = ClusteringPipeline(
    windower=AudioWindower(window_sec=2.0, hop_sec=1.0),
    embedder=AvesEmbedder(model_url=AVES_BASE_BIO, pooling="mean", batch_size=32),
    reducer=UmapReducer(n_components=2, n_neighbors=30, min_dist=0.05, metric="cosine"),
    clusterer=HdbscanClusterer(min_cluster_size=15, min_samples=5),
)

result = pipe.run(["MARS-20260330T000000Z-16kHz.wav", "MARS-20260331T000000Z-16kHz.wav"])
print(result.cluster_summary())
# e.g. {-1: 81, 0: 33, 1: 486}
```

### Score-guided windowing

```python
from hbws_clustering import ClusteringPipeline, ScoreGuidedWindower, AudioWindower

pipe = ClusteringPipeline(
    windower=ScoreGuidedWindower(
        windower=AudioWindower(window_sec=2.0, hop_sec=1.0),
        threshold=0.7,
    ),
)

result = pipe.run_scored([
    ("MARS-20260330T000000Z-16kHz.wav", "Scores/2026/03/Scores-20260330.npy"),
    ("MARS-20260331T000000Z-16kHz.wav", "Scores/2026/03/Scores-20260331.npy"),
])
print(result.cluster_summary())
```

### Inspecting results

```python
# Which windows belong to cluster 0, and when do they occur?
windows_c0 = [(w.source_file, w.start_sec, w.end_sec)
              for w, l in zip(result.windows, result.labels) if l == 0]
print(windows_c0[:5])
```

## Project structure

```
hbws_clustering/
├── pyproject.toml
├── justfile                   # recipes: run, inspect-npz, plot-umap
├── scripts/
│   ├── inspect_npz.py         # per-cluster summary table
│   └── plot_umap.py           # UMAP scatter plot
├── src/hbws_clustering/
│   ├── __init__.py
│   ├── __main__.py            # hbws-cluster CLI (typer)
│   ├── windowing.py           # AudioWindower, ScoreGuidedWindower
│   ├── embedding.py           # AvesEmbedder — TorchAudio AVES inference
│   ├── reduction.py           # UmapReducer
│   ├── clustering.py          # HdbscanClusterer
│   └── pipeline.py            # ClusteringPipeline — run() and run_scored()
└── tests/
    ├── test_windowing.py
    ├── test_score_guided_windowing.py
    └── test_reduction_clustering.py
```

## Design notes

- **AVES model:** originally from the [Earth Species Project](https://github.com/earthspecies/aves),
  now maintained as part of [avex](https://github.com/earthspecies/avex). Checkpoints are loaded
  directly from Google Cloud Storage via TorchAudio (no `avex` install required — `avex` pulls in
  TensorFlow/Keras for its other backends and is not worth the dependency cost here).
  Two variants are available via module-level constants:
    - `AVES_BASE_BIO` — pretrained on bioacoustic audio (default, recommended for wildlife)
    - `AVES_BASE_ALL` — pretrained on a broader audio corpus
- **Score-guided windowing:** uses HWSD per-second scores (`Scores-YYYYMMDD.npy`) to skip ambient
  noise. Day files are read with `soundfile` seek — the 4+ GB WAV is never fully loaded into memory.
- **Pooling:** mean-pools AVES frame embeddings → one fixed-size vector per window. `max` also supported.
- **Windowing:** configurable `window_sec` / `hop_sec`, resamples to 16 kHz, zero-pads the final
  partial window when it meets `min_window_sec`.
- **Output:** `--output results.npz` saves embeddings, UMAP coordinates, cluster labels, and soft
  membership probabilities.

## Development

```bash
uv run pytest -v                                  # all tests
uv run pytest tests/test_windowing.py             # windowing only (fast, no GPU/network)
uv run pytest tests/test_score_guided_windowing.py
uv run ruff check src/                            # lint
```
