# hbwsc

Humpback whale vocalization analysis using [AVES](https://github.com/earthspecies/aves)
bioacoustic embeddings, UMAP dimensionality reduction, and HDBSCAN clustering.

Integrates with the
[NOAA/Google Humpback Whale Song Detector](https://www.kaggle.com/models/google/humpback-whale/tensorFlow2/humpback-whale/1),
as exercised with <https://github.com/mbari-org/humpback-whale-song-detection>,
to restrict the clustering to high-confidence whale song regions.

**Pipeline:** audio files → score-guided windows → AVES embeddings → UMAP projection → HDBSCAN clusters

> [!NOTE]
> This is an exploratory exercise!
> The documentation and parameter guidance in this README are the result of exchanges
> with Claude (Anthropic) while building the code and conducting initial exploration.
> Various aspects have been revised during these exchanges,
> but no exhaustive review or independent validation has been performed.

## Running

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mbari-org/hbwsc
cd hbwsc
uv sync
```

> **First run:** `AvesEmbedder` downloads the AVES checkpoint (~360 MB) from Google Cloud Storage
> on first use and caches it under `$(python -c "import torch; print(torch.hub.get_dir())")/aves/`.

### Plain windowing

Windows every second of every file — useful for short recordings or initial exploration.

```bash
# Single file, all defaults
uv run hbws-cluster MARS-20260330T000000Z-16kHz.wav

# Multiple files, 50% overlap, custom cluster size, save results
uv run hbws-cluster \
  MARS-20260330T000000Z-16kHz.wav \
  MARS-20260331T000000Z-16kHz.wav \
  --window-sec 0.5 \
  --hop-sec 0.25 \
  --min-cluster-size 10 \
  --output results.npz
```

### Score-guided windowing

Pass `--score-dir` to use pre-computed HWSD detection scores. The score file for each WAV is
located automatically from the filename date
(`MARS-YYYYMMDDTHHMMSSZ-16kHz.wav` → `{score-dir}/YYYY/MM/Scores-YYYYMMDD.npy`).
Only seconds where the detector score exceeds `--score-threshold` (default `0.7`) are extracted.

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

### Inspect and plot results

```bash
just inspect-npz results.npz   # per-cluster count table with UMAP centroids
just plot-umap results.npz     # saves results.png
```

### All CLI options

| Option | Default | Description |
|---|---|---|
| `--score-dir` | — | Base directory for HWSD score `.npy` files; score file auto-located from WAV filename date |
| `--score-file` | — | Explicit path to a score `.npy` file; use when filename doesn't follow the MARS convention (requires exactly one audio file) |
| `--score-threshold` | `0.7` | Minimum HWSD score to include a second |
| `--window-sec` | `2.0` | Window duration in seconds |
| `--hop-sec` | `window-sec` | Step between windows (overlap when < `window-sec`) |
| `--sample-rate` | `16000` | Target sample rate for resampling |
| `--model` | `AVES_BASE_BIO` (GCS URL) | URL to a TorchAudio AVES checkpoint (`.pt`) |
| `--pooling` | `mean` | Embedding pooling: `mean` or `max` |
| `--batch-size` | `16` | AVES inference batch size — increase to 64–128 on GPU |
| `--umap-components` | `2` | UMAP output dimensions for visualization (2-D scatter plot) |
| `--umap-cluster-components` | `10` | UMAP output dimensions fed to HDBSCAN; set equal to `--umap-components` to skip the second UMAP pass |
| `--umap-neighbors` | `15` | UMAP `n_neighbors` (shared by both UMAP passes) |
| `--min-cluster-size` | `5` | HDBSCAN `min_cluster_size` |
| `--embeddings-cache` / `-e` | — | Cache AVES embeddings to `.npy`; reloaded on subsequent runs |
| `--output` / `-o` | — | Save results as `.npz` file |

## Parameters

### Step 1 — Score-guided windowing

**`--score-threshold`**
The HWSD scores are one value per second in [0, 1]. Only seconds at or above the threshold are kept.

**`--window-sec`**
The fundamental unit of analysis. Every window is exactly this many seconds of audio and produces
exactly one embedding vector. This choice involves a tradeoff:
- Too short: may clip individual calls; AVES has less acoustic context to work with
- Too long: may smear different call types together in one window; embeddings become less specific
- The right value depends on the temporal scale of the structure you want to cluster
  and should be informed by domain knowledge and listening to the data

**`--hop-sec`** (default `window-sec`, i.e. no overlap)
Step between successive window starts. More overlap gives finer temporal coverage but also more redundancy:
adjacent windows will be very similar, which can inflate cluster sizes without adding new information.

### Step 2 — AVES embeddings

Each window becomes one 768-dimensional vector. AVES was pretrained on bioacoustic audio so it captures
acoustic structure meaningful to animal vocalizations.
Nothing to tune here except `--batch-size` for throughput (default 16 on CPU; 64–128 on GPU).

> **Note on input audio quality.** AVES (like its parent HuBERT) takes raw waveform with no
> built-in noise-reduction or loudness-normalization step. On variable-SNR field recordings,
> this means quiet and loud realizations of the same call can land in different regions of
> the embedding space. If your downstream goal is sensitive to that (e.g. unit-type
> discovery), consider whether to apply per-file or per-window pre-processing — loudness
> normalization (RMS or peak), denoising, or PCEN — *before* feeding audio into this
> pipeline. The pipeline itself is intentionally agnostic about this choice; it depends on
> the embedder you use (some embedders are trained with noise augmentation and don't need
> it) and on whether you favor cluster purity (helps) over detection coverage
> (slightly hurts).

### Step 3 — UMAP

The pipeline runs two independent UMAP projections by default:

- **`--umap-cluster-components`** (default `10`) — high-dimensional projection fed to HDBSCAN.
  10 dimensions retains far more structure than 2-D while still being much cheaper to cluster than
  the raw 768-D embeddings. Values in the 5–20 range are reasonable; higher gives the clusterer
  more to work with but increases UMAP runtime.

- **`--umap-components`** (default `2`) — 2-D projection used only for the scatter plot.
  Visualization and clustering are purposely decoupled: the 2-D plot gives an intuitive picture
  of the embedding space, but clustering on 2-D discards too much information.

Set `--umap-cluster-components` equal to `--umap-components` (e.g., both `2`) to collapse back
to a single UMAP pass — useful for quick experiments or when runtime is the bottleneck.

**`--umap-neighbors`** (default `15`)
Controls local vs. global structure. Applies to both UMAP passes. Small values (5–10) preserve
fine local clusters; large values (30–50) pull the global shape together but may merge distinct
call types (per the
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

## Intended analysis levels

The clustering pipeline is intended to support the definition of a **unit vocabulary**,
that is, the set of atomic elements that can be used as a basis to characterize humpback whale song.

The window length should be chosen such that it allows to capture the smallest unit
in the training data.

Once good quality unit types are obtained according to plausible clustering metrics,
higher-level structure can be recovered.

While the score-guided windowing discards low-confidence detection regions, the temporal gaps
are structural, that is, inter-unit, inter-phrase, and inter-song silences carry information
needed for segmentation at higher levels. Unit clustering can be done entirely on detected
regions, but phrase/theme/song analysis will require combining the unit-type labels and
timestamps with the gap information available in the score files.

HDBSCAN's soft membership probabilities (`result.probabilities`, saved in the `.npz`) give a
per-window confidence score that can inform boundary decisions when unit-type assignment is
ambiguous.

## Hyperparameter tuning workflow

### Role of each component

**AVES** does the acoustic heavy lifting: it converts each audio window into a
768-dimensional vector where acoustic similarity is reflected in geometric proximity.
Because it was pretrained on bioacoustic audio, there is good reason to expect it encodes
features meaningful to animal vocalizations. The quality of the clustering depends directly
on how well AVES captures the acoustic structure relevant to the target call types.

**UMAP** addresses a practical necessity: HDBSCAN breaks down in 768 dimensions due to the
curse of dimensionality (distances become nearly indistinguishable, so no density structure
can be found). UMAP compresses the embedding space while preserving neighborhood relationships,
giving HDBSCAN a tractable input. Visualization (2-D) and clustering (default 10-D) are
intentionally decoupled — the 2-D projection gives an intuitive picture of the space but
discards too much information to cluster on directly.

**HDBSCAN** discovers clusters without requiring the number of clusters to be specified in
advance, and without assuming spherical shapes. The number of clusters is an output, not an
input — which is the appropriate posture when the unit vocabulary is unknown.

### Tuning `--min-cluster-size`

`min_cluster_size` is the primary tuning parameter. It sets a floor on what counts as a
real cluster: anything smaller is labeled noise. It encodes a prior about the minimum
prevalence of a unit type worth retaining in the vocabulary.

The specific value is not knowable in advance without ground truth, but it does not need to
be pinned precisely. The intended workflow is:

1. **Sweep** a range of candidate values (e.g. with `scripts/sweep.py`) and examine how
   cluster count, noise fraction, and DBCV vary.
2. **Identify the stable region** — values where DBCV is positive and cluster count is
   plausible for the recording (a single song session is unlikely to contain dozens of
   distinct unit types).
3. **Validate acoustically** — listen to samples from each cluster (e.g. with
   `scripts/export_cluster.py`). If two clusters sound like the same unit type, raise
   `min_cluster_size`. If a cluster sounds like a mixture, lower it.

Domain judgment makes the final call; the sweep narrows the search to a small set of
candidates worth listening to.

### Choosing a clustering metric

The sweep script reports **DBCV** (Density-Based Clustering Validation), which measures
how compact and well-separated clusters are in terms of density. It is the appropriate
metric here because HDBSCAN is density-based and finds clusters of arbitrary shape —
classical metrics such as inertia, silhouette score, or Davies-Bouldin implicitly assume
spherical clusters and would give misleading guidance in this setting.

DBCV range is [-1, 1]: values near 1 indicate tight, well-separated clusters; negative
values indicate that cluster boundaries cut through dense regions (a sign of over- or
under-fragmentation).

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

- **AVES model:** maintained as part of [avex](https://github.com/earthspecies/avex). 
  Checkpoints are loaded directly from Google Cloud Storage via TorchAudio.
  Two variants are available via module-level constants:
    - `AVES_BASE_BIO` — pretrained on bioacoustic audio (default, recommended for wildlife)
    - `AVES_BASE_ALL` — pretrained on a broader audio corpus
- **Score-guided windowing:** uses HWSD per-second scores (`Scores-YYYYMMDD.npy`)
- **Pooling:** mean-pools AVES frame embeddings → one fixed-size vector per window. `max` also supported.
- **Windowing:** configurable `window_sec` / `hop_sec`, resamples to 16 kHz, zero-pads the final
  partial window when it meets `min_window_sec`.
- **Output:** `--output results.npz` saves embeddings, UMAP coordinates, cluster labels, and soft
  membership probabilities.

## Development

```bash
uv run pytest
uv run ruff check src
```
