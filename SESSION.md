# Session-based workflow

A **session** is a directory that groups all inputs, parameters, and outputs for
a set of clustering runs on the same audio recording(s) with the same windowing
settings. The directory name encodes the key context; files within it use simple
names since the parent provides the rest.

## Directory layout

```
experiments/
  song_20161221_w0.5_h0.25/     ← session dir (base + window + hop)
    parameters.yml               ← all inputs and parameters (edit before running)
    embeddings.npy               ← AVES embeddings, shared across all mcs runs
    sweep/
      results.csv                ← hyperparameter sweep table
    mcs100/                      ← one subdir per min_cluster_size value
      results.npz
      umap.png
      timeline.png
      raven.txt
      raven_aggregate.txt
      clusters/
        cluster_0/
        cluster_1/
        noise/
        ...
    mcs200/
      ...
```

## Step-by-step

### 1. Create a session

```bash
just new-session experiments/song_20161221 0.5 0.25
```

Creates `experiments/song_20161221_w0.5_h0.25/parameters.yml` with `window_sec`
and `hop_sec` pre-filled. Open it and set at minimum:

- `audio_files` — paths to the WAV file(s);
  relative paths are resolved relative to the session directory
- `score_file` or `score_dir` — for score-guided windowing
  (comment out both for plain windowing)
- `min_cluster_size` — default value used by `run-session` when no mcs is given

### 2. Run the pipeline

```bash
just run-session experiments/song_20161221_w0.5_h0.25
```

Uses `min_cluster_size` from `parameters.yml`. Results go to `mcs<N>/results.npz`.
AVES embeddings are cached at the session root (`embeddings.npy`) and reused on
subsequent runs, so only UMAP and HDBSCAN are re-run when changing `mcs`.

To override `min_cluster_size` without editing `parameters.yml`:

```bash
just run-session experiments/song_20161221_w0.5_h0.25 200
```

### 3. Sweep hyperparameters

```bash
just sweep-session experiments/song_20161221_w0.5_h0.25
```

Requires `embeddings.npy` to exist (i.e., `run-session` must have been called at least once).
Sweeps the UMAP dimension and mcs combinations defined in `parameters.yml` under `sweep_dims`
and `sweep_mcs`. Results go to `sweep/results.csv`.

Use the sweep to identify promising `min_cluster_size` values before committing
to a full analysis. See the **Hyperparameter tuning workflow** section in README.md.

### 4. Analyze results

```bash
just analyze-session experiments/song_20161221_w0.5_h0.25 100
```

Runs all analysis steps for `mcs100/`:

| Output | Description |
|---|---|
| `umap.png` | 2-D UMAP scatter plot coloured by cluster |
| `timeline.png` | Cluster labels plotted over time |
| `raven.txt` | Raven Pro selection table (one row per window) |
| `raven_aggregate.txt` | Same, with consecutive same-cluster windows merged |
| `clusters/cluster_N/` | Up to `n_cluster_samples` WAV snippets per cluster |

### 5. Inspect cluster summary

```bash
just inspect-session experiments/song_20161221_w0.5_h0.25 100
```

Prints a per-cluster count table with UMAP centroids to the terminal.

## parameters.yml reference

```yaml
# Audio input — one or more WAV files
audio_files:
  - /path/to/recording.wav

# Score-guided windowing — choose one or neither
# score_file: /path/to/scores.npy       # explicit score file (single WAV only)
# score_dir:  /path/to/scores/base/dir  # auto-locate by date from WAV filename

score_threshold: 0.7

# Windowing
window_sec: 0.5
hop_sec: 0.25
sample_rate: 16000

# UMAP
umap_components: 2           # for visualization (2-D scatter plot)
umap_cluster_components: 10  # for HDBSCAN clustering
umap_neighbors: 15

# HDBSCAN
min_cluster_size: 100        # default; can be overridden on the run-session command line

# Hyperparameter sweep
sweep_dims: "2,5,10,15,20,30"
sweep_mcs: "50,100,200"

# Audio export (samples per cluster for analyze-session)
n_cluster_samples: 10
```

## Tips

- **Different window sizes** → create a new session with a different `window_sec`/`hop_sec`;
  the directory name keeps them distinct and embeddings are not shared across sessions.
- **Same audio, different UMAP settings** → edit `parameters.yml` and re-run; embeddings
  are reused automatically.
- **Raven Pro** — open the WAV file, then load `raven.txt` or `raven_aggregate.txt`
  via *File → Open Selection Table*. The `Cluster` column can be used to filter or
  colour selections. `Score` is the HDBSCAN soft membership probability.
