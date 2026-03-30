## Project structure

```
  hbws_clustering/
  ├── pyproject.toml            # uv project, hatchling build, CLI script
  ├── src/hbws_clustering/
  │   ├── __init__.py
  │   ├── __main__.py           # hbws-cluster CLI (typer)
  │   ├── windowing.py          # AudioWindower — load + slice audio into windows
  │   ├── embedding.py          # AvesEmbedder — HuggingFace AVES inference
  │   ├── reduction.py          # UmapReducer — UMAP projection
  │   ├── clustering.py         # HdbscanClusterer — HDBSCAN fit + labels
  │   └── pipeline.py           # ClusteringPipeline — orchestrates all 4 steps
  └── tests/
      ├── test_windowing.py     # 7 tests (no network/model required)
      └── test_reduction_clustering.py  # 7 tests (no network/model required)
```

## Key design decisions

- AVES model: defaults to m-a-p/AVES-base-bio (pretrained on bioacoustic data). 
  Swap to AVES-base-all via --model.
- Pooling: mean-pools AVES frame embeddings → one vector per window. 
  pooling max is also supported.
- Windowing: configurable window_sec / hop_sec, resamples to 16 kHz, 
  zero-pads the last partial window if it meets min_window_sec.
- Output: --output results.npz saves embeddings, reduced coords, labels, and probabilities.

## Usage

Run the pipeline on WAV files:
```bash
uv run hbws-cluster call1.wav call2.wav --window-sec 2.0 --min-cluster-size 10 -o results.npz
```

Run tests
```bash
uv run pytest
```

Programmatic use
```python
from hbws_clustering import ClusteringPipeline
result = ClusteringPipeline().run(["call1.wav"])
print(result.cluster_summary())
```