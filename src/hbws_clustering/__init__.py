"""Humpback whale vocalization clustering pipeline using AVES embeddings."""

from __future__ import annotations

__all__ = [
    "AudioWindower",
    "ScoreGuidedWindower",
    "AvesEmbedder",
    "UmapReducer",
    "HdbscanClusterer",
    "ClusteringPipeline",
]

# Modules and the names they export, resolved lazily on first attribute access
# so that importing this package (e.g. via the CLI entry point) does not pull
# in torch / torchaudio / librosa / umap / hdbscan at startup.
_lazy: dict[str, str] = {
    "AudioWindower": "hbws_clustering.windowing",
    "ScoreGuidedWindower": "hbws_clustering.windowing",
    "AvesEmbedder": "hbws_clustering.embedding",
    "UmapReducer": "hbws_clustering.reduction",
    "HdbscanClusterer": "hbws_clustering.clustering",
    "ClusteringPipeline": "hbws_clustering.pipeline",
}


def __getattr__(name: str):
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name])
        obj = getattr(mod, name)
        # Cache in module globals so subsequent accesses are a plain dict lookup.
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
