"""Humpback whale vocalization clustering pipeline using AVES embeddings."""

from hbws_clustering.windowing import AudioWindower, ScoreGuidedWindower
from hbws_clustering.embedding import AvesEmbedder
from hbws_clustering.reduction import UmapReducer
from hbws_clustering.clustering import HdbscanClusterer
from hbws_clustering.pipeline import ClusteringPipeline

__all__ = [
    "AudioWindower",
    "ScoreGuidedWindower",
    "AvesEmbedder",
    "UmapReducer",
    "HdbscanClusterer",
    "ClusteringPipeline",
]
