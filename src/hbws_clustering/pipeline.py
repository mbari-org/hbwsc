"""End-to-end clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from hbws_clustering.windowing import AudioWindower, Window
from hbws_clustering.embedding import AvesEmbedder, AVES_BASE_BIO
from hbws_clustering.reduction import UmapReducer
from hbws_clustering.clustering import HdbscanClusterer


@dataclass
class PipelineResult:
    """Holds all outputs produced by a pipeline run."""

    windows: list[Window]
    embeddings: np.ndarray       # (N, D)
    reduced: np.ndarray          # (N, n_components)
    labels: np.ndarray           # (N,) int
    probabilities: np.ndarray    # (N,) float

    def cluster_summary(self) -> dict[int, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


@dataclass
class ClusteringPipeline:
    """Orchestrate windowing → AVES embedding → UMAP → HDBSCAN.

    All component configurations can be overridden at construction time.

    Example
    -------
    >>> pipe = ClusteringPipeline()
    >>> result = pipe.run(["call1.wav", "call2.wav"])
    >>> print(result.cluster_summary())
    """

    windower: AudioWindower = field(default_factory=AudioWindower)
    embedder: AvesEmbedder = field(default_factory=lambda: AvesEmbedder(model_name=AVES_BASE_BIO))
    reducer: UmapReducer = field(default_factory=UmapReducer)
    clusterer: HdbscanClusterer = field(default_factory=HdbscanClusterer)
    verbose: bool = True

    def run(self, audio_paths: Sequence[str | Path]) -> PipelineResult:
        """Run the full pipeline on a list of audio files."""
        # 1. Windowing
        self._log("Step 1/4: Windowing audio files...")
        windows: list[Window] = []
        for path in audio_paths:
            file_windows = self.windower.window_file(path)
            windows.extend(file_windows)
            self._log(f"  {Path(path).name}: {len(file_windows)} windows")
        self._log(f"  Total windows: {len(windows)}")

        if not windows:
            raise ValueError("No windows extracted — check audio paths and window settings.")

        # 2. AVES embedding
        self._log("Step 2/4: Extracting AVES embeddings...")
        embeddings = self.embedder.embed_windows(windows)
        self._log(f"  Embeddings shape: {embeddings.shape}")

        # 3. UMAP reduction
        self._log("Step 3/4: UMAP dimensionality reduction...")
        reduced = self.reducer.fit_transform(embeddings)
        self._log(f"  Reduced shape: {reduced.shape}")

        # 4. HDBSCAN clustering
        self._log("Step 4/4: HDBSCAN clustering...")
        self.clusterer.fit(reduced)
        labels = self.clusterer.labels
        probabilities = self.clusterer.probabilities
        summary = self.clusterer.cluster_summary()
        n_clusters = sum(1 for k in summary if k >= 0)
        n_noise = summary.get(-1, 0)
        self._log(f"  Clusters found: {n_clusters}  |  Noise points: {n_noise}")

        return PipelineResult(
            windows=windows,
            embeddings=embeddings,
            reduced=reduced,
            labels=labels,
            probabilities=probabilities,
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
