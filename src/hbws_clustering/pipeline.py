"""End-to-end clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from hbws_clustering.windowing import AudioWindower, ScoreGuidedWindower, Window
from hbws_clustering.embedding import AvesEmbedder, AVES_BASE_BIO
from hbws_clustering.reduction import UmapReducer
from hbws_clustering.clustering import HdbscanClusterer

# Pair of (audio_path, scores) accepted by run_scored.
# scores may be a numpy array or a path to a .npy file.
ScoredPair = tuple[str | Path, np.ndarray | str | Path]


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

    Plain windowing example
    -----------------------
    >>> pipe = ClusteringPipeline()
    >>> result = pipe.run(["call1.wav", "call2.wav"])
    >>> print(result.cluster_summary())

    Score-guided windowing example
    ------------------------------
    >>> pipe = ClusteringPipeline(windower=ScoreGuidedWindower())
    >>> result = pipe.run_scored([
    ...     ("MARS-20260330T000000Z-16kHz.wav", "Scores-20260330.npy"),
    ...     ("MARS-20260331T000000Z-16kHz.wav", "Scores-20260331.npy"),
    ... ])
    """

    windower: AudioWindower | ScoreGuidedWindower = field(default_factory=AudioWindower)
    embedder: AvesEmbedder = field(default_factory=lambda: AvesEmbedder(model_url=AVES_BASE_BIO))
    reducer: UmapReducer = field(default_factory=UmapReducer)
    clusterer: HdbscanClusterer = field(default_factory=HdbscanClusterer)
    verbose: bool = True

    def run(self, audio_paths: Sequence[str | Path]) -> PipelineResult:
        """Run the full pipeline on a list of plain audio files."""
        self._log("Step 1/4: Windowing audio files...")
        windows: list[Window] = []
        for path in audio_paths:
            file_windows = self.windower.window_file(path)
            windows.extend(file_windows)
            self._log(f"  {Path(path).name}: {len(file_windows)} windows")
        self._log(f"  Total windows: {len(windows)}")

        return self._run_from_windows(windows)

    def run_scored(self, pairs: Sequence[ScoredPair]) -> PipelineResult:
        """Run the pipeline using detector scores to guide windowing.

        Parameters
        ----------
        pairs:
            Sequence of ``(audio_path, scores)`` where *scores* is either a
            1-D numpy array of per-second detection scores or a path to a
            ``.npy`` file containing one.

        Raises
        ------
        TypeError
            If ``self.windower`` is not a :class:`ScoreGuidedWindower`.
        """
        if not isinstance(self.windower, ScoreGuidedWindower):
            raise TypeError(
                "run_scored() requires a ScoreGuidedWindower. "
                f"Got {type(self.windower).__name__}. "
                "Construct the pipeline with windower=ScoreGuidedWindower(...)."
            )

        self._log("Step 1/4: Score-guided windowing...")
        windows: list[Window] = []
        for audio_path, scores in pairs:
            file_windows = self.windower.window_file(audio_path, scores)
            windows.extend(file_windows)
            self._log(f"  {Path(audio_path).name}: {len(file_windows)} windows")
        self._log(f"  Total windows: {len(windows)}")

        return self._run_from_windows(windows)

    def _run_from_windows(self, windows: list[Window]) -> PipelineResult:
        if not windows:
            raise ValueError("No windows extracted — check audio paths and window/score settings.")

        self._log("Step 2/4: Extracting AVES embeddings...")
        embeddings = self.embedder.embed_windows(windows)
        self._log(f"  Embeddings shape: {embeddings.shape}")

        self._log("Step 3/4: UMAP dimensionality reduction...")
        reduced = self.reducer.fit_transform(embeddings)
        self._log(f"  Reduced shape: {reduced.shape}")

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
