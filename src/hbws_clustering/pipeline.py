"""End-to-end clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from hbws_clustering.clustering import HdbscanClusterer
from hbws_clustering.embedding import AVES_BASE_BIO, AvesEmbedder, PerchEmbedder
from hbws_clustering.reduction import UmapReducer
from hbws_clustering.windowing import AudioWindower, ScoreGuidedWindower, Window

# Pair of (audio_path, scores) accepted by run_scored.
# scores may be a numpy array or a path to a .npy file.
ScoredPair = tuple[str | Path, np.ndarray | str | Path]


@dataclass
class PipelineResult:
    """Holds all outputs produced by a pipeline run."""

    windows: list[Window]
    embeddings: np.ndarray  # (N, D)
    reduced: np.ndarray  # (N, 2) — 2-D UMAP projection for visualization
    reduced_cluster: np.ndarray  # (N, k) — k-D UMAP projection used for clustering
    labels: np.ndarray  # (N,) int
    probabilities: np.ndarray  # (N,) float

    def cluster_summary(self) -> dict[int, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


@dataclass
class ClusteringPipeline:
    """Orchestrate windowing → AVES embedding → UMAP → HDBSCAN.

    Two UMAP passes are supported: ``reducer_cluster`` (high-dimensional, fed to
    HDBSCAN) and ``reducer`` (2-D, for visualization).  When ``reducer_cluster``
    is ``None`` the single ``reducer`` is used for both clustering and plotting —
    which is only appropriate when ``reducer.n_components`` is high enough.

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
    embedder: AvesEmbedder | PerchEmbedder = field(default_factory=lambda: AvesEmbedder(model_url=AVES_BASE_BIO))
    reducer: UmapReducer = field(default_factory=UmapReducer)
    reducer_cluster: UmapReducer | None = None  # high-D UMAP for clustering; None → use reducer
    clusterer: HdbscanClusterer = field(default_factory=HdbscanClusterer)
    verbose: bool = True

    def run(
        self,
        audio_paths: Sequence[str | Path],
        embeddings_cache: Path | None = None,
    ) -> PipelineResult:
        """Run the full pipeline on a list of plain audio files."""
        self._log("Step 1/4: Windowing audio files...")
        windows: list[Window] = []
        embeddings_parts: list[np.ndarray] = []
        use_cache = embeddings_cache is not None and embeddings_cache.exists()

        if use_cache:
            cached = np.load(embeddings_cache)
            self._log(f"  Loaded embeddings from cache: {embeddings_cache}")

        for path in audio_paths:
            file_windows = self.windower.window_file(path)
            self._log(f"  {Path(path).name}: {len(file_windows)} windows")
            if not use_cache and file_windows:
                embeddings_parts.append(self.embedder.embed_windows(file_windows))
            for w in file_windows:
                w.audio = None
            windows.extend(file_windows)

        self._log(f"  Total windows: {len(windows)}")
        embeddings = self._finalize_embeddings(windows, embeddings_parts, use_cache, cached if use_cache else None, embeddings_cache)
        return self._run_from_embeddings(windows, embeddings)

    def run_scored(self, pairs: Sequence[ScoredPair], embeddings_cache: Path | None = None) -> PipelineResult:
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
        embeddings_parts: list[np.ndarray] = []
        use_cache = embeddings_cache is not None and embeddings_cache.exists()

        if use_cache:
            cached = np.load(embeddings_cache)
            self._log(f"  Loaded embeddings from cache: {embeddings_cache}")

        for audio_path, scores in pairs:
            file_windows = self.windower.window_file(audio_path, scores)
            self._log(f"  {Path(audio_path).name}: {len(file_windows)} windows")
            if not use_cache and file_windows:
                embeddings_parts.append(self.embedder.embed_windows(file_windows))
            for w in file_windows:
                w.audio = None
            windows.extend(file_windows)

        self._log(f"  Total windows: {len(windows)}")
        embeddings = self._finalize_embeddings(windows, embeddings_parts, use_cache, cached if use_cache else None, embeddings_cache)
        return self._run_from_embeddings(windows, embeddings)

    def _finalize_embeddings(
        self,
        windows: list[Window],
        parts: list[np.ndarray],
        use_cache: bool,
        cached: np.ndarray | None,
        cache_path: Path | None,
    ) -> np.ndarray:
        """Validate cached embeddings or concatenate per-file parts and save."""
        if not windows:
            raise ValueError("No windows extracted — check audio paths and window/score settings.")

        if use_cache:
            assert cached is not None
            if cached.shape[0] != len(windows):
                raise ValueError(
                    f"Cache has {cached.shape[0]} rows but {len(windows)} windows. "
                    "Delete the cache file and re-run."
                )
            return cached

        self._log("Step 2/4: Extracting embeddings...")
        embeddings = np.concatenate(parts, axis=0) if parts else np.empty((0, 0))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            self._log(f"  Saved to cache: {cache_path}")
        return embeddings

    def _run_from_embeddings(self, windows: list[Window], embeddings: np.ndarray) -> PipelineResult:
        self._log(f"  Embeddings shape: {embeddings.shape}")

        if self.reducer_cluster is not None:
            self._log(
                f"Step 3/4: UMAP dimensionality reduction ({self.reducer_cluster.n_components}-D for clustering,"
                f" {self.reducer.n_components}-D for visualization)..."
            )
            reduced_cluster = self.reducer_cluster.fit_transform(embeddings)
            self._log(f"  Cluster-space shape: {reduced_cluster.shape}")
            reduced = self.reducer.fit_transform(embeddings)
            self._log(f"  Visualization shape: {reduced.shape}")
        else:
            self._log(f"Step 3/4: UMAP dimensionality reduction ({self.reducer.n_components}-D)...")
            reduced = self.reducer.fit_transform(embeddings)
            reduced_cluster = reduced
            self._log(f"  Reduced shape: {reduced.shape}")

        self._log("Step 4/4: HDBSCAN clustering...")
        self.clusterer.fit(reduced_cluster)
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
            reduced_cluster=reduced_cluster,
            labels=labels,
            probabilities=probabilities,
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

