"""HDBSCAN clustering of UMAP-reduced embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import hdbscan


@dataclass
class HdbscanClusterer:
    """Cluster low-dimensional embeddings with HDBSCAN.

    Cluster label ``-1`` denotes noise / unclustered points.

    Parameters
    ----------
    min_cluster_size:
        Smallest group that can be considered a cluster.
    min_samples:
        Number of samples in the neighborhood for a point to be considered a
        core point. Defaults to ``min_cluster_size``.
    cluster_selection_epsilon:
        Distance threshold below which clusters are merged.
    metric:
        Distance metric for the clusterer.
    cluster_selection_method:
        ``"eom"`` (Excess of Mass, default) or ``"leaf"``.
    """

    min_cluster_size: int = 5
    min_samples: int | None = None
    cluster_selection_epsilon: float = 0.0
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"

    _clusterer: hdbscan.HDBSCAN = field(init=False, repr=False, default=None)

    def fit(self, reduced: np.ndarray) -> "HdbscanClusterer":
        """Fit on *reduced* (N, d) projected embeddings."""
        min_samples = self.min_samples if self.min_samples is not None else self.min_cluster_size
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
        )
        self._clusterer.fit(reduced)
        return self

    @property
    def labels(self) -> np.ndarray:
        """Integer cluster labels, ``-1`` for noise."""
        if self._clusterer is None:
            raise RuntimeError("Call fit() first.")
        return self._clusterer.labels_

    @property
    def probabilities(self) -> np.ndarray:
        """Soft cluster membership probabilities (0–1 per point)."""
        if self._clusterer is None:
            raise RuntimeError("Call fit() first.")
        return self._clusterer.probabilities_

    def cluster_summary(self) -> dict[int, int]:
        """Return a ``{label: count}`` dict sorted by label."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
