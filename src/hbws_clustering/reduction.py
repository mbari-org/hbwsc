"""UMAP dimensionality reduction for AVES embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import umap


@dataclass
class UmapReducer:
    """Wrap UMAP to project high-dimensional embeddings to 2-D (or n-D).

    Parameters
    ----------
    n_components:
        Target dimensionality. ``2`` for visualization, higher for clustering.
    n_neighbors:
        UMAP neighbourhood size. Controls local vs. global structure trade-off.
    min_dist:
        Minimum distance between points in the low-dimensional representation.
    metric:
        Distance metric used in the input space.
    random_state:
        Seed for reproducibility.
    """

    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42

    _reducer: umap.UMAP = field(init=False, repr=False, default=None)

    def fit(self, embeddings: np.ndarray) -> "UmapReducer":
        """Fit the reducer on *embeddings* (N, D)."""
        self._reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        self._reducer.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project *embeddings* to low-dimensional space."""
        if self._reducer is None:
            raise RuntimeError("Call fit() before transform().")
        return self._reducer.transform(embeddings).astype(np.float32)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and project in one step."""
        self.fit(embeddings)
        return self.transform(embeddings)
