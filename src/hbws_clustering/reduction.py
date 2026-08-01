"""UMAP dimensionality reduction for AVES embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

# Prefer GPU-accelerated cuML UMAP when available (e.g. Linux + NVIDIA GPU
# with RAPIDS installed).  Falls back to the CPU umap-learn package otherwise.
try:
    from cuml.manifold import UMAP as _UMAP
    import rmm

    # RMM pool allocator prevents memory fragmentation and cudaMalloc bottlenecks
    rmm.reinitialize(pool_allocator=True)

    _BACKEND = "cuml"
except ImportError:
    import umap

    _UMAP = umap.UMAP
    _BACKEND = "cpu"

log = logging.getLogger(__name__)


@dataclass
class UmapReducer:
    """Wrap UMAP to project high-dimensional embeddings to 2-D (or n-D).

    Parameters
    ----------
    n_components:
        Target dimensionality. ``2`` for visualization, higher for clustering.
    n_neighbors:
        UMAP neighborhood size. Controls local vs. global structure trade-off.
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

    _reducer: _UMAP = field(init=False, repr=False, default=None)

    def fit(self, embeddings: np.ndarray) -> "UmapReducer":
        """Fit the reducer on *embeddings* (N, D)."""
        print(f"  UMAP backend: {_BACKEND}")
        self._reducer = _UMAP(
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
        print(f"  UMAP backend: {_BACKEND}")
        if self._reducer is None:
            self._reducer = _UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
            )
        return self._reducer.fit_transform(embeddings).astype(np.float32)
