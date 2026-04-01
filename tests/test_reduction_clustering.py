"""Tests for UMAP reduction and HDBSCAN clustering."""

import numpy as np
import pytest

from hbws_clustering.clustering import HdbscanClusterer
from hbws_clustering.reduction import UmapReducer


def make_blobs(n: int = 200, d: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Two well-separated Gaussian clusters
    c1 = rng.normal(loc=0.0, scale=0.3, size=(n // 2, d))
    c2 = rng.normal(loc=10.0, scale=0.3, size=(n // 2, d))
    return np.vstack([c1, c2]).astype(np.float32)


def test_umap_output_shape():
    data = make_blobs(100, 32)
    reducer = UmapReducer(n_components=2, n_neighbors=10)
    reduced = reducer.fit_transform(data)
    assert reduced.shape == (100, 2)


def test_umap_output_dtype():
    data = make_blobs(100, 32)
    reducer = UmapReducer(n_components=2, n_neighbors=10)
    reduced = reducer.fit_transform(data)
    assert reduced.dtype == np.float32


def test_umap_transform_before_fit_raises():
    reducer = UmapReducer()
    with pytest.raises(RuntimeError, match="fit"):
        reducer.transform(np.zeros((10, 8)))


def test_hdbscan_finds_clusters():
    data = make_blobs(200, 32)
    reducer = UmapReducer(n_components=2, n_neighbors=15, metric="euclidean")
    reduced = reducer.fit_transform(data)
    clusterer = HdbscanClusterer(min_cluster_size=10)
    clusterer.fit(reduced)
    labels = clusterer.labels
    n_clusters = sum(1 for k in np.unique(labels) if k >= 0)
    assert n_clusters >= 2, f"Expected >=2 clusters, got {n_clusters}"


def test_hdbscan_labels_shape():
    data = make_blobs(100, 16)
    reducer = UmapReducer(n_components=2, n_neighbors=10)
    reduced = reducer.fit_transform(data)
    clusterer = HdbscanClusterer(min_cluster_size=5)
    clusterer.fit(reduced)
    assert clusterer.labels.shape == (100,)
    assert clusterer.probabilities.shape == (100,)


def test_hdbscan_before_fit_raises():
    clusterer = HdbscanClusterer()
    with pytest.raises(RuntimeError, match="fit"):
        _ = clusterer.labels


def test_cluster_summary_keys():
    data = make_blobs(200, 32)
    reducer = UmapReducer(n_components=2, n_neighbors=15, metric="euclidean")
    reduced = reducer.fit_transform(data)
    clusterer = HdbscanClusterer(min_cluster_size=10)
    clusterer.fit(reduced)
    summary = clusterer.cluster_summary()
    for label, count in summary.items():
        assert isinstance(label, int)
        assert count > 0
