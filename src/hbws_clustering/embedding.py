"""AVES embedding extraction via HuggingFace Transformers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoFeatureExtractor

from hbws_clustering.windowing import Window


# AVES models published on HuggingFace by the AVES authors.
# "aves-base-bio" is pretrained on bioacoustic audio (recommended for wildlife).
AVES_BASE_BIO = "m-a-p/AVES-base-bio"
AVES_BASE_ALL = "m-a-p/AVES-base-all"


@dataclass
class AvesEmbedder:
    """Extract frame-level or pooled AVES embeddings for audio windows.

    Parameters
    ----------
    model_name:
        HuggingFace model ID for the AVES checkpoint.
    pooling:
        How to aggregate frame-level features into a single vector per window.
        ``"mean"`` (default) averages over time; ``"max"`` takes the maximum.
    device:
        Torch device string. Defaults to CUDA if available, else CPU.
    batch_size:
        Number of windows processed per forward pass.
    """

    model_name: str = AVES_BASE_BIO
    pooling: str = "mean"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 16

    _model: AutoModel = field(init=False, repr=False, default=None)
    _feature_extractor: AutoFeatureExtractor = field(init=False, repr=False, default=None)

    def _load(self) -> None:
        if self._model is not None:
            return
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

    @property
    def embedding_dim(self) -> int:
        self._load()
        return self._model.config.hidden_size

    def embed_windows(self, windows: Sequence[Window]) -> np.ndarray:
        """Return an (N, D) float32 array of embeddings for *windows*."""
        self._load()
        all_embeddings: list[np.ndarray] = []

        for batch_start in range(0, len(windows), self.batch_size):
            batch = windows[batch_start : batch_start + self.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def _embed_batch(self, batch: list[Window]) -> np.ndarray:
        sr = batch[0].sample_rate
        raw_arrays = [w.audio for w in batch]

        inputs = self._feature_extractor(
            raw_arrays,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.last_hidden_state: (B, T, D)
        hidden = outputs.last_hidden_state  # (B, T, D)

        if self.pooling == "mean":
            pooled = hidden.mean(dim=1)
        elif self.pooling == "max":
            pooled = hidden.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling!r}. Use 'mean' or 'max'.")

        return pooled.cpu().float().numpy()
