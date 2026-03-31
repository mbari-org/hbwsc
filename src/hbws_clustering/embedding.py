"""AVES embedding extraction via TorchAudio.

AVES (Audio Visual Embeddings for Self-supervised learning) models are hosted
by the Earth Species Project on Google Cloud Storage:
  https://github.com/earthspecies/aves
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchaudio

from hbws_clustering.windowing import Window


# TorchAudio-format checkpoints published by the Earth Species Project.
AVES_BASE_BIO = "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt"
AVES_BASE_ALL = "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.torchaudio.pt"

# Local cache directory (mirrors torch.hub convention)
_CACHE_DIR = Path(torch.hub.get_dir()) / "aves"


@dataclass
class AvesEmbedder:
    """Extract frame-level or pooled AVES embeddings for audio windows.

    The model checkpoint is downloaded once and cached locally under
    ``torch.hub.get_dir()/aves/``.

    Parameters
    ----------
    model_url:
        URL to a TorchAudio-format AVES checkpoint (.pt). Use the module-level
        constants ``AVES_BASE_BIO`` or ``AVES_BASE_ALL``.
    pooling:
        How to aggregate frame-level features into a single vector per window.
        ``"mean"`` (default) averages over time; ``"max"`` takes the maximum.
    device:
        Torch device string. Defaults to CUDA if available, else CPU.
    batch_size:
        Number of windows processed per forward pass.
    """

    model_url: str = AVES_BASE_BIO
    pooling: str = "mean"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 16

    _model: torchaudio.models.Wav2Vec2Model = field(init=False, repr=False, default=None)

    def _load(self) -> None:
        if self._model is not None:
            return
        checkpoint_path = self._download(self.model_url)
        self._model = torchaudio.models.wav2vec2_base()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._model.to(self.device)

    @staticmethod
    def _download(url: str) -> Path:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filename = Path(url).name
        dest = _CACHE_DIR / filename
        if not dest.exists():
            print(f"Downloading AVES checkpoint to {dest} ...")
            torch.hub.download_url_to_file(url, str(dest))
        return dest

    def embed_windows(self, windows: Sequence[Window]) -> np.ndarray:
        """Return an (N, D) float32 array of embeddings for *windows*."""
        from tqdm import tqdm

        self._load()
        all_embeddings: list[np.ndarray] = []

        with tqdm(total=len(windows), unit="win", desc="AVES embeddings") as bar:
            for batch_start in range(0, len(windows), self.batch_size):
                batch = list(windows[batch_start : batch_start + self.batch_size])
                all_embeddings.append(self._embed_batch(batch))
                bar.update(len(batch))

        return np.concatenate(all_embeddings, axis=0)

    def _embed_batch(self, batch: list[Window]) -> np.ndarray:
        # Stack waveforms; all windows are the same length (AudioWindower pads them)
        waveforms = torch.stack(
            [torch.from_numpy(w.audio) for w in batch]
        ).to(self.device)  # (B, T)

        with torch.no_grad():
            # Returns (features, lengths): features is (B, T', D)
            features, _ = self._model(waveforms)

        if self.pooling == "mean":
            pooled = features.mean(dim=1)
        elif self.pooling == "max":
            pooled = features.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling!r}. Use 'mean' or 'max'.")

        return pooled.cpu().float().numpy()
