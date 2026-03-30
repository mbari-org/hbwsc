"""Audio windowing: load audio files and slice into fixed-length windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import librosa


@dataclass
class Window:
    """A single audio window with metadata."""

    audio: np.ndarray          # shape (n_samples,), float32, mono
    sample_rate: int
    source_file: Path
    start_sec: float
    end_sec: float


@dataclass
class AudioWindower:
    """Slice audio files into overlapping fixed-length windows.

    Parameters
    ----------
    window_sec:
        Duration of each window in seconds.
    hop_sec:
        Step between successive window starts. Defaults to ``window_sec``
        (non-overlapping).
    target_sr:
        Sample rate to resample to. Set to ``None`` to keep the original rate.
    min_window_sec:
        Minimum duration required for the final partial window to be kept.
        Partial windows shorter than this are discarded.
    """

    window_sec: float = 2.0
    hop_sec: float | None = None
    target_sr: int = 16_000
    min_window_sec: float = 0.5
    _hop_sec: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._hop_sec = self.hop_sec if self.hop_sec is not None else self.window_sec

    def window_file(self, path: str | Path) -> list[Window]:
        """Load *path* and return all windows extracted from it."""
        path = Path(path)
        audio, sr = librosa.load(path, sr=self.target_sr, mono=True)
        return self._slice(audio, sr, path)

    def window_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        source_file: Path | None = None,
    ) -> list[Window]:
        """Window a pre-loaded numpy array."""
        if source_file is None:
            source_file = Path("<array>")
        if self.target_sr is not None and sample_rate != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.target_sr)
            sample_rate = self.target_sr
        return self._slice(audio, sample_rate, source_file)

    def _slice(self, audio: np.ndarray, sr: int, path: Path) -> list[Window]:
        win_samples = int(self.window_sec * sr)
        hop_samples = int(self._hop_sec * sr)
        min_samples = int(self.min_window_sec * sr)

        windows: list[Window] = []
        start = 0
        while start < len(audio):
            end = start + win_samples
            chunk = audio[start:end]
            if len(chunk) < min_samples:
                break
            # zero-pad the last (possibly short) window
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))
            windows.append(
                Window(
                    audio=chunk.astype(np.float32),
                    sample_rate=sr,
                    source_file=path,
                    start_sec=start / sr,
                    end_sec=(start + win_samples) / sr,
                )
            )
            start += hop_samples

        return windows
