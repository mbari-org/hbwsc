"""Audio windowing: load audio files and slice into fixed-length windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


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

    def _slice(self, audio: np.ndarray, sr: int, path: Path, offset_sec: float = 0.0) -> list[Window]:
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
                    start_sec=offset_sec + start / sr,
                    end_sec=offset_sec + (start + win_samples) / sr,
                )
            )
            start += hop_samples

        return windows


@dataclass
class ScoreGuidedWindower:
    """Extract windows only from high-confidence regions of a day-long audio file.

    Uses detector score arrays (one float per second, shape ``(86400,)``) produced
    by the NOAA/Google Humpback Whale Song Detector to skip silence and ambient
    noise, reading only the relevant slices from disk.  The 4+ GB day file is
    never loaded into memory in full.

    Parameters
    ----------
    windower:
        Underlying :class:`AudioWindower` used to slice each high-score chunk.
    threshold:
        Minimum score to include a second in the analysis.
    """

    windower: AudioWindower = field(default_factory=AudioWindower)
    threshold: float = 0.7

    def window_file(
        self,
        audio_path: str | Path,
        scores: np.ndarray | str | Path,
    ) -> list[Window]:
        """Return windows from high-score regions of *audio_path*.

        Parameters
        ----------
        audio_path:
            Path to a day-long WAV file (any sample rate; resampling is handled
            by the underlying ``AudioWindower``).
        scores:
            Either a 1-D numpy array of per-second scores, or a path to a
            ``.npy`` file containing one.
        """
        if not isinstance(scores, np.ndarray):
            scores = np.load(scores)

        segments = self._high_score_segments(scores)
        if not segments:
            return []

        audio_path = Path(audio_path)
        windows: list[Window] = []

        with sf.SoundFile(audio_path) as f:
            sr_native = f.samplerate
            for start_sec, end_sec in segments:
                start_frame = int(start_sec * sr_native)
                # read enough frames to cover at least one full window past end_sec
                stop_frame = int((end_sec + self.windower.window_sec) * sr_native)
                stop_frame = min(stop_frame, f.frames)

                f.seek(start_frame)
                chunk = f.read(stop_frame - start_frame, dtype="float32", always_2d=False)

                chunk_windows = self.windower.window_array(
                    chunk, sr_native, source_file=audio_path
                )
                # shift timestamps to be absolute within the day
                for w in chunk_windows:
                    w.start_sec += start_sec
                    w.end_sec += start_sec

                windows.extend(chunk_windows)

        return windows

    def _high_score_segments(self, scores: np.ndarray) -> list[tuple[int, int]]:
        """Return ``[(start_sec, end_sec), ...]`` for contiguous runs above threshold."""
        high = np.where(scores >= self.threshold)[0]
        if len(high) == 0:
            return []

        segments = []
        gaps = np.where(np.diff(high) > 1)[0] + 1
        for run in np.split(high, gaps):
            segments.append((int(run[0]), int(run[-1]) + 1))
        return segments
