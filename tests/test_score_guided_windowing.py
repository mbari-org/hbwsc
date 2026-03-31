"""Tests for ScoreGuidedWindower."""

import io
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from hbws_clustering.windowing import AudioWindower, ScoreGuidedWindower


SR = 16_000


def make_wav(duration_sec: float, sr: int = SR, freq: float = 440.0) -> Path:
    """Write a sine-wave WAV to a temp file and return the path."""
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return Path(tmp.name)


def make_scores(n_seconds: int, high: list[int], threshold: float = 0.7) -> np.ndarray:
    """Return a scores array with `high` seconds set above threshold."""
    scores = np.zeros(n_seconds, dtype=np.float32)
    for s in high:
        scores[s] = threshold + 0.1
    return scores


def test_high_score_segments_contiguous():
    sgw = ScoreGuidedWindower(threshold=0.7)
    scores = make_scores(20, high=[5, 6, 7])
    segs = sgw._high_score_segments(scores)
    assert segs == [(5, 8)]


def test_high_score_segments_two_runs():
    sgw = ScoreGuidedWindower(threshold=0.7)
    scores = make_scores(20, high=[2, 3, 10, 11, 12])
    segs = sgw._high_score_segments(scores)
    assert segs == [(2, 4), (10, 13)]


def test_high_score_segments_empty():
    sgw = ScoreGuidedWindower(threshold=0.7)
    scores = np.zeros(20, dtype=np.float32)
    assert sgw._high_score_segments(scores) == []


def test_window_file_returns_windows_only_in_high_score_region():
    # 10-second audio; only seconds 4-5 are above threshold
    audio_path = make_wav(10.0)
    scores = make_scores(10, high=[4, 5])

    sgw = ScoreGuidedWindower(
        windower=AudioWindower(window_sec=1.0, hop_sec=1.0, target_sr=None),
        threshold=0.7,
    )
    windows = sgw.window_file(audio_path, scores)

    assert len(windows) > 0
    # All window starts must fall within the high-score region
    for w in windows:
        assert w.start_sec >= 4.0


def test_window_file_timestamps_are_absolute():
    audio_path = make_wav(10.0)
    scores = make_scores(10, high=[6, 7, 8])

    sgw = ScoreGuidedWindower(
        windower=AudioWindower(window_sec=1.0, hop_sec=1.0, target_sr=None),
        threshold=0.7,
    )
    windows = sgw.window_file(audio_path, scores)

    assert len(windows) > 0
    assert windows[0].start_sec >= 6.0


def test_window_file_accepts_npy_path(tmp_path):
    audio_path = make_wav(10.0)
    scores = make_scores(10, high=[3, 4])
    npy_path = tmp_path / "scores.npy"
    np.save(npy_path, scores)

    sgw = ScoreGuidedWindower(
        windower=AudioWindower(window_sec=1.0, hop_sec=1.0, target_sr=None),
        threshold=0.7,
    )
    windows = sgw.window_file(audio_path, npy_path)
    assert len(windows) > 0


def test_window_file_all_below_threshold_returns_empty():
    audio_path = make_wav(10.0)
    scores = np.zeros(10, dtype=np.float32)

    sgw = ScoreGuidedWindower(threshold=0.7)
    windows = sgw.window_file(audio_path, scores)
    assert windows == []
