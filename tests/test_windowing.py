"""Tests for the audio windowing module."""

import numpy as np
import pytest

from hbws_clustering.windowing import AudioWindower


def make_sine(duration_sec: float = 5.0, sr: int = 16_000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_non_overlapping_window_count():
    sr = 16_000
    audio = make_sine(10.0, sr)
    windower = AudioWindower(window_sec=2.0, target_sr=None, min_window_sec=0.5)
    windows = windower.window_array(audio, sr)
    assert len(windows) == 5


def test_overlapping_window_count():
    sr = 16_000
    audio = make_sine(10.0, sr)
    windower = AudioWindower(window_sec=2.0, hop_sec=1.0, target_sr=None, min_window_sec=0.5)
    windows = windower.window_array(audio, sr)
    # starts at 0, 1, 2, ..., 9 sec → 9 full windows + the last short one depends on min_window_sec
    assert len(windows) >= 9


def test_window_shape():
    sr = 16_000
    audio = make_sine(5.0, sr)
    windower = AudioWindower(window_sec=2.0, target_sr=None)
    windows = windower.window_array(audio, sr)
    expected_samples = int(2.0 * sr)
    for w in windows:
        assert w.audio.shape == (expected_samples,)


def test_window_dtype():
    sr = 16_000
    audio = make_sine(4.0, sr)
    windower = AudioWindower(window_sec=2.0, target_sr=None)
    windows = windower.window_array(audio, sr)
    for w in windows:
        assert w.audio.dtype == np.float32


def test_zero_padding_on_last_window():
    sr = 16_000
    # 3-second audio with 2-second windows → last window needs 1 second of padding
    audio = make_sine(3.0, sr)
    windower = AudioWindower(window_sec=2.0, target_sr=None, min_window_sec=0.5)
    windows = windower.window_array(audio, sr)
    assert len(windows) == 2
    assert windows[-1].audio.shape == (int(2.0 * sr),)


def test_timestamps():
    sr = 16_000
    audio = make_sine(6.0, sr)
    windower = AudioWindower(window_sec=2.0, target_sr=None)
    windows = windower.window_array(audio, sr)
    assert windows[0].start_sec == pytest.approx(0.0)
    assert windows[1].start_sec == pytest.approx(2.0)
    assert windows[2].start_sec == pytest.approx(4.0)


def test_short_file_below_min_window_discarded():
    sr = 16_000
    audio = make_sine(0.3, sr)
    windower = AudioWindower(window_sec=2.0, target_sr=None, min_window_sec=0.5)
    windows = windower.window_array(audio, sr)
    assert len(windows) == 0
