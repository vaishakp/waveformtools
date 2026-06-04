"""Tests for local transform convention helpers."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools.transforms import compute_fft, compute_ifft


def test_compute_ifft_returns_consistent_time_axis():
    delta_t = 0.1
    data = np.arange(8, dtype=float)

    frequency_axis, frequency_data = compute_fft(data, delta_t)
    delta_f = float(frequency_axis[1] - frequency_axis[0])
    time_axis, roundtrip = compute_ifft(frequency_data, delta_f)

    assert len(time_axis) == len(data)
    assert len(roundtrip) == len(data)
    assert np.diff(time_axis) == pytest.approx(delta_t)
    assert np.max(np.abs(roundtrip - data)) < 1e-12
