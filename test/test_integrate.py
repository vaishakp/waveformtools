"""Tests for integration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools import integrate


def test_fixed_frequency_integrator_accepts_supplied_fft_without_mutating_input(
    monkeypatch,
):
    freq_axis = np.array([-1.0, 0.0, 1.0, 2.0])
    utilde = np.ones(4, dtype=np.complex128)
    captured = {}

    def fake_ifft(utilde_integ_n, delta_f):
        captured["utilde"] = np.array(utilde_integ_n, copy=True)
        captured["delta_f"] = delta_f
        return np.arange(len(utilde_integ_n)), utilde_integ_n

    monkeypatch.setattr(integrate, "compute_ifft", fake_ifft)

    integrate.fixed_frequency_integrator(
        udata_time=np.zeros(4),
        delta_t=0.1,
        utilde_conven=utilde,
        freq_axis=freq_axis,
        omega0=0.1,
        order=0,
        zero_mode=7.0,
    )

    assert np.all(utilde == 1.0)
    assert captured["utilde"][1] == pytest.approx(7.0)
    assert captured["delta_f"] == pytest.approx(1.0)


def test_fixed_frequency_integrator_handles_frequency_axis_without_zero_bin(
    monkeypatch,
):
    freq_axis = np.array([-2.0, -1.0, 1.0, 2.0])
    utilde = np.ones(4, dtype=np.complex128)
    captured = {}

    def fake_ifft(utilde_integ_n, delta_f):
        captured["utilde"] = np.array(utilde_integ_n, copy=True)
        return np.arange(len(utilde_integ_n)), utilde_integ_n

    monkeypatch.setattr(integrate, "compute_ifft", fake_ifft)

    integrate.fixed_frequency_integrator(
        udata_time=np.zeros(4),
        delta_t=0.1,
        utilde_conven=utilde,
        freq_axis=freq_axis,
        omega0=0.1,
        order=0,
        zero_mode=7.0,
    )

    assert np.all(captured["utilde"] == 1.0)
    assert np.all(utilde == 1.0)


def test_fixed_frequency_integrator_requires_frequency_axis_for_supplied_fft():
    with pytest.raises(ValueError, match="freq_axis"):
        integrate.fixed_frequency_integrator(
            udata_time=np.zeros(4),
            delta_t=0.1,
            utilde_conven=np.ones(4, dtype=np.complex128),
            freq_axis=None,
            omega0=0.1,
        )
