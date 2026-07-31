"""Tests for integration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools import integrate
from waveformtools.grids import UniformGrid, GLGrid
from waveformtools.integrate import TwoDIntegral


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


# The area element sqrt(det(g_ab)) for the round unit sphere metric
# diag(1, sin^2(theta)) is |sin(theta)|, which TwoDIntegral applies as its
# integration measure. Integrating the constant function 1 therefore yields
# the surface area of the unit sphere, 4*pi.
def test_twodintegral_sphere_area_uniform_grid():
    """Midpoint/Simpson/Driscoll-Healy recover the unit-sphere area (4*pi)."""
    info = UniformGrid(ntheta=48, nphi=96)
    func = np.ones(info.shape)

    # Tolerances reflect the order of each rule on the uniform grid: the
    # midpoint and Driscoll-Healy rules converge quickly, Simpson's is coarser.
    assert TwoDIntegral(func, info, int_method="MP") / (4 * np.pi) == pytest.approx(
        1.0, abs=1e-3
    )
    assert TwoDIntegral(func, info, int_method="SP") / (4 * np.pi) == pytest.approx(
        1.0, abs=5e-2
    )
    assert TwoDIntegral(func, info, int_method="DH") / (4 * np.pi) == pytest.approx(
        1.0, abs=1e-2
    )


def test_twodintegral_sphere_area_gauss_legendre():
    """Gauss-Legendre integration is spectrally accurate for the sphere area."""
    info = GLGrid(L=48)
    func = np.ones(info.shape)

    assert TwoDIntegral(func, info, int_method="GL") / (4 * np.pi) == pytest.approx(
        1.0, abs=1e-10
    )


def test_twodintegral_does_not_mutate_input():
    """TwoDIntegral must not multiply the caller's array by the area measure
    in place. This regressed when Simpson/Driscoll-Healy shared a mutated
    array across successive calls, giving an area of pi/4 instead of 4*pi."""
    info = UniformGrid(ntheta=48, nphi=96)
    func = np.ones(info.shape)

    for method in ("MP", "SP", "DH"):
        TwoDIntegral(func, info, int_method=method)
        assert np.all(func == 1.0), f"{method} mutated its input array"
