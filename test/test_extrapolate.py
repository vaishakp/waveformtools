"""Tests for perturbative extrapolation of Psi4 to null infinity.

Each test builds a finite-radius r*Psi4 from a known H(u) with the forward
asymptotic expansion of Nakano et al. (2015), Phys. Rev. D 91, 104022, then
checks that the extrapolation formula recovers Hddot(u) = r*Psi4(r = infinity)
up to the neglected O(1/r^3) terms.
"""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools.extrapolate import (
    waveextract_to_inf_perturbative_two_order,
    waveextract_to_inf_perturbative_twop5_order,
)
from waveformtools.integrate import fixed_frequency_integrator

DELTA_T = 0.1
N = 2**15
MASS = 1.0


def _tone(sign, cycles=400):
    """A periodic tone exp(sign * i * w * u) with an integer number of cycles."""
    u = np.arange(N) * DELTA_T
    w = 2 * np.pi * cycles / (N * DELTA_T)
    H = np.exp(sign * 1j * w * u)
    return H, sign * 1j * w


@pytest.mark.parametrize("sign", [-1, +1])
@pytest.mark.parametrize("order", [1, 2])
def test_ffi_integrates_both_frequency_signs(sign, order):
    H, iw = _tone(sign)
    _, integ = fixed_frequency_integrator(H, DELTA_T, omega0=0.01, order=order)
    np.testing.assert_allclose(integ, H / iw**order, rtol=1e-8, atol=1e-10)


def test_ffi_first_return_value_is_the_time_axis():
    H, _ = _tone(-1)
    time_axis, integ = fixed_frequency_integrator(H, DELTA_T, omega0=0.01)
    assert time_axis[0] == 0
    assert np.all(np.diff(time_axis) > 0)
    assert np.iscomplexobj(integ)


@pytest.mark.parametrize("sign", [-1, +1])
@pytest.mark.parametrize("ell", [2, 3])
def test_two_order_recovers_infinity(sign, ell):
    H, iw = _tone(sign)
    Hdot, Hddot = iw * H, iw**2 * H
    for r in (100.0, 200.0, 400.0):
        rpsi4 = (
            Hddot
            + (ell - 1) * (ell + 2) / (2 * r) * Hdot
            + (ell - 1) * ell * (ell + 1) * (ell + 2) / (8 * r**2) * H
            + 3 * MASS / (2 * r**2) * Hdot
        )
        out = waveextract_to_inf_perturbative_two_order(
            rpsi4, DELTA_T, areal_radius=r, mass=MASS, ell=ell, omega0=0.01
        )
        err = np.max(np.abs(out - Hddot)) / np.max(np.abs(Hddot))
        uncorrected = np.max(np.abs(rpsi4 - Hddot)) / np.max(np.abs(Hddot))
        # Residual is O(1/r^3); the uncorrected waveform is off at O(1/r).
        assert err < 20 * (ell**2 / r) ** 3 / abs(iw) ** 3
        assert err < 1e-2 * uncorrected


@pytest.mark.parametrize("sign", [-1, +1])
@pytest.mark.parametrize("ell", [2, 3])
def test_twop5_order_without_spin_recovers_infinity(sign, ell):
    H, iw = _tone(sign)
    Hdot, Hddot = iw * H, iw**2 * H
    for r in (100.0, 200.0, 400.0):
        # Forward model with the numerical-tetrad factor of the improved formula.
        rpsi4 = (
            Hddot
            + (ell - 1) * (ell + 2) / (2 * r) * Hdot
            + (ell - 1) * ell * (ell + 1) * (ell + 2) / (8 * r**2) * H
        ) / (1 - 2 * MASS / r)
        out = waveextract_to_inf_perturbative_twop5_order(
            rpsi4, DELTA_T, areal_radius=r, mass=MASS, ell=ell, emm=ell, omega0=0.01
        )
        err = np.max(np.abs(out - Hddot)) / np.max(np.abs(Hddot))
        uncorrected = np.max(np.abs(rpsi4 - Hddot)) / np.max(np.abs(Hddot))
        assert err < 1e-2 * uncorrected


def test_twop5_order_integral_term_has_the_published_sign():
    """With the 1/r integral term entering with the wrong sign the error is O(1/r)."""
    H, iw = _tone(-1)
    Hdot, Hddot = iw * H, iw**2 * H
    r, ell = 100.0, 2
    rpsi4 = Hddot + (ell - 1) * (ell + 2) / (2 * r) * Hdot
    out = waveextract_to_inf_perturbative_twop5_order(
        rpsi4 * (1 / (1 - 2 * MASS / r)), DELTA_T, areal_radius=r, mass=MASS,
        ell=ell, emm=2, omega0=0.01,
    )
    assert np.max(np.abs(out - Hddot)) / np.max(np.abs(Hddot)) < 1e-3


def test_twop5_order_spin_needs_the_neighbouring_modes():
    H, _ = _tone(-1)
    with pytest.raises(ValueError, match="ell \\+ 1"):
        waveextract_to_inf_perturbative_twop5_order(
            H, DELTA_T, areal_radius=100.0, spin=0.5, ell=2, emm=2, omega0=0.01
        )
    # For ell = 2 the coupling to ell - 1 vanishes, so only ell + 1 is needed.
    out = waveextract_to_inf_perturbative_twop5_order(
        H, DELTA_T, areal_radius=100.0, spin=0.5, ell=2, emm=2, omega0=0.01,
        rPsi4_lp1m=np.zeros_like(H),
    )
    assert out.shape == H.shape
    with pytest.raises(ValueError, match="ell - 1"):
        waveextract_to_inf_perturbative_twop5_order(
            H, DELTA_T, areal_radius=100.0, spin=0.5, ell=3, emm=2, omega0=0.01,
            rPsi4_lp1m=np.zeros_like(H),
        )
