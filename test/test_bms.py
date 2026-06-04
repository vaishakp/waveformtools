"""Tests for BMS waveform transformation helpers."""

from __future__ import annotations

import numpy as np

from spectools.spherical.swsh import Yslm_vec

from waveformtools.BMS import compute_supertransl_alpha


def test_compute_supertransl_alpha_uses_supplied_angles():
    alpha_modes = {"l1": [0.0, 1.0, 0.0]}
    theta_a, phi_a = 0.3, 0.1
    theta_b, phi_b = 1.1, 0.1

    value_a = compute_supertransl_alpha(alpha_modes, theta_a, phi_a)
    value_b = compute_supertransl_alpha(alpha_modes, theta_b, phi_b)

    assert value_a == np.asarray(Yslm_vec(0, 1, 0, theta_a, phi_a)).item()
    assert value_b == np.asarray(Yslm_vec(0, 1, 0, theta_b, phi_b)).item()
    assert not np.allclose(value_a, value_b)
