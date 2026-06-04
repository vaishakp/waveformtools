"""Tests for dependency-light rotation math helpers."""

from __future__ import annotations

import numpy as np

from waveformtools.rotation_math import euler_zyz_quaternion, wigner_d


def test_rotation_math_imports_without_scri_side_effects():
    quat = euler_zyz_quaternion(0.0, 0.0, 0.0)

    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_wigner_d_z_axis_limit():
    angle = 0.4
    quat = euler_zyz_quaternion(angle, 0.0, 0.0)

    assert np.allclose(wigner_d(quat, 2, 2, 2), np.exp(2j * angle))
    assert np.allclose(wigner_d(quat, 2, -2, -2), np.exp(-2j * angle))
    assert np.allclose(wigner_d(quat, 2, -2, 2), 0.0)
