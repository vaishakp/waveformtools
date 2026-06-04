"""Tests for dependency-light rotation math helpers."""

from __future__ import annotations

import numpy as np

from waveformtools.rotation_math import (
    euler_zyz_quaternion,
    quaternion_from_two_vectors,
    quaternion_rotate_vector,
    quaternion_to_euler_zyz,
    wigner_d,
)


def test_rotation_math_imports_without_scri_side_effects():
    quat = euler_zyz_quaternion(0.0, 0.0, 0.0)

    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_wigner_d_z_axis_limit():
    angle = 0.4
    quat = euler_zyz_quaternion(angle, 0.0, 0.0)

    assert np.allclose(wigner_d(quat, 2, 2, 2), np.exp(2j * angle))
    assert np.allclose(wigner_d(quat, 2, -2, -2), np.exp(-2j * angle))
    assert np.allclose(wigner_d(quat, 2, -2, 2), 0.0)


def test_quaternion_from_two_vectors_rotates_source_to_target():
    source = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])

    quat = quaternion_from_two_vectors(source, target)
    rotated = quaternion_rotate_vector(quat, source)

    np.testing.assert_allclose(rotated, target, atol=1e-14)


def test_quaternion_to_euler_zyz_round_trips_local_convention():
    quat = quaternion_from_two_vectors(
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 1.0]),
    )

    alpha, beta, gamma = quaternion_to_euler_zyz(quat)
    round_trip = euler_zyz_quaternion(alpha, beta, gamma)

    np.testing.assert_allclose(round_trip, quat, atol=1e-14)
