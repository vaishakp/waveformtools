"""Tests for dependency-light rotation math helpers."""

from __future__ import annotations

import numpy as np

from waveformtools.rotation_math import (
    axis_angle_quaternion,
    euler_zyz_quaternion,
    quaternion_from_two_vectors,
    quaternion_multiply,
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


def test_axis_angle_quaternion_rotates_about_axis():
    quat = axis_angle_quaternion(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)

    rotated = quaternion_rotate_vector(quat, np.array([1.0, 0.0, 0.0]))

    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-14)


def test_quaternion_multiply_composes_rotations():
    rotate_x_to_z = quaternion_from_two_vectors(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    rotate_y_to_x_about_z = axis_angle_quaternion(
        np.array([0.0, 0.0, 1.0]),
        -np.pi / 2.0,
    )
    composed = quaternion_multiply(rotate_y_to_x_about_z, rotate_x_to_z)

    np.testing.assert_allclose(
        quaternion_rotate_vector(composed, np.array([1.0, 0.0, 0.0])),
        [0.0, 0.0, 1.0],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        quaternion_rotate_vector(composed, np.array([0.0, 1.0, 0.0])),
        [1.0, 0.0, 0.0],
        atol=1e-14,
    )
