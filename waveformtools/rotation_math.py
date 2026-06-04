"""Dependency-light quaternion and Wigner-D rotation helpers."""

from __future__ import annotations

from math import atan2, factorial, hypot

import numpy as np


def euler_zyz_quaternion(
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Return the local convention quaternion for z-y-z Euler angles."""

    half_beta = 0.5 * beta
    half_alpha_gamma_sum = 0.5 * (alpha + gamma)
    half_gamma_alpha_diff = 0.5 * (gamma - alpha)
    return np.array(
        [
            np.cos(half_beta) * np.cos(half_alpha_gamma_sum),
            np.sin(half_beta) * np.sin(half_gamma_alpha_diff),
            np.sin(half_beta) * np.cos(half_gamma_alpha_diff),
            np.cos(half_beta) * np.sin(half_alpha_gamma_sum),
        ],
        dtype=float,
    )


def z_rotation_quaternion(angle: float) -> np.ndarray:
    """Return a unit quaternion for a rotation about the z-axis."""

    return np.array([np.cos(0.5 * angle), 0.0, 0.0, np.sin(0.5 * angle)])


def quaternion_from_two_vectors(
    source: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Return a unit quaternion rotating ``source`` onto ``target``."""

    source_unit = _unit_vector(source, "source")
    target_unit = _unit_vector(target, "target")
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if dot > 1.0 - tolerance:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + tolerance:
        axis = _orthogonal_unit_vector(source_unit)
        return np.array([0.0, axis[0], axis[1], axis[2]])

    quat = np.array(
        [
            1.0 + dot,
            *np.cross(source_unit, target_unit),
        ],
        dtype=float,
    )
    return _normalize_quaternion(quat)


def quaternion_rotate_vector(
    quat: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Rotate a three-vector by a unit quaternion."""

    q_array = _normalize_quaternion(np.asarray(quat, dtype=float))
    vector_array = np.asarray(vector, dtype=float)
    if vector_array.shape != (3,):
        raise ValueError("vector must have shape (3,).")
    vector_quat = np.array(
        [0.0, vector_array[0], vector_array[1], vector_array[2]],
        dtype=float,
    )
    rotated = _quaternion_multiply(
        _quaternion_multiply(q_array, vector_quat),
        _quaternion_conjugate(q_array),
    )
    return rotated[1:]


def quaternion_to_euler_zyz(quat: np.ndarray) -> tuple[float, float, float]:
    """Return z-y-z Euler angles matching ``euler_zyz_quaternion``."""

    q_array = _normalize_quaternion(np.asarray(quat, dtype=float))
    if q_array[0] < 0.0:
        q_array = -q_array
    ww, xx, yy, zz = q_array
    sin_half_beta = hypot(xx, yy)
    cos_half_beta = hypot(ww, zz)
    beta = 2.0 * atan2(sin_half_beta, cos_half_beta)

    if sin_half_beta < 1e-14:
        alpha = 2.0 * atan2(zz, ww)
        gamma = 0.0
    elif cos_half_beta < 1e-14:
        gamma_minus_alpha = 2.0 * atan2(xx, yy)
        alpha = -0.5 * gamma_minus_alpha
        gamma = 0.5 * gamma_minus_alpha
    else:
        alpha_plus_gamma = 2.0 * atan2(zz, ww)
        gamma_minus_alpha = 2.0 * atan2(xx, yy)
        alpha = 0.5 * (alpha_plus_gamma - gamma_minus_alpha)
        gamma = 0.5 * (alpha_plus_gamma + gamma_minus_alpha)
    return float(alpha), float(beta), float(gamma)


def wigner_d(quat: np.ndarray, ell: int, emp: int, emm: int) -> complex:
    """Return one Wigner-D matrix element in the local mode convention."""

    if abs(emp) > ell or abs(emm) > ell:
        raise ValueError("Bad Wigner-D indices.")

    q_array = np.asarray(quat, dtype=float)
    ra = np.array([q_array[0] + 1j * q_array[3]])
    rb = np.array([q_array[2] + 1j * q_array[1]])
    ra_small = np.abs(ra) < 1e-12
    rb_small = np.abs(rb) < 1e-12
    regular = np.where((~ra_small) & (~rb_small))[0]
    ra_zero = np.where(ra_small)[0]
    rb_zero = np.where((~ra_small) & rb_small)[0]
    result = np.zeros_like(ra, dtype=np.complex128)

    if emp == -emm:
        result[ra_zero] = rb[ra_zero] ** (2 * emm)

    if emp == emm:
        result[rb_zero] = ra[rb_zero] ** (2 * emm)

    if len(regular):
        ra_reg = ra[regular]
        rb_reg = rb[regular]
        ratio_abs_squared = (np.abs(rb_reg) / np.abs(ra_reg)) ** 2
        rho_min = max(0, emp - emm)
        rho_max = min(ell + emp, ell - emm)
        factor = (
            _wigner_coefficient(ell, emp, emm)
            * (np.abs(ra_reg) ** (2 * (ell - emm)))
            * (ra_reg ** (emm + emp))
            * (rb_reg ** (emm - emp))
        )
        series = 0.0
        for rho in range(rho_max, rho_min - 1, -1):
            term = ((-1) ** rho) * _binomial(ell + emp, rho)
            term *= _binomial(
                ell - emp,
                ell - rho - emm,
            )
            series = term + series * ratio_abs_squared
        result[regular] = factor * series * (ratio_abs_squared**rho_min)
    return complex(result[0])


def _wigner_coefficient(ell: int, emp: int, emm: int) -> float:
    return float(
        np.sqrt(
            factorial(ell + emm)
            * factorial(ell - emm)
            / (factorial(ell + emp) * factorial(ell - emp))
        )
    )


def _binomial(nn: int, kk: int) -> float:
    if kk < 0 or kk > nn:
        return 0.0
    return float(factorial(nn) / (factorial(kk) * factorial(nn - kk)))


def _unit_vector(vector: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} vector must have shape (3,).")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} vector must have nonzero finite norm.")
    return array / norm


def _orthogonal_unit_vector(vector: np.ndarray) -> np.ndarray:
    axis_index = int(np.argmin(np.abs(vector)))
    trial = np.zeros(3, dtype=float)
    trial[axis_index] = 1.0
    orthogonal = np.cross(vector, trial)
    return _unit_vector(orthogonal, "orthogonal")


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    array = np.asarray(quat, dtype=float)
    if array.shape != (4,):
        raise ValueError("quat must have shape (4,).")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("quat must have nonzero finite norm.")
    return array / norm


def _quaternion_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )
