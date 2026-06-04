"""Dependency-light quaternion and Wigner-D rotation helpers."""

from __future__ import annotations

from math import factorial

import numpy as np


def euler_zyz_quaternion(alpha: float, beta: float, gamma: float) -> np.ndarray:
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
            series = (
                ((-1) ** rho)
                * _binomial(ell + emp, rho)
                * _binomial(ell - emp, ell - rho - emm)
                + series * ratio_abs_squared
            )
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
