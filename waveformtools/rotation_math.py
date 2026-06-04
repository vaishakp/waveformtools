"""Dependency-light quaternion and Wigner-D rotation helpers."""

from __future__ import annotations

from math import factorial

import numpy as np


def z_rotation_quaternion(angle: float) -> np.ndarray:
    """Return a unit quaternion for a rotation about the z-axis."""

    return np.array([np.cos(0.5 * angle), 0.0, 0.0, np.sin(0.5 * angle)])
