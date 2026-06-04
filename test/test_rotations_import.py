"""Import-hygiene tests for the legacy rotations module."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from waveformtools.rotation_math import z_rotation_quaternion


def test_rotations_import_without_optional_dependencies_at_import_time():
    completed = subprocess.run(
        [sys.executable, "-c", "import waveformtools.rotations; print('ok')"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"


def test_rotations_compatibility_wrappers_use_pure_rotation_math():
    from waveformtools import rotations

    angle = 0.4
    quat = rotations.zRotationQuat(angle)

    assert np.allclose(quat, z_rotation_quaternion(angle))
    assert np.allclose(rotations._wignerD(quat, 2, 2, 2), [np.exp(2j * angle)])
