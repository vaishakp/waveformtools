"""Tests for the waveformtools.comparison fixed-frame core."""

from __future__ import annotations

import numpy as np
import pytest

import waveformtools.comparison  # noqa: F401 - installs ModesArray methods
from waveformtools.comparison import WaveformMetadata, mode_match, residue_distance
from waveformtools.modes_array import ModesArray


def make_test_modes(phase: float = 0.0) -> ModesArray:
    """Construct a small synthetic spin -2 mode set."""

    time_axis = np.linspace(-10.0, 10.0, 256)
    envelope = np.exp(-0.05 * time_axis**2)
    carrier = np.exp(1j * 0.7 * time_axis)

    modes = ModesArray(ell_max=2, time_axis=time_axis, spin_weight=-2)
    modes.create_modes_array(ell_max=2, data_len=len(time_axis))
    modes.set_mode_data(ell=2, emm=2, data=envelope * carrier * np.exp(1j * phase))
    modes.set_mode_data(ell=2, emm=-2, data=0.3 * envelope * np.conjugate(carrier) * np.exp(1j * phase))
    return modes


def test_attach_metadata_round_trip():
    modes = make_test_modes()
    modes.attach_metadata(
        approximant="synthetic",
        parameters={"mass1": 40.0, "mass2": 30.0},
        mode_output_frame="test_frame",
    )

    metadata = modes.get_comparison_metadata()
    assert isinstance(metadata, WaveformMetadata)
    assert metadata.approximant == "synthetic"
    assert metadata.mode_output_frame == "test_frame"
    assert metadata.parameters["mass1"] == 40.0


def test_identity_mode_match_is_one():
    modes = make_test_modes()
    result = modes.match(modes)

    assert result.objective_name == "fixed_frame_mode_match"
    assert result.match == pytest.approx(1.0, abs=1e-12)
    assert result.mismatch == pytest.approx(0.0, abs=1e-12)
    assert result.diagnostics["n_modes"] >= 1


def test_phase_maximized_mode_match_recovers_global_phase():
    reference = make_test_modes(phase=0.0)
    shifted = make_test_modes(phase=0.8)

    phase_maximized = mode_match(reference, shifted, phase_maximize=True)
    fixed_phase = mode_match(reference, shifted, phase_maximize=False)

    assert phase_maximized.match == pytest.approx(1.0, abs=1e-12)
    assert fixed_phase.match < 1.0


def test_residue_distance_zero_for_identical_modes():
    modes = make_test_modes()

    def residual(obj):
        return np.abs(obj.mode(2, 2))

    result = residue_distance(modes, modes, residue_function=residual)
    assert result.distance == pytest.approx(0.0, abs=1e-14)
    assert result.normalized_distance == pytest.approx(0.0, abs=1e-14)
