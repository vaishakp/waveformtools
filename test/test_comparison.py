"""Tests for the waveformtools.comparison fixed-frame core."""

from __future__ import annotations

import numpy as np
import pytest

import waveformtools.comparison  # noqa: F401 - installs ModesArray methods
from waveformtools.comparison import AlignmentSpec, WaveformMetadata, mode_match, residue_distance
from waveformtools.modes_array import ModesArray


def make_test_modes(
    phase: float = 0.0,
    *,
    time_axis: np.ndarray | None = None,
    time_shift: float = 0.0,
    orbital_phase: float = 0.0,
) -> ModesArray:
    """Construct a small synthetic spin -2 mode set.

    ``time_shift`` shifts the waveform content relative to the supplied time
    axis. ``orbital_phase`` applies the physical mode rotation
    ``h_lm -> exp(i m orbital_phase) h_lm``.
    """

    if time_axis is None:
        time_axis = np.linspace(-10.0, 10.0, 256)
    source_time = time_axis - time_shift
    envelope = np.exp(-0.05 * source_time**2)
    carrier = np.exp(1j * 0.7 * source_time)

    modes = ModesArray(ell_max=2, time_axis=time_axis, spin_weight=-2)
    modes.create_modes_array(ell_max=2, data_len=len(time_axis))
    modes.set_mode_data(
        ell=2,
        emm=2,
        data=envelope * carrier * np.exp(1j * phase) * np.exp(2j * orbital_phase),
    )
    modes.set_mode_data(
        ell=2,
        emm=-2,
        data=0.3
        * envelope
        * np.conjugate(carrier)
        * np.exp(1j * phase)
        * np.exp(-2j * orbital_phase),
    )
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

    assert result.objective_name == "mode_match"
    assert result.match == pytest.approx(1.0, abs=1e-12)
    assert result.mismatch == pytest.approx(0.0, abs=1e-12)
    assert result.diagnostics["n_modes"] >= 1
    assert result.diagnostics["alignment"]["time_alignment"] == "peak_22"
    assert result.diagnostics["alignment"]["time_domain_policy"] == "crop_to_overlap"


def test_phase_maximized_mode_match_recovers_global_phase():
    reference = make_test_modes(phase=0.0)
    shifted = make_test_modes(phase=0.8)

    phase_maximized = mode_match(reference, shifted, phase_alignment="global_complex")
    fixed_phase = mode_match(reference, shifted, phase_alignment="none")

    assert phase_maximized.match == pytest.approx(1.0, abs=1e-12)
    assert fixed_phase.match < 1.0


def test_orbital_phase_alignment_uses_single_m_dependent_phase():
    reference = make_test_modes(orbital_phase=0.0)
    shifted = make_test_modes(orbital_phase=0.5)

    fixed_phase = mode_match(reference, shifted, phase_alignment="none")
    aligned = mode_match(
        reference,
        shifted,
        phase_alignment="orbital_phase_and_global",
        alignment=AlignmentSpec(orbital_phase_samples=721),
    )

    assert fixed_phase.match < 1.0
    assert aligned.match > 0.999
    assert aligned.best_parameters["orbital_phase"] is not None


def test_default_alignment_crops_unequal_lengths_after_peak_alignment():
    dt = 0.1
    reference_time = np.arange(-10.0, 10.0 + 0.5 * dt, dt)
    candidate_time = np.arange(-7.0, 6.0 + 0.5 * dt, dt)
    reference = make_test_modes(time_axis=reference_time, time_shift=0.0)
    candidate = make_test_modes(time_axis=candidate_time, time_shift=0.0)

    result = mode_match(reference, candidate)

    assert result.match == pytest.approx(1.0, abs=1e-12)
    assert result.diagnostics["time_axis"]["n_samples"] == len(candidate_time)
    assert result.diagnostics["time_axis"]["policy"] == "crop_to_overlap"
    assert result.diagnostics["alignment"]["time_alignment"] == "peak_22"


def test_strict_time_policy_rejects_different_lengths():
    reference = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 256))
    shorter = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 200))

    with pytest.raises(ValueError, match="common time grid"):
        mode_match(reference, shorter, time_domain_policy="error")


def test_crop_to_overlap_rejects_different_sampling_rates():
    reference = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 256))
    different_dt = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 300))

    with pytest.raises(ValueError, match="compatible sampling rates"):
        mode_match(reference, different_dt, time_domain_policy="crop_to_overlap")


def test_resample_to_reference_handles_different_sampling_rates():
    reference = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 256))
    candidate = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 300))

    result = mode_match(
        reference,
        candidate,
        time_domain_policy="resample_to_reference",
        resample_method="linear",
    )

    assert result.match > 0.999
    assert result.diagnostics["time_axis"]["n_samples"] == len(reference.time_axis)
    assert result.diagnostics["time_axis"]["policy"] == "resample_to_reference"


def test_peak_total_power_alignment_handles_shifted_peak():
    reference = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 256), time_shift=0.0)
    shifted = make_test_modes(time_axis=np.linspace(-7.0, 13.0, 256), time_shift=3.0)

    unaligned = mode_match(
        reference,
        shifted,
        time_alignment="none",
        time_domain_policy="resample_to_reference",
        phase_alignment="global_complex",
    )
    aligned = mode_match(
        reference,
        shifted,
        time_alignment="peak_total_power",
        time_domain_policy="resample_to_reference",
        phase_alignment="global_complex",
    )

    assert aligned.match > unaligned.match
    assert aligned.match > 0.999
    assert aligned.diagnostics["time_axis"]["reference_time_b"] != 0.0


def test_residue_distance_zero_for_identical_modes():
    modes = make_test_modes()

    def residual(obj):
        return np.abs(obj.mode(2, 2))

    result = residue_distance(modes, modes, residue_function=residual)
    assert result.distance == pytest.approx(0.0, abs=1e-14)
    assert result.normalized_distance == pytest.approx(0.0, abs=1e-14)
