"""Tests for the waveformtools.comparison fixed-frame core."""

from __future__ import annotations

import numpy as np
import pytest

import waveformtools.comparison  # noqa: F401 - installs ModesArray methods
from waveformtools.comparison import (
    AlignmentSpec,
    FittingFactorConfig,
    ModeComparisonConfig,
    RotationSpec,
    WaveformMetadata,
    canonicalize_modes_for_comparison,
    fixed_candidate_fitting_factor,
    mode_convention_for_approximant,
    mode_match,
    prepare_aligned_mode_data,
    residue_distance,
    rotate_modes,
    standardize_generated_modes_in_place,
)
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
        data=envelope
        * carrier
        * np.exp(1j * phase)
        * np.exp(2j * orbital_phase),
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


def make_full_ell2_modes(*, time_axis: np.ndarray | None = None) -> ModesArray:
    """Construct synthetic data with all ell=2 modes populated."""

    if time_axis is None:
        time_axis = np.linspace(-6.0, 6.0, 160)
    envelope = np.exp(-0.04 * time_axis**2)
    modes = ModesArray(ell_max=2, time_axis=time_axis, spin_weight=-2)
    modes.create_modes_array(ell_max=2, data_len=len(time_axis))
    for index, emm in enumerate(range(-2, 3), start=1):
        amplitude = 0.15 + 0.12 * index
        frequency = 0.25 + 0.08 * index
        phase = 0.17 * index
        modes.set_mode_data(
            ell=2,
            emm=emm,
            data=amplitude
            * envelope
            * np.exp(1j * (frequency * time_axis + phase)),
        )
    return modes


def circular_phase_distance(phase_a: float, phase_b: float) -> float:
    return float(abs(np.angle(np.exp(1j * (phase_a - phase_b)))))


def normalized_rms_from_arrays(reference, candidate, selected_modes) -> float:
    numerator = 0.0
    denominator = 0.0
    for mode in selected_modes:
        diff = reference[mode] - candidate[mode]
        numerator += float(np.mean(np.abs(diff) ** 2))
        denominator += float(np.mean(np.abs(reference[mode]) ** 2))
    return float(np.sqrt(numerator / max(denominator, 1e-300)))


def normalized_inner_from_arrays(
    reference, candidate, selected_modes, time_axis
) -> float:
    inner = 0.0j
    norm_a = 0.0j
    norm_b = 0.0j
    for mode in selected_modes:
        inner += np.trapezoid(
            np.conjugate(reference[mode]) * candidate[mode], time_axis
        )
        norm_a += np.trapezoid(
            np.conjugate(reference[mode]) * reference[mode], time_axis
        )
        norm_b += np.trapezoid(
            np.conjugate(candidate[mode]) * candidate[mode], time_axis
        )
    return float(inner.real / np.sqrt(norm_a.real * norm_b.real))


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
    assert (
        result.diagnostics["alignment"]["time_alignment"]
        == "peak_total_news_power"
    )
    assert (
        result.diagnostics["alignment"]["time_domain_policy"]
        == "crop_to_overlap"
    )


def test_phase_maximized_mode_match_recovers_global_phase():
    reference = make_test_modes(phase=0.0)
    shifted = make_test_modes(phase=0.8)

    phase_maximized = mode_match(
        reference, shifted, phase_alignment="global_complex"
    )
    fixed_phase = mode_match(reference, shifted, phase_alignment="none")

    assert phase_maximized.match == pytest.approx(1.0, abs=1e-12)
    assert fixed_phase.match < 1.0


def test_global_phase_alignment_is_replayable():
    phase = 0.8
    reference = make_test_modes(phase=0.0)
    shifted = make_test_modes(phase=phase)

    unaligned = mode_match(
        reference,
        shifted,
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="none",
    )
    aligned = mode_match(
        reference,
        shifted,
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="global_complex",
    )
    replay = prepare_aligned_mode_data(reference, shifted, aligned)

    assert aligned.match == pytest.approx(1.0, abs=1e-12)
    assert aligned.best_parameters["orbital_phase"] == 0.0
    assert aligned.best_parameters["global_phase"] == pytest.approx(-phase)
    assert (
        normalized_rms_from_arrays(
            replay.reference_modes,
            replay.candidate_modes,
            replay.selected_modes,
        )
        < 1e-12
    )
    assert aligned.match > unaligned.match


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


def test_h22_only_orbital_and_global_phase_alignment_is_degenerate():
    phase = 0.7
    reference = make_test_modes(phase=0.0)
    shifted = make_test_modes(phase=phase)

    result = mode_match(
        reference,
        shifted,
        modes=[(2, 2)],
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="orbital_phase_and_global",
    )
    replay = prepare_aligned_mode_data(reference, shifted, result)

    orbital_only_candidate = {
        mode: np.exp(1j * mode[1] * result.best_parameters["orbital_phase"])
        * shifted.mode(*mode)
        for mode in replay.selected_modes
    }

    assert result.match == pytest.approx(1.0, abs=1e-12)
    assert result.diagnostics["orbital_phase_degenerate"] is True
    assert result.best_parameters["orbital_phase"] == pytest.approx(0.0)
    assert result.best_parameters["global_phase"] == pytest.approx(-phase)
    assert (
        normalized_rms_from_arrays(
            replay.reference_modes,
            replay.candidate_modes,
            replay.selected_modes,
        )
        < 1e-12
    )
    assert (
        normalized_rms_from_arrays(
            replay.reference_modes,
            orbital_only_candidate,
            replay.selected_modes,
        )
        > 0.1
    )


@pytest.mark.parametrize("approximant", ["SEOBNRv5PHM", "SEOBNRv5HM"])
def test_registered_v5_eob_modes_are_canonicalized_before_match(approximant):
    reference = make_test_modes()
    reference.attach_metadata(
        approximant="NRSur7dq4",
        mode_convention="canonical_strain_lm",
    )
    raw_seob = make_test_modes().bar()
    raw_seob.attach_metadata(approximant=approximant)

    uncanonicalized = mode_match(
        reference,
        raw_seob,
        modes=[(2, 2)],
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="orbital_phase",
        canonicalize_mode_conventions=False,
    )
    canonicalized = mode_match(
        reference,
        raw_seob,
        modes=[(2, 2)],
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="orbital_phase",
    )
    replay = prepare_aligned_mode_data(reference, raw_seob, canonicalized)

    assert mode_convention_for_approximant("NRSur7dq4") is None
    assert mode_convention_for_approximant(approximant) is not None
    assert uncanonicalized.match < 0.1
    assert canonicalized.match == pytest.approx(1.0, abs=1e-12)
    assert canonicalized.diagnostics["mode_conventions"]["candidate"][
        "applied"
    ] is True
    assert canonicalized.diagnostics["mode_conventions"]["candidate"][
        "canonical_transform"
    ] == "complex_conjugate"
    assert (
        canonicalized.candidate_metadata.canonicalization_applied
        == "complex_conjugate"
    )
    assert (
        normalized_rms_from_arrays(
            replay.reference_modes,
            replay.candidate_modes,
            replay.selected_modes,
        )
        < 1e-10
    )


def test_nrsur_modes_are_treated_as_canonical():
    reference = make_test_modes()
    raw_nrsur = make_test_modes()
    raw_nrsur.attach_metadata(approximant="NRSur7dq4")

    canonicalized, diagnostics = canonicalize_modes_for_comparison(raw_nrsur)
    result = mode_match(
        reference,
        raw_nrsur,
        modes=[(2, 2)],
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="orbital_phase",
    )

    assert canonicalized is raw_nrsur
    assert diagnostics["registered"] is False
    assert diagnostics["applied"] is False
    assert result.match == pytest.approx(1.0, abs=1e-12)


def test_mode_convention_registry_uses_exact_approximant_keys():
    reference = make_test_modes()
    fake_sur = make_test_modes().bar()
    fake_sur.attach_metadata(approximant="FakeSurModel")

    result = mode_match(
        reference,
        fake_sur,
        modes=[(2, 2)],
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="orbital_phase",
    )

    assert mode_convention_for_approximant("FakeSurModel") is None
    assert result.match < 0.1
    assert result.diagnostics["mode_conventions"]["candidate"][
        "registered"
    ] is False
    assert result.diagnostics["mode_conventions"]["candidate"][
        "applied"
    ] is False


@pytest.mark.parametrize("approximant", ["SEOBNRv5PHM", "SEOBNRv5HM"])
def test_mode_convention_canonicalization_is_idempotent(approximant):
    reference = make_test_modes()
    raw_seob = make_test_modes().bar()
    raw_seob.attach_metadata(approximant=approximant)

    canonical_seob, first_diagnostics = canonicalize_modes_for_comparison(
        raw_seob
    )
    canonical_seob, second_diagnostics = canonicalize_modes_for_comparison(
        canonical_seob
    )
    result = mode_match(
        reference,
        canonical_seob,
        modes=[(2, 2)],
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="orbital_phase",
    )

    assert first_diagnostics["applied"] is True
    assert second_diagnostics["already_canonical"] is True
    assert second_diagnostics["applied"] is False
    assert result.match == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("approximant", ["SEOBNRv5PHM", "SEOBNRv5HM"])
def test_generation_standardization_mutates_v5_eob_modes_and_records_history(
    approximant,
):
    reference = make_test_modes()
    raw_seob = make_test_modes().bar()
    raw_seob.attach_metadata(approximant=approximant)

    standardized, diagnostics = standardize_generated_modes_in_place(raw_seob)

    assert standardized is raw_seob
    assert diagnostics["applied"] is True
    assert diagnostics["canonical_transform"] == "complex_conjugate"
    assert np.allclose(raw_seob.mode(2, 2), reference.mode(2, 2))
    metadata = raw_seob.get_comparison_metadata()
    assert metadata.mode_convention == "canonical_strain_lm"
    assert metadata.raw_mode_convention == (
        "pyseobnr_complex_conjugate_strain_lm"
    )
    assert metadata.canonicalization_applied == "complex_conjugate"
    assert metadata.mode_convention_history[-1]["stage"] == (
        "waveform_generation"
    )
    assert (
        "standardize_mode_convention(complex_conjugate)"
        in raw_seob.actions
    )

    saved_metadata = raw_seob.get_metadata()["comparison_metadata"]
    assert isinstance(saved_metadata, dict)
    assert saved_metadata["mode_convention_history"][-1]["stage"] == (
        "waveform_generation"
    )


@pytest.mark.parametrize("approximant", ["SEOBNRv5PHM", "SEOBNRv5HM"])
def test_generation_standardization_is_idempotent(approximant):
    raw_seob = make_test_modes().bar()
    raw_seob.attach_metadata(approximant=approximant)

    standardize_generated_modes_in_place(raw_seob)
    modes_after_first = np.array(raw_seob.modes_data, copy=True)
    actions_after_first = raw_seob.actions
    _, diagnostics = standardize_generated_modes_in_place(raw_seob)

    assert diagnostics["already_canonical"] is True
    assert diagnostics["applied"] is False
    assert np.allclose(raw_seob.modes_data, modes_after_first)
    assert raw_seob.actions == actions_after_first
    assert len(raw_seob.get_comparison_metadata().mode_convention_history) == 1


@pytest.mark.parametrize("approximant", ["SEOBNRv5PHM", "SEOBNRv5HM"])
def test_waveform_model_generation_hook_standardizes_and_merges_metadata(
    approximant,
):
    from waveformtools.models.waveform_models import WaveformModel

    model = WaveformModel(
        {
            "approximant": approximant,
            "mass1": 40.0,
            "mass2": 20.0,
            "spin1x": 0.0,
            "spin1y": 0.0,
            "spin1z": 0.0,
            "spin2x": 0.0,
            "spin2y": 0.0,
            "spin2z": 0.0,
            "f_lower": 0.0,
            "f_ref": 20.0,
        }
    )
    reference = make_test_modes()
    raw_seob = make_test_modes().bar()
    raw_seob.attach_metadata(
        parameters={"preexisting": "kept"},
        mode_convention_history=[{"stage": "pre_generation"}],
    )

    standardized = model._standardize_generated_modes(
        raw_seob,
        domain="td",
        dimensionless=True,
        generator="synthetic_generator",
    )

    assert standardized is raw_seob
    assert np.allclose(standardized.mode(2, 2), reference.mode(2, 2))
    metadata = standardized.get_comparison_metadata()
    assert metadata.approximant == approximant
    assert metadata.parameters["preexisting"] == "kept"
    assert metadata.parameters["generation_domain"] == "td"
    assert metadata.parameters["dimensionless_output"] is True
    assert metadata.mass_ratio == pytest.approx(2.0)
    assert metadata.generator == "synthetic_generator"
    assert [item["stage"] for item in metadata.mode_convention_history] == [
        "pre_generation",
        "waveform_generation",
    ]
    assert standardized.mode_convention_diagnostics["applied"] is True


def test_continuous_orbital_phase_alignment_refines_grid_result():
    orbital_phase = 0.241
    reference = make_full_ell2_modes()
    shifted = rotate_modes(
        reference,
        RotationSpec(kind="z_axis", angle=orbital_phase),
    )
    grid_alignment = AlignmentSpec(
        time_alignment="none",
        phase_alignment="orbital_phase",
        orbital_phase_samples=16,
        orbital_phase_optimizer="grid",
    )
    continuous_alignment = AlignmentSpec(
        time_alignment="none",
        phase_alignment="orbital_phase",
        orbital_phase_samples=16,
        orbital_phase_optimizer="continuous",
    )

    grid_result = mode_match(reference, shifted, alignment=grid_alignment)
    continuous_result = mode_match(
        reference,
        shifted,
        alignment=continuous_alignment,
    )
    replay = prepare_aligned_mode_data(reference, shifted, continuous_result)

    expected_phase = (-orbital_phase) % (2.0 * np.pi)
    assert continuous_result.match >= grid_result.match
    assert continuous_result.match == pytest.approx(1.0, abs=1e-10)
    assert grid_result.match < 0.999
    assert (
        circular_phase_distance(
            continuous_result.best_parameters["orbital_phase"],
            expected_phase,
        )
        < 1e-5
    )
    assert continuous_result.best_parameters["global_phase"] == 0.0
    assert (
        normalized_rms_from_arrays(
            replay.reference_modes,
            replay.candidate_modes,
            replay.selected_modes,
        )
        < 1e-5
    )
    assert continuous_result.optimizer == "continuous_orbital_phase"


def test_continuous_orbital_phase_and_global_refines_grid_result():
    orbital_phase = 0.317
    reference = make_full_ell2_modes()
    shifted = rotate_modes(
        reference,
        RotationSpec(kind="z_axis", angle=orbital_phase),
    )
    grid_alignment = AlignmentSpec(
        time_alignment="none",
        phase_alignment="orbital_phase_and_global",
        orbital_phase_samples=16,
        orbital_phase_optimizer="grid",
    )
    continuous_alignment = AlignmentSpec(
        time_alignment="none",
        phase_alignment="orbital_phase_and_global",
        orbital_phase_samples=16,
        orbital_phase_optimizer="continuous",
    )

    grid_result = mode_match(reference, shifted, alignment=grid_alignment)
    continuous_result = mode_match(
        reference,
        shifted,
        alignment=continuous_alignment,
    )
    replay = prepare_aligned_mode_data(reference, shifted, continuous_result)

    assert continuous_result.match >= grid_result.match
    assert continuous_result.match == pytest.approx(1.0, abs=1e-10)
    assert continuous_result.diagnostics["orbital_phase_degenerate"] is False
    assert (
        normalized_rms_from_arrays(
            replay.reference_modes,
            replay.candidate_modes,
            replay.selected_modes,
        )
        < 1e-5
    )
    assert continuous_result.optimizer == "continuous_orbital_phase"


def test_alignment_rejects_unknown_orbital_phase_optimizer():
    with pytest.raises(ValueError, match="orbital_phase_optimizer"):
        AlignmentSpec(orbital_phase_optimizer="not-an-optimizer")


def test_z_axis_rotation_applies_one_coherent_mode_phase():
    angle = 0.4
    modes = make_test_modes(orbital_phase=0.0)

    rotated = rotate_modes(modes, RotationSpec(kind="z_axis", angle=angle))

    assert np.allclose(
        rotated.mode(2, 2), np.exp(2j * angle) * modes.mode(2, 2)
    )
    assert np.allclose(
        rotated.mode(2, -2), np.exp(-2j * angle) * modes.mode(2, -2)
    )
    assert np.allclose(
        modes.mode(2, 2), make_test_modes(orbital_phase=0.0).mode(2, 2)
    )


def test_wigner_identity_rotation_matches_input_modes():
    modes = make_test_modes()

    rotated = rotate_modes(modes, RotationSpec(kind="wigner"))

    assert np.allclose(rotated.mode(2, 2), modes.mode(2, 2))
    assert np.allclose(rotated.mode(2, -2), modes.mode(2, -2))


def test_wigner_z_axis_limit_matches_z_axis_rotation():
    angle = 0.4
    modes = make_test_modes()

    z_axis = rotate_modes(modes, RotationSpec(kind="z_axis", angle=angle))
    wigner = rotate_modes(modes, RotationSpec(kind="wigner", alpha=angle))

    assert np.allclose(wigner.mode(2, 2), z_axis.mode(2, 2))
    assert np.allclose(wigner.mode(2, -2), z_axis.mode(2, -2))


def test_wigner_rotation_preserves_unselected_modes():
    angle = 0.4
    modes = make_test_modes()
    original_m2 = np.array(modes.mode(2, -2), copy=True)

    rotated = rotate_modes(
        modes,
        RotationSpec(kind="wigner", alpha=angle),
        modes=[(2, 2)],
    )

    assert np.allclose(
        rotated.mode(2, 2), np.exp(2j * angle) * modes.mode(2, 2)
    )
    assert np.allclose(rotated.mode(2, -2), original_m2)


def test_wigner_rotation_preserves_same_ell_power():
    time_axis = np.linspace(-1.0, 1.0, 16)
    modes = ModesArray(ell_max=2, time_axis=time_axis, spin_weight=-2)
    modes.create_modes_array(ell_max=2, data_len=len(time_axis))
    signal = 1.0 + 0.1 * time_axis
    modes.set_mode_data(ell=2, emm=2, data=signal)

    rotated = rotate_modes(modes, RotationSpec(kind="wigner", beta=0.3))

    original_power = sum(
        np.abs(modes.mode(2, emm)) ** 2 for emm in range(-2, 3)
    )
    rotated_power = sum(
        np.abs(rotated.mode(2, emm)) ** 2 for emm in range(-2, 3)
    )
    assert np.allclose(rotated_power, original_power, atol=1e-12)
    assert np.max(np.abs(rotated.mode(2, 1))) > 0.0


def test_mode_match_accepts_fixed_candidate_rotation():
    angle = 0.4
    reference = make_test_modes(orbital_phase=0.0)
    shifted = make_test_modes(orbital_phase=angle)

    unrotated = mode_match(
        reference,
        shifted,
        time_alignment="none",
        phase_alignment="none",
    )
    rotated = mode_match(
        reference,
        shifted,
        time_alignment="none",
        phase_alignment="none",
        rotation={"kind": "z_axis", "angle": -angle},
    )

    assert unrotated.match < 1.0
    assert rotated.match == pytest.approx(1.0, abs=1e-12)
    assert rotated.diagnostics["rotation"]["kind"] == "z_axis"
    assert rotated.best_parameters["rotation"]["angle"] == pytest.approx(-angle)


def test_mode_match_optimizes_one_shared_z_rotation():
    angle = 0.35
    reference = make_test_modes(orbital_phase=0.0)
    shifted = make_test_modes(orbital_phase=angle)

    unoptimized = mode_match(
        reference,
        shifted,
        time_alignment="none",
        phase_alignment="none",
    )
    optimized = mode_match(
        reference,
        shifted,
        time_alignment="none",
        phase_alignment="none",
        rotation={
            "kind": "z_axis",
            "optimize_angle": True,
            "angle_bounds": (-0.8, 0.0),
        },
    )

    assert optimized.match > unoptimized.match
    assert optimized.match == pytest.approx(1.0, abs=1e-10)
    assert optimized.best_parameters["rotation"]["angle"] == pytest.approx(
        -angle, abs=2e-3
    )
    assert optimized.diagnostics["rotation"]["optimize_angle"] is True
    assert (
        optimized.diagnostics["rotation_optimization"]["parameter"]
        == "rotation.angle"
    )
    assert optimized.optimizer == "bounded_z_rotation"


def test_z_rotation_optimization_rejects_orbital_phase_degeneracy():
    reference = make_test_modes(orbital_phase=0.0)
    shifted = make_test_modes(orbital_phase=0.2)

    with pytest.raises(ValueError, match="degenerate"):
        mode_match(
            reference,
            shifted,
            time_alignment="none",
            phase_alignment="orbital_phase_and_global",
            rotation=RotationSpec(kind="z_axis", optimize_angle=True),
        )


def test_mode_match_optimizes_wigner_beta_rotation():
    reference = make_full_ell2_modes()
    candidate = rotate_modes(reference, RotationSpec(kind="wigner", beta=0.25))

    unoptimized = mode_match(
        reference,
        candidate,
        time_alignment="none",
        phase_alignment="none",
    )
    optimized = mode_match(
        reference,
        candidate,
        time_alignment="none",
        phase_alignment="none",
        rotation=RotationSpec(
            kind="wigner",
            optimize_parameters=("beta",),
            parameter_bounds={"beta": (-0.5, 0.1)},
        ),
    )

    assert optimized.match > unoptimized.match
    assert optimized.match > 0.999
    assert optimized.best_parameters["rotation"]["beta"] == pytest.approx(
        -0.25, abs=2e-3
    )
    assert optimized.diagnostics["rotation_optimization"]["parameters"] == (
        "beta",
    )
    assert optimized.optimizer == "bounded_wigner_rotation"


def test_mode_match_optimizes_full_wigner_rotation():
    reference = make_full_ell2_modes()
    alpha = 0.12
    beta = 0.18
    gamma = -0.09
    candidate = rotate_modes(
        reference,
        RotationSpec(kind="wigner", alpha=alpha, beta=beta, gamma=gamma),
    )

    optimized = mode_match(
        reference,
        candidate,
        time_alignment="none",
        phase_alignment="none",
        rotation=RotationSpec(
            kind="wigner",
            optimize_parameters=("alpha", "beta", "gamma"),
            parameter_bounds={
                "alpha": (-0.1, 0.2),
                "beta": (-0.3, 0.0),
                "gamma": (-0.2, 0.1),
            },
        ),
    )

    assert optimized.match > 0.999
    assert optimized.best_parameters["rotation"]["alpha"] == pytest.approx(
        -gamma, abs=1e-2
    )
    assert optimized.best_parameters["rotation"]["beta"] == pytest.approx(
        -beta, abs=1e-2
    )
    assert optimized.best_parameters["rotation"]["gamma"] == pytest.approx(
        -alpha, abs=1e-2
    )


def test_wigner_rotation_optimization_rejects_phase_degeneracy_by_default():
    reference = make_full_ell2_modes()
    candidate = rotate_modes(reference, RotationSpec(kind="wigner", beta=0.1))

    with pytest.raises(ValueError, match="degenerate"):
        mode_match(
            reference,
            candidate,
            time_alignment="none",
            phase_alignment="orbital_phase_and_global",
            rotation=RotationSpec(
                kind="wigner",
                optimize_parameters=("alpha", "beta"),
                parameter_bounds={"alpha": (-0.2, 0.2), "beta": (-0.2, 0.0)},
            ),
        )


def test_wigner_rotation_optimization_can_opt_into_phase_degeneracy():
    reference = make_full_ell2_modes()
    candidate = rotate_modes(reference, RotationSpec(kind="wigner", beta=0.1))

    result = mode_match(
        reference,
        candidate,
        alignment=AlignmentSpec(
            time_alignment="none",
            phase_alignment="orbital_phase_and_global",
            allow_phase_rotation_degeneracy=True,
        ),
        rotation=RotationSpec(
            kind="wigner",
            optimize_parameters=("alpha", "beta"),
            parameter_bounds={"alpha": (-0.2, 0.2), "beta": (-0.2, 0.0)},
        ),
    )

    assert result.match > 0.999
    assert (
        result.diagnostics["rotation_optimization"]["phase_degeneracy_possible"]
        is True
    )


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
    assert (
        result.diagnostics["alignment"]["time_alignment"]
        == "peak_total_news_power"
    )


def test_strict_time_policy_rejects_different_lengths():
    reference = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 256))
    shorter = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 200))

    with pytest.raises(ValueError, match="common time grid"):
        mode_match(reference, shorter, time_domain_policy="error")


def test_crop_to_overlap_rejects_different_sampling_rates():
    reference = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 256))
    different_dt = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 300))

    with pytest.raises(ValueError, match="compatible sampling rates"):
        mode_match(
            reference, different_dt, time_domain_policy="crop_to_overlap"
        )


def test_resample_to_reference_handles_different_sampling_rates():
    reference = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 256))
    candidate = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 300))

    result = mode_match(
        reference,
        candidate,
        time_alignment="none",
        time_domain_policy="resample_to_reference",
        resample_method="linear",
    )

    assert result.match > 0.999
    assert result.diagnostics["time_axis"]["n_samples"] == len(
        reference.time_axis
    )
    assert result.diagnostics["time_axis"]["policy"] == "resample_to_reference"
    assert result.diagnostics["alignment"]["time_alignment"] == "none"


def test_peak_22_resample_to_reference_uses_valid_overlap_for_different_sampling_rates():
    reference = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 256))
    candidate = make_test_modes(time_axis=np.linspace(-10.0, 10.0, 300))

    result = mode_match(
        reference,
        candidate,
        time_alignment="peak_22",
        time_domain_policy="resample_to_reference",
        resample_method="linear",
    )

    assert result.match > 0.999
    assert result.diagnostics["time_axis"]["n_samples"] <= len(
        reference.time_axis
    )
    assert (
        result.diagnostics["time_axis"]["n_samples"]
        >= len(reference.time_axis) - 1
    )
    assert result.diagnostics["time_axis"]["policy"] == "resample_to_reference"
    assert result.diagnostics["alignment"]["time_alignment"] == "peak_22"


def test_peak_total_power_alignment_handles_shifted_peak():
    reference = make_test_modes(
        time_axis=np.linspace(-10.0, 10.0, 256), time_shift=0.0
    )
    shifted = make_test_modes(
        time_axis=np.linspace(-7.0, 13.0, 256), time_shift=3.0
    )

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


def test_peak_total_news_power_alignment_handles_shifted_peak():
    reference = make_test_modes(
        time_axis=np.linspace(-10.0, 10.0, 256), time_shift=0.0
    )
    shifted = make_test_modes(
        time_axis=np.linspace(-7.0, 13.0, 256), time_shift=3.0
    )

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
        time_alignment="peak_total_news_power",
        time_domain_policy="resample_to_reference",
        phase_alignment="global_complex",
    )

    assert aligned.match > unaligned.match
    assert aligned.match > 0.999
    assert (
        aligned.diagnostics["alignment"]["time_alignment"]
        == "peak_total_news_power"
    )


def test_get_news_from_strain_honors_method_argument():
    time_axis = np.linspace(-10.0, 10.0, 512)
    omega = 0.7
    modes = ModesArray(ell_max=2, time_axis=time_axis, spin_weight=-2)
    modes.create_modes_array(ell_max=2, data_len=len(time_axis))
    signal = np.exp(1j * omega * time_axis)
    modes.set_mode_data(ell=2, emm=2, data=signal)

    news = modes.get_news_from_strain(method="spline")
    expected = 1j * omega * signal

    assert np.max(np.abs(news.mode(2, 2)[8:-8] - expected[8:-8])) < 1e-5


def test_modes_array_reflected_subtraction_uses_left_operand():
    time_axis = np.linspace(-1.0, 1.0, 8)
    left = make_test_modes(time_axis=time_axis)
    right = make_test_modes(time_axis=time_axis)
    left.set_mode_data(ell=2, emm=2, data=3.0 * np.ones_like(time_axis))
    right.set_mode_data(ell=2, emm=2, data=np.ones_like(time_axis))

    reflected = right.__rsub__(left)

    assert np.allclose(reflected.mode(2, 2), 2.0)


def test_time_shift_optimization_recovers_one_shared_shift():
    time_axis = np.linspace(-12.0, 12.0, 512)
    known_shift = 0.35
    reference = make_test_modes(time_axis=time_axis, time_shift=0.0)
    shifted = make_test_modes(time_axis=time_axis, time_shift=known_shift)

    unoptimized = mode_match(
        reference,
        shifted,
        time_alignment="none",
        time_domain_policy="resample_to_reference",
        phase_alignment="global_complex",
    )
    optimized = mode_match(
        reference,
        shifted,
        time_alignment="none",
        time_domain_policy="resample_to_reference",
        phase_alignment="global_complex",
        optimize_time_shift=True,
        time_shift_bounds=(-0.7, 0.0),
    )

    assert optimized.match > unoptimized.match
    assert optimized.match > 0.999
    assert optimized.best_parameters["candidate_time_shift"] == pytest.approx(
        -known_shift, abs=2e-2
    )
    assert optimized.diagnostics["alignment"]["optimize_time_shift"] is True
    assert optimized.diagnostics["time_axis"][
        "candidate_time_shift"
    ] == pytest.approx(-known_shift, abs=2e-2)
    assert (
        optimized.diagnostics["time_shift_optimization"]["parameter"]
        == "candidate_time_shift"
    )


def test_time_shift_optimization_default_bounds_recover_large_shift():
    time_axis = np.linspace(-12.0, 12.0, 512)
    known_shift = 2.4
    reference = make_test_modes(time_axis=time_axis, time_shift=0.0)
    shifted = make_test_modes(time_axis=time_axis, time_shift=known_shift)

    unoptimized = mode_match(
        reference,
        shifted,
        time_alignment="none",
        time_domain_policy="resample_to_reference",
        phase_alignment="global_complex",
    )
    optimized = mode_match(
        reference,
        shifted,
        time_alignment="none",
        time_domain_policy="resample_to_reference",
        phase_alignment="global_complex",
        optimize_time_shift=True,
    )

    assert optimized.match > unoptimized.match
    assert optimized.match > 0.999
    assert optimized.best_parameters["candidate_time_shift"] == pytest.approx(
        -known_shift, abs=5e-2
    )
    assert (
        optimized.diagnostics["time_shift_optimization"]["n_grid_evaluations"]
        >= 401
    )
    assert (
        optimized.diagnostics["time_shift_optimization"][
            "coarse_grid_best_mismatch"
        ]
        >= optimized.diagnostics["time_shift_optimization"]["best_mismatch"]
    )


def test_replay_helper_inner_product_matches_mode_match():
    time_axis = np.linspace(-12.0, 12.0, 512)
    reference = make_test_modes(time_axis=time_axis, orbital_phase=0.0)
    shifted = make_test_modes(
        time_axis=time_axis,
        time_shift=0.45,
        orbital_phase=0.23,
    )
    unaligned = mode_match(
        reference,
        shifted,
        time_alignment="none",
        time_domain_policy="resample_to_reference",
        phase_alignment="none",
    )
    result = mode_match(
        reference,
        shifted,
        time_alignment="none",
        time_domain_policy="resample_to_reference",
        phase_alignment="orbital_phase_and_global",
        optimize_time_shift=True,
        time_shift_bounds=(-0.8, 0.0),
    )
    unaligned_replay = prepare_aligned_mode_data(reference, shifted, unaligned)
    aligned_replay = prepare_aligned_mode_data(reference, shifted, result)

    replay_match = normalized_inner_from_arrays(
        aligned_replay.reference_modes,
        aligned_replay.candidate_modes,
        aligned_replay.selected_modes,
        aligned_replay.time_axis,
    )
    assert replay_match == pytest.approx(result.match, abs=1e-12)
    assert normalized_rms_from_arrays(
        aligned_replay.reference_modes,
        aligned_replay.candidate_modes,
        aligned_replay.selected_modes,
    ) < normalized_rms_from_arrays(
        unaligned_replay.reference_modes,
        unaligned_replay.candidate_modes,
        unaligned_replay.selected_modes,
    )


def test_comparison_config_normalizes_nested_alignment():
    config = FittingFactorConfig.from_value(
        {
            "comparison": {
                "ell_min": 2,
                "ell_max": 2,
                "alignment": {
                    "time_alignment": "none",
                    "phase_alignment": "none",
                },
            },
            "variable_parameters": {"phase": (-1.0, 1.0)},
            "initial_parameters": {"phase": 0.25},
        }
    )

    assert isinstance(config.comparison, ModeComparisonConfig)
    assert isinstance(config.comparison.alignment, AlignmentSpec)
    assert config.comparison.alignment.time_alignment == "none"
    assert config.variable_parameters["phase"] == (-1.0, 1.0)


def test_fixed_candidate_fitting_factor_matches_mode_match():
    reference = make_test_modes()
    candidate = make_test_modes()
    comparison = ModeComparisonConfig(
        alignment=AlignmentSpec(
            time_alignment="none", phase_alignment="global_complex"
        )
    )

    match = mode_match(reference, candidate, alignment=comparison.alignment)
    fitting = fixed_candidate_fitting_factor(
        reference, candidate, config=comparison
    )

    assert fitting.objective_name == "fixed_candidate_fitting_factor"
    assert fitting.match == pytest.approx(match.match, abs=1e-12)
    assert fitting.candidate_generation_parameters == {}
    assert fitting.best_parameters["generator"] == {}
    assert (
        fitting.best_parameters["alignment"]["phase_alignment"]
        == "global_complex"
    )
    assert (
        fitting.diagnostics["comparison_config"]["alignment"]["time_alignment"]
        == "none"
    )


def test_fixed_candidate_fitting_factor_uses_comparison_rotation():
    angle = 0.4
    reference = make_test_modes(orbital_phase=0.0)
    shifted = make_test_modes(orbital_phase=angle)
    comparison = ModeComparisonConfig(
        alignment=AlignmentSpec(time_alignment="none", phase_alignment="none"),
        rotation=RotationSpec(kind="z_axis", angle=-angle),
    )

    result = fixed_candidate_fitting_factor(
        reference, shifted, config=comparison
    )

    assert result.match == pytest.approx(1.0, abs=1e-12)
    assert result.best_parameters["alignment"]["rotation"]["kind"] == "z_axis"
    assert result.best_parameters["alignment"]["rotation"][
        "angle"
    ] == pytest.approx(-angle)


def test_fixed_candidate_fitting_factor_uses_optimized_rotation():
    angle = 0.35
    reference = make_test_modes(orbital_phase=0.0)
    shifted = make_test_modes(orbital_phase=angle)
    comparison = ModeComparisonConfig(
        alignment=AlignmentSpec(time_alignment="none", phase_alignment="none"),
        rotation=RotationSpec(
            kind="z_axis",
            optimize_angle=True,
            angle_bounds=(-0.8, 0.0),
        ),
    )

    result = fixed_candidate_fitting_factor(
        reference, shifted, config=comparison
    )

    assert result.match == pytest.approx(1.0, abs=1e-10)
    assert result.best_parameters["alignment"]["rotation"][
        "angle"
    ] == pytest.approx(-angle, abs=2e-3)


def test_modes_array_fitting_factor_uses_generator_defaults():
    reference = make_test_modes(phase=0.2)
    calls = []

    def generator(phase=0.2, amplitude=1.0):
        calls.append({"phase": phase, "amplitude": amplitude})
        modes = make_test_modes(phase=phase)
        modes.set_mode_data(ell=2, emm=2, data=amplitude * modes.mode(2, 2))
        return modes

    result = reference.fitting_factor(
        generator,
        config=FittingFactorConfig(
            comparison=ModeComparisonConfig(
                alignment=AlignmentSpec(
                    time_alignment="none", phase_alignment="none"
                )
            ),
            fixed_parameters={},
            optimizer="none",
        ),
    )

    assert result.match == pytest.approx(1.0, abs=1e-12)
    assert calls == [{"phase": 0.2, "amplitude": 1.0}]
    assert result.n_waveform_generations == 1


def test_fitting_factor_optimizes_user_selected_parameter():
    target_phase = 0.35
    reference = make_test_modes(phase=target_phase)

    def generator(phase):
        return make_test_modes(phase=phase)

    result = reference.fitting_factor(
        generator,
        config=FittingFactorConfig(
            comparison=ModeComparisonConfig(
                alignment=AlignmentSpec(
                    time_alignment="none", phase_alignment="none"
                )
            ),
            variable_parameters={"phase": (-1.0, 1.0)},
            initial_parameters={"phase": 0.0},
            optimizer="scipy_minimize",
            optimizer_options={"options": {"maxiter": 40}},
        ),
    )

    assert result.match > 0.999
    assert result.candidate_generation_parameters["phase"] == pytest.approx(
        target_phase, abs=2e-3
    )
    assert result.best_parameters["generator"]["phase"] == pytest.approx(
        target_phase, abs=2e-3
    )
    assert result.best_parameters["alignment"]["candidate_time_shift"] == 0.0
    assert result.n_waveform_generations >= 1


def test_fitting_factor_forwards_arbitrary_user_parameter_dict():
    target_phase = 0.45
    reference = make_test_modes(phase=target_phase)
    calls = []

    def generator(parameters):
        calls.append(dict(parameters))
        assert parameters["approximant"] == "synthetic-family"
        assert parameters["total_mass"] == 75.0
        assert parameters["spin1_theta"] == 0.6
        phase = parameters["reference_phase"] + 0.05 * parameters["mass_ratio"]
        return make_test_modes(phase=phase)

    result = reference.fitting_factor(
        generator,
        config=FittingFactorConfig(
            comparison=ModeComparisonConfig(
                alignment=AlignmentSpec(
                    time_alignment="none", phase_alignment="none"
                )
            ),
            variable_parameters={"reference_phase": (-1.0, 1.0)},
            fixed_parameters={
                "approximant": "synthetic-family",
                "total_mass": 75.0,
                "mass_ratio": 2.0,
                "spin1_theta": 0.6,
            },
            initial_parameters={"reference_phase": 0.0},
            optimizer="scipy_minimize",
            optimizer_options={"options": {"maxiter": 40}},
            generator_call_style="dict",
        ),
    )

    assert result.match > 0.999
    best_reference_phase = result.best_parameters["generator"][
        "reference_phase"
    ]
    assert best_reference_phase == pytest.approx(target_phase - 0.1, abs=2e-3)
    assert calls
    assert "reference_phase" in calls[0]
    assert "mass_ratio" in calls[0]


def test_fitting_factor_namespaces_generator_and_alignment_parameters():
    reference = make_test_modes()

    def generator(candidate_time_shift=0.25):
        assert candidate_time_shift == 0.25
        return make_test_modes()

    result = reference.fitting_factor(
        generator,
        config=FittingFactorConfig(
            comparison=ModeComparisonConfig(
                alignment=AlignmentSpec(
                    time_alignment="none",
                    phase_alignment="global_complex",
                    candidate_time_shift=0.0,
                )
            ),
            fixed_parameters={"candidate_time_shift": 0.25},
            optimizer="none",
        ),
    )

    assert result.match == pytest.approx(1.0, abs=1e-12)
    assert (
        result.candidate_generation_parameters["candidate_time_shift"] == 0.25
    )
    assert result.best_parameters["generator"]["candidate_time_shift"] == 0.25
    assert result.best_parameters["alignment"]["candidate_time_shift"] == 0.0


def test_residue_distance_zero_for_identical_modes():
    modes = make_test_modes()

    def residual(obj):
        return np.abs(obj.mode(2, 2))

    result = residue_distance(modes, modes, residue_function=residual)
    assert result.distance == pytest.approx(0.0, abs=1e-14)
    assert result.normalized_distance == pytest.approx(0.0, abs=1e-14)
