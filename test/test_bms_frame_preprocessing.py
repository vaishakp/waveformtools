"""Tests for conservative BMS-frame preprocessing reports."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools.bms_frame_diagnostics import BMSFrameDiagnosticsResult
from waveformtools.bms_frame_preprocessing import (
    BMSFramePreprocessingSpec,
    bms_synchronized_fixed_candidate_fitting_factor,
    preprocess_bms_frame,
)
from waveformtools.comparison import AlignmentSpec, ModeComparisonConfig
from waveformtools.modes_array import ModesArray


def make_preprocessing_modes(amplitude: float = 1.0) -> ModesArray:
    time_axis = np.linspace(-16.0, 16.0, 128)
    envelope = amplitude * np.exp(-0.012 * time_axis**2)
    phase = 0.3 * time_axis + 0.002 * time_axis**2
    modes = ModesArray(ell_max=3, time_axis=time_axis, spin_weight=-2)
    modes.create_modes_array(ell_max=3, data_len=len(time_axis))
    modes.set_mode_data(ell=2, emm=2, data=envelope * np.exp(1j * phase))
    modes.set_mode_data(
        ell=2,
        emm=-2,
        data=0.4 * envelope * np.exp(-1j * phase),
    )
    modes.set_mode_data(
        ell=3,
        emm=1,
        data=0.1 * envelope * np.exp(0.15j * time_axis),
    )
    return modes


def make_diagnostics(
    vector,
    *,
    angular_vector=None,
) -> BMSFrameDiagnosticsResult:
    linear = np.asarray(vector, dtype=float)
    angular = (
        linear
        if angular_vector is None
        else np.asarray(angular_vector, dtype=float)
    )
    return BMSFrameDiagnosticsResult(
        energy_radiated=0.1,
        radiated_linear_momentum=linear,
        kick_velocity=linear / 0.95,
        angular_momentum_radiated=angular,
        memory_finite_time=None,
        omitted_inspiral={"omitted_inspiral_likely": False},
        assumptions={},
        diagnostics={},
    )


def test_bms_frame_preprocessing_returns_copies_and_report():
    target = make_preprocessing_modes()
    candidate = make_preprocessing_modes()

    target_pre, candidate_pre, report = preprocess_bms_frame(
        target,
        candidate,
        spec=BMSFramePreprocessingSpec(),
    )

    assert target_pre is not target
    assert candidate_pre is not candidate
    np.testing.assert_allclose(target_pre.modes_data, target.modes_data)
    np.testing.assert_allclose(candidate_pre.modes_data, candidate.modes_data)
    assert report.compatibility["compatible"]
    assert not report.transforms["applied"]
    assert not report.assumptions["superrest_frame_fixed"]
    assert report.target_diagnostics is not None
    assert report.candidate_diagnostics is not None


def test_bms_frame_preprocessing_can_skip_diagnostics():
    target = make_preprocessing_modes()
    candidate = make_preprocessing_modes(amplitude=1.5)

    _target_pre, _candidate_pre, report = preprocess_bms_frame(
        target,
        candidate,
        diagnose=False,
    )

    assert report.target_diagnostics is None
    assert report.candidate_diagnostics is None
    assert report.compatibility["compatible"]
    assert not report.compatibility["diagnostics_available"]


def test_bms_frame_preprocessing_flags_incompatible_energy():
    target = make_preprocessing_modes(amplitude=1.0)
    candidate = make_preprocessing_modes(amplitude=1.6)

    _target_pre, _candidate_pre, report = preprocess_bms_frame(
        target,
        candidate,
        relative_energy_tolerance=0.01,
        relative_angular_momentum_tolerance=1.0,
        relative_linear_momentum_tolerance=1.0,
    )

    assert not report.compatibility["compatible"]
    assert "energy_radiated" in report.compatibility["failed_checks"]


def test_bms_frame_preprocessing_can_require_compatibility():
    target = make_preprocessing_modes(amplitude=1.0)
    candidate = make_preprocessing_modes(amplitude=1.6)

    with pytest.raises(ValueError, match="outside requested tolerances"):
        preprocess_bms_frame(
            target,
            candidate,
            require_compatible_diagnostics=True,
            relative_energy_tolerance=0.01,
            relative_angular_momentum_tolerance=1.0,
            relative_linear_momentum_tolerance=1.0,
        )


def test_bms_frame_preprocessing_rejects_unimplemented_transforms():
    with pytest.raises(NotImplementedError, match="rotation"):
        BMSFramePreprocessingSpec(
            rotation="bad_rotation"  # type: ignore[arg-type]
        )

    with pytest.raises(NotImplementedError, match="boost"):
        BMSFramePreprocessingSpec(boost="final_rest")  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="supertranslation"):
        BMSFramePreprocessingSpec(
            supertranslation="superrest"  # type: ignore[arg-type]
        )


def test_bms_frame_preprocessing_report_is_json_friendly():
    target = make_preprocessing_modes()
    candidate = make_preprocessing_modes()

    _target_pre, _candidate_pre, report = preprocess_bms_frame(
        target,
        candidate,
    )
    data = report.to_dict()

    assert data["compatibility"]["compatible"]
    assert data["transforms"]["rotation"] == "none"
    assert "pn_eob_phenom_frame_comment" in data["assumptions"]


def test_bms_frame_preprocessing_aligns_radiated_momentum(monkeypatch):
    from waveformtools import bms_frame_preprocessing as preprocessing

    target = make_preprocessing_modes()
    candidate = make_preprocessing_modes()

    def fake_diagnostics(modes, _config):
        if modes is target:
            return make_diagnostics([0.0, 0.0, 1.0])
        return make_diagnostics([1.0, 0.0, 0.0])

    monkeypatch.setattr(
        preprocessing,
        "compute_bms_frame_diagnostics",
        fake_diagnostics,
    )

    _target_pre, candidate_pre, report = preprocess_bms_frame(
        target,
        candidate,
        rotation="align_kick",
        relative_linear_momentum_tolerance=1e-12,
        relative_angular_momentum_tolerance=1e-12,
    )

    assert report.transforms["applied"]
    assert report.transforms["rotation"] == "align_radiated_linear_momentum"
    assert report.transforms["applied_operations"] == ["rotation"]
    direction_residual = report.transforms["rotation_details"][
        "direction_residual"
    ]
    assert direction_residual == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(
        report.candidate_diagnostics.radiated_linear_momentum,
        [0.0, 0.0, 1.0],
        atol=1e-14,
    )
    assert report.compatibility["compatible"]
    assert not np.allclose(candidate_pre.modes_data, candidate.modes_data)


def test_bms_frame_preprocessing_aligns_radiated_angular_momentum(
    monkeypatch,
):
    from waveformtools import bms_frame_preprocessing as preprocessing

    target = make_preprocessing_modes()
    candidate = make_preprocessing_modes()

    def fake_diagnostics(modes, _config):
        if modes is target:
            return make_diagnostics(
                [0.2, 0.0, 0.0],
                angular_vector=[0.0, 0.0, 1.0],
            )
        return make_diagnostics(
            [0.2, 0.0, 0.0],
            angular_vector=[1.0, 0.0, 0.0],
        )

    monkeypatch.setattr(
        preprocessing,
        "compute_bms_frame_diagnostics",
        fake_diagnostics,
    )

    _target_pre, candidate_pre, report = preprocess_bms_frame(
        target,
        candidate,
        rotation="align_angular_momentum",
        relative_linear_momentum_tolerance=2.0,
        relative_angular_momentum_tolerance=1e-12,
    )

    assert report.transforms["applied"]
    assert report.transforms["rotation"] == "align_radiated_angular_momentum"
    assert (
        report.transforms["rotation_details"]["source_quantity"]
        == "angular_momentum_radiated"
    )
    np.testing.assert_allclose(
        report.candidate_diagnostics.angular_momentum_radiated,
        [0.0, 0.0, 1.0],
        atol=1e-14,
    )
    assert report.compatibility["compatible"]
    assert not np.allclose(candidate_pre.modes_data, candidate.modes_data)


def test_bms_frame_preprocessing_aligns_angular_then_linear_momentum(
    monkeypatch,
):
    from waveformtools import bms_frame_preprocessing as preprocessing

    target = make_preprocessing_modes()
    candidate = make_preprocessing_modes()

    def fake_diagnostics(modes, _config):
        if modes is target:
            return make_diagnostics(
                [1.0, 0.0, 0.0],
                angular_vector=[0.0, 0.0, 1.0],
            )
        return make_diagnostics(
            [0.0, 1.0, 0.0],
            angular_vector=[1.0, 0.0, 0.0],
        )

    monkeypatch.setattr(
        preprocessing,
        "compute_bms_frame_diagnostics",
        fake_diagnostics,
    )

    _target_pre, _candidate_pre, report = preprocess_bms_frame(
        target,
        candidate,
        rotation="align_angular_then_linear_momentum",
        relative_linear_momentum_tolerance=1e-12,
        relative_angular_momentum_tolerance=1e-12,
    )

    details = report.transforms["rotation_details"]
    assert details["method"] == "align_angular_then_linear_momentum"
    assert details["secondary_alignment"]["applied"]
    assert details["direction_residual"] == pytest.approx(0.0, abs=1e-14)
    assert details["secondary_alignment"][
        "projected_direction_residual"
    ] == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(
        report.candidate_diagnostics.angular_momentum_radiated,
        [0.0, 0.0, 1.0],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        report.candidate_diagnostics.radiated_linear_momentum,
        [1.0, 0.0, 0.0],
        atol=1e-14,
    )
    assert report.compatibility["compatible"]


def test_bms_frame_preprocessing_rejects_zero_momentum_alignment(monkeypatch):
    from waveformtools import bms_frame_preprocessing as preprocessing

    target = make_preprocessing_modes()
    candidate = make_preprocessing_modes()

    def fake_diagnostics(modes, _config):
        if modes is target:
            return make_diagnostics([0.0, 0.0, 0.0])
        return make_diagnostics([1.0, 0.0, 0.0])

    monkeypatch.setattr(
        preprocessing,
        "compute_bms_frame_diagnostics",
        fake_diagnostics,
    )

    with pytest.raises(ValueError, match="nonzero momentum"):
        preprocess_bms_frame(
            target,
            candidate,
            rotation="align_radiated_linear_momentum",
        )


def test_bms_synchronized_fitting_factor_defaults_to_angular_momentum(
    monkeypatch,
):
    from waveformtools import bms_frame_preprocessing as preprocessing

    target = make_preprocessing_modes()
    candidate = make_preprocessing_modes()

    def fake_diagnostics(modes, _config):
        if modes is target:
            return make_diagnostics(
                [0.2, 0.0, 0.0],
                angular_vector=[0.0, 0.0, 1.0],
            )
        return make_diagnostics(
            [0.2, 0.0, 0.0],
            angular_vector=[1.0, 0.0, 0.0],
        )

    monkeypatch.setattr(
        preprocessing,
        "compute_bms_frame_diagnostics",
        fake_diagnostics,
    )
    comparison = ModeComparisonConfig(
        alignment=AlignmentSpec(
            time_alignment="none",
            phase_alignment="global_complex",
        ),
        ell_max=3,
    )

    result = bms_synchronized_fixed_candidate_fitting_factor(
        target,
        candidate,
        comparison=comparison,
        preprocessing={
            "relative_linear_momentum_tolerance": 1.0,
            "relative_angular_momentum_tolerance": 1e-12,
        },
    )

    assert result.objective_name == "fixed_candidate_fitting_factor"
    sync = result.diagnostics["bms_frame_synchronization"]
    assert sync["rotation"] == "align_radiated_angular_momentum"
    assert sync["phase_alignment_after_preprocessing"] == "global_complex"
    report = result.diagnostics["bms_frame_preprocessing"]
    assert report["transforms"]["applied"]
    assert (
        report["transforms"]["rotation_details"]["source_quantity"]
        == "angular_momentum_radiated"
    )
