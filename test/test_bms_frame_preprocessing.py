"""Tests for conservative BMS-frame preprocessing reports."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools.bms_frame_preprocessing import (
    BMSFramePreprocessingSpec,
    preprocess_bms_frame,
)
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
    assert report.charge_diagnostics["available"]
    assert report.charge_diagnostics["comparison"]["supermomentum_l0_energy"][
        "available"
    ]
    assert report.charge_diagnostics["comparison"][
        "supermomentum_l1_linear_momentum"
    ]["available"]
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
    assert not report.charge_diagnostics["available"]


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
    assert data["charge_diagnostics"]["available"]
    assert data["charge_diagnostics"]["comparison"]["angular_momentum"][
        "available"
    ]
    assert data["transforms"]["boost"] == "none"
    assert "pn_eob_phenom_frame_comment" in data["assumptions"]
