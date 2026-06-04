"""Optional real-waveform integration tests for comparison features.

These tests are skipped by default because LAL mode generation can be slow and
may require local surrogate data. Run them explicitly with

    WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 pytest test/test_comparison_real_waveforms.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from waveformtools.comparison import (
    AlignmentSpec,
    ModeComparisonConfig,
    RotationSpec,
    fixed_candidate_fitting_factor,
    mode_match,
    rotate_modes,
)

pytestmark = [pytest.mark.integration, pytest.mark.real_waveform]


def _run_real_waveform_tests() -> bool:
    return os.environ.get("WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS") == "1"


def _base_parameters(approximant: str) -> dict[str, float | str | int]:
    return {
        "approximant": approximant,
        "mass1": 40.0,
        "mass2": 35.0,
        "spin1x": 0.08,
        "spin1y": -0.03,
        "spin1z": 0.25,
        "spin2x": -0.04,
        "spin2y": 0.02,
        "spin2z": -0.15,
        "distance": 400.0,
        "inclination": 0.7,
        "phi_ref": 0.2,
        "f_lower": 20.0,
        "f_ref": 20.0,
        "f_max": 512.0,
        "delta_t": 1.0 / 2048.0,
        "delta_f": 1.0 / 4.0,
        "ell_max": 2,
    }


@pytest.fixture(scope="session", params=["NRSur7dq4", "IMRPhenomXPHM"])
def real_modes(request: pytest.FixtureRequest):
    if not _run_real_waveform_tests():
        pytest.skip(
            "Set WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 to run real-waveform tests."
        )

    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")

    from waveformtools.models.lal import LALWaveformModel

    approximant = str(request.param)
    model = LALWaveformModel(parameters_dict=_base_parameters(approximant))
    try:
        modes = model.get_td_waveform_modes(dimensionless=True)
    except Exception as exc:
        pytest.skip(f"{approximant} mode generation unavailable: {exc}")

    _require_populated_modes(modes, approximant)
    return modes


def _require_populated_modes(modes, label: str) -> None:
    time_axis = np.asarray(modes.time_axis, dtype=float)
    if len(time_axis) < 16:
        pytest.skip(f"{label} generated too few time samples: {len(time_axis)}")
    if not np.all(np.isfinite(time_axis)):
        pytest.skip(f"{label} generated non-finite time samples.")

    mode_power = 0.0
    for emm in range(-2, 3):
        try:
            data = np.asarray(modes.mode(2, emm), dtype=np.complex128)
        except Exception:
            continue
        mode_power += float(np.sum(np.abs(data) ** 2))
    if not np.isfinite(mode_power) or mode_power <= 0.0:
        pytest.skip(f"{label} generated no finite ell=2 mode power.")


def _available_ell2_modes(modes) -> list[tuple[int, int]]:
    available = []
    for emm in range(-2, 3):
        try:
            data = np.asarray(modes.mode(2, emm), dtype=np.complex128)
        except Exception:
            continue
        if data.size and np.any(np.isfinite(data)):
            available.append((2, emm))
    if len(available) < 2:
        pytest.skip("Need at least two populated ell=2 modes for this test.")
    return available


def test_real_waveform_self_match(real_modes):
    result = mode_match(
        real_modes,
        real_modes,
        modes=_available_ell2_modes(real_modes),
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="none",
    )

    assert result.match == pytest.approx(1.0, abs=1e-12)
    assert result.mismatch == pytest.approx(0.0, abs=1e-12)


def test_real_waveform_z_rotation_recovery(real_modes):
    selected_modes = _available_ell2_modes(real_modes)
    angle = 0.2
    candidate = rotate_modes(
        real_modes,
        RotationSpec(kind="z_axis", angle=angle),
        modes=selected_modes,
    )

    result = mode_match(
        real_modes,
        candidate,
        modes=selected_modes,
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="none",
        rotation=RotationSpec(
            kind="z_axis",
            optimize_angle=True,
            angle_bounds=(-0.4, 0.0),
        ),
    )

    assert result.match > 0.999
    assert result.best_parameters["rotation"]["angle"] == pytest.approx(
        -angle, abs=2e-3
    )


def test_real_waveform_wigner_beta_rotation_recovery(real_modes):
    selected_modes = _available_ell2_modes(real_modes)
    if len(selected_modes) < 5:
        pytest.skip("Need all ell=2 modes for Wigner-mixing recovery.")

    beta = 0.12
    candidate = rotate_modes(
        real_modes,
        RotationSpec(kind="wigner", beta=beta),
        modes=selected_modes,
    )

    result = mode_match(
        real_modes,
        candidate,
        modes=selected_modes,
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="none",
        rotation=RotationSpec(
            kind="wigner",
            optimize_parameters=("beta",),
            parameter_bounds={"beta": (-0.25, 0.0)},
        ),
    )

    assert result.match > 0.999
    assert result.best_parameters["rotation"]["beta"] == pytest.approx(
        -beta, abs=4e-3
    )


def test_real_waveform_fixed_candidate_fitting_factor(real_modes):
    selected_modes = _available_ell2_modes(real_modes)
    candidate = rotate_modes(
        real_modes,
        RotationSpec(kind="z_axis", angle=0.15),
        modes=selected_modes,
    )

    result = fixed_candidate_fitting_factor(
        real_modes,
        candidate,
        config=ModeComparisonConfig(
            modes=selected_modes,
            alignment=AlignmentSpec(
                time_alignment="none",
                time_domain_policy="error",
                phase_alignment="none",
            ),
            rotation=RotationSpec(
                kind="z_axis",
                optimize_angle=True,
                angle_bounds=(-0.3, 0.0),
            ),
        ),
    )

    assert result.match > 0.999
    assert result.best_parameters["alignment"]["rotation"]["angle"] == pytest.approx(
        -0.15, abs=2e-3
    )
