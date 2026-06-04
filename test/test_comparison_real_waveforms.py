"""Optional real-waveform integration tests for comparison features.

These tests are skipped by default because LAL mode generation can be slow and
may require local surrogate data. Run them explicitly with

    NUMBA_CACHE_DIR=/tmp/waveformtools_numba_cache \
    WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 \
    pytest test/test_comparison_real_waveforms.py

PyCBC plugin discovery may import pyseobnr/qnm. In some conda environments that
path needs a writable ``NUMBA_CACHE_DIR``. Older PyCBC releases also still import
``pkg_resources``, which is provided by ``setuptools<81``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from waveformtools.comparison import (
    AlignmentSpec,
    FittingFactorConfig,
    ModeComparisonConfig,
    RotationSpec,
    fixed_candidate_fitting_factor,
    mode_match,
    rotate_modes,
)
from waveformtools.comparison.alignment import prepare_mode_data

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
    return _generate_real_modes(str(request.param))


@pytest.fixture(scope="session")
def real_model_pair():
    return {
        approximant: _generate_real_modes(approximant)
        for approximant in ("NRSur7dq4", "IMRPhenomXPHM")
    }


def _generate_real_modes(approximant: str):
    if not _run_real_waveform_tests():
        pytest.skip(
            "Set WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 to run real-waveform tests."
        )

    modes = _generate_real_modes_from_parameters(_base_parameters(approximant))
    _require_populated_modes(modes, approximant)
    return modes


def _generate_real_modes_from_parameters(parameters):
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")

    from waveformtools.models.lal import LALWaveformModel

    model = LALWaveformModel(parameters_dict=dict(parameters))
    try:
        modes = model.get_td_waveform_modes(dimensionless=True)
    except Exception as exc:
        approximant = parameters.get("approximant", "unknown")
        pytest.skip(f"{approximant} mode generation unavailable: {exc}")
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


def _normalized_rms_residue(
    modes_a,
    modes_b,
    selected_modes,
    alignment,
    rotation,
    *,
    orbital_phase: float | None = None,
) -> float:
    alignment_spec = AlignmentSpec.from_value(alignment)
    rotation_spec = RotationSpec.from_value(rotation)
    rotated_b = rotate_modes(modes_b, rotation_spec, modes=selected_modes)
    prepared = prepare_mode_data(modes_a, rotated_b, selected_modes, alignment_spec)
    phase_adjusted_b = {}
    orbital_phase = 0.0 if orbital_phase is None else float(orbital_phase)
    for ell, emm in prepared.selected_modes:
        phase_adjusted_b[(ell, emm)] = (
            np.exp(1j * emm * orbital_phase) * prepared.modes_b[(ell, emm)]
        )

    if alignment_spec.phase_alignment in {"global_complex", "orbital_phase_and_global"}:
        overlap = 0.0j
        for mode in prepared.selected_modes:
            overlap += np.vdot(phase_adjusted_b[mode], prepared.modes_a[mode])
        global_phase = np.angle(overlap)
        for mode in prepared.selected_modes:
            phase_adjusted_b[mode] = np.exp(1j * global_phase) * phase_adjusted_b[mode]

    numerator = 0.0
    denominator = 0.0
    for mode in prepared.selected_modes:
        diff = prepared.modes_a[mode] - phase_adjusted_b[mode]
        numerator += float(np.mean(np.abs(diff) ** 2))
        denominator += float(np.mean(np.abs(prepared.modes_a[mode]) ** 2))
    return float(np.sqrt(numerator / max(denominator, 1e-300)))


def test_real_waveform_self_match(real_modes):
    selected_modes = _available_ell2_modes(real_modes)
    result = mode_match(
        real_modes,
        real_modes,
        modes=selected_modes,
        time_alignment="none",
        time_domain_policy="error",
        phase_alignment="none",
    )

    assert result.match == pytest.approx(1.0, abs=1e-12)
    assert result.mismatch == pytest.approx(0.0, abs=1e-12)
    assert _normalized_rms_residue(
        real_modes,
        real_modes,
        selected_modes,
        AlignmentSpec(
            time_alignment="none",
            time_domain_policy="error",
            phase_alignment="none",
        ),
        RotationSpec(),
    ) == pytest.approx(0.0, abs=1e-12)


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
    assert _normalized_rms_residue(
        real_modes,
        candidate,
        selected_modes,
        AlignmentSpec(
            time_alignment="none",
            time_domain_policy="error",
            phase_alignment="none",
        ),
        RotationSpec.from_value(result.best_parameters["rotation"]),
    ) < 5e-3
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
    assert _normalized_rms_residue(
        real_modes,
        candidate,
        selected_modes,
        AlignmentSpec(
            time_alignment="none",
            time_domain_policy="error",
            phase_alignment="none",
        ),
        RotationSpec.from_value(result.best_parameters["rotation"]),
    ) < 5e-3
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
    assert _normalized_rms_residue(
        real_modes,
        candidate,
        selected_modes,
        AlignmentSpec(
            time_alignment="none",
            time_domain_policy="error",
            phase_alignment="none",
        ),
        RotationSpec.from_value(result.best_parameters["alignment"]["rotation"]),
    ) < 5e-3
    assert result.best_parameters["alignment"]["rotation"]["angle"] == pytest.approx(
        -0.15, abs=2e-3
    )


def test_real_waveform_cross_model_fitting_factor_smoke(real_model_pair):
    nrsur = real_model_pair["NRSur7dq4"]
    phenom = real_model_pair["IMRPhenomXPHM"]
    selected_modes = sorted(
        set(_available_ell2_modes(nrsur)).intersection(_available_ell2_modes(phenom))
    )
    if len(selected_modes) < 2:
        pytest.skip("Need at least two common ell=2 modes for cross-model comparison.")

    fixed = fixed_candidate_fitting_factor(
        nrsur,
        phenom,
        config=ModeComparisonConfig(
            modes=selected_modes,
            alignment=AlignmentSpec(
                time_alignment="peak_total_news_power",
                time_domain_policy="resample_to_reference",
                phase_alignment="orbital_phase_and_global",
                optimize_time_shift=False,
                orbital_phase_samples=257,
            ),
        ),
    )
    optimized = fixed_candidate_fitting_factor(
        nrsur,
        phenom,
        config=ModeComparisonConfig(
            modes=selected_modes,
            alignment=AlignmentSpec(
                time_alignment="peak_total_news_power",
                time_domain_policy="resample_to_reference",
                phase_alignment="orbital_phase_and_global",
                optimize_time_shift=True,
                orbital_phase_samples=257,
            ),
            rotation=RotationSpec(
                kind="z_axis",
                optimize_angle=True,
                angle_bounds=(-np.pi, np.pi),
            ),
        ),
    )

    assert np.isfinite(fixed.match)
    assert np.isfinite(optimized.match)
    assert np.isfinite(fixed.mismatch)
    assert np.isfinite(optimized.mismatch)
    fixed_rms_residue = _normalized_rms_residue(
        nrsur,
        phenom,
        selected_modes,
        AlignmentSpec.from_value(
            fixed.diagnostics["comparison_config"]["alignment"],
            candidate_time_shift=fixed.best_parameters["alignment"][
                "candidate_time_shift"
            ],
        ),
        fixed.best_parameters["alignment"]["rotation"],
        orbital_phase=fixed.best_parameters["alignment"]["orbital_phase"],
    )
    optimized_rms_residue = _normalized_rms_residue(
        nrsur,
        phenom,
        selected_modes,
        AlignmentSpec.from_value(
            optimized.diagnostics["comparison_config"]["alignment"],
            candidate_time_shift=optimized.best_parameters["alignment"][
                "candidate_time_shift"
            ],
        ),
        optimized.best_parameters["alignment"]["rotation"],
        orbital_phase=optimized.best_parameters["alignment"]["orbital_phase"],
    )
    assert np.isfinite(fixed_rms_residue)
    assert np.isfinite(optimized_rms_residue)
    assert -1.0 <= fixed.match <= 1.0
    assert -1.0 <= optimized.match <= 1.0
    assert optimized.match >= fixed.match - 1e-8
    assert optimized_rms_residue <= fixed_rms_residue + 1e-8
    assert optimized.best_parameters["alignment"]["candidate_time_shift"] != 0.0
    assert optimized.best_parameters["alignment"]["rotation"]["kind"] == "z_axis"


def test_real_waveform_generator_fitting_factor_accepts_user_params(real_model_pair):
    nrsur = real_model_pair["NRSur7dq4"]
    selected_modes = _available_ell2_modes(nrsur)
    base_candidate_parameters = _base_parameters("IMRPhenomXPHM")
    calls = []

    def candidate_generator(**user_parameters):
        parameters = dict(base_candidate_parameters)
        parameters.update(user_parameters)
        calls.append(dict(user_parameters))
        return _generate_real_modes_from_parameters(parameters)

    result = nrsur.fitting_factor(
        candidate_generator,
        config=FittingFactorConfig(
            comparison=ModeComparisonConfig(
                modes=selected_modes,
                alignment=AlignmentSpec(
                    time_alignment="peak_total_news_power",
                    time_domain_policy="resample_to_reference",
                    phase_alignment="orbital_phase_and_global",
                    optimize_time_shift=False,
                    orbital_phase_samples=65,
                ),
            ),
            variable_parameters={"phi_ref": (-0.1, 0.5)},
            fixed_parameters={
                "approximant": "IMRPhenomXPHM",
                "ell_max": 2,
            },
            initial_parameters={
                "phi_ref": base_candidate_parameters["phi_ref"]
            },
            optimizer="scipy_minimize",
            optimizer_options={"options": {"maxiter": 2}},
        ),
    )

    assert np.isfinite(result.match)
    assert np.isfinite(result.mismatch)
    assert result.n_waveform_generations >= 1
    assert "phi_ref" in result.best_parameters["generator"]
    assert "phi_ref" in calls[0]
