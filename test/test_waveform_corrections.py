"""Tests for fractional modal waveform corrections."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools.modes_array import ModesArray
from waveformtools.waveform_corrections import (
    FractionalModeCorrectionSpec,
    apply_fractional_mode_correction,
    coefficient_bounds,
    coefficient_index,
    coefficient_layout,
    coefficient_scales,
    fractional_correction_basis,
    fractional_delta_from_vector,
    n_fractional_correction_coefficients,
    zero_fractional_correction_vector,
)


def make_correction_test_modes() -> ModesArray:
    time_axis = np.linspace(-8.0, 8.0, 96)
    envelope = np.exp(-0.04 * time_axis**2)
    modes = ModesArray(ell_max=3, time_axis=time_axis, spin_weight=-2)
    modes.create_modes_array(ell_max=3, data_len=len(time_axis))
    modes.set_mode_data(
        ell=2,
        emm=2,
        data=envelope * np.exp(0.5j * time_axis),
    )
    modes.set_mode_data(
        ell=2,
        emm=-2,
        data=0.4 * envelope * np.exp(-0.5j * time_axis),
    )
    modes.set_mode_data(
        ell=3,
        emm=1,
        data=0.2 * envelope * np.exp(0.2j * time_axis),
    )
    return modes


def test_fractional_correction_spec_validates_layout():
    spec = FractionalModeCorrectionSpec(
        modes=[(2, 2), (2, -2)],
        n_time_basis=4,
        max_abs_delta=0.1,
    )

    layout = coefficient_layout(spec)

    assert n_fractional_correction_coefficients(spec) == 16
    assert layout[0].mode == (2, 2)
    assert layout[0].basis_index == 0
    assert layout[0].component == "real"
    assert layout[1].component == "imag"

    with pytest.raises(ValueError, match="Duplicate"):
        FractionalModeCorrectionSpec(modes=[(2, 2), (2, 2)])

    with pytest.raises(ValueError, match="Invalid mode"):
        FractionalModeCorrectionSpec(modes=[(2, 3)])


def test_zero_fractional_correction_is_identity():
    modes = make_correction_test_modes()
    spec = FractionalModeCorrectionSpec(modes=[(2, 2)], n_time_basis=5)
    vector = zero_fractional_correction_vector(spec)

    result = apply_fractional_mode_correction(modes, vector, spec)

    np.testing.assert_allclose(
        result.corrected_modes.mode(2, 2), modes.mode(2, 2)
    )
    np.testing.assert_allclose(
        result.corrected_modes.mode(2, -2), modes.mode(2, -2)
    )
    assert result.diagnostics["max_abs_delta"] == pytest.approx(0.0)


def test_real_fractional_correction_rescales_selected_mode():
    modes = make_correction_test_modes()
    spec = FractionalModeCorrectionSpec(
        modes=[(2, 2)],
        n_time_basis=5,
        max_abs_delta=0.2,
    )
    vector = zero_fractional_correction_vector(spec)
    vector[coefficient_index(spec, (2, 2), 2, "real")] = 0.05

    result = apply_fractional_mode_correction(modes, vector, spec)
    delta = result.delta_by_mode[(2, 2)]

    np.testing.assert_allclose(
        result.corrected_modes.mode(2, 2),
        modes.mode(2, 2) * (1.0 + delta),
    )
    np.testing.assert_allclose(
        result.corrected_modes.mode(2, -2), modes.mode(2, -2)
    )
    assert np.max(np.real(delta)) > 0.0
    assert np.max(np.abs(np.imag(delta))) == pytest.approx(0.0)


def test_imaginary_fractional_correction_is_phase_like():
    modes = make_correction_test_modes()
    spec = FractionalModeCorrectionSpec(
        modes=[(2, 2)],
        n_time_basis=5,
        max_abs_delta=0.2,
    )
    vector = zero_fractional_correction_vector(spec)
    vector[coefficient_index(spec, (2, 2), 2, "imag")] = 0.04

    result = apply_fractional_mode_correction(modes, vector, spec)
    ratio = result.corrected_modes.mode(2, 2) / modes.mode(2, 2)
    delta = result.delta_by_mode[(2, 2)]

    np.testing.assert_allclose(ratio, 1.0 + delta)
    assert np.max(np.imag(delta)) > 0.0
    assert np.max(np.abs(np.real(delta))) == pytest.approx(0.0)


def test_fractional_correction_supports_multiple_modes():
    modes = make_correction_test_modes()
    spec = FractionalModeCorrectionSpec(
        modes=[(2, 2), (2, -2)],
        n_time_basis=4,
        max_abs_delta={(2, 2): 0.1, (2, -2): 0.05},
    )
    vector = zero_fractional_correction_vector(spec)
    vector[coefficient_index(spec, (2, 2), 1, "real")] = 0.02
    vector[coefficient_index(spec, (2, -2), 2, "imag")] = -0.01

    result = apply_fractional_mode_correction(modes, vector, spec)

    assert result.diagnostics["max_abs_delta_by_mode"][(2, 2)] > 0.0
    assert result.diagnostics["max_abs_delta_by_mode"][(2, -2)] > 0.0
    np.testing.assert_allclose(
        result.corrected_modes.mode(3, 1), modes.mode(3, 1)
    )


def test_fractional_correction_bounds_and_scales_are_solver_ready():
    spec = FractionalModeCorrectionSpec(
        modes=[(2, 2)],
        n_time_basis=4,
        max_abs_delta=0.2,
    )

    lower, upper = coefficient_bounds(spec)
    scales = coefficient_scales(spec)

    assert lower.shape == (8,)
    assert upper.shape == (8,)
    assert scales.shape == (8,)
    np.testing.assert_allclose(lower, -upper)
    np.testing.assert_allclose(scales, upper)
    assert np.all(upper < 0.2)


def test_fractional_correction_enforces_actual_delta_bound():
    modes = make_correction_test_modes()
    spec = FractionalModeCorrectionSpec(
        modes=[(2, 2)],
        n_time_basis=4,
        max_abs_delta=0.01,
    )
    vector = zero_fractional_correction_vector(spec)
    vector[coefficient_index(spec, (2, 2), 1, "real")] = 0.5

    with pytest.raises(ValueError, match="exceeds max_abs_delta"):
        apply_fractional_mode_correction(modes, vector, spec)


def test_fractional_correction_basis_and_delta_diagnostics_are_finite():
    modes = make_correction_test_modes()
    spec = FractionalModeCorrectionSpec(
        modes=[(2, 2)],
        n_time_basis=5,
        max_abs_delta=0.2,
    )
    vector = zero_fractional_correction_vector(spec)
    vector[coefficient_index(spec, (2, 2), 1, "real")] = 0.02
    vector[coefficient_index(spec, (2, 2), 3, "imag")] = -0.03

    basis = fractional_correction_basis(modes.time_axis, spec)
    delta_by_mode = fractional_delta_from_vector(vector, spec, modes.time_axis)
    result = apply_fractional_mode_correction(modes, vector, spec)

    assert basis.shape == (modes.data_len, spec.n_time_basis)
    assert np.allclose(basis[0], 0.0)
    assert np.allclose(basis[-1], 0.0)
    assert np.all(np.isfinite(delta_by_mode[(2, 2)]))
    assert np.isfinite(result.diagnostics["basis_condition_number"])
    assert result.diagnostics["basis_rank"] == spec.n_time_basis
