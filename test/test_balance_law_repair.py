"""Tests for projected balance-law repair coordinates."""

from __future__ import annotations

import numpy as np

from waveformtools.balance_law_repair import (
    BalanceLawRepairSpec,
    apply_balance_law_repair_coefficients,
    build_balance_law_repair_projection,
)
from waveformtools.modes_array import ModesArray
from waveformtools.waveform_corrections import (
    FractionalModeCorrectionSpec,
    coefficient_index,
    zero_fractional_correction_vector,
)


def make_repair_test_modes() -> ModesArray:
    time_axis = np.linspace(-12.0, 12.0, 128)
    envelope = np.exp(-0.025 * time_axis**2)
    chirp_phase = 0.25 * time_axis + 0.01 * time_axis**2
    modes = ModesArray(ell_max=3, time_axis=time_axis, spin_weight=-2)
    modes.create_modes_array(ell_max=3, data_len=len(time_axis))
    modes.set_mode_data(
        ell=2,
        emm=2,
        data=envelope * np.exp(1j * chirp_phase),
    )
    modes.set_mode_data(
        ell=2,
        emm=-2,
        data=0.35 * envelope * np.exp(-1j * chirp_phase),
    )
    modes.set_mode_data(
        ell=3,
        emm=1,
        data=0.2 * envelope * np.exp(0.15j * time_axis),
    )
    return modes


def weighted_inner_products(projection):
    weights = np.tile(projection.time_weights, len(projection.modes))
    weighted_protected = projection.protected_basis * weights[:, np.newaxis]
    weighted_projected = projection.projected_basis * weights[:, np.newaxis]
    return weighted_protected.conj().T @ weighted_projected


def test_balance_law_repair_projection_removes_global_phase_direction():
    modes = make_repair_test_modes()
    correction = FractionalModeCorrectionSpec(
        modes=[(2, 2)],
        n_time_basis=1,
        endpoint_constraint="free",
        include_real=False,
        include_imag=True,
    )
    spec = BalanceLawRepairSpec(
        correction=correction,
        protect_time_shift=False,
        protect_global_phase=True,
        protect_orbital_phase=False,
    )

    projection = build_balance_law_repair_projection(modes, spec)

    assert projection.diagnostics["protected_rank"] == 1
    assert np.linalg.norm(projection.projected_basis) < 1e-10


def test_balance_law_repair_projection_is_orthogonal_to_time_shift():
    modes = make_repair_test_modes()
    correction = FractionalModeCorrectionSpec(
        modes=[(2, 2), (2, -2)],
        n_time_basis=5,
        max_abs_delta=0.1,
    )
    spec = BalanceLawRepairSpec(
        correction=correction,
        protect_time_shift=True,
        protect_global_phase=False,
        protect_orbital_phase=False,
    )

    projection = build_balance_law_repair_projection(modes, spec)
    inner_products = weighted_inner_products(projection)

    assert projection.diagnostics["protected_rank"] == 1
    assert np.max(np.abs(inner_products)) < 1e-10


def test_balance_law_repair_projection_removes_user_tangent():
    modes = make_repair_test_modes()
    correction = FractionalModeCorrectionSpec(
        modes=[(2, 2)],
        n_time_basis=1,
        endpoint_constraint="free",
        include_real=True,
        include_imag=False,
    )
    spec = BalanceLawRepairSpec(
        correction=correction,
        protect_time_shift=False,
        protect_global_phase=False,
        protect_orbital_phase=False,
        user_tangent_labels=("mass_ratio",),
    )
    user_tangents = {"mass_ratio": {(2, 2): modes.mode(2, 2)}}

    projection = build_balance_law_repair_projection(
        modes,
        spec,
        user_tangents=user_tangents,
    )

    assert projection.protected_labels == ("user:mass_ratio",)
    assert projection.diagnostics["protected_rank"] == 1
    assert np.linalg.norm(projection.projected_basis) < 1e-10


def test_balance_law_repair_projection_reports_redundant_tangent_rank():
    modes = make_repair_test_modes()
    correction = FractionalModeCorrectionSpec(
        modes=[(2, 2)],
        n_time_basis=3,
        max_abs_delta=0.1,
    )
    spec = BalanceLawRepairSpec(
        correction=correction,
        protect_time_shift=False,
        protect_global_phase=True,
        protect_orbital_phase=True,
    )

    projection = build_balance_law_repair_projection(modes, spec)

    assert projection.diagnostics["n_protected_input_directions"] == 2
    assert projection.diagnostics["protected_rank"] == 1


def test_apply_balance_law_repair_coefficients_preserves_unselected_modes():
    modes = make_repair_test_modes()
    correction = FractionalModeCorrectionSpec(
        modes=[(2, 2)],
        n_time_basis=4,
        max_abs_delta=0.1,
    )
    spec = BalanceLawRepairSpec(
        correction=correction,
        protect_time_shift=False,
        protect_global_phase=False,
        protect_orbital_phase=False,
    )
    projection = build_balance_law_repair_projection(modes, spec)
    vector = zero_fractional_correction_vector(correction)
    vector[coefficient_index(correction, (2, 2), 1, "real")] = 0.03

    result = apply_balance_law_repair_coefficients(modes, vector, projection)

    assert not np.allclose(result.corrected_modes.mode(2, 2), modes.mode(2, 2))
    np.testing.assert_allclose(
        result.corrected_modes.mode(2, -2), modes.mode(2, -2)
    )
    np.testing.assert_allclose(
        result.corrected_modes.mode(3, 1), modes.mode(3, 1)
    )
    assert result.diagnostics["max_fractional_change"] > 0.0
