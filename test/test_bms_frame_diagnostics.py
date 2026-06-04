"""Tests for waveform-derived BMS-frame diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools.bms_frame_diagnostics import (
    BMSFrameDiagnosticsConfig,
    compute_bms_frame_diagnostics,
)
from waveformtools.modes_array import ModesArray


def make_bms_diagnostic_modes(spin_weight: int = -2) -> ModesArray:
    time_axis = np.linspace(-20.0, 20.0, 128)
    envelope = np.exp(-0.01 * time_axis**2)
    phase = 0.35 * time_axis + 0.003 * time_axis**2
    modes = ModesArray(ell_max=3, time_axis=time_axis, spin_weight=spin_weight)
    modes.create_modes_array(ell_max=3, data_len=len(time_axis))
    modes.set_mode_data(
        ell=2,
        emm=2,
        data=envelope * np.exp(1j * phase),
    )
    modes.set_mode_data(
        ell=2,
        emm=-2,
        data=0.4 * envelope * np.exp(-1j * phase),
    )
    modes.set_mode_data(
        ell=3,
        emm=1,
        data=0.15 * envelope * np.exp(0.2j * time_axis),
    )
    return modes


def test_bms_frame_diagnostics_config_validation():
    config = BMSFrameDiagnosticsConfig(
        initial_mass=1.0,
        final_mass=0.95,
        omitted_inspiral_mode=(2, 2),
    )

    assert config.news_method == "spline"
    assert config.final_mass == 0.95

    with pytest.raises(ValueError, match="final_mass"):
        BMSFrameDiagnosticsConfig(final_mass=0.0)

    with pytest.raises(ValueError, match="t_start"):
        BMSFrameDiagnosticsConfig(t_start=2.0, t_end=1.0)


def test_bms_frame_diagnostics_computes_flux_quantities():
    modes = make_bms_diagnostic_modes()

    result = compute_bms_frame_diagnostics(
        modes,
        initial_mass=1.0,
        final_mass=0.95,
    )

    assert result.energy_radiated is not None
    assert result.energy_radiated > 0.0
    assert result.radiated_linear_momentum is not None
    assert result.radiated_linear_momentum.shape == (3,)
    assert np.all(np.isfinite(result.radiated_linear_momentum))
    assert result.kick_velocity is not None
    np.testing.assert_allclose(
        result.kick_velocity,
        result.radiated_linear_momentum / 0.95,
    )
    assert result.angular_momentum_radiated is not None
    assert result.angular_momentum_radiated.shape == (3,)
    assert np.all(np.isfinite(result.angular_momentum_radiated))
    assert result.omitted_inspiral is not None
    assert result.omitted_inspiral["available"]


def test_bms_frame_diagnostics_records_charge_assumptions():
    modes = make_bms_diagnostic_modes()

    result = compute_bms_frame_diagnostics(modes, initial_mass=1.0)

    assert result.kick_velocity is None
    assert result.assumptions["kick_requires_final_mass"]
    assert result.assumptions["absolute_bms_charges_require_endpoint_data"]
    assert result.assumptions["superrest_frame_not_fixed"]
    assert result.assumptions["moreschi_supermomentum_not_computed"]
    assert result.assumptions["final_mass_from_initial_minus_radiation"] < 1.0
    assert not result.diagnostics["superrest_frame_fixed"]
    assert not result.diagnostics["absolute_bms_charges_computed"]


def test_bms_frame_diagnostics_can_disable_optional_fluxes():
    modes = make_bms_diagnostic_modes()

    result = compute_bms_frame_diagnostics(
        modes,
        compute_linear_momentum=False,
        compute_angular_momentum=False,
        compute_omitted_inspiral=False,
    )

    assert result.energy_radiated is not None
    assert result.radiated_linear_momentum is None
    assert result.kick_velocity is None
    assert result.angular_momentum_radiated is None
    assert result.omitted_inspiral is None


def test_bms_frame_diagnostics_reports_optional_memory_errors():
    modes = make_bms_diagnostic_modes()

    result = compute_bms_frame_diagnostics(
        modes,
        compute_memory_finite_time=True,
    )

    assert result.memory_finite_time is not None
    assert "available" in result.memory_finite_time


def test_bms_frame_diagnostics_to_dict_is_json_friendly():
    modes = make_bms_diagnostic_modes()

    result = compute_bms_frame_diagnostics(modes, final_mass=0.95)
    data = result.to_dict()

    assert isinstance(data["radiated_linear_momentum"], list)
    assert isinstance(data["kick_velocity"], list)
    assert data["diagnostics"]["news"]["spin_weight"] == -2


def test_bms_frame_diagnostics_validates_spin_weight():
    modes = make_bms_diagnostic_modes(spin_weight=0)

    with pytest.raises(ValueError, match="spin_weight=-2"):
        compute_bms_frame_diagnostics(modes)
