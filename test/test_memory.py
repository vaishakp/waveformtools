"""Tests for the opt-in displacement-memory API surface."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools.memory import (
    DisplacementMemoryConfig,
    _source_modes_to_memory_strain,
    compute_displacement_memory_from_news,
    compute_displacement_memory_from_strain,
    compute_displacement_memory_source_from_news,
    diagnose_displacement_memory_finite_time,
    diagnose_omitted_inspiral,
    remove_displacement_memory_in_place,
    with_displacement_memory,
    without_displacement_memory,
)
from waveformtools.modes_array import ModesArray


def make_memory_test_modes(spin_weight: int = -2) -> ModesArray:
    from spectools.spherical.grids import GLGrid

    time_axis = np.linspace(-4.0, 4.0, 64)
    modes = ModesArray(
        ell_max=2,
        time_axis=time_axis,
        spin_weight=spin_weight,
        Grid=GLGrid(L=4),
    )
    modes.create_modes_array(ell_max=2, data_len=len(time_axis))
    signal = np.exp(-0.1 * time_axis**2) * np.exp(0.2j * time_axis)
    modes.set_mode_data(ell=2, emm=2, data=signal)
    modes.set_mode_data(ell=2, emm=-2, data=np.conjugate(signal))
    return modes


def test_displacement_memory_config_validation():
    config = DisplacementMemoryConfig(ell_min=2, ell_max=4, memory_ell_max=4)

    assert config.to_dict()["integration_constant"] == "zero_at_start"
    assert config.to_dict()["source_normalization"] == "news_squared"

    with pytest.raises(ValueError, match="ell_min"):
        DisplacementMemoryConfig(ell_min=1)

    with pytest.raises(ValueError, match="integration_constant"):
        DisplacementMemoryConfig(
            integration_constant="free"  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="source_normalization"):
        DisplacementMemoryConfig(
            source_normalization="unknown"  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="source_scale"):
        DisplacementMemoryConfig(source_scale=np.inf)


def test_compute_displacement_memory_validates_and_marks_kernel_boundary():
    modes = make_memory_test_modes()

    memory_modes = modes.compute_displacement_memory()

    assert memory_modes.spin_weight == -2
    assert memory_modes.ell_max == modes.ell_max
    assert np.allclose(memory_modes.time_axis, modes.time_axis)
    assert np.allclose(memory_modes.modes_data[..., 0], 0.0)
    assert np.max(np.abs(memory_modes.modes_data)) > 0.0
    assert (
        memory_modes.displacement_memory_metadata["implementation"]
        == "bar_eth2_inverse"
    )


def test_compute_displacement_memory_source_projects_news_intensity():
    news_modes = make_memory_test_modes()

    source_modes = compute_displacement_memory_source_from_news(news_modes)

    assert source_modes.spin_weight == 0
    assert np.allclose(source_modes.time_axis, news_modes.time_axis)
    assert source_modes.modes_data.shape[-1] == news_modes.data_len
    assert np.all(np.isfinite(source_modes.modes_data))
    assert np.max(np.abs(source_modes.modes_data)) > 0.0
    assert (
        source_modes.displacement_memory_source_metadata["source"]
        == "|news|^2"
    )
    assert np.isclose(
        source_modes.displacement_memory_source_metadata["source_factor"],
        1.0,
    )


def test_memory_source_normalization_is_configurable():
    news_modes = make_memory_test_modes()

    news_squared_source = compute_displacement_memory_source_from_news(
        news_modes
    )
    legacy_source = compute_displacement_memory_source_from_news(
        news_modes,
        source_normalization="balance_law_16pi",
    )
    scaled_source = compute_displacement_memory_source_from_news(
        news_modes,
        source_normalization="news_squared",
        source_scale=0.25,
    )

    np.testing.assert_allclose(
        news_squared_source.modes_data,
        legacy_source.modes_data * (16.0 * np.pi),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        scaled_source.modes_data,
        0.25 * news_squared_source.modes_data,
        rtol=1e-12,
        atol=1e-12,
    )
    assert (
        news_squared_source.displacement_memory_source_metadata["source"]
        == "|news|^2"
    )
    assert (
        legacy_source.displacement_memory_source_metadata["source"]
        == "|news|^2/(16*pi)"
    )
    assert (
        scaled_source.displacement_memory_source_metadata["source"]
        == "0.25*|news|^2"
    )


def test_compute_displacement_memory_source_zero_news_gives_zero_modes():
    news_modes = make_memory_test_modes()
    news_modes._modes_data = np.zeros_like(news_modes.modes_data)

    source_modes = compute_displacement_memory_source_from_news(news_modes)

    assert np.allclose(source_modes.modes_data, 0.0)


def test_compute_displacement_memory_from_news_inverts_bar_eth2_source():
    news_modes = make_memory_test_modes()
    source_modes = compute_displacement_memory_source_from_news(
        news_modes,
        memory_ell_max=news_modes.ell_max,
    )

    memory_modes = compute_displacement_memory_from_news(news_modes)

    eigenvalue = np.sqrt((2 - 1) * 2 * (2 + 1) * (2 + 2))
    expected_integral = cumulative_trapezoid_zero_at_start(
        news_modes.time_axis,
        source_modes.mode(2, 0),
    )
    np.testing.assert_allclose(
        memory_modes.mode(2, 0) * eigenvalue,
        expected_integral,
        rtol=1e-12,
        atol=1e-12,
    )


def test_memory_kernel_matches_balance_law_eth2_conjugate_operator():
    spin_coefficient = pytest.importorskip("qlmtools.spin_coefficient")
    eth_n_modes_from_modes = spin_coefficient.eth_n_modes_from_modes

    source_modes = make_scalar_memory_source_modes()

    memory_modes = _source_modes_to_memory_strain(
        source_modes,
        DisplacementMemoryConfig(memory_ell_max=source_modes.ell_max),
        memory_ell_max=source_modes.ell_max,
    )
    recovered_source = eth_n_modes_from_modes(
        memory_modes.time_derivative(method="spline").bar(),
        memory_modes.Grid,
        times=2,
    )

    for emm in range(-2, 3):
        np.testing.assert_allclose(
            recovered_source.mode(2, emm),
            source_modes.mode(2, emm),
            rtol=2e-2,
            atol=2e-12,
        )


def test_compute_displacement_memory_from_news_zero_news_gives_zero_memory():
    news_modes = make_memory_test_modes()
    news_modes._modes_data = np.zeros_like(news_modes.modes_data)

    memory_modes = compute_displacement_memory_from_news(news_modes)

    assert memory_modes.spin_weight == -2
    assert np.allclose(memory_modes.modes_data, 0.0)


def test_modes_array_memory_source_uses_existing_news_derivative_path():
    strain_modes = make_memory_test_modes()

    source_modes = strain_modes.compute_displacement_memory_source()

    assert source_modes.spin_weight == 0
    assert np.allclose(source_modes.time_axis, strain_modes.time_axis)
    assert np.max(np.abs(source_modes.modes_data)) > 0.0


def test_finite_time_memory_diagnostic_reports_endpoint_sensitivity():
    strain_modes = make_memory_test_modes()

    diagnostic = diagnose_displacement_memory_finite_time(
        strain_modes,
        window_fraction=0.2,
        start_indices=(0, 8, 16),
    )
    method_diagnostic = strain_modes.diagnose_displacement_memory_finite_time(
        window_fraction=0.2,
        start_indices=(0, 8, 16),
    )

    assert diagnostic["memory_endpoint_norm"] > 0.0
    assert diagnostic["window_size"] == 13
    assert len(diagnostic["start_index_sensitivity"]) == 3
    assert np.isfinite(diagnostic["early_window_fraction_of_endpoint"])
    assert np.isfinite(diagnostic["late_window_fraction_of_endpoint"])
    assert method_diagnostic["memory_endpoint_norm"] == pytest.approx(
        diagnostic["memory_endpoint_norm"]
    )


def test_omitted_inspiral_diagnostic_reports_initial_frequency():
    strain_modes = make_memory_test_modes()

    diagnostic = diagnose_omitted_inspiral(
        strain_modes,
        min_cycles=0.0,
        high_initial_power_fraction=2.0,
    )
    method_diagnostic = strain_modes.diagnose_omitted_inspiral(
        min_cycles=0.0,
        high_initial_power_fraction=2.0,
    )

    assert diagnostic["mode"] == (2, 2)
    assert diagnostic["initial_angular_frequency"] == pytest.approx(0.2)
    assert diagnostic["total_cycles"] > 0.0
    assert diagnostic["early_energy_fraction"] >= 0.0
    assert diagnostic["omitted_inspiral_likely"] is False
    assert method_diagnostic["initial_period"] == pytest.approx(
        diagnostic["initial_period"]
    )


def test_compute_displacement_memory_rejects_non_strain_spin_weight():
    modes = make_memory_test_modes(spin_weight=0)

    with pytest.raises(ValueError, match="spin_weight=-2"):
        modes.compute_displacement_memory()


def test_compute_displacement_memory_from_news_rejects_bad_time_axis():
    modes = make_memory_test_modes()
    modes._time_axis = modes.time_axis[::-1]

    with pytest.raises(ValueError, match="strictly increasing"):
        compute_displacement_memory_from_news(modes)


def test_with_displacement_memory_accepts_explicit_memory_modes():
    modes = make_memory_test_modes()
    zero_memory = modes.deepcopy()
    zero_memory._modes_data = np.zeros_like(modes.modes_data)

    with_memory = modes.with_displacement_memory(memory_modes=zero_memory)

    assert with_memory is not modes
    assert np.allclose(with_memory.modes_data, modes.modes_data)
    assert with_memory.displacement_memory_metadata["included"] is True


def test_add_displacement_memory_in_place_accepts_explicit_memory_modes():
    modes = make_memory_test_modes()
    original_data = np.array(modes.modes_data, copy=True)
    zero_memory = modes.deepcopy()
    zero_memory._modes_data = np.zeros_like(modes.modes_data)

    returned = modes.add_displacement_memory_in_place(memory_modes=zero_memory)

    assert returned is modes
    assert np.allclose(modes.modes_data, original_data)
    assert modes.displacement_memory_metadata["included"] is True


def test_with_displacement_memory_rejects_incompatible_memory_modes():
    modes = make_memory_test_modes()
    memory = make_memory_test_modes()
    memory._time_axis = memory.time_axis + 1.0

    with pytest.raises(ValueError, match="same time axis"):
        modes.with_displacement_memory(memory_modes=memory)


def test_without_displacement_memory_exact_subtraction_roundtrip():
    modes = make_memory_test_modes()
    memory_modes = compute_displacement_memory_from_strain(modes)

    with_memory = with_displacement_memory(modes, memory_modes=memory_modes)
    recovered = without_displacement_memory(
        with_memory,
        memory_modes=memory_modes,
    )

    scale = np.max(np.abs(modes.modes_data))
    np.testing.assert_allclose(
        recovered.modes_data,
        modes.modes_data,
        rtol=0.0,
        atol=1e-15 * scale,
    )
    metadata = recovered.displacement_memory_metadata
    assert metadata["included"] is False
    assert metadata["removed"] is True
    assert metadata["removal"]["mode"] == "exact_subtraction"


def test_without_displacement_memory_fixed_point_roundtrip():
    modes = make_memory_test_modes()

    with_memory = with_displacement_memory(modes)
    recovered = without_displacement_memory(with_memory)

    scale = np.max(np.abs(modes.modes_data))
    np.testing.assert_allclose(
        recovered.modes_data,
        modes.modes_data,
        atol=1e-9 * scale,
    )
    removal = recovered.displacement_memory_metadata["removal"]
    assert removal["mode"] == "fixed_point"
    assert removal["converged"] is True
    assert removal["iterations"] >= 2
    assert removal["residual_history"][-1] <= removal["tolerance"]


def test_remove_displacement_memory_in_place_fixed_point():
    modes = make_memory_test_modes()
    original_data = np.array(modes.modes_data, copy=True)
    modes.add_displacement_memory_in_place()

    returned = remove_displacement_memory_in_place(modes)

    assert returned is modes
    scale = np.max(np.abs(original_data))
    np.testing.assert_allclose(
        modes.modes_data,
        original_data,
        atol=1e-9 * scale,
    )
    assert modes.displacement_memory_metadata["removed"] is True


def test_fixed_point_removal_raises_on_non_convergence():
    modes = make_memory_test_modes()
    with_memory = with_displacement_memory(modes)

    with pytest.raises(RuntimeError, match="did not converge"):
        without_displacement_memory(
            with_memory,
            removal_tolerance=1e-16,
            removal_max_iterations=1,
        )


def test_removal_config_validation():
    with pytest.raises(ValueError, match="removal_tolerance"):
        DisplacementMemoryConfig(removal_tolerance=0.0)

    with pytest.raises(ValueError, match="removal_tolerance"):
        DisplacementMemoryConfig(removal_tolerance=np.inf)

    with pytest.raises(ValueError, match="removal_max_iterations"):
        DisplacementMemoryConfig(removal_max_iterations=0)


def test_arithmetic_ops_invalidate_news_cache():
    modes = make_memory_test_modes()
    news_before = modes.get_news_from_strain()

    doubled = modes + modes
    news_doubled = doubled.get_news_from_strain()

    np.testing.assert_allclose(
        news_doubled.modes_data,
        2.0 * news_before.modes_data,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "operate",
    [
        lambda m: m + m,
        lambda m: 1.0 + m,
        lambda m: m - 0.5 * m,
        lambda m: 1.0 - m,
        lambda m: m * 3.0,
        lambda m: 3.0 * m,
        lambda m: m / 2.0,
        lambda m: m[5],
        lambda m: np.conjugate(m),
        lambda m: m.time_derivative(method="FD"),
    ],
    ids=[
        "add",
        "radd",
        "sub",
        "rsub",
        "mul",
        "rmul",
        "truediv",
        "getitem",
        "conjugate",
        "time_derivative_FD",
    ],
)
def test_mutating_ops_drop_cached_news_spline(operate):
    modes = make_memory_test_modes()
    modes.get_news_from_strain()
    assert "_news_modes_spline" in modes.__dict__

    result = operate(modes)

    assert "_news_modes_spline" not in result.__dict__


def test_add_displacement_memory_in_place_invalidates_news_cache():
    modes = make_memory_test_modes()
    news_before = np.array(
        modes.get_news_from_strain().modes_data, copy=True
    )

    modes.add_displacement_memory_in_place()

    news_after = modes.get_news_from_strain().modes_data
    assert not np.allclose(news_after, news_before)


def cumulative_trapezoid_zero_at_start(xdata, ydata):
    integral = np.zeros_like(ydata, dtype=np.result_type(ydata, complex))
    increments = 0.5 * (ydata[..., 1:] + ydata[..., :-1]) * np.diff(xdata)
    integral[..., 1:] = np.cumsum(increments, axis=-1)
    return integral


def make_scalar_memory_source_modes() -> ModesArray:
    from spectools.spherical.grids import GLGrid

    time_axis = np.linspace(-2.0, 2.0, 64)
    source_modes = ModesArray(
        ell_max=2,
        time_axis=time_axis,
        spin_weight=0,
        Grid=GLGrid(L=4),
    )
    source_modes.create_modes_array(ell_max=2, data_len=len(time_axis))
    source_modes.set_mode_data(ell=2, emm=-2, data=(0.3 - 0.2j) * time_axis)
    source_modes.set_mode_data(ell=2, emm=-1, data=(-0.1 + 0.4j) * time_axis)
    source_modes.set_mode_data(ell=2, emm=0, data=0.7 * time_axis)
    source_modes.set_mode_data(ell=2, emm=1, data=(0.2 + 0.1j) * time_axis)
    source_modes.set_mode_data(ell=2, emm=2, data=(-0.5 - 0.3j) * time_axis)
    return source_modes
