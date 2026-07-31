"""Tests for the opt-in spin-memory API surface."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools.modes_array import ModesArray
from waveformtools.spin_memory import (
    SpinMemoryConfig,
    add_spin_memory_in_place,
    compute_spin_memory_from_news,
    compute_spin_memory_from_strain,
    compute_spin_memory_source_from_news,
    compute_spin_memory_source_from_strain,
    remove_spin_memory_in_place,
    with_spin_memory,
    without_spin_memory,
)


def make_spin_memory_test_modes(
    spin_weight: int = -2,
    with_admixture: bool = False,
    amplitude: float = 1.0,
) -> ModesArray:
    from spectools.spherical.grids import GLGrid

    time_axis = np.linspace(-4.0, 4.0, 64)
    modes = ModesArray(
        ell_max=3,
        time_axis=time_axis,
        spin_weight=spin_weight,
        Grid=GLGrid(L=6),
    )
    modes.create_modes_array(ell_max=3, data_len=len(time_axis))
    signal = (
        amplitude
        * np.exp(-0.1 * time_axis**2)
        * np.exp(0.2j * time_axis)
    )
    modes.set_mode_data(ell=2, emm=2, data=signal)
    modes.set_mode_data(ell=2, emm=-2, data=np.conjugate(signal))
    if with_admixture:
        modes.set_mode_data(
            ell=2, emm=1, data=0.3 * signal * np.exp(0.05j * time_axis)
        )
        modes.set_mode_data(
            ell=3, emm=0, data=0.1 * signal.real.astype(np.complex128)
        )
    return modes


def test_spin_memory_config_validation():
    config = SpinMemoryConfig(ell_min=2, ell_max=3, memory_ell_max=4)

    assert config.to_dict()["source_normalization"] == "curl_flux"
    assert config.to_dict()["time_profile"] == "cumulative_flux"

    with pytest.raises(ValueError, match="ell_min"):
        SpinMemoryConfig(ell_min=1)

    with pytest.raises(ValueError, match="integration_constant"):
        SpinMemoryConfig(integration_constant="free")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="source_normalization"):
        SpinMemoryConfig(source_normalization="unknown")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="source_scale"):
        SpinMemoryConfig(source_scale=np.inf)

    with pytest.raises(ValueError, match="source_sign"):
        SpinMemoryConfig(source_sign=np.nan)

    with pytest.raises(ValueError, match="time_profile"):
        SpinMemoryConfig(time_profile="instantaneous")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="removal_tolerance"):
        SpinMemoryConfig(removal_tolerance=-1.0)

    with pytest.raises(ValueError, match="removal_max_iterations"):
        SpinMemoryConfig(removal_max_iterations=0)

    with pytest.raises(TypeError, match="spin-memory config"):
        SpinMemoryConfig.from_value(3.14)  # type: ignore[arg-type]


def test_spin_memory_source_is_magnetic_parity():
    modes = make_spin_memory_test_modes(with_admixture=True)

    source_modes = compute_spin_memory_source_from_strain(modes)

    for ell in range(1, source_modes.ell_max + 1):
        for emm in range(-ell, ell + 1):
            lhs = np.asarray(source_modes.mode(ell, emm))
            rhs = (-1.0) ** emm * np.conjugate(
                np.asarray(source_modes.mode(ell, -emm))
            )
            np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=1e-14)


def test_spin_memory_strain_is_magnetic_parity():
    modes = make_spin_memory_test_modes(with_admixture=True)

    memory_modes = compute_spin_memory_from_strain(modes)

    assert memory_modes.spin_weight == -2
    assert np.max(np.abs(memory_modes.modes_data)) > 0.0
    for ell, emm_list in memory_modes.modes_list:
        for emm in emm_list:
            lhs = np.asarray(memory_modes.mode(ell, emm))
            rhs = -((-1.0) ** emm) * np.conjugate(
                np.asarray(memory_modes.mode(ell, -emm))
            )
            np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=1e-14)


def test_spin_memory_m_structure_equal_mass_aligned():
    modes = make_spin_memory_test_modes(with_admixture=False)

    memory_modes = compute_spin_memory_from_strain(modes)

    for ell, emm_list in memory_modes.modes_list:
        for emm in emm_list:
            data = np.asarray(memory_modes.mode(ell, emm))
            if emm in (0, 4, -4):
                continue
            np.testing.assert_allclose(data, 0.0, atol=1e-14)
    m0_total = sum(
        np.max(np.abs(np.asarray(memory_modes.mode(ell, 0))))
        for ell in range(2, memory_modes.ell_max + 1)
    )
    assert m0_total > 0.0
    for ell in range(2, memory_modes.ell_max + 1):
        m0_data = np.asarray(memory_modes.mode(ell, 0))
        np.testing.assert_allclose(m0_data.real, 0.0, atol=1e-14)


def test_spin_memory_zero_news_gives_zero_memory():
    modes = make_spin_memory_test_modes()
    constant = np.full(len(modes.time_axis), 0.3 - 0.1j)
    modes._modes_data = np.zeros_like(modes.modes_data)
    modes.set_mode_data(ell=2, emm=2, data=constant)

    memory_modes = compute_spin_memory_from_strain(modes)

    np.testing.assert_allclose(memory_modes.modes_data, 0.0, atol=1e-13)


def test_spin_memory_starts_at_zero():
    modes = make_spin_memory_test_modes(with_admixture=True)

    memory_modes = compute_spin_memory_from_strain(modes)

    np.testing.assert_allclose(
        memory_modes.modes_data[..., 0], 0.0, atol=1e-15
    )


def test_spin_memory_curl_source_l1_matches_bms_jz_flux():
    from waveformtools.BMS import compute_angular_momentum_evolution

    modes = make_spin_memory_test_modes(with_admixture=True)
    news_modes = modes.get_news_from_strain()

    source_modes = compute_spin_memory_source_from_news(
        news_modes,
        strain_modes=modes,
    )
    s10 = np.asarray(source_modes.mode(1, 0))

    _, _, _, djz_dt = compute_angular_momentum_evolution(modes, news_modes)

    np.testing.assert_allclose(s10.imag, 0.0, atol=1e-14)
    expected = np.sqrt(4.0 * np.pi / 3.0) / (16.0 * np.pi) * s10.real
    np.testing.assert_allclose(djz_dt, expected, rtol=1e-10, atol=1e-16)


def test_spin_memory_source_normalization_is_configurable():
    modes = make_spin_memory_test_modes(with_admixture=True)

    default_source = compute_spin_memory_source_from_strain(modes)
    legacy_source = compute_spin_memory_source_from_strain(
        modes,
        source_normalization="balance_law_16pi",
    )
    flipped_source = compute_spin_memory_source_from_strain(
        modes,
        source_sign=-1.0,
    )

    np.testing.assert_allclose(
        default_source.modes_data,
        legacy_source.modes_data * (16.0 * np.pi),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        flipped_source.modes_data,
        -default_source.modes_data,
        rtol=1e-12,
        atol=1e-12,
    )


def test_spin_memory_from_news_matches_from_strain():
    modes = make_spin_memory_test_modes(with_admixture=True)
    news_modes = modes.get_news_from_strain()

    from_strain = compute_spin_memory_from_strain(modes)
    from_news = compute_spin_memory_from_news(
        news_modes,
        strain_modes=modes,
    )

    np.testing.assert_allclose(
        from_news.modes_data,
        from_strain.modes_data,
        rtol=1e-12,
        atol=1e-14,
    )


def test_spin_memory_from_news_without_strain_uses_integrated_news():
    modes = make_spin_memory_test_modes(with_admixture=True)
    news_modes = modes.get_news_from_strain()

    memory_modes = compute_spin_memory_from_news(news_modes)

    assert memory_modes.spin_weight == -2
    assert np.max(np.abs(memory_modes.modes_data)) > 0.0


def test_without_spin_memory_exact_subtraction_roundtrip():
    modes = make_spin_memory_test_modes(with_admixture=True)
    memory_modes = compute_spin_memory_from_strain(modes)

    with_memory = with_spin_memory(modes, memory_modes=memory_modes)
    recovered = without_spin_memory(with_memory, memory_modes=memory_modes)

    scale = np.max(np.abs(modes.modes_data))
    np.testing.assert_allclose(
        recovered.modes_data,
        modes.modes_data,
        rtol=0.0,
        atol=1e-15 * scale,
    )
    metadata = recovered.spin_memory_metadata
    assert metadata["included"] is False
    assert metadata["removed"] is True
    assert metadata["removal"]["mode"] == "exact_subtraction"


def test_without_spin_memory_fixed_point_roundtrip():
    # The fixed-point map contracts with factor ~|memory|/|h|, which is
    # bilinear in the strain: a perturbative amplitude (the physical
    # regime -- real memory is a few percent of the strain) is required
    # for convergence.
    modes = make_spin_memory_test_modes(with_admixture=True, amplitude=0.1)

    with_memory = with_spin_memory(modes)
    recovered = without_spin_memory(with_memory)

    scale = np.max(np.abs(modes.modes_data))
    np.testing.assert_allclose(
        recovered.modes_data,
        modes.modes_data,
        atol=1e-9 * scale,
    )
    removal = recovered.spin_memory_metadata["removal"]
    assert removal["mode"] == "fixed_point"
    assert removal["converged"] is True


def test_spin_memory_removal_after_discarding_memory_modes():
    modes = make_spin_memory_test_modes(with_admixture=True, amplitude=0.1)
    original_data = np.array(modes.modes_data, copy=True)

    add_spin_memory_in_place(modes)
    assert not np.allclose(modes.modes_data, original_data)

    returned = remove_spin_memory_in_place(modes)

    assert returned is modes
    scale = np.max(np.abs(original_data))
    np.testing.assert_allclose(
        modes.modes_data,
        original_data,
        atol=1e-9 * scale,
    )
    assert modes.spin_memory_metadata["removed"] is True


def test_modes_array_spin_memory_wrappers_match_module_functions():
    modes = make_spin_memory_test_modes(with_admixture=True, amplitude=0.1)

    memory_via_method = modes.compute_spin_memory()
    memory_via_module = compute_spin_memory_from_strain(modes)
    np.testing.assert_allclose(
        memory_via_method.modes_data,
        memory_via_module.modes_data,
        rtol=1e-14,
        atol=1e-16,
    )

    source_via_method = modes.compute_spin_memory_source()
    assert source_via_method.spin_weight == 0
    assert np.max(np.abs(source_via_method.modes_data)) > 0.0

    with_memory = modes.with_spin_memory()
    recovered = with_memory.without_spin_memory()
    scale = np.max(np.abs(modes.modes_data))
    np.testing.assert_allclose(
        recovered.modes_data,
        modes.modes_data,
        atol=1e-9 * scale,
    )


def test_modes_array_displacement_removal_wrapper():
    modes = make_spin_memory_test_modes(with_admixture=True)

    with_memory = modes.with_displacement_memory()
    recovered = with_memory.without_displacement_memory()

    scale = np.max(np.abs(modes.modes_data))
    np.testing.assert_allclose(
        recovered.modes_data,
        modes.modes_data,
        atol=1e-9 * scale,
    )

    in_place = modes.deepcopy()
    in_place.add_displacement_memory_in_place()
    in_place.remove_displacement_memory_in_place()
    np.testing.assert_allclose(
        in_place.modes_data,
        modes.modes_data,
        atol=1e-9 * scale,
    )


def test_spin_memory_rejects_non_strain_spin_weight():
    modes = make_spin_memory_test_modes(spin_weight=0)

    with pytest.raises(ValueError, match="spin_weight=-2"):
        compute_spin_memory_from_strain(modes)


def test_spin_memory_rejects_mismatched_time_axes():
    modes = make_spin_memory_test_modes()
    news_modes = modes.get_news_from_strain()
    strain_shifted = modes.deepcopy()
    strain_shifted._time_axis = modes.time_axis + 1.0

    with pytest.raises(ValueError, match="same time axis"):
        compute_spin_memory_source_from_news(
            news_modes,
            strain_modes=strain_shifted,
        )
