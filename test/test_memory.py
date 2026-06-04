"""Tests for the opt-in displacement-memory API surface."""

from __future__ import annotations

import numpy as np
import pytest

from waveformtools.memory import (
    DisplacementMemoryConfig,
    compute_displacement_memory_from_news,
    compute_displacement_memory_source_from_news,
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

    with pytest.raises(ValueError, match="ell_min"):
        DisplacementMemoryConfig(ell_min=1)

    with pytest.raises(ValueError, match="integration_constant"):
        DisplacementMemoryConfig(
            integration_constant="free"  # type: ignore[arg-type]
        )


def test_compute_displacement_memory_validates_and_marks_kernel_boundary():
    modes = make_memory_test_modes()

    with pytest.raises(NotImplementedError, match="spectral kernel"):
        modes.compute_displacement_memory()


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
        == "|news|^2/(16*pi)"
    )


def test_compute_displacement_memory_source_zero_news_gives_zero_modes():
    news_modes = make_memory_test_modes()
    news_modes._modes_data = np.zeros_like(news_modes.modes_data)

    source_modes = compute_displacement_memory_source_from_news(news_modes)

    assert np.allclose(source_modes.modes_data, 0.0)


def test_modes_array_memory_source_uses_existing_news_derivative_path():
    strain_modes = make_memory_test_modes()

    source_modes = strain_modes.compute_displacement_memory_source()

    assert source_modes.spin_weight == 0
    assert np.allclose(source_modes.time_axis, strain_modes.time_axis)
    assert np.max(np.abs(source_modes.modes_data)) > 0.0


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
