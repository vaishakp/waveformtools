"""Tests for memory-bounded balance-law helpers."""

from __future__ import annotations

import numpy as np

from waveformtools.chunked_balance_laws import compare_chunked_to_full_rhs
from waveformtools.modes_array import ModesArray


def test_chunked_rhs_matches_full_rhs_for_smooth_synthetic_modes():
    from spectools.spherical.grids import GLGrid

    grid = GLGrid(L=4)
    time_axis = np.linspace(-2.0, 2.0, 96)
    modes = ModesArray(
        ell_max=2,
        time_axis=time_axis,
        spin_weight=-2,
        Grid=grid,
    )
    modes.create_modes_array(ell_max=2, data_len=len(time_axis))
    signal = np.exp(-0.2 * time_axis**2) * np.exp(0.3j * time_axis)
    modes.set_mode_data(ell=2, emm=2, data=signal)
    modes.set_mode_data(ell=2, emm=-2, data=np.conjugate(signal))

    comparison = compare_chunked_to_full_rhs(modes, grid, chunk_size=24)

    assert comparison["relative_l2_error"] < 1e-3
    assert np.all(np.isfinite(comparison["rhs_full"]))
    assert np.all(np.isfinite(comparison["rhs_chunked"]))
