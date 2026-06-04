"""Focused tests for spherical-array mode projection."""

import numpy as np
from spectools.spherical.grids import GLGrid
from spectools.spherical.swsh import Yslm_vec

from waveformtools.spherical_array import SphericalArray


def test_to_modes_array_projects_time_dependent_angular_data():
    grid = GLGrid(L=4)
    theta, phi = grid.meshgrid
    time_axis = np.array([0.0, 1.0, 2.0])
    amplitudes = np.array([1.0 + 0.5j, -0.25 + 0.2j, 0.75 - 0.1j])
    basis = Yslm_vec(
        spin_weight=0,
        ell=2,
        emm=1,
        theta_grid=theta,
        phi_grid=phi,
    )
    angular_data = (
        basis[:, :, np.newaxis] * amplitudes[np.newaxis, np.newaxis, :]
    )
    spherical_data = SphericalArray(
        label="test_time_domain",
        time_axis=time_axis,
        data=angular_data,
        data_len=len(time_axis),
        Grid=grid,
        spin_weight=0,
        ell_max=2,
    )

    modes = spherical_data.to_modes_array(Grid=grid, spin_weight=0, ell_max=2)

    np.testing.assert_allclose(modes.mode(2, 1), amplitudes, atol=1e-12)
