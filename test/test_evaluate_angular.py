"""Tests for angular reconstruction target selection."""

from __future__ import annotations

import numpy as np
import pytest
from spectools.spherical.grids import GLGrid
from spectools.spherical.swsh import Yslm_vec

from waveformtools.modes_array import ModesArray


def _make_strain_modes(grid=None):
    time_axis = np.array([0.0, 1.0, 2.0])
    modes = ModesArray(
        ell_max=2,
        time_axis=time_axis,
        spin_weight=-2,
        Grid=grid,
    )
    modes.create_modes_array(data_len=len(time_axis))

    h22 = np.array([1.0 + 2.0j, 2.0 + 3.0j, 3.0 + 4.0j])
    h2m2 = np.array([0.5 - 0.25j, 0.75 - 0.5j, 1.0 - 0.75j])
    modes.set_mode_data(ell=2, emm=2, data=h22)
    modes.set_mode_data(ell=2, emm=-2, data=h2m2)
    return modes, h22, h2m2


def _explicit_spin_minus_two_sum(theta, phi, h22, h2m2):
    basis_22 = np.asarray(
        Yslm_vec(
            spin_weight=-2,
            ell=2,
            emm=2,
            theta_grid=theta,
            phi_grid=phi,
        )
    )
    basis_2m2 = np.asarray(
        Yslm_vec(
            spin_weight=-2,
            ell=2,
            emm=-2,
            theta_grid=theta,
            phi_grid=phi,
        )
    )
    angular_shape = (1,) * basis_22.ndim
    return (
        h22.reshape((-1, *angular_shape)) * basis_22
        + h2m2.reshape((-1, *angular_shape)) * basis_2m2
    )


def test_evaluate_angular_uses_object_grid_when_angles_are_omitted():
    grid = GLGrid(L=8)
    modes, h22, h2m2 = _make_strain_modes(grid=grid)
    theta, phi = grid.meshgrid

    reconstructed = modes.evaluate_angular()
    expected = _explicit_spin_minus_two_sum(theta, phi, h22, h2m2)

    assert reconstructed.shape == (modes.data_len, *theta.shape)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-13)


def test_evaluate_angular_scalar_angles_override_attached_grid():
    grid = GLGrid(L=8)
    modes, h22, h2m2 = _make_strain_modes(grid=grid)
    theta = 1.1
    phi = 0.7

    reconstructed = modes.evaluate_angular(theta=theta, phi=phi)
    expected = _explicit_spin_minus_two_sum(theta, phi, h22, h2m2)

    assert reconstructed.shape == (modes.data_len,)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-13)


def test_evaluate_angular_user_meshgrid_overrides_attached_grid():
    grid = GLGrid(L=8)
    modes, h22, h2m2 = _make_strain_modes(grid=grid)
    theta, phi = np.meshgrid(
        np.array([0.4, 1.0, 1.6]),
        np.array([0.2, 2.5]),
        indexing="ij",
    )

    reconstructed = modes.evaluate_angular(theta=theta, phi=phi)
    expected = _explicit_spin_minus_two_sum(theta, phi, h22, h2m2)

    assert reconstructed.shape == (modes.data_len, *theta.shape)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-13)


def test_evaluate_angular_requires_paired_matching_angles():
    modes, _h22, _h2m2 = _make_strain_modes()

    with pytest.raises(ValueError, match="supplied together"):
        modes.evaluate_angular(theta=1.0)

    with pytest.raises(ValueError, match="matching shapes"):
        modes.evaluate_angular(theta=np.array([1.0, 1.2]), phi=np.array([0.1]))
