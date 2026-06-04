"""Tests for frequency-domain to time-domain mode conversion."""

from __future__ import annotations

import numpy as np

from waveformtools.fd_to_td import lal_fd_modes_to_td_modes
from waveformtools.modes_array import ModesArray


def test_lal_fd_modes_to_td_modes_round_trips_loader_convention():
    data_len = 65
    delta_f = 0.25
    time_axis = np.arange(data_len, dtype=float) / (data_len * delta_f)
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(data_len, d=time_axis[1]))
    phase = 2.0 * np.pi * 1.25 * time_axis + 0.1 * time_axis**2
    envelope = np.exp(-0.2 * (time_axis - time_axis.mean()) ** 2)
    expected_h22 = envelope * np.exp(1j * phase)
    raw_lal_fd = np.fft.fftshift(
        np.fft.fft(np.conjugate(expected_h22) / (data_len * delta_f))
    )
    stored_lal_loader_data = np.conjugate(raw_lal_fd) / data_len

    fd_modes = ModesArray(
        label="synthetic_lal_fd",
        ell_max=2,
        frequency_axis=frequency_axis,
        spin_weight=-2,
    )
    fd_modes.create_modes_array(ell_max=2, data_len=data_len)
    fd_modes.set_mode_data(
        ell=2,
        emm=2,
        data=stored_lal_loader_data,
    )

    td_modes = lal_fd_modes_to_td_modes(fd_modes, undo_warp=False)

    np.testing.assert_allclose(td_modes.time_axis, time_axis)
    np.testing.assert_allclose(td_modes.mode(2, 2), expected_h22, atol=1e-14)
