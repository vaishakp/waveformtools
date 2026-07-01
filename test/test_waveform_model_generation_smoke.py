from __future__ import annotations

import numpy as np


def _assert_populated_h22_modes(modes):
    h22 = np.asarray(modes.mode(2, 2), dtype=np.complex128)

    assert modes.data_len > 10
    assert h22.shape == (modes.data_len,)
    assert np.all(np.isfinite(modes.time_axis))
    assert np.all(np.isfinite(h22))
    assert np.nanmax(np.abs(h22)) > 0.0


def test_lal_waveform_model_generates_modes():
    from waveformtools.models.lal import LALWaveformModel

    model = LALWaveformModel(
        parameters_dict={
            "approximant": "IMRPhenomXPHM",
            "mass1": 30.0,
            "mass2": 30.0,
            "spin1x": 0.0,
            "spin1y": 0.0,
            "spin1z": 0.0,
            "spin2x": 0.0,
            "spin2y": 0.0,
            "spin2z": 0.0,
            "distance": 400.0,
            "inclination": 0.0,
            "phi_ref": 0.0,
            "f_lower": 30.0,
            "f_ref": 30.0,
            "f_max": 256.0,
            "delta_f": 1.0,
            "delta_t": 1.0 / 1024.0,
            "ell_max": 2,
        }
    )

    _assert_populated_h22_modes(model.get_td_waveform_modes(dimensionless=True))


def test_eob_waveform_model_generates_modes():
    from waveformtools.models.eob import EOBWaveformModel

    model = EOBWaveformModel(
        parameters_dict={
            "approximant": "SEOBNRv5PHM",
            "mass1": 30.0,
            "mass2": 30.0,
            "spin1x": 0.0,
            "spin1y": 0.0,
            "spin1z": 0.0,
            "spin2x": 0.0,
            "spin2y": 0.0,
            "spin2z": 0.0,
            "distance": 400.0,
            "inclination": 0.0,
            "phi_ref": 0.0,
            "f_lower": 30.0,
            "f_ref": 30.0,
            "delta_t": 1.0 / 1024.0,
            "ell_max": 2,
            "lmax_nyquist": 2,
        }
    )

    _assert_populated_h22_modes(
        model.get_td_waveform_modes(dimensionless=True, L=2)
    )


def test_teobresums_waveform_model_generates_modes():
    from waveformtools.models.teobresums import TEOBResumSWaveformModel

    model = TEOBResumSWaveformModel(
        parameters_dict={
            "approximant": "TEOBResumS-Dali",
            "mass1": 30.0,
            "mass2": 30.0,
            "spin1x": 0.0,
            "spin1y": 0.0,
            "spin1z": 0.0,
            "spin2x": 0.0,
            "spin2y": 0.0,
            "spin2z": 0.0,
            "phi_ref": 0.0,
            "inclination": 0.0,
            "distance": 1.0,
            "f_lower": 30.0,
            "f_ref": 30.0,
            "delta_t": 1.0 / 4096.0,
            "ell_max": 2,
        }
    )

    _assert_populated_h22_modes(
        model.get_td_waveform_modes(dimensionless=True, L=2)
    )
