from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from waveformtools.models.teobresums import (
    TEOBResumSWaveformModel,
    is_teobresums_approximant,
    mode_to_teob_key,
)


def test_teobresums_approximant_detection():
    assert is_teobresums_approximant("TEOBResumS-Dali")
    assert is_teobresums_approximant("TEOBResumS_GIOTTO")
    assert is_teobresums_approximant("Dali")
    assert not is_teobresums_approximant("SEOBNRv5PHM")


def test_teobresums_python_extension_is_installed():
    import EOBRun_module

    assert callable(EOBRun_module.EOBRunPy)


def test_teobresums_capabilities_match_public_api():
    model = TEOBResumSWaveformModel(parameters_dict={"f_lower": 20.0})

    assert model.capabilities()["td_modes"] is True
    assert model.capabilities()["td_polarizations"] is False


def test_signed_mode_key_mapping():
    assert mode_to_teob_key(2, 2) == "1"
    assert mode_to_teob_key(2, -2) == "-1"
    assert mode_to_teob_key(2, 1) == "0"
    assert mode_to_teob_key(2, -1) == "-0"
    assert mode_to_teob_key(4, 0) == "40"


def test_model_populates_native_signed_modes(monkeypatch):
    time_axis = np.array([0.0, 1.0, 2.0])

    def fake_eob_run(_pars):
        hlm = {
            "1": [np.array([2.0, 3.0, 4.0]), np.array([0.0, 0.1, 0.2])],
            "-1": [np.array([5.0, 6.0, 7.0]), np.array([0.3, 0.4, 0.5])],
            "20": [np.array([0.1, 0.2, 0.3]), np.zeros(3)],
        }
        dynamics = {"E": np.array([0.99, 0.98, 0.97])}
        return time_axis, np.zeros(3), np.zeros(3), hlm, dynamics

    monkeypatch.setitem(
        sys.modules,
        "EOBRun_module",
        SimpleNamespace(EOBRunPy=fake_eob_run),
    )

    model = TEOBResumSWaveformModel(
        parameters_dict={
            "approximant": "TEOBResumS-Dali",
            "mass1": 35.0,
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
            "f_lower": 20.0,
            "f_ref": 20.0,
            "f_max": 512.0,
            "delta_t": 1.0 / 4096.0,
            "delta_f": 0.25,
            "ell_max": 2,
            "teobresums_scale_by_eta": False,
        }
    )
    modes = model.get_td_waveform_modes(dimensionless=True, L=2)

    assert np.allclose(
        modes.mode(2, 2),
        np.array([2.0, 3.0, 4.0]) * np.exp(1j * np.array([0.0, 0.1, 0.2])),
    )
    assert np.allclose(
        modes.mode(2, -2),
        np.array([5.0, 6.0, 7.0]) * np.exp(1j * np.array([0.3, 0.4, 0.5])),
    )
    assert np.allclose(modes.mode(2, 0), np.array([0.1, 0.2, 0.3]))
    assert np.allclose(modes.mode(2, 1), 0.0)
    assert model.get_corresponding_eob_hamiltonian() == 0.99


def test_model_reconstructs_missing_negative_modes(monkeypatch):
    time_axis = np.array([0.0, 1.0, 2.0])

    def fake_eob_run(_pars):
        hlm = {
            "1": [np.array([2.0, 3.0, 4.0]), np.array([0.0, 0.1, 0.2])],
        }
        dynamics = {"E": np.array([0.99, 0.98, 0.97])}
        return time_axis, np.zeros(3), np.zeros(3), hlm, dynamics

    monkeypatch.setitem(
        sys.modules,
        "EOBRun_module",
        SimpleNamespace(EOBRunPy=fake_eob_run),
    )

    model = TEOBResumSWaveformModel(
        parameters_dict={
            "approximant": "TEOBResumS-Dali",
            "mass1": 35.0,
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
            "f_lower": 20.0,
            "f_ref": 20.0,
            "f_max": 512.0,
            "delta_t": 1.0 / 4096.0,
            "delta_f": 0.25,
            "ell_max": 2,
            "teobresums_scale_by_eta": False,
        }
    )
    modes = model.get_td_waveform_modes(dimensionless=True, L=2)
    positive = np.array([2.0, 3.0, 4.0]) * np.exp(1j * np.array([0.0, 0.1, 0.2]))

    assert np.allclose(modes.mode(2, 2), positive)
    assert np.allclose(modes.mode(2, -2), np.conjugate(positive))
