from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


def _base_eob_params():
    return {
        "mass1": 40.0,
        "mass2": 20.0,
        "spin1x": 0.0,
        "spin1y": 0.0,
        "spin1z": 0.1,
        "spin2x": 0.0,
        "spin2y": 0.0,
        "spin2z": -0.2,
        "phi_ref": 0.0,
        "distance": 400.0,
        "inclination": 0.0,
        "f_lower": 10.0,
        "f_ref": 20.0,
        "delta_t": 1.0 / 4096.0,
        "approximant": "SEOBNRv5PHM",
        "lmax_nyquist": 2,
    }


def test_waveform_model_initializes_and_updates_omega0_from_f_lower():
    from waveformtools.models.waveform_models import WaveformModel

    model = WaveformModel(parameters_dict={"f_lower": 10.0})

    assert model.omega0 == pytest.approx(np.pi * 10.0)

    model.update_parameters({"f_lower": 5.0})

    assert model.omega0 == pytest.approx(np.pi * 5.0)


def test_eob_compute_model_refreshes_parameters_and_passes_mass_with_dt(monkeypatch):
    from waveformtools.models.eob import EOBWaveformModel
    import waveformtools.models.utils as model_utils

    captured = {}

    fake_pyseobnr = ModuleType("pyseobnr")
    fake_generate_waveform = ModuleType("pyseobnr.generate_waveform")

    def fake_generate_modes_opt(q, chi_1, chi_2, omega0, omega_ref, **kwargs):
        captured.update(
            {
                "q": q,
                "chi_1": np.array(chi_1),
                "chi_2": np.array(chi_2),
                "omega0": omega0,
                "omega_ref": omega_ref,
                "kwargs": kwargs,
            }
        )
        model = SimpleNamespace(
            dynamics=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            final_mass=0.95,
        )
        return np.array([0.0, 0.1]), {"2,2": np.array([1.0 + 0.0j, 0.5 + 0.0j])}, model

    fake_generate_waveform.generate_modes_opt = fake_generate_modes_opt
    monkeypatch.setitem(sys.modules, "pyseobnr", fake_pyseobnr)
    monkeypatch.setitem(
        sys.modules,
        "pyseobnr.generate_waveform",
        fake_generate_waveform,
    )
    monkeypatch.setattr(
        model_utils,
        "get_modes_array_from_eob_modes_dict",
        lambda time_axis, modes_dict, L: SimpleNamespace(
            time_axis=time_axis,
            modes_dict=modes_dict,
            ell_max=L,
        ),
    )

    model = EOBWaveformModel(parameters_dict=_base_eob_params())
    model.compute_model(
        L=2,
        mass1=60.0,
        mass2=20.0,
        spin1x=0.3,
        spin1y=-0.1,
        spin1z=0.2,
        spin2z=-0.4,
        delta_t=1.0 / 8192.0,
    )

    assert captured["q"] == pytest.approx(3.0)
    np.testing.assert_allclose(captured["chi_1"], [0.3, -0.1, 0.2])
    np.testing.assert_allclose(captured["chi_2"], [0.0, 0.0, -0.4])
    assert model.delta_t_dimless == pytest.approx(
        (1.0 / 8192.0) / (80.0 * model.MTSUN_SI)
    )
    settings = captured["kwargs"]["settings"]
    assert settings["dt"] == pytest.approx(1.0 / 8192.0)
    assert settings["M"] == pytest.approx(80.0)
    assert settings["lmax_nyquist"] == 2


def test_corresponding_eob_hamiltonian_respects_explicit_start(monkeypatch):
    import waveformtools.models.eob as eob_module
    from waveformtools.models.waveform_models import WaveformModel

    captured = {}

    class FakeEOBWaveformModel:
        def __init__(self, parameters_dict):
            captured["parameters_dict"] = dict(parameters_dict)
            self.model = None

        def compute_model(self, L=29):
            self.model = SimpleNamespace(
                dynamics=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
            )

    class FakeSource:
        f_lower = 0.0
        omega0 = 0.0

        def get_td_waveform_modes(self, **_kwargs):
            raise AssertionError("explicit EOB start should not infer source start")

    monkeypatch.setattr(eob_module, "EOBWaveformModel", FakeEOBWaveformModel)

    params = _base_eob_params()
    params.update({"f_lower": 10.0, "omega0": np.pi * 10.0})

    e0 = WaveformModel.get_corresponding_eob_hamiltonian(FakeSource(), **params)

    assert e0 == pytest.approx(2.0 * (40.0 / 60.0) * (20.0 / 60.0))
    assert captured["parameters_dict"]["f_lower"] == pytest.approx(10.0)
    assert captured["parameters_dict"]["omega0"] == pytest.approx(np.pi * 10.0)
