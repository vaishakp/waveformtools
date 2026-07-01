"""TEOBResumS waveform generation backend."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from waveformtools.models.waveform_models import WaveformModel


def is_teobresums_approximant(approximant: str) -> bool:
    """Return whether ``approximant`` should be generated through TEOBResumS."""

    token = str(approximant).replace("_", "-").upper()
    return token.startswith("TEOBRESUMS") or token in {"DALI", "GIOTTO"}


def mode_to_teob_key(ell: int, emm: int) -> str:
    """Return TEOBResumS' ``hlm`` key for a signed ``(ell, m)`` mode."""

    ell = int(ell)
    emm = int(emm)
    if emm == 0:
        return f"{ell}0"
    key = int(ell * (ell - 1) / 2 + abs(emm) - 2)
    return f"-{key}" if emm < 0 else str(key)


def mode_to_teob_index(ell: int, emm_abs: int) -> int:
    """Return TEOBResumS' positive ``use_mode_lm`` index for ``(ell, |m|)``."""

    return int(int(ell) * (int(ell) - 1) / 2 + int(emm_abs) - 2)


class TEOBResumSWaveformModel(WaveformModel):
    """Waveformtools model wrapper for TEOBResumS-Dali/Giotto modes."""

    def __init__(self, parameters_dict: Mapping[str, Any] | None = None, **kwargs: Any):
        params = dict(parameters_dict or {})
        params.update(kwargs)
        params.setdefault("approximant", "TEOBResumS-Dali")
        super().__init__(parameters_dict=params)
        self.model = SimpleNamespace()
        self.time_axis = None
        self.modes_dict = {}
        self.dynamics = {}
        self.last_teob_parameters = {}
        self.td_waveform_modes = None

    def capabilities(self):
        """Return the output capabilities advertised by this backend."""

        return {
            "td_modes": True,
            "fd_modes": False,
            "fd_modes_as_td": False,
            "td_polarizations": True,
            "fd_polarizations": False,
            "td_projection": True,
            "fd_projection": False,
            "nr_hdf5": False,
        }

    def compute_model(self, L=28, dimensionless=True, **parameters_dict):
        """Run TEOBResumS and store the generated modes."""

        self.update_parameters(parameters_dict)
        pars = self._teob_parameters(dimensionless=dimensionless)
        eob = self._eob_module()
        time_axis, hp, hc, hlm, dynamics = eob.EOBRunPy(pars)

        self.time_axis = np.asarray(time_axis, dtype=float)
        self.hp = np.asarray(hp)
        self.hc = np.asarray(hc)
        self.dynamics = dynamics if isinstance(dynamics, Mapping) else {}
        self.last_teob_parameters = pars
        self.model = SimpleNamespace(
            initial_hamiltonian=self._initial_hamiltonian_from_dynamics(),
            final_mass=None,
        )
        self.td_waveform_modes = self._modes_array_from_hlm(
            hlm,
            time_axis=self.time_axis,
            ell_max=int(self.parameters_dict.get("ell_max", 4)),
            L=L,
        )

    def get_td_waveform_modes(
        self,
        dimensionless=True,
        L=29,
        **parameters_dict,
    ):
        """Generate TEOBResumS time-domain modes as a ``ModesArray``."""

        self.compute_model(L=L, dimensionless=dimensionless, **parameters_dict)
        return self._standardize_generated_modes(
            self.td_waveform_modes,
            domain="td",
            dimensionless=dimensionless,
            generator="TEOBResumSWaveformModel.get_td_waveform_modes",
        )

    def get_td_modes(self, **parameters_dict):
        return self.get_td_waveform_modes(**parameters_dict)

    def get_corresponding_eob_hamiltonian(self, L=None, **parameters_dict):
        """Return TEOBResumS' initial EOB energy for the requested parameters."""

        self.update_parameters(parameters_dict)
        if self.dynamics:
            energy = self._initial_hamiltonian_from_dynamics()
            if np.isfinite(energy):
                return float(energy)
        pars = self._teob_parameters(dimensionless=False)
        pars["use_mode_lm"] = [mode_to_teob_index(2, 2)]
        eob = self._eob_module()
        _t, _hp, _hc, _hlm, dynamics = eob.EOBRunPy(pars)
        if not isinstance(dynamics, Mapping) or "E" not in dynamics:
            raise RuntimeError("TEOBResumS dynamics did not include an E energy array.")
        energy = np.asarray(dynamics["E"], dtype=float)
        if energy.size == 0:
            raise RuntimeError("TEOBResumS returned an empty E energy array.")
        return float(energy[0])

    def _teob_parameters(self, *, dimensionless: bool) -> dict[str, Any]:
        params = dict(self.parameters_dict)
        mass1 = float(params.get("mass1", params.get("mass_1", 1.0)))
        mass2 = float(params.get("mass2", params.get("mass_2", 1.0)))
        total_mass = mass1 + mass2
        q = max(mass1, mass2) / min(mass1, mass2)
        ell_max = int(params.get("ell_max", 4))

        use_geometric_units = "yes" if dimensionless else "no"
        f_lower = float(params.get("f_lower", params.get("f_min", 20.0)))
        initial_frequency = (
            f_lower * total_mass * self.MTSUN_SI if dimensionless else f_lower
        )

        delta_t = float(params.get("delta_t", 1.0 / 4096.0))
        if dimensionless:
            dt_over_m = delta_t / max(
                total_mass * self.MTSUN_SI,
                np.finfo(float).tiny,
            )
            srate_interp = 1.0 / dt_over_m if dt_over_m > 0 else 4096.0
        else:
            srate_interp = 1.0 / delta_t if delta_t > 0 else 4096.0

        teob: dict[str, Any] = {
            "M": 1.0 if dimensionless else total_mass,
            "q": q,
            "chi1x": float(params.get("spin1x", params.get("chi1x", 0.0))),
            "chi1y": float(params.get("spin1y", params.get("chi1y", 0.0))),
            "chi1z": float(
                params.get("spin1z", params.get("chi1z", params.get("chi1", 0.0)))
            ),
            "chi2x": float(params.get("spin2x", params.get("chi2x", 0.0))),
            "chi2y": float(params.get("spin2y", params.get("chi2y", 0.0))),
            "chi2z": float(
                params.get("spin2z", params.get("chi2z", params.get("chi2", 0.0)))
            ),
            "LambdaAl2": float(params.get("LambdaAl2", 0.0)),
            "LambdaBl2": float(params.get("LambdaBl2", 0.0)),
            "ecc": float(params.get("ecc", params.get("eccentricity", 0.0))),
            "anomaly": float(params.get("anomaly", math.pi)),
            "distance": float(params.get("distance", 1.0)),
            "inclination": float(params.get("inclination", 0.0)),
            "coalescence_angle": float(
                params.get("phi_ref", params.get("coalescence_angle", 0.0))
            ),
            "domain": 0,
            "use_geometric_units": use_geometric_units,
            "initial_frequency": initial_frequency,
            "interp_uniform_grid": "yes",
            "srate_interp": float(params.get("srate_interp", srate_interp)),
            "use_mode_lm": [
                mode_to_teob_index(ell, emm)
                for ell in range(2, ell_max + 1)
                for emm in range(1, ell + 1)
            ],
            "arg_out": "yes",
            "output_hpc": "no",
        }
        teob["chi1"] = teob["chi1z"]
        teob["chi2"] = teob["chi2z"]
        if "GIOTTO" in str(self.approximant).upper():
            teob["model"] = "Giotto"
        elif "teobresums_model" in params:
            teob["model"] = params["teobresums_model"]
        if any(abs(teob[key]) > 0.0 for key in ("chi1x", "chi1y", "chi2x", "chi2y")):
            teob.setdefault("spin_flx", params.get("spin_flx", "EOB"))
            teob.setdefault("spin_interp_domain", int(params.get("spin_interp_domain", 0)))
        return teob

    def _modes_array_from_hlm(self, hlm, *, time_axis, ell_max, L):
        from spectools.spherical.grids import GLGrid
        from waveformtools.modes_array import ModesArray

        grid = GLGrid(L=L)
        modes = ModesArray(
            ell_max=ell_max,
            time_axis=time_axis,
            spin_weight=-2,
            Grid=grid,
        )
        modes.create_modes_array(ell_max=ell_max, data_len=time_axis.size)
        scale = self._mode_scale()
        self.modes_dict = {}
        for ell in range(2, ell_max + 1):
            for emm in range(-ell, ell + 1):
                data = self._complex_mode_for_mode(hlm, ell, emm, time_axis.size) * scale
                modes.set_mode_data(ell=ell, emm=emm, data=data)
                self.modes_dict[(ell, emm)] = data
        return modes

    def _mode_scale(self) -> float:
        if not bool(self.parameters_dict.get("teobresums_scale_by_eta", True)):
            return 1.0
        mass1 = float(self.parameters_dict.get("mass1", self.parameters_dict.get("mass_1", 1.0)))
        mass2 = float(self.parameters_dict.get("mass2", self.parameters_dict.get("mass_2", 1.0)))
        total_mass = mass1 + mass2
        if total_mass <= 0:
            return 1.0
        return mass1 * mass2 / total_mass**2

    @staticmethod
    def _complex_mode_from_hlm(hlm: Mapping[str, Any], key: str, data_len: int):
        if key not in hlm:
            return np.zeros(data_len, dtype=np.complex128)
        amplitude, phase = hlm[key]
        return np.asarray(amplitude, dtype=float) * np.exp(1j * np.asarray(phase, dtype=float))

    @classmethod
    def _complex_mode_for_mode(cls, hlm, ell: int, emm: int, data_len: int):
        key = mode_to_teob_key(ell, emm)
        if key in hlm:
            return cls._complex_mode_from_hlm(hlm, key, data_len)
        if emm < 0:
            positive_key = mode_to_teob_key(ell, -emm)
            if positive_key in hlm:
                positive_mode = cls._complex_mode_from_hlm(hlm, positive_key, data_len)
                return ((-1) ** ell) * np.conjugate(positive_mode)
        return np.zeros(data_len, dtype=np.complex128)

    @staticmethod
    def _eob_module():
        try:
            import EOBRun_module
        except ImportError as exc:
            raise ImportError(
                "TEOBResumS support requires the Python extension EOBRun_module. "
                "Install the TEOBResumS Python package in the active environment."
            ) from exc
        return EOBRun_module

    def _initial_hamiltonian_from_dynamics(self) -> float:
        if not isinstance(self.dynamics, Mapping) or "E" not in self.dynamics:
            return math.nan
        energy = np.asarray(self.dynamics["E"], dtype=float)
        return float(energy[0]) if energy.size else math.nan
