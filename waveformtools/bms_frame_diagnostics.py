"""BMS-frame diagnostics from waveform mode data.

This module computes waveform-derived flux diagnostics that are useful before
fitting, balance-law repair, or cross-approximant comparison.  It intentionally
does not fix a BMS frame or compute absolute BMS charges without endpoint
assumptions; those choices belong in higher-level user workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from waveformtools.BMS import compute_impulse_from_force


@dataclass(slots=True)
class BMSFrameDiagnosticsConfig:
    """Configuration for waveform-derived BMS-frame diagnostics."""

    news_method: str = "spline"
    t_start: float | None = None
    t_end: float | None = None
    since_peak: bool = False
    inspiral_only: bool = False
    initial_mass: float | None = None
    final_mass: float | None = None
    reference_frame: str = "input"
    reference_cut: str = "input_time_axis"
    compute_energy: bool = True
    compute_linear_momentum: bool = True
    compute_angular_momentum: bool = True
    compute_memory_finite_time: bool = False
    compute_omitted_inspiral: bool = True
    omitted_inspiral_mode: tuple[int, int] = (2, 2)
    fail_on_optional_error: bool = False

    def __post_init__(self) -> None:
        self.news_method = str(self.news_method)
        self.reference_frame = str(self.reference_frame)
        self.reference_cut = str(self.reference_cut)
        self.omitted_inspiral_mode = (
            int(self.omitted_inspiral_mode[0]),
            int(self.omitted_inspiral_mode[1]),
        )
        if self.t_start is not None:
            self.t_start = float(self.t_start)
        if self.t_end is not None:
            self.t_end = float(self.t_end)
        if self.t_start is not None and self.t_end is not None:
            if self.t_start >= self.t_end:
                raise ValueError("t_start must be less than t_end.")
        if self.initial_mass is not None:
            self.initial_mass = _positive_finite(
                self.initial_mass, "initial_mass"
            )
        if self.final_mass is not None:
            self.final_mass = _positive_finite(self.final_mass, "final_mass")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        return asdict(self)

    @classmethod
    def from_value(
        cls,
        value: "BMSFrameDiagnosticsConfig | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "BMSFrameDiagnosticsConfig":
        """Construct diagnostics config from a dataclass, mapping, or none."""

        if value is None:
            data: dict[str, Any] = {}
        elif isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "BMS frame diagnostics config must be a "
                "BMSFrameDiagnosticsConfig, mapping, or None; "
                f"got {type(value)!r}."
            )
        data.update(
            {key: val for key, val in overrides.items() if val is not None}
        )
        return cls(**data)


@dataclass(slots=True)
class BMSFrameDiagnosticsResult:
    """Waveform-derived diagnostics relevant to BMS-frame comparisons."""

    energy_radiated: float | None
    radiated_linear_momentum: np.ndarray | None
    kick_velocity: np.ndarray | None
    angular_momentum_radiated: np.ndarray | None
    memory_finite_time: dict[str, Any] | None
    omitted_inspiral: dict[str, Any] | None
    assumptions: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""

        return {
            "energy_radiated": self.energy_radiated,
            "radiated_linear_momentum": _to_jsonable(
                self.radiated_linear_momentum
            ),
            "kick_velocity": _to_jsonable(self.kick_velocity),
            "angular_momentum_radiated": _to_jsonable(
                self.angular_momentum_radiated
            ),
            "memory_finite_time": _to_jsonable(self.memory_finite_time),
            "omitted_inspiral": _to_jsonable(self.omitted_inspiral),
            "assumptions": _to_jsonable(self.assumptions),
            "diagnostics": _to_jsonable(self.diagnostics),
        }


def compute_bms_frame_diagnostics(
    strain_modes: Any,
    config: BMSFrameDiagnosticsConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> BMSFrameDiagnosticsResult:
    """Compute frame diagnostics from a strain ``ModesArray``.

    The result contains radiated quantities and assumption metadata.  It does
    not claim to place the waveform in a superrest frame or determine absolute
    BMS charges from waveform data alone.
    """

    diagnostics_config = BMSFrameDiagnosticsConfig.from_value(
        config, **overrides
    )
    _validate_strain_modes(strain_modes)
    time_axis = np.asarray(strain_modes.time_axis, dtype=float)
    news_modes = strain_modes.get_news_from_strain(
        method=diagnostics_config.news_method
    )

    energy_radiated = None
    if diagnostics_config.compute_energy:
        energy_radiated = float(
            strain_modes.compute_energy_radiated(
                news_modes=news_modes,
                t_start=diagnostics_config.t_start,
                t_end=diagnostics_config.t_end,
                since_peak=diagnostics_config.since_peak,
                inspiral_only=diagnostics_config.inspiral_only,
            )
        )

    radiated_linear_momentum = None
    kick_velocity = None
    momentum_flux = None
    if diagnostics_config.compute_linear_momentum:
        momentum_flux = strain_modes.compute_momentum_flux(news_modes)
        radiated_linear_momentum = compute_impulse_from_force(
            news_modes.time_axis,
            momentum_flux[0],
            momentum_flux[1],
            momentum_flux[2],
        )
        if diagnostics_config.final_mass is not None:
            kick_velocity = (
                radiated_linear_momentum / diagnostics_config.final_mass
            )

    angular_momentum_radiated = None
    if diagnostics_config.compute_angular_momentum:
        angular_momentum_radiated = (
            strain_modes.compute_angular_momentum_radiated(
                news_modes=news_modes,
                t_start=diagnostics_config.t_start,
                t_end=diagnostics_config.t_end,
                since_peak=diagnostics_config.since_peak,
                inspiral_only=diagnostics_config.inspiral_only,
            )
        )

    memory_finite_time = None
    if diagnostics_config.compute_memory_finite_time:
        memory_finite_time = _optional_diagnostic(
            "memory_finite_time",
            diagnostics_config,
            strain_modes.diagnose_displacement_memory_finite_time,
            news_method=diagnostics_config.news_method,
        )

    omitted_inspiral = None
    if diagnostics_config.compute_omitted_inspiral:
        omitted_inspiral = _optional_diagnostic(
            "omitted_inspiral",
            diagnostics_config,
            strain_modes.diagnose_omitted_inspiral,
            news_modes=news_modes,
            mode=diagnostics_config.omitted_inspiral_mode,
            news_method=diagnostics_config.news_method,
        )

    assumptions = _assumption_metadata(
        diagnostics_config,
        energy_radiated=energy_radiated,
    )
    diagnostics = {
        "config": diagnostics_config.to_dict(),
        "news": _news_metadata(news_modes),
        "time": {
            "time_start": float(time_axis[0]),
            "time_end": float(time_axis[-1]),
            "duration": float(time_axis[-1] - time_axis[0]),
            "n_samples": int(time_axis.size),
        },
        "linear_momentum_flux_available": momentum_flux is not None,
        "superrest_frame_fixed": False,
        "absolute_bms_charges_computed": False,
        "notes": (
            "Radiated fluxes are waveform-derived. Absolute BMS charges and "
            "a superrest frame require additional endpoint/cut assumptions."
        ),
    }
    return BMSFrameDiagnosticsResult(
        energy_radiated=energy_radiated,
        radiated_linear_momentum=radiated_linear_momentum,
        kick_velocity=kick_velocity,
        angular_momentum_radiated=angular_momentum_radiated,
        memory_finite_time=memory_finite_time,
        omitted_inspiral=omitted_inspiral,
        assumptions=assumptions,
        diagnostics=diagnostics,
    )


def _assumption_metadata(
    config: BMSFrameDiagnosticsConfig,
    *,
    energy_radiated: float | None,
) -> dict[str, Any]:
    final_mass_from_initial = None
    if config.initial_mass is not None and energy_radiated is not None:
        final_mass_from_initial = config.initial_mass - energy_radiated
    return {
        "reference_frame": config.reference_frame,
        "reference_cut": config.reference_cut,
        "initial_mass": config.initial_mass,
        "final_mass": config.final_mass,
        "final_mass_from_initial_minus_radiation": final_mass_from_initial,
        "kick_requires_final_mass": config.final_mass is None,
        "absolute_bms_charges_require_endpoint_data": True,
        "superrest_frame_not_fixed": True,
        "moreschi_supermomentum_not_computed": True,
    }


def _news_metadata(news_modes: Any) -> dict[str, Any]:
    time_axis = np.asarray(news_modes.time_axis, dtype=float)
    return {
        "spin_weight": int(getattr(news_modes, "spin_weight")),
        "ell_max": int(getattr(news_modes, "ell_max")),
        "data_len": int(getattr(news_modes, "data_len")),
        "time_start": float(time_axis[0]),
        "time_end": float(time_axis[-1]),
    }


def _optional_diagnostic(
    name: str,
    config: BMSFrameDiagnosticsConfig,
    function: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        result = function(**kwargs)
    except Exception as exc:
        if config.fail_on_optional_error:
            raise
        return {
            "available": False,
            "name": name,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    if isinstance(result, dict):
        return {"available": True, **result}
    return {"available": True, "value": result}


def _validate_strain_modes(strain_modes: Any) -> None:
    raw_time_axis = getattr(strain_modes, "time_axis", None)
    time_axis = np.asarray(raw_time_axis, dtype=float)
    if time_axis.ndim != 1 or time_axis.size < 6:
        raise ValueError(
            "BMS frame diagnostics require at least 6 time samples."
        )
    if not np.all(np.isfinite(time_axis)):
        raise ValueError("BMS frame diagnostics require finite time values.")
    if np.any(np.diff(time_axis) <= 0.0):
        raise ValueError(
            "BMS frame diagnostics require increasing time values."
        )
    if int(getattr(strain_modes, "spin_weight", -2)) != -2:
        raise ValueError("BMS frame diagnostics expect spin_weight=-2 strain.")
    required_methods = (
        "get_news_from_strain",
        "compute_energy_radiated",
        "compute_momentum_flux",
        "compute_angular_momentum_radiated",
    )
    for method_name in required_methods:
        if not hasattr(strain_modes, method_name):
            raise TypeError(f"strain_modes must provide {method_name}().")


def _positive_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return numeric


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [
                {"real": float(item.real), "imag": float(item.imag)}
                for item in value.ravel()
            ]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
