"""Opt-in displacement-memory helpers for waveform mode arrays.

The current implementation exposes the public memory API and the first
spectral building block: projection of the nonlinear memory source
``|N|**2`` from an angular grid back to scalar modes.  The source
normalization is configurable so convention checks can compare this default
memory-flux source with the legacy ``|N|**2 / (16*pi)`` diagnostic convention.
The default memory strain kernel treats those scalar modes as the right-hand
side of ``bar_eth**2 h_mem`` and uses the corresponding spectral inverse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np

MemoryIntegrationConstant = Literal["zero_at_start"]
MemoryMethod = Literal["spectral"]
MemorySourceNormalization = Literal["news_squared", "balance_law_16pi"]


@dataclass(slots=True)
class DisplacementMemoryConfig:
    """Configuration for opt-in displacement-memory construction."""

    ell_min: int = 2
    ell_max: int | None = None
    memory_ell_max: int | None = None
    integration_constant: MemoryIntegrationConstant = "zero_at_start"
    method: MemoryMethod = "spectral"
    news_method: str = "spline"
    source_normalization: MemorySourceNormalization = "news_squared"
    source_scale: float = 1.0

    def __post_init__(self) -> None:
        self.ell_min = int(self.ell_min)
        if self.ell_min < 2:
            raise ValueError(
                "ell_min must be at least 2 for displacement memory."
            )
        if self.ell_max is not None:
            self.ell_max = int(self.ell_max)
            if self.ell_max < self.ell_min:
                raise ValueError(
                    "ell_max must be greater than or equal to ell_min."
                )
        if self.memory_ell_max is not None:
            self.memory_ell_max = int(self.memory_ell_max)
            if self.memory_ell_max < 2:
                raise ValueError("memory_ell_max must be at least 2.")
        if self.integration_constant != "zero_at_start":
            raise ValueError(
                "Only integration_constant='zero_at_start' is supported."
            )
        if self.method != "spectral":
            raise ValueError("Only method='spectral' is supported.")
        if self.source_normalization not in {
            "news_squared",
            "balance_law_16pi",
        }:
            raise ValueError(
                "source_normalization must be 'news_squared' or "
                "'balance_law_16pi'."
            )
        self.source_scale = float(self.source_scale)
        if not np.isfinite(self.source_scale):
            raise ValueError("source_scale must be finite.")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        return asdict(self)

    @classmethod
    def from_value(
        cls,
        value: "DisplacementMemoryConfig | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "DisplacementMemoryConfig":
        """Construct a memory config from a dataclass, mapping, or ``None``."""

        if value is None:
            data: dict[str, Any] = {}
        elif isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "memory config must be a DisplacementMemoryConfig, a mapping, "
                f"or None; got {type(value)!r}."
            )
        data.update(
            {key: val for key, val in overrides.items() if val is not None}
        )
        return cls(**data)


def compute_displacement_memory_from_strain(
    strain_modes: Any,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return displacement-memory strain modes for an input strain waveform."""

    memory_config = DisplacementMemoryConfig.from_value(config, **overrides)
    _validate_strain_modes(strain_modes, memory_config)
    news_modes = strain_modes.get_news_from_strain(
        method=memory_config.news_method
    )
    return compute_displacement_memory_from_news(news_modes, memory_config)


def compute_displacement_memory_from_news(
    news_modes: Any,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return displacement-memory strain modes for input news modes."""

    memory_config = DisplacementMemoryConfig.from_value(config, **overrides)
    _validate_news_modes(news_modes, memory_config)
    memory_ell_max = _memory_ell_max(news_modes, memory_config)
    source_modes = compute_displacement_memory_source_from_news(
        news_modes,
        memory_config,
        memory_ell_max=memory_ell_max,
    )
    memory_modes = _source_modes_to_memory_strain(
        source_modes,
        memory_config,
        memory_ell_max=memory_ell_max,
    )
    _record_memory_metadata(memory_modes, memory_config, {})
    return memory_modes


def compute_displacement_memory_source_from_news(
    news_modes: Any,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return scalar modes of the nonlinear memory source.

    The returned object has ``spin_weight=0`` and contains the angular
    projection of the configured ``|N|**2`` source normalization on the same
    time axis as ``news_modes``.
    This is the source data needed by the later spin -2 displacement-memory
    kernel; it is not itself the memory strain.
    """

    memory_config = DisplacementMemoryConfig.from_value(config, **overrides)
    _validate_news_modes(news_modes, memory_config)
    _validate_grid(news_modes)

    ell_max = _source_ell_max(news_modes, memory_config)
    angular_news = news_modes.evaluate_angular(ell_max=ell_max)
    source_factor = _source_normalization_factor(memory_config)
    source = source_factor * np.abs(angular_news) ** 2
    source = _angular_data_time_first(source, news_modes)
    projection_ell_max = _source_projection_ell_max(
        news_modes,
        memory_config,
        ell_max,
    )
    from waveformtools.spherical_array import SphericalArray

    source_array = SphericalArray(
        label="memory_source_time_domain",
        time_axis=np.asarray(news_modes.time_axis, dtype=float),
        data=np.moveaxis(source, 0, -1),
        data_len=len(news_modes.time_axis),
        Grid=news_modes.Grid,
        spin_weight=0,
        ell_max=projection_ell_max,
    )
    source_modes = source_array.to_modes_array(
        Grid=news_modes.Grid,
        spin_weight=0,
        ell_max=projection_ell_max,
    )
    setattr(
        source_modes,
        "displacement_memory_source_metadata",
        {
            "included": True,
            "config": memory_config.to_dict(),
            "source": _source_description(memory_config),
            "source_factor": source_factor,
            "implementation": "angular_projection",
        },
    )
    return source_modes


def diagnose_displacement_memory_finite_time(
    strain_modes: Any,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None = None,
    window_fraction: float = 0.1,
    start_indices: list[int] | tuple[int, ...] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return diagnostics for finite-time displacement-memory sensitivity."""

    memory_config = DisplacementMemoryConfig.from_value(config, **overrides)
    _validate_strain_modes(strain_modes, memory_config)
    time_axis = np.asarray(strain_modes.time_axis, dtype=float)
    news_modes = strain_modes.get_news_from_strain(
        method=memory_config.news_method
    )
    source_modes = compute_displacement_memory_source_from_news(
        news_modes,
        memory_config,
        memory_ell_max=_memory_ell_max(news_modes, memory_config),
    )
    memory_modes = _source_modes_to_memory_strain(
        source_modes,
        memory_config,
        memory_ell_max=_memory_ell_max(news_modes, memory_config),
    )

    window_size = _diagnostic_window_size(len(time_axis), window_fraction)
    starts = _diagnostic_start_indices(len(time_axis), start_indices)
    full_endpoint_norm = _mode_data_vector_norm(
        memory_modes.modes_data[..., -1]
    )
    early_endpoint_norm = _memory_endpoint_norm_from_interval(
        source_modes,
        memory_config,
        0,
        window_size,
    )
    late_endpoint_norm = _memory_endpoint_norm_from_interval(
        source_modes,
        memory_config,
        len(time_axis) - window_size,
        len(time_axis),
    )

    return {
        "time_start": float(time_axis[0]),
        "time_end": float(time_axis[-1]),
        "duration": float(time_axis[-1] - time_axis[0]),
        "window_size": window_size,
        "window_fraction": float(window_fraction),
        "memory_endpoint_norm": full_endpoint_norm,
        "early_window_endpoint_norm": early_endpoint_norm,
        "late_window_endpoint_norm": late_endpoint_norm,
        "early_window_fraction_of_endpoint": _safe_ratio(
            early_endpoint_norm,
            full_endpoint_norm,
        ),
        "late_window_fraction_of_endpoint": _safe_ratio(
            late_endpoint_norm,
            full_endpoint_norm,
        ),
        "start_index_sensitivity": [
            {
                "start_index": int(start_index),
                "start_time": float(time_axis[start_index]),
                "endpoint_norm": _memory_endpoint_norm_from_interval(
                    source_modes,
                    memory_config,
                    start_index,
                    len(time_axis),
                ),
            }
            for start_index in starts
        ],
        "config": memory_config.to_dict(),
    }


def diagnose_omitted_inspiral(
    strain_modes: Any,
    news_modes: Any | None = None,
    mode: tuple[int, int] = (2, 2),
    news_method: str = "spline",
    early_window_fraction: float = 0.1,
    min_cycles: float = 3.0,
    high_initial_power_fraction: float = 0.05,
) -> dict[str, Any]:
    """Return start-of-waveform diagnostics for omitted inspiral."""

    config = DisplacementMemoryConfig(news_method=news_method)
    _validate_strain_modes(strain_modes, config)
    time_axis = np.asarray(strain_modes.time_axis, dtype=float)
    if news_modes is None:
        news_modes = strain_modes.get_news_from_strain(method=news_method)
    _validate_news_modes(news_modes, config)

    ell, emm = mode
    mode_data = np.asarray(strain_modes.mode(ell, emm), dtype=np.complex128)
    phase = np.unwrap(np.angle(mode_data))
    omega = np.gradient(phase, time_axis)
    window_size = _diagnostic_window_size(
        len(time_axis),
        early_window_fraction,
    )
    initial_omega = float(np.median(np.abs(omega[:window_size])))
    total_cycles = float(abs(phase[-1] - phase[0]) / (2.0 * np.pi))
    early_cycles = float(
        abs(phase[window_size - 1] - phase[0]) / (2.0 * np.pi)
    )
    initial_period = np.inf
    if initial_omega > 0.0:
        initial_period = float(2.0 * np.pi / initial_omega)

    power = np.asarray(strain_modes.get_power_from_news_modes(news_modes))
    peak_power = float(np.max(power))
    initial_power = float(power[0])
    early_power_median = float(np.median(power[:window_size]))
    total_energy = float(_trapezoid(power, time_axis))
    early_energy = float(
        _trapezoid(power[:window_size], time_axis[:window_size])
    )
    initial_power_fraction = _safe_ratio(initial_power, peak_power)
    early_power_fraction = _safe_ratio(early_power_median, peak_power)
    early_energy_fraction = _safe_ratio(early_energy, total_energy)

    starts_near_peak = initial_power_fraction > high_initial_power_fraction
    starts_short = total_cycles < min_cycles
    return {
        "mode": (int(ell), int(emm)),
        "time_start": float(time_axis[0]),
        "time_end": float(time_axis[-1]),
        "duration": float(time_axis[-1] - time_axis[0]),
        "initial_angular_frequency": initial_omega,
        "initial_period": initial_period,
        "total_cycles": total_cycles,
        "early_window_cycles": early_cycles,
        "early_window_size": window_size,
        "initial_power_fraction_of_peak": initial_power_fraction,
        "early_power_fraction_of_peak": early_power_fraction,
        "early_energy_fraction": early_energy_fraction,
        "starts_short": bool(starts_short),
        "starts_near_peak": bool(starts_near_peak),
        "omitted_inspiral_likely": bool(starts_short or starts_near_peak),
    }


def with_displacement_memory(
    strain_modes: Any,
    memory_modes: Any | None = None,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return a copy of ``strain_modes`` with displacement memory added."""

    if memory_modes is None:
        memory_modes = compute_displacement_memory_from_strain(
            strain_modes,
            config=config,
            **overrides,
        )
    _validate_compatible_memory_modes(strain_modes, memory_modes)
    out = strain_modes + memory_modes
    _record_memory_metadata(out, config, overrides)
    return out


def add_displacement_memory_in_place(
    strain_modes: Any,
    memory_modes: Any | None = None,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Add displacement memory to ``strain_modes`` in place and return it."""

    with_memory = with_displacement_memory(
        strain_modes,
        memory_modes=memory_modes,
        config=config,
        **overrides,
    )
    strain_modes._modes_data = with_memory.modes_data
    _record_memory_metadata(strain_modes, config, overrides)
    return strain_modes


def _validate_strain_modes(
    strain_modes: Any,
    config: DisplacementMemoryConfig,
) -> None:
    _validate_modes_common(strain_modes, config)
    if int(getattr(strain_modes, "spin_weight", -2)) != -2:
        raise ValueError(
            "Displacement memory requires spin_weight=-2 strain modes."
        )
    if not hasattr(strain_modes, "get_news_from_strain"):
        raise TypeError("strain_modes must provide get_news_from_strain().")


def _validate_news_modes(
    news_modes: Any,
    config: DisplacementMemoryConfig,
) -> None:
    _validate_modes_common(news_modes, config)
    if int(getattr(news_modes, "spin_weight", -2)) != -2:
        raise ValueError(
            "Displacement memory requires spin_weight=-2 news modes."
        )


def _validate_modes_common(
    modes_obj: Any,
    config: DisplacementMemoryConfig,
) -> None:
    time_axis = np.asarray(getattr(modes_obj, "time_axis", None), dtype=float)
    if time_axis.ndim != 1 or len(time_axis) < 2:
        raise ValueError(
            "Displacement memory requires a one-dimensional time axis."
        )
    if not np.all(np.isfinite(time_axis)):
        raise ValueError(
            "Displacement memory requires finite time-axis values."
        )
    if not np.all(np.diff(time_axis) > 0.0):
        raise ValueError(
            "Displacement memory requires a strictly increasing time axis."
        )
    ell_max = int(getattr(modes_obj, "ell_max"))
    if ell_max < config.ell_min:
        raise ValueError("Input modes do not contain the requested ell range.")


def _validate_grid(modes_obj: Any) -> None:
    grid = getattr(modes_obj, "Grid", None)
    if grid is None or not hasattr(grid, "meshgrid"):
        raise ValueError("Memory source projection requires a spherical Grid.")


def _source_ell_max(modes_obj: Any, config: DisplacementMemoryConfig) -> int:
    ell_max = int(getattr(modes_obj, "ell_max"))
    if config.ell_max is not None:
        ell_max = min(ell_max, config.ell_max)
    return ell_max


def _angular_data_time_first(data: np.ndarray, modes_obj: Any) -> np.ndarray:
    time_len = len(modes_obj.time_axis)
    theta, phi = modes_obj.Grid.meshgrid
    angular_shape = theta.shape
    if data.shape == (time_len, *angular_shape):
        return data
    if data.shape == (*angular_shape, time_len):
        return np.moveaxis(data, -1, 0)
    raise ValueError(
        "Angular data has shape "
        f"{data.shape}; expected {(time_len, *angular_shape)} "
        f"or {(*angular_shape, time_len)}."
    )


def _source_projection_ell_max(
    modes_obj: Any,
    config: DisplacementMemoryConfig,
    source_ell_max: int,
) -> int:
    if config.memory_ell_max is not None:
        return config.memory_ell_max
    grid_limit = int(getattr(modes_obj.Grid, "L", 2 * source_ell_max))
    return min(2 * source_ell_max, grid_limit)


def _source_normalization_factor(config: DisplacementMemoryConfig) -> float:
    if config.source_normalization == "news_squared":
        base_factor = 1.0
    elif config.source_normalization == "balance_law_16pi":
        base_factor = 1.0 / (16.0 * np.pi)
    else:  # pragma: no cover - guarded by config validation.
        raise ValueError(
            f"Unknown source_normalization={config.source_normalization!r}."
        )
    return config.source_scale * base_factor


def _source_description(config: DisplacementMemoryConfig) -> str:
    if config.source_normalization == "news_squared":
        base = "|news|^2"
    elif config.source_normalization == "balance_law_16pi":
        base = "|news|^2/(16*pi)"
    else:  # pragma: no cover - guarded by config validation.
        base = f"|news|^2[{config.source_normalization}]"
    if config.source_scale == 1.0:
        return base
    return f"{config.source_scale:g}*{base}"


def _diagnostic_window_size(data_len: int, fraction: float) -> int:
    fraction = float(fraction)
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError("diagnostic window fraction must be in (0, 1].")
    return max(2, min(int(data_len), int(np.ceil(fraction * data_len))))


def _diagnostic_start_indices(
    data_len: int,
    start_indices: list[int] | tuple[int, ...] | None,
) -> list[int]:
    if start_indices is None:
        proposed = [0, int(round(0.05 * data_len)), int(round(0.1 * data_len))]
    else:
        proposed = [int(index) for index in start_indices]
    return sorted({max(0, min(index, data_len - 2)) for index in proposed})


def _memory_endpoint_norm_from_interval(
    source_modes: Any,
    config: DisplacementMemoryConfig,
    start_index: int,
    stop_index: int,
) -> float:
    endpoint_values = []
    ell_min = max(2, config.ell_min)
    for ell, emm_list in source_modes.modes_list:
        if ell < ell_min:
            continue
        eigenvalue = _bar_eth2_eigenvalue(ell)
        for emm in emm_list:
            integrated_source = _trapezoid(
                np.asarray(source_modes.mode(ell, emm))[
                    start_index:stop_index
                ],
                np.asarray(source_modes.time_axis)[start_index:stop_index],
            )
            endpoint_values.append(
                np.conjugate(integrated_source) / eigenvalue
            )
    return _mode_data_vector_norm(np.asarray(endpoint_values))


def _mode_data_vector_norm(data: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.abs(data) ** 2)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return np.nan
    return float(numerator / denominator)


def _trapezoid(ydata: np.ndarray, xdata: np.ndarray) -> Any:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(ydata, x=xdata, axis=-1)
    return np.trapz(ydata, x=xdata, axis=-1)


def _memory_ell_max(
    modes_obj: Any,
    config: DisplacementMemoryConfig,
) -> int:
    ell_max = int(getattr(modes_obj, "ell_max"))
    if config.memory_ell_max is None:
        return ell_max
    return config.memory_ell_max


def _source_modes_to_memory_strain(
    source_modes: Any,
    config: DisplacementMemoryConfig,
    memory_ell_max: int,
) -> Any:
    from waveformtools.dataIO import construct_mode_list
    from waveformtools.modes_array import ModesArray

    time_axis = np.asarray(source_modes.time_axis, dtype=float)
    memory_modes = ModesArray(
        label="displacement_memory_time_domain",
        ell_max=memory_ell_max,
        time_axis=time_axis,
        spin_weight=-2,
        Grid=source_modes.Grid,
    )
    memory_modes.create_modes_array(
        ell_max=memory_ell_max,
        data_len=len(time_axis),
    )
    memory_modes.modes_list = construct_mode_list(
        memory_ell_max,
        spin_weight=-2,
    )

    ell_min = max(2, config.ell_min)
    for ell, emm_list in memory_modes.modes_list:
        if ell < ell_min:
            continue
        eigenvalue = _bar_eth2_eigenvalue(ell)
        for emm in emm_list:
            integrated_source = _cumulative_trapezoid_zero_at_start(
                time_axis,
                source_modes.mode(ell, emm),
            )
            memory_modes.set_mode_data(
                ell=ell,
                emm=emm,
                data=np.conjugate(integrated_source) / eigenvalue,
            )
    return memory_modes


def _bar_eth2_eigenvalue(ell: int) -> float:
    return float(np.sqrt((ell - 1) * ell * (ell + 1) * (ell + 2)))


def _cumulative_trapezoid_zero_at_start(
    xdata: np.ndarray,
    ydata: np.ndarray,
) -> np.ndarray:
    integral = np.zeros_like(ydata, dtype=np.result_type(ydata, complex))
    increments = 0.5 * (ydata[..., 1:] + ydata[..., :-1]) * np.diff(xdata)
    integral[..., 1:] = np.cumsum(increments, axis=-1)
    return integral


def _validate_compatible_memory_modes(
    strain_modes: Any,
    memory_modes: Any,
) -> None:
    if int(getattr(memory_modes, "spin_weight", -2)) != -2:
        raise ValueError("memory_modes must have spin_weight=-2.")
    if int(getattr(strain_modes, "ell_max")) != int(
        getattr(memory_modes, "ell_max")
    ):
        raise ValueError(
            "memory_modes must use the same ell_max as strain_modes."
        )
    if not np.allclose(strain_modes.time_axis, memory_modes.time_axis):
        raise ValueError(
            "memory_modes must use the same time axis as strain_modes."
        )


def _record_memory_metadata(
    modes_obj: Any,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
) -> None:
    memory_config = DisplacementMemoryConfig.from_value(
        config,
        **dict(overrides),
    )
    setattr(
        modes_obj,
        "displacement_memory_metadata",
        {
            "included": True,
            "config": memory_config.to_dict(),
            "implementation": "bar_eth2_inverse",
        },
    )
