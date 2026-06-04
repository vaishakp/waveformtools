"""Opt-in displacement-memory helpers for waveform mode arrays.

The current implementation exposes the public memory API and the first
spectral building block: projection of the nonlinear memory source
``|N|**2 / (16*pi)`` from an angular grid back to scalar modes.  The default
memory strain kernel treats those scalar modes as the right-hand side of
``bar_eth**2 h_mem`` and uses the corresponding spectral inverse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np

MemoryIntegrationConstant = Literal["zero_at_start"]
MemoryMethod = Literal["spectral"]


@dataclass(slots=True)
class DisplacementMemoryConfig:
    """Configuration for opt-in displacement-memory construction."""

    ell_min: int = 2
    ell_max: int | None = None
    memory_ell_max: int | None = None
    integration_constant: MemoryIntegrationConstant = "zero_at_start"
    method: MemoryMethod = "spectral"
    news_method: str = "spline"

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
    projection of ``|N|**2 / (16*pi)`` on the same time axis as ``news_modes``.
    This is the source data needed by the later spin -2 displacement-memory
    kernel; it is not itself the memory strain.
    """

    memory_config = DisplacementMemoryConfig.from_value(config, **overrides)
    _validate_news_modes(news_modes, memory_config)
    _validate_grid(news_modes)

    ell_max = _source_ell_max(news_modes, memory_config)
    angular_news = news_modes.evaluate_angular(ell_max=ell_max)
    source = np.abs(angular_news) ** 2 / (16.0 * np.pi)
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
            "source": "|news|^2/(16*pi)",
            "implementation": "angular_projection",
        },
    )
    return source_modes


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
                data=integrated_source / eigenvalue,
            )
    return memory_modes


def _bar_eth2_eigenvalue(ell: int) -> float:
    from qlmtools.spin_coefficient import analytic_spin_raise_basis_factor

    return float(
        analytic_spin_raise_basis_factor(
            ell=ell,
            emm=0,
            spin_weight=0,
            times=2,
        )
    )


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
