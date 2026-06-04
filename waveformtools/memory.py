"""Opt-in displacement-memory helpers for waveform mode arrays.

This module intentionally starts with validation and API plumbing only.  The
spectral memory kernel will be added in a later batch once the angular
projection conventions are fixed and tested.
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
            raise ValueError("ell_min must be at least 2 for displacement memory.")
        if self.ell_max is not None:
            self.ell_max = int(self.ell_max)
            if self.ell_max < self.ell_min:
                raise ValueError("ell_max must be greater than or equal to ell_min.")
        if self.memory_ell_max is not None:
            self.memory_ell_max = int(self.memory_ell_max)
            if self.memory_ell_max < 2:
                raise ValueError("memory_ell_max must be at least 2.")
        if self.integration_constant != "zero_at_start":
            raise ValueError("Only integration_constant='zero_at_start' is supported.")
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
        data.update({key: val for key, val in overrides.items() if val is not None})
        return cls(**data)


def compute_displacement_memory_from_strain(
    strain_modes: Any,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return displacement-memory strain modes for an input strain waveform."""

    memory_config = DisplacementMemoryConfig.from_value(config, **overrides)
    _validate_strain_modes(strain_modes, memory_config)
    news_modes = strain_modes.get_news_from_strain(method=memory_config.news_method)
    return compute_displacement_memory_from_news(news_modes, memory_config)


def compute_displacement_memory_from_news(
    news_modes: Any,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return displacement-memory strain modes for input news modes."""

    memory_config = DisplacementMemoryConfig.from_value(config, **overrides)
    _validate_news_modes(news_modes, memory_config)
    raise NotImplementedError(
        "The displacement-memory spectral kernel is not implemented yet. "
        "This batch only exposes the opt-in API and validation surface."
    )


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


def _validate_strain_modes(strain_modes: Any, config: DisplacementMemoryConfig) -> None:
    _validate_modes_common(strain_modes, config)
    if int(getattr(strain_modes, "spin_weight", -2)) != -2:
        raise ValueError("Displacement memory requires spin_weight=-2 strain modes.")
    if not hasattr(strain_modes, "get_news_from_strain"):
        raise TypeError("strain_modes must provide get_news_from_strain().")


def _validate_news_modes(news_modes: Any, config: DisplacementMemoryConfig) -> None:
    _validate_modes_common(news_modes, config)
    if int(getattr(news_modes, "spin_weight", -2)) != -2:
        raise ValueError("Displacement memory requires spin_weight=-2 news modes.")


def _validate_modes_common(modes_obj: Any, config: DisplacementMemoryConfig) -> None:
    time_axis = np.asarray(getattr(modes_obj, "time_axis", None), dtype=float)
    if time_axis.ndim != 1 or len(time_axis) < 2:
        raise ValueError("Displacement memory requires a one-dimensional time axis.")
    if not np.all(np.isfinite(time_axis)):
        raise ValueError("Displacement memory requires finite time-axis values.")
    if not np.all(np.diff(time_axis) > 0.0):
        raise ValueError("Displacement memory requires a strictly increasing time axis.")
    ell_max = int(getattr(modes_obj, "ell_max"))
    if ell_max < config.ell_min:
        raise ValueError("Input modes do not contain the requested ell range.")


def _validate_compatible_memory_modes(strain_modes: Any, memory_modes: Any) -> None:
    if int(getattr(memory_modes, "spin_weight", -2)) != -2:
        raise ValueError("memory_modes must have spin_weight=-2.")
    if int(getattr(strain_modes, "ell_max")) != int(getattr(memory_modes, "ell_max")):
        raise ValueError("memory_modes must use the same ell_max as strain_modes.")
    if not np.allclose(strain_modes.time_axis, memory_modes.time_axis):
        raise ValueError("memory_modes must use the same time axis as strain_modes.")


def _record_memory_metadata(
    modes_obj: Any,
    config: DisplacementMemoryConfig | Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
) -> None:
    memory_config = DisplacementMemoryConfig.from_value(config, **dict(overrides))
    setattr(
        modes_obj,
        "displacement_memory_metadata",
        {
            "included": True,
            "config": memory_config.to_dict(),
            "implementation": "api_skeleton",
        },
    )
