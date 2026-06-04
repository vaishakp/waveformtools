"""Configuration objects for waveform-mode comparisons and fitting factors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from waveformtools.comparison.alignment import AlignmentSpec
from waveformtools.comparison.rotation import RotationSpec

ObjectiveName = Literal["mode_match"]
FittingFactorOptimizer = Literal[
    "none", "scipy_minimize", "differential_evolution"
]
GeneratorCallStyle = Literal["kwargs", "dict"]


@dataclass(slots=True)
class ModeComparisonConfig:
    """Declarative settings for a fixed pair of mode arrays."""

    ell_min: int = 2
    ell_max: int | None = None
    modes: Sequence[tuple[int, int]] | None = None
    alignment: AlignmentSpec = field(default_factory=AlignmentSpec)
    rotation: RotationSpec = field(default_factory=RotationSpec)
    objective: ObjectiveName = "mode_match"

    def __post_init__(self) -> None:
        if self.ell_min < 0:
            raise ValueError("ell_min must be non-negative.")
        if self.ell_max is not None and self.ell_max < self.ell_min:
            raise ValueError(
                "ell_max must be greater than or equal to ell_min."
            )
        if self.objective != "mode_match":
            raise ValueError(
                "Only objective='mode_match' is currently supported."
            )
        if not isinstance(self.alignment, AlignmentSpec):
            self.alignment = AlignmentSpec.from_value(self.alignment)
        if not isinstance(self.rotation, RotationSpec):
            self.rotation = RotationSpec.from_value(self.rotation)
        if self.modes is not None:
            self.modes = tuple((int(ell), int(emm)) for ell, emm in self.modes)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        data = asdict(self)
        data["alignment"] = self.alignment.to_dict()
        data["rotation"] = self.rotation.to_dict()
        return data

    @classmethod
    def from_value(
        cls,
        value: "ModeComparisonConfig | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "ModeComparisonConfig":
        """Construct a config from a dataclass, mapping, or ``None``."""

        if value is None:
            data: dict[str, Any] = {}
        elif isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "comparison config must be a ModeComparisonConfig, a mapping, "
                f"or None; got {type(value)!r}."
            )

        data.update(
            {key: val for key, val in overrides.items() if val is not None}
        )
        if "alignment" in data:
            data["alignment"] = AlignmentSpec.from_value(data["alignment"])
        if "rotation" in data:
            data["rotation"] = RotationSpec.from_value(data["rotation"])
        return cls(**data)


# pylint: disable=too-many-instance-attributes
@dataclass(slots=True)
class FittingFactorConfig:
    """Declarative settings for fitting-factor evaluations.

    ``variable_parameters`` maps optimizer parameter names to ``(lower, upper)``
    bounds.  Parameters not present in ``fixed_parameters`` or a trial point are
    not passed to the generator, allowing model defaults to apply.
    """

    comparison: ModeComparisonConfig = field(
        default_factory=ModeComparisonConfig
    )
    variable_parameters: Mapping[str, tuple[float, float]] = field(
        default_factory=dict
    )
    fixed_parameters: Mapping[str, Any] = field(default_factory=dict)
    initial_parameters: Mapping[str, float] = field(default_factory=dict)
    use_generator_defaults: bool = True
    optimizer: FittingFactorOptimizer = "scipy_minimize"
    optimizer_options: Mapping[str, Any] = field(default_factory=dict)
    generator_call_style: GeneratorCallStyle = "kwargs"

    def __post_init__(self) -> None:
        if not isinstance(self.comparison, ModeComparisonConfig):
            self.comparison = ModeComparisonConfig.from_value(self.comparison)
        if self.optimizer not in {
            "none",
            "scipy_minimize",
            "differential_evolution",
        }:
            raise ValueError(
                "optimizer must be one of 'none', 'scipy_minimize', "
                "or 'differential_evolution'."
            )
        if self.generator_call_style not in {"kwargs", "dict"}:
            raise ValueError("generator_call_style must be 'kwargs' or 'dict'.")

        variable_parameters: dict[str, tuple[float, float]] = {}
        for name, bounds in self.variable_parameters.items():
            if len(bounds) != 2:
                raise ValueError(
                    f"Bounds for parameter {name!r} must have length 2."
                )
            lo, hi = float(bounds[0]), float(bounds[1])
            if lo >= hi:
                raise ValueError(
                    f"Bounds for parameter {name!r} must be increasing."
                )
            variable_parameters[str(name)] = (lo, hi)
        self.variable_parameters = variable_parameters
        self.fixed_parameters = dict(self.fixed_parameters)
        self.initial_parameters = {
            str(name): float(value)
            for name, value in self.initial_parameters.items()
        }
        self.optimizer_options = dict(self.optimizer_options)

        for name, value in self.initial_parameters.items():
            if name not in self.variable_parameters:
                raise ValueError(
                    f"Initial parameter {name!r} has no variable bounds."
                )
            lo, hi = self.variable_parameters[name]
            if not lo <= value <= hi:
                raise ValueError(
                    f"Initial parameter {name!r}={value} is outside its bounds."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        data = asdict(self)
        data["comparison"] = self.comparison.to_dict()
        return data

    @classmethod
    def from_value(
        cls,
        value: "FittingFactorConfig | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "FittingFactorConfig":
        """Construct a config from a dataclass, mapping, or ``None``."""

        if value is None:
            data: dict[str, Any] = {}
        elif isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "fitting-factor config must be a FittingFactorConfig, a mapping, "
                f"or None; got {type(value)!r}."
            )

        data.update(
            {key: val for key, val in overrides.items() if val is not None}
        )
        if "comparison" in data:
            data["comparison"] = ModeComparisonConfig.from_value(
                data["comparison"]
            )
        return cls(**data)
