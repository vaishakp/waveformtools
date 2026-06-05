"""Metadata helpers for waveform-mode comparison.

This module deliberately keeps metadata lightweight. Existing ``ModesArray``
objects in waveformtools were not originally constructed around a rich
provenance object, so the helpers here attach and retrieve optional comparison
metadata without changing the storage layout of the modes themselves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class WaveformMetadata:
    """Metadata describing how a mode object was generated.

    Parameters are intentionally optional. Fixed-frame comparisons should not
    require all metadata to be present, but fitting-factor searches and
    convention-sensitive comparisons should record as much of this object as
    possible.
    """

    approximant: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    mass_ratio: float | None = None
    total_mass: float | None = None
    mass1: float | None = None
    mass2: float | None = None
    spin1: tuple[float, float, float] | None = None
    spin2: tuple[float, float, float] | None = None
    f_ref: float | None = None
    f_lower: float | None = None

    spin_input_frame: str | None = None
    mode_output_frame: str | None = None
    mode_convention: str | None = None
    raw_mode_convention: str | None = None
    mode_convention_source: str | None = None
    canonicalization_applied: str | None = None
    mode_convention_history: list[dict[str, Any]] = field(default_factory=list)
    z_axis_definition: str | None = None
    x_axis_definition: str | None = None
    reference_time_or_frequency: str | None = None
    generator: str | None = None
    generator_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "WaveformMetadata":
        """Build metadata from an arbitrary mapping.

        Unknown keys are preserved inside ``parameters`` rather than discarded.
        This keeps the API tolerant of model-specific parameter dictionaries.
        """

        if mapping is None:
            return cls()

        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        known: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in dict(mapping).items():
            if key in fields:
                known[key] = value
            else:
                extra[key] = value

        metadata = cls(**known)
        if extra:
            metadata.parameters = {**extra, **metadata.parameters}
        return metadata


def get_comparison_metadata(modes: Any) -> WaveformMetadata:
    """Return comparison metadata attached to a modes object.

    If no explicit comparison metadata is attached, this function attempts a
    best-effort construction from common attributes such as ``approximant`` and
    ``parameters_dict``.
    """

    metadata = getattr(modes, "comparison_metadata", None)
    if isinstance(metadata, WaveformMetadata):
        return metadata
    if isinstance(metadata, Mapping):
        return WaveformMetadata.from_mapping(metadata)

    parameters: dict[str, Any] = {}
    for attr in ("parameters_dict", "generation_parameters", "parameters"):
        value = getattr(modes, attr, None)
        if isinstance(value, Mapping):
            parameters.update(dict(value))

    approximant = getattr(modes, "approximant", None)
    if approximant is None:
        approximant = parameters.get("approximant")

    mass1 = parameters.get("mass1")
    mass2 = parameters.get("mass2")
    total_mass = parameters.get("total_mass")
    mass_ratio = parameters.get("q")
    if mass1 is not None and mass2 is not None:
        total_mass = mass1 + mass2
        if mass2:
            mass_ratio = mass1 / mass2

    return WaveformMetadata(
        approximant=approximant,
        parameters=parameters,
        mass1=mass1,
        mass2=mass2,
        total_mass=total_mass,
        mass_ratio=mass_ratio,
        spin1=_spin_tuple(parameters, "spin1"),
        spin2=_spin_tuple(parameters, "spin2"),
        f_ref=parameters.get("f_ref"),
        f_lower=parameters.get("f_lower"),
    )


def attach_comparison_metadata(
    modes: Any,
    metadata: WaveformMetadata | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Attach comparison metadata to an existing modes object and return it."""

    if metadata is None:
        metadata_obj = WaveformMetadata.from_mapping(kwargs)
    elif isinstance(metadata, WaveformMetadata):
        metadata_obj = metadata
        if kwargs:
            data = metadata_obj.to_dict()
            data.update(kwargs)
            metadata_obj = WaveformMetadata.from_mapping(data)
    else:
        data = dict(metadata)
        data.update(kwargs)
        metadata_obj = WaveformMetadata.from_mapping(data)

    setattr(modes, "comparison_metadata", metadata_obj)
    return modes


def _spin_tuple(
    parameters: Mapping[str, Any],
    prefix: str,
) -> tuple[float, float, float] | None:
    keys = (f"{prefix}x", f"{prefix}y", f"{prefix}z")
    if all(key in parameters for key in keys):
        return tuple(float(parameters[key]) for key in keys)  # type: ignore[return-value]
    return None
