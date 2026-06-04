"""Mode-space rotation helpers for waveform comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

RotationKind = Literal["none", "z_axis"]

_ALLOWED_ROTATION_KINDS = {"none", "z_axis"}


@dataclass(slots=True)
class RotationSpec:
    """Fixed mode-space rotation applied coherently to all selected modes."""

    kind: RotationKind = "none"
    angle: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_ROTATION_KINDS:
            raise ValueError(
                f"Unsupported rotation kind={self.kind!r}; "
                f"choose one of {sorted(_ALLOWED_ROTATION_KINDS)}."
            )
        self.angle = float(self.angle)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        return asdict(self)

    @classmethod
    def from_value(
        cls,
        value: "RotationSpec | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "RotationSpec":
        """Construct a rotation spec from a dataclass, mapping, or ``None``."""

        if value is None:
            data: dict[str, Any] = {}
        elif isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "rotation must be a RotationSpec, a mapping, or None; "
                f"got {type(value)!r}."
            )
        data.update({key: val for key, val in overrides.items() if val is not None})
        return cls(**data)


def rotate_modes(
    modes_obj: Any,
    rotation: RotationSpec | Mapping[str, Any] | None = None,
    *,
    modes: Sequence[tuple[int, int]] | None = None,
) -> Any:
    """Return a rotated copy of a modes object.

    The first supported nontrivial case is a z-axis rotation, which acts on each
    mode as ``h_lm -> exp(i m angle) h_lm``.  This keeps the operation entirely
    in ``ModesArray`` mode space and applies one coherent physical rotation to
    all selected modes.
    """

    rotation_spec = RotationSpec.from_value(rotation)
    if rotation_spec.kind == "none":
        return modes_obj.deepcopy() if hasattr(modes_obj, "deepcopy") else modes_obj

    if modes is None:
        modes = _available_modes_from_object(modes_obj)

    rotated = modes_obj.deepcopy() if hasattr(modes_obj, "deepcopy") else modes_obj
    for ell, emm in modes:
        data = np.asarray(modes_obj.mode(ell, emm), dtype=np.complex128)
        if rotation_spec.kind == "z_axis":
            rotated.set_mode_data(
                ell=ell,
                emm=emm,
                data=np.exp(1j * emm * rotation_spec.angle) * data,
            )
        else:  # pragma: no cover - guarded by RotationSpec validation
            raise ValueError(f"Unsupported rotation kind={rotation_spec.kind!r}.")
    return rotated


def _available_modes_from_object(modes_obj: Any) -> list[tuple[int, int]]:
    mode_list = getattr(modes_obj, "modes_list", None)
    if mode_list:
        out: list[tuple[int, int]] = []
        for ell, emm_values in mode_list:
            out.extend((int(ell), int(emm)) for emm in emm_values)
        return out

    ell_max = int(getattr(modes_obj, "ell_max"))
    return [(ell, emm) for ell in range(2, ell_max + 1) for emm in range(-ell, ell + 1)]
