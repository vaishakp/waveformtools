"""Mode-space rotation helpers for waveform comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from waveformtools.rotation_math import euler_zyz_quaternion, wigner_d

RotationKind = Literal["none", "z_axis", "wigner"]

_ALLOWED_ROTATION_KINDS = {"none", "z_axis", "wigner"}


@dataclass(slots=True)
class RotationSpec:
    """Fixed mode-space rotation applied coherently to all selected modes."""

    kind: RotationKind = "none"
    angle: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    optimize_angle: bool = False
    angle_bounds: tuple[float, float] | None = None
    optimize_parameters: tuple[str, ...] = ()
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_ROTATION_KINDS:
            raise ValueError(
                f"Unsupported rotation kind={self.kind!r}; "
                f"choose one of {sorted(_ALLOWED_ROTATION_KINDS)}."
            )
        self.angle = float(self.angle)
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.gamma = float(self.gamma)
        if self.optimize_angle and self.kind != "z_axis":
            raise ValueError("optimize_angle=True currently requires kind='z_axis'.")
        self.optimize_parameters = tuple(str(name) for name in self.optimize_parameters)
        if self.optimize_parameters and self.kind != "wigner":
            raise ValueError("optimize_parameters currently requires kind='wigner'.")
        allowed_parameters = {"alpha", "beta", "gamma"}
        invalid_parameters = set(self.optimize_parameters) - allowed_parameters
        if invalid_parameters:
            raise ValueError(
                "Unsupported rotation optimize_parameters entries "
                f"{sorted(invalid_parameters)}; choose from {sorted(allowed_parameters)}."
            )
        if self.angle_bounds is not None:
            lo, hi = self.angle_bounds
            if not lo < hi:
                raise ValueError("angle_bounds must be an increasing (lower, upper) pair.")
            self.angle_bounds = (float(lo), float(hi))
        if self.parameter_bounds is not None:
            bounds: dict[str, tuple[float, float]] = {}
            for name, value in self.parameter_bounds.items():
                if name not in allowed_parameters:
                    raise ValueError(
                        f"Unsupported rotation parameter_bounds entry {name!r}; "
                        f"choose from {sorted(allowed_parameters)}."
                    )
                lo, hi = value
                if not lo < hi:
                    raise ValueError(
                        f"Bounds for rotation parameter {name!r} must be increasing."
                    )
                bounds[str(name)] = (float(lo), float(hi))
            self.parameter_bounds = bounds

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

    ``kind='z_axis'`` acts on each mode as
    ``h_lm -> exp(i m angle) h_lm``.  ``kind='wigner'`` applies the full
    same-ell Wigner-D mixing implied by the supplied Euler angles.  Both modes
    keep the operation entirely in ``ModesArray`` mode space and apply one
    coherent physical rotation to all selected modes.
    """

    rotation_spec = RotationSpec.from_value(rotation)
    if rotation_spec.kind == "none":
        return modes_obj.deepcopy() if hasattr(modes_obj, "deepcopy") else modes_obj

    if modes is None:
        modes = _available_modes_from_object(modes_obj)

    rotated = modes_obj.deepcopy() if hasattr(modes_obj, "deepcopy") else modes_obj
    if rotation_spec.kind == "wigner":
        _rotate_wigner_in_place(rotated, modes_obj, rotation_spec, modes)
        return rotated

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


def _rotate_wigner_in_place(
    rotated: Any,
    modes_obj: Any,
    rotation: RotationSpec,
    modes: Sequence[tuple[int, int]],
) -> None:
    quat = euler_zyz_quaternion(rotation.alpha, rotation.beta, rotation.gamma)
    modes_by_ell: dict[int, list[int]] = {}
    for ell, emm in modes:
        modes_by_ell.setdefault(int(ell), []).append(int(emm))

    for ell, emm_values in modes_by_ell.items():
        unique_emms = sorted(set(emm_values))
        source_data = {
            emm: np.asarray(modes_obj.mode(ell, emm), dtype=np.complex128)
            for emm in unique_emms
        }
        for emm in unique_emms:
            rotated_data = np.zeros_like(source_data[emm], dtype=np.complex128)
            for emp in unique_emms:
                rotated_data += wigner_d(quat, ell, emp, emm) * source_data[emp]
            rotated.set_mode_data(ell=ell, emm=emm, data=rotated_data)


def _available_modes_from_object(modes_obj: Any) -> list[tuple[int, int]]:
    mode_list = getattr(modes_obj, "modes_list", None)
    if mode_list:
        out: list[tuple[int, int]] = []
        for ell, emm_values in mode_list:
            out.extend((int(ell), int(emm)) for emm in emm_values)
        return out

    ell_max = int(getattr(modes_obj, "ell_max"))
    return [(ell, emm) for ell in range(2, ell_max + 1) for emm in range(-ell, ell + 1)]
