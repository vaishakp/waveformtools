"""Mode-space rotation helpers for waveform comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import factorial
from typing import Any, Literal, Mapping, Sequence

import numpy as np

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
        if self.angle_bounds is not None:
            lo, hi = self.angle_bounds
            if not lo < hi:
                raise ValueError("angle_bounds must be an increasing (lower, upper) pair.")

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
    quat = _euler_zyz_quaternion(rotation.alpha, rotation.beta, rotation.gamma)
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
                rotated_data += _wigner_d(quat, ell, emp, emm) * source_data[emp]
            rotated.set_mode_data(ell=ell, emm=emm, data=rotated_data)


def _euler_zyz_quaternion(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return the repository convention quaternion for z-y-z Euler angles."""

    half_beta = 0.5 * beta
    half_alpha_gamma_sum = 0.5 * (alpha + gamma)
    half_gamma_alpha_diff = 0.5 * (gamma - alpha)
    return np.array(
        [
            np.cos(half_beta) * np.cos(half_alpha_gamma_sum),
            np.sin(half_beta) * np.sin(half_gamma_alpha_diff),
            np.sin(half_beta) * np.cos(half_gamma_alpha_diff),
            np.cos(half_beta) * np.sin(half_alpha_gamma_sum),
        ],
        dtype=float,
    )


def _wigner_d(quat: np.ndarray, ell: int, emp: int, emm: int) -> complex:
    """Return one Wigner-D matrix element in the local mode convention."""

    if abs(emp) > ell or abs(emm) > ell:
        raise ValueError("Bad Wigner-D indices.")

    q_array = np.asarray(quat, dtype=float)
    ra = np.array([q_array[0] + 1j * q_array[3]])
    rb = np.array([q_array[2] + 1j * q_array[1]])
    ra_small = np.abs(ra) < 1e-12
    rb_small = np.abs(rb) < 1e-12
    regular = np.where((~ra_small) & (~rb_small))[0]
    ra_zero = np.where(ra_small)[0]
    rb_zero = np.where((~ra_small) & rb_small)[0]
    result = np.zeros_like(ra, dtype=np.complex128)

    if emp == -emm:
        result[ra_zero] = rb[ra_zero] ** (2 * emm)

    if emp == emm:
        result[rb_zero] = ra[rb_zero] ** (2 * emm)

    if len(regular):
        ra_reg = ra[regular]
        rb_reg = rb[regular]
        ratio_abs_squared = (np.abs(rb_reg) / np.abs(ra_reg)) ** 2
        rho_min = max(0, emp - emm)
        rho_max = min(ell + emp, ell - emm)
        factor = (
            _wigner_coefficient(ell, emp, emm)
            * (np.abs(ra_reg) ** (2 * (ell - emm)))
            * (ra_reg ** (emm + emp))
            * (rb_reg ** (emm - emp))
        )
        series = 0.0
        for rho in range(rho_max, rho_min - 1, -1):
            series = (
                ((-1) ** rho)
                * _binomial(ell + emp, rho)
                * _binomial(ell - emp, ell - rho - emm)
                + series * ratio_abs_squared
            )
        result[regular] = factor * series * (ratio_abs_squared**rho_min)
    return complex(result[0])


def _wigner_coefficient(ell: int, emp: int, emm: int) -> float:
    return float(
        np.sqrt(
            factorial(ell + emm)
            * factorial(ell - emm)
            / (factorial(ell + emp) * factorial(ell - emp))
        )
    )


def _binomial(nn: int, kk: int) -> float:
    if kk < 0 or kk > nn:
        return 0.0
    return float(factorial(nn) / (factorial(kk) * factorial(nn - kk)))


def _available_modes_from_object(modes_obj: Any) -> list[tuple[int, int]]:
    mode_list = getattr(modes_obj, "modes_list", None)
    if mode_list:
        out: list[tuple[int, int]] = []
        for ell, emm_values in mode_list:
            out.extend((int(ell), int(emm)) for emm in emm_values)
        return out

    ell_max = int(getattr(modes_obj, "ell_max"))
    return [(ell, emm) for ell in range(2, ell_max + 1) for emm in range(-ell, ell + 1)]
