"""Structured perturbations of waveform mode coefficients.

This module provides the first low-level coordinate system for balance-law
repair: small fractional corrections applied directly to the SWSH mode
coefficients in a ``ModesArray``.  The higher-level repair workflow should
project these raw coordinates away from alignment, BMS, and protected
intrinsic-parameter tangent directions before optimizing them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

TimeBasisKind = Literal["cubic_b_spline"]
EndpointConstraint = Literal["zero", "free"]
CorrectionComponent = Literal["real", "imag"]


@dataclass(slots=True)
class FractionalModeCorrectionSpec:
    """Configuration for multiplicative modal corrections.

    The correction is applied as ``h_lm -> h_lm * (1 + delta_lm(t))``.
    ``delta_lm`` is expanded in a smooth time basis, with independent real
    and imaginary coefficient blocks by default.  The real part is a
    fractional amplitude correction; the imaginary part is a small phase-like
    correction.
    """

    modes: Sequence[tuple[int, int]]
    n_time_basis: int = 6
    time_basis: TimeBasisKind = "cubic_b_spline"
    endpoint_constraint: EndpointConstraint = "zero"
    max_abs_delta: float | Mapping[tuple[int, int], float] = 0.05
    include_real: bool = True
    include_imag: bool = True

    def __post_init__(self) -> None:
        self.modes = _normalize_modes(self.modes)
        self.n_time_basis = int(self.n_time_basis)
        if self.n_time_basis < 1:
            raise ValueError("n_time_basis must be at least 1.")
        if self.time_basis != "cubic_b_spline":
            raise ValueError("Only time_basis='cubic_b_spline' is supported.")
        if self.endpoint_constraint not in {"zero", "free"}:
            raise ValueError("endpoint_constraint must be 'zero' or 'free'.")
        if not self.include_real and not self.include_imag:
            raise ValueError("At least one correction component is required.")
        self.max_abs_delta = _normalize_delta_limits(
            self.max_abs_delta, self.modes
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        return asdict(self)

    @classmethod
    def from_value(
        cls,
        value: "FractionalModeCorrectionSpec | Mapping[str, Any]",
        **overrides: Any,
    ) -> "FractionalModeCorrectionSpec":
        """Construct a correction spec from a dataclass or mapping."""

        if isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "correction spec must be a FractionalModeCorrectionSpec "
                f"or mapping; got {type(value)!r}."
            )
        data.update(
            {key: val for key, val in overrides.items() if val is not None}
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class FractionalCorrectionCoefficient:
    """One flat optimization coefficient in a correction vector."""

    flat_index: int
    mode: tuple[int, int]
    basis_index: int
    component: CorrectionComponent
    scale: float
    lower_bound: float
    upper_bound: float


@dataclass(slots=True)
class FractionalModeCorrectionResult:
    """Corrected modes plus diagnostics for a fractional correction."""

    corrected_modes: Any
    delta_by_mode: dict[tuple[int, int], np.ndarray]
    diagnostics: dict[str, Any]


def coefficient_layout(
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
) -> tuple[FractionalCorrectionCoefficient, ...]:
    """Return the flat coefficient layout implied by ``spec``."""

    corr_spec = FractionalModeCorrectionSpec.from_value(spec)
    components = _components(corr_spec)
    entries: list[FractionalCorrectionCoefficient] = []
    flat_index = 0
    for mode in corr_spec.modes:
        scale = _component_scale_for_mode(corr_spec, mode)
        for basis_index in range(corr_spec.n_time_basis):
            for component in components:
                entries.append(
                    FractionalCorrectionCoefficient(
                        flat_index=flat_index,
                        mode=mode,
                        basis_index=basis_index,
                        component=component,
                        scale=scale,
                        lower_bound=-scale,
                        upper_bound=scale,
                    )
                )
                flat_index += 1
    return tuple(entries)


def n_fractional_correction_coefficients(
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
) -> int:
    """Return the length of the flat correction vector."""

    return len(coefficient_layout(spec))


def zero_fractional_correction_vector(
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
) -> np.ndarray:
    """Return a zero vector with the right flat coefficient length."""

    return np.zeros(n_fractional_correction_coefficients(spec), dtype=float)


def coefficient_index(
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
    mode: tuple[int, int],
    basis_index: int,
    component: CorrectionComponent,
) -> int:
    """Return the flat index for one mode/basis/component coefficient."""

    mode = (int(mode[0]), int(mode[1]))
    for entry in coefficient_layout(spec):
        if (
            entry.mode == mode
            and entry.basis_index == int(basis_index)
            and entry.component == component
        ):
            return entry.flat_index
    raise KeyError(
        f"No coefficient for mode={mode}, basis_index={basis_index}, "
        f"component={component!r}."
    )


def coefficient_bounds(
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds for the flat coefficient vector."""

    layout = coefficient_layout(spec)
    lower = np.array([entry.lower_bound for entry in layout], dtype=float)
    upper = np.array([entry.upper_bound for entry in layout], dtype=float)
    return lower, upper


def coefficient_scales(
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
) -> np.ndarray:
    """Return solver scaling factors for the flat coefficient vector."""

    return np.array([entry.scale for entry in coefficient_layout(spec)])


def fractional_correction_basis(
    time_axis: Sequence[float],
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
) -> np.ndarray:
    """Evaluate the time basis on ``time_axis``.

    The default endpoint constraint multiplies the cubic B-spline basis by a
    smooth taper that vanishes at the first and last samples.
    """

    corr_spec = FractionalModeCorrectionSpec.from_value(spec)
    x_axis = _normalized_time_axis(time_axis)
    basis = _cubic_b_spline_basis(x_axis, corr_spec.n_time_basis)
    if corr_spec.endpoint_constraint == "zero":
        basis = basis * (np.sin(np.pi * x_axis) ** 2)[:, np.newaxis]
    return basis


def fractional_delta_from_vector(
    coefficient_vector: Sequence[float],
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
    time_axis: Sequence[float],
) -> dict[tuple[int, int], np.ndarray]:
    """Convert a flat real vector into complex ``delta_lm(t)`` arrays."""

    corr_spec = FractionalModeCorrectionSpec.from_value(spec)
    vector = _coerce_coefficient_vector(coefficient_vector, corr_spec)
    basis = fractional_correction_basis(time_axis, corr_spec)
    mode_coefficients = {
        mode: np.zeros(corr_spec.n_time_basis, dtype=np.complex128)
        for mode in corr_spec.modes
    }
    layout = coefficient_layout(corr_spec)
    for entry, value in zip(layout, vector, strict=True):
        if entry.component == "real":
            mode_coefficients[entry.mode][entry.basis_index] += value
        else:
            mode_coefficients[entry.mode][entry.basis_index] += 1j * value

    return {
        mode: basis @ coefficients
        for mode, coefficients in mode_coefficients.items()
    }


def fractional_correction_diagnostics(
    coefficient_vector: Sequence[float],
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
    time_axis: Sequence[float],
) -> dict[str, Any]:
    """Return conditioning and size diagnostics for a correction vector."""

    corr_spec = FractionalModeCorrectionSpec.from_value(spec)
    vector = _coerce_coefficient_vector(coefficient_vector, corr_spec)
    basis = fractional_correction_basis(time_axis, corr_spec)
    delta_by_mode = fractional_delta_from_vector(vector, corr_spec, time_axis)
    max_abs_delta_by_mode = {
        mode: float(np.max(np.abs(delta))) if delta.size else 0.0
        for mode, delta in delta_by_mode.items()
    }
    rms_delta_by_mode = {}
    for mode, delta in delta_by_mode.items():
        if delta.size:
            rms_delta = float(np.sqrt(np.mean(np.abs(delta) ** 2)))
        else:
            rms_delta = 0.0
        rms_delta_by_mode[mode] = rms_delta
    limits = {
        mode: _max_abs_delta_for_mode(corr_spec, mode)
        for mode in corr_spec.modes
    }
    return {
        "n_coefficients": int(vector.size),
        "coefficient_norm": float(np.linalg.norm(vector)),
        "basis_rank": int(np.linalg.matrix_rank(basis)),
        "basis_condition_number": _basis_condition_number(basis),
        "max_abs_delta": max(max_abs_delta_by_mode.values(), default=0.0),
        "max_abs_delta_by_mode": max_abs_delta_by_mode,
        "rms_delta_by_mode": rms_delta_by_mode,
        "max_abs_delta_limit_by_mode": limits,
    }


def apply_fractional_mode_correction(
    modes_obj: Any,
    coefficient_vector: Sequence[float],
    spec: FractionalModeCorrectionSpec | Mapping[str, Any],
    *,
    enforce_bounds: bool = True,
) -> FractionalModeCorrectionResult:
    """Return a copy with ``h_lm -> h_lm * (1 + delta_lm)`` applied."""

    corr_spec = FractionalModeCorrectionSpec.from_value(spec)
    _validate_modes_object(modes_obj, corr_spec)
    delta_by_mode = fractional_delta_from_vector(
        coefficient_vector, corr_spec, modes_obj.time_axis
    )
    diagnostics = fractional_correction_diagnostics(
        coefficient_vector, corr_spec, modes_obj.time_axis
    )
    if enforce_bounds:
        _enforce_delta_limits(corr_spec, diagnostics)

    corrected = modes_obj.deepcopy()
    for ell, emm in corr_spec.modes:
        original_mode = np.asarray(
            modes_obj.mode(ell, emm), dtype=np.complex128
        )
        corrected.set_mode_data(
            ell=ell,
            emm=emm,
            data=original_mode * (1.0 + delta_by_mode[(ell, emm)]),
        )

    return FractionalModeCorrectionResult(
        corrected_modes=corrected,
        delta_by_mode=delta_by_mode,
        diagnostics=diagnostics,
    )


def _normalize_modes(
    modes: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for ell, emm in modes:
        mode = (int(ell), int(emm))
        if mode[0] < 0:
            raise ValueError("ell must be non-negative.")
        if abs(mode[1]) > mode[0]:
            raise ValueError(f"Invalid mode {mode}; require |m| <= ell.")
        if mode in seen:
            raise ValueError(f"Duplicate correction mode {mode}.")
        normalized.append(mode)
        seen.add(mode)
    if not normalized:
        raise ValueError("At least one correction mode is required.")
    return tuple(normalized)


def _normalize_delta_limits(
    max_abs_delta: float | Mapping[tuple[int, int], float],
    modes: Sequence[tuple[int, int]],
) -> float | dict[tuple[int, int], float]:
    if isinstance(max_abs_delta, Mapping):
        normalized: dict[tuple[int, int], float] = {}
        for mode in modes:
            if mode not in max_abs_delta:
                raise ValueError(f"Missing max_abs_delta for mode {mode}.")
            normalized[mode] = _validate_delta_limit(max_abs_delta[mode])
        return normalized
    return _validate_delta_limit(max_abs_delta)


def _validate_delta_limit(value: float) -> float:
    limit = float(value)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_abs_delta values must be positive and finite.")
    return limit


def _components(
    spec: FractionalModeCorrectionSpec,
) -> tuple[CorrectionComponent, ...]:
    components: list[CorrectionComponent] = []
    if spec.include_real:
        components.append("real")
    if spec.include_imag:
        components.append("imag")
    return tuple(components)


def _component_scale_for_mode(
    spec: FractionalModeCorrectionSpec,
    mode: tuple[int, int],
) -> float:
    n_components = len(_components(spec))
    return _max_abs_delta_for_mode(spec, mode) / np.sqrt(float(n_components))


def _max_abs_delta_for_mode(
    spec: FractionalModeCorrectionSpec,
    mode: tuple[int, int],
) -> float:
    if isinstance(spec.max_abs_delta, Mapping):
        return float(spec.max_abs_delta[mode])
    return float(spec.max_abs_delta)


def _normalized_time_axis(time_axis: Sequence[float]) -> np.ndarray:
    axis = np.asarray(time_axis, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("time_axis must be one-dimensional with length >= 2.")
    if not np.all(np.isfinite(axis)):
        raise ValueError("time_axis must contain only finite values.")
    span = axis[-1] - axis[0]
    if span <= 0.0:
        raise ValueError("time_axis must be strictly increasing overall.")
    x_axis = (axis - axis[0]) / span
    if np.any(np.diff(x_axis) < 0.0):
        raise ValueError("time_axis must be monotonically increasing.")
    return np.clip(x_axis, 0.0, 1.0)


def _cubic_b_spline_basis(x_axis: np.ndarray, n_basis: int) -> np.ndarray:
    try:
        from scipy.interpolate import BSpline
    except Exception as exc:  # pragma: no cover - scipy is normally present
        raise ImportError(
            "Fractional mode corrections require scipy.interpolate.BSpline."
        ) from exc

    degree = min(3, n_basis - 1)
    interior_count = n_basis - degree - 1
    if interior_count > 0:
        interior_knots = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    else:
        interior_knots = np.array([], dtype=float)
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            interior_knots,
            np.ones(degree + 1),
        )
    )
    basis = np.empty((x_axis.size, n_basis), dtype=float)
    for basis_index in range(n_basis):
        coefficients = np.zeros(n_basis, dtype=float)
        coefficients[basis_index] = 1.0
        basis[:, basis_index] = BSpline(
            knots,
            coefficients,
            degree,
            extrapolate=False,
        )(x_axis)
    basis = np.nan_to_num(basis, nan=0.0)
    basis[np.abs(basis) < 1e-15] = 0.0
    return basis


def _coerce_coefficient_vector(
    coefficient_vector: Sequence[float],
    spec: FractionalModeCorrectionSpec,
) -> np.ndarray:
    vector = np.asarray(coefficient_vector, dtype=float)
    expected = n_fractional_correction_coefficients(spec)
    if vector.ndim != 1 or vector.size != expected:
        raise ValueError(
            f"coefficient_vector must be one-dimensional with length "
            f"{expected}; got shape {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("coefficient_vector must contain only finite values.")
    return vector


def _basis_condition_number(basis: np.ndarray) -> float:
    if basis.size == 0:
        return np.inf
    singular_values = np.linalg.svd(basis, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] == 0.0:
        return np.inf
    return float(singular_values[0] / singular_values[-1])


def _validate_modes_object(
    modes_obj: Any,
    spec: FractionalModeCorrectionSpec,
) -> None:
    if getattr(modes_obj, "extra_mode_axes", False):
        raise NotImplementedError(
            "Fractional corrections do not yet support extra mode axes."
        )
    ell_max = getattr(modes_obj, "ell_max", None)
    if ell_max is None:
        raise ValueError("modes_obj must define ell_max.")
    for ell, _emm in spec.modes:
        if ell > int(ell_max):
            raise ValueError(
                f"Correction mode ell={ell} exceeds "
                f"modes_obj.ell_max={ell_max}."
            )
    _normalized_time_axis(modes_obj.time_axis)


def _enforce_delta_limits(
    spec: FractionalModeCorrectionSpec,
    diagnostics: Mapping[str, Any],
) -> None:
    max_abs_delta_by_mode = diagnostics["max_abs_delta_by_mode"]
    for mode in spec.modes:
        limit = _max_abs_delta_for_mode(spec, mode)
        observed = float(max_abs_delta_by_mode[mode])
        if observed > limit * (1.0 + 1e-12):
            raise ValueError(
                f"Fractional correction for mode {mode} exceeds "
                f"max_abs_delta: "
                f"{observed} > {limit}."
            )
