"""Projected correction spaces for balance-law repair.

The repair workflow starts from fractional mode-correction coordinates and
projects them away from protected tangent directions.  Protected directions
include alignment/gauge directions such as time shifts and phase rotations,
plus user-supplied tangents for intrinsic parameters or BMS transformations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from waveformtools.waveform_corrections import (
    FractionalCorrectionCoefficient,
    FractionalModeCorrectionSpec,
    coefficient_layout,
    fractional_correction_basis,
)


@dataclass(slots=True)
class BalanceLawRepairSpec:
    """Configuration for projecting raw corrections into a repair space."""

    correction: FractionalModeCorrectionSpec | Mapping[str, Any]
    protect_time_shift: bool = True
    protect_global_phase: bool = True
    protect_orbital_phase: bool = True
    user_tangent_labels: Sequence[str] = field(default_factory=tuple)
    projection_rtol: float = 1e-10
    projection_atol: float = 1e-12

    def __post_init__(self) -> None:
        self.correction = FractionalModeCorrectionSpec.from_value(
            self.correction
        )
        self.user_tangent_labels = tuple(
            str(label) for label in self.user_tangent_labels
        )
        self.projection_rtol = float(self.projection_rtol)
        self.projection_atol = float(self.projection_atol)
        if self.projection_rtol < 0.0 or self.projection_atol < 0.0:
            raise ValueError("Projection tolerances must be non-negative.")

    @property
    def modes(self) -> tuple[tuple[int, int], ...]:
        """Return selected correction modes."""

        return self.correction.modes

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        data = asdict(self)
        data["correction"] = self.correction.to_dict()
        return data

    @classmethod
    def from_value(
        cls,
        value: "BalanceLawRepairSpec | Mapping[str, Any]",
        **overrides: Any,
    ) -> "BalanceLawRepairSpec":
        """Construct a repair spec from a dataclass or mapping."""

        if isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "repair spec must be a BalanceLawRepairSpec or mapping; "
                f"got {type(value)!r}."
            )
        data.update(
            {key: val for key, val in overrides.items() if val is not None}
        )
        return cls(**data)


@dataclass(slots=True)
class BalanceLawRepairProjection:
    """Projected basis and diagnostics for balance-law repair."""

    raw_basis: np.ndarray
    projected_basis: np.ndarray
    protected_basis: np.ndarray
    orthonormal_protected_basis: np.ndarray
    coefficient_layout: tuple[FractionalCorrectionCoefficient, ...]
    modes: tuple[tuple[int, int], ...]
    protected_labels: tuple[str, ...]
    time_weights: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class BalanceLawRepairResult:
    """Corrected modes produced from projected repair coefficients."""

    corrected_modes: Any
    correction_by_mode: dict[tuple[int, int], np.ndarray]
    diagnostics: dict[str, Any]


def build_balance_law_repair_projection(
    modes_obj: Any,
    spec: BalanceLawRepairSpec | Mapping[str, Any],
    *,
    user_tangents: Mapping[str, Any] | None = None,
) -> BalanceLawRepairProjection:
    """Build a projected correction basis for a modes object.

    The returned ``projected_basis`` maps the same flat coefficient vector as
    the raw fractional correction layer, but removes components parallel to the
    protected tangent space in the weighted mode-time inner product.
    """

    repair_spec = BalanceLawRepairSpec.from_value(spec)
    _validate_modes_object(modes_obj, repair_spec.modes)
    time_axis = np.asarray(modes_obj.time_axis, dtype=float)
    time_weights = _trapezoid_sqrt_weights(time_axis)
    raw_basis = _raw_repair_basis(modes_obj, repair_spec.correction)
    protected_basis, protected_labels = _protected_tangent_basis(
        modes_obj,
        repair_spec,
        user_tangents=user_tangents,
    )
    weighted_raw_basis = _apply_weights(
        raw_basis, time_weights, repair_spec.modes
    )
    weighted_protected_basis = _apply_weights(
        protected_basis, time_weights, repair_spec.modes
    )
    q_basis, protected_singular_values = _orthonormal_column_basis(
        weighted_protected_basis,
        rtol=repair_spec.projection_rtol,
        atol=repair_spec.projection_atol,
    )
    if q_basis.size:
        weighted_projected_basis = weighted_raw_basis - q_basis @ (
            q_basis.conj().T @ weighted_raw_basis
        )
    else:
        weighted_projected_basis = weighted_raw_basis.copy()
    projected_basis = _remove_weights(
        weighted_projected_basis, time_weights, repair_spec.modes
    )
    projected_singular_values = np.linalg.svd(
        weighted_projected_basis, compute_uv=False
    )
    raw_singular_values = np.linalg.svd(weighted_raw_basis, compute_uv=False)
    diagnostics = {
        "n_raw_directions": int(raw_basis.shape[1]),
        "n_protected_input_directions": int(protected_basis.shape[1]),
        "protected_rank": int(q_basis.shape[1]),
        "projected_rank": _matrix_rank_from_singular_values(
            projected_singular_values,
            rtol=repair_spec.projection_rtol,
            atol=repair_spec.projection_atol,
        ),
        "protected_labels": protected_labels,
        "raw_condition_number": _condition_number(raw_singular_values),
        "projected_condition_number": _condition_number(
            projected_singular_values
        ),
        "protected_condition_number": _condition_number(
            protected_singular_values
        ),
        "raw_singular_values": raw_singular_values.tolist(),
        "projected_singular_values": projected_singular_values.tolist(),
        "protected_singular_values": protected_singular_values.tolist(),
    }
    return BalanceLawRepairProjection(
        raw_basis=raw_basis,
        projected_basis=projected_basis,
        protected_basis=protected_basis,
        orthonormal_protected_basis=q_basis,
        coefficient_layout=coefficient_layout(repair_spec.correction),
        modes=repair_spec.modes,
        protected_labels=protected_labels,
        time_weights=time_weights,
        diagnostics=diagnostics,
    )


def apply_balance_law_repair_coefficients(
    modes_obj: Any,
    coefficient_vector: Sequence[float],
    projection: BalanceLawRepairProjection,
) -> BalanceLawRepairResult:
    """Apply projected repair coefficients to a copy of ``modes_obj``."""

    vector = np.asarray(coefficient_vector, dtype=float)
    if vector.ndim != 1 or vector.size != projection.projected_basis.shape[1]:
        raise ValueError(
            "coefficient_vector must be one-dimensional with length "
            f"{projection.projected_basis.shape[1]}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("coefficient_vector must contain only finite values.")

    flat_correction = projection.projected_basis @ vector
    correction_by_mode = _unflatten_modes(
        flat_correction,
        projection.modes,
        len(modes_obj.time_axis),
    )
    corrected = modes_obj.deepcopy()
    for ell, emm in projection.modes:
        repaired_mode = (
            modes_obj.mode(ell, emm) + correction_by_mode[(ell, emm)]
        )
        corrected.set_mode_data(ell=ell, emm=emm, data=repaired_mode)

    diagnostics = _repair_application_diagnostics(
        modes_obj,
        correction_by_mode,
        projection,
    )
    return BalanceLawRepairResult(
        corrected_modes=corrected,
        correction_by_mode=correction_by_mode,
        diagnostics=diagnostics,
    )


def _raw_repair_basis(
    modes_obj: Any,
    correction_spec: FractionalModeCorrectionSpec,
) -> np.ndarray:
    time_axis = np.asarray(modes_obj.time_axis, dtype=float)
    basis = fractional_correction_basis(time_axis, correction_spec)
    layout = coefficient_layout(correction_spec)
    raw_basis = np.zeros(
        (len(correction_spec.modes) * time_axis.size, len(layout)),
        dtype=np.complex128,
    )
    mode_offsets = _mode_offsets(correction_spec.modes, time_axis.size)
    for entry in layout:
        ell, emm = entry.mode
        mode_data = np.asarray(modes_obj.mode(ell, emm), dtype=np.complex128)
        multiplier = basis[:, entry.basis_index]
        if entry.component == "imag":
            multiplier = 1j * multiplier
        start, stop = mode_offsets[entry.mode]
        raw_basis[start:stop, entry.flat_index] = mode_data * multiplier
    return raw_basis


def _protected_tangent_basis(
    modes_obj: Any,
    spec: BalanceLawRepairSpec,
    *,
    user_tangents: Mapping[str, Any] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns: list[np.ndarray] = []
    labels: list[str] = []
    if spec.protect_time_shift:
        columns.append(_time_shift_tangent(modes_obj, spec.modes))
        labels.append("alignment:time_shift")
    if spec.protect_global_phase:
        columns.append(_global_phase_tangent(modes_obj, spec.modes))
        labels.append("alignment:global_phase")
    if spec.protect_orbital_phase:
        columns.append(_orbital_phase_tangent(modes_obj, spec.modes))
        labels.append("alignment:orbital_phase")

    user_tangents = {} if user_tangents is None else dict(user_tangents)
    requested_labels = spec.user_tangent_labels or tuple(user_tangents)
    for label in requested_labels:
        if label not in user_tangents:
            raise ValueError(f"Missing user tangent {label!r}.")
        columns.append(_flatten_tangent(user_tangents[label], spec.modes))
        labels.append(f"user:{label}")

    n_rows = len(spec.modes) * len(modes_obj.time_axis)
    if not columns:
        return np.zeros((n_rows, 0), dtype=np.complex128), tuple()
    return np.column_stack(columns), tuple(labels)


def _time_shift_tangent(
    modes_obj: Any,
    modes: Sequence[tuple[int, int]],
) -> np.ndarray:
    time_axis = np.asarray(modes_obj.time_axis, dtype=float)
    columns = []
    edge_order = 2 if time_axis.size > 2 else 1
    for ell, emm in modes:
        mode_data = np.asarray(modes_obj.mode(ell, emm), dtype=np.complex128)
        columns.append(
            -np.gradient(mode_data, time_axis, edge_order=edge_order)
        )
    return np.concatenate(columns)


def _global_phase_tangent(
    modes_obj: Any,
    modes: Sequence[tuple[int, int]],
) -> np.ndarray:
    return np.concatenate(
        [
            1j * np.asarray(modes_obj.mode(ell, emm), dtype=np.complex128)
            for ell, emm in modes
        ]
    )


def _orbital_phase_tangent(
    modes_obj: Any,
    modes: Sequence[tuple[int, int]],
) -> np.ndarray:
    columns = []
    for ell, emm in modes:
        mode_data = np.asarray(modes_obj.mode(ell, emm), dtype=np.complex128)
        columns.append(1j * emm * mode_data)
    return np.concatenate(columns)


def _flatten_tangent(
    tangent: Any,
    modes: Sequence[tuple[int, int]],
) -> np.ndarray:
    if hasattr(tangent, "mode"):
        return np.concatenate(
            [
                np.asarray(tangent.mode(ell, emm), dtype=np.complex128)
                for ell, emm in modes
            ]
        )
    if isinstance(tangent, Mapping):
        arrays = []
        for mode in modes:
            if mode not in tangent:
                raise ValueError(f"Missing tangent data for mode {mode}.")
            arrays.append(np.asarray(tangent[mode], dtype=np.complex128))
        return np.concatenate(arrays)
    raise TypeError(
        "User tangents must be ModesArray-like objects or mappings from "
        "(ell, m) to arrays."
    )


def _apply_weights(
    matrix: np.ndarray,
    time_weights: np.ndarray,
    modes: Sequence[tuple[int, int]],
) -> np.ndarray:
    if matrix.shape[1] == 0:
        return matrix.copy()
    repeated_weights = np.tile(time_weights, len(modes))
    return matrix * repeated_weights[:, np.newaxis]


def _remove_weights(
    matrix: np.ndarray,
    time_weights: np.ndarray,
    modes: Sequence[tuple[int, int]],
) -> np.ndarray:
    if matrix.shape[1] == 0:
        return matrix.copy()
    repeated_weights = np.tile(time_weights, len(modes))
    return matrix / repeated_weights[:, np.newaxis]


def _orthonormal_column_basis(
    matrix: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0 or matrix.shape[1] == 0:
        empty_basis = np.zeros((matrix.shape[0], 0), dtype=np.complex128)
        return empty_basis, np.array([])
    svd_result = np.linalg.svd(matrix, full_matrices=False)
    left, singular_values, _right_h = svd_result
    rank = _matrix_rank_from_singular_values(
        singular_values,
        rtol=rtol,
        atol=atol,
    )
    return left[:, :rank], singular_values


def _matrix_rank_from_singular_values(
    singular_values: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> int:
    if singular_values.size == 0:
        return 0
    threshold = max(float(atol), float(rtol) * float(singular_values[0]))
    return int(np.count_nonzero(singular_values > threshold))


def _condition_number(singular_values: np.ndarray) -> float:
    if singular_values.size == 0:
        return 0.0
    nonzero = singular_values[singular_values > 0.0]
    if nonzero.size == 0:
        return np.inf
    return float(nonzero[0] / nonzero[-1])


def _trapezoid_sqrt_weights(time_axis: np.ndarray) -> np.ndarray:
    if time_axis.ndim != 1 or time_axis.size < 2:
        raise ValueError("time_axis must be one-dimensional with length >= 2.")
    if not np.all(np.isfinite(time_axis)):
        raise ValueError("time_axis must contain only finite values.")
    if np.any(np.diff(time_axis) <= 0.0):
        raise ValueError("time_axis must be strictly increasing.")
    weights = np.empty_like(time_axis, dtype=float)
    weights[0] = 0.5 * (time_axis[1] - time_axis[0])
    weights[-1] = 0.5 * (time_axis[-1] - time_axis[-2])
    if time_axis.size > 2:
        weights[1:-1] = 0.5 * (time_axis[2:] - time_axis[:-2])
    if np.any(weights <= 0.0):
        raise ValueError("time integration weights must be positive.")
    return np.sqrt(weights)


def _mode_offsets(
    modes: Sequence[tuple[int, int]],
    data_len: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    return {
        mode: (index * data_len, (index + 1) * data_len)
        for index, mode in enumerate(modes)
    }


def _unflatten_modes(
    flat_values: np.ndarray,
    modes: Sequence[tuple[int, int]],
    data_len: int,
) -> dict[tuple[int, int], np.ndarray]:
    offsets = _mode_offsets(modes, data_len)
    return {
        mode: np.array(flat_values[start:stop], copy=True)
        for mode, (start, stop) in offsets.items()
    }


def _repair_application_diagnostics(
    modes_obj: Any,
    correction_by_mode: Mapping[tuple[int, int], np.ndarray],
    projection: BalanceLawRepairProjection,
) -> dict[str, Any]:
    max_fraction_by_mode = {}
    rms_fraction_by_mode = {}
    for ell, emm in projection.modes:
        original = np.asarray(modes_obj.mode(ell, emm), dtype=np.complex128)
        correction = np.asarray(correction_by_mode[(ell, emm)])
        denominator = np.maximum(np.abs(original), 1e-300)
        fraction = np.abs(correction) / denominator
        max_fraction_by_mode[(ell, emm)] = float(np.max(fraction))
        rms_fraction = np.sqrt(np.mean(fraction**2))
        rms_fraction_by_mode[(ell, emm)] = float(rms_fraction)
    return {
        "max_fractional_change": max(
            max_fraction_by_mode.values(), default=0.0
        ),
        "max_fractional_change_by_mode": max_fraction_by_mode,
        "rms_fractional_change_by_mode": rms_fraction_by_mode,
        "projection": dict(projection.diagnostics),
    }


def _validate_modes_object(
    modes_obj: Any,
    modes: Sequence[tuple[int, int]],
) -> None:
    if getattr(modes_obj, "extra_mode_axes", False):
        raise NotImplementedError(
            "Balance-law repair projection does not yet support extra "
            "mode axes."
        )
    ell_max = getattr(modes_obj, "ell_max", None)
    if ell_max is None:
        raise ValueError("modes_obj must define ell_max.")
    for ell, _emm in modes:
        if ell > int(ell_max):
            raise ValueError(
                f"Repair mode ell={ell} exceeds modes_obj.ell_max={ell_max}."
            )
    _trapezoid_sqrt_weights(np.asarray(modes_obj.time_axis, dtype=float))
