"""Core comparison routines for waveform mode arrays.

This module compares already-generated mode objects.  It now makes time- and
phase-alignment choices explicit through ``AlignmentSpec`` while still avoiding
full intrinsic-parameter fitting-factor searches.  Expensive waveform-family
optimization is layered on top of this core in later PRs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from waveformtools.comparison.alignment import (
    AlignmentSpec,
    PreparedModeData,
    prepare_mode_data,
)
from waveformtools.comparison.conventions import (
    canonicalize_modes_for_comparison,
)
from waveformtools.comparison.metadata import get_comparison_metadata
from waveformtools.comparison.results import ComparisonResult
from waveformtools.comparison.rotation import RotationSpec, rotate_modes

ModeSelector = Sequence[tuple[int, int]] | None


@dataclass(slots=True)
class AlignedModeData:
    """Replayable aligned arrays from a comparison result.

    Candidate arrays include the configured rotation, time-origin alignment,
    ``candidate_time_shift``, ``orbital_phase``, and ``global_phase`` using the
    same conventions as ``mode_match``.
    """

    time_axis: np.ndarray
    reference_modes: dict[tuple[int, int], np.ndarray]
    candidate_modes: dict[tuple[int, int], np.ndarray]
    selected_modes: list[tuple[int, int]]
    alignment: AlignmentSpec
    rotation: RotationSpec
    orbital_phase: float
    global_phase: float
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class _PhaseAlignmentEvaluation:
    inner: float
    raw_inner: complex
    phase_inner: complex
    orbital_phase: float
    global_phase: float
    diagnostics: dict[str, Any]


def available_modes(
    modes: Any, ell_min: int = 2, ell_max: int | None = None
) -> list[tuple[int, int]]:
    """Return available ``(ell, m)`` pairs for a modes object.

    The function prefers ``modes.modes_list`` when available because that is how
    ``ModesArray`` records loaded/generated modes. If it is missing, a dense
    list up to ``ell_max`` is constructed.
    """

    if ell_max is None:
        ell_max = int(getattr(modes, "ell_max"))

    mode_list = getattr(modes, "modes_list", None)
    if mode_list:
        out: list[tuple[int, int]] = []
        for ell, emm_values in mode_list:
            if ell_min <= int(ell) <= ell_max:
                out.extend((int(ell), int(emm)) for emm in emm_values)
        return out

    return [
        (ell, emm)
        for ell in range(ell_min, ell_max + 1)
        for emm in range(-ell, ell + 1)
    ]


def common_modes(
    modes_a: Any,
    modes_b: Any,
    ell_min: int = 2,
    ell_max: int | None = None,
    modes: ModeSelector = None,
) -> list[tuple[int, int]]:
    """Return modes available in both inputs, respecting an optional selector."""

    if modes is not None:
        return [(int(ell), int(emm)) for ell, emm in modes]

    a = set(available_modes(modes_a, ell_min=ell_min, ell_max=ell_max))
    b = set(available_modes(modes_b, ell_min=ell_min, ell_max=ell_max))
    return sorted(a.intersection(b))


def assert_common_time_axis(
    modes_a: Any, modes_b: Any, *, rtol: float = 1e-10, atol: float = 1e-12
) -> np.ndarray:
    """Validate and return a common time axis.

    This is kept as a compatibility helper for callers that want the original
    strict behavior.  New code should prefer ``AlignmentSpec`` via
    ``mode_match(..., alignment=...)``.
    """

    time_a = np.asarray(modes_a.time_axis, dtype=float)
    time_b = np.asarray(modes_b.time_axis, dtype=float)
    alignment = AlignmentSpec(
        time_domain_policy="error", time_axis_rtol=rtol, time_axis_atol=atol
    )
    prepared = prepare_mode_data(
        modes_a, modes_b, common_modes(modes_a, modes_b), alignment
    )
    if (
        time_a.shape != time_b.shape
    ):  # defensive; prepare_mode_data already checks
        raise ValueError(
            "Mode comparisons currently require a common time grid; "
            f"got {time_a.shape} and {time_b.shape}."
        )
    return prepared.time_axis


def _trapz_complex(values: np.ndarray, time_axis: np.ndarray) -> complex:
    if hasattr(np, "trapezoid"):
        return complex(np.trapezoid(values, time_axis))
    return complex(np.trapz(values, time_axis))


def _trapz_complex_axis(values: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    """Vectorized complex trapezoidal integral along the last (time) axis.

    ``values`` has shape ``(n_modes, n_time)``; returns a length ``n_modes``
    complex array holding each mode's integral. Equivalent to calling
    :func:`_trapz_complex` per row, but in a single numpy call.
    """
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return np.asarray(trapz(values, time_axis, axis=-1), dtype=complex)


def mode_inner_product(
    modes_a: Any,
    modes_b: Any,
    *,
    ell_min: int = 2,
    ell_max: int | None = None,
    modes: ModeSelector = None,
    time_axis: np.ndarray | None = None,
) -> complex:
    """Compute an unweighted time-domain inner product over common modes.

    The convention is

    ``<a,b> = integral dt sum_lm conj(a_lm(t)) b_lm(t)``.

    This compatibility function requires a common time grid.  Use ``mode_match``
    with an ``AlignmentSpec`` for peak alignment, cropping, or resampling.
    """

    if time_axis is None:
        time_axis = assert_common_time_axis(modes_a, modes_b)

    total = 0.0j
    for ell, emm in common_modes(
        modes_a, modes_b, ell_min=ell_min, ell_max=ell_max, modes=modes
    ):
        a_lm = np.asarray(modes_a.mode(ell, emm), dtype=np.complex128)
        b_lm = np.asarray(modes_b.mode(ell, emm), dtype=np.complex128)
        total += _trapz_complex(np.conjugate(a_lm) * b_lm, time_axis)
    return total


def mode_norm(
    modes_obj: Any,
    *,
    ell_min: int = 2,
    ell_max: int | None = None,
    modes: ModeSelector = None,
    time_axis: np.ndarray | None = None,
) -> float:
    """Return the positive mode-space norm of a modes object."""

    value = mode_inner_product(
        modes_obj,
        modes_obj,
        ell_min=ell_min,
        ell_max=ell_max,
        modes=modes,
        time_axis=time_axis,
    )
    return float(np.sqrt(max(value.real, 0.0)))


def mode_match(
    modes_a: Any,
    modes_b: Any,
    *,
    ell_min: int = 2,
    ell_max: int | None = None,
    modes: ModeSelector = None,
    phase_maximize: bool | None = None,
    alignment: AlignmentSpec | dict[str, Any] | None = None,
    rotation: RotationSpec | dict[str, Any] | None = None,
    time_alignment: str | None = None,
    time_domain_policy: str | None = None,
    phase_alignment: str | None = None,
    resample_method: str | None = None,
    candidate_time_shift: float | None = None,
    optimize_time_shift: bool | None = None,
    time_shift_bounds: tuple[float, float] | None = None,
    time_axis_rtol: float | None = None,
    time_axis_atol: float | None = None,
    orbital_phase_optimizer: str | None = None,
    phase_degeneracy_tol: float | None = None,
    time_shift_grid_samples: int | None = None,
    allow_phase_rotation_degeneracy: bool | None = None,
    canonicalize_mode_conventions: bool = True,
) -> ComparisonResult:
    """Compute a normalized mode match with explicit alignment choices.

    This is an unweighted mode-space time-domain match,
    not a detector PSD-weighted gravitational-wave overlap.  Returned
    alignment parameters are replayable on the candidate waveform with
    ``prepare_aligned_mode_data``.  Positive ``candidate_time_shift`` shifts
    the candidate time axis forward after reference-time alignment.
    ``orbital_phase`` applies ``exp(i m orbital_phase)`` to candidate modes and
    ``global_phase`` applies ``exp(i global_phase)`` to all candidate modes.

    Parameters
    ----------
    alignment:
        Optional ``AlignmentSpec`` or mapping.  Individual keyword arguments
        such as ``time_alignment`` and ``phase_alignment`` override it.
    phase_maximize:
        Backwards-compatible shorthand.  If supplied, it overrides
        ``phase_alignment`` with ``global_complex`` when true and ``none`` when
        false.
    canonicalize_mode_conventions:
        If true, transform registered approximant-specific raw mode
        conventions into the canonical comparison convention before mode
        selection, rotation, alignment, and inner products.
    """

    t0 = time.perf_counter()
    if phase_maximize is not None:
        phase_alignment = "global_complex" if phase_maximize else "none"

    alignment_spec = AlignmentSpec.from_value(
        alignment,
        time_alignment=time_alignment,
        time_domain_policy=time_domain_policy,
        phase_alignment=phase_alignment,
        resample_method=resample_method,
        candidate_time_shift=candidate_time_shift,
        optimize_time_shift=optimize_time_shift,
        time_shift_bounds=time_shift_bounds,
        time_axis_rtol=time_axis_rtol,
        time_axis_atol=time_axis_atol,
        orbital_phase_optimizer=orbital_phase_optimizer,
        phase_degeneracy_tol=phase_degeneracy_tol,
        time_shift_grid_samples=time_shift_grid_samples,
        allow_phase_rotation_degeneracy=allow_phase_rotation_degeneracy,
    )

    comparison_modes_a, target_mode_conventions = (
        canonicalize_modes_for_comparison(
            modes_a,
            enabled=canonicalize_mode_conventions,
        )
    )
    comparison_modes_b, candidate_mode_conventions = (
        canonicalize_modes_for_comparison(
            modes_b,
            enabled=canonicalize_mode_conventions,
        )
    )

    selected_modes = common_modes(
        comparison_modes_a,
        comparison_modes_b,
        ell_min=ell_min,
        ell_max=ell_max,
        modes=modes,
    )
    rotation_spec = RotationSpec.from_value(rotation)
    _validate_phase_rotation_degeneracy(alignment_spec, rotation_spec)

    rotation_optimization = None
    rotation_evaluations = 0
    if rotation_spec.optimize_angle:
        rotation_spec, rotation_optimization = _optimize_z_rotation(
            comparison_modes_a,
            comparison_modes_b,
            selected_modes,
            alignment_spec,
            rotation_spec,
        )
        rotation_evaluations = rotation_optimization["n_evaluations"]
    elif rotation_spec.optimize_parameters:
        rotation_spec, rotation_optimization = _optimize_wigner_rotation(
            comparison_modes_a,
            comparison_modes_b,
            selected_modes,
            alignment_spec,
            rotation_spec,
        )
        rotation_evaluations = rotation_optimization["n_evaluations"]

    comparison_modes_b = rotate_modes(
        comparison_modes_b, rotation_spec, modes=selected_modes
    )

    time_shift_optimization = None
    time_shift_evaluations = 0
    n_evaluations = 1
    if alignment_spec.optimize_time_shift:
        alignment_spec, time_shift_optimization = _optimize_time_shift(
            comparison_modes_a,
            comparison_modes_b,
            selected_modes,
            alignment_spec,
        )
        time_shift_evaluations = time_shift_optimization["n_evaluations"]

    evaluation = _evaluate_prepared_match(
        comparison_modes_a,
        comparison_modes_b,
        selected_modes,
        alignment_spec,
    )
    prepared = evaluation["prepared"]
    match_value = evaluation["match"]
    if np.isfinite(match_value):
        match_value = float(np.clip(match_value, -1.0, 1.0))
        mismatch_value = 1.0 - match_value
    else:
        mismatch_value = np.nan

    diagnostics = {
        "raw_inner_product_real": float(evaluation["raw_inner"].real),
        "raw_inner_product_imag": float(evaluation["raw_inner"].imag),
        "raw_inner_product_abs": float(abs(evaluation["raw_inner"])),
        "phase_aligned_inner": float(evaluation["inner"]),
        "orbital_phase": float(evaluation["orbital_phase"]),
        "global_phase": float(evaluation["global_phase"]),
        "orbital_phase_degenerate": bool(
            evaluation["phase_diagnostics"].get(
                "orbital_phase_degenerate", False
            )
        ),
        "phase_objective_relative_span": evaluation["phase_diagnostics"].get(
            "phase_objective_relative_span"
        ),
        "phase_optimization": evaluation["phase_diagnostics"],
        "norm_a": evaluation["norm_a"],
        "norm_b": evaluation["norm_b"],
        "n_modes": len(selected_modes),
        "modes": selected_modes,
        "alignment": alignment_spec.to_dict(),
        "rotation": rotation_spec.to_dict(),
        "time_axis": prepared.diagnostics.to_dict(),
        "mode_conventions": {
            "canonicalize_mode_conventions": bool(
                canonicalize_mode_conventions
            ),
            "target": target_mode_conventions,
            "candidate": candidate_mode_conventions,
        },
    }
    if rotation_optimization is not None:
        diagnostics["rotation_optimization"] = rotation_optimization
    if time_shift_optimization is not None:
        diagnostics["time_shift_optimization"] = time_shift_optimization

    if rotation_evaluations or time_shift_evaluations:
        n_evaluations = rotation_evaluations + time_shift_evaluations + 1

    return ComparisonResult(
        objective_name="mode_match",
        match=match_value,
        mismatch=mismatch_value,
        best_parameters={
            "phase_alignment": alignment_spec.phase_alignment,
            "orbital_phase": evaluation["orbital_phase"],
            "global_phase": evaluation["global_phase"],
            "time_alignment": alignment_spec.time_alignment,
            "time_domain_policy": alignment_spec.time_domain_policy,
            "candidate_time_shift": alignment_spec.candidate_time_shift,
            "rotation": rotation_spec.to_dict(),
        },
        optimizer=_optimizer_name(alignment_spec, rotation_spec),
        optimizer_status="ok",
        n_objective_evaluations=n_evaluations,
        elapsed_s=time.perf_counter() - t0,
        diagnostics=diagnostics,
        target_metadata=get_comparison_metadata(comparison_modes_a),
        candidate_metadata=get_comparison_metadata(comparison_modes_b),
    )


def mode_mismatch(*args: Any, **kwargs: Any) -> float:
    """Convenience wrapper returning only ``1 - match``."""

    return float(mode_match(*args, **kwargs).mismatch)


def prepare_aligned_mode_data(
    reference_modes: Any,
    candidate_modes: Any,
    result: ComparisonResult,
    modes: ModeSelector = None,
) -> AlignedModeData:
    """Replay a comparison result and return aligned mode arrays.

    The candidate replay convention is:

    ``t_candidate_aligned = t_candidate - reference_time_candidate + candidate_time_shift``

    and, after any configured rotation,

    ``h_lm_candidate_aligned = exp(i global_phase) * exp(i m orbital_phase) * h_lm_candidate``.

    This helper should be used by plotting and diagnostics instead of manually
    applying only ``orbital_phase``.
    """

    mode_conventions = result.diagnostics.get("mode_conventions", {})
    canonicalize_mode_conventions = bool(
        mode_conventions.get("canonicalize_mode_conventions", False)
    )
    reference_modes, _reference_mode_conventions = (
        canonicalize_modes_for_comparison(
            reference_modes,
            enabled=canonicalize_mode_conventions,
        )
    )
    candidate_modes, _candidate_mode_conventions = (
        canonicalize_modes_for_comparison(
            candidate_modes,
            enabled=canonicalize_mode_conventions,
        )
    )

    parameters = _result_alignment_parameters(result)
    alignment = AlignmentSpec.from_value(
        result.diagnostics.get("alignment", {}),
        candidate_time_shift=parameters.get("candidate_time_shift", 0.0),
    )
    rotation = RotationSpec.from_value(
        parameters.get("rotation", result.diagnostics.get("rotation"))
    )
    selected_modes = _result_selected_modes(
        reference_modes, candidate_modes, result, modes
    )
    rotated_candidate = rotate_modes(
        candidate_modes, rotation, modes=selected_modes
    )
    prepared = prepare_mode_data(
        reference_modes,
        rotated_candidate,
        selected_modes,
        alignment,
    )
    orbital_phase = float(parameters.get("orbital_phase", 0.0) or 0.0)
    global_phase = float(parameters.get("global_phase", 0.0) or 0.0)
    phase = np.exp(1j * global_phase)
    aligned_candidate = {
        (ell, emm): phase
        * np.exp(1j * emm * orbital_phase)
        * prepared.modes_b[(ell, emm)]
        for ell, emm in prepared.selected_modes
    }
    return AlignedModeData(
        time_axis=prepared.time_axis,
        reference_modes={
            mode: prepared.modes_a[mode] for mode in prepared.selected_modes
        },
        candidate_modes=aligned_candidate,
        selected_modes=list(prepared.selected_modes),
        alignment=alignment,
        rotation=rotation,
        orbital_phase=orbital_phase,
        global_phase=global_phase,
        diagnostics={
            "time_axis": prepared.diagnostics.to_dict(),
            "alignment": alignment.to_dict(),
            "rotation": rotation.to_dict(),
        },
    )


def aligned_mode_arrays(
    reference_modes: Any,
    candidate_modes: Any,
    result: ComparisonResult,
    modes: ModeSelector = None,
) -> AlignedModeData:
    """Alias for ``prepare_aligned_mode_data``."""

    return prepare_aligned_mode_data(
        reference_modes,
        candidate_modes,
        result,
        modes=modes,
    )


def _result_alignment_parameters(result: ComparisonResult) -> Mapping[str, Any]:
    alignment_parameters = result.best_parameters.get("alignment")
    if isinstance(alignment_parameters, Mapping):
        return alignment_parameters
    return result.best_parameters


def _result_selected_modes(
    reference_modes: Any,
    candidate_modes: Any,
    result: ComparisonResult,
    modes: ModeSelector,
) -> list[tuple[int, int]]:
    if modes is not None:
        return [(int(ell), int(emm)) for ell, emm in modes]
    diagnostic_modes = result.diagnostics.get("modes")
    if diagnostic_modes is not None:
        return [(int(ell), int(emm)) for ell, emm in diagnostic_modes]
    return common_modes(reference_modes, candidate_modes)


def residue_distance(
    modes_a: Any,
    modes_b: Any,
    residue_function: Callable[[Any], np.ndarray],
    *,
    norm_function: Callable[[np.ndarray], float] | None = None,
) -> ComparisonResult:
    """Compare residual fields computed from two modes objects.

    ``residue_function`` is supplied by downstream packages, e.g.
    ``waveform_balance_laws``.  This keeps waveformtools independent of any
    particular balance-law implementation.

    Alignment for residues will be added as a separate layer because the safer
    balance-law workflow is usually: align modes first, compute residues second,
    compare residues third.
    """

    t0 = time.perf_counter()
    residue_a = np.asarray(residue_function(modes_a))
    residue_b = np.asarray(residue_function(modes_b))
    if residue_a.shape != residue_b.shape:
        raise ValueError(
            "Residue arrays must have the same shape for fixed-frame distance; "
            f"got {residue_a.shape} and {residue_b.shape}."
        )

    diff = residue_a - residue_b
    if norm_function is None:
        distance = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
        norm_a = float(np.sqrt(np.mean(np.abs(residue_a) ** 2)))
        norm_b = float(np.sqrt(np.mean(np.abs(residue_b) ** 2)))
    else:
        distance = float(norm_function(diff))
        norm_a = float(norm_function(residue_a))
        norm_b = float(norm_function(residue_b))

    denom = max(norm_a, norm_b, 1e-300)
    normalized = distance / denom

    return ComparisonResult(
        objective_name="fixed_frame_residue_distance",
        distance=distance,
        normalized_distance=normalized,
        best_parameters={},
        optimizer_status="ok",
        elapsed_s=time.perf_counter() - t0,
        diagnostics={
            "norm_a": norm_a,
            "norm_b": norm_b,
            "shape": residue_a.shape,
        },
        target_metadata=get_comparison_metadata(modes_a),
        candidate_metadata=get_comparison_metadata(modes_b),
    )


def _prepared_mode_overlaps(
    arrays_a: dict[tuple[int, int], np.ndarray],
    arrays_b: dict[tuple[int, int], np.ndarray],
    selected_modes: Sequence[tuple[int, int]],
    time_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-mode time-domain overlaps ``J_lm = integral conj(a_lm) b_lm dt``.

    The orbital-phase factor ``exp(i m phi)`` is constant in time, so it
    factors out of the integral and these overlaps do not depend on the
    orbital phase. Computing them once -- vectorized over all modes in a single
    trapezoidal integration -- lets the orbital-phase search reuse them without
    re-integrating, and replaces the per-mode Python loop with one array op.

    Returns ``(m_values, overlaps)``: the ``m`` index of each selected mode and
    the corresponding complex overlap, both ordered like ``selected_modes``.
    """
    if not selected_modes:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=complex)
    stack_a = np.stack([arrays_a[mode] for mode in selected_modes])
    stack_b = np.stack([arrays_b[mode] for mode in selected_modes])
    overlaps = _trapz_complex_axis(np.conjugate(stack_a) * stack_b, time_axis)
    m_values = np.fromiter(
        (emm for _ell, emm in selected_modes),
        dtype=float,
        count=len(selected_modes),
    )
    return m_values, overlaps


def _inner_from_overlaps(
    m_values: np.ndarray,
    overlaps: np.ndarray,
    orbital_phase: float = 0.0,
) -> complex:
    """Combine precomputed per-mode overlaps at a given orbital phase.

    ``<a, b>(phi) = sum_lm exp(i m phi) J_lm`` -- a scalar reduction with no
    time integration, since ``overlaps`` already holds the integrals ``J_lm``.
    """
    if overlaps.size == 0:
        return 0.0j
    phase = np.exp(1j * m_values * float(orbital_phase))
    return complex(np.sum(phase * overlaps))


def _prepared_inner_product(
    arrays_a: dict[tuple[int, int], np.ndarray],
    arrays_b: dict[tuple[int, int], np.ndarray],
    selected_modes: Sequence[tuple[int, int]],
    time_axis: np.ndarray,
    *,
    orbital_phase: float = 0.0,
) -> complex:
    m_values, overlaps = _prepared_mode_overlaps(
        arrays_a, arrays_b, selected_modes, time_axis
    )
    return _inner_from_overlaps(m_values, overlaps, orbital_phase)


def _prepared_norm(
    arrays: dict[tuple[int, int], np.ndarray],
    selected_modes: Sequence[tuple[int, int]],
    time_axis: np.ndarray,
) -> float:
    if not selected_modes:
        return 0.0
    stack = np.stack([arrays[mode] for mode in selected_modes])
    total = complex(
        np.sum(_trapz_complex_axis(np.conjugate(stack) * stack, time_axis))
    )
    return float(np.sqrt(max(total.real, 0.0)))


def _phase_aligned_inner_product(
    prepared: PreparedModeData,
) -> _PhaseAlignmentEvaluation:
    m_values, overlaps = _prepared_mode_overlaps(
        prepared.modes_a,
        prepared.modes_b,
        prepared.selected_modes,
        prepared.time_axis,
    )
    raw = _inner_from_overlaps(m_values, overlaps, 0.0)
    phase_alignment = prepared.alignment.phase_alignment

    if phase_alignment == "none":
        return _PhaseAlignmentEvaluation(
            inner=float(raw.real),
            raw_inner=raw,
            phase_inner=raw,
            orbital_phase=0.0,
            global_phase=0.0,
            diagnostics={
                "phase_alignment": phase_alignment,
                "orbital_phase_optimizer": None,
                "orbital_phase_degenerate": False,
                "phase_objective_relative_span": None,
            },
        )
    if phase_alignment == "global_complex":
        global_phase = _global_phase_from_inner(raw)
        return _PhaseAlignmentEvaluation(
            inner=float(abs(raw)),
            raw_inner=raw,
            phase_inner=raw,
            orbital_phase=0.0,
            global_phase=global_phase,
            diagnostics={
                "phase_alignment": phase_alignment,
                "orbital_phase_optimizer": "analytic_global_phase",
                "orbital_phase_degenerate": False,
                "phase_objective_relative_span": None,
                "global_phase": global_phase,
                "objective_value": float(abs(raw)),
            },
        )

    phase_result = _maximize_orbital_phase(prepared, raw, m_values, overlaps)
    if phase_alignment == "orbital_phase":
        return _PhaseAlignmentEvaluation(
            inner=float(phase_result["inner"].real),
            raw_inner=raw,
            phase_inner=complex(phase_result["inner"]),
            orbital_phase=float(phase_result["orbital_phase"]),
            global_phase=0.0,
            diagnostics=phase_result["diagnostics"],
        )
    if phase_alignment == "orbital_phase_and_global":
        best_inner = complex(phase_result["inner"])
        global_phase = _global_phase_from_inner(best_inner)
        diagnostics = dict(phase_result["diagnostics"])
        diagnostics["global_phase"] = global_phase
        return _PhaseAlignmentEvaluation(
            inner=float(abs(best_inner)),
            raw_inner=raw,
            phase_inner=best_inner,
            orbital_phase=float(phase_result["orbital_phase"]),
            global_phase=global_phase,
            diagnostics=diagnostics,
        )
    raise ValueError(f"Unsupported phase_alignment={phase_alignment!r}")


def _evaluate_prepared_match(
    modes_a: Any,
    modes_b: Any,
    selected_modes: Sequence[tuple[int, int]],
    alignment: AlignmentSpec,
) -> dict[str, Any]:
    prepared = prepare_mode_data(modes_a, modes_b, selected_modes, alignment)
    phase_evaluation = _phase_aligned_inner_product(prepared)
    norm_a = _prepared_norm(
        prepared.modes_a, prepared.selected_modes, prepared.time_axis
    )
    norm_b = _prepared_norm(
        prepared.modes_b, prepared.selected_modes, prepared.time_axis
    )
    if norm_a == 0.0 or norm_b == 0.0:
        match_value = np.nan
    else:
        match_value = float(phase_evaluation.inner / (norm_a * norm_b))
    return {
        "prepared": prepared,
        "inner": phase_evaluation.inner,
        "orbital_phase": phase_evaluation.orbital_phase,
        "global_phase": phase_evaluation.global_phase,
        "phase_inner": phase_evaluation.phase_inner,
        "raw_inner": phase_evaluation.raw_inner,
        "phase_diagnostics": phase_evaluation.diagnostics,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "match": match_value,
    }


def _optimize_time_shift(
    modes_a: Any,
    modes_b: Any,
    selected_modes: Sequence[tuple[int, int]],
    alignment: AlignmentSpec,
) -> tuple[AlignmentSpec, dict[str, Any]]:
    method = getattr(alignment, "time_shift_method", "grid")
    if method == "roll":
        return _optimize_time_shift_roll(modes_a, modes_b, selected_modes, alignment)
    if method == "ifft":
        return _optimize_time_shift_ifft(modes_a, modes_b, selected_modes, alignment)

    base_alignment = replace(
        alignment, time_domain_policy="resample_to_reference"
    )
    bounds = _time_shift_bounds(modes_a, modes_b, base_alignment)
    n_refinement_evaluations = 0

    def objective(candidate_shift: float) -> float:
        trial_alignment = replace(
            base_alignment, candidate_time_shift=float(candidate_shift)
        )
        try:
            evaluation = _evaluate_prepared_match(
                modes_a, modes_b, selected_modes, trial_alignment
            )
        except ValueError:
            return np.inf
        match_value = evaluation["match"]
        if not np.isfinite(match_value):
            return np.inf
        return 1.0 - float(np.clip(match_value, -1.0, 1.0))

    n_grid = _time_shift_grid_sample_count(
        modes_a, modes_b, bounds, base_alignment
    )
    grid_shifts = np.linspace(bounds[0], bounds[1], n_grid)
    grid_objectives = np.array(
        [objective(float(shift)) for shift in grid_shifts], dtype=float
    )
    finite = np.isfinite(grid_objectives)
    if not np.any(finite):
        raise ValueError(
            "Time-shift optimization found no finite objective values."
        )
    finite_indices = np.flatnonzero(finite)
    grid_best_index = int(
        finite_indices[int(np.argmin(grid_objectives[finite]))]
    )
    grid_best_shift = float(grid_shifts[grid_best_index])
    grid_best_objective = float(grid_objectives[grid_best_index])

    left_index = max(grid_best_index - 1, 0)
    right_index = min(grid_best_index + 1, len(grid_shifts) - 1)
    refine_bounds = (
        float(grid_shifts[left_index]),
        float(grid_shifts[right_index]),
    )
    if not refine_bounds[0] < refine_bounds[1]:
        refine_bounds = tuple(float(item) for item in bounds)

    try:
        from scipy.optimize import minimize_scalar
    except Exception as exc:  # pragma: no cover - scipy is normally available
        raise ImportError(
            "optimize_time_shift=True requires scipy.optimize.minimize_scalar"
        ) from exc

    def counted_objective(candidate_shift: float) -> float:
        nonlocal n_refinement_evaluations
        n_refinement_evaluations += 1
        return objective(candidate_shift)

    result = minimize_scalar(
        counted_objective, bounds=refine_bounds, method="bounded"
    )
    refined_objective = float(result.fun)
    if (
        result.success
        and np.isfinite(refined_objective)
        and refined_objective <= grid_best_objective
    ):
        best_shift = float(result.x)
        best_objective = refined_objective
    else:
        best_shift = grid_best_shift
        best_objective = grid_best_objective
    best_alignment = replace(base_alignment, candidate_time_shift=best_shift)
    diagnostics = {
        "parameter": "candidate_time_shift",
        "bounds": tuple(float(item) for item in bounds),
        "refinement_bounds": refine_bounds,
        "initial_shift": float(alignment.candidate_time_shift),
        "coarse_grid_best_shift": grid_best_shift,
        "coarse_grid_best_mismatch": grid_best_objective,
        "refined_best_shift": float(result.x),
        "refined_best_mismatch": refined_objective,
        "best_shift": best_shift,
        "best_mismatch": best_objective,
        "success": bool(result.success),
        "message": str(result.message),
        "n_grid_evaluations": int(n_grid),
        "n_refinement_evaluations": int(n_refinement_evaluations),
        "n_evaluations": int(n_grid + n_refinement_evaluations),
    }
    return best_alignment, diagnostics


def _time_shift_grid_search_and_refine(
    objective: "Callable[[float], float]",
    bounds: tuple[float, float],
    n_grid: int,
    base_alignment: Any,
) -> tuple[Any, dict]:
    """Shared grid-search + scalar-refine logic for all time-shift methods.

    ``objective(shift) -> 1 - match`` is called by the caller-supplied
    function; this helper handles the coarse grid, ``minimize_scalar``
    refinement, and building the diagnostics dict in the standard format.
    """
    from dataclasses import replace as _replace

    try:
        from scipy.optimize import minimize_scalar
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "optimize_time_shift=True requires scipy.optimize.minimize_scalar"
        ) from exc

    grid_shifts = np.linspace(bounds[0], bounds[1], n_grid)
    grid_objectives = np.array(
        [objective(float(s)) for s in grid_shifts], dtype=float
    )
    finite = np.isfinite(grid_objectives)
    if not np.any(finite):
        raise ValueError(
            "Time-shift optimization found no finite objective values."
        )
    finite_indices = np.flatnonzero(finite)
    grid_best_index = int(
        finite_indices[int(np.argmin(grid_objectives[finite]))]
    )
    grid_best_shift = float(grid_shifts[grid_best_index])
    grid_best_objective = float(grid_objectives[grid_best_index])

    left_index = max(grid_best_index - 1, 0)
    right_index = min(grid_best_index + 1, len(grid_shifts) - 1)
    refine_bounds = (
        float(grid_shifts[left_index]),
        float(grid_shifts[right_index]),
    )
    if not refine_bounds[0] < refine_bounds[1]:
        refine_bounds = tuple(float(x) for x in bounds)

    n_refinement_evaluations = 0

    def counted_objective(shift: float) -> float:
        nonlocal n_refinement_evaluations
        n_refinement_evaluations += 1
        return objective(shift)

    result = minimize_scalar(
        counted_objective, bounds=refine_bounds, method="bounded"
    )
    refined_objective = float(result.fun)
    if (
        result.success
        and np.isfinite(refined_objective)
        and refined_objective <= grid_best_objective
    ):
        best_shift = float(result.x)
        best_objective = refined_objective
    else:
        best_shift = grid_best_shift
        best_objective = grid_best_objective

    best_alignment = _replace(base_alignment, candidate_time_shift=best_shift)
    diagnostics = {
        "parameter": "candidate_time_shift",
        "bounds": tuple(float(x) for x in bounds),
        "refinement_bounds": refine_bounds,
        "initial_shift": float(base_alignment.candidate_time_shift),
        "coarse_grid_best_shift": grid_best_shift,
        "coarse_grid_best_mismatch": grid_best_objective,
        "refined_best_shift": float(result.x),
        "refined_best_mismatch": refined_objective,
        "best_shift": best_shift,
        "best_mismatch": best_objective,
        "success": bool(result.success),
        "message": str(result.message),
        "n_grid_evaluations": int(n_grid),
        "n_refinement_evaluations": int(n_refinement_evaluations),
        "n_evaluations": int(n_grid + n_refinement_evaluations),
    }
    return best_alignment, diagnostics


def _optimize_time_shift_roll(
    modes_a: Any,
    modes_b: Any,
    selected_modes: Sequence[tuple[int, int]],
    alignment: Any,
) -> tuple[Any, dict]:
    """Time-shift optimizer: prepare once, shift via integer roll + FD correction.

    Calls ``prepare_mode_data`` exactly once at Δt=0 to build the stacked
    mode arrays on the common reference grid.  Each candidate shift is applied
    by rolling the candidate stack by the nearest integer number of samples
    (O(n_modes × N) in-place) followed by a sub-sample phase-ramp correction
    in the frequency domain to reach spectral accuracy.  The mode overlaps
    ``J_lm`` are then computed with a single vectorized trapezoid call.

    Cost relative to ``"grid"``: one ``prepare_mode_data`` call instead of
    ``n_grid + n_refine`` calls; the per-step cost drops from full resampling
    to one FFT/IFFT pair on the stacked array.
    """
    from dataclasses import replace as _replace

    base_alignment = _replace(alignment, time_domain_policy="resample_to_reference")
    bounds = _time_shift_bounds(modes_a, modes_b, base_alignment)

    prepared_0 = prepare_mode_data(
        modes_a, modes_b, selected_modes,
        _replace(base_alignment, candidate_time_shift=0.0),
    )
    time_axis = prepared_0.time_axis
    N = len(time_axis)
    dt = float(time_axis[1] - time_axis[0]) if N > 1 else 1.0

    norm_a = _prepared_norm(prepared_0.modes_a, selected_modes, time_axis)
    norm_b = _prepared_norm(prepared_0.modes_b, selected_modes, time_axis)

    if not selected_modes or norm_a == 0.0 or norm_b == 0.0:
        best_alignment = _replace(base_alignment, candidate_time_shift=0.0)
        return best_alignment, {
            "parameter": "candidate_time_shift", "bounds": bounds,
            "best_shift": 0.0, "n_evaluations": 0,
            "n_grid_evaluations": 0, "n_refinement_evaluations": 0,
        }

    stack_a = np.stack([prepared_0.modes_a[m] for m in selected_modes])
    stack_b0 = np.stack([prepared_0.modes_b[m] for m in selected_modes])
    m_values = np.fromiter(
        (emm for _ell, emm in selected_modes), dtype=float, count=len(selected_modes)
    )
    freqs = np.fft.fftfreq(N, d=dt)

    def j_lm_at_shift(delta_t: float) -> np.ndarray:
        n_samples = int(np.round(delta_t / dt))
        residual_dt = delta_t - n_samples * dt
        # candidate_time_shift=delta_t shifts time_b by +delta_t, so the grid
        # method evaluates b at (t - delta_t).  Rolling by n_samples reproduces
        # this: roll(b, n)[i] = b[i - n] = b(t_i - n*dt) = b(t_i - delta_t).
        rolled = np.roll(stack_b0, n_samples, axis=-1)
        if abs(residual_dt) > 1e-14 * abs(dt):
            phase = np.exp(-2j * np.pi * freqs * residual_dt)
            B_hat = np.fft.fft(rolled, axis=-1)
            rolled = np.fft.ifft(B_hat * phase[np.newaxis, :], axis=-1)
        return _trapz_complex_axis(np.conj(stack_a) * rolled, time_axis)

    def objective(delta_t: float) -> float:
        j_lm = j_lm_at_shift(float(delta_t))
        raw = _inner_from_overlaps(m_values, j_lm, 0.0)
        phase_result = _maximize_orbital_phase(prepared_0, raw, m_values, j_lm)
        inner = complex(phase_result["inner"])
        match = float(np.real(inner) / (norm_a * norm_b))
        return 1.0 - float(np.clip(match, -1.0, 1.0))

    n_grid = _time_shift_grid_sample_count(modes_a, modes_b, bounds, base_alignment)
    return _time_shift_grid_search_and_refine(objective, bounds, n_grid, base_alignment)


def _optimize_time_shift_ifft(
    modes_a: Any,
    modes_b: Any,
    selected_modes: Sequence[tuple[int, int]],
    alignment: Any,
) -> tuple[Any, dict]:
    """Time-shift optimizer: precompute full cross-correlation via batched FFT.

    Calls ``prepare_mode_data`` once, zero-pads to 2N to suppress circular
    aliasing, then computes the per-mode cross-correlation array
    ``C_lm[k] ≈ ∫ conj(a_lm(t)) b_lm(t + k·dt) dt`` in one batched
    ``rfft`` / ``irfft`` pass.  At each candidate shift Δt, the per-mode
    overlaps ``J_lm(Δt) = C_lm[-Δt/dt]`` are read off as O(1) array lookups
    — no integration in the time-shift loop.

    Cost: three batched FFTs (fixed, independent of grid size) plus the same
    phase-search arithmetic as the other methods.  The phase optimizer reuses
    the existing ``_maximize_orbital_phase`` machinery on the looked-up
    ``J_lm`` values.
    """
    from dataclasses import replace as _replace

    base_alignment = _replace(alignment, time_domain_policy="resample_to_reference")
    bounds = _time_shift_bounds(modes_a, modes_b, base_alignment)

    prepared_0 = prepare_mode_data(
        modes_a, modes_b, selected_modes,
        _replace(base_alignment, candidate_time_shift=0.0),
    )
    time_axis = prepared_0.time_axis
    N = len(time_axis)
    dt = float(time_axis[1] - time_axis[0]) if N > 1 else 1.0

    norm_a = _prepared_norm(prepared_0.modes_a, selected_modes, time_axis)
    norm_b = _prepared_norm(prepared_0.modes_b, selected_modes, time_axis)

    if not selected_modes or norm_a == 0.0 or norm_b == 0.0:
        best_alignment = _replace(base_alignment, candidate_time_shift=0.0)
        return best_alignment, {
            "parameter": "candidate_time_shift", "bounds": bounds,
            "best_shift": 0.0, "n_evaluations": 0,
            "n_grid_evaluations": 0, "n_refinement_evaluations": 0,
        }

    stack_a = np.stack([prepared_0.modes_a[m] for m in selected_modes])
    stack_b = np.stack([prepared_0.modes_b[m] for m in selected_modes])
    m_values = np.fromiter(
        (emm for _ell, emm in selected_modes), dtype=float, count=len(selected_modes)
    )

    # Zero-pad to 2N to avoid circular aliasing across the full shift range.
    N_pad = 2 * N
    A_hat = np.fft.fft(stack_a, n=N_pad, axis=-1)  # (n_modes, N_pad)
    B_hat = np.fft.fft(stack_b, n=N_pad, axis=-1)

    # Circular cross-correlation via DFT:
    # C_lm[k] = Σ_n conj(a_lm[n]) b_lm[(n+k) mod N_pad]
    #          = N_pad * ifft(conj(A_hat) * B_hat)[k]
    # Scaled to approximate the trapezoid integral:
    # J_lm(lag=k·dt) ≈ dt * C_lm[k]
    # ifft(conj(A)*B)[k] = (1/N_pad) Σ_f conj(A[f])*B[f]*exp(2πikf/N_pad)
    # At k=0 this equals Σ_n conj(a[n])*b[n] (Parseval for zero-padded arrays),
    # so dt * ifft(...)[k] gives the trapezoid-approximated cross-correlation
    # at lag k·dt without any additional N_pad factor.
    xcorr = dt * np.fft.ifft(
        np.conj(A_hat) * B_hat, axis=-1
    )  # (n_modes, N_pad), complex

    def j_lm_at_shift(delta_t: float) -> np.ndarray:
        # b_lm(t - Δt): lag = -Δt → index round(-Δt/dt) mod N_pad
        k = int(np.round(-delta_t / dt)) % N_pad
        return xcorr[:, k].astype(complex)

    def objective(delta_t: float) -> float:
        j_lm = j_lm_at_shift(float(delta_t))
        raw = _inner_from_overlaps(m_values, j_lm, 0.0)
        phase_result = _maximize_orbital_phase(prepared_0, raw, m_values, j_lm)
        inner = complex(phase_result["inner"])
        match = float(np.real(inner) / (norm_a * norm_b))
        return 1.0 - float(np.clip(match, -1.0, 1.0))

    n_grid = _time_shift_grid_sample_count(modes_a, modes_b, bounds, base_alignment)
    return _time_shift_grid_search_and_refine(objective, bounds, n_grid, base_alignment)


def _optimize_z_rotation(
    modes_a: Any,
    modes_b: Any,
    selected_modes: Sequence[tuple[int, int]],
    alignment: AlignmentSpec,
    rotation: RotationSpec,
) -> tuple[RotationSpec, dict[str, Any]]:
    bounds = _rotation_angle_bounds(rotation)
    n_evaluations = 0

    def objective(angle: float) -> float:
        nonlocal n_evaluations
        n_evaluations += 1
        trial_rotation = replace(rotation, kind="z_axis", angle=float(angle))
        trial_modes_b = rotate_modes(
            modes_b, trial_rotation, modes=selected_modes
        )
        try:
            evaluation = _evaluate_prepared_match(
                modes_a,
                trial_modes_b,
                selected_modes,
                alignment,
            )
        except ValueError:
            return np.inf
        match_value = evaluation["match"]
        if not np.isfinite(match_value):
            return np.inf
        return 1.0 - float(np.clip(match_value, -1.0, 1.0))

    try:
        from scipy.optimize import minimize_scalar
    except Exception as exc:  # pragma: no cover - scipy is normally available
        raise ImportError(
            "rotation optimize_angle=True requires scipy.optimize.minimize_scalar"
        ) from exc

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    best_angle = float(result.x)
    best_rotation = replace(rotation, kind="z_axis", angle=best_angle)
    diagnostics = {
        "parameter": "rotation.angle",
        "bounds": tuple(float(item) for item in bounds),
        "initial_angle": float(rotation.angle),
        "best_angle": best_angle,
        "best_mismatch": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "n_evaluations": int(n_evaluations),
    }
    return best_rotation, diagnostics


def _optimize_wigner_rotation(
    modes_a: Any,
    modes_b: Any,
    selected_modes: Sequence[tuple[int, int]],
    alignment: AlignmentSpec,
    rotation: RotationSpec,
) -> tuple[RotationSpec, dict[str, Any]]:
    parameters = tuple(rotation.optimize_parameters)
    bounds = _wigner_parameter_bounds(rotation, parameters)
    initial_values = np.array(
        [float(getattr(rotation, parameter)) for parameter in parameters],
        dtype=float,
    )
    n_evaluations = 0

    def objective(values: np.ndarray) -> float:
        nonlocal n_evaluations
        n_evaluations += 1
        trial_kwargs = {
            parameter: float(value)
            for parameter, value in zip(parameters, values)
        }
        trial_rotation = replace(rotation, **trial_kwargs)
        trial_modes_b = rotate_modes(
            modes_b, trial_rotation, modes=selected_modes
        )
        try:
            evaluation = _evaluate_prepared_match(
                modes_a,
                trial_modes_b,
                selected_modes,
                alignment,
            )
        except ValueError:
            return np.inf
        match_value = evaluation["match"]
        if not np.isfinite(match_value):
            return np.inf
        return 1.0 - float(np.clip(match_value, -1.0, 1.0))

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - scipy is normally available
        raise ImportError(
            "rotation optimize_parameters requires scipy.optimize.minimize"
        ) from exc

    result = minimize(
        objective,
        initial_values,
        bounds=[bounds[parameter] for parameter in parameters],
        method="L-BFGS-B",
    )
    best_values = {
        parameter: float(value)
        for parameter, value in zip(parameters, result.x)
    }
    best_rotation = replace(rotation, **best_values)
    diagnostics = {
        "parameters": parameters,
        "bounds": bounds,
        "initial_values": {
            parameter: float(getattr(rotation, parameter))
            for parameter in parameters
        },
        "best_values": best_values,
        "best_mismatch": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "n_evaluations": int(n_evaluations),
        "phase_degeneracy_possible": _rotation_phase_degeneracy_possible(
            alignment,
            parameters,
        ),
    }
    return best_rotation, diagnostics


def _rotation_angle_bounds(rotation: RotationSpec) -> tuple[float, float]:
    if rotation.angle_bounds is not None:
        return tuple(float(item) for item in rotation.angle_bounds)
    center = float(rotation.angle)
    return center - np.pi, center + np.pi


def _wigner_parameter_bounds(
    rotation: RotationSpec,
    parameters: Sequence[str],
) -> dict[str, tuple[float, float]]:
    user_bounds = rotation.parameter_bounds or {}
    bounds: dict[str, tuple[float, float]] = {}
    for parameter in parameters:
        if parameter in user_bounds:
            bounds[parameter] = tuple(
                float(item) for item in user_bounds[parameter]
            )
            continue
        center = float(getattr(rotation, parameter))
        bounds[parameter] = (center - np.pi, center + np.pi)
    return bounds


def _rotation_phase_degeneracy_possible(
    alignment: AlignmentSpec,
    parameters: Sequence[str],
) -> bool:
    return alignment.phase_alignment in {
        "orbital_phase",
        "orbital_phase_and_global",
    } and bool({"alpha", "gamma"}.intersection(parameters))


def _validate_phase_rotation_degeneracy(
    alignment: AlignmentSpec,
    rotation: RotationSpec,
) -> None:
    if alignment.allow_phase_rotation_degeneracy:
        return
    if alignment.phase_alignment not in {
        "orbital_phase",
        "orbital_phase_and_global",
    }:
        return
    degenerate = False
    if rotation.optimize_angle and rotation.kind == "z_axis":
        degenerate = True
    if rotation.optimize_parameters and {"alpha", "gamma"}.intersection(
        rotation.optimize_parameters
    ):
        degenerate = True
    if degenerate:
        raise ValueError(
            "Simultaneously optimizing orbital phase and z-axis-like rotation "
            "is degenerate. Use phase_alignment='global_complex' with z-axis "
            "rotation optimization, use orbital-phase alignment without z-axis "
            "rotation optimization, or set allow_phase_rotation_degeneracy=True."
        )


def _time_shift_bounds(
    modes_a: Any, modes_b: Any, alignment: AlignmentSpec
) -> tuple[float, float]:
    if alignment.time_shift_bounds is not None:
        return tuple(float(item) for item in alignment.time_shift_bounds)

    time_a = np.asarray(modes_a.time_axis, dtype=float)
    time_b = np.asarray(modes_b.time_axis, dtype=float)
    duration_a = float(time_a[-1] - time_a[0])
    duration_b = float(time_b[-1] - time_b[0])
    half_width = 0.25 * min(duration_a, duration_b)
    center = float(alignment.candidate_time_shift)
    return center - half_width, center + half_width


def _time_shift_grid_sample_count(
    modes_a: Any,
    modes_b: Any,
    bounds: tuple[float, float],
    alignment: AlignmentSpec,
) -> int:
    if alignment.time_shift_grid_samples is not None:
        return int(alignment.time_shift_grid_samples)

    dt_values = []
    for modes_obj in (modes_a, modes_b):
        time_axis = np.asarray(modes_obj.time_axis, dtype=float)
        dt, uniform = _sampling_interval_for_bounds(time_axis)
        if uniform and dt is not None:
            dt_values.append(abs(dt))
    width = float(bounds[1] - bounds[0])
    if dt_values:
        dt = min(value for value in dt_values if value > 0.0)
        return max(401, int(np.ceil(width / dt)) + 1)
    return 401


def _sampling_interval_for_bounds(
    time_axis: np.ndarray,
) -> tuple[float | None, bool]:
    if len(time_axis) < 2:
        return None, False
    diffs = np.diff(time_axis)
    dt = float(np.median(diffs))
    return dt, bool(np.allclose(diffs, dt, rtol=1e-10, atol=1e-12))


def _orbital_phase_inner(prepared: PreparedModeData, phase: float) -> complex:
    return _prepared_inner_product(
        prepared.modes_a,
        prepared.modes_b,
        prepared.selected_modes,
        prepared.time_axis,
        orbital_phase=float(phase),
    )


def _global_phase_from_inner(inner: complex) -> float:
    if abs(inner) == 0.0:
        return 0.0
    return float(-np.angle(inner))


def _orbital_phase_score(prepared: PreparedModeData, inner: complex) -> float:
    if prepared.alignment.phase_alignment == "orbital_phase_and_global":
        return float(abs(inner))
    return float(inner.real)


def _maximize_orbital_phase_grid(
    prepared: PreparedModeData,
    m_values: np.ndarray,
    overlaps: np.ndarray,
) -> dict[str, Any]:
    n = int(prepared.alignment.orbital_phase_samples)
    phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    values = np.array(
        [_inner_from_overlaps(m_values, overlaps, float(phase)) for phase in phases]
    )
    scores = np.array(
        [_orbital_phase_score(prepared, complex(value)) for value in values],
        dtype=float,
    )
    if prepared.alignment.phase_alignment == "orbital_phase_and_global":
        index = int(np.argmax(np.abs(values)))
    else:
        index = int(np.argmax(values.real))
    phase = float(phases[index])
    inner = complex(values[index])
    score = _orbital_phase_score(prepared, inner)
    step = float(2.0 * np.pi / n)
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))
    scale = max(abs(max_score), 1e-300)
    relative_span = float((max_score - min_score) / scale)
    return {
        "phase": phase,
        "inner": inner,
        "score": score,
        "step": step,
        "relative_span": relative_span,
        "max_score": max_score,
        "min_score": min_score,
        "n_grid_evaluations": int(n),
    }


def _maximize_orbital_phase(
    prepared: PreparedModeData,
    raw_inner: complex,
    m_values: np.ndarray,
    overlaps: np.ndarray,
) -> dict[str, Any]:
    grid = _maximize_orbital_phase_grid(prepared, m_values, overlaps)
    diagnostics = {
        "phase_alignment": prepared.alignment.phase_alignment,
        "orbital_phase_optimizer": prepared.alignment.orbital_phase_optimizer,
        "orbital_phase_samples": int(prepared.alignment.orbital_phase_samples),
        "coarse_grid_best_phase": float(grid["phase"]),
        "coarse_grid_objective": float(grid["score"]),
        "phase_objective_relative_span": float(grid["relative_span"]),
        "phase_objective_grid_min": float(grid["min_score"]),
        "phase_objective_grid_max": float(grid["max_score"]),
        "orbital_phase_degenerate": False,
        "n_grid_evaluations": int(grid["n_grid_evaluations"]),
        "n_refinement_evaluations": 0,
    }
    if (
        prepared.alignment.phase_alignment == "orbital_phase_and_global"
        and grid["relative_span"] <= prepared.alignment.phase_degeneracy_tol
    ):
        diagnostics.update(
            {
                "orbital_phase_degenerate": True,
                "refined_orbital_phase": 0.0,
                "objective_value": float(abs(raw_inner)),
                "degenerate_resolution": "global_phase_only",
            }
        )
        return {
            "orbital_phase": 0.0,
            "inner": raw_inner,
            "diagnostics": diagnostics,
        }

    if prepared.alignment.orbital_phase_optimizer == "grid":
        diagnostics.update(
            {
                "refined_orbital_phase": float(grid["phase"]),
                "objective_value": float(grid["score"]),
            }
        )
        return {
            "orbital_phase": float(grid["phase"]),
            "inner": complex(grid["inner"]),
            "diagnostics": diagnostics,
        }

    try:
        from scipy.optimize import minimize_scalar
    except Exception as exc:  # pragma: no cover - scipy is normally available
        raise ImportError(
            "orbital_phase_optimizer='continuous' requires scipy.optimize.minimize_scalar"
        ) from exc

    def objective(candidate_phase: float) -> float:
        wrapped_phase = float(candidate_phase % (2.0 * np.pi))
        candidate_inner = _inner_from_overlaps(m_values, overlaps, wrapped_phase)
        return -_orbital_phase_score(prepared, candidate_inner)

    phase = float(grid["phase"])
    step = float(grid["step"])
    n_refinement_evaluations = 0

    def counted_objective(candidate_phase: float) -> float:
        nonlocal n_refinement_evaluations
        n_refinement_evaluations += 1
        return objective(candidate_phase)

    result = minimize_scalar(
        counted_objective,
        bounds=(phase - step, phase + step),
        method="bounded",
        options={"xatol": 1e-12},
    )
    refined_phase = float(result.x % (2.0 * np.pi))
    refined_inner = _inner_from_overlaps(m_values, overlaps, refined_phase)
    refined_score = _orbital_phase_score(prepared, refined_inner)
    if (
        result.success
        and np.isfinite(refined_score)
        and refined_score >= grid["score"]
    ):
        diagnostics.update(
            {
                "refined_orbital_phase": refined_phase,
                "objective_value": float(refined_score),
                "n_refinement_evaluations": int(n_refinement_evaluations),
                "refinement_success": bool(result.success),
                "refinement_message": str(result.message),
            }
        )
        return {
            "orbital_phase": refined_phase,
            "inner": refined_inner,
            "diagnostics": diagnostics,
        }
    diagnostics.update(
        {
            "refined_orbital_phase": float(grid["phase"]),
            "objective_value": float(grid["score"]),
            "n_refinement_evaluations": int(n_refinement_evaluations),
            "refinement_success": bool(result.success),
            "refinement_message": str(result.message),
            "refinement_fallback": "coarse_grid",
        }
    )
    return {
        "orbital_phase": float(grid["phase"]),
        "inner": complex(grid["inner"]),
        "diagnostics": diagnostics,
    }


def _phase_optimizer_name(alignment: AlignmentSpec) -> str | None:
    if alignment.phase_alignment == "none":
        return None
    if alignment.phase_alignment == "global_complex":
        return "analytic_global_phase"
    if alignment.phase_alignment in {
        "orbital_phase",
        "orbital_phase_and_global",
    }:
        return f"{alignment.orbital_phase_optimizer}_orbital_phase"
    return None


def _optimizer_name(
    alignment: AlignmentSpec, rotation: RotationSpec | None = None
) -> str | None:
    phase_optimizer = _phase_optimizer_name(alignment)
    parts: list[str] = []
    if rotation is not None and rotation.optimize_angle:
        parts.append("bounded_z_rotation")
    if rotation is not None and rotation.optimize_parameters:
        parts.append("bounded_wigner_rotation")
    if alignment.optimize_time_shift:
        parts.append("grid_refined_time_shift")
    if phase_optimizer is not None:
        parts.append(phase_optimizer)
    if parts:
        return "+".join(parts)
    return None
