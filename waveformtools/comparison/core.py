"""Core comparison routines for waveform mode arrays.

This module compares already-generated mode objects.  It now makes time- and
phase-alignment choices explicit through ``AlignmentSpec`` while still avoiding
full intrinsic-parameter fitting-factor searches.  Expensive waveform-family
optimization is layered on top of this core in later PRs.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from waveformtools.comparison.alignment import AlignmentSpec, PreparedModeData, prepare_mode_data
from waveformtools.comparison.metadata import get_comparison_metadata
from waveformtools.comparison.results import ComparisonResult

ModeSelector = Sequence[tuple[int, int]] | None


def available_modes(modes: Any, ell_min: int = 2, ell_max: int | None = None) -> list[tuple[int, int]]:
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

    return [(ell, emm) for ell in range(ell_min, ell_max + 1) for emm in range(-ell, ell + 1)]


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


def assert_common_time_axis(modes_a: Any, modes_b: Any, *, rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
    """Validate and return a common time axis.

    This is kept as a compatibility helper for callers that want the original
    strict behavior.  New code should prefer ``AlignmentSpec`` via
    ``mode_match(..., alignment=...)``.
    """

    time_a = np.asarray(modes_a.time_axis, dtype=float)
    time_b = np.asarray(modes_b.time_axis, dtype=float)
    alignment = AlignmentSpec(time_domain_policy="error", time_axis_rtol=rtol, time_axis_atol=atol)
    prepared = prepare_mode_data(modes_a, modes_b, common_modes(modes_a, modes_b), alignment)
    if time_a.shape != time_b.shape:  # defensive; prepare_mode_data already checks
        raise ValueError(
            "Mode comparisons currently require a common time grid; "
            f"got {time_a.shape} and {time_b.shape}."
        )
    return prepared.time_axis


def _trapz_complex(values: np.ndarray, time_axis: np.ndarray) -> complex:
    if hasattr(np, "trapezoid"):
        return complex(np.trapezoid(values, time_axis))
    return complex(np.trapz(values, time_axis))


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
    for ell, emm in common_modes(modes_a, modes_b, ell_min=ell_min, ell_max=ell_max, modes=modes):
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
    time_alignment: str | None = None,
    time_domain_policy: str | None = None,
    phase_alignment: str | None = None,
    resample_method: str | None = None,
    candidate_time_shift: float | None = None,
    time_axis_rtol: float | None = None,
    time_axis_atol: float | None = None,
) -> ComparisonResult:
    """Compute a normalized mode match with explicit alignment choices.

    Parameters
    ----------
    alignment:
        Optional ``AlignmentSpec`` or mapping.  Individual keyword arguments
        such as ``time_alignment`` and ``phase_alignment`` override it.
    phase_maximize:
        Backwards-compatible shorthand.  If supplied, it overrides
        ``phase_alignment`` with ``global_complex`` when true and ``none`` when
        false.
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
        time_axis_rtol=time_axis_rtol,
        time_axis_atol=time_axis_atol,
    )

    selected_modes = common_modes(modes_a, modes_b, ell_min=ell_min, ell_max=ell_max, modes=modes)
    prepared = prepare_mode_data(modes_a, modes_b, selected_modes, alignment_spec)
    inner, best_phase, raw_inner = _phase_aligned_inner_product(prepared)
    norm_a = _prepared_norm(prepared.modes_a, prepared.selected_modes, prepared.time_axis)
    norm_b = _prepared_norm(prepared.modes_b, prepared.selected_modes, prepared.time_axis)

    if norm_a == 0.0 or norm_b == 0.0:
        match_value = np.nan
    else:
        match_value = float(inner / (norm_a * norm_b))

    if np.isfinite(match_value):
        # Guard against harmless numerical overshoots for identical arrays.
        match_value = float(np.clip(match_value, -1.0, 1.0))
        mismatch_value = 1.0 - match_value
    else:
        mismatch_value = np.nan

    return ComparisonResult(
        objective_name="mode_match",
        match=match_value,
        mismatch=mismatch_value,
        best_parameters={
            "phase_alignment": alignment_spec.phase_alignment,
            "orbital_phase": best_phase,
            "time_alignment": alignment_spec.time_alignment,
            "time_domain_policy": alignment_spec.time_domain_policy,
            "candidate_time_shift": alignment_spec.candidate_time_shift,
        },
        optimizer=_phase_optimizer_name(alignment_spec.phase_alignment),
        optimizer_status="ok",
        elapsed_s=time.perf_counter() - t0,
        diagnostics={
            "raw_inner_product_real": float(raw_inner.real),
            "raw_inner_product_imag": float(raw_inner.imag),
            "raw_inner_product_abs": float(abs(raw_inner)),
            "phase_aligned_inner": float(inner),
            "norm_a": norm_a,
            "norm_b": norm_b,
            "n_modes": len(selected_modes),
            "modes": selected_modes,
            "alignment": alignment_spec.to_dict(),
            "time_axis": prepared.diagnostics.to_dict(),
        },
        target_metadata=get_comparison_metadata(modes_a),
        candidate_metadata=get_comparison_metadata(modes_b),
    )


def mode_mismatch(*args: Any, **kwargs: Any) -> float:
    """Convenience wrapper returning only ``1 - match``."""

    return float(mode_match(*args, **kwargs).mismatch)


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
        diagnostics={"norm_a": norm_a, "norm_b": norm_b, "shape": residue_a.shape},
        target_metadata=get_comparison_metadata(modes_a),
        candidate_metadata=get_comparison_metadata(modes_b),
    )


def _prepared_inner_product(
    arrays_a: dict[tuple[int, int], np.ndarray],
    arrays_b: dict[tuple[int, int], np.ndarray],
    selected_modes: Sequence[tuple[int, int]],
    time_axis: np.ndarray,
    *,
    orbital_phase: float = 0.0,
) -> complex:
    total = 0.0j
    for ell, emm in selected_modes:
        phase_factor = np.exp(1j * emm * orbital_phase)
        total += _trapz_complex(np.conjugate(arrays_a[(ell, emm)]) * phase_factor * arrays_b[(ell, emm)], time_axis)
    return total


def _prepared_norm(
    arrays: dict[tuple[int, int], np.ndarray],
    selected_modes: Sequence[tuple[int, int]],
    time_axis: np.ndarray,
) -> float:
    total = 0.0j
    for mode in selected_modes:
        data = arrays[mode]
        total += _trapz_complex(np.conjugate(data) * data, time_axis)
    return float(np.sqrt(max(total.real, 0.0)))


def _phase_aligned_inner_product(prepared: PreparedModeData) -> tuple[float, float | None, complex]:
    raw = _prepared_inner_product(prepared.modes_a, prepared.modes_b, prepared.selected_modes, prepared.time_axis)
    phase_alignment = prepared.alignment.phase_alignment

    if phase_alignment == "none":
        return float(raw.real), None, raw
    if phase_alignment == "global_complex":
        return float(abs(raw)), None, raw

    best_phase, best_inner = _maximize_orbital_phase(prepared)
    if phase_alignment == "orbital_phase":
        return float(best_inner.real), best_phase, raw
    if phase_alignment == "orbital_phase_and_global":
        return float(abs(best_inner)), best_phase, raw
    raise ValueError(f"Unsupported phase_alignment={phase_alignment!r}")


def _maximize_orbital_phase(prepared: PreparedModeData) -> tuple[float, complex]:
    n = int(prepared.alignment.orbital_phase_samples)
    phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    values = np.array(
        [
            _prepared_inner_product(
                prepared.modes_a,
                prepared.modes_b,
                prepared.selected_modes,
                prepared.time_axis,
                orbital_phase=float(phase),
            )
            for phase in phases
        ]
    )
    if prepared.alignment.phase_alignment == "orbital_phase_and_global":
        index = int(np.argmax(np.abs(values)))
    else:
        index = int(np.argmax(values.real))
    return float(phases[index]), complex(values[index])


def _phase_optimizer_name(phase_alignment: str) -> str | None:
    if phase_alignment == "none":
        return None
    if phase_alignment == "global_complex":
        return "analytic_global_phase"
    if phase_alignment in {"orbital_phase", "orbital_phase_and_global"}:
        return "grid_orbital_phase"
    return None
