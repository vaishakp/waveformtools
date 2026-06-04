"""Core fixed-frame comparison routines for waveform mode arrays.

This first implementation is intentionally conservative: it compares two
already-generated mode objects on a common time grid and optionally maximizes
over a single global complex phase. It does not yet rotate frames, optimize
time shifts, or generate new candidate waveforms.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

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

    Later comparison stages will include resampling and time-shift
    optimization. This fixed-frame first pass requires identical grids so the
    objective is unambiguous.
    """

    time_a = np.asarray(modes_a.time_axis, dtype=float)
    time_b = np.asarray(modes_b.time_axis, dtype=float)
    if time_a.shape != time_b.shape:
        raise ValueError(
            "Mode comparisons currently require a common time grid; "
            f"got {time_a.shape} and {time_b.shape}."
        )
    if not np.allclose(time_a, time_b, rtol=rtol, atol=atol):
        raise ValueError("Mode comparisons currently require matching time-axis values.")
    return time_a


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

    This is a mode-space diagnostic, not a detector PSD-weighted match.
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
    phase_maximize: bool = True,
    time_axis_rtol: float = 1e-10,
    time_axis_atol: float = 1e-12,
) -> ComparisonResult:
    """Compute a fixed-frame normalized mode match.

    If ``phase_maximize`` is true, maximize over a single global complex phase
    by taking ``abs(<a,b>)``. This is not yet a per-mode or orbital-phase
    maximization.
    """

    t0 = time.perf_counter()
    time_axis = assert_common_time_axis(modes_a, modes_b, rtol=time_axis_rtol, atol=time_axis_atol)
    selected_modes = common_modes(modes_a, modes_b, ell_min=ell_min, ell_max=ell_max, modes=modes)
    inner = mode_inner_product(modes_a, modes_b, modes=selected_modes, time_axis=time_axis)
    norm_a = mode_norm(modes_a, modes=selected_modes, time_axis=time_axis)
    norm_b = mode_norm(modes_b, modes=selected_modes, time_axis=time_axis)

    if norm_a == 0.0 or norm_b == 0.0:
        match_value = np.nan
    elif phase_maximize:
        match_value = float(np.abs(inner) / (norm_a * norm_b))
    else:
        match_value = float(inner.real / (norm_a * norm_b))

    if np.isfinite(match_value):
        # Guard against harmless numerical overshoots for identical arrays.
        match_value = float(np.clip(match_value, -1.0, 1.0))
        mismatch_value = 1.0 - match_value
    else:
        mismatch_value = np.nan

    return ComparisonResult(
        objective_name="fixed_frame_mode_match",
        match=match_value,
        mismatch=mismatch_value,
        best_parameters={"phase_maximized": bool(phase_maximize)},
        optimizer="analytic_global_phase" if phase_maximize else None,
        optimizer_status="ok",
        elapsed_s=time.perf_counter() - t0,
        diagnostics={
            "inner_product_real": float(inner.real),
            "inner_product_imag": float(inner.imag),
            "inner_product_abs": float(abs(inner)),
            "norm_a": norm_a,
            "norm_b": norm_b,
            "n_modes": len(selected_modes),
            "modes": selected_modes,
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
