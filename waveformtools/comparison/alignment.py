"""Alignment policies for mode-space waveform comparisons.

The comparison core uses this module to make all alignment choices explicit.
The default behavior is deliberately useful for unequal-length waveforms:

1. align the two time axes by the peak of the total selected-mode news power,
2. require compatible sampling rates,
3. crop/truncate both waveforms to their common overlapping interval.

The first implemented subset supports:

- no time alignment,
- peak-of-(2,2) alignment,
- peak-of-total-mode-power alignment,
- peak-of-total-news-power alignment,
- strict common-grid comparisons,
- crop/truncate-to-overlap comparisons,
- resampling the candidate waveform to the reference grid,
- no phase alignment,
- global complex phase maximization,
- orbital-phase maximization with an ``exp(i m phi)`` factor.

Returned alignment parameters use a replay convention: after any configured
rotation, the candidate time axis is shifted by ``candidate_time_shift`` and
candidate modes are multiplied by
``exp(i global_phase) * exp(i m orbital_phase)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

TimeAlignment = Literal[
    "none", "peak_22", "peak_total_power", "peak_total_news_power"
]
TimeDomainPolicy = Literal["error", "crop_to_overlap", "resample_to_reference"]
PhaseAlignment = Literal[
    "none",
    "global_complex",
    "orbital_phase",
    "orbital_phase_and_global",
]
ResampleMethod = Literal["linear", "cubic"]
OrbitalPhaseOptimizer = Literal["grid", "continuous"]

_ALLOWED_TIME_ALIGNMENTS = {
    "none",
    "peak_22",
    "peak_total_power",
    "peak_total_news_power",
}
_ALLOWED_TIME_DOMAIN_POLICIES = {
    "error",
    "crop_to_overlap",
    "resample_to_reference",
}
_ALLOWED_PHASE_ALIGNMENTS = {
    "none",
    "global_complex",
    "orbital_phase",
    "orbital_phase_and_global",
}
_ALLOWED_RESAMPLE_METHODS = {"linear", "cubic"}
_ALLOWED_ORBITAL_PHASE_OPTIMIZERS = {"grid", "continuous"}


@dataclass(slots=True)
class AlignmentSpec:
    """User-facing alignment choices for a mode comparison.

    Defaults are chosen for the common case where two generated waveforms have
    the same sampling rate but different start/end times. The comparison aligns
    peaks and crops both waveforms to their common post-alignment interval.

    Parameters
    ----------
    time_alignment:
        How the time origins are chosen before comparing modes.
    time_domain_policy:
        How to handle unequal time axes after time-origin alignment.
    phase_alignment:
        What phase freedom is maximized before reporting a match.
    resample_method:
        Interpolation method used when resampling is required.
    candidate_time_shift:
        Manual candidate time shift applied after the chosen reference-time
        alignment. Positive values shift the candidate time axis forward.
    time_axis_rtol, time_axis_atol:
        Tolerances for strict grid checks and dt compatibility checks.
    minimum_overlap_samples:
        Required number of samples after overlap/cropping.
    orbital_phase_samples:
        Number of grid points used by the orbital-phase search. For continuous
        optimization this grid seeds the local scalar refinement.
    orbital_phase_optimizer:
        ``"grid"`` keeps the best discrete phase from ``orbital_phase_samples``.
        ``"continuous"`` refines that grid result with a bounded scalar
        optimizer.
    phase_degeneracy_tol:
        Relative objective-span tolerance below which
        ``orbital_phase_and_global`` is treated as orbital-phase degenerate.
    time_shift_grid_samples:
        Optional coarse-grid sample count for time-shift optimization. When
        omitted the grid is chosen from the overlap bounds and sampling rate.
    allow_phase_rotation_degeneracy:
        Opt in to simultaneous orbital-phase and z-axis rotation optimization.
        By default this degenerate combination raises a clear error.
    """

    time_alignment: TimeAlignment = "peak_total_news_power"
    time_domain_policy: TimeDomainPolicy = "crop_to_overlap"
    phase_alignment: PhaseAlignment = "global_complex"
    resample_method: ResampleMethod = "linear"
    candidate_time_shift: float = 0.0
    optimize_time_shift: bool = False
    time_shift_bounds: tuple[float, float] | None = None
    time_axis_rtol: float = 1e-10
    time_axis_atol: float = 1e-12
    minimum_overlap_samples: int = 8
    orbital_phase_samples: int = 257
    orbital_phase_optimizer: OrbitalPhaseOptimizer = "continuous"
    phase_degeneracy_tol: float = 1e-10
    time_shift_grid_samples: int | None = None
    allow_phase_rotation_degeneracy: bool = False

    def __post_init__(self) -> None:
        if self.time_alignment not in _ALLOWED_TIME_ALIGNMENTS:
            raise ValueError(
                f"Unsupported time_alignment={self.time_alignment!r}; "
                f"choose one of {sorted(_ALLOWED_TIME_ALIGNMENTS)}."
            )
        if self.time_domain_policy not in _ALLOWED_TIME_DOMAIN_POLICIES:
            raise ValueError(
                f"Unsupported time_domain_policy={self.time_domain_policy!r}; "
                f"choose one of {sorted(_ALLOWED_TIME_DOMAIN_POLICIES)}."
            )
        if self.phase_alignment not in _ALLOWED_PHASE_ALIGNMENTS:
            raise ValueError(
                f"Unsupported phase_alignment={self.phase_alignment!r}; "
                f"choose one of {sorted(_ALLOWED_PHASE_ALIGNMENTS)}."
            )
        if self.resample_method not in _ALLOWED_RESAMPLE_METHODS:
            raise ValueError(
                f"Unsupported resample_method={self.resample_method!r}; "
                f"choose one of {sorted(_ALLOWED_RESAMPLE_METHODS)}."
            )
        if (
            self.orbital_phase_optimizer
            not in _ALLOWED_ORBITAL_PHASE_OPTIMIZERS
        ):
            raise ValueError(
                "Unsupported orbital_phase_optimizer="
                f"{self.orbital_phase_optimizer!r}; choose one of "
                f"{sorted(_ALLOWED_ORBITAL_PHASE_OPTIMIZERS)}."
            )
        if self.minimum_overlap_samples < 2:
            raise ValueError("minimum_overlap_samples must be at least 2.")
        if self.orbital_phase_samples < 8:
            raise ValueError("orbital_phase_samples must be at least 8.")
        if self.phase_degeneracy_tol < 0.0:
            raise ValueError("phase_degeneracy_tol must be non-negative.")
        if (
            self.time_shift_grid_samples is not None
            and self.time_shift_grid_samples < 3
        ):
            raise ValueError("time_shift_grid_samples must be at least 3.")
        if self.time_shift_bounds is not None:
            lo, hi = self.time_shift_bounds
            if not lo < hi:
                raise ValueError(
                    "time_shift_bounds must be an increasing (lower, upper) pair."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        return asdict(self)

    @classmethod
    def from_value(
        cls,
        value: "AlignmentSpec | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "AlignmentSpec":
        """Construct an ``AlignmentSpec`` from a dataclass, mapping, or None."""

        if value is None:
            data: dict[str, Any] = {}
        elif isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "alignment must be an AlignmentSpec, a mapping, or None; "
                f"got {type(value)!r}."
            )
        data.update(
            {key: val for key, val in overrides.items() if val is not None}
        )
        return cls(**data)


@dataclass(slots=True)
class TimeAxisDiagnostics:
    """Diagnostics describing the time grids used in a comparison."""

    dt_a: float | None
    dt_b: float | None
    uniform_a: bool
    uniform_b: bool
    original_range_a: tuple[float, float]
    original_range_b: tuple[float, float]
    aligned_range_a: tuple[float, float]
    aligned_range_b: tuple[float, float]
    overlap: tuple[float, float] | None
    n_samples: int
    policy: str
    resample_method: str
    reference_time_a: float
    reference_time_b: float
    candidate_time_shift: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreparedModeData:
    """Mode arrays after time-origin alignment and grid handling."""

    time_axis: np.ndarray
    modes_a: dict[tuple[int, int], np.ndarray]
    modes_b: dict[tuple[int, int], np.ndarray]
    selected_modes: list[tuple[int, int]]
    alignment: AlignmentSpec
    diagnostics: TimeAxisDiagnostics


def prepare_mode_data(
    modes_a: Any,
    modes_b: Any,
    selected_modes: Sequence[tuple[int, int]],
    alignment: AlignmentSpec,
) -> PreparedModeData:
    """Prepare two modes objects for a fixed-frame mode comparison."""

    time_a_original = np.asarray(modes_a.time_axis, dtype=float)
    time_b_original = np.asarray(modes_b.time_axis, dtype=float)
    _check_time_axis(time_a_original, "modes_a")
    _check_time_axis(time_b_original, "modes_b")

    ref_a = reference_time(modes_a, selected_modes, alignment.time_alignment)
    ref_b = reference_time(modes_b, selected_modes, alignment.time_alignment)
    time_a = time_a_original - ref_a
    time_b = time_b_original - ref_b + float(alignment.candidate_time_shift)

    dt_a, uniform_a = sampling_interval(
        time_a_original, alignment.time_axis_rtol, alignment.time_axis_atol
    )
    dt_b, uniform_b = sampling_interval(
        time_b_original, alignment.time_axis_rtol, alignment.time_axis_atol
    )

    if alignment.time_domain_policy == "error":
        reference_grid = _strict_common_grid(time_a, time_b, alignment)
        arrays_a = {
            mode: np.asarray(modes_a.mode(*mode), dtype=np.complex128)
            for mode in selected_modes
        }
        arrays_b = {
            mode: np.asarray(modes_b.mode(*mode), dtype=np.complex128)
            for mode in selected_modes
        }
        overlap = (float(reference_grid[0]), float(reference_grid[-1]))
    else:
        overlap = overlap_interval(time_a, time_b)
        if overlap is None:
            raise ValueError(
                "No overlapping time interval after time alignment."
            )

        if alignment.time_domain_policy == "crop_to_overlap":
            _require_compatible_sampling(dt_a, dt_b, alignment)
            reference_grid, arrays_a, arrays_b = _crop_to_common_overlap(
                modes_a,
                modes_b,
                time_a,
                time_b,
                selected_modes,
                overlap,
                alignment,
            )
        else:
            mask = (time_a >= overlap[0] - alignment.time_axis_atol) & (
                time_a <= overlap[1] + alignment.time_axis_atol
            )
            reference_grid = time_a[mask]
            if len(reference_grid) < alignment.minimum_overlap_samples:
                raise ValueError(
                    "Insufficient overlap after alignment: "
                    f"{len(reference_grid)} samples < {alignment.minimum_overlap_samples}."
                )
            arrays_a = {
                mode: np.asarray(modes_a.mode(*mode), dtype=np.complex128)[mask]
                for mode in selected_modes
            }
            arrays_b = {
                mode: interpolate_complex(
                    time_b,
                    np.asarray(modes_b.mode(*mode), dtype=np.complex128),
                    reference_grid,
                    method=alignment.resample_method,
                )
                for mode in selected_modes
            }

    diagnostics = TimeAxisDiagnostics(
        dt_a=dt_a,
        dt_b=dt_b,
        uniform_a=uniform_a,
        uniform_b=uniform_b,
        original_range_a=(
            float(time_a_original[0]),
            float(time_a_original[-1]),
        ),
        original_range_b=(
            float(time_b_original[0]),
            float(time_b_original[-1]),
        ),
        aligned_range_a=(float(time_a[0]), float(time_a[-1])),
        aligned_range_b=(float(time_b[0]), float(time_b[-1])),
        overlap=overlap,
        n_samples=int(len(reference_grid)),
        policy=alignment.time_domain_policy,
        resample_method=alignment.resample_method,
        reference_time_a=float(ref_a),
        reference_time_b=float(ref_b),
        candidate_time_shift=float(alignment.candidate_time_shift),
    )

    return PreparedModeData(
        time_axis=reference_grid,
        modes_a=arrays_a,
        modes_b=arrays_b,
        selected_modes=list(selected_modes),
        alignment=alignment,
        diagnostics=diagnostics,
    )


def reference_time(
    modes_obj: Any, selected_modes: Sequence[tuple[int, int]], mode: str
) -> float:
    """Return the reference time implied by ``mode``."""

    time_axis = np.asarray(modes_obj.time_axis, dtype=float)
    if mode == "none":
        return 0.0
    if mode == "peak_22":
        amp = np.abs(np.asarray(modes_obj.mode(2, 2), dtype=np.complex128))
        return float(time_axis[int(np.argmax(amp))])
    if mode == "peak_total_power":
        power = _total_mode_power(modes_obj, selected_modes)
        return float(time_axis[int(np.argmax(power))])
    if mode == "peak_total_news_power":
        if not hasattr(modes_obj, "get_news_from_strain") or len(time_axis) < 4:
            power = _total_mode_power(modes_obj, selected_modes)
            return float(time_axis[int(np.argmax(power))])
        news_modes = modes_obj.get_news_from_strain()
        news_time_axis = np.asarray(news_modes.time_axis, dtype=float)
        power = _total_mode_power(news_modes, selected_modes)
        return float(news_time_axis[int(np.argmax(power))])
    raise ValueError(f"Unsupported time alignment mode: {mode!r}")


def _total_mode_power(
    modes_obj: Any, selected_modes: Sequence[tuple[int, int]]
) -> np.ndarray:
    time_axis = np.asarray(modes_obj.time_axis, dtype=float)
    power = np.zeros_like(time_axis, dtype=float)
    for ell, emm in selected_modes:
        data = np.asarray(modes_obj.mode(ell, emm), dtype=np.complex128)
        power += np.abs(data) ** 2
    return power


def sampling_interval(
    time_axis: np.ndarray, rtol: float, atol: float
) -> tuple[float | None, bool]:
    """Return median dt and whether the time axis is uniformly sampled."""

    if len(time_axis) < 2:
        return None, False
    diffs = np.diff(time_axis)
    dt = float(np.median(diffs))
    uniform = bool(np.allclose(diffs, dt, rtol=rtol, atol=atol))
    return dt, uniform


def overlap_interval(
    time_a: np.ndarray, time_b: np.ndarray
) -> tuple[float, float] | None:
    """Return the common interval of two increasing time arrays."""

    start = max(float(time_a[0]), float(time_b[0]))
    stop = min(float(time_a[-1]), float(time_b[-1]))
    if not start < stop:
        return None
    return start, stop


def interpolate_complex(
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
    *,
    method: ResampleMethod = "linear",
) -> np.ndarray:
    """Interpolate a complex time series onto a new real grid."""

    if method == "linear":
        real = np.interp(x_new, x_old, y_old.real)
        imag = np.interp(x_new, x_old, y_old.imag)
        return real + 1j * imag

    if method == "cubic":
        try:
            from scipy.interpolate import CubicSpline
        except Exception as exc:  # pragma: no cover - scipy is normally present
            raise ImportError(
                "cubic resampling requires scipy.interpolate.CubicSpline"
            ) from exc
        real = CubicSpline(x_old, y_old.real)(x_new)
        imag = CubicSpline(x_old, y_old.imag)(x_new)
        return real + 1j * imag

    raise ValueError(f"Unsupported resampling method: {method!r}")


def _crop_to_common_overlap(
    modes_a: Any,
    modes_b: Any,
    time_a: np.ndarray,
    time_b: np.ndarray,
    selected_modes: Sequence[tuple[int, int]],
    overlap: tuple[float, float],
    alignment: AlignmentSpec,
) -> tuple[
    np.ndarray,
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    """Crop both waveforms to the common overlap without interpolation."""

    mask_a = (time_a >= overlap[0] - alignment.time_axis_atol) & (
        time_a <= overlap[1] + alignment.time_axis_atol
    )
    mask_b = (time_b >= overlap[0] - alignment.time_axis_atol) & (
        time_b <= overlap[1] + alignment.time_axis_atol
    )
    grid_a = time_a[mask_a]
    grid_b = time_b[mask_b]
    n_samples = min(len(grid_a), len(grid_b))
    if n_samples < alignment.minimum_overlap_samples:
        raise ValueError(
            "Insufficient overlap after alignment: "
            f"{n_samples} samples < {alignment.minimum_overlap_samples}."
        )

    grid_a = grid_a[:n_samples]
    grid_b = grid_b[:n_samples]
    if not np.allclose(
        grid_a,
        grid_b,
        rtol=alignment.time_axis_rtol,
        atol=alignment.time_axis_atol,
    ):
        raise ValueError(
            "time_domain_policy='crop_to_overlap' requires aligned samples after peak/time alignment. "
            "Use time_domain_policy='resample_to_reference' if the grids have a sub-sample offset."
        )

    arrays_a = {
        mode: np.asarray(modes_a.mode(*mode), dtype=np.complex128)[mask_a][
            :n_samples
        ]
        for mode in selected_modes
    }
    arrays_b = {
        mode: np.asarray(modes_b.mode(*mode), dtype=np.complex128)[mask_b][
            :n_samples
        ]
        for mode in selected_modes
    }
    return grid_a, arrays_a, arrays_b


def _strict_common_grid(
    time_a: np.ndarray, time_b: np.ndarray, alignment: AlignmentSpec
) -> np.ndarray:
    if time_a.shape != time_b.shape:
        raise ValueError(
            "Mode comparisons with time_domain_policy='error' require a common time grid; "
            f"got shapes {time_a.shape} and {time_b.shape}. Use "
            "time_domain_policy='crop_to_overlap' or 'resample_to_reference' if this is intentional."
        )
    if not np.allclose(
        time_a,
        time_b,
        rtol=alignment.time_axis_rtol,
        atol=alignment.time_axis_atol,
    ):
        raise ValueError(
            "Mode comparisons with time_domain_policy='error' require matching time-axis values. "
            "Use an explicit alignment/time-domain policy if the difference is intentional."
        )
    return time_a


def _check_time_axis(time_axis: np.ndarray, name: str) -> None:
    if time_axis.ndim != 1:
        raise ValueError(f"{name}.time_axis must be one-dimensional.")
    if len(time_axis) < 2:
        raise ValueError(f"{name}.time_axis must contain at least two samples.")
    if np.any(np.diff(time_axis) <= 0):
        raise ValueError(f"{name}.time_axis must be strictly increasing.")


def _require_compatible_sampling(
    dt_a: float | None, dt_b: float | None, alignment: AlignmentSpec
) -> None:
    if dt_a is None or dt_b is None:
        raise ValueError(
            "Cannot check sampling-rate compatibility with fewer than two samples."
        )
    if not np.isclose(
        dt_a, dt_b, rtol=alignment.time_axis_rtol, atol=alignment.time_axis_atol
    ):
        raise ValueError(
            "time_domain_policy='crop_to_overlap' requires compatible sampling rates; "
            f"got dt_a={dt_a} and dt_b={dt_b}. Use "
            "time_domain_policy='resample_to_reference' to resample explicitly."
        )
