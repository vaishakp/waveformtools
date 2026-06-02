"""Memory-bounded balance-law utilities.

The legacy balance-law RHS evaluates full angular-time arrays with shape
``(nt, ntheta, nphi)``.  For long low-frequency waveforms this can dominate
memory.  This module keeps the angular algebra vectorized but evaluates and
integrates it in time chunks.

The waveform-generation step itself is still all-at-once; this only reduces
memory used after modes have been generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Union

import numpy as np

SliceLike = Union[slice, Tuple[int, int]]


@dataclass(frozen=True)
class ChunkedBalanceLawResult:
    """Return object for debug chunked RHS evaluations."""

    rhs: np.ndarray
    first_term: Optional[np.ndarray] = None
    second_term: Optional[np.ndarray] = None


def _normalize_time_slice(time_slice: Optional[SliceLike], data_len: int) -> slice:
    if time_slice is None:
        return slice(0, data_len)
    if isinstance(time_slice, tuple):
        if len(time_slice) != 2:
            raise ValueError("time_slice tuple must be (start, stop)")
        time_slice = slice(time_slice[0], time_slice[1])
    if not isinstance(time_slice, slice):
        raise TypeError("time_slice must be None, a slice, or a (start, stop) tuple")
    start, stop, step = time_slice.indices(data_len)
    if step != 1:
        raise ValueError("time_slice with step != 1 is not supported")
    if stop <= start:
        raise ValueError("time_slice selects no samples")
    return slice(start, stop)


def iter_time_chunks(data_len: int, chunk_size: int) -> Iterator[slice]:
    """Yield consecutive non-overlapping slices over a time axis."""

    if chunk_size is None:
        raise ValueError("chunk_size must be an integer")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, int(data_len), chunk_size):
        stop = min(start + chunk_size, int(data_len))
        if stop > start:
            yield slice(start, stop)


def _trapezoid(y: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def evaluate_angular_chunk(
    modes,
    Grid=None,
    time_slice: Optional[SliceLike] = None,
    theta=None,
    phi=None,
    ell_max: Optional[int] = None,
):
    """Evaluate a ModesArray on an angular grid for a time slice.

    This mirrors ``ModesArray.evaluate_angular`` but slices the last mode-data
    axis before the tensor contraction.  For ordinary two-index modes, the
    returned shape is ``(nt_chunk, ntheta, nphi)``.
    """

    from spectools.spherical.Yslm_mp import Yslm_mp

    if ell_max is None:
        ell_max = modes.ell_max
    if Grid is None:
        Grid = modes.Grid
    if Grid is None and (theta is None or phi is None):
        raise ValueError("Grid must be supplied when theta/phi are not supplied")
    if theta is None or phi is None:
        theta, phi = Grid.meshgrid
    if getattr(modes, "extra_mode_axes", False):
        raise NotImplementedError(
            "chunked angular evaluation currently supports ordinary two-index "
            "ModesArray objects only"
        )

    sl = _normalize_time_slice(time_slice, modes.data_len)
    sYlm = Yslm_mp(
        ell_max=ell_max,
        spin_weight=modes.spin_weight,
        theta=theta,
        phi=phi,
        Grid=Grid,
        cache=False,
    )
    sYlm.run()
    swsh_modes = sYlm.sYlm_modes._modes_data
    n_swsh_modes = swsh_modes.shape[0]
    return np.tensordot(
        modes._modes_data[:n_swsh_modes, ..., sl],
        swsh_modes,
        axes=((0), (0)),
    )


def _accumulate_time_integral(
    accumulator: np.ndarray,
    y_chunk: np.ndarray,
    t_chunk: np.ndarray,
    prev_t: Optional[float],
    prev_y: Optional[np.ndarray],
):
    """Accumulate a trapezoidal integral with chunk-boundary continuity."""

    if y_chunk.shape[0] != len(t_chunk):
        raise ValueError("time axis of y_chunk does not match t_chunk")
    if prev_y is not None:
        dt_boundary = t_chunk[0] - prev_t
        accumulator += 0.5 * dt_boundary * (prev_y + y_chunk[0])
    if len(t_chunk) > 1:
        accumulator += _trapezoid(y_chunk, x=t_chunk, axis=0)
    return t_chunk[-1], y_chunk[-1]


def rhs_balance_law_from_modes_chunked(
    strain_modes,
    Grid,
    chunk_size: int = 4096,
    ell_max: Optional[int] = None,
    debug: bool = False,
) -> Union[np.ndarray, ChunkedBalanceLawResult]:
    """Compute the infinite-time balance-law RHS in memory-bounded chunks.

    The returned RHS is on the angular grid.  This function uses trapezoidal
    time integration.  The legacy full-array path uses spline integration via
    ``twod_time_integral`` after materializing the full angular-time array, so
    short-waveform regression comparisons should allow small integration-method
    differences.
    """

    from qlmtools.spin_coefficient import eth_n_modes_from_modes

    news_modes = strain_modes.get_news_from_strain()
    news_modes._Grid = Grid
    conj_news_modes = news_modes.bar()
    eth2_conj_news_modes = eth_n_modes_from_modes(conj_news_modes, Grid, times=2)
    eth2_conj_news_modes._Grid = Grid

    theta, phi = Grid.meshgrid
    rhs_integral = np.zeros(theta.shape, dtype=np.complex128)
    first_integral = np.zeros(theta.shape, dtype=np.complex128) if debug else None
    second_integral = np.zeros(theta.shape, dtype=np.complex128) if debug else None

    prev_rhs_t = prev_rhs = None
    prev_first_t = prev_first = None
    prev_second_t = prev_second = None

    for sl in iter_time_chunks(strain_modes.data_len, chunk_size):
        t_chunk = strain_modes.time_axis[sl]
        news = evaluate_angular_chunk(
            news_modes, Grid=Grid, time_slice=sl, theta=theta, phi=phi, ell_max=ell_max
        )
        eth2_conj_news = evaluate_angular_chunk(
            eth2_conj_news_modes,
            Grid=Grid,
            time_slice=sl,
            theta=theta,
            phi=phi,
            ell_max=ell_max,
        )
        first_term = np.absolute(news) ** 2
        second_term = -eth2_conj_news.real
        rhs = first_term + second_term
        prev_rhs_t, prev_rhs = _accumulate_time_integral(
            rhs_integral, rhs, t_chunk, prev_rhs_t, prev_rhs
        )
        if debug:
            prev_first_t, prev_first = _accumulate_time_integral(
                first_integral, first_term, t_chunk, prev_first_t, prev_first
            )
            prev_second_t, prev_second = _accumulate_time_integral(
                second_integral, second_term, t_chunk, prev_second_t, prev_second
            )

    if debug:
        return ChunkedBalanceLawResult(
            rhs=rhs_integral, first_term=first_integral, second_term=second_integral
        )
    return rhs_integral


def lhs_balance_law(M_adm, M_final, v_kick, grid_info):
    """Compute the infinite-time balance-law LHS on the angular grid."""

    theta, phi = grid_info.meshgrid
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    if isinstance(v_kick, (list, tuple, np.ndarray)):
        vx, vy, vz = v_kick
    else:
        vx, vy, vz = v_kick.real, v_kick.imag, 0.0
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    v_dot_x = vx * st * cp + vy * st * sp + vz * ct
    lhs = M_adm - M_final / (gamma * (1.0 - v_dot_x)) ** 3
    return 4.0 * lhs


def balance_law_chunked(
    strain_modes,
    ginfo,
    M_adm,
    M_final,
    v_kick,
    chunk_size: int = 4096,
    ell_max: Optional[int] = None,
    debug: bool = False,
):
    """Compute the infinite-time balance-law residual with chunked RHS."""

    lhs = lhs_balance_law(M_adm, M_final, v_kick, ginfo) / (4.0 * np.pi)
    rhs_result = rhs_balance_law_from_modes_chunked(
        strain_modes, ginfo, chunk_size=chunk_size, ell_max=ell_max, debug=debug
    )
    if debug:
        from waveformtools.integrate import TwoDIntegral

        rhs = rhs_result.rhs / (4.0 * np.pi)
        first = rhs_result.first_term / (4.0 * np.pi)
        second = rhs_result.second_term / (4.0 * np.pi)
        delta = lhs - rhs
        info = {
            "lhs": lhs,
            "rhs": rhs,
            "lhs_int": TwoDIntegral(lhs, ginfo),
            "rhs_int": TwoDIntegral(rhs, ginfo),
            "res_int": TwoDIntegral(delta, ginfo),
            "first_term": first,
            "first_term_int": TwoDIntegral(first, ginfo),
            "second_term": second,
            "second_term_int": TwoDIntegral(second, ginfo),
            "chunk_size": chunk_size,
            "time_integrator": "trapezoid",
        }
        return delta, info
    rhs = rhs_result / (4.0 * np.pi)
    return lhs - rhs


def compare_chunked_to_full_rhs(strain_modes, Grid, chunk_size: int = 4096):
    """Compare chunked RHS to the legacy full-array RHS on a short waveform."""

    from waveform_balance_laws.laws import rhs_balance_law_from_modes

    rhs_full = rhs_balance_law_from_modes(strain_modes, Grid)
    rhs_chunked = rhs_balance_law_from_modes_chunked(
        strain_modes, Grid, chunk_size=chunk_size, debug=False
    )
    diff = rhs_full - rhs_chunked
    denom = max(np.linalg.norm(rhs_full), np.linalg.norm(rhs_chunked), 1e-300)
    return {
        "rhs_full": rhs_full,
        "rhs_chunked": rhs_chunked,
        "max_abs_error": np.max(np.abs(diff)),
        "relative_l2_error": np.linalg.norm(diff) / denom,
        "note": (
            "The legacy path uses spline time integration after full angular "
            "evaluation; the chunked path uses trapezoidal accumulation."
        ),
    }
