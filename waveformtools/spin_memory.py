"""Opt-in spin-memory helpers for waveform mode arrays.

Spin memory is sourced by the magnetic-parity (curl) part of the hard
angular-momentum flux carried by the news, in contrast to displacement
memory, which is sourced by the electric-parity energy flux ``|N|**2``.
This module implements the curl-flux source and the corresponding
spectral kernel, mirroring the displacement-memory API of
:mod:`waveformtools.memory`.

Conventions
-----------
Strain ``h`` has spin weight ``-2`` and news ``N = dh/du``.  The shear
tensor is ``C_AB = h m_A m_B + conj(h) mbar_A mbar_B``.  SWSH and eth
conventions follow spectools (Goldberg et al.):
``eth sYlm = +sqrt((l-s)(l+s+1)) (s+1)Ylm`` and
``ethbar sYlm = -sqrt((l+s)(l-s+1)) (s-1)Ylm``, so
``ethbar^2: s=0 -> s=-2`` has the per-``ell`` eigenvalue
``lam_l = sqrt((l-1)l(l+1)(l+2))``.

The hard (null) angular-momentum-aspect flux one-form is

``T_A = (1/16) D_A(N_BC C^BC) - (1/4) N^BC D_A C_BC
        - (1/4) D_B(C^BC N_CA - N^BC C_CA)``

[Flanagan & Nichols, PRD 95, 044002 (2017)].  Its spin ``+1`` component
``Tau = sqrt(2) m^A T_A`` reduces, via the dyad algebra and the product
rule ``N eth(conj(h)) = eth(N conj(h)) - conj(h) eth(N)``, to

``Tau = eth(Xi) + kappa (conj(N) eth(h) - conj(h) eth(N))``

with a complex spin-0 scalar ``Xi`` whose imaginary part is
``Im(Xi) = -2 kappa Im(N conj(h))``.  Only ``Im(Xi)`` and the
anti-electric part of the vector piece contribute to the curl.  The
value ``kappa = -1/2`` (i.e. ``Im(Xi) = Im(N conj(h))/2``) is fixed by
calibration: the ``l=1`` curl modes must reproduce the physical
angular-momentum flux, ``dJz/dt = sqrt(4*pi/3)/(16*pi) S_10`` with the
default ``'curl_flux'`` normalization (the ratio is time-independent to
machine precision, which pins the relative dyad algebra as well).
Decomposing ``T_A = D_A f + eps_AB D^B g`` with real ``f, g``, the
spin-0 modes of the curl potential are

``g_lm = [Im(Xi)]_lm
         + [V_lm - (-1)^m conj(V_{l,-m})] / (2i sqrt(l(l+1)))``

where ``V`` is the spin ``+1`` projection of the vector piece.  The
magnetic source modes are ``S_lm = -l(l+1) g_lm`` (the curl ``D^2 g``),
scaled by the configured normalization and sign.

The null spin-memory balance law [Nichols, PRD 95, 084048 (2017);
conventions as in Mitman et al., PRD 103, 024031 (2021)] is
``(1/8) D^2(D^2+2) Sigma_lm = integral of S_lm du`` with eigenvalue
``lam_l**2 / 8``, and the memory is placed as magnetic-parity strain
``h^B = i ethbar^2 Sigma``:

``(h^B_mem)_lm(u) = i (8/lam_l) conj( integral_{u0}^{u} S_lm du' )``.

Residual orientation/normalization freedom (the epsilon-orientation of
the dyad and the ``16*pi`` charge normalization) is exposed through
``source_sign`` and ``source_normalization``/``source_scale`` and pinned
by the ``l=1`` calibration against the angular-momentum flux
``dJz/dt`` of :func:`waveformtools.BMS.compute_angular_momentum_evolution`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np

from waveformtools.memory import (
    _angular_data_time_first,
    _bar_eth2_eigenvalue,
    _cumulative_trapezoid_zero_at_start,
    _fixed_point_memory_removal,
    _validate_compatible_memory_modes,
    _validate_grid,
)

SpinMemoryIntegrationConstant = Literal["zero_at_start"]
SpinMemoryMethod = Literal["spectral"]
SpinMemorySourceNormalization = Literal["curl_flux", "balance_law_16pi"]
SpinMemoryTimeProfile = Literal["cumulative_flux"]


@dataclass(slots=True)
class SpinMemoryConfig:
    """Configuration for opt-in spin-memory construction."""

    ell_min: int = 2
    ell_max: int | None = None
    memory_ell_max: int | None = None
    integration_constant: SpinMemoryIntegrationConstant = "zero_at_start"
    method: SpinMemoryMethod = "spectral"
    news_method: str = "spline"
    source_normalization: SpinMemorySourceNormalization = "curl_flux"
    source_scale: float = 1.0
    source_sign: float = 1.0
    time_profile: SpinMemoryTimeProfile = "cumulative_flux"
    removal_tolerance: float = 1e-10
    removal_max_iterations: int = 25

    def __post_init__(self) -> None:
        self.ell_min = int(self.ell_min)
        if self.ell_min < 2:
            raise ValueError("ell_min must be at least 2 for spin memory.")
        if self.ell_max is not None:
            self.ell_max = int(self.ell_max)
            if self.ell_max < self.ell_min:
                raise ValueError(
                    "ell_max must be greater than or equal to ell_min."
                )
        if self.memory_ell_max is not None:
            self.memory_ell_max = int(self.memory_ell_max)
            if self.memory_ell_max < 2:
                raise ValueError("memory_ell_max must be at least 2.")
        if self.integration_constant != "zero_at_start":
            raise ValueError(
                "Only integration_constant='zero_at_start' is supported."
            )
        if self.method != "spectral":
            raise ValueError("Only method='spectral' is supported.")
        if self.source_normalization not in {
            "curl_flux",
            "balance_law_16pi",
        }:
            raise ValueError(
                "source_normalization must be 'curl_flux' or "
                "'balance_law_16pi'."
            )
        self.source_scale = float(self.source_scale)
        if not np.isfinite(self.source_scale):
            raise ValueError("source_scale must be finite.")
        self.source_sign = float(self.source_sign)
        if not np.isfinite(self.source_sign):
            raise ValueError("source_sign must be finite.")
        if self.time_profile != "cumulative_flux":
            raise ValueError(
                "Only time_profile='cumulative_flux' is supported."
            )
        self.removal_tolerance = float(self.removal_tolerance)
        if (
            not np.isfinite(self.removal_tolerance)
            or self.removal_tolerance <= 0.0
        ):
            raise ValueError(
                "removal_tolerance must be finite and positive."
            )
        self.removal_max_iterations = int(self.removal_max_iterations)
        if self.removal_max_iterations < 1:
            raise ValueError("removal_max_iterations must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        return asdict(self)

    @classmethod
    def from_value(
        cls,
        value: "SpinMemoryConfig | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "SpinMemoryConfig":
        """Construct a spin-memory config from a dataclass, mapping, or ``None``."""

        if value is None:
            data: dict[str, Any] = {}
        elif isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "spin-memory config must be a SpinMemoryConfig, a mapping, "
                f"or None; got {type(value)!r}."
            )
        data.update(
            {key: val for key, val in overrides.items() if val is not None}
        )
        return cls(**data)


def compute_spin_memory_from_strain(
    strain_modes: Any,
    config: SpinMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return spin-memory strain modes for an input strain waveform."""

    memory_config = SpinMemoryConfig.from_value(config, **overrides)
    _validate_strain_modes(strain_modes, memory_config)
    news_modes = strain_modes.get_news_from_strain(
        method=memory_config.news_method
    )
    return compute_spin_memory_from_news(
        news_modes,
        strain_modes=strain_modes,
        config=memory_config,
    )


def compute_spin_memory_from_news(
    news_modes: Any,
    strain_modes: Any | None = None,
    config: SpinMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return spin-memory strain modes for input news modes.

    The curl-flux source is bilinear in the strain and the news, so both
    are required.  If ``strain_modes`` is omitted, the strain is
    reconstructed as the cumulative time integral of the news with
    ``h(u0) = 0``; the discarded integration constant is a constant
    shear offset that feeds the ``conj(h) eth(N)``-type terms, so pass
    the actual strain whenever it is available.
    """

    memory_config = SpinMemoryConfig.from_value(config, **overrides)
    _validate_news_modes(news_modes, memory_config)
    memory_ell_max = _memory_ell_max(news_modes, memory_config)
    source_modes = compute_spin_memory_source_from_news(
        news_modes,
        strain_modes=strain_modes,
        config=memory_config,
    )
    memory_modes = _source_modes_to_spin_memory_strain(
        source_modes,
        memory_config,
        memory_ell_max=memory_ell_max,
    )
    _record_spin_memory_metadata(memory_modes, memory_config, {})
    return memory_modes


def compute_spin_memory_source_from_strain(
    strain_modes: Any,
    config: SpinMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return the magnetic-parity source modes for an input strain."""

    memory_config = SpinMemoryConfig.from_value(config, **overrides)
    _validate_strain_modes(strain_modes, memory_config)
    news_modes = strain_modes.get_news_from_strain(
        method=memory_config.news_method
    )
    return compute_spin_memory_source_from_news(
        news_modes,
        strain_modes=strain_modes,
        config=memory_config,
    )


def compute_spin_memory_source_from_news(
    news_modes: Any,
    strain_modes: Any | None = None,
    config: SpinMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return scalar modes of the magnetic-parity (curl) flux source.

    The returned object has ``spin_weight=0`` and contains
    ``S_lm(u) = -l(l+1) g_lm(u)`` scaled by the configured normalization
    and sign, where ``g`` is the curl potential of the hard
    angular-momentum flux one-form.  The ``l=1`` modes carry the
    instantaneous angular-momentum flux (used for calibration); spin
    memory itself is sourced by ``l >= 2``.
    """

    memory_config = SpinMemoryConfig.from_value(config, **overrides)
    _validate_news_modes(news_modes, memory_config)
    _validate_grid(news_modes)
    if strain_modes is None:
        strain_modes = _strain_from_news(news_modes)
    else:
        _validate_strain_modes(strain_modes, memory_config)
        if not np.allclose(strain_modes.time_axis, news_modes.time_axis):
            raise ValueError(
                "strain_modes must use the same time axis as news_modes."
            )

    ell_max = _source_ell_max(news_modes, memory_config)
    projection_ell_max = _source_projection_ell_max(
        news_modes,
        memory_config,
        ell_max,
    )

    h_grid = _angular_data_time_first(
        strain_modes.evaluate_angular(ell_max=ell_max), strain_modes
    )
    n_grid = _angular_data_time_first(
        news_modes.evaluate_angular(ell_max=ell_max), news_modes
    )
    eth_h_grid = _angular_data_time_first(
        _eth_modes(strain_modes).evaluate_angular(ell_max=ell_max),
        strain_modes,
    )
    eth_n_grid = _angular_data_time_first(
        _eth_modes(news_modes).evaluate_angular(ell_max=ell_max),
        news_modes,
    )

    # Curl-relevant pieces of the flux one-form (module docstring):
    # gradient scalar Im(Xi) and the spin +1 vector piece V.  The
    # coefficients are calibrated so that with the default 'curl_flux'
    # normalization dJz/dt = sqrt(4*pi/3)/(16*pi) * S_10 against
    # BMS.compute_angular_momentum_evolution (time-independent ratio,
    # machine precision).
    nh_bar = n_grid * np.conjugate(h_grid)
    im_xi = 0.5 * np.imag(nh_bar)
    v_field = -0.5 * (
        np.conjugate(n_grid) * eth_h_grid
        - np.conjugate(h_grid) * eth_n_grid
    )

    im_xi_modes = _project_to_modes(
        im_xi.astype(np.complex128),
        news_modes,
        spin_weight=0,
        ell_max=projection_ell_max,
        label="spin_memory_gradient_scalar_time_domain",
    )
    v_modes = _project_to_modes(
        v_field,
        news_modes,
        spin_weight=1,
        ell_max=projection_ell_max,
        label="spin_memory_vector_piece_time_domain",
    )

    source_factor = (
        memory_config.source_sign
        * _source_normalization_factor(memory_config)
    )
    source_modes = _new_modes_array(
        news_modes,
        label="spin_memory_source_time_domain",
        ell_max=projection_ell_max,
        spin_weight=0,
    )
    for ell in range(1, projection_ell_max + 1):
        laplacian_eigenvalue = float(ell * (ell + 1))
        for emm in range(-ell, ell + 1):
            v_lm = np.asarray(v_modes.mode(ell, emm))
            v_lmm = np.asarray(v_modes.mode(ell, -emm))
            anti_electric = (
                v_lm - (-1.0) ** emm * np.conjugate(v_lmm)
            ) / (2.0j * np.sqrt(laplacian_eigenvalue))
            g_lm = np.asarray(im_xi_modes.mode(ell, emm)) + anti_electric
            source_modes.set_mode_data(
                ell=ell,
                emm=emm,
                data=-laplacian_eigenvalue * source_factor * g_lm,
            )
    setattr(
        source_modes,
        "spin_memory_source_metadata",
        {
            "included": True,
            "config": memory_config.to_dict(),
            "source": _source_description(memory_config),
            "source_factor": source_factor,
            "implementation": "curl_flux_projection",
        },
    )
    return source_modes


def with_spin_memory(
    strain_modes: Any,
    memory_modes: Any | None = None,
    config: SpinMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return a copy of ``strain_modes`` with spin memory added."""

    if memory_modes is None:
        memory_modes = compute_spin_memory_from_strain(
            strain_modes,
            config=config,
            **overrides,
        )
    _validate_compatible_memory_modes(strain_modes, memory_modes)
    out = strain_modes + memory_modes
    _record_spin_memory_metadata(out, config, overrides)
    return out


def add_spin_memory_in_place(
    strain_modes: Any,
    memory_modes: Any | None = None,
    config: SpinMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Add spin memory to ``strain_modes`` in place and return it."""

    with_memory = with_spin_memory(
        strain_modes,
        memory_modes=memory_modes,
        config=config,
        **overrides,
    )
    strain_modes._modes_data = with_memory.modes_data
    invalidate = getattr(
        strain_modes, "_invalidate_strain_derived_caches", None
    )
    if invalidate is not None:
        invalidate()
    _record_spin_memory_metadata(strain_modes, config, overrides)
    return strain_modes


def without_spin_memory(
    strain_modes: Any,
    memory_modes: Any | None = None,
    config: SpinMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Return a copy of ``strain_modes`` with spin memory removed.

    If ``memory_modes`` is given, the removal is the exact subtraction
    inverse of :func:`with_spin_memory` called with the same
    ``memory_modes``.  Otherwise the memory-free strain is found by
    fixed-point iteration ``h0 <- h - M(h0)`` starting from ``h0 = h``,
    where ``M`` is :func:`compute_spin_memory_from_strain` with the
    supplied config; the fixed point satisfies ``h0 + M(h0) = h``
    exactly, so the round trip ``without(with(h0)) == h0`` holds to
    ``removal_tolerance``.
    """

    memory_config = SpinMemoryConfig.from_value(config, **overrides)
    if memory_modes is not None:
        _validate_compatible_memory_modes(strain_modes, memory_modes)
        out = strain_modes - memory_modes
        removal_info: dict[str, Any] = {"mode": "exact_subtraction"}
    else:
        _validate_strain_modes(strain_modes, memory_config)
        out, removal_info = _fixed_point_memory_removal(
            strain_modes,
            lambda modes: compute_spin_memory_from_strain(
                modes,
                memory_config,
            ),
            tolerance=memory_config.removal_tolerance,
            max_iterations=memory_config.removal_max_iterations,
            quantity_label="spin memory",
        )
    _record_spin_memory_removal_metadata(out, memory_config, removal_info)
    return out


def remove_spin_memory_in_place(
    strain_modes: Any,
    memory_modes: Any | None = None,
    config: SpinMemoryConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Remove spin memory from ``strain_modes`` in place."""

    without_memory = without_spin_memory(
        strain_modes,
        memory_modes=memory_modes,
        config=config,
        **overrides,
    )
    strain_modes._modes_data = without_memory.modes_data
    invalidate = getattr(
        strain_modes, "_invalidate_strain_derived_caches", None
    )
    if invalidate is not None:
        invalidate()
    setattr(
        strain_modes,
        "spin_memory_metadata",
        getattr(without_memory, "spin_memory_metadata"),
    )
    return strain_modes


def _eth_modes(modes_obj: Any) -> Any:
    """Spin-raise a spin ``-2`` modes object to spin ``-1`` in mode space.

    Uses ``eth sYlm = +sqrt((l-s)(l+s+1)) (s+1)Ylm`` with ``s = -2``, so
    each mode is scaled by ``sqrt((l+2)(l-1))``.
    """

    out = _new_modes_array(
        modes_obj,
        label=f"{getattr(modes_obj, 'label', 'modes')}_eth_time_domain",
        ell_max=int(modes_obj.ell_max),
        spin_weight=-1,
    )
    for ell, emm_list in modes_obj.modes_list:
        eigenvalue = float(np.sqrt((ell + 2) * (ell - 1)))
        for emm in emm_list:
            out.set_mode_data(
                ell=ell,
                emm=emm,
                data=eigenvalue * np.asarray(modes_obj.mode(ell, emm)),
            )
    return out


def _strain_from_news(news_modes: Any) -> Any:
    """Reconstruct the strain as the ``zero_at_start`` integral of the news."""

    strain_modes = news_modes.deepcopy()
    strain_modes._modes_data = _cumulative_trapezoid_zero_at_start(
        np.asarray(news_modes.time_axis, dtype=float),
        np.asarray(news_modes.modes_data),
    )
    invalidate = getattr(
        strain_modes, "_invalidate_strain_derived_caches", None
    )
    if invalidate is not None:
        invalidate()
    return strain_modes


def _new_modes_array(
    template_modes: Any,
    label: str,
    ell_max: int,
    spin_weight: int,
) -> Any:
    from waveformtools.dataIO import construct_mode_list
    from waveformtools.modes_array import ModesArray

    time_axis = np.asarray(template_modes.time_axis, dtype=float)
    out = ModesArray(
        label=label,
        ell_max=ell_max,
        time_axis=time_axis,
        spin_weight=spin_weight,
        Grid=template_modes.Grid,
    )
    out.create_modes_array(ell_max=ell_max, data_len=len(time_axis))
    out.modes_list = construct_mode_list(ell_max, spin_weight=spin_weight)
    return out


def _project_to_modes(
    grid_data_time_first: np.ndarray,
    template_modes: Any,
    spin_weight: int,
    ell_max: int,
    label: str,
) -> Any:
    """Project time-first angular grid data onto SWSH modes.

    ``grid_data_time_first`` must be the spin ``+|spin_weight|``
    component when ``spin_weight`` is nonzero:
    :meth:`SphericalArray.to_modes_array` decomposes against
    ``abs(spin_weight)`` harmonics.
    """

    from waveformtools.spherical_array import SphericalArray

    time_axis = np.asarray(template_modes.time_axis, dtype=float)
    spherical = SphericalArray(
        label=label,
        time_axis=time_axis,
        data=np.moveaxis(grid_data_time_first, 0, -1),
        data_len=len(time_axis),
        Grid=template_modes.Grid,
        spin_weight=spin_weight,
        ell_max=ell_max,
    )
    projected = spherical.to_modes_array(
        Grid=template_modes.Grid,
        spin_weight=spin_weight,
        ell_max=ell_max,
    )
    projected._Grid = template_modes.Grid
    return projected


def _source_modes_to_spin_memory_strain(
    source_modes: Any,
    config: SpinMemoryConfig,
    memory_ell_max: int,
) -> Any:
    """Invert the spin-memory operator and place the magnetic strain.

    ``(1/8) D^2(D^2+2)`` has eigenvalue ``lam_l**2 / 8`` and the
    magnetic placement ``h^B = i ethbar^2 Sigma`` contributes another
    ``lam_l``, leaving the single net factor ``8 / lam_l``.
    """

    time_axis = np.asarray(source_modes.time_axis, dtype=float)
    memory_modes = _new_modes_array(
        source_modes,
        label="spin_memory_time_domain",
        ell_max=memory_ell_max,
        spin_weight=-2,
    )
    ell_min = max(2, config.ell_min)
    for ell, emm_list in memory_modes.modes_list:
        if ell < ell_min:
            continue
        eigenvalue = _bar_eth2_eigenvalue(ell)
        for emm in emm_list:
            integrated_source = _cumulative_trapezoid_zero_at_start(
                time_axis,
                np.asarray(source_modes.mode(ell, emm)),
            )
            memory_modes.set_mode_data(
                ell=ell,
                emm=emm,
                data=8.0j * np.conjugate(integrated_source) / eigenvalue,
            )
    return memory_modes


def _memory_ell_max(modes_obj: Any, config: SpinMemoryConfig) -> int:
    ell_max = int(getattr(modes_obj, "ell_max"))
    if config.memory_ell_max is None:
        return ell_max
    return config.memory_ell_max


def _source_ell_max(modes_obj: Any, config: SpinMemoryConfig) -> int:
    ell_max = int(getattr(modes_obj, "ell_max"))
    if config.ell_max is not None:
        ell_max = min(ell_max, config.ell_max)
    return ell_max


def _source_projection_ell_max(
    modes_obj: Any,
    config: SpinMemoryConfig,
    source_ell_max: int,
) -> int:
    if config.memory_ell_max is not None:
        return config.memory_ell_max
    grid_limit = int(getattr(modes_obj.Grid, "L", 2 * source_ell_max))
    return min(2 * source_ell_max, grid_limit)


def _source_normalization_factor(config: SpinMemoryConfig) -> float:
    if config.source_normalization == "curl_flux":
        base_factor = 1.0
    elif config.source_normalization == "balance_law_16pi":
        base_factor = 1.0 / (16.0 * np.pi)
    else:  # pragma: no cover - guarded by config validation.
        raise ValueError(
            f"Unknown source_normalization={config.source_normalization!r}."
        )
    return config.source_scale * base_factor


def _source_description(config: SpinMemoryConfig) -> str:
    if config.source_normalization == "curl_flux":
        base = "curl(T_A)"
    elif config.source_normalization == "balance_law_16pi":
        base = "curl(T_A)/(16*pi)"
    else:  # pragma: no cover - guarded by config validation.
        base = f"curl(T_A)[{config.source_normalization}]"
    prefix = ""
    if config.source_sign != 1.0:
        prefix = f"{config.source_sign:g}*"
    if config.source_scale != 1.0:
        prefix = f"{prefix}{config.source_scale:g}*"
    return f"{prefix}{base}"


def _validate_strain_modes(
    strain_modes: Any,
    config: SpinMemoryConfig,
) -> None:
    _validate_modes_common(strain_modes, config)
    if int(getattr(strain_modes, "spin_weight", -2)) != -2:
        raise ValueError(
            "Spin memory requires spin_weight=-2 strain modes."
        )
    if not hasattr(strain_modes, "get_news_from_strain"):
        raise TypeError("strain_modes must provide get_news_from_strain().")


def _validate_news_modes(
    news_modes: Any,
    config: SpinMemoryConfig,
) -> None:
    _validate_modes_common(news_modes, config)
    if int(getattr(news_modes, "spin_weight", -2)) != -2:
        raise ValueError(
            "Spin memory requires spin_weight=-2 news modes."
        )


def _validate_modes_common(
    modes_obj: Any,
    config: SpinMemoryConfig,
) -> None:
    time_axis = np.asarray(getattr(modes_obj, "time_axis", None), dtype=float)
    if time_axis.ndim != 1 or len(time_axis) < 2:
        raise ValueError(
            "Spin memory requires a one-dimensional time axis."
        )
    if not np.all(np.isfinite(time_axis)):
        raise ValueError(
            "Spin memory requires finite time-axis values."
        )
    if not np.all(np.diff(time_axis) > 0.0):
        raise ValueError(
            "Spin memory requires a strictly increasing time axis."
        )
    ell_max = int(getattr(modes_obj, "ell_max"))
    if ell_max < config.ell_min:
        raise ValueError("Input modes do not contain the requested ell range.")


def _record_spin_memory_metadata(
    modes_obj: Any,
    config: SpinMemoryConfig | Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
) -> None:
    memory_config = SpinMemoryConfig.from_value(config, **dict(overrides))
    setattr(
        modes_obj,
        "spin_memory_metadata",
        {
            "included": True,
            "config": memory_config.to_dict(),
            "implementation": "curl_flux_Dop_inverse",
        },
    )


def _record_spin_memory_removal_metadata(
    modes_obj: Any,
    config: SpinMemoryConfig,
    removal_info: Mapping[str, Any],
) -> None:
    setattr(
        modes_obj,
        "spin_memory_metadata",
        {
            "included": False,
            "removed": True,
            "removal": dict(removal_info),
            "config": config.to_dict(),
            "implementation": "curl_flux_Dop_inverse",
        },
    )
