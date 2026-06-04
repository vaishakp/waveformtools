"""Optional real-waveform balance-law check for displacement memory.

This test is skipped by default because it generates real LAL waveform modes
and evaluates an angular balance-law residual. Run it explicitly with

    NUMBA_CACHE_DIR=/tmp/waveformtools_numba_cache \
    WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 \
    pytest test/test_memory_real_balance_law.py -vv -s

The longer all-mode cases are additionally gated by
``WAVEFORMTOOLS_RUN_LARGE_MEMORY_TESTS=1`` and are intended for a machine with
about 64 GB of RAM.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_waveform]


def _run_real_waveform_tests() -> bool:
    return os.environ.get("WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS") == "1"


def _run_large_memory_tests() -> bool:
    return os.environ.get("WAVEFORMTOOLS_RUN_LARGE_MEMORY_TESTS") == "1"


def _memory_balance_law_parameters() -> dict[str, float | str | int]:
    return {
        "approximant": "NRSur7dq4",
        "mass1": 40.0,
        "mass2": 35.0,
        "spin1x": 0.08,
        "spin1y": -0.03,
        "spin1z": 0.25,
        "spin2x": -0.04,
        "spin2y": 0.02,
        "spin2z": -0.15,
        "distance": 400.0,
        "inclination": 0.7,
        "phi_ref": 0.2,
        "f_lower": 20.0,
        "f_ref": 20.0,
        "f_max": 512.0,
        "delta_t": 1.0 / 2048.0,
        "delta_f": 1.0 / 4.0,
        "ell_max": 2,
    }


def _large_memory_balance_law_parameters(
    approximant: str,
) -> dict[str, float | str | int]:
    params = _memory_balance_law_parameters()
    f_lower = 0.0 if approximant == "NRSur7dq4" else 15.0
    f_ref = 20.0 if approximant == "NRSur7dq4" else f_lower
    params.update(
        {
            "approximant": approximant,
            "f_lower": f_lower,
            "f_ref": f_ref,
            "delta_t": 1.0 / 4096.0,
            "ell_max": 4,
        }
    )
    return params


def test_real_waveform_memory_lowers_infinite_time_balance_law_residue():
    if not _run_real_waveform_tests():
        pytest.skip(
            "Set WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 to run this test."
        )

    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    pytest.importorskip("pyseobnr")

    from spectools.spherical.grids import GLGrid

    from waveformtools.chunked_balance_laws import balance_law_chunked
    from waveformtools.integrate import TwoDIntegral
    from waveformtools.models.lal import LALWaveformModel

    params = _memory_balance_law_parameters()
    grid = GLGrid(L=4)
    model = LALWaveformModel(parameters_dict=params)
    try:
        strain_modes = model.get_td_waveform_modes(dimensionless=True)
        strain_modes._Grid = grid
        e0 = model.get_corresponding_eob_hamiltonian(**params)
    except Exception as exc:
        pytest.skip(f"Real waveform/balance-law setup unavailable: {exc}")

    _require_real_modes(strain_modes)

    original_result = _balance_law_residue(
        strain_modes,
        e0,
        grid,
        balance_law_chunked,
        TwoDIntegral,
    )
    with_memory = strain_modes.with_displacement_memory()
    with_memory._Grid = grid
    memory_result = _balance_law_residue(
        with_memory,
        e0,
        grid,
        balance_law_chunked,
        TwoDIntegral,
    )

    print(
        "memory_balance_law_residue "
        f"original={original_result.rms:.8e} "
        f"with_memory={memory_result.rms:.8e} "
        f"ratio={memory_result.rms / original_result.rms:.8e} "
        f"l00_original={original_result.l00:.8e} "
        f"l00_with_memory={memory_result.l00:.8e} "
        f"l00_ratio={abs(memory_result.l00) / abs(original_result.l00):.8e}"
    )
    assert memory_result.rms < original_result.rms


@pytest.mark.parametrize("approximant", ["NRSur7dq4", "SEOBNRv5PHM"])
def test_large_real_waveform_memory_balance_law_diagnostics(approximant):
    if not _run_real_waveform_tests() or not _run_large_memory_tests():
        pytest.skip(
            "Set WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 and "
            "WAVEFORMTOOLS_RUN_LARGE_MEMORY_TESTS=1 to run this test."
        )

    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    pytest.importorskip("pyseobnr")

    from spectools.spherical.grids import GLGrid

    from waveformtools.chunked_balance_laws import balance_law_chunked
    from waveformtools.integrate import TwoDIntegral

    params = _large_memory_balance_law_parameters(approximant)
    grid = GLGrid(L=8)
    try:
        strain_modes, e0 = _generate_large_test_waveform_and_e0(
            approximant,
            params,
            grid,
        )
    except Exception as exc:
        pytest.skip(f"{approximant} balance-law setup unavailable: {exc}")

    _require_real_modes(strain_modes)
    original_result = _balance_law_residue(
        strain_modes,
        e0,
        grid,
        balance_law_chunked,
        TwoDIntegral,
        chunk_size=1024,
    )
    with_memory = strain_modes.with_displacement_memory()
    with_memory._Grid = grid
    memory_result = _balance_law_residue(
        with_memory,
        e0,
        grid,
        balance_law_chunked,
        TwoDIntegral,
        chunk_size=1024,
    )
    print(
        "large_memory_balance_law_residue "
        f"approximant={approximant} "
        f"n_times={strain_modes.data_len} "
        f"ell_max={strain_modes.ell_max} "
        f"original={original_result.rms:.8e} "
        f"with_memory={memory_result.rms:.8e} "
        f"ratio={memory_result.rms / original_result.rms:.8e} "
        f"l00_original={original_result.l00:.8e} "
        f"l00_with_memory={memory_result.l00:.8e} "
        f"l00_ratio={abs(memory_result.l00) / abs(original_result.l00):.8e}"
    )


def _balance_law_residue(
    strain_modes,
    e0,
    grid,
    balance_law_chunked,
    two_d_integral,
    chunk_size=512,
):
    news_modes = strain_modes.get_news_from_strain()
    energy_radiated = strain_modes.compute_energy_radiated(
        news_modes=news_modes
    )
    final_mass = e0 - energy_radiated
    kick = strain_modes.compute_kick(Mfinal=final_mass)
    residual = balance_law_chunked(
        strain_modes=strain_modes,
        ginfo=grid,
        M_adm=e0,
        M_final=final_mass,
        v_kick=kick,
        chunk_size=chunk_size,
        ell_max=strain_modes.ell_max,
    )
    residual_power = two_d_integral(np.abs(residual) ** 2, grid)
    return BalanceLawResidue(
        rms=float(np.sqrt(np.real(residual_power) / (4.0 * np.pi))),
        l00=_residual_l00(residual, grid),
    )


class BalanceLawResidue:
    def __init__(self, rms: float, l00: complex):
        self.rms = rms
        self.l00 = l00


def _residual_l00(residual, grid) -> complex:
    from waveformtools.spherical_array import SphericalArray

    spherical_residual = SphericalArray(
        label="residual_time_domain",
        time_axis=np.array([0.0]),
        data=residual[:, :, np.newaxis],
        data_len=1,
        Grid=grid,
        spin_weight=0,
        ell_max=0,
    )
    residual_modes = spherical_residual.to_modes_array(
        Grid=grid,
        spin_weight=0,
        ell_max=0,
    )
    return complex(residual_modes.mode(0, 0)[0])


def _require_real_modes(modes) -> None:
    time_axis = np.asarray(modes.time_axis, dtype=float)
    if len(time_axis) < 256:
        pytest.skip(f"Generated too few time samples: {len(time_axis)}")
    if not np.all(np.isfinite(time_axis)):
        pytest.skip("Generated non-finite time samples.")
    mode_power = 0.0
    for emm in range(-2, 3):
        try:
            data = np.asarray(modes.mode(2, emm), dtype=np.complex128)
        except Exception:
            continue
        mode_power += float(np.sum(np.abs(data) ** 2))
    if not np.isfinite(mode_power) or mode_power <= 0.0:
        pytest.skip("Generated no finite ell=2 mode power.")


def _generate_large_test_waveform_and_e0(approximant, params, grid):
    if approximant == "SEOBNRv5PHM":
        return _generate_eob_waveform_and_e0(params, grid)
    return _generate_lal_waveform_and_e0(params, grid)


def _generate_lal_waveform_and_e0(params, grid):
    from waveformtools.models.lal import LALWaveformModel

    model = LALWaveformModel(parameters_dict=params)
    strain_modes = model.get_td_waveform_modes(dimensionless=True)
    strain_modes._Grid = grid
    e0 = model.get_corresponding_eob_hamiltonian(**params)
    return strain_modes, e0


def _generate_eob_waveform_and_e0(params, grid):
    from waveformtools.models.eob import EOBWaveformModel

    model = EOBWaveformModel(parameters_dict=params)
    strain_modes = model.get_td_waveform_modes(
        dimensionless=True,
        L=int(params["ell_max"]),
    )
    strain_modes._Grid = grid
    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    total_mass = mass1 + mass2
    mu = (mass1 / total_mass) * (mass2 / total_mass)
    e0 = model.model.dynamics[0, 5] * mu
    return strain_modes, e0
