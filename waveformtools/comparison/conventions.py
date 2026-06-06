"""Mode-convention registry and canonicalization helpers for comparisons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from waveformtools.comparison.metadata import (
    attach_comparison_metadata,
    get_comparison_metadata,
)


CANONICAL_MODE_CONVENTION = "canonical_strain_lm"


@dataclass(frozen=True, slots=True)
class ModeConventionEntry:
    """Ledger entry describing how an approximant's raw modes are interpreted."""

    approximant: str
    raw_mode_convention: str
    canonical_transform: str = "identity"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        return asdict(self)


APPROXIMANT_MODE_CONVENTIONS: dict[str, ModeConventionEntry] = {
    "SEOBNRv5PHM": ModeConventionEntry(
        approximant="SEOBNRv5PHM",
        raw_mode_convention="pyseobnr_complex_conjugate_strain_lm",
        canonical_transform="complex_conjugate",
        note=(
            "The canonical comparison convention follows the raw NRSur7dq4 "
            "and IMRPhenomXPHM mode convention. Local pyseobnr "
            "SEOBNRv5PHM diagnostics show this approximant matches the "
            "canonical convention after complex conjugation."
        ),
    ),
    "SEOBNRv5HM": ModeConventionEntry(
        approximant="SEOBNRv5HM",
        raw_mode_convention="pyseobnr_complex_conjugate_strain_lm",
        canonical_transform="complex_conjugate",
        note=(
            "The canonical comparison convention follows the raw NRSur7dq4 "
            "and IMRPhenomXPHM mode convention. Local pyseobnr SEOBNRv5HM "
            "h22 diagnostics show this approximant matches the canonical "
            "convention after complex conjugation."
        ),
    ),
}


def mode_convention_for_approximant(
    approximant: str | None,
) -> ModeConventionEntry | None:
    """Return the exact approximant convention entry, if one is registered."""

    if approximant is None:
        return None
    return APPROXIMANT_MODE_CONVENTIONS.get(str(approximant))


def canonicalize_modes_for_comparison(
    modes: Any,
    *,
    enabled: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return modes in the canonical comparison convention plus diagnostics.

    The registry is keyed by exact approximant names. Unknown approximants are
    treated as already canonical and are left untouched.
    """

    metadata = get_comparison_metadata(modes)
    entry = mode_convention_for_approximant(metadata.approximant)
    already_canonical = metadata.mode_convention == CANONICAL_MODE_CONVENTION
    diagnostics: dict[str, Any] = {
        "enabled": bool(enabled),
        "approximant": metadata.approximant,
        "registered": entry is not None,
        "already_canonical": already_canonical,
        "raw_mode_convention": (
            entry.raw_mode_convention
            if entry is not None
            else metadata.mode_convention
        ),
        "canonical_mode_convention": CANONICAL_MODE_CONVENTION,
        "canonical_transform": "identity",
        "applied": False,
    }

    if (
        already_canonical
        or not enabled
        or entry is None
        or entry.canonical_transform == "identity"
    ):
        return modes, diagnostics

    if entry.canonical_transform == "complex_conjugate":
        canonical_modes = _complex_conjugate_modes(modes)
    else:
        raise ValueError(
            f"Unsupported mode canonicalization transform "
            f"{entry.canonical_transform!r} for {entry.approximant}."
        )

    metadata_dict = metadata.to_dict()
    metadata_dict.update(
        {
            "mode_convention": CANONICAL_MODE_CONVENTION,
            "raw_mode_convention": entry.raw_mode_convention,
            "mode_convention_source": entry.approximant,
            "canonicalization_applied": entry.canonical_transform,
        }
    )
    attach_comparison_metadata(canonical_modes, metadata_dict)
    diagnostics.update(
        {
            "canonical_transform": entry.canonical_transform,
            "applied": True,
            "entry": entry.to_dict(),
        }
    )
    return canonical_modes, diagnostics


def standardize_generated_modes_in_place(
    modes: Any,
    *,
    enabled: bool = True,
    stage: str = "waveform_generation",
) -> tuple[Any, dict[str, Any]]:
    """Standardize generated modes in-place and record provenance.

    This is intended for waveform-generation return paths. It applies the same
    canonical convention registry used by comparisons, but mutates the returned
    modes object so downstream balance-law diagnostics see standardized modes
    immediately.
    """

    metadata = get_comparison_metadata(modes)
    entry = mode_convention_for_approximant(metadata.approximant)
    already_canonical = metadata.mode_convention == CANONICAL_MODE_CONVENTION
    diagnostics: dict[str, Any] = {
        "enabled": bool(enabled),
        "stage": stage,
        "approximant": metadata.approximant,
        "registered": entry is not None,
        "already_canonical": already_canonical,
        "raw_mode_convention": (
            entry.raw_mode_convention
            if entry is not None
            else metadata.mode_convention
        ),
        "canonical_mode_convention": CANONICAL_MODE_CONVENTION,
        "canonical_transform": "identity",
        "applied": False,
        "in_place": True,
    }

    if (
        already_canonical
        or not enabled
        or entry is None
        or entry.canonical_transform == "identity"
    ):
        return modes, diagnostics

    if entry.canonical_transform == "complex_conjugate":
        _complex_conjugate_modes_in_place(modes)
    else:
        raise ValueError(
            f"Unsupported mode canonicalization transform "
            f"{entry.canonical_transform!r} for {entry.approximant}."
        )

    history_entry = {
        "stage": stage,
        "approximant": entry.approximant,
        "raw_mode_convention": entry.raw_mode_convention,
        "mode_convention": CANONICAL_MODE_CONVENTION,
        "canonical_transform": entry.canonical_transform,
        "mode_convention_source": entry.approximant,
    }
    metadata_dict = metadata.to_dict()
    history = list(metadata_dict.get("mode_convention_history") or [])
    history.append(history_entry)
    metadata_dict.update(
        {
            "mode_convention": CANONICAL_MODE_CONVENTION,
            "raw_mode_convention": entry.raw_mode_convention,
            "mode_convention_source": entry.approximant,
            "canonicalization_applied": entry.canonical_transform,
            "mode_convention_history": history,
        }
    )
    attach_comparison_metadata(modes, metadata_dict)
    _append_modes_action(
        modes,
        f"standardize_mode_convention({entry.canonical_transform})",
    )
    diagnostics.update(
        {
            "canonical_transform": entry.canonical_transform,
            "applied": True,
            "entry": entry.to_dict(),
        }
    )
    return modes, diagnostics


def _complex_conjugate_modes(modes: Any) -> Any:
    if hasattr(modes, "bar"):
        return modes.bar()

    copied = deepcopy(modes)
    if hasattr(copied, "_modes_data") and hasattr(modes, "modes_data"):
        copied._modes_data = np.conjugate(modes.modes_data)
        return copied
    raise TypeError(
        "Cannot complex-conjugate modes object without bar() or modes_data."
    )


def _complex_conjugate_modes_in_place(modes: Any) -> Any:
    if hasattr(modes, "_modes_data") and hasattr(modes, "modes_data"):
        modes._modes_data = np.conjugate(modes.modes_data)
        return modes
    raise TypeError("Cannot complex-conjugate modes object without modes_data.")


def _append_modes_action(modes: Any, action: str) -> None:
    existing = getattr(modes, "_actions", None)
    if existing is None:
        return
    setattr(modes, "_actions", f"{existing}->{action}")
