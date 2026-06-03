"""Thin object API for attaching comparison methods to ``ModesArray``.

The existing ``ModesArray`` class is large and used throughout waveformtools.
For the first comparison PR we avoid invasive edits to that file and install
small forwarding methods when ``waveformtools.comparison`` is imported.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from waveformtools.comparison.core import mode_match, mode_mismatch, residue_distance
from waveformtools.comparison.metadata import (
    WaveformMetadata,
    attach_comparison_metadata,
    get_comparison_metadata,
)
from waveformtools.comparison.results import ComparisonResult, FittingFactorResult


def modes_attach_metadata(self: Any, metadata: WaveformMetadata | dict[str, Any] | None = None, **kwargs: Any) -> Any:
    """Attach comparison metadata to this modes object and return ``self``."""

    return attach_comparison_metadata(self, metadata, **kwargs)


def modes_get_comparison_metadata(self: Any) -> WaveformMetadata:
    """Return comparison metadata for this modes object."""

    return get_comparison_metadata(self)


def modes_match(self: Any, other: Any, **kwargs: Any) -> ComparisonResult:
    """Compute the fixed-frame mode match against another modes object."""

    return mode_match(self, other, **kwargs)


def modes_mismatch(self: Any, other: Any, **kwargs: Any) -> float:
    """Return ``1 - match`` against another modes object."""

    return mode_mismatch(self, other, **kwargs)


def modes_residue_distance(
    self: Any,
    other: Any,
    residue_function: Callable[[Any], np.ndarray],
    **kwargs: Any,
) -> ComparisonResult:
    """Compare residual fields computed from two modes objects."""

    return residue_distance(self, other, residue_function=residue_function, **kwargs)


def modes_fitting_factor(self: Any, *args: Any, **kwargs: Any) -> FittingFactorResult:
    """Placeholder for waveform-family fitting-factor optimization.

    The object API is reserved in this first PR, but full intrinsic/extrinsic
    optimization is intentionally deferred until the fixed-frame comparison,
    metadata, and convention scaffolding have tests.
    """

    raise NotImplementedError(
        "ModesArray.fitting_factor is reserved but not implemented in this "
        "first comparison-core PR. Use ModesArray.match for fixed-frame mode "
        "matches. The optimizer-backed fitting-factor implementation will be "
        "added in the next comparison PR."
    )


def install_modes_array_methods() -> None:
    """Install comparison methods onto ``waveformtools.modes_array.ModesArray``.

    The operation is idempotent.  It is called from ``waveformtools.comparison``.
    """

    from waveformtools.modes_array import ModesArray

    methods = {
        "attach_metadata": modes_attach_metadata,
        "get_comparison_metadata": modes_get_comparison_metadata,
        "match": modes_match,
        "mismatch": modes_mismatch,
        "residue_distance": modes_residue_distance,
        "fitting_factor": modes_fitting_factor,
    }
    for name, method in methods.items():
        current = getattr(ModesArray, name, None)
        if current is None or getattr(current, "__module__", None) == __name__:
            setattr(ModesArray, name, method)
