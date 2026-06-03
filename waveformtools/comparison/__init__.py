"""Waveform-mode comparison utilities.

Importing this package installs lightweight comparison methods on
``waveformtools.modes_array.ModesArray``:

- ``attach_metadata``
- ``get_comparison_metadata``
- ``match``
- ``mismatch``
- ``residue_distance``
- ``fitting_factor`` placeholder reserved for the optimizer-backed follow-up
"""

from waveformtools.comparison.core import (
    available_modes,
    common_modes,
    mode_inner_product,
    mode_match,
    mode_mismatch,
    mode_norm,
    residue_distance,
)
from waveformtools.comparison.metadata import (
    WaveformMetadata,
    attach_comparison_metadata,
    get_comparison_metadata,
)
from waveformtools.comparison.modes_api import install_modes_array_methods
from waveformtools.comparison.results import ComparisonResult, FittingFactorResult

install_modes_array_methods()

__all__ = [
    "ComparisonResult",
    "FittingFactorResult",
    "WaveformMetadata",
    "attach_comparison_metadata",
    "available_modes",
    "common_modes",
    "get_comparison_metadata",
    "mode_inner_product",
    "mode_match",
    "mode_mismatch",
    "mode_norm",
    "residue_distance",
]
