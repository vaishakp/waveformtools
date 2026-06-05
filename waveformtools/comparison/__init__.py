"""Waveform-mode comparison utilities.

Importing this package installs lightweight comparison methods on
``waveformtools.modes_array.ModesArray``:

- ``attach_metadata``
- ``get_comparison_metadata``
- ``match``
- ``mismatch``
- ``residue_distance``
- ``fitting_factor``
"""

from waveformtools.comparison.alignment import (
    AlignmentSpec,
    PreparedModeData,
    TimeAxisDiagnostics,
    interpolate_complex,
    overlap_interval,
    prepare_mode_data,
    reference_time,
    sampling_interval,
)
from waveformtools.comparison.config import (
    FittingFactorConfig,
    ModeComparisonConfig,
)
from waveformtools.comparison.conventions import (
    APPROXIMANT_MODE_CONVENTIONS,
    CANONICAL_MODE_CONVENTION,
    ModeConventionEntry,
    canonicalize_modes_for_comparison,
    mode_convention_for_approximant,
    standardize_generated_modes_in_place,
)
from waveformtools.comparison.core import (
    AlignedModeData,
    aligned_mode_arrays,
    available_modes,
    common_modes,
    mode_inner_product,
    mode_match,
    mode_mismatch,
    mode_norm,
    prepare_aligned_mode_data,
    residue_distance,
)
from waveformtools.comparison.fitting_factor import (
    fitting_factor,
    fixed_candidate_fitting_factor,
)
from waveformtools.comparison.metadata import (
    WaveformMetadata,
    attach_comparison_metadata,
    get_comparison_metadata,
)
from waveformtools.comparison.modes_api import install_modes_array_methods
from waveformtools.comparison.results import (
    ComparisonResult,
    FittingFactorResult,
)
from waveformtools.comparison.rotation import RotationSpec, rotate_modes

install_modes_array_methods()

__all__ = [
    "AlignmentSpec",
    "AlignedModeData",
    "APPROXIMANT_MODE_CONVENTIONS",
    "CANONICAL_MODE_CONVENTION",
    "ComparisonResult",
    "FittingFactorResult",
    "FittingFactorConfig",
    "ModeComparisonConfig",
    "ModeConventionEntry",
    "PreparedModeData",
    "RotationSpec",
    "TimeAxisDiagnostics",
    "WaveformMetadata",
    "aligned_mode_arrays",
    "attach_comparison_metadata",
    "available_modes",
    "canonicalize_modes_for_comparison",
    "common_modes",
    "fitting_factor",
    "fixed_candidate_fitting_factor",
    "get_comparison_metadata",
    "interpolate_complex",
    "mode_inner_product",
    "mode_match",
    "mode_mismatch",
    "mode_norm",
    "mode_convention_for_approximant",
    "overlap_interval",
    "prepare_mode_data",
    "prepare_aligned_mode_data",
    "reference_time",
    "residue_distance",
    "rotate_modes",
    "sampling_interval",
    "standardize_generated_modes_in_place",
]
