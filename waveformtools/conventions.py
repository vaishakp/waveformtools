"""Public waveform convention descriptors.

This module is the lightweight metadata surface for downstream tools that need
to record waveformtools' mode, frame, and LAL-storage conventions without
inspecting implementation details.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from waveformtools.comparison.conventions import (
    APPROXIMANT_MODE_CONVENTIONS,
    CANONICAL_MODE_CONVENTION,
)


WAVEFORM_CONVENTIONS: dict[str, Any] = {
    "schema_name": "waveformtools.waveform_conventions",
    "schema_version": 1,
    "owner": "waveformtools",
    "mode_indexing": {
        "labels": ["ell", "m"],
        "ordering": "mode data are addressed by explicit (ell, m) labels",
        "negative_m_modes": "stored when supplied by the waveform source",
    },
    "angular_basis": {
        "basis": "spin-weighted spherical harmonics",
        "spin_weight": -2,
        "reconstruction": (
            "ModesArray.evaluate_angular(theta, phi) reconstructs strain as "
            "sum_lm stored[l,m] * _{-2}Y_lm(theta, phi)."
        ),
    },
    "lal_mode_storage": {
        "loader": "waveformtools.waveformtools.load_lal_modes_to_modes_array",
        "td_storage": "stored_td[l,m](t) = conjugate(raw_lal_mode_data)",
        "fd_storage": "stored_fd[l,m](f) = conjugate(raw_lal_mode_data) / N",
        "fd_axis": "LAL linked-list fdata axis is two-sided and sorted",
        "fd_positive_m_support": (
            "positive-m FD modes carry physical support on the negative-frequency "
            "side of the two-sided LAL linked-list representation"
        ),
        "fd_negative_m_support": (
            "negative-m FD modes carry physical support on the positive-frequency "
            "side of the two-sided LAL linked-list representation"
        ),
        "all_modes_policy": "store all linked-list modes, including negative m",
        "single_mode_helper_warning": (
            "LAL SphHarmFrequencySeriesGetMode returns a nonnegative-frequency "
            "helper grid and can miss positive-m inspiral content for FD modes"
        ),
    },
    "comparison": {
        "canonical_mode_convention": CANONICAL_MODE_CONVENTION,
        "registered_approximant_conventions": {
            name: entry.to_dict()
            for name, entry in APPROXIMANT_MODE_CONVENTIONS.items()
        },
    },
    "known_gaps": [
        (
            "Full detector-polarization sign and phase conventions for every "
            "external approximant are not yet encoded in this descriptor."
        ),
        (
            "Coprecessing/inertial-frame metadata should be supplied by each "
            "waveform model adapter when mode outputs are generated."
        ),
    ],
}


def get_waveform_conventions() -> dict[str, Any]:
    """Return a JSON-serializable copy of waveformtools convention metadata."""

    return deepcopy(WAVEFORM_CONVENTIONS)
