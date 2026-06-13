"""Tests for public waveform convention descriptors."""

from __future__ import annotations

import json

import waveformtools
from waveformtools.conventions import get_waveform_conventions


def test_public_waveform_conventions_are_json_serializable():
    conventions = get_waveform_conventions()

    encoded = json.dumps(conventions, sort_keys=True)

    assert "waveformtools.waveform_conventions" in encoded
    assert conventions["owner"] == "waveformtools"
    assert conventions["lal_mode_storage"]["fd_axis"]
    assert (
        conventions["comparison"]["canonical_mode_convention"]
        == "canonical_strain_lm"
    )


def test_waveform_conventions_are_exported_as_copies():
    first = waveformtools.get_waveform_conventions()
    second = waveformtools.get_waveform_conventions()

    first["owner"] = "mutated"

    assert second["owner"] == "waveformtools"
    assert waveformtools.WAVEFORM_CONVENTIONS["owner"] == "waveformtools"
