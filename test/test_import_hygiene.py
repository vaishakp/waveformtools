"""Tests for package import hygiene."""

from __future__ import annotations

import subprocess
import sys


def _assert_imports_cleanly(module_name: str):
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module_name}; print('ok')"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"


def test_import_waveformtools_does_not_print_to_stdout():
    completed = subprocess.run(
        [sys.executable, "-c", "import waveformtools"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""


def test_quarantined_model_modules_import_cleanly():
    _assert_imports_cleanly("waveformtools.models.imr")
    _assert_imports_cleanly("waveformtools.models.bilby")


def test_eob_model_module_imports_without_pyseobnr_side_effects():
    _assert_imports_cleanly("waveformtools.models.eob")


def test_lal_model_module_imports_without_heavy_side_effects():
    script = """
import sys
import waveformtools.models.lal

for module_name in (
    "matplotlib.pyplot",
    "scipy.stats",
    "waveformtools.modes_array",
    "waveformtools.waveformtools",
    "pycbc.waveform",
    "pyseobnr",
):
    assert module_name not in sys.modules, module_name

print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"


def test_simulations_module_imports_cleanly():
    _assert_imports_cleanly("waveformtools.simulations")
