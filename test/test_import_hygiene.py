"""Tests for package import hygiene."""

from __future__ import annotations

import subprocess
import sys


def test_import_waveformtools_does_not_print_to_stdout():
    completed = subprocess.run(
        [sys.executable, "-c", "import waveformtools"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
