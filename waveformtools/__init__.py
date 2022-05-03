# Copyright (c) 2020, Vaishak Prasad
# See LICENSE file for details: <https://gitlab.com/vaishakp/waveformtools/-/blob/main/LICENSE>
"""Module for the analysis and handling of numerical relativity and gravitational waveform data.
Classes
-------
sim : Base class for NR data
    This is needed to retrieve and handle Numerical Relativity data.

"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:  # pragma: no cover
    import importlib_metadata

import sys

__version__ = importlib_metadata.version(__name__)
