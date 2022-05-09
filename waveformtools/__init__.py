# Copyright (c) 2020, Vaishak Prasad
# See LICENSE file for details: <https://gitlab.com/vaishakp/waveformtools/-/blob/main/LICENSE>
"""Module for the analysis and handling of numerical relativity and gravitational waveform data.
Classes
-------
sim : Base class for NR data
    This is needed to retrieve and handle Numerical Relativity data.

"""

# try:
#    import importlib.metadata as importlib_metadata
# except ModuleNotFoundError:  # pragma: no cover
#    import importlib_metadata

import os
import sys

import numpy as np

# __version__ = importlib_metadata.version(__name__)

package_directory = os.path.dirname(os.path.abspath(__file__))


def get_version():
    """ Get the latest version number based on the last commit date. 

	"""

    # print(package_directory)
    # Open the file
    with open(package_directory + "/../date.txt", "r") as vers_file:
        vers = vers_file.read()[:10]

    # print(vers)
    return vers
