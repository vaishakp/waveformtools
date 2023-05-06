# Copyright (c) 2020, Vaishak Prasad
# See [LICENSE](https://gitlab.com/vaishakp/waveformtools/-/blob/main/LICENSE) file for details.

"""
Module for the analysis and handling of numerical relativity and gravitational waveform data.

Classes
-------
sim: Base class for NR simulations data.
				This is needed to retrieve and handle Numerical Relativity data.
spherical_array:	A 2D data-type.
					Stores and manages two-dimensional data on surfaces of spherical topology.
modes_array: A data-type.
			 Handle and work with mode coefficients.

"""


import os
import sys

package_directory = os.path.dirname(os.path.abspath(__file__))


def get_version():
    """Get the latest version number based on the last commit date."""

    # print(package_directory)
    # Open the file
    with open(package_directory + "/../public/date.txt", "r") as vers_file:
        vers = vers_file.read()[:10]

    return vers
