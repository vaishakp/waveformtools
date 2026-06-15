# Copyright (c) 2020, Vaishak Prasad
# See [LICENSE](https://github.com/vaishakp/waveformtools/blob/main/LICENSE) file for details.
"""
Module for the analysis and handling of numerical relativity and gravitational waveform data.

Classes
-------
sim: Base class for NR simulations data.
                This is needed to retrieve and handle Numerical Relativity data.
spherical_array:    A 2D data-type.
                    Stores and manages two-dimensional data on surfaces of spherical topology.
modes_array: A data-type.
             Handle and work with mode coefficients.

"""
import os

package_directory = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    "WAVEFORM_CONVENTIONS",
    "get_waveform_conventions",
    "read_git_version",
]


def get_waveform_conventions():
    """Return waveform convention descriptors without eager import overhead."""

    from waveformtools.conventions import get_waveform_conventions as _get

    return _get()


def __getattr__(name):
    if name == "WAVEFORM_CONVENTIONS":
        from waveformtools.conventions import WAVEFORM_CONVENTIONS

        return WAVEFORM_CONVENTIONS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def read_git_version():
    """Get the latest version number based on the last commit date,
    iff this is a git repo."""

    # print(package_directory)
    # Open the file
    vers = "-1"
    try:
        with open(
            package_directory + "/../public/version", "r", encoding="utf-8"
        ) as vers_file:
            vers = vers_file.readlines()[0]
    except Exception as excep:
        print(
            excep,
            "\n",
            "This is not a git repo! please use the version attribute instead!",
        )
    # with open(package_directory + "/../public/date.txt", "r") as vers_file:
    # vers = vers_file.read()[:10]

    return vers


__version__ = "2026.03.03.3"
