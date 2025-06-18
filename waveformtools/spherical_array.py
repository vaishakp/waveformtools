from copy import deepcopy
import sys

import h5py
import numpy as np
import math

from sympy import comp

from waveformtools import dataIO
from waveformtools.dataIO import (
    construct_mode_list,
    get_iteration_numbers_from_keys,
    sort_keys,
)
from spectools.spherical.grids import UniformGrid
from spectools.spherical.grids import UniformGrid
from spectools.spherical.swsh import Yslm_vec
from spectools.spherical.swsh import Yslm_vec
from waveformtools.waveformtools import interp_resam_wfs, message
from pathlib import Path
from waveformtools.integrate import fixed_frequency_integrator
from waveformtools.waveformtools import (
    get_starting_angular_frequency as sang_f,
)

from waveformtools.single_mode import SingleMode
from scipy.interpolate import InterpolatedUnivariateSpline
from waveformtools.BMS import compute_linear_momentum_contribution_from_news, compute_impulse_from_force




class SphericalArray:
    """A class for handling waveforms on a sphere.


    Attributes
    ----------
    label: str
            The label of the waveform data.
    time_axis: 1d array
                The time axis of the data.
    frequency_axis: 1d array
                     The frequency axis if the data
                     is represented in frequency domain.
    Grid: UniformGrid
                An instance of the `UniformGrid` class.
    data_len: int
               The length of the data along the time axis.

    Methods
    -------
    delta_t:
              Fetch the time stepping `delta_t`.
    to_modes_array:
                     Find the waveform expressed in the
                     SWSH basis.
    boost:
           Boost the waveform.
    supertranslate:
                    Supertranslate the waveform.


    Notes
    -----
    The modes array has the axis in the form
    (flattened_modes_1d, *shape of extra modes, )
    """

    def __init__(
        self,
        label=None,
        time_axis=None,
        frequency_axis=None,
        data=None,
        data_len=None,
        data_dir=None,
        file_name=None,
        Grid=None,
        spin_weight=-2,
        ell_max=8,
    ):
        self._label = label
        self._data_len = data_len
        # self.base_dir = base_dir  # The base directory containing the
        self._data = data
        self._file_name = file_name
        self._data_dir = data_dir
        self._time_axis = time_axis
        self._frequency_axis = frequency_axis
        self._Grid = Grid
        self._spin_weight = spin_weight
        self._ell_max = ell_max

    @property
    def label(self):
        return self._label

    @property
    def ell_max(self):
        return self._ell_max

    @property
    def time_axis(self):
        return self._time_axis

    @property
    def frequency_axis(self):
        return self._frequency_axis

    @property
    def data(self):
        return self._data

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def file_name(self):
        return self._file_name

    @property
    def spin_weight(self):
        return self._spin_weight

    @property
    def Grid(self):
        return self._Grid

    def delta_t(self, value=None):
        """Sets and returns the value of time stepping :math:`dt`.

        Parameters
        ----------
        value: float, optional
                The value of :math:`dt`
                to set to the attribute.

        Returns
        -------
        delta_t: float
                  Sets the attribute.
        """

        if not self._delta_t:
            if not value:
                try:
                    self._delta_t = self.time_axis[1] - self.time_axis[0]
                except Exception as ex:
                    message(
                        "Please input the value of `delta_t`"
                        "or supply the `time_axis` to the waveform.",
                        ex,
                    )
            else:
                self._delta_t = value

        return self._delta_t

    @property
    def data_len(self):
        """Returns the length of the time/frequency axis.

        Returns
        -------
        data_len: int
                   The length of the time/frequency axis.
        """

        if self._data_len is None:
            try:
                data_len = len(self.time_axis)

            except Exception as ex:
                data_len = len(self.frequency_axis)
                message(ex)

            self._data_len = data_len

        return self._data_len

    def to_modes_array(self, Grid=None, spin_weight=None, ell_max=8):
        """Decompose a given spherical array function on a sphere
        into Spin Weighted Spherical Harmonic modes.

        Parameters
        ----------
        spin_weight: int, optional
                      The spin weight of the waveform.
                      It defaults to -2 for a
                      gravitational waveform.
        ell_max: int, optional
                  The maximum value of the :math:`\\ell`
                  polar quantum number. Defaults to 8.
        Grid: class instance
                    The class instance that contains
                    the properties of the spherical grid.

        Returns
        -------
        waveforms_modes: modes_array
                          An instance of the `modes_array` class
                          containing the decomposed modes.

        Notes
        -----
        1. Assumes that the sphere on which
           this decomposition is carried out is so far out
           that the coordinate system is spherical polar
           on a round sphere.
        2. Assumes that the poper area is the same
           as its co-ordinate area.
        3. Ensure that the label of the
           input spherical array indicates whether
           it is a time domain data or frequency domain data.
        """

        if self.Grid is None:
            if Grid is None:
                message(
                    "Please specify the grid specs. Assuming a default"
                    " uniform grid.",
                    message_verbosity=1,
                )
                Grid = UniformGrid()

            self._Grid = Grid

        if self._ell_max is None:
            if ell_max is None:
                try:
                    self._ell_max = Grid.L + 1
                except Exception as ex:
                    message(
                        ex,
                        "\n ell_max not provided. Assuming 8",
                        message_verbosity=1,
                    )
                    self._ell_max = 8

            else:
                try:
                    ell_max_grid = Grid.L + 1

                    assert ell_max <= ell_max_grid, (
                        "This GL grid"
                        "can only decompose upto"
                        f"ell_max of {ell_max_grid}"
                    )

                    self._ell_max = ell_max

                except Exception as ex:
                    message(
                        ex,
                        "No ell_max constraints" "due to uniform grid",
                        message_verbosity=3,
                    )

        if spin_weight is None:
            if self.spin_weight is None:
                message("Please specify spin weight. Assuming 2")
                spin_weight = 2
                self.spin_weight = spin_weight

            else:
                spin_weight = self.spin_weight

        spin_weight = abs(spin_weight)
        # Compute the meshgrid for theta and phi.
        theta, phi = self.Grid.meshgrid

        # Create a modes array object

        # Create a modes list.
        modes_list = construct_mode_list(ell_max, spin_weight=spin_weight)

        if not self.label:
            label = "decomposed_time_domain"
        else:
            label = self.label

        # Create a mode array for the decomposed_waveform
        waveform_modes = modes_array(
            label=label, ell_max=ell_max, spin_weight=spin_weight
        )

        # Inherit the time or frequency axis.
        if "time" in label:
            # axis = self.time_axis
            waveform_modes.time_axis = self.time_axis
        else:
            # axis = self.frequency_axis
            waveform_modes.frequency_axis = self.frequency_axis

        # Create the modes_array
        waveform_modes.create_modes_array(
            ell_max=ell_max, data_len=self.data_len
        )
        waveform_modes.modes_list = modes_list
        # The area element on the sphere
        # Compute the meshgrid for theta and phi.
        theta, phi = self.Grid.meshgrid

        # sqrt_met_det = np.sin(theta)
        # sqrt_met_det = np.sqrt(np.power(np.sin(theta), 2))

        # darea = sqrt_met_det * Grid.dtheta * Grid.dphi

        modes_list = [item for item in modes_list if item[0] >= spin_weight]

        from waveformtools.integrate import TwoDIntegral

        for mode in modes_list:
            ell, all_emms = mode

            for emm in all_emms:
                # m value.
                # Spin weighted spherical harmonic function at (theta, phi)
                Ybasis_fun = np.conj(
                    Yslm_vec(
                        spin_weight,
                        ell=ell,
                        emm=emm,
                        theta_grid=theta,
                        phi_grid=phi,
                    )
                )

                # Ydarea = Ybasis_fun * darea
                # print(Ydarea.shape)
                # print(full_integrand)
                # Using quad
                # multipole_ell_emm = np.tensordot(
                #    self.data, Ydarea, axes=((0, 1), (0, 1))
                # )
                # print("Shape", Ybasis_fun.shape, self.data.shape)
                # integrand = np.tensordot(self.data,
                # Ybasis_fun, axes=((0, 1), (0, 1)))
                integrand = np.transpose(
                    np.transpose(self.data, (2, 0, 1)) * Ybasis_fun, (1, 2, 0)
                )

                # integrand = self.data * Ybasis_fun
                multipole_ell_emm = TwoDIntegral(integrand, self.Grid)
                # print(f'l{ell}m{emm}', multipole_ell_emm)
                waveform_modes.set_mode_data(
                    ell=ell, emm=emm, data=multipole_ell_emm
                )

        return waveform_modes

    def boost(self, conformal_factor, Grid=None):
        """Boost the waveform given the unboosted
        waveform and the boost conformal factor.

        Parameters
        ----------
        self: spherical_array
               A class instance of `spherical array`.

        conformal_factor: 2d array
                           The conformal factor for the
                           Lorentz transformation.
                           It may be a single floating point number
                           or an array on a spherical grid. The array
                           will be of dimensions [ntheta, nphi]

        Grid: class instance
                    The class instance that contains
                    the properties of the spherical grid.


        Returns
        -------
        boosted_waveform: sp_array
                           The class instance `sp_array` that
                           contains the boosted waveform.
        """

        from waveformtools.waveforms import spherical_array

        if Grid is None:
            Grid = self.Grid

        # Compute the boosted waveform
        # on the spherical grid
        # on all the elements.
        boosted_waveform_data = (
            np.transpose(self.data, (2, 0, 1)) * conformal_factor
        )
        # boosted_waveform_data = np.multiply(self.data,
        # conformal_factor[:, :, None])
        # boosted_waveform_data = boosted_waveform_item
        # boosted_waveform_data = np.array([np.transpose(item)
        # *conformal_factor
        # for item in np.transpose(self.data)])

        # Construct a 2d waveform array object
        boosted_waveform = spherical_array(
            Grid=Grid,
            data=np.transpose(np.array(boosted_waveform_data), (1, 2, 0)),
        )

        # Assign other attributes.
        boosted_waveform.label = "boosted " + self.label
        boosted_waveform.time_axis = self.time_axis

        return boosted_waveform

    def supertranslate(self, supertransl_alpha_sp, order=1):
        """Supertranslate the :math:`\\Psi_{4\\ell m}` waveform modes,
        given the, the supertranslation parameter and the order.

        Parameters
        ----------
        supertransl_alpha_modes: modes_array
                                  The modes_array containing the
                                  supertranslation mode coefficients.
        Grid: class instance
                    The class instance that contains
                    the properties of the spherical grid
                    using which the computations are
                    carried out.
        order: int
                The number of terms to use to
                approximate the supertranslation.

        Returns
        -------
        Psi4_supertransl: modes_array
                           A class instance containg the
                           modes of the supertranslated
                           waveform:math:`\\Psi_4`.
        """

        # Create a spherical_array to hold the supertranslated waveform
        Psi4_supertransl_sp = spherical_array(
            Grid=self.Grid,
            label="{} -> supertranslated time".format(self.label),
        )

        delta_t = float(self.delta_t())
        # Set the data.
        data = 0
        # data = self.data
        # Psi4_supertransl_sp.data = self.data
        # delta = 0
        # count = 0
        from waveformtools.differentiate import differentiate5_vec_numba

        for index in range(order):
            # print(f'{index} loop')
            dPsidu = self.data
            for dorder in range(index + 1):
                # print(f'differentiating {dorder+1} times')
                dPsidu = differentiate5_vec_numba(dPsidu, delta_t)

            message("Incrementing...")
            # delta = np.power(supertransl_alpha_sp.data, index+1)
            # * dPsidu / np.math.factorial(index+1)
            # print(delta/self.data)

            data += (
                np.power(supertransl_alpha_sp.data, index + 1)
                * dPsidu
                / math.factorial(index + 1)
            )  # delta

        data += self.data
        message("Equal to original waveform?", (data == self.data).all())

        Psi4_supertransl_sp.data = data
        Psi4_supertransl_sp.time_axis = self.time_axis
        message("Done.")
        return Psi4_supertransl_sp

    def load_qlm_data(
        self, data_dir=None, Grid=None, bh=0, variable="sigma"
    ):
        """Load the 2D shear data from h5 files.

        Parameters
        ----------
        file_name: str
                    The name of the file containing data.
        data_dir: str
                   The name of the directory containing data.
        Grid: class instance
                    An instance of the Grid class.
        bh: int
             The black hole number (0, 1 or 2)
        """

        if data_dir is None:
            if self.data_dir is None:
                print("Please supply the data directory!")
            else:
                data_dir = self.data_dir
        else:
            if self.data_dir is None:
                self.data_dir = data_dir

        if Grid is None:
            if self.Grid is None:
                message("Please supply the grid spec!")
            else:
                Grid = self.Grid
        else:
            if self.Grid is None:
                self.Grid = Grid
        # get the full path.

        file_name = f"qlm_{variable}[{bh}].xy.h5"

        full_path = self.data_dir + file_name

        # cflag = 0

        nghosts = Grid.nghosts
        ntheta = Grid.ntheta
        nphi = Grid.nphi

        # Open the modes file.
        with h5py.File(full_path, "r") as wfile:
            # Get all the mode keys.
            modes_keys_list = list(wfile.keys())
            # modes_keys_list = sorted(modes_keys_list)

            # Get the mode keys containing the data.
            modes_keys_list = [
                item for item in modes_keys_list if "it=" in item
            ]

            # Get the itaration numbers.
            iteration_numbers = sorted(
                get_iteration_numbers_from_keys(modes_keys_list)
            )
            # sargs = np.argsort(iteration_numbers)
            # iteration_numbers = iteration_numbers[sargs]
            modes_keys_list = sort_keys(modes_keys_list)
            # Construct the data array.

            data_array = []

            for key in modes_keys_list:
                # data_item = np.array(wfile[key])
                # print(data_item.shape)
                data_item = np.array(wfile[key])[
                    nghosts : nphi - nghosts, nghosts : ntheta - nghosts
                ]
                try:
                    data_item = data_item["real"] + 1j * data_item["imag"]
                except Exception as ex:
                    message(ex)
                    pass

                data_array.append(data_item)

        self.data = np.transpose(np.array(data_array), (2, 1, 0))

        self.iteration_axis = np.array(iteration_numbers)

        #########################################################
        # Load inv_coords data
        #########################################################

        inv_file_name = f"qlm_inv_z[{bh}].xy.h5"

        # get the full path.
        full_path = self.data_dir + inv_file_name

        # Open the modes file.
        with h5py.File(full_path, "r") as wfile:
            # Get all the mode keys.
            modes_keys_list = list(wfile.keys())
            # modes_keys_list = sorted(modes_keys_list)

            # Get the mode keys containing the data.
            modes_keys_list = [
                item for item in modes_keys_list if "it=" in item
            ]

            modes_keys_list = sort_keys(modes_keys_list)
            data_array = []

            for key in modes_keys_list:
                data_item = np.array(wfile[key])[
                    nghosts : nphi - nghosts, nghosts : ntheta - nghosts
                ]
                # data_item = data_item['real'] + 1j*data_item['imag']
                data_array.append(data_item)

        self.invariant_coordinates_data = np.transpose(
            np.array(data_array), (2, 1, 0)
        )

        #########################################################
        # Load metric determinant  data
        #########################################################

        twometric_qtt_file_name = f"qlm_qtt[{bh}].xy.h5"
        twometric_qtp_file_name = f"qlm_qtp[{bh}].xy.h5"
        twometric_qpp_file_name = f"qlm_qpp[{bh}].xy.h5"

        # set the full path.
        full_path = self.data_dir + twometric_qtt_file_name

        # Open the modes file.
        with h5py.File(full_path, "r") as wfile:
            # Get all the mode keys.
            modes_keys_list = list(wfile.keys())
            # modes_keys_list = sorted(modes_keys_list)

            # Get the mode keys containing the data.
            modes_keys_list = [
                item for item in modes_keys_list if "it=" in item
            ]

            modes_keys_list = sort_keys(modes_keys_list)

            qtt_data_array = []

            for key in modes_keys_list:
                data_item = np.array(wfile[key])[
                    nghosts : nphi - nghosts, nghosts : ntheta - nghosts
                ]
                # data_item = data_item['real'] + 1j*data_item['imag']
                qtt_data_array.append(data_item)

        qtt_data_array = np.array(qtt_data_array)
        qtt_data_array = np.transpose(qtt_data_array, (2, 1, 0))

        # set the full path.
        full_path = self.data_dir + twometric_qtp_file_name

        # Open the modes file.
        with h5py.File(full_path, "r") as wfile:
            # Get all the mode keys.
            modes_keys_list = list(wfile.keys())
            # modes_keys_list = sorted(modes_keys_list)

            # Get the mode keys containing the data.
            modes_keys_list = [
                item for item in modes_keys_list if "it=" in item
            ]

            modes_keys_list = sort_keys(modes_keys_list)

            qtp_data_array = []

            for key in modes_keys_list:
                data_item = np.array(wfile[key])[
                    nghosts : nphi - nghosts, nghosts : ntheta - nghosts
                ]
                # data_item = data_item['real'] + 1j*data_item['imag']
                qtp_data_array.append(data_item)

        qtp_data_array = np.array(qtp_data_array)
        qtp_data_array = np.transpose(qtp_data_array, (2, 1, 0))

        # set the full path.
        full_path = self.data_dir + twometric_qpp_file_name

        # Open the modes file.
        with h5py.File(full_path, "r") as wfile:
            # Get all the mode keys.
            modes_keys_list = list(wfile.keys())
            # modes_keys_list = sorted(modes_keys_list)

            # Get the mode keys containing the data.
            modes_keys_list = [
                item for item in modes_keys_list if "it=" in item
            ]

            modes_keys_list = sort_keys(modes_keys_list)

            qpp_data_array = []

            for key in modes_keys_list:
                data_item = np.array(wfile[key])[
                    nghosts : nphi - nghosts, nghosts : ntheta - nghosts
                ]
                # data_item = data_item['real'] + 1j*data_item['imag']
                qpp_data_array.append(data_item)

        qpp_data_array = np.array(qpp_data_array)
        qpp_data_array = np.transpose(qpp_data_array, (2, 1, 0))

        sqrt_met_det = np.sqrt(
            np.linalg.det(
                np.transpose(
                    np.array(
                        [
                            [qtt_data_array, qtp_data_array],
                            [qtp_data_array, qpp_data_array],
                        ]
                    ),
                    (2, 3, 4, 0, 1),
                )
            )
        )

        self.sqrt_met_det_data = sqrt_met_det

    def to_shear_modes_array(self, Grid=None, spin_weight=None, ell_max=8):
        """Decompose a given spherical array function on a sphere
        into Spin Weighted Spherical Harmonic modes.

        Parameters
        ----------
        spin_weight: int, optional
                      The spin weight of the waveform.
                      It defaults to -2 for
                      a gravitational waveform.
        ell_max: int, optional
                  The maximum value of the :math:`\\ell`
                  polar quantum number. Defaults to 8.
        Grid: class instance
                    The class instance that contains
                    the properties of the spherical grid.

        Returns
        -------
        waveforms_modes: modes_array
                          An instance of the `modes_array`
                          class containing the decomposed modes.

        Notes
        -----
        1. Assumes that the sphere on which
           this decomposition is carried out is so far out
           that the coordinate system is spherical polar
           on a round sphere.
        2. Assumes that the poper area is
           the same as its co-ordinate area.
        3. Ensure that the label of the input
           spherical array indicates whether
           it is a time domain data or
           frequency domain data.
        """

        if Grid is None:
            if self.Grid is None:
                message("Please specify the grid specs. Assuming defaults.")
                Grid = UniformGrid()
                self.Grid = Grid
            else:
                Grid = self.Grid

        if spin_weight is None:
            if self.spin_weight is None:
                message("Please specify spin weight. Assuming 2")
                spin_weight = 2
                self.spin_weight = spin_weight

            else:
                spin_weight = self.spin_weight

        spin_weight = abs(spin_weight)
        # Compute the meshgrid for theta and phi.
        theta, phi = Grid.meshgrid

        # Create a modes array object

        # Create a modes list.
        modes_list = construct_mode_list(ell_max, spin_weight=spin_weight)

        if not self.label:
            label = "decomposed_time_domain"
        else:
            label = self.label

        # Create a mode array for the decomposed_waveform
        waveform_modes = modes_array(
            label=label, ell_max=ell_max, spin_weight=spin_weight
        )

        # Inherit the time or frequency axis.
        if "time" in label:
            # axis = self.time_axis
            waveform_modes.time_axis = self.time_axis
        else:
            # axis = self.frequency_axis
            waveform_modes.frequency_axis = self.frequency_axis

        # Create the modes_array
        waveform_modes.time_axis = self.time_axis[:]
        # sargs = np.argsort(waveform_modes.time_axis)
        # message(sargs)
        waveform_modes.time_axis = waveform_modes.time_axis

        waveform_modes.create_modes_array(
            ell_max=ell_max, data_len=self.data_len
        )
        waveform_modes.modes_list = modes_list
        # The area element on the sphere
        # Compute the meshgrid for theta and phi.
        theta, phi = Grid.meshgrid

        phi = np.transpose(
            np.array([phi for index in range(len(self.time_axis))]), (1, 2, 0)
        )

        # sqrt_met_det = np.sin(theta)
        # sqrt_met_det = np.sqrt(np.power(np.sin(theta), 2))

        darea = self.sqrt_met_det_data * Grid.dtheta * Grid.dphi

        theta = np.emath.arccos(self.invariant_coordinates_data)

        modes_list = [item for item in modes_list if item[0] >= spin_weight]

        for mode in modes_list:
            ell, all_emms = mode

            for emm in all_emms:
                # m value.
                # message(f'Processing l{ell} m{emm}')
                # Spin weighted spherical harmonic function at (theta, phi)

                Ybasis_fun = np.conj(
                    Yslm_vec(
                        spin_weight=spin_weight,
                        ell=ell,
                        emm=emm,
                        theta_grid=theta,
                        phi_grid=phi,
                    )
                )
                # Ybasis_fun = np.array([np.conj(Yslm_vec(spin_weight=
                # spin_weight, ell=ell, emm=emm,
                # theta_grid=theta[:, :, index],
                # phi_grid=phi[:, :, index])) for index in
                # range(self.data_len)])
                # Ybasis_fun = np.transpose(Ybasis_fun, (1, 2, 0))
                # message('Ybasis_fun', Ybasis_fun.shape)
                Ydarea = Ybasis_fun * darea
                # message('Ydarea', Ydarea.shape)
                # message(full_integrand)
                # Using quad
                # message('self.data', self.data.shape)
                # multipole_ell_emm = np.tensordot(self.data, Ydarea,
                # axes=((0, 1), (0, 1)))
                multipole_ell_emm = np.sum(self.data * Ydarea, (0, 1))

                # message(f'l{ell}m{emm}', multipole_ell_emm)

                # message('multipole_ell_emm', multipole_ell_emm.shape)
                waveform_modes.set_mode_data(
                    ell=ell, emm=emm, data=multipole_ell_emm
                )

        return waveform_modes