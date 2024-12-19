""" Centre of mass correction for the waveforms. """

#################
# Imports
#################

import numpy as np


def X_com_moments(time_axis, Xcom, order):
    """Compute the nth order temporal moment of the COM coordinates.

    Parameters
    ----------
    time_axis : 1d array
                The time axis.
    Xcom : list
           A list of three 1d arrays, each a 1d array containing the
           time series of the x, y and z co-ordinates in that order.
    order : int
            The order of the moment.

    Returns
    -------
    moments : list
              A list containing three real numbers, one each for the moment
              of x, y and z locations.

    """
    # Initial and final times
    time_i = time_axis[0]
    time_f = time_axis[-1]

    duration_t = time_f - time_i
    # Split the data
    x_all, y_all, z_all = Xcom

    # Interpolate
    from scipy.interpolate import interp1d

    x_all_int_fun = interp1d(
        time_axis,
        np.power(time_axis, order) * x_all / duration_t,
        kind="quadratic",
    )
    y_all_int_fun = interp1d(
        time_axis,
        np.power(time_axis, order) * y_all / duration_t,
        kind="quadratic",
    )
    z_all_int_fun = interp1d(
        time_axis,
        np.power(time_axis, order) * z_all / duration_t,
        kind="quadratic",
    )

    int_funcs = [x_all_int_fun, y_all_int_fun, z_all_int_fun]

    # Integrate
    from scipy.integrate import quad

    moments = {}
    labels = ["x", "y", "z"]

    count = 0
    # Find the moments
    for item in int_funcs:
        moment, err = quad(item, time_i, time_f)
        moments.update({labels[count]: [moment, err]})
        count += 1

    return moments


def compute_com_alpha(time_i, time_f, Xcom_0, Xcom_1):
    """Computes the CoM correction alpha parameter:
    the mean displacement of the system,
    of the COM correction as defined in
    Woodford et al. 2019 (Phys. Rev. D 100, 124010).

    Parameters
    ----------
    time_i : float
             initial time
    time_f : float
             final time
    Xcom_0 : list
             A list containing the zeroth order moments of the COM.
    Xcom_1 : list
             A list containing the first order moments of the COM.

    Returns
    -------
    com_alpha : list
                The list containig the alpha parameter vector

    """

    com_alpha = (
        4 * (time_f**2 + time_f * time_i + time_i**2) * np.array(Xcom_0)
        - 6 * (time_f + time_i) * np.array(Xcom_1)
    ) / (time_f - time_i) ** 2

    return com_alpha


def compute_com_beta(time_i, time_f, Xcom_0, Xcom_1):
    """Computes the CoM beta parameter: the mean drift of the system,
    of the COM correction as defined in
    Woodford et al. 2019 (Phys. Rev. D 100, 124010).

    Parameters
    ----------
    time_i: float
            initial time
    time_f: float
            final time
    Xcom_0: list
            A list containing the zeroth order moments of the COM.
    Xcom_1: list
            A list containing the first order moments of the COM.

    Returns
    -------
    com_beta: list
              The list containig the alpha parameter vector
    """

    com_beta = (12 * (Xcom_1) - 6 * (time_f + time_i) * Xcom_0) / (
        time_f - time_i
    ) ** 2

    return com_beta


def compute_conformal_k(vec_v, info, spin_phase=0):
    """Compute the conformal factor for the boost transformation
            :math:`k = \\exp(-2i \\lambda) \\gamma^3 (1 -
            \\mathbf{v} \\cdot \\mathbf{r})^3

    Parameters
    ----------
    vec_v: list
           A list of 2d arrays containing
           the velocity vector in the form
           [vec_x, vec_y, vec_z].

    spin_phase: float, optional
                The spin phase :math:`\\lambda'. Defaults to 0.

    info: obj
          An instance of the class `grids.sp_grid`
          that contains information about the
          spherical grid being used for the
          transformations.

    Returns
    -------

    conformal_k: 2d array
                 The conformal factor for the boost transformation
                 as defined above.


    """

    # unpack the velocity vector
    vel_x, vel_y, vel_z = vec_v

    # magnitude of velocity
    mag_v = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

    # Compute the 2d co-ordinate axis on the sphere
    theta, phi = info.meshgrid

    # compute the dot product
    v_dot_r = np.sin(theta) * (
        vel_x * np.cos(phi) + vel_y * np.sin(phi)
    ) + vel_z * np.cos(theta)

    # Lorentz factor
    gamma = 1.0 / np.sqrt(1 - mag_v**2)

    # spin_phase
    spin_factor = np.exp(-2 * 1j * spin_phase)

    # Finally, the conformal factor
    conformal_factor = spin_factor * np.power(gamma * (1 - v_dot_r), 3)

    return conformal_factor


def compute_translation_alpha_modes(time_axis, com_alpha, com_beta):
    """Compute the translation scalar :math:`\\alpha` in its spherical harmonic
    components given the mean motion of the centre of mass. These are basically
    the quantities in Eq. (4-5d) in the reference Woodford et al. 2019.

    Parameters
    ----------
    time_axis: 1d array
               The 1D array containing the time axis of the simulation.
    alpha: 1d array
           The 1D array containing the mean co-ordinate displacement
           of the COM of the system.
    beta: 1d array
          The 1D array containing the mean co-ordinate velocity of the COM.

    Returns
    -------

    alpha_modes: modes_array
                 A `waveforms.modes_array` object containing the SH
                 decomposition of the 'Alpha' supertranslation variable.

    """

    # Define the total displacement
    delta_t = 0
    delta_x = com_alpha[0] + com_beta[0] * time_axis
    delta_y = com_alpha[1] + com_beta[1] * time_axis
    delta_z = com_alpha[2] + com_beta[2] * time_axis

    Alpha_00 = np.sqrt(4 * np.pi) * delta_t
    Alpha_1m1 = -2 * np.sqrt(2 * np.pi / 3) * (delta_x + 1j * delta_y)
    Alpha_10 = -np.sqrt(4 * np.pi / 3) * delta_z
    Alpha_11 = -2 * np.sqrt(2 * np.pi / 3) * (-delta_x + 1j * delta_y)

    # Construct a mode array
    from waveformtools.waveforms import modes_array

    # Compute the data length
    data_len = len(time_axis)

    alpha_modes = modes_array(label="CoM alpha modes")

    alpha_modes._create_modes_array(ell_max=1, data_len=data_len)

    # l0m0
    alpha_modes.set_mode_data(ell_value=0, emm_value=0, data=Alpha_00)
    # l1mm1
    alpha_modes.set_mode_data(ell_value=1, emm_value=-1, data=Alpha_1m1)
    # l1m0
    alpha_modes.set_mode_data(ell_value=1, emm_value=0, data=Alpha_10)
    # l1m1
    alpha_modes.set_mode_data(ell_value=1, emm_value=1, data=Alpha_11)

    # Combine into one list
    # modes    = { 'l0' : [Alpha_00], 'l1' : [Alpha_1m1, Alpha_10, Alpha_11]}
    modes_list = [[0, [0]], [1, [-1, 0, 1]]]
    alpha_modes.modes_list = modes_list
    alpha_modes.time_axis = time_axis
    alpha_modes.ell_max = 1
    alpha_modes.spin_weight = -1
    return alpha_modes


def boost_waveform(unboosted_waveform, conformal_factor):
    """Boost the waveform given the unboosted waveform
    and the boost conformal factor.

    Parameters
    ----------
    unboosted_waveform: spherical_array
                        A class instance of `spherical array`.

    conformal_factor: 2d array
                      The conformal factor for the Lorentz transformation.
                      It may be a single floating point number or an array
                      on a spherical grid. The array will be of dimensions
                      [ntheta, nphi].

    gridinfo: class instance
              The class instance that contains the properties
              of the spherical grid.


    Returns
    -------
    boosted_waveform: sp_array
                      The class instance `sp_array` that
                      contains the boosted waveform.
    """

    from waveforms import spherical_array

    # Compute the meshgrid for theta and phi.
    # theta, phi = unboosted_waveform.gridinfo.meshgrid
    #   = unboosted_waveform.gridinfo.phi
    # A list to store the boosted waveform.
    boosted_waveform_data = []

    for item in unboosted_waveform.data:
        # Compute the boosted waveform on the spherical grid
        # on all the elements.

        # conformal_k_on_sphere = compute_conformal_k(vec_v, theta, phi)
        boosted_waveform_item = conformal_factor * item

        boosted_waveform_data.append(boosted_waveform_item)

    # Construct a 2d waveform array object
    boosted_waveform = spherical_array(
        gridinfo=unboosted_waveform.gridinfo,
        data=np.array(boosted_waveform_data),
    )
    boosted_waveform.label = "boosted"

    return boosted_waveform
