""" The implementation of BMS transformations on the waveforms. """

#############################
# Imports
#############################

import numpy as np

from waveformtools.waveformtools import message


def compute_conformal_k(vec_v, theta, phi, spin_phase=0):
    """Compute the conformal factor for the boost transformation
            :math:`k = \\exp(-2i \\lambda) \\gamma^3
            (1 - \\mathbf{v} \\cdot \\mathbf{r})^3`

    Parameters
    ----------
    vec_v : list
            The velocity vector.

    theta : float
            The polar angle :math:`\\theta' in radians.

    phi : float
          The azimuthal angle :math:`\\phi' in radians.

    spin_phase : float, optional
                 The spin phase :math:`\\lambda'. Defaults to 0.

    Returns
    -------
    conformal_k : float
                  The conformal factor for the
                  boost transformation as defined above.
    """

    # unpack the velocity vector
    vel_x, vel_y, vel_z = vec_v

    # magnitude of velocity
    mag_v = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
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


def compute_supertransl_alpha(supertransl_alpha_modes, theta, phi):
    """Compute the spherical Alpha supertranslation variable
    :math:`\\alpha(\\theta, \\phi)` given its modes. This method
    just multiplies the alpha modes with their corresponding spherical
    harmonic basis functions and returns the summed result.


    Parameters
    ----------
    supertransl_alpha_modes : dict
                              A dictionary of lists, each sublist
                              containing the set of super-translation
                              modes corresponding to a particular
                              :math:`\\ell'.
    theta :	float
            The polar angle :math:`\\theta'.
    phi : float
          The azimuthal angle :math:`\\phi'.

    Returns
    --------
    supertransl_alpha_sphere : func
                               A function on the sphere
                               (arguments :math:`\\theta', math:`\\phi').
    """

    # For partial evaluation of functions
    # from functools import partial
    message(supertransl_alpha_modes.keys())
    # Find the extreme ell values.
    keys_list = sorted(list(supertransl_alpha_modes.keys()))

    # ell_min = int(keys_list[0][1])
    # ell_max = int(keys_list[-1][1])
    # Import the Spherical Harmonic function
    from spectral.spherical.swsh import Yslm_vec

    spin_weight = 0
    # Ylm = partial(Yslm, spin_weight=0)
    # The final function
    supertransl_alpha_sphere = 0

    theta = np.pi / 2
    phi = 0.0
    for item in keys_list:
        ell = int(item[1])
        for m_index in range(2 * ell + 1):
            emm = m_index - ell
            message("ell is", ell, type(ell), "emm is ", emm)
            supertransl_alpha_sphere += supertransl_alpha_modes[item][
                m_index
            ] * Yslm_vec(spin_weight, ell, emm, theta, phi)

    return supertransl_alpha_sphere


def boost_waveform(unboosted_waveform, conformal_factor):
    """Boost the waveform given the unboosted waveform and
    the boost conformal factor.

    Parameters
    ----------
    non_boosted_waveform : list
                           A list with a single floating point number
                           or a numpy array of the unboosted waveform.
                           The waveform can have angular as well as
                           time dimentions.

                           The nesting order should be that, given the
                           list `non_boosted_waveform', each item in the
                           list refers to an array defined on the sphere
                           at a particular time or frequency. The subitem
                           will have dimensions [ntheta, nphi].



    conformal_factor : float/array
                       The conformal factor for the Lorentz transformation.
                       It may be a single floating point number or an array
                       on a spherical grid. The array will be of dimensions
                       [ntheta, nphi]

    gridinfo : class instance
               The class instance that contains the properties
               of the spherical grid.
    """

    # Find out if the unboosted waveform is a single number
    # or defined on a spherical grid.
    # onepoint = isinstance(unboosted_waveform[0], float)

    # if not onepoint:
    # Get the spherical grid shape.
    # 	ntheta, nphi = np.array(unboosted_waveform[0]).shape

    # Compute the meshgrid for theta and phi.
    # theta, phi = gridinfo.meshgrid

    # A list to store the boosted waveform.
    boosted_waveform = []

    for item in unboosted_waveform:
        # Compute the boosted waveform on the spherical grid
        # on all the elements.

        # conformal_k_on_sphere = compute_conformal_k(vec_v, theta, phi)
        boosted_waveform_item = conformal_factor * item

        boosted_waveform.append(boosted_waveform_item)

    return boosted_waveform
