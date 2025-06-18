""" Methods to handle functions on a sphere. """

##################################
# Imports
#################################

import numpy as np
from spectools.spherical.swsh import Yslm_vec
from spectools.spherical.swsh import Yslm_vec


def decompose_in_SWSHs(
    waveform, gridinfo, spin_weight=-2, ell_max=8, emm_list="all"
):
    """Decompose a given function on a sphere
    in Spin Weighted Spherical Harmonics

    Parameters
    ----------

    waveform:       list
                    A list that contains as its items the waveform
                    defined on the sphere as an array of shape
                    [ntheta, nphi]. Each item in the list may denote
                    an instant of time or frequency.

    spin_weight:     int, optional
                    The spin weight of the waveform.
                    It defaults to -2 for a gravitational waveform.

    ell_max:    int, optional
                The maximum value of the :math:`\\ell'
                polar quantum number. Defaults to 8.

    gridinfo:       class instance
                    The class instance that contains
                    the properties of the spherical grid.


    Returns
    -------

    SWSH_coeffs:    list
                    The SWSH coefficients of the waveform.
                    It may be a list composed of a single
                    floating point number or a 1d array
                    (denoting time or frequency dimension).
                    The waveform can have angular as well
                    as time dimentions. The nesting order
                    will be that, given the list
                    `non_boosted_waveform', each
                    item refers to a one dimensional array in
                    time/ frequency of SWSH coefficients.


    Notes
    -----
    Assumes that the sphere on which this decomposition
    is carried out is so far out
    that the coordinate system is spherical polar
    and the poper area is the
    same as its co-ordinate area.

    """

    # Find out if the unboosted waveform is
    # a single number or defined on a spherical grid.
    onepoint = isinstance(waveform[0], float)

    if not onepoint:
        # Get the spherical grid shape.
        ntheta, nphi = np.array(waveform[0]).shape

    # Compute the meshgrid for theta and phi.
    theta = gridinfo.theta(ntheta=ntheta, nphi=nphi)

    # Compute the meshgrid for theta and phi.
    phi = gridinfo.phi(ntheta=ntheta, nphi=nphi)

    # decomposed_waveforms = {}

    multipoles_all = {}
    for item in waveform:
        # Integrate on the sphere for decomposition into SWSHs
        """This should be fixed with summation over ell"""

        # define m values.
        if emm_list == "all":
            emm_list = np.arange(-ell_max, ell_max + 1)

        # Convert input to arrays.

        integrand_data = np.array(item)

        # Step 1: Compute the surface integral.

        # Assign data vectors.
        # lpole  = {}

        # Check if data includes ghost zones or not.

        # Compute the meshgrid for theta and phi.
        theta = gridinfo.theta(ntheta=ntheta, nphi=nphi)

        # Compute the meshgrid for theta and phi.
        phi = gridinfo.phi(ntheta=ntheta, nphi=nphi)

        sqrt_met_det = np.sqrt(np.power(np.sin(theta), 2))

        integrand_ij = integrand_data

        darea = sqrt_met_det * gridinfo.dtheta * gridinfo.dphi

        for ell_index in range(ell_max):
            multipoles_ell = {}
            for emm_index in range(len(emm_list)):
                # Decompose into seperate m modes.

                # m value.
                emm_val = int(emm_list[emm_index])

                # Spin weighted spherical harmonic function at (theta, phi)
                Ybasis_fun = Yslm_vec(
                    spin_weight, ell_index, emm_val, theta, phi
                )

                # Integrate to obtain the multipole of order l.

                # Integration for real and imaginary parts of
                # the data separately.
                # Integrate the function

                # Using quad
                multipole_emm = quad_on_sphere(
                    integrand_ij * Ybasis_fun * darea, gridinfo
                )
                # multipole_emm  = np.sum(integrand_ij * Ybasis_fun * darea)

                multipoles_ell.update({emm_val: multipole_emm})
            multipoles_all.update({ell_index: multipoles_ell})

        # Return the computed multipole.
        return multipoles_all


def quad_on_sphere(integrand, gridinfo, kind="third"):
    """Integrate on a sphere using the scipy.quad method

    Parameters
    ----------
    integrand: 2d array
               The two dimensional integrand array defined on the sphere.

    info: class instance
          The class instance that contains
          the properties of the spherical grid.

    kind: str
          The interpolation order to use in integration.

    Returns
    -------
    final_integral: float
                    The given integrand integrated over the sphere.

    final_errs: float
                The accumulated errors.

    Notes
    -----
    Assumes that the sphere is a unit round sphere.
    """

    # Step 0: Get the grid properties

    # Compute the meshgrid for theta and phi.
    theta_1d = gridinfo.theta_1d
    # theta
    # Compute the meshgrid for theta and phi.
    phi_1d = gridinfo.phi_1d

    # imports
    from scipy.integrate import quad
    from scipy.interpolate import interp1d

    theta_first_integral_vals = []
    theta_first_integral_errs = []
    # Step 1: integrate along the theta direction
    for phi_index in range(gridinfo.nphi):
        # Interpolate the integrand.

        integrand_phi = integrand[:, phi_index]

        integrand_phi_interp_func = interp1d(theta_1d, integrand_phi, kind=kind)

        # Integrate on the phi plane
        integral_phi_vals, integral_phi_errs = quad(
            integrand_phi_interp_func, 0, np.pi
        )

        theta_first_integral_vals.append(integral_phi_vals)
        theta_first_integral_errs.append(integral_phi_errs)

    # Step 2: integrate along the phi direction

    # Interpolate the integrand.

    integrand_theta = theta_first_integral_vals

    integrand_theta_interp_func = interp1d(phi_1d, integrand_theta, kind=kind)

    # Integrate on the theta plane
    final_integral, semi_final_errs = quad(
        integrand_theta_interp_func, 0, 2 * np.pi
    )

    # Get final errors
    final_errs = semi_final_errs + np.sum(np.array(theta_first_integral_errs))

    return final_integral, final_errs
