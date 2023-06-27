""" Methods to integrate functions """

#################################################
# Imports
################################################

import numpy as np

from waveformtools.waveformtools import message

##################################################
# Fixed frequency integration
##################################################


def fixed_frequency_integrator(
    udata_time, delta_t, utilde_conven=None, omega0=0, order=1, zero_mode=0
):
    """Fixed frequency integrator as presented in Reisswig et. al.


    Parameters
    ----------
    udata_time:	1d array
                            The input data in time.
    delta_t:	float
                            The time stepping.

    utilde_conven:		1d array, optional
                                            The conventional FFT of the samples udata_time.
    omega0:	float, optional
            The cutoff angular frequency in the integration.
            Must be lower than the starting angular frequency
            of the input waveform. All frequencies whose absolute
            value is below this value will be neglected.
            The default cutoff-value is 0.

    order:		int, optional
                The number of times to integrate
                the integrand in time. Defaults to 1.

    zero_mode:	float, optional
                            The zero mode amplitude of the FFT required.
                            Defaults to 0 i.e. the zero mode is removed.

    Returns
    -------

    u_integ_n_time:	1d array
                                            The input waveform in time-space, integrated in frequency space using FFI.

    u_integ_integ_n:	1d array
                                            The integrated u samples in Fourier space.

    """

    if not utilde_conven:
        # Compute the FFT of data
        from numpy.fft import ifft

        from waveformtools.transforms import compute_fft, unset_fft_conven

        # from waveformtools import taper
        # udata_x_re = taper(u_time.real, delta_t=delta_t)
        # udata_x_im = taper(u_time.imag, delta_t=delta_t)
        # udata_x	   = np.array(udata_x_re) + 1j * np.array(udata_x_im)
        # x_axis = udata_x_re.sample_times
        # udata_x = np.array(udata_x)
        freq_axis, utilde_conven = compute_fft(udata_time, delta_t)

        # Find the length of the input data.
        Nlen = len(udata_time)

    else:
        Nlen = len(utilde_conven)

    # Find the location of the zero index.
    if Nlen % 2 == 0:
        zero_index = int(Nlen / 2)
    else:
        zero_index = int((Nlen + 1) / 2)

    # Construct the angular frequency axis.
    omega_axis = 2 * np.pi * freq_axis

    # print("The chosen cutoff angular frequency is", omega0)

    if omega0 > 0:
        for index, element in enumerate(omega_axis):
            # Loop over the samples.

            # Skip the zero index
            if index != zero_index:
                # print(freq_integ[index])
                try:
                    # Get the sign of the angular frequency.
                    sign = int(element / abs(element))
                except Exception as excep:
                    message(excep)
                    sign = 1

                # print(sign)
                # Change the angular frequency if its magnitude is below a given omega0.
                if abs(element) < omega0:
                    omega_axis[index] = sign * omega0

    # Set the zero frequency element separately.
    if not zero_mode:
        utilde_conven[zero_index] = 0
    else:
        utilde_conven[zero_index] = zero_mode

    # Integrate in frequency space
    utilde_integ_n = np.power((-1j / omega_axis), order) * utilde_conven

    # Get the inverse fft
    utilde_integ_n_orig = unset_fft_conven(utilde_integ_n)

    u_integ_n_time = ifft(utilde_integ_n_orig)

    return u_integ_n_time, utilde_integ_n


#############################################
# 2D integrals
#############################################


def TwoDIntegral(func, info, method="DH"):
    """Integrate a function over a sphere.

    Parameters
    ----------
    func : function
            The function to be integrated
    NTheta, NPhi : int
             The number of grid points in the theta and phi directions.
             Note that NTheta must be even.
    ht, hp : float
             The grid spacings.
    method : string
             The method to use for the integration. Options are DH (Driscoll Healy), SP (Simpson's), MP (Midpoint).

    Returns
    -------
    integ : float
            The function f integrated over the sphere.
    """

    # NTheta = info.ntheta_act
    # NPhi  = info.nphi_act

    # NTheta, NPhi = func.shape

    # ht = info.dtheta
    # hp = info.dphi

    if method == "DH":
        integral = DriscollHealy2DInteg(func, info)
    elif method == "MP":
        integral = MidPoint2DInteg(func, info)
    elif method == "SP":
        integral = Simpson2DInteg(func, info)
    elif method == "GL":
        integral = GaussLegendre2DInteg(func, info)

    else:
        raise ValueError("Unknown method!")
    return integral


def MidPoint2DInteg(func, info):
    """Evaulate the 2D surface integral using the midpoint rule.

    Parameters
    ----------
    func : ndarray
           The data to be integrated
    info : surface_grid_info
            An instance of the surface grid info class containing information about the grid.
    Returns
    -------
     integ : float
            The function f integrated over the sphere.
    """

    ht = info.dtheta
    hp = info.dphi

    theta_grid, _ = info.meshgrid

    integral = np.sum(func) * ht * hp

    return integral


def DriscollHealy2DInteg(func, info):
    """Implementation of the Driscoll Healy 2D integration that
    exhibits near spectral convergence.

    Parameters
    ----------
    func : function
            The function to be integrated
    NTheta, NPhi : int
             The number of grid points in the theta and phi directions.
             Note that NTheta must be even.
    ht, hp : float
             The grid spacings.

    Returns
    -------
    integ : float
            The function f integrated over the sphere.
    """

    NTheta = info.ntheta_act
    NPhi = info.nphi_act

    # NTheta, NPhi = func.shape

    ht = info.dtheta
    hp = info.dphi

    if NTheta < 0:
        raise ValueError("N_x is negative!")
    elif NPhi < 0:
        raise ValueError("N_y is negative!")

    if (NTheta % 2) != 0:
        raise ValueError("NTheta must be even!")

    integrand_sum = 0.0

    # Skip the poles (ix=0 and ix=NTheta), as the weight there is zero

    for index_theta in range(1, NTheta):
        # These weights lead to an almost spectral convergence
        this_theta = np.pi * index_theta / NTheta
        weight = 0
        # CCTK_REAL const theta = M_PI * ix / NTheta;
        # CCTK_REAL weight = 0.0;
        for ell in range(int(NTheta / 2)):
            # for (int l = 0; l < NTheta/2; ++ l)
            weight += np.sin((2 * ell + 1) * this_theta) / (2 * ell + 1)

        weight *= 4.0 / np.pi
        latitude_sum = 0
        # CCTK_REAL local_sum = 0.0;
        # Skip the last point (iy=NPhi), since we assume periodicity and
        # therefore it has the same value as the first point. We don't use
        # weights in this direction, which leads to spectral convergence.
        # (Yay periodicity!)
        for index_phi in range(NPhi):
            # for (int iy = 0; iy < NPhi; ++ iy)
            latitude_sum += func[index_theta, index_phi]

        integrand_sum += weight * latitude_sum

    return ht * hp * integrand_sum


def Simpson2DInteg(func, info):
    """Implementation of Simpson's 2D integration
    scheme.

    Parameters
    ----------
    func : function
            The function to be integrated
    NTheta, NPhi : int
             The number of grid points in the theta and phi directions.
             Note that NTheta must be even.
    ht, hp : float
             The grid spacings.

    Returns
    -------
    integ : float
            The function f integrated over the sphere.
    """

    NTheta = info.ntheta_act
    NPhi = info.nphi_act

    # NTheta, NPhi = func.shape

    ht = info.dtheta
    hp = info.dphi

    integrand_sum = 0
    index_theta = 0
    index_phi = 0

    assert NTheta > 0
    assert NPhi > 0
    # assert(func.all())
    assert NTheta % 2 == 0
    assert NPhi % 2 == 0

    Px = int(NTheta / 2)
    Py = int(NPhi / 2)

    # Around corners
    integrand_sum += (
        func[0, 0]
        + func[NTheta - 1, 0]
        + func[0, NPhi - 1]
        + func[NTheta - 1, NPhi - 1]
    )

    # Arount edges
    for index_phi in range(1, Py):
        integrand_sum += (
            4 * func[0, 2 * index_phi - 1]
            + 4 * func[NTheta - 1, 2 * index_phi - 1]
        )

    # for (iy = 1; iy <= py-1; iy++)
    for index_phi in range(1, Py - 1):
        integrand_sum += (
            2 * func[0, 2 * index_phi] + 2 * func[NTheta - 1, 2 * index_phi]
        )

    # for (ix = 1; ix <= px; ix++)
    for index_theta in range(1, Px):
        integrand_sum += (
            4 * func[2 * index_theta - 1, 0]
            + 4 * func[2 * index_theta - 1, NPhi - 1]
        )

    # for (ix = 1; ix <= px-1; ix++)
    for index_theta in range(1, Px - 1):
        integrand_sum += (
            2 * func[2 * index_theta, 0] + 2 * func[2 * index_theta, NPhi - 1]
        )

    # In the Interiors
    # for (iy = 1; iy <= py; iy++)
    for index_phi in range(1, Py):
        # for (ix = 1; ix <= px; ix++)
        for index_theta in range(1, Px):
            integrand_sum += 16 * func[2 * index_theta - 1, 2 * index_phi - 1]

    # for (iy = 1; iy <= py-1; iy++)
    for index_phi in range(1, Py - 1):
        # for (ix = 1; ix <= px; ix++)
        for index_theta in range(Px):
            integrand_sum += 8 * func[2 * index_theta - 1, 2 * index_phi]

    # for (iy = 1; iy <= py; iy++)
    for index_phi in range(1, Py):
        # for (ix = 1; ix <= px-1; ix++)
        for index_theta in range(1, Px - 1):
            integrand_sum += 8 * func[2 * index_theta, 2 * index_phi - 1]

    # for (iy = 1; iy <= py-1; iy++)
    for index_phi in range(1, Py - 1):
        # for (ix = 1; ix <= px-1; ix++)
        for index_theta in range(1, Px - 1):
            integrand_sum += 4 * func[2 * index_theta, 2 * index_phi]

    return (1 / 9) * ht * hp * integrand_sum


def GaussLegendre2DInteg(func, info):
    """Evaulate the 2D surface integral using the Gauss-Legendre rule.

    Parameters
    ----------
    func : ndarray
           The data to be integrated
    info : surface_grid_info
            An instance of the surface grid info class containing information about the grid.
    Returns
    -------
     integ : float
            The function f integrated over the sphere.
    """

    # NTheta = info.ntheta_act
    # NPhi  = info.nphi_act

    # NTheta, NPhi = func.shape

    # ht = info.dtheta
    # hp = info.dphi

    # theta_grid, _ = info.meshgrid
    # Norm = 1#(info.L +1)/4
    integral = np.sum(func * info.weights_grid) * info.dphi

    return integral
