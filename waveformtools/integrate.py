""" Methods to integrate functions """

#################################################
# Imports
################################################
import numpy as np
import scipy
from spectools.fourier.transforms import compute_ifft
from spectools.fourier.transforms import compute_ifft
from waveformtools.waveformtools import message
from waveformtools.waveformtools import (
    get_starting_angular_frequency as sang_f,
)

from scipy.interpolate import InterpolatedUnivariateSpline

##################################################
# Fixed frequency integration
##################################################


def fixed_frequency_integrator(
    udata_time,
    delta_t,
    utilde_conven=None,
    freq_axis=None,
    omega0="auto",
    order=1,
    zero_mode=0,
    omega_threshold_factor=10,
):
    """Fixed frequency integrator as presented in Reisswig et. al.

    Parameters
    ----------
    udata_time:	1d array
                The input data in time.
    delta_t: float
             The time stepping.

    utilde_conven: 1d array, optional
                   The conventional FFT of the samples udata_time.
    freq_axis: 1darray, optional
               The frequency axis of the FFT. Must be supplied
               along with `utilde_conven`.
    omega0:	float, optional
            The cutoff angular frequency in the integration.
            Must be lower than the starting angular frequency
            of the input waveform. All frequencies whose absolute
            value is below this value will be neglected.
            The default cutoff-value is 0.

    order: int, optional
           The number of times to integrate
           the integrand in time. Defaults to 1.

    zero_mode: float, optional
               The zero mode amplitude of the FFT required.
               Defaults to 0 i.e. the zero mode is removed.

    Returns
    -------
    u_integ_n_time:	1d array
                    The input waveform in time-space,
                    integrated in frequency space using FFI.

    u_integ_integ_n: 1d array
                     The integrated u samples in Fourier space.

    """

    if omega0 == "auto":
        omega0 = max(
            abs(sang_f(udata_time, delta_t) / omega_threshold_factor), 1e-5
        )
        message(f"Using cutoff frequency {omega0}", message_verbosity=1)

    if not utilde_conven:
        # Compute the FFT of data
        from numpy.fft import ifft
        from spectools.fourier.transforms import compute_fft, unset_fft_conven
        from spectools.fourier.transforms import compute_fft, unset_fft_conven

        # from waveformtools import taper
        # udata_x_re = taper(u_time.real, delta_t=delta_t)
        # udata_x_im = taper(u_time.imag, delta_t=delta_t)
        # udata_x	   = np.array(udata_x_re) + 1j * np.array(udata_x_im)
        # x_axis = udata_x_re.sample_times
        # udata_x = np.array(udata_x)
        freq_axis, utilde_conven = compute_fft(udata_time, delta_t)

        if np.isnan(utilde_conven).any():
            message("Nan Found in utilde_conven!")

        if np.isnan(freq_axis).any():
            message("Nan Found in freq_axis!")

        # Find the length of the input data.
        Nlen = len(udata_time)

    else:
        Nlen = len(utilde_conven)
        assert (
            np.array(freq_axis) != np.array(None)
        ).all(), "Please supply the frequency axis along with utilde_conven"

    # df = np.diff(freq_axis)[0]
    df = scipy.stats.mode(np.diff(freq_axis))[0]
    message("df ", df, message_verbosity=3)

    # Find the location of the zero index.
    if (freq_axis == 0).any():
        zero_index = np.where(freq_axis == 0)[0][0]
    else:
        zero_index = None

    omega_axis = construct_ffi_omega_axis(freq_axis, omega0, zero_index)

    # Set the zero frequency element separately.
    utilde_conven[zero_index] = zero_mode

    # Integrate in frequency space
    utilde_integ_n = np.power((-1j / omega_axis), order) * utilde_conven

    if np.isnan(utilde_integ_n).any():
        message("Nan Found in utilde_integ_n!", message_verbosity=0)

    # Get the inverse fft
    # utilde_integ_n_orig = unset_fft_conven(utilde_integ_n)

    # u_integ_n_time = ifft(utilde_integ_n_orig)
    u_integ_n_time, u_integ = compute_ifft(utilde_integ_n, df)

    return u_integ_n_time, u_integ


def construct_ffi_omega_axis(freq_axis, omega0, zero_index):
    """Construct an angular frequency axis for
    use with FFI"""

    assert (
        omega0 > 0
    ), "Please supply a non-zero positive value of cutoff angular frequency"

    # Construct the angular frequency axis.
    omega_axis = 2 * np.pi * freq_axis
    # omega_axis[zero_index] = 1

    message(
        "The chosen cutoff angular frequency is", omega0, message_verbosity=2
    )
    message("Omega axis", omega_axis, message_verbosity=4)

    # Change the angular frequency if its magnitude is below a given omega0.
    for index, element in enumerate(omega_axis):
        # Loop over the samples.
        # Skip the zero index
        if index != zero_index:
            # print(freq_integ[index])
            # try:
            # Get the sign of the angular frequency.
            sign = int(element / abs(element))
            if abs(element) < omega0:
                omega_axis[index] = sign * omega0

        else:
            sign = 1
            assert (
                omega_axis[index] == 0
            ), f"The zero mode element must be zero frequency. Instead it is {omega_axis[index]}"
            omega_axis[index] = 1

    return omega_axis


#############################################
# 2D integrals
#############################################


def TwoDIntegral(func, grid_info, int_method=None):
    """Integrate a function over a sphere.

    Parameters
    ----------
    func: function
          The function to be integrated
    NTheta, NPhi: int
                  The number of grid points in the theta and phi directions.
                  Note that NTheta must be even.
    ht, hp: float
            The grid spacings.
    int_method: string
                The method to use for the integration.
                Options are DH (Driscoll Healy),
                SP (Simpson's),
                MP (Midpoint).

    Returns
    -------
    integ : float
            The function f integrated over the sphere.
    """

    if int_method is None:
        if grid_info.grid_type == "GL":
            int_method = "GL"
        elif grid_info.grid_type == "Uniform":
            int_method = "MP"
        else:
            raise KeyError(
                "Unable to discern default integration"
                "method due to unknown grid type. Please provide"
                "method explicitely"
            )

    if int_method == "DH":
        integral = DriscollHealy2DInteg(func, grid_info)
    elif int_method == "MP":
        integral = MidPoint2DInteg(func, grid_info)
    elif int_method == "SP":
        integral = Simpson2DInteg(func, grid_info)
    elif int_method == "GL":
        integral = GaussLegendre2DInteg(func, grid_info)

    else:
        raise ValueError("Unknown method!")
    return integral


def MidPoint2DInteg(func, info):
    """Evaulate the 2D surface integral using the midpoint rule.

    Parameters
    ----------
    func: ndarray
          The data to be integrated
    info: surface_grid_info
          An instance of the surface grid info class
          containing information about the grid.
    Returns
    -------
    integ: float
           The function f integrated over the sphere.
    """

    ht = info.dtheta
    hp = info.dphi

    theta_grid, _ = info.meshgrid

    integral = (
        np.tensordot(func, np.sin(theta_grid), axes=((-2, -1), (0, 1)))
        * ht
        * hp
    )

    return integral


def DriscollHealy2DInteg(func, info):
    """Implementation of the Driscoll Healy 2D integration that
    exhibits near spectools convergence.
    exhibits near spectools convergence.

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
    if len(func.shape) > 2:
        raise NotImplementedError(
            "Driscoll-Healy's method cannot currently handle 3d arrays"
        )

    NTheta = info.ntheta_act
    NPhi = info.nphi_act

    # NTheta, NPhi = func.shape
    theta_grid, _ = info.meshgrid

    ht = info.dtheta
    hp = info.dphi

    if NTheta < 0:
        raise ValueError("Ntheta is negative!")
    elif NPhi < 0:
        raise ValueError("Nphi is negative!")

    if (NTheta % 2) != 0:
        raise ValueError("NTheta must be even!")

    integrand_sum = 0.0

    func *= np.sin(theta_grid)

    # Skip the poles (ix=0 and ix=NTheta), as the weight there is zero
    # theta_1d = np.pi* np.arange(1, NTheta)/NTheta
    # ell_weight_axis = np.arange(int(NTheta/2))
    # theta_2d, ell_2d = np.meshgrid(theta_1d, ell_weight_axis)
    # weights_grid = (4/ np.pi) * np.sin((2 * ell_2d + 1) * theta_2d) /
    # (2 * ell_2d + 1)
    # weights_axis = np.sum(weights_grid, axis=1)
    # latitude_sum_axis = np.sum(func, axis=1)
    # integrand_axis = latitude_sum_axis * weights_axis

    for theta_index in range(1, NTheta):
        # These weights lead to an almost spectools convergence
        # These weights lead to an almost spectools convergence
        this_theta = np.pi * theta_index / NTheta

        # this_theta = theta_1d[theta_index]

        weight = 0

        # theta = M_PI * ix / NTheta;
        # weight = 0.0;
        for ell in range(int(NTheta / 2)):
            # for (int l = 0; l < NTheta/2; ++ l)
            weight += np.sin((2 * ell + 1) * this_theta) / (2 * ell + 1)

        # weight_axis = np.sin((2 * ell_weight_axis + 1)
        # * this_theta) / (2 * ell_weight_axis + 1)

        weight *= 4.0 / np.pi
        latitude_sum = 0

        # local_sum = 0.0;
        # Skip the last point (iy=NPhi), since we assume periodicity and
        # therefore it has the same value as the first point. We don't use
        # weights in this direction, which leads to spectools convergence.
        # weights in this direction, which leads to spectools convergence.
        # (Yay periodicity!)

        for index_phi in range(NPhi):
            # for (int iy = 0; iy < NPhi; ++ iy)
            latitude_sum += func[theta_index, index_phi]

        # latitude_sum = np.sum(func[index_theta, :])

        integrand_sum += weight * latitude_sum

    return ht * hp * integrand_sum


def DriscollHealy2DInteg_v2(func, info):
    """Implementation of the Driscoll Healy 2D integration that
    exhibits near spectools convergence.
    exhibits near spectools convergence.

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

    if len(func.shape) > 2:
        raise NotImplementedError(
            "Driscoll-Healy's method cannot currently handle 3d arrays"
        )

    NTheta = info.ntheta_act
    NPhi = info.nphi_act

    # NTheta, NPhi = func.shape
    theta_grid, _ = info.meshgrid

    ht = info.dtheta
    hp = info.dphi

    if NTheta < 0:
        raise ValueError("Ntheta is negative!")
    elif NPhi < 0:
        raise ValueError("Nphi is negative!")

    if (NTheta % 2) != 0:
        raise ValueError("NTheta must be even!")

    integrand_sum = 0.0

    # func*=np.sin(theta_grid)

    # Skip the poles (ix=0 and ix=NTheta), as the weight there is zero

    theta_1d = np.pi * np.arange(1, NTheta) / NTheta
    ell_weight_axis = np.arange(int(NTheta / 2))

    print("theta 1d", len(theta_1d))
    print("ell axis", len(ell_weight_axis))

    theta_2d, ell_2d = np.meshgrid(theta_1d, ell_weight_axis)
    print("T2d", theta_2d.shape)

    weights_grid = (
        (4 / np.pi) * np.sin((2 * ell_2d + 1) * theta_2d) / (2 * ell_2d + 1)
    )

    print("WG", weights_grid.shape)

    weights_axis = np.sum(weights_grid, axis=1)
    print("WG", weights_axis.shape)

    latitude_sum_axis = np.sum(func, axis=1)
    print("LSA", latitude_sum_axis.shape)

    integrand_axis = latitude_sum_axis * weights_axis

    integrand_sum = np.sum(integrand_axis)

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

    if len(func.shape) > 2:
        raise NotImplementedError(
            "Simpson's method cannot currently handle 3d arrays"
        )

    NTheta = info.ntheta_act
    NPhi = info.nphi_act

    theta_grid, _ = info.meshgrid
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

    func *= np.sin(theta_grid)

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
    func: ndarray
          The data to be integrated
    info: surface_grid_info
           An instance of the surface grid info class
           containing information about the grid.
    Returns
    -------
    integ: float
           The function f integrated over the sphere.
    """

    integral = (
        np.tensordot(func, info.weights_grid, axes=((-2, -1), (0, 1)))
        * info.dphi
    )
    return integral


def twod_time_integral(times, twod_func_ts, a=None, b=None):
    ''' Integrate a twoD array in time '''
    
    xdata = times
    if a is None:
        a = xdata[0]
    if b is None:
        b = xdata[-1]
        
    ntime, ntheta, nphi = twod_func_ts.shape
    ans = np.zeros((ntheta ,nphi), dtype=np.complex128)

    for th_idx in range(ntheta):
        for ph_idx in range(nphi):
            ydata = twod_func_ts[:, th_idx, ph_idx]
            yinterp_re = InterpolatedUnivariateSpline(xdata, ydata.real, k=5)
            yinterp_im = InterpolatedUnivariateSpline(xdata, ydata.imag, k=5)
            integral_re = yinterp_re.integral(a=a, b=b)
            integral_im = yinterp_im.integral(a=a, b=b)
            ans[th_idx, ph_idx]  = integral_re + 1j*integral_im

    return ans
