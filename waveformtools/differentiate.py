"""
Tools for differentiating data.

"""

#######################################################
# Imports
#######################################################

import matplotlib.pyplot as plt
import numpy as np

# from numba import jit, njit

from waveformtools.waveformtools import message


def derivative(x_data, y_data, method="FD", degree=3):
    """Compute the derivative of the y data w.r.t the x data
    using the specified method. x_data can be non-uniformly sampled
    in which case the derivative will be computed by resampling.

    Parameters
    ----------
    x_data, y_data: 1d array
                    The x and y data. x_data is assumed
                    to be sorted.
    method: str
            The method to use for differentiation.
            Presently supported values are
            'SP' : Spline
            'CS' : Chebyshev
            'FS' : Fourier
            'FD' : Finite difference
    degree: int
            The algorithm degree to use for differentiating. This
            is applicable when dealing with the 'CS'
            method, which is the number of basis functions
            to use and 'FD' method, which is the number
            of points on either side to use.

    Returns
    -------
    dydx: 1d array
          The first order derivative of y w.r.t. x.
    """

    if method == "spline":
        dydx = spline_differential(x_data, y_data, k=degree)
    elif method == "CS":
        dydx = Chebyshev_differential(x_data, y_data, order=1, degree=degree)
    else:
        delta_x_all = np.diff(x_data)

        if (delta_x_all - delta_x_all[0] < 1e-14).all():
            message("No interpolation required.")
            interp = False
            x_uniform = x_data
            y_uniform = y_data
        else:
            message("INterpolation required!")
            interp = True
            from scipy.interpolate import interp1d

            # Resample the axis
            x_uniform = np.linspace(x_data[0], x_data[-1], 2 * len(x_data))
            y_uniform = interp1d(x_data, y_data, kind="cubic")(x_uniform)

        delta_x = np.diff(x_uniform)[0]

        if method == "FS":
            dydx_new, _, x_new, _ = Fourier_differential(
                delta_x=delta_x,
                udata_x=y_uniform,
                order=1,
                zero_mode=0,
                taper=False,
            )

        elif method == "FD":
            if degree == 1:
                dydx_new = differentiate(y_uniform, delta_x)
            elif degree == 2:
                dydx_new = differentiate2(y_uniform, delta_x)
            elif degree == 3:
                dydx_new = differentiate3(y_uniform, delta_x)
            elif degree == 4:
                dydx_new = differentiate4(y_uniform, delta_x)
            elif degree == 5:
                dydx_new = differentiate5(y_uniform, delta_x)
            else:
                raise NotImplementedError(f"Unknown degree {degree}")

        elif method == "FD_vec":
            dydx_new = differentiate5_vec_nonumba(y_uniform, delta_x)

        else:
            raise NotImplementedError(f"Unknown method {method}")

        if interp:
            dydx = interp1d(x_uniform, dydx_new, kind="cubic")(x_data)
        else:
            dydx = dydx_new

    return dydx


#######################################################
# Spline differentiation
########################################################


def spline_differential(x_data, y_data, k=5):
    """Spline differentiation

        Parameters
        ----------
    x_data, y_data : 1d array
                     The x and y data. x_data is assumed
                     to be sorted.

        k : int
                The interpolation order

        Returns
        -------
    dydx : 1d array
           The first order derivative of y w.r.t. x.
    """

    from scipy.interpolate import InterpolatedUnivariateSpline as interpolator

    interp = interpolator(x_data, y_data, k=k)

    dydx = interp.derivative()(x_data)

    return dydx


########################################################
# Chebyshev differentiation
########################################################


def Chebyshev_differential(x_data, y_data, order=1, degree=8):
    """Differentiate a function using the Chebyshev expansion.


    Parameters
    ----------
    x_data: 1d array
            The x data.
    y_data: 1d array
            The y data.
    order: int
           The number of times to differentiate.

    degree: int
            The number of basis functions to use.

    Returns
    -------
    dydx_data: 1d array
               The differentiated data.

    """

    # Find the basis coefficients.
    from numpy.polynomial.chebyshev import chebder, chebfit, chebval

    # L2errs = []
    # p_res = 1e21
    # for deg_index in range(degree):
    #   cheb_coeffs, result = chebfit(x_data, y_data, deg=deg_index, full=True)
    #   res = result[0][0]
    # if res%2==0:
    #   L2errs.append(res)
    # print(x_data, y_data)
    cheb_coeffs, result = chebfit(x_data, y_data, deg=degree, full=True)

    message("\n CS derivative Result\n", result, result[0], message_verbosity=4)

    res = result[0][0]

    # L2errs = [(a + b)  for a, b in zip(L2errs[::2], L2errs[1::2])]

    # best_deg = 2*np.argmin(L2errs)+2
    # if best_deg<degree:
    # plt.plot(L2errs)
    # plt.show()

    #   print(f'Optimizing degree to {best_deg}')
    #   degree=best_deg
    #   cheb_coeffs, result = chebfit(x_data, y_data, deg=degree, full=True)

    #   res = result[0][0]
    if res >= 1e-3:
        if res <= 1e-1 and res >= 1e-3:
            message(f"Residue warning {res}")
        elif res > 1e-1:
            import traceback

            traceback.print_stack()
            y_fit_data = chebval(x_data, cheb_coeffs)
            plt.scatter(
                x_data, y_data, label="Input", s=3, c="magenta", marker="o"
            )
            plt.scatter(
                x_data, y_fit_data, label="fit", s=3, c="blue", marker="X"
            )
            plt.grid()
            plt.legend()
            plt.show()

            message(f"Residue warning {res}: Bad fit!")

    # compute the derivative
    cheb_der_coeffs = chebder(cheb_coeffs, m=order)

    # Change the basis to that of x_data
    dydx_data = chebval(x_data, cheb_der_coeffs)

    return dydx_data


########################################################
# Fourier differentiation
########################################################


def Fourier_differential(
    delta_x,
    udata_x=None,
    utilde_conven=None,
    omega0=np.inf,
    order=1,
    zero_mode=0,
    taper=True,
):
    """Fixed frequency differentiation, the inverse of the
    Fixed frequency integration as presented in Reisswig et al.
    This function takes in a function and returns its nth order
    derivative differential taken in the frequency domain.


    Parameters
    ----------
    xaxis: 1d array
           The co-ordinate space axis.
    udata_x: 1d array
             The data to be differentiated,
             expressed in coordinate space.
    omega0: float, optional
            The cutoff angular frequency in the integration.
            Must be lower than the starting angular frequency
            of the input waveform.
    order: int, optional
           The number of times to differentiate
           the integrand in time.
    zero_mode: float, optional
               The zero mode amplitude of the FFT required.
    taper: bool
           Whether or not to taper the real co-ordinate space data.

    Returns
    -------
    udata_differentiated: 1d array
                          The input waveform in time-space, integrated
                          in frequency space using FFI.

    utilde_differentiated: 1d array
                           The FFT of the frixed frequency
                           differentiated array in good conventions.


    new_x_axis: 1d array
                The new x-axis, assuming the
                data may have been changed in length

    freq_axis: 1d array
               The frequency axis of the FFT of data.

    Notes
    -----
    The returned differentiated function of a real udata_x
    in real co-ordinate space is a complex number
    due to the numerical inaccuracies. Take
    the real part of udata_differentiated if the input udata_x
    is real.

    """

    if not utilde_conven:
        # Compute the FFT of data
        from numpy.fft import ifft
        from transforms import compute_fft, unset_fft_conven

        from waveformtools import taper

        udata_x = taper(udata_x, delta_t=delta_x)
        new_x_axis = udata_x.sample_times
        udata_x = np.array(udata_x)
        freq_axis, utilde_conven = compute_fft(udata_x, delta_x)

        # Find the length of the input data.
        Nlen = len(udata_x)

    else:
        Nlen = len(utilde_conven)

    # Find the location of the zero index.
    if Nlen % 2 == 0:
        zero_index = int(Nlen / 2)
    else:
        zero_index = int((Nlen + 1) / 2)

    # Construct the angular frequency axis.
    omega_axis = 2 * np.pi * freq_axis

    # print(omega_axis)
    message("The cutoff angular frequency is", omega0)

    # Alter the frequency axis if omega0 < inf

    if omega0 < np.inf:
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
                # Change the angular frequency
                # if its magnitude is below a given omega0.
                if abs(element) > omega0:
                    omega_axis[index] = sign * omega0

    # Set the zero frequency element separately.
    omega_axis[zero_index] = omega0

    # print(omega_axis)
    # Differentiate in frequency space
    utilde_differentiated = np.power((-1j * omega_axis), order) * utilde_conven

    # Set the zero mode amplitude
    if not zero_mode:
        utilde_differentiated[zero_index] = 0

    # Get the inverse fft
    utilde_differentiated_np = unset_fft_conven(utilde_differentiated)

    udata_differentiated = ifft(utilde_differentiated_np)

    return udata_differentiated, utilde_differentiated, new_x_axis, freq_axis


#########################################################
# Finite difference differentiation
########################################################


def differentiate(data, delta_t):
    """Central difference derivative calculator.
    Forward/ backward Euler near the boundaries.

    Parameters
    ----------

    data: 1d array
          The 1d data.
    delta_t: float
             The time step in units of t/M.

    Returns
    -------
    dAdt: 1d array
          The derivative.
    """

    # A list to hold the derivatives.
    dAdt = []

    # Near boundaries: For n=0
    val = (data[1] - data[0]) / delta_t
    dAdt.append(val)

    for index in range(1, len(data) - 1):
        # For interior points.
        val = (data[index + 1] - data[index - 1]) / (2 * delta_t)
        dAdt.append(val)

    # Near boundaries: For n = N-1

    val = (data[-1] - data[-2]) / delta_t
    dAdt.append(val)

    return np.array(dAdt)


def differentiate2(data, delta_t):
    """Five point difference derivative calculator.
    Not accurate near the boundaries.


    Parameters
    ----------
    data: 1d array
          The 1d data.
    delta_t: float
             The time step in t/M.

    Returns
    -------
    dAdt: 1d array
          The derivative.
    """

    # Number of points on right side.
    order = 2

    # The five point derivative stencil.
    coeffs = np.array([1, -8, 0, 8, -1])
    # The divison factor.
    divide = 12

    # list to hold the derivative
    der_data = []

    # Near boundaries. For n=0, N
    der0 = (data[1] - data[0]) / delta_t
    derNm1 = (data[-1] - data[-2]) / delta_t

    der_data.append(der0)

    # for n=1
    der1 = (data[2] - data[0]) / (2 * delta_t)
    # FOr n=-2
    derNm2 = (data[-1] - data[-3]) / (2 * delta_t)

    der_data.append(der1)

    for index in range(order, len(data) - order):
        # For the interior points, use the five point stencil.
        data_subarray = data[index - order : index + order + 1]
        der_data.append(np.dot(coeffs, data_subarray) / (divide * delta_t))

    der_data.append(derNm2)
    der_data.append(derNm1)

    return np.array(der_data)


def differentiate3(data, delta_t):
    """Seven point difference derivative calculator.
    Not accurate near the boundaries.


    Parameters
    ----------
    data: 1d array
          The 1d data.
    delta_t: float
             The time step in t/M.

    Returns
    -------
    dAdt: 1d array
          The derivative.
    """

    # The number of points on one direction.
    order = 3
    # The seven point stencil.
    coeffs = np.array([-1, 9, -45, 0, 45, -9, 1])
    divide = 60

    # A list to hold the derivatives.
    der_data = []

    # Near the boundaries

    # n=0, N-1
    der0 = (data[1] - data[0]) / delta_t
    derNm1 = (data[-1] - data[-2]) / delta_t

    der_data.append(der0)

    # for n=1, N-2
    der1 = (data[2] - data[0]) / (2 * delta_t)
    derNm2 = (data[-1] - data[-3]) / (2 * delta_t)

    der_data.append(der1)

    # For n=2, N-3
    stencil = np.array([1, -8, 0, 8, -1]) / 12
    data_vec = data[:5]

    der2 = np.dot(stencil, data_vec) / delta_t

    data_vec = data[-5:]

    derNm3 = np.dot(stencil, data_vec) / delta_t

    der_data.append(der2)

    for index in range(order, len(data) - order):
        data_subarray = data[index - order : index + order + 1]
        der_data.append(np.dot(coeffs, data_subarray) / (divide * delta_t))

    der_data.append(derNm3)
    der_data.append(derNm2)
    der_data.append(derNm1)

    return der_data


def differentiate4(data, delta_t):
    """Nine point difference derivative calculator.
    Not accurate near the boundaries.

    Parameters
    ----------
    data: 1d array
          The 1d data.
    delta_t: float
             The time step in t/M.

    Returns
    -------
    dAdt: 1d array
          The derivative.

    """

    # The number of points on one side.
    order = 4
    # THe stencil.
    coeffs = np.array([3, -32, 168, -672, 0, 672, -168, 32, 3])
    # The divison factor.
    divide = 840

    # A list to hold the points.
    der_data = []

    # Near the boundaries

    # n=0, N-1
    der0 = (data[1] - data[0]) / delta_t
    derNm1 = (data[-1] - data[-2]) / delta_t

    der_data.append(der0)

    # for n=1, N-2
    der1 = (data[2] - data[0]) / (2 * delta_t)
    derNm2 = (data[-1] - data[-3]) / (2 * delta_t)

    der_data.append(der1)

    # For n=2, N-3
    stencil = np.array([1, -8, 0, 8, -1]) / 12
    data_vec = data[:5]

    der2 = np.dot(stencil, data_vec) / delta_t

    data_vec = data[-5:]

    derNm3 = np.dot(stencil, data_vec) / delta_t

    der_data.append(der2)

    # For n=3, N-4
    stencil = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60

    data_vec = data[:7]

    der3 = np.dot(stencil, data_vec) / delta_t

    data_vec = data[-7:]

    derNm4 = np.dot(stencil, data_vec) / delta_t

    der_data.append(der3)

    for index in range(order, len(data) - order):
        # For the interior points.
        data_subarray = data[index - order : index + order + 1]
        der_data.append(np.dot(coeffs, data_subarray) / (divide * delta_t))

    der_data.append(derNm4)
    der_data.append(derNm3)
    der_data.append(derNm2)
    der_data.append(derNm1)

    return der_data


def differentiate5(data, delta_t):
    """Eleven point difference derivative calculator.
    Not accurate near the boundaries.

    Parameters
    ----------
    data: 1d array
          The 1d data.
    delta_t: float
             The time step in t/M.

    Returns
    -------
    dAdt: 1d array
          The derivative of the data.

    """

    # The number of points on one side.
    order = 5
    # The stencil.
    coeffs = np.array([-2, 25, -150, 600, -2100, 0, 2100, -600, 150, -25, 2])
    # The divison factor.
    divide = 2520

    # A list to hold the derivatives.
    der_data = []

    # Near the boundaries

    # n=0, N-1
    der0 = (data[1] - data[0]) / delta_t
    derNm1 = (data[-1] - data[-2]) / delta_t
    der_data.append(der0)

    # for n=1, N-2
    der1 = (data[2] - data[0]) / (2 * delta_t)
    derNm2 = (data[-1] - data[-3]) / (2 * delta_t)
    der_data.append(der1)

    # For n=2, N-3
    stencil = np.array([1, -8, 0, 8, -1]) / 12
    data_vec = data[:5]

    der2 = np.dot(stencil, data_vec) / delta_t
    data_vec = data[-5:]

    derNm3 = np.dot(stencil, data_vec) / delta_t
    der_data.append(der2)

    # For n=3, N-4
    stencil = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60
    data_vec = data[:7]

    der3 = np.dot(stencil, data_vec) / delta_t
    data_vec = data[-7:]

    derNm4 = np.dot(stencil, data_vec) / delta_t
    der_data.append(der3)

    # For n=4, N-5
    stencil = np.array([3, -32, 168, -672, 0, 672, -168, 32, 3]) / 840
    data_vec = data[:9]

    der4 = np.dot(stencil, data_vec) / delta_t
    data_vec = data[-9:]

    derNm5 = np.dot(stencil, data_vec) / delta_t
    der_data.append(der4)

    for index in range(order, len(data) - order):
        # For the interior points.
        data_subarray = data[index - order : index + order + 1]
        der_data.append(np.dot(coeffs, data_subarray) / (divide * delta_t))

    der_data.append(derNm5)
    der_data.append(derNm4)
    der_data.append(derNm3)
    der_data.append(derNm2)
    der_data.append(derNm1)

    return der_data


def differentiate5_vec_nonumba(data, delta_t):
    """Eleven point difference derivative calculator
    for data on a sphere.

    Not accurate near the boundaries.


    Parameters
    ----------
    data:   3d array
            The axis order being (theta, phi, time)

    delta_t: float
             The time step in t/M.

    Returns
    -------
    dAdt: 3d array
          The derivative of the data.

    """

    # data = np.transpose(data, (2, 0, 1))
    data = np.moveaxis(data, -1, 0)

    der_data = np.zeros(data.shape, dtype=np.complex128)

    # The number of points on one side.
    order = 5
    # The stencil.
    coeffs = np.array([-2, 25, -150, 600, -2100, 0, 2100, -600, 150, -25, 2])
    # The divison factor.
    divide = 2520

    # A list to hold the derivatives.
    # der_data = np.array([])

    # Near the edges

    # n=0, N-1, forward/ backward Euler
    der0 = (data[1] - data[0]) / delta_t
    derNm1 = (data[-1] - data[-2]) / delta_t
    # der_data = np.append(der_data, [der0], axis=aax)
    der_data[0] = der0

    # for n=1, N-2, Central difference
    der1 = (data[2] - data[0]) / (2 * delta_t)
    derNm2 = (data[-1] - data[-3]) / (2 * delta_t)
    # der_data = np.append(der_data, [der1], axis=aax)
    der_data[1] = der1

    # For n=2, N-3, 5 point stencil
    stencil = np.array([1, -8, 0, 8, -1]) / 12
    data_vec = data[:5]
    der2 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t
    data_vec = data[-5:]
    derNm3 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t
    # der_data = np.append(der_data, [der2], axis=aax)
    der_data[2] = der2

    # For n=3, N-4, 7 point stencil
    stencil = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60
    data_vec = data[:7]
    der3 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t
    data_vec = data[-7:]
    derNm4 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t
    # der_data = np.append(der_data, [der3], axis=aax)
    der_data[3] = der3

    # For n=4, N-5, 9 point stencil
    stencil = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840
    data_vec = data[:9]
    der4 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t
    data_vec = data[-9:]
    derNm5 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t

    # For interior points
    der_data[4] = der4
    for index in range(order, len(data) - order):
        data_subarray = data[index - order : index + order + 1]
        # der_data = np.append(der_data,
        # [np.tensordot(coeffs, data_subarray, axes=((0), (0)))
        # / (divide * delta_t)], axis=aax)
        der_data[index] = np.tensordot(
            coeffs, data_subarray, axes=((0), (0))
        ) / (divide * delta_t)

    # der_data = np.append(der_data, [derNm5], axis=aax)
    der_data[-5] = derNm5
    # der_data = np.append(der_data, [derNm4], axis=aax)
    der_data[-4] = derNm4
    # der_data = np.append(der_data, [derNm3], axis=aax)
    der_data[-3] = derNm3
    # der_data = np.append(der_data, [derNm2], axis=aax)
    der_data[-2] = derNm2
    # der_data = np.append(der_data, [derNm1], axis=aax)
    der_data[-1] = derNm1

    message("Shape of data", der_data.shape, message_verbosity=2)

    der_data = np.moveaxis(der_data, 0, -1)

    return der_data


# @njit(parallel=True)
def differentiate5_vec_numba(data, delta_t):
    """Eleven point difference derivative calculator.
    Not accurate near the boundaries.


    Parameters
    ----------
    data: 1d array
          The 1d data.
    delta_t: float
             The time step in t/M.

    Returns
    -------
    dAdt: 1d array
          The derivative of the data.

    """

    # import pdb
    # pdb.set_trace()

    s1, s2, s3 = data.shape
    data = np.transpose(data, (2, 0, 1))
    der_data = np.zeros(data.shape, dtype=np.complex128)

    # aax = 0
    # The number of points on one side.
    order = 5
    # The stencil.
    coeffs = np.array([-2, 25, -150, 600, -2100, 0, 2100, -600, 150, -25, 2])
    # The divison factor.
    divide = 2520

    # A list to hold the derivatives.
    # der_data = np.array([])

    # Near the boundaries

    # n=0, N-1
    der_data[0] = (data[1] - data[0]) / delta_t
    der_data[-1] = (data[-1] - data[-2]) / delta_t

    # der_data = np.append(der_data, [der0], axis=aax)

    # der_data[0] = der0
    # for n=1, N-2
    der_data[1] = (data[2] - data[0]) / (2 * delta_t)
    der_data[-2] = (data[-1] - data[-3]) / (2 * delta_t)

    # der_data = np.append(der_data, [der1], axis=aax)
    # der_data[1] = der1

    # For n=2, N-3
    stencil = np.array([1, -8, 0, 8, -1]) / 12
    data_vec = data[:5]

    # der2 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t

    for index, val in enumerate(stencil):
        der_data[2] += val * data_vec[index] / delta_t

    data_vec = data[-5:]

    # derNm3 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t

    for index, val in enumerate(stencil):
        der_data[-3] += val * data_vec[index] / delta_t

    # der_data = np.append(der_data, [der2], axis=aax)

    # der_data[2] = der2
    # For n=3, N-4
    stencil = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60

    data_vec = data[:7]

    # der3 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t
    for index, val in enumerate(stencil):
        der_data[3] += val * data_vec[index] / delta_t

    data_vec = data[-7:]

    # derNm4 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t
    for index, val in enumerate(stencil):
        der_data[-4] += val * data_vec[index] / delta_t

    # der_data = np.append(der_data, [der3], axis=aax)
    # der_data[3] = der3
    # For n=4, N-5
    stencil = np.array([3, -32, 168, -672, 0, 672, -168, 32, 3]) / 840

    data_vec = data[:9]

    # der4 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t
    for index, val in enumerate(stencil):
        der_data[4] += val * data_vec[index] / delta_t

    data_vec = data[-9:]

    # derNm5 = np.tensordot(stencil, data_vec, axes=((0), (0))) / delta_t

    for index, val in enumerate(stencil):
        der_data[-5] += val * data_vec[index] / delta_t

    # der_data = np.append(der_data, [der4], axis=aax)
    # der_data[4] = der4

    for index in range(order, len(data) - order):
        # For the interior points.
        data_subarray = data[index - order : index + order + 1]
        # der_data = np.append(der_data,
        # [np.tensordot(coeffs, data_subarray, axes=((0), (0)))
        # / (divide * delta_t)], axis=aax)
        # der_data[index] = np.tensordot(coeffs, data_subarray,
        # axes=((0), (0))) / (divide * delta_t)
        for inner_index, val in enumerate(coeffs):
            der_data[index] += (
                val * data_subarray[inner_index] / (divide * delta_t)
            )

    # der_data = np.append(der_data, [derNm5], axis=aax)
    # der_data[-5] = derNm5
    # der_data = np.append(der_data, [derNm4], axis=aax)
    # der_data[-4] = derNm4
    # der_data = np.append(der_data, [derNm3], axis=aax)
    # der_data[-3] = derNm3
    # der_data = np.append(der_data, [derNm2], axis=aax)
    # der_data[-2] = derNm2
    # der_data = np.append(der_data, [derNm1], axis=aax)
    # der_data[-1] = derNm1

    message("Shape of data", der_data.shape, message_verbosity=2)
    return np.transpose(der_data, (1, 2, 0))


############################################
# Complex Amplitude-Phase differentiation
############################################


def differentiate_cwaveform(time_axis, waveform, method="SP", degree=5):
    """Differentiate a given waveform by differentiating
    the Amplitude-Phase form.

    Parameters
    ----------
    time_axis:  1d array
                The time axis of the waveform.

    waveform:   1d array
                The complex 1d array of the waveform timeseries.


    Returns
    -------
    differentiated_waveform:    1d array
                                The waveform differentiated in time.

    """

    # Get the amplitude and phase of the complex 1d waveform.
    from waveformtools.waveformtools import xtract_camp_phase

    waveform_amp, waveform_phase = xtract_camp_phase(
        waveform.real, waveform.imag
    )

    # Differentiate the waveform.

    Amplitude_dot = derivative(
        time_axis, waveform_amp, method=method, degree=degree
    )
    Phase_dot = derivative(
        time_axis, waveform_phase, method=method, degree=degree
    )

    differentiated_waveform = (
        Amplitude_dot + waveform_amp * 1j * Phase_dot
    ) * np.exp(1j * waveform_phase)

    return differentiated_waveform
