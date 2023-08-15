""" Methods to transform the waveform """

import numpy as np

from waveformtools.waveformtools import message

# from numba import jit, njit


# @njit(parallel=True)
def compute_fft(udata_x, delta_x):
    """Find the FFT of the samples in time-space, and return with the frequencies.

    Parameters
    ----------

    udata_x:	1d array
                                                    The samples in time-space.

    delta_x:	float
                                                    The stepping delta_x

    Returns
    -------

    freqs:	1d array
                                    The frequency axis, shifted approriately.
    utilde:	1d array
                                                    The samples in frequency space, with conventions applied.

    """

    # import necessary libraries.
    from numpy.fft import fft

    # FFT
    utilde_orig = fft(udata_x)

    # Apply conventions.
    utilde = set_fft_conven(utilde_orig)

    # Get frequency axes.
    Nlen = len(utilde)
    # message(Nlen)
    # Naxis			= np.arange(Nlen)
    # freq_orig		= fftfreq(Nlen)
    # freq_axis		= fftshift(freq_orig)*Nlen
    # delta_x		 = xdata[1] - xdata[0]

    # Naxis			 = np.arange(Nlen)
    #freq_axis = np.linspace(-0.5 / delta_x, 0.5 / delta_x, Nlen)
    freq_axis = np.fft.fftshift(np.fft.fftfreq(Nlen, delta_x))

    return freq_axis, utilde


# @njit(parallel=True)
def compute_ifft(utilde, delta_f):
    """Find the inverse FFT of the samples in frequency-space, and return with the time axis.

    Parameters
    ----------

    utilde	:	1d array
                                                    The samples in frequency-space.

    delta_f:	float
                                                    The frequency stepping

    Returns
    -------

    time_axis:	1d array
                                                    The time axis.

    udata_time:	1d array
                                                                    The samples in time domain.

    """

    # import necessary libraries.
    from numpy.fft import ifft

    # FFT
    utilde_orig = unset_fft_conven(utilde)

    # Inverse transform
    udata_time = ifft(utilde_orig)

    # Get frequency axes.
    Nlen = len(udata_time)
    # message(Nlen)
    # Naxis			= np.arange(Nlen)
    # freq_orig		= fftfreq(Nlen)
    # freq_axis		= fftshift(freq_orig)*Nlen
    # delta_x		 = xdata[1] - xdata[0]

    # Naxis			 = np.arange(Nlen)
    delta_t = 1.0 / (delta_f * Nlen)
    # Dt				= Nlen * delta_f/2

    # time_axis = np.linspace(0, delta_t * Nlen, Nlen)
    time_axis = np.arange(0, delta_t * Nlen, 1/Nlen)

    return time_axis, udata_time


# @njit(parallel=True)
def set_fft_conven(utilde_orig):
    """Make a numppy fft consistent with the chosen conventions.
                    This takes care of the zero mode factor and array position.
                    Also, it shifts the negative frequencies using numpy's fftshift.

    Parameters
    ----------

    utilde_orig:	1d array
                                                                    The result of a numpy fft.

    Returns
    -------

    utilde_conven:	1d array
                                                                    The fft with set conventions.
    """

    # Multiply by 2, take conjugate.
    utilde_conven = 2 * np.conj(utilde_orig) / len(utilde_orig)
    # Restore the zero mode.
    utilde_conven[0] = utilde_conven[0] / 2
    # Shift the frequency axis.
    utilde_conven = np.fft.fftshift(utilde_conven)

    return utilde_conven


# @njit(parallel=True)
def unset_fft_conven(utilde_conven):
    """Make an actual conventional fft consistent with numpy's conventions.
                    The inverse of set_conv.


    Parameters
    ----------

    utilde_conven:	1d array
                                                                    The conventional fft data vector.

    Returns
    -------

    utilde_np
    """

    utilde_np = np.fft.ifftshift(utilde_conven)

    utilde_np = len(utilde_np) * np.conj(utilde_np) / 2
    # message(utilde_original[0])
    utilde_np[0] *= 2
    # message(utilde_original[0])

    return utilde_np


def Yslm(spin_weight, ell, emm, theta, phi):
    """Spin-weighted spherical harmonics fast evaluation.

    Parameters
    ----------

    spin_weight :	int
                                    The Spin weight.
    ell :	int
                    The mode number :math:`\\ell'.
    emm :	int
                    The azimuthal mode number :math:`m'.
    theta : float
                    The polar angle  :math:`\\theta` in radians,
    phi :	float
                    The aximuthal angle :math:`\\phi' in radians.

    Returns
    --------
    Yslm :	float
                    The value of Yslm at :math:`\\theta, phi'.

            Note
            ----
            This is accurate upto 14 decimals for L upto 25.

    """
    import sympy as sp

    # theta, phi = sp.symbols('theta phi')

    fact = np.math.factorial
    # fact = sp.factorial
    Sum = 0

    factor = 1
    if spin_weight < 0:
        factor = (-1) ** ell
        theta = np.pi - theta
        phi += np.pi

    abs_spin_weight = abs(spin_weight)

    for aar in range(ell - abs_spin_weight + 1):
        if (aar + abs_spin_weight - emm) < 0 or (
            ell - aar - abs_spin_weight
        ) < 0:
            message(f"Skippin r {aar}", message_verbosity=3)
            continue
        else:
            Sum += (
                sp.binomial(ell - abs_spin_weight, aar)
                * sp.binomial(
                    ell + abs_spin_weight, aar + abs_spin_weight - emm
                )
                * np.power((-1), (ell - aar - abs_spin_weight))
                * np.exp(1j * emm * phi)
                / np.power(
                    np.tan(theta / 2), (2 * aar + abs_spin_weight - emm)
                )
            )

    Sum = complex(Sum)
    Yslm = (-1) ** emm * (
        np.sqrt(
            fact(ell + emm)
            * fact(ell - emm)
            * (2 * ell + 1)
            / (
                4
                * np.pi
                * fact(ell + abs_spin_weight)
                * fact(ell - abs_spin_weight)
            )
        )
        * np.sin(theta / 2) ** (2 * ell)
        * Sum
    )

    return factor * Yslm


def Yslm_vec(spin_weight, ell, emm, theta_grid, phi_grid):
    """Spin-weighted spherical harmonics fast evaluations on numpy arrays for vectorized evaluations.

    Inputs
    -----------

    spin_weight :	int
                                    The Spin weight.
    ell :	int
                    The mode number :math:`\\ell'.
    emm :	int
                    The azimuthal mode number :math:`m'.
    theta : float
                    The polar angle  :math:`\\theta` in radians,
    phi :	float
                    The aximuthal angle :math:`\\phi' in radians.

    Returns
    --------
    Yslm :	float
                    The value of Yslm at :math:`\\theta, phi'.

            Note
            ----
            This is accurate upto 14 decimals for L upto 25.
    """

    from math import comb

    fact = np.math.factorial

    theta_grid = np.array(theta_grid)
    phi_grid = np.array(phi_grid)

    Sum = 0 + 1j * 0

    factor = 1
    if spin_weight < 0:
        factor = (-1) ** ell
        theta_grid = np.pi - theta_grid
        phi_grid += np.pi

    abs_spin_weight = abs(spin_weight)

    for aar in range(0, ell - abs_spin_weight + 1):
        subterm = 0

        if (aar + abs_spin_weight - emm) < 0 or (
            ell - aar - abs_spin_weight
        ) < 0:
            message(f"Skipping r {aar}", message_verbosity=3)
            continue
        else:
            term1 = comb(ell - abs_spin_weight, aar)
            term2 = comb(ell + abs_spin_weight, aar + abs_spin_weight - emm)
            term3 = np.power(float(-1), (ell - aar - abs_spin_weight))
            term4 = np.exp(1j * emm * phi_grid)
            term5 = np.power(
                np.tan(theta_grid / 2), (-2 * aar - abs_spin_weight + emm)
            )
            subterm = term1 * term2 * term3 * term4 * term5

            Sum += subterm

    Yslmv = float(-1) ** emm * (
        np.sqrt(
            fact(ell + emm)
            * fact(ell - emm)
            * (2 * ell + 1)
            / (
                4
                * np.pi
                * fact(ell + abs_spin_weight)
                * fact(ell - abs_spin_weight)
            )
        )
        * np.sin(theta_grid / 2) ** (2 * ell)
        * Sum
    )

    return factor * Yslmv


def Yslm_prec(spin_weight, ell, emm, theta, phi, prec=24):
    """Spin-weighted spherical harmonics function with precise computations.
                            Uses a symbolic method evaluated at the degree of precision requested
                            by the user.
    Parameters
    ----------

    spin_weight :	int
                                    The Spin weight.
    ell :	int
                    The mode number :math:`\\ell'.
    emm :	int
                    The azimuthal mode number :math:`m'.
    theta : float
                    The polar angle  :math:`\\theta` in radians,
    phi :	float
                    The aximuthal angle :math:`\\phi' in radians.
    pres : int, optional
               The precision i.e. number of digits to compute
               upto. Default value is 16.
    Returns
    --------
    Yslm :	float
                    The value of Yslm at :math:`\\theta, phi'.

    """
    import sympy as sp

    # tv, pv = theta, phi
    th, ph = sp.symbols("theta phi")

    Yslm_expr = Yslm_prec_sym(spin_weight, ell, emm)

    if spin_weight < 0:
        theta = np.pi - theta
        phi = np.pi + phi

    return Yslm_expr.evalf(
        prec, subs={th: sp.Float(f"{theta}"), ph: sp.Float(f"{phi}")}
    )


def Yslm_prec_sym(spin_weight, ell, emm):
    """Spin-weighted spherical harmonics precise, symbolic computation for deferred evaluations.
       Is dependent on variables th: theta and ph:phi.
    Parameters
    ----------

    spin_weight :	int
                                    The Spin weight.
    ell :	int
                    The mode number :math:`\\ell'.
    emm :	int
                    The azimuthal mode number :math:`m'.
    theta : float
                    The polar angle  :math:`\\theta` in radians,
    phi :	float
                    The aximuthal angle :math:`\\phi' in radians.
    pres : int, optional
               The precision i.e. number of digits to compute
               upto. Default value is 16.

    Returns
    --------
    Yslm :	sym
                    The value of Yslm at :math:`\\theta, phi'.

    """
    import sympy as sp

    th, ph = sp.symbols("theta phi")

    fact = sp.factorial
    Sum = 0

    abs_spin_weight = abs(spin_weight)
    # To get negative spin weight SWSH
    # in terms of positive spin weight
    factor = 1
    if spin_weight < 0:
        factor = sp.Pow(-1, ell)

    for aar in range(ell - abs_spin_weight + 1):
        if (aar + abs_spin_weight - emm) < 0 or (
            ell - aar - abs_spin_weight
        ) < 0:
            # message('Continuing')
            continue
        else:
            # message('r, l, s, m', r, l, s, m)
            # a1 = sp.binomial(ell - spin_weight, aar)
            # message(a1)
            # a2 = sp.binomial(ell + spin_weight, aar + spin_weight - emm)
            # message(a2)
            # a3 = sp.exp(1j * emm * phi)
            # message(a3)
            # a4 = sp.tan(theta / 2)
            # message(a4)

            Sum += (
                sp.binomial(ell - abs_spin_weight, aar)
                * sp.binomial(
                    ell + abs_spin_weight, aar + abs_spin_weight - emm
                )
                * sp.Pow((-1), (ell - aar - abs_spin_weight))
                * sp.exp(sp.I * emm * ph)
                * sp.Pow(sp.cot(th / 2), (2 * aar + abs_spin_weight - emm))
            )

    Yslm_expr = sp.Pow(-1, emm) * (
        sp.sqrt(
            fact(ell + emm)
            * fact(ell - emm)
            * (2 * ell + 1)
            / (
                4
                * sp.pi
                * fact(ell + abs_spin_weight)
                * fact(ell - abs_spin_weight)
            )
        )
        * sp.Pow(sp.sin(th / 2), (2 * ell))
        * Sum
    )

    Yslm_expr = factor * sp.simplify(Yslm_expr)

    return Yslm_expr


def rotate_polarizations(wf, alpha):
    """Rotate the polarizations of the time domain
    observer waveform by :math:`2\alpha`

    Parameters
    ----------
    wf : 1d array
             The complex observer waveform to rotate.
    alpha : float
                    The coordinate angle to rotate the polarizations
                    in radians. Note that the polarizarions would
                    rotate by :math:`2 \alpha` on a cordinate
                    rotation of :math:`\alpha`.

    Returns
    -------
    rot_wf : 1d array
                     The rotated waveform.
    """

    h1, h2 = wf.real, wf.imag

    rh1 = np.cos(2 * alpha) * h1 - np.sin(2 * alpha) * h2
    rh2 = np.sin(2 * alpha) * h1 + np.cos(2 * alpha) * h2

    return rh1 + 1j * rh2
