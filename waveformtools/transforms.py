""" Methods to transform the waveform """

import numpy as np
import math
from waveformtools.waveformtools import message
from waveformtools.integrate import TwoDIntegral


from waveformtools.single_mode import SingleMode

# from numba import jit, njit

fact_dict = {0: 1, 1: 1}


def factorial(number):
    if number not in fact_dict.keys():
        fact_dict.update({number: factorial(number - 1) * number})

    return fact_dict[number]


# @njit(parallel=True)
def compute_fft(udata_x, delta_x):
    """Find the FFT of the samples in time-space,
    and return with the frequencies.

    Parameters
    ----------
    udata_x : 1d array
              The samples in time-space.

    delta_x : float
              The stepping delta_x

    Returns
    -------
    freqs :	1d array
            The frequency axis, shifted approriately.
    utilde : 1d array
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
    # freq_axis = np.linspace(-0.5 / delta_x, 0.5 / delta_x, Nlen)
    freq_axis = np.fft.fftshift(np.fft.fftfreq(Nlen, delta_x))

    return freq_axis, utilde


# @njit(parallel=True)
def compute_ifft(utilde, delta_f):
    """Find the inverse FFT of the samples in frequency-space,
    and return with the time axis.

    Parameters
    ----------
    utilde : 1d array
             The samples in frequency-space.

    delta_f : float
              The frequency stepping

    Returns
    -------
    time_axis : 1d array
                The time axis.

    udata_time : 1d array
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
    time_axis = np.arange(0, delta_t * Nlen, 1 / Nlen)

    return time_axis, udata_time


# @njit(parallel=True)
def set_fft_conven(utilde_orig):
    """Make a numppy fft consistent with the chosen conventions.
    This takes care of the zero mode factor and array position.
    Also, it shifts the negative frequencies using numpy's fftshift.

    Parameters
    ----------
    utilde_orig : 1d array
                  The result of a numpy fft.

    Returns
    -------
    utilde_conven :	1d array
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
    """Make an actual conventional fft
    consistent with numpy's conventions.
    The inverse of set_conv.


    Parameters
    ----------
    utilde_conven : 1d array
                    The conventional fft data vector.

    Returns
    -------
    utilde_np : 1darray
                The fft data vector in numpy conventions.
    """

    utilde_np = np.fft.ifftshift(utilde_conven)

    utilde_np = len(utilde_np) * np.conj(utilde_np) / 2
    # message(utilde_original[0])
    utilde_np[0] *= 2
    # message(utilde_original[0])

    return utilde_np


def check_Yslm_args(spin_weight, ell, emm):
    """Check if the arguments to a Yslm functions
    makes sense

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    """

    assert ell >= abs(spin_weight), (
        " ell should be greater than"
        "or equal to the absolute value of spin weight "
    )

    assert abs(emm) <= ell, (
        "absolute value of emm should be" "less than or equal to ell"
    )


def Yslm(spin_weight, ell, emm, theta, phi):
    """Spin-weighted spherical harmonics fast evaluation.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic.
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta : float
            The polar angle  :math:`\\theta` in radians,
    phi : float
          The aximuthal angle :math:`\\phi' in radians.

    Returns
    --------
    Yslm : float
           The value of Yslm at :math:`\\theta, phi'.

    Note
    ----
    This is accurate upto 14 decimals for L upto 25.
    """

    check_Yslm_args(spin_weight, ell, emm)
    import sympy as sp

    # theta, phi = sp.symbols('theta phi')

    fact = math.factorial
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
                / np.power(np.tan(theta / 2), (2 * aar + abs_spin_weight - emm))
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


def check_Yslm_theta(theta_grid, threshold=1e-6):
    theta_list = np.array(theta_grid).flatten()

    locs = np.where(abs(theta_list) < threshold)

    # print("Locs", locs, locs[0])

    for index in locs[0]:

        theta = theta_list[index]

        # print("Theta val \t ", theta, "\n")

        if theta == 0:
            sign = 1
        else:
            sign = theta_list[index] / abs(theta_list[index])

        theta_list[index] = theta_list[index] + sign * threshold

    return theta_list.reshape(np.array(theta_grid).shape)


def Yslm_vec(spin_weight, ell, emm, theta_grid, phi_grid):
    """Spin-weighted spherical harmonics fast evaluations
    on numpy arrays for vectorized evaluations.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta : float
            The polar angle  :math:`\\theta` in radians,
    phi : float
          The aximuthal angle :math:`\\phi' in radians.

    Returns
    --------
    Yslm : float
           The value of Yslm at :math:`\\theta, phi'.

    Note
    ----
    This is accurate upto 14 decimals for L upto 25.
    """

    check_Yslm_args(spin_weight, ell, emm)

    theta_grid = check_Yslm_theta(theta_grid)

    from math import comb

    fact = math.factorial

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
            message(f"Skipping r {aar}", message_verbosity=4)
            continue
        else:
            term1 = comb(ell - abs_spin_weight, aar)
            term2 = comb(ell + abs_spin_weight, aar + abs_spin_weight - emm)
            term3 = np.power(float(-1), (ell - aar - abs_spin_weight))
            term4 = np.exp(1j * emm * phi_grid)
            term5 = np.longdouble(
                np.power(
                    np.tan(theta_grid / 2), (-2 * aar - abs_spin_weight + emm)
                )
            )
            subterm = term1 * term2 * term3 * term4 * term5

            Sum += subterm

    Yslmv = float(-1) ** emm * (
        np.sqrt(
            np.longdouble(fact(ell + emm))
            * np.longdouble(fact(ell - emm))
            * (2 * ell + 1)
            / (
                4
                * np.pi
                * np.longdouble(fact(ell + abs_spin_weight))
                * np.longdouble(fact(ell - abs_spin_weight))
            )
        )
        * np.sin(theta_grid / 2) ** (2 * ell)
        * Sum
    )

    value = factor * Yslmv

    if np.isnan(np.array(value)).any():
        message(
            "Nan discovered. Falling back to Yslm_prec on defaulted locations",
            message_verbosity=1,
        )

        nan_locs = np.where(np.isnan(np.array(value).flatten()))[0]

        message("Nan locations", nan_locs, message_verbosity=1)

        theta_list = np.array(theta_grid).flatten()
        phi_list = np.array(phi_grid).flatten()

        message("Theta values", theta_list[nan_locs], message_verbosity=1)

        value_list = np.array(value, dtype=np.complex128).flatten()

        for index in nan_locs:
            replaced_value = Yslm_prec(
                spin_weight=spin_weight,
                theta=theta_list[index],
                phi=phi_list[index],
                ell=ell,
                emm=emm,
            )

            value_list[index] = replaced_value

        value = np.array(value_list).reshape(theta_grid.shape)

        message("nan corrected", value, message_verbosity=1)

        if np.isnan(np.array(value)).any():
            message(
                "Nan re discovered. Falling back to Yslm_prec_grid",
                message_verbosity=1,
            )

            value = np.complex128(
                Yslm_prec_grid(
                    spin_weight, ell, emm, theta_grid, phi_grid, prec=16
                )
            )

            if np.isnan(np.array(value)).any():
                if (abs(np.array(theta_grid)) < 1e-14).any():
                    # print("!!! Warning: setting to zero manually.
                    # Please check again !!!")
                    # value = 0
                    raise ValueError(
                        "Possible zero value encountered due to"
                        f"small theta {np.amin(theta_grid)}"
                    )

                else:
                    raise ValueError(
                        "Although theta>1e-14, couldnt compute Yslm."
                        "Please check theta"
                    )

    return value


def Yslm_prec_grid(spin_weight, ell, emm, theta_grid, phi_grid, prec=24):
    """Spin-weighted spherical harmonics function with precise computations
    on an angular grid. Uses a symbolic method evaluated at the degree
    of precision requested by the user.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta_grid : 2darray
                 The polar angle  :math:`\\theta` in radians,
    phi_grid : 2darray
               The aximuthal angle :math:`\\phi' in radians.
    pres : int, optional
           The precision i.e. number of digits to compute
           upto. Default value is 16.

    Returns
    --------
    Yslm_vals : float
               The value of Yslm at the grid
               :math:`\\theta, phi'.
    """

    theta_grid_1d, phi_grid_1d = theta_grid.flatten(), phi_grid.flatten()
    from itertools import zip_longest

    ang_set = zip_longest(theta_grid_1d, phi_grid_1d)

    Yslm_vals = np.array(
        [
            Yslm_prec(
                spin_weight=spin_weight,
                theta=thetav,
                phi=phiv,
                ell=ell,
                emm=emm,
                prec=prec,
            )
            for thetav, phiv in ang_set
        ]
    ).reshape(theta_grid.shape)

    return Yslm_vals


def Yslm_prec(spin_weight, ell, emm, theta, phi, prec=24):
    """Spin-weighted spherical harmonics function with precise computations.
    Uses a symbolic method evaluated at the degree of precision requested
    by the user.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta : float
            The polar angle  :math:`\\theta` in radians,
    phi : float
          The aximuthal angle :math:`\\phi' in radians.
    pres : int, optional
           The precision i.e. number of digits to compute
           upto. Default value is 16.

    Returns
    --------
    Yslm : float
           The value of Yslm at :math:`\\theta, phi'.
    """

    check_Yslm_args(spin_weight, ell, emm)

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
    """Spin-weighted spherical harmonics precise,
    symbolic computation for deferred evaluations.
    Is dependent on variables th: theta and ph:phi.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta : float
            The polar angle  :math:`\\theta` in radians,
    phi : float
          The aximuthal angle :math:`\\phi' in radians.
    pres : int, optional
           The precision i.e. number of digits to compute
           upto. Default value is 16.

    Returns
    --------
    Yslm : sym
           The value of Yslm at :math:`\\theta, phi'.
    """

    check_Yslm_args(spin_weight, ell, emm)

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


def CheckRegReq(data):
    """Check if a function requires regularization.

    Parameters
    ----------
    data : 1d array
           A 1d array of the data to check.

    Returns
    -------
    check_reg : list
                a list containg the list of boundary points where
                regularization may be required.
    """
    nlen = len(data)
    nhlen = int(nlen / 2)
    nrlen = nlen - nhlen

    first_half = data[:nhlen]
    second_half = data[nhlen:]

    check_reg = [0, 0]

    toln = int(nlen / 10)
    if np.argmax(np.absolute(first_half)) <= toln:  # Added tolerence Apr 8 2023
        check_reg[0] = 1

    if np.argmax(np.absolute(second_half)) >= nrlen - toln:  # Here as well
        check_reg[1] = 1

    # if 1 in check_reg:
    # print('Reqularization required at', check_reg)

    return check_reg


def SHExpand(
    func,
    info,
    method_info,
    err_info=False,
    auto_ell_max=False,
    res_tol_percent=3,
    reg=False,
    reg_order=1,
):
    """Expand a given function in spin weight 0 spherical harmonics
    upto an optimal :math:`\\ell \\leq \\ell_{max}`.

    Parameters
    ----------
    func : ndarray
           The function to be expanded.
    info : Grid
           An instance of the Spherical grid class
           that stores the details of the structure
           of a grid on a topological sphere.
    method_info : MethodInfo
                  An instance of the method info
                  class that contains informations
                  about the numerical methods
                  to be used during the following
                  operations.
    err_info : bool
               Whether or not to compute and return
               the error measures related to the
               SH representation.

    Returns
    -------
    modes : dict
            The modes as a dictionary whose keys are lm.
    """

    if info.grid_type == "GL":
        assert method_info.ell_max == info.L, (
            "The GL grid L must be same" " as ell_max of requested expansion"
        )

    if auto_ell_max:
        message(
            "Using SHExpandAuto: "
            " Will automatically find optimal "
            " ell_max",
            message_verbosity=2,
        )

        results = SHExpandAuto(
            func,
            info,
            method_info,
            err_info,
            res_tol_percent,
            reg,
            reg_order=reg_order,
        )

    else:
        message(
            "Using ShExpandSimple:"
            " Expanding upto user prescribed"
            f" ell_max {method_info.ell_max}",
            message_verbosity=2,
        )

        results = SHExpandSimple(
            func, info, method_info, err_info, reg=reg, reg_order=reg_order
        )

    return results


def SHRegularize(func, theta_grid, check_reg, order=1):
    """Regularize an SH expansion"""

    reg_func = func.copy()

    if bool(check_reg[0]):
        message("Regularizing north end ", message_verbosity=2)
        reg_func *= (theta_grid) ** order

    if bool(check_reg[1]):
        message("Regularizing south end ", message_verbosity=2)
        reg_func *= (theta_grid - np.pi) ** order

    return reg_func


def SHDeRegularize(func, theta_grid, check_reg, order=1):
    """Return the original funtion given the regularized functions"""

    orig_func = func.copy()

    if bool(check_reg[0]):
        orig_func /= (theta_grid) ** order

    if bool(check_reg[1]):
        orig_func /= (theta_grid - np.pi) ** order

    return orig_func


def SHExpandAuto(
    func,
    info,
    method_info,
    err_info=False,
    res_tol_percent=3,
    reg=False,
    reg_order=1,
    check_reg=None,
):
    """Expand a given function in spin weight 0 spherical harmonics
    upto an optimal :math:`\\ell \\leq \\ell_{max}` that is
    automatically found.

    Additionally, if requested, this routine can:

    1. regularize a function and expand and return the
       modes of the regularized function and the associated
       regularization details.
    2. Compute diagnostic information in terms of residue
       per mode.
    3. The RMS deviation of the reconstructed expansion from the
       original function.

    Parameters
    ----------
    func : ndarray
           The function to be expanded.
    info : Grid
           An instance of the Spherical grid class
           that stores the details of the structure
           of a grid on a topological sphere.
    method_info : MethodInfo
                  An instance of the method info
                  class that contains informations
                  about the numerical methods
                  to be used during the following
                  operations.
    err_info : bool
               Whether or not to compute and return
               the error measures related to the
               SH representation.
    check_reg : list, optional
                A list of two integers (0,1)
                that depicts whether or not to
                regularize the input function
                at the poles.
    Returns
    -------
    modes : dict
            The modes as a dictionary whose keys are lm.


    Notes
    -----
    When regularization is requested,
        1. To compute the total RMS deviation,
           the orginal form is used.
        2. To compute the rms deviation per mode,
           regularized expression is used.

    """

    #####################
    # Prepare
    #####################

    orig_func = func.copy()

    # from scipy.special import sph_harm

    theta_grid, phi_grid = info.meshgrid

    ell_max = method_info.ell_max
    method = method_info.int_method

    # from waveformtools.single_mode import SingleMode

    modes = {}

    # if method != "GL":
    #    SinTheta = np.sin(theta_grid)
    # else:
    #    SinTheta = 1

    #####################

    ####################
    # Regularize
    ####################
    if reg:
        if check_reg is None:
            check_reg = CheckRegReq(func)

        if np.array(check_reg).any() > 0:
            message("Regularizing function", message_verbosity=2)
            func = SHRegularize(func, theta_grid, check_reg, order=reg_order)

    #####################

    #################
    # Zeroth residue
    #################

    recon_func = np.zeros(func.shape, dtype=np.complex128)

    # The first residue is the maximum residue
    # with zero as reconstructed function
    res1 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))

    # The list holding all residues
    all_res = [res1]

    #################

    #######################
    # Expand
    #######################

    for ell in range(ell_max + 1):
        emm_list = np.arange(-ell, ell + 1)

        emmCoeffs = {}

        for emm in emm_list:
            Ylm = Yslm_vec(
                spin_weight=0,
                emm=emm,
                ell=ell,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )

            integrand = func * np.conjugate(Ylm)

            uu = np.isnan(integrand.any())

            if uu:
                raise ValueError("Nan found!")

            Clm = TwoDIntegral(integrand, info, method=method)

            recon_func += Clm * Ylm

            emmCoeffs.update({f"m{emm}": Clm})

        if ell % 2 == 0:
            res2 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))

            dres_percent = 100 * (res2 / res1 - 1)

            if dres_percent > res_tol_percent:
                all_res.append(res2)
                message(
                    f" ell_max residue increase error of {dres_percent} %",
                    message_verbosity=1,
                )

                ell_max = ell - 1
                message(
                    "Auto setting ell max to {ell_max} instead",
                    ell_max,
                    message_verbosity=1,
                )
                break

            else:
                res1 = res2
                all_res.append(res1)

        elif ell == ell_max:
            res2 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))
            all_res.append(res2)

        modes.update({f"l{ell}": emmCoeffs})

    ############################

    #################################
    # update details
    #################################

    result = SingleMode(modes_dict=modes)
    result._Grid = info

    if reg:
        result.reg_order = reg_order
        result.reg_details = check_reg

    else:
        result.reg_order = 0
        result.reg_details = "NA"

    if err_info:
        from waveformtools.diagnostics import RMSerrs

        recon_func = SHContract(modes, info, ell_max)

        ################################
        # Compute total RMS deviation
        # of the expansion
        ###############################

        if reg:
            if np.array(check_reg).any() > 0:
                message(
                    "De-regularizing function" "for RMS deviation computation",
                    message_verbosity=2,
                )

                recon_func = SHDeRegularize(
                    recon_func, theta_grid, check_reg, order=reg_order
                )

        Rerr, Amin, Amax = RMSerrs(orig_func, recon_func, info)
        err_info_dict = {"RMS": Rerr, "Amin": Amin, "Amax": Amax}

        ############################
        # Update error details
        ############################

        result.error_info = err_info_dict
        result.residuals = all_res

        even_mode_nums = np.arange(0, ell_max, 2)

        residual_axis = [-1] + list(even_mode_nums)

        if ell_max % 2 == 1:
            residual_axis += [ell_max]

        result.residual_axis = residual_axis

        if Rerr > 0.1:
            message(
                f"Residue warning {Rerr}!  Inaccurate representation.",
                message_verbosity=0,
            )

    #####################################

    return result


def SHExpandSimple(
    func,
    info,
    method_info,
    err_info=False,
    reg=False,
    reg_order=1,
    check_reg=None,
):
    """Expand a given function in spin weight 0 spherical harmonics
    upto a user prescribed :math:`\\ell_{max}`.

    Additionally, if requested, this routine can:

    1. regularize a function and expand and return the
       modes of the regularized function and the associated
       regularization details.
    2. Compute diagnostic information in terms of residue
       per mode.
    3. The RMS deviation of the reconstructed expansion from the
       original function.


    Parameters
    ----------
    func : ndarray
           The function to be expanded.
    info : Grid
           An instance of the Spherical grid class
           that stores the details of the structure
           of a grid on a topological sphere.
    method_info : MethodInfo
                  An instance of the method info
                  class that contains informations
                  about the numerical methods
                  to be used during the following
                  operations.
    err_info : bool
               Whether or not to compute and return
               the error measures related to the
               SH representation.

    check_reg : list, optional
                A list of two integers (0,1)
                that depicts whether or not to
                regularize the input function
                at the poles.

    Returns
    -------
    modes : dict
            The modes as a dictionary whose keys are lm.

    Notes
    -----
    When regularization is requested,
        1. To compute the total RMS deviation,
           the orginal form is used.
        2. To compute the rms deviation per mode,
           regularized expression is used.


    """
    # from scipy.special import sph_harm

    # from waveformtools.single_mode import SingleMode

    orig_func = func.copy()

    theta_grid, phi_grid = info.meshgrid

    ell_max = method_info.ell_max

    method = method_info.int_method

    message(
        f"SHExpandSimple: expansion ell max is {ell_max}", message_verbosity=3
    )

    # Good old Modes dict
    # modes = {}

    # if method != "GL":
    #    SinTheta = np.sin(theta_grid)
    # else:
    #    SinTheta = 1

    if reg:
        if check_reg is None:
            check_reg = CheckRegReq(func)

        if np.array(check_reg).any() > 0:
            message("Regularizing function", message_verbosity=2)
            func = SHRegularize(func, theta_grid, check_reg, order=reg_order)

    result = SingleMode(ell_max=ell_max)

    recon_func = np.zeros(func.shape, dtype=np.complex128)

    res1 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))

    all_res = [res1]

    for ell in range(ell_max + 1):
        emm_list = np.arange(-ell, ell + 1)

        # Subdict of modes
        # emmCoeffs = {}

        for emm in emm_list:
            Ylm = Yslm_vec(
                spin_weight=0,
                emm=emm,
                ell=ell,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )

            integrand = func * np.conjugate(Ylm)

            uu = np.isnan(integrand.any())

            # print(uu)
            if uu:
                raise ValueError("Nan found!")

            Clm = TwoDIntegral(integrand, info, int_method=method)

            recon_func += Clm * Ylm

            # emmCoeffs.update({f"m{emm}": Clm})
            # print(Clm)
            # message("Clm ", Clm, message_verbosity=2)

            result.set_mode_data(ell=ell, emm=emm, value=Clm)

        res = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))
        all_res.append(res)

        # modes.update({f"l{ell}": emmCoeffs})

    # result2 = SingleMode(modes_dict=modes)

    # message(f"result2 ell max {result2.ell_max}", message_verbosity=1)

    result._Grid = info

    if reg:
        result.reg_order = reg_order
        result.reg_details = check_reg

    else:
        result.reg_order = 0
        result.reg_details = "NA"

    if err_info:
        from waveformtools.diagnostics import RMSerrs

        recon_func = SHContract(result, info, ell_max)

        if reg:
            if np.array(check_reg).any() > 0:
                message(
                    "De-regularizing function"
                    " for total RMS deviation"
                    " computation",
                    message_verbosity=2,
                )

                recon_func = SHDeRegularize(
                    recon_func, theta_grid, check_reg, order=reg_order
                )

        Rerr, Amin, Amax = RMSerrs(orig_func, recon_func, info)
        err_info_dict = {"RMS": Rerr, "Amin": Amin, "Amax": Amax}

        result.error_info = err_info_dict
        result.residuals = all_res
        result.residual_axis = np.arange(-1, ell_max + 1)

        if Rerr > 0.1:
            message("Residue warning!", message_verbosity=0)

    return result


def SHExpandSimpleSPack(
    func,
    info,
    method_info,
    err_info=False,
    reg=False,
    reg_order=1,
    check_reg=None,
):
    """Expand a given function in spin weight 0 spherical harmonics
    upto a user prescribed :math:`\\ell_{max}`.

    Additionally, if requested, this routine can:

    1. regularize a function and expand and return the
       modes of the regularized function and the associated
       regularization details.
    2. Compute diagnostic information in terms of residue
       per mode.
    3. The RMS deviation of the reconstructed expansion from the
       original function.


    Parameters
    ----------
    func : ndarray
           The function to be expanded.
    info : Grid
           An instance of the Spherical grid class
           that stores the details of the structure
           of a grid on a topological sphere.
    method_info : MethodInfo
                  An instance of the method info
                  class that contains informations
                  about the numerical methods
                  to be used during the following
                  operations.
    err_info : bool
               Whether or not to compute and return
               the error measures related to the
               SH representation.

    check_reg : list, optional
                A list of two integers (0,1)
                that depicts whether or not to
                regularize the input function
                at the poles.

    Returns
    -------
    modes : dict
            The modes as a dictionary whose keys are lm.

    Notes
    -----
    When regularization is requested,
        1. To compute the total RMS deviation,
           the orginal form is used.
        2. To compute the rms deviation per mode,
           regularized expression is used.


    """
    # from scipy.special import sph_harm
    # from waveformtools.single_mode import SingleMode

    orig_func = func.copy()

    theta_grid, phi_grid = info.meshgrid

    ell_max = method_info.ell_max

    method = method_info.int_method

    message(
        f"SHExpandSimple: expansion ell max is {ell_max}", message_verbosity=3
    )

    # Good old Modes dict
    # modes = {}

    # if method != "GL":
    #    SinTheta = np.sin(theta_grid)
    # else:
    #    SinTheta = 1

    if reg:
        if check_reg is None:
            check_reg = CheckRegReq(func)

        if np.array(check_reg).any() > 0:
            message("Regularizing function", message_verbosity=2)
            func = SHRegularize(func, theta_grid, check_reg, order=reg_order)

    result = SingleMode(ell_max=ell_max)

    recon_func = np.zeros(func.shape, dtype=np.complex128)

    res1 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))

    all_res = [res1]

    for ell in range(ell_max + 1):
        emm_list = np.arange(-ell, ell + 1)

        # Subdict of modes
        # emmCoeffs = {}

        for emm in emm_list:
            Ylm = Yslm_vec(
                spin_weight=0,
                emm=emm,
                ell=ell,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )

            integrand = func * np.conjugate(Ylm)

            uu = np.isnan(integrand.any())

            # print(uu)
            if uu:
                raise ValueError("Nan found!")

            Clm = TwoDIntegral(integrand, info, method=method)

            recon_func += Clm * Ylm

            # emmCoeffs.update({f"m{emm}": Clm})
            # print(Clm)
            # message("Clm ", Clm, message_verbosity=2)

            result.set_mode_data(ell, emm, Clm)

        res = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))
        all_res.append(res)

        # modes.update({f"l{ell}": emmCoeffs})

    # result2 = SingleMode(modes_dict=modes)

    # message(f"result2 ell max {result2.ell_max}", message_verbosity=1)

    result._Grid = info

    if reg:
        result.reg_order = reg_order
        result.reg_details = check_reg

    else:
        result.reg_order = 0
        result.reg_details = "NA"

    if err_info:
        from waveformtools.diagnostics import RMSerrs

        recon_func = SHContract(result, info, ell_max)

        if reg:
            if np.array(check_reg).any() > 0:
                message(
                    "De-regularizing function"
                    " for total RMS deviation"
                    " computation",
                    message_verbosity=2,
                )

                recon_func = SHDeRegularize(
                    recon_func, theta_grid, check_reg, order=reg_order
                )

        Rerr, Amin, Amax = RMSerrs(orig_func, recon_func, info)
        err_info_dict = {"RMS": Rerr, "Amin": Amin, "Amax": Amax}

        result.error_info = err_info_dict
        result.residuals = all_res
        result.residual_axis = np.arange(-1, ell_max + 1)

        if Rerr > 0.1:
            message("Residue warning!", message_verbosity=0)

    return result


def SHContract(modes, info=None, ell_max=None):
    """Reconstruct a function on a grid given its SH modes.

    Parameters
    ----------
    modes : list
            A list of modes, in the convention [[l, [m list]], ]
    info : surfacegridinfo
           An instance of the surfacegridinfo.
    ell_max : int
              The max l mode to include.
    Returns
    -------
    recon_func : ndarray
                 The reconstructed grid function.
    """

    # if isinstance(modes, SingleMode):
    # message("SingleMode obj input.
    # Converting to modes dictionary", message_verbosity=3)

    # modes = modes.get_modes_dict()
    if info is None:
        info = modes.Grid

    if ell_max is None:
        ell_max = modes.ell_max

    # message(f"Modes in SHContract {modes}", message_verbosity=4)

    # print(modes)
    from waveformtools.waveforms import construct_mode_list

    # Construct modes list
    modes_list = construct_mode_list(ell_max=ell_max, spin_weight=0)

    message(f"Modes list in SHContract {modes_list}", message_verbosity=4)

    theta_grid, phi_grid = info.meshgrid

    recon_func = np.zeros(theta_grid.shape, dtype=np.complex128)

    for ell, emm_list in modes_list:
        for emm in emm_list:
            # Clm = modes[f"l{ell}"][f"m{emm}"]

            Clm = modes.mode(ell, emm)
            message(f"Clm shape in SHContract {Clm.shape}", message_verbosity=4)

            recon_func += Clm * Yslm_vec(
                spin_weight=0,
                ell=ell,
                emm=emm,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )

    return recon_func
