""" The implementation of BMS transformations on the waveforms. """

#############################
# Imports
#############################

import numpy as np

from waveformtools.waveformtools import message
from scipy.interpolate import InterpolatedUnivariateSpline
from waveformtools.dataIO import construct_mode_list

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
    from spectools.spherical.swsh import Yslm_vec
    from spectools.spherical.swsh import Yslm_vec

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


def compute_linear_momentum_contribution_from_news(news_modes, ell, emm):

    #dPxdt = np.zeros(len(hdot_lm), dtype=np.complex128)
    #dPydt = np.zeros(len(hdot_lm), dtype=np.complex128)

    dpdt_xy_lm = news_modes.mode(ell, emm)*(
        linear_momentum_alm_func(ell, emm)*np.conj(news_modes.mode(ell, emm+1)) + 
        linear_momentum_blm_func(ell, -emm)*np.conj(news_modes.mode(ell-1, emm+1)) - 
        linear_momentum_blm_func(ell+1, emm+1)*np.conj(news_modes.mode(ell+1, emm+1))
                                  )

    dpdt_z_lm = news_modes.mode(ell, emm)*(
        linear_momentum_clm_func(ell, emm)*np.conj(news_modes.mode(ell, emm))   + 
        linear_momentum_dlm_func(ell, emm)*np.conj(news_modes.mode(ell-1, emm)) +
        linear_momentum_dlm_func(ell+1, emm)*np.conj(news_modes.mode(ell+1, emm))
                                  )
    
    dPxdt_lm = dpdt_xy_lm.real/(8*np.pi)
    dPydt_lm = dpdt_xy_lm.imag/(8*np.pi)
    dPzdt_lm = dpdt_z_lm/(16*np.pi)

    #print("dPzdt", dPzdt_lm)
    return dPxdt_lm, dPydt_lm, dPzdt_lm

def linear_momentum_alm_func(ell, emm):
    return np.sqrt((ell-emm)*(ell+emm+1))/(ell*(ell+1))

def linear_momentum_blm_func(ell, emm):
    return (1/(2*ell))*np.sqrt((ell-2)*(ell+2)*(ell+emm)*(ell+emm-1)/((2*ell-1)*(2*ell+1)))

def linear_momentum_clm_func(ell, emm):
    return 2*emm/(ell*(ell+1))

def linear_momentum_dlm_func(ell, emm):
    return (1/ell)*np.sqrt((ell-2)*(ell+2)*(ell-emm)*(ell+emm)/((2*ell-1)*(2*ell+1)))

def compute_impulse_from_force(time_axis, dPxdt, dPydt, dPzdt):
    
    spline_dPxdt = InterpolatedUnivariateSpline(time_axis, dPxdt, k=5)
    #spline_dPxdt_imag = InterpolatedUnivariateSpline(time_axis, dPxdt.imag, k=5)
    spline_dPydt = InterpolatedUnivariateSpline(time_axis, dPydt, k=5)
    #spline_dPydt_imag = InterpolatedUnivariateSpline(time_axis, dPydt.imag, k=5)

    spline_dPzdt_real = InterpolatedUnivariateSpline(time_axis, dPzdt.real, k=5)
    spline_dPzdt_imag = InterpolatedUnivariateSpline(time_axis, dPzdt.imag, k=5)

    dPx = spline_dPxdt.integral(time_axis[0], time_axis[-1]) #+ 1j*spline_dPxdt_imag.integral(time_axis[0], time_axis[-1])
    dPy = spline_dPydt.integral(time_axis[0], time_axis[-1]) #+ 1j*spline_dPydt_imag.integral(time_axis[0], time_axis[-1])
    dPz = spline_dPzdt_real.integral(time_axis[0], time_axis[-1]) + 1j*spline_dPzdt_imag.integral(time_axis[0], time_axis[-1])

    # Factor = lal.C_SI/1000 to get in km/s
    return np.array([dPx, dPy, dPz])


def f_lm(ell, emm):
    ''' The flm factor for the angular momentum modes '''
    f_lm = np.sqrt( ell*(ell+1) - emm*(emm+1) )
    return f_lm

def compute_angular_momentum_evolution(strain_modes, news_modes):

    dJcx_dt = 0
    dJcy_dt = 0
    dJcz_dt = 0
    factor = 16*np.pi
    news_modes_conj = np.conjugate(news_modes)
    
    modes_list = construct_mode_list(ell_max=strain_modes.ell_max-1, spin_weight=-2)

    #for ell, emm_list in strain_modes.modes_list:
    for ell, emm_list in modes_list:
        for emm in emm_list:
            #if abs(emm+1)>ell or abs(emm-1)>ell:
            #    continue
            t1 = f_lm(ell, emm)*news_modes_conj.mode(ell, emm+1)
            t2 = f_lm(ell, -emm)*news_modes_conj.mode(ell, emm-1)
            dJcx_dt += strain_modes.mode(ell, emm)*(t1 + t2)
            dJcy_dt += strain_modes.mode(ell, emm)*(t1 - t2)
            dJcz_dt += emm*(strain_modes.mode(ell, emm) * news_modes_conj.mode(ell, emm)) 
    
    dJx_dt = dJcx_dt.imag/(2*factor)
    dJy_dt = -dJcy_dt.real/(2*factor)
    dJz_dt = dJcz_dt.imag/factor

    return np.array([strain_modes.time_axis, dJx_dt, dJy_dt, dJz_dt])


def compute_angular_momentum(strain_modes, 
                             news_modes,
                             t_start=None,
                             t_end=None,
                             since_peak=False,
                             inspiral_only=False):
    
    #modes_list = construct_mode_list(ell_max=strain_modes.ell_max-1, spin_weight=-2)

    if since_peak or inspiral_only:
        power = strain_modes.get_power_from_news_modes(news_modes)
        if since_peak:
            t_start = strain_modes.time_axis[np.argmax(power)]

        if inspiral_only:
            t_end = strain_modes.time_axis[np.argmax(power)]

    if t_start is None:
        t_start = strain_modes.time_axis[0]

    if t_end is None:
        t_end = strain_modes.time_axis[-1]


    dJ_dt_vec = compute_angular_momentum_evolution(strain_modes, news_modes)
    time_axis, dJx_dt, dJy_dt, dJz_dt = dJ_dt_vec
    spline_dJx_dt = InterpolatedUnivariateSpline(time_axis, dJx_dt, k=5)
    spline_dJy_dt = InterpolatedUnivariateSpline(time_axis, dJy_dt, k=5)
    spline_dJz_dt = InterpolatedUnivariateSpline(time_axis, dJz_dt, k=5)
    dJx = spline_dJx_dt.integral(t_start, t_end)
    dJy = spline_dJy_dt.integral(t_start, t_end)
    dJz = spline_dJz_dt.integral(t_start, t_end)

    return np.array([dJx, dJy, dJz])

