"""Methods for waveform extrapolation."""

##############################################
# Imports
##############################################
import numpy as np

import waveformtools

###############################################
# Basic utilities
###############################################


def r_to_ra_conversion(coord_radius, mass=1, spin=0):
    """Convert the isotropic co-ordinate radius parameter `r` in the ETK
    simulations into the approximate areal radius.

    Parameters
    ----------
    coord_radius: float
                  The coordinate radius in the Einstein toolkit

    mass: float, optional
          The sum of the quasi-local horizon (Christodolou) masses
          of the black holes. Defaults to 1.

    spin: float, optional
          The magnitude of the spin of the system,
          as approximated by a single Kerr black hole
          far away from the system. Defaults to 0.

    Returns
    -------
    areal_radius: float
                  The appriximate areal radius of the sphere.

    Notes
    -----
    Assumes that the system interoir to the sphere
    at co-ordinate radius `r_coord` is well approximated by a
    Kerr black hole.

    References
    ----------
    Nakano et al., (2015), Phys. Rev. D 91, 104022, in-text below Eq.[30].
    """

    areal_radius = (
        coord_radius
        * (1 + (mass + spin) / (2 * coord_radius))
        * (1 + (mass - spin) / (2 * coord_radius))
    )

    return areal_radius


############################################################################
# Perturbative extraction
############################################################################


def waveextract_to_inf_perturbative_twop5_order(
    rPsi4_rlm,
    delta_t,
    areal_radius=500,
    mass=1,
    spin=0,
    ell=2,
    emm=2,
    degree=24,
    method="CS",
):
    """Extract a numerical waveform to null infinity using perturbative
    techniques.

    This is:
            * accurate to second order in :math:`1/r`.
            * accurate to first order in Kerr mass and spin.
            * corrects for spheroidal harmonics

    Parameters
    ----------

    rPsi4_rlm: 1d array
               The extracted Weyl scalar :math:`r\\Psi_{4\\ell m}` data array
    delta_t: float
             The time stepping.
    areal_radius: 1d array
                  The areal radius of the extraction sphere.
    mass: float
          The total horizon mass of the system.
    spin: float, optional
          The effective spin of the spacetime. Defaults to 0.
    ell: int
         The polar quantum number :math:`\\ell`.
    emm: int
         The azimuthal quantum number :math:`m`.
    method: str
            The method to use for differentiation.
    degree: int
            The degree to use for dfferentiation.
    Returns
    -------
    rPsi4_inflm: 1d array
                 The waveform extracted to
                 null infninity :math:`\\mathcal{I}^+`

    References
    ----------
    This implements the definition in
    Nakano et al., (2015),  Phys. Rev. D 91, 104022 Eq.[29].
    """

    from waveformtools.differentiate import differentiate_cwaveform
    from waveformtools.integrate import fixed_frequency_integrator

    # Timeaxis

    timeaxis = np.arange(0, len(rPsi4_rlm) * delta_t, delta_t)
    # Assigning the terms. Each set of subterms in
    # a pair of paranthesis is a term.
    term_1_prefac = 1 - 2 * mass / areal_radius
    subterm_1_1 = rPsi4_rlm
    subterm_1_2_prefac = (ell - 1) * (ell + 2) / (2 * areal_radius)
    subterm_1_2, _ = fixed_frequency_integrator(
        rPsi4_rlm, delta_t, omega0=0.015
    )  # Integral_rPsi4_rlm
    subterm_1_3_prefac = (
        (ell - 1) * (ell + 2) * (ell**2 + ell - 4) / (8 * areal_radius**2)
    )
    subterm_1_3, _ = fixed_frequency_integrator(
        rPsi4_rlm, delta_t, order=2, omega0=0.015
    )  # Double_integral_rPsi4_rlm

    term_1 = term_1_prefac * (
        subterm_1_1
        + subterm_1_2_prefac * subterm_1_2
        + subterm_1_3_prefac * subterm_1_3
    )

    term_2_prefac = (2 * 1j * spin / (ell + 1) ** 2) * np.sqrt(
        (ell + 3)
        * (ell - 1)
        * (ell + emm + 1)
        * (ell - emm + 1)
        / ((2 * ell + 1) * (2 * ell + 3))
    )
    subterm_2_1 = differentiate_cwaveform(
        timeaxis, rPsi4_rlm, method=method, degree=degree
    )  # Differential of waveform
    subterm_2_2_prefac = -ell * (ell + 3) / areal_radius
    subterm_2_2 = subterm_1_1

    term_2 = term_2_prefac * (subterm_2_1 + subterm_2_2_prefac * subterm_2_2)

    term_3_prefac = (-2 * 1j * spin / ell**2) * np.sqrt(
        ((ell + 2) * (ell - 2) * (ell + emm) * (ell - emm))
        / ((2 * ell - 1) * (2 * ell + 1))
    )
    subterm_3_1 = subterm_2_1
    subterm_3_2_prefac = -(ell - 2) * (ell + 1) / areal_radius
    subterm_3_2 = subterm_1_1

    term_3 = term_3_prefac * (subterm_3_1 + subterm_3_2_prefac * subterm_3_2)

    rPsi4_inflm = term_1 + term_2 + term_3

    return rPsi4_inflm


def waveextract_to_inf_perturbative_two_order(
    rPsi4_rlm, delta_t, areal_radius=500, mass=1, ell=2
):
    """Extract a numerical waveform to null infinity using perturbative
    techniques.

    This is:
            * accurate to second order in :math:`1/r`.
            * accurate to first order in Kerr mass and spin.
            * corrects for spheroidal harmonics

    Parameters
    ----------
    rPsi4_rlm: 1d array
               The extracted Weyl scalar :math:`r\\Psi_{4\\ell m}` data array.

    delta_t: float
             The time stepping.

    areal_radius: 1d array
                  The areal radius of the extraction sphere.

    mass: float
          The total horizon mass of the system.

    spin: float, optional
          The effective spin of the spacetime. Defaults to 0.

    ell: int
         The polar quantum number :math:`\\ell`.

    emm: int
         The azimuthal quantum number :math:`m`.

    Returns
    -------
    rPsi4_inflm: 1d array
                 The waveform extracted to null infninity :math:`\\mathcal{I}^+`

    References
    ----------
    This implements the definition in Nakano et al., (2015),
    Phys. Rev. D 91, 104022 Eq.[29].
    """

    from integrate import fixed_frequency_integrator

    # Assigning the terms. Each set of subterms in
    # a pair of paranthesis is a term.
    term_1 = rPsi4_rlm

    term_2_prefac = -(ell - 1) * (ell + 2) / (2 * areal_radius)
    subterm_2_1, _ = fixed_frequency_integrator(
        rPsi4_rlm, delta_t, omega0=0.015
    )  # Integral_rPsi4_rlm
    term_2 = term_2_prefac * subterm_2_1

    term_3_prefac = (
        (ell - 1) * (ell + 2) * (ell**2 + ell - 4) / (8 * areal_radius**2)
    )
    subterm_3_1, _ = fixed_frequency_integrator(
        rPsi4_rlm, delta_t, order=2, omega0=0.015
    )  # Double_Integral_rPsi4_rlm
    term_3 = term_3_prefac * subterm_3_1

    term_4_prefac = -3 * mass / (2 * areal_radius**2)
    subterm_4_1 = subterm_2_1
    term_4 = term_4_prefac * subterm_4_1

    rPsi4_inflm = term_1 + term_2 + term_3 + term_4

    return rPsi4_inflm


def waveextract_to_inf_perturbative_one_order(
    u_ret, rPsi4_rlm, areal_radius=500, ell=2
):
    """Extract a numerical waveform to null infinity using perturbative
    techniques.

    This is:
            * accurate to second order in :math:`1/r`.
            * accurate to first order in Kerr mass and spin.
            * corrects for spheroidal harmonics

    Parameters
    ----------
    u_ret: 1d array
           The retarted time array at the location r = areal_radius.

    rPsi4_rlm: 1d array
               The extracted Weyl scalar :math:`r\\Psi_{4\\ell m}` data array

    areal_radius: 1d array
                  The areal radius of the extraction sphere.

    mass: float
          The total horizon mass of the system.

    ell: int
         The polar quantum number :math:`\\ell`.

    emm: int
         The azimuthal quantum number :math:`m`.

    Returns
    -------
    rPsi4_inflm: 1d array
                 The waveform extracted to null infninity :math:`\\mathcal{I}^+`

    References
    ----------
    This implements the definition in
    Nakano et al., (2015),  Phys. Rev. D 91, 104022 Eq.[29].
    """

    # Get the amplitude and phase
    A_lm, _ = waveformtools.xtract_camp_phase(rPsi4_rlm.real, rPsi4_rlm.imag)

    # Get the time stepping
    delta_t = u_ret[1] - u_ret[0]

    # Get the waveform instantaneous frequency
    omega_lm = waveformtools.get_waveform_angular_frequency(
        rPsi4_rlm, delta_t=delta_t, timeaxis=u_ret
    )

    # Assigning the terms. Each set of subterms in
    # a pair of paranthesis is a term.

    # The amplitude correction factor
    A_lm_correction = 0.5 * np.power(
        ((ell * (ell + 1) / (2 * omega_lm * areal_radius))), 2
    )

    # The phase correction factor.
    sin_Phase_correction = ell * (ell + 1) / (2 * omega_lm * areal_radius)
    cos_Phase_correction = np.sqrt(1 - np.power(sin_Phase_correction, 2))
    Phase_correction = cos_Phase_correction + 1j * sin_Phase_correction

    # The extrapolated waveform

    A_lm_corrected = A_lm * (1 + A_lm_correction)
    Phi_lm_corrected = np.exp(-1j * omega_lm * u_ret) * Phase_correction

    rPsi4_inflm = A_lm_corrected * Phi_lm_corrected

    return rPsi4_inflm
