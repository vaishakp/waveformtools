''' Methods to handle functions on a sphere. '''





##################################
# Imports
#################################

import numpy as np




def decompose_in_SWSHs(waveform, spin_weight=-2, ell_max=8, emm='all'):
	''' Decompose a given function on a sphere in Spin Weighted Spherical Harmonics

	Parameters
	----------

	waveform :		list
					A list that contains as its items the waveform defined on the sphere as an array of shape [ntheta, nphi]. Each item in the list may donote
					an instant of time or frequency.

	spin_weight :	 int, optional
					 The spin weight of the waveform. It defaults to -2 for a gravitational waveform.

	ell_max :	int, optional
				The maximum value of the :math:`\\ell' polar quantum number. Defaults to 6=8.

	gridinfo :		class instance
					The class instance that contains the properties of the spherical grid.


	Returns
	-------

	SWSH_coeffs :	list
					The SWSH coefficients of the waveform. It may be a list composed of a single floating point number or a 1d array (denoting time or frequency dimension).
					The waveform can have angular as well as time dimentions. The nesting order will be that, given the list `non_boosted_waveform', each
					item refers to a one dimensional array in time/ frequency of SWSH coefficients.


	Notes
	-----

	Assumes that the sphere on which this decomposition is carried out is so far out
	that the coordinate system is spherical polar and the poper area is the
	same as its co-ordinate area.

	'''


	# Find out if the unboosted waveform is a single number or defined on a spherical grid.
	onepoint = isinstance(waveform[0], float)


	if not onepoint:
		# Get the spherical grid shape.
		ntheta, nphi = np.array(waveform[0]).shape

	# Compute the meshgrid for theta and phi.
	theta = gridinfo.theta(ntheta=ntheta, nphi=nphi)

	 # Compute the meshgrid for theta and phi.
	phi = gridinfo.phi(ntheta=ntheta, nphi=nphi)

	decomposed_waveforms = {}

	for item in waveform:
		# Integrate on the sphere for decomposition into SWSHs

		# define m values.
		if m=='all':
			m = np.arange(-l,l+1)

		# Convert input to arrays.

		integrand_data		  = np.array(item)


		# Step 1: Compute the surface integral.

		# Assign data vectors.
		lpole  = {}

		# Check if data includes ghost zones or not.
		s1, s2 = integrand_data.shape

		if s1*s2 < info.npixmax:

			start			= 0
			end_dtheta		= 2*info.nghosts
			end_dphi		= 2*info.nghosts
		else:

			start			= info.nghosts
			end_dtheta		= info.nghosts
			end_dphi		= info.nghosts

		# Compute the meshgrid for theta and phi.
		theta		  = gridinfo.theta(ntheta=ntheta, nphi=nphi)

		# Compute the meshgrid for theta and phi.
		phi			  = gridinfo.phi(ntheta=ntheta, nphi=nphi)

		sqrt_met_det  = np.sqrt(np.power(np.sin(theta), 2))


		integrand_ij  = integrand_data

		darea		  = sqrt_metdet * info.dtheta * info.dphi

		for ell_index in range(ell_max):

			multipoles_all = {}
			for m_index in range(len(m)):

				# Decompose into seperate m modes.

				# m value.
				emm_val			=	int(emm_all[m_index])

				# Spin weighted spherical harmonic function at (theta, phi)
				Ybasis_fun		=	Yslm(spin_weight, ell, m_val, theta, phi)

				# Integrate to obtain the multipole of order l.

				# Integration for real and imaginary parts of the data separately.
				# Integrate the function

				# Using quad
				multipole_emm	=	quad_on_sphere(integrand_ij * Ybasis_fun * darea, info)
				#multipole_emm	 =	 np.sum(integrand_ij * Ybasis_fun * darea)

				multipole_ell.update({ emm_val : multipole_emm })
			multipoles_all.update({ ell_index : multipole_ell })

		# Return the computed multipole.
		return multipoles_all


def quad_on_sphere(integrand, info, kind='third'):
	''' Integrate on a sphere using the scipy.quad method

	Parameters
	----------

	integrand :		2d array
					The two dimensional integrand array defined on the sphere.

	info :		class instance
				The class instance that contains the properties of the spherical grid.

	kind :		str
				The interpolation order to use in integration.
	Returns
	-------

	final_integral : float
				  The given integrand integrated over the sphere.

	final_errs :	float
					The accumulated errors.

	Notes
	-----

	Assumes that the sphere is a unit round sphere.


	'''

	# Step 0: Get the grid properties

	# Compute the meshgrid for theta and phi.
	theta_1d = info.theta_1d(ntheta=ntheta, nphi=nphi)
	#theta
	 # Compute the meshgrid for theta and phi.
	phi_1d	 = info.phi_1d(ntheta=ntheta, nphi=nphi)


	# imports
	from scipy.interpolate import interp1d
	from scipy.integrate import quad

	theta_first_integral_val  = []
	theta_first_integral_errs = []
	# Step 1: integrate along the theta direction
	for phi_index in range(info.nphi):
		# Interpolate the integrand.

		integrand_phi = integrand[:, phi_index]

		integrand_phi_interp_func = interp1d(theta_1d, integrand_phi, kind=kind)

		# Integrate on the phi plane
		integral_phi_val, integral_phi_errs  = quad(integrand_phi_interp_func, 0, np.pi)

		theta_first_integral_vals.append(integral_phi_vals)
		theta_first_integral_errs.append(integral_phi_errs)

	# Step 2: integrate along the phi direction

	# Interpolate the integrand.

	integrand_theta = theta_first_integral

	integrand_theta_interp_func = interp1d(phi_1d, integrand_theta, kind=kind)

	# Integrate on the theta plane
	final_integral, semi_final_errs = quad(integrand_theta_interp_func, 0, 2*np.pi)

	# Get final errors
	final_errs = semi_final_errs + np.sum(np.array(theta_first_integral_errs))

	return final_integral, final_errs
