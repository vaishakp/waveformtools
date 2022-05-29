""" Methods to integrate functions """


#################################################
# Imports
################################################


import numpy as np

##################################################
# Fixed frequency integration
##################################################


def fixed_frequency_integrator(udata_time, delta_t, utilde_conven=None, omega0=0, order=1, zero_mode=0):
	""" Fixed frequency integrator as presented in Reisswig


	Parameters
	----------
	udata_time :	1d array
				The input data in time.
	delta_t :	float
				The time stepping.

	utilde_conven :		1d array, optional
						The conventional FFT of the samples udata_time.
	omega0 :	float, optional
				The cutoff angular frequency in the integration. Must be lower than the starting angular frequency of the input waveform.
				All frequencies below this will be neglected.
				The default value is 0.

	order :		int, optional
				The number of times to integrate the integrand in time.
				Defaults to 1.

	zero_mode :	float, optional
				The zero mode amplitude of the FFT required.
				Defaults to 0 i.e. the zero mode is removed.

	Returns
	-------

	u_integ_n_time :	1d array
						The input waveform in time-space, integrated in frequency space using FFI.

	u_integ_integ_n :	1d array
						The integrated u samples in Fourier space.

	"""

	if not utilde_conven:
		# Compute the FFT of data
		from numpy.fft import ifft
		from transforms import compute_fft, unset_fft_conven

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

	print("The chosen cutoff angular frequency is", omega0)

	if omega0 > 0:
		for index, element in enumerate(omega_axis):
			# Loop over the samples.

			# Skip the zero index
			if index != zero_index:
				# print(freq_integ[index])
				try:
					# Get the sign of the angular frequency.
					sign = int(element / abs(element))
				except:
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
