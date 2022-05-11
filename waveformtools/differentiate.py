''' Tools for differentiating a function '''

#######################################################
# Imports
#######################################################

import numpy as np

########################################################
# Chebyshev differentiation
########################################################

def Chebyshev_differential(x_data, y_data, order = 1, degree = 5):
	''' Differentiate a function using the Chebyshev expansion.


	Inputs
	---------


	x_data :	1d array
				The x data.
	y_data :	1d array
			The y data.

	order :	int
			The number of times to differentiate.

	degree :	int
				The number of basis functions to use.


	Returns
	----------

	dydx_data :	1d array
				The differentiated data.

	'''

	# Find the basis coefficients.
	from numpy.polynomial.chebyshev import chebfit, chebder, chebval

	cheb_coeffs = chebfit(x_data, y_data, deg=degree)

	# compute the derivative
	cheb_der_coeffs = chebder(cheb_coeffs, m=order)


	# Change the basis to that of x_data
	dydx_data = chebval(x_data, cheb_der_coeffs)


	return dydx_data







########################################################
# Fourier differentiation
########################################################


def Fourier_differential(delta_x, udata_x=None, utilde_conven=None, omega0=np.inf, order=1, zero_mode=0, taper=True):
	''' Fixed frequency differentiation, the inverse of the Fixed frequency integration as presented in Reisswig et al.
		This function takes in a function and returns its nth order derivative differential taken in the frequency domain.


	Inputs
	-------
	xaxis :	1d array
			The co-ordinate space axis.
	udata_x :	1d array
				The data to be differentiated, expressed in coordinate space.
	omega0 :	float, optional
				The cutoff angular frequency in the integration. Must be lower than the starting angular frequency of the input waveform.
	order :	int, optional
			The number of times to differentiate the integrand in time.
	zero_mode :	float, optional
				The zero mode amplitude of the FFT required.
	taper :	bool
			Whether or not to taper the real co-ordinate space data.
	Returns
	-------
	udata_differentiated :	1d array
							The input waveform in time-space, integrated in frequency space using FFI.

	utilde_differentiated :	1d array
							The FFT of the frixed frequency differentiated array in good conventions.

	freq_axis :	1d array
				The frequency axis of the FFT of data.

	Note
	----

	The returned differentiated function of a real udata_x in real co-ordinate space is a complex number
	due to the numerical inaccuracies. Take the real part of udata_differentiated if the iunput udata_x
	is real.

	'''


	if not utilde_conven:
		# Compute the FFT of data
		from numpy.fft import ifft
		from waveformtools.transforms import find_fft, unset_fft_conven
		from waveformtools.waveformtools import taper
		udata_x = taper(udata_x, delta_t = delta_x)
		x_axis = udata_x.sample_times
		udata_x = np.array(udata_x)
		freq_axis, utilde_conven		= find_fft(udata_x, delta_x)


		# Find the length of the input data.
		Nlen					= len(udata_x)

	else:
		Nlen = len(utilde_conven)


	# Find the location of the zero index.
	if Nlen%2==0:
		zero_index = int(Nlen/2)
	else:
		zero_index = int((Nlen+1)/2)

	# Construct the angular frequency axis.
	omega_axis = 2*np.pi*freq_axis

	#print(omega_axis)
	print('The cutoff angular frequency is', omega0)

	# Alter the frequency axis if omega0 < inf

	if omega0<np.inf:
		for index, element in enumerate(omega_axis):
			# Loop over the samples.

			# Skip the zero index
			if index!=zero_index:
			#print(freq_integ[index])
				try:
				# Get the sign of the angular frequency.
					sign = int(element/ abs(element))
				except:
					sign = 1

				#print(sign)
				# Change the angular frequency if its magnitude is below a given omega0.
				if abs(element) > omega0:
					omega_axis[index] = sign*omega0

		# Set the zero frequency element separately.
		omega_axis[zero_index]		= omega0

	#print(omega_axis)
	# Differentiate in frequency space
	utilde_differentiated		= np.power((-1j*omega_axis), order) * utilde_conven

	# Set the zero mode amplitude
	if not zero_mode:
		utilde_differentiated[zero_index] = 0

	# Get the inverse fft
	utilde_differentiated_np	= unset_fft_conven(utilde_differentiated)


	udata_differentiated		= ifft(utilde_differentiated_np)

	return udata_differentiated, utilde_differentiated, x_axis, freq_axis

#########################################################
# Finite difference differentiation
########################################################



def differentiate(data, delta_t):
	""" Central difference derivative calculator. Forward/ backward Euler near the boundaries.

	Inputs
	--------

	data :	1d array
			The 1d data.
	delta_t :	float
				The time step in units of t/M.

	Returns
	--------

	dAdt :	1d array
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
	""" Five point difference derivative calculator.  Not accurate near the boundaries.


	Inputs
	--------

	data :	1d array
			The 1d data.
	delta_t :	float
				The time step in t/M.

	Returns
	---------

	dAdt :	1d array
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
	der_data.append(derN)

	return der_data



def differentiate3(data, delta_t):
	""" Seven point difference derivative calculator. Not accurate near the boundaries.


	Inputs
	---------

	data :	1d array
			The 1d data.
	delta_t :	float
				The time step in t/M.

	Returns
	----------

	dAdt :	1d array
			The derivative.

	"""

	# The number of points on one direction.
	order		= 3
	# The seven point stencil.
	coeffs		= np.array([-1, 9, -45, 0, 45, -9, 1])
	divide		= 60

	# A list to hold the derivatives.
	der_data	= []


	# Near the boundaries

	# n=0, N-1
	der0		= (data[1] - data[0]) / delta_t
	derNm1		= (data[-1] - data[-2]) / delta_t

	der_data.append(der0)


	# for n=1, N-2
	der1		= (data[2] - data[0]) / (2 * delta_t)
	derNm2		= (data[-1] - data[-3]) / (2 * delta_t)

	der_data.append(der1)

	# For n=2, N-3
	stencil		= np.array([1, -8, 0, 8, -1])/12
	data_vec	= data[:5]


	der2		= np.dot(stencil, data_vec) /delta_t

	data_vec	= data[-5:]

	derNm3		= np.dot(stencil, data_vec) /delta_t



	for index in range(order, len(data) - order):
		data_subarray = data[index - order : index + order + 1]
		der_data.append(np.dot(coeffs, data_subarray) / (divide * delta_t))

	der_data.append(derNm3)
	der_data.append(derNm2)
	der_data.append(derNm1)

	return der_data


def differentiate4(data, delta_t):
	""" Nine point difference derivative calculator. Not accurate near the boundaries.


	Inputs
	--------

	data :	1d array
			The 1d data.
	delta_t :	float
				The time step in t/M.

	Returns
	---------

	dAdt :	1d array
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
	der0		= (data[1] - data[0]) / delta_t
	derNm1		= (data[-1] - data[-2]) / delta_t

	der_data.append(der0)


	# for n=1, N-2
	der1		= (data[2] - data[0]) / (2 * delta_t)
	derNm2		= (data[-1] - data[-3]) / (2 * delta_t)

	der_data.append(der1)

	# For n=2, N-3
	stencil		= np.array([1, -8, 0, 8, -1])/12
	data_vec	= data[:5]


	der2		= np.dot(stencil, data_vec) /delta_t

	data_vec	= data[-5:]

	derNm3		= np.dot(stencil, data_vec) /delta_t


	der_data,append(der2)

	# For n=3, N-4
	stencil		= np.array([-1, 9, -45, 0, 45, -9, 1])/60

	data_vec	= data[:7]


	der3		= np.dot(stencil, data_vec) /delta_t

	data_vec	= data[-7:]

	derNm4		= np.dot(stencil, data_vec) /delta_t


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
	""" Eleven point difference derivative calculator. Not accurate near the boundaries.


	Inputs
	---------

	data :	1d array
			The 1d data.
	delta_t :	float
				The time step in t/M.

	Returns
	---------

	dAdt :	1d array
			The derivative of the data.

	"""

	#The number of points on one side.
	order = 5
	# The stencil.
	coeffs = np.array([-2, 25, -150, 600, -2100, 0, 2100, -600, 150, -25, 2])
	# The divison factor.
	divide = 2520

	# A list to hold the derivatives.
	der_data = []

	# Near the boundaries

	# n=0, N-1
	der0		= (data[1] - data[0]) / delta_t
	derNm1		= (data[-1] - data[-2]) / delta_t

	der_data.append(der0)


	# for n=1, N-2
	der1		= (data[2] - data[0]) / (2 * delta_t)
	derNm2		= (data[-1] - data[-3]) / (2 * delta_t)

	der_data.append(der1)

	# For n=2, N-3
	stencil		= np.array([1, -8, 0, 8, -1])/12
	data_vec	= data[:5]


	der2		= np.dot(stencil, data_vec) /delta_t

	data_vec	= data[-5:]

	derNm3		= np.dot(stencil, data_vec) /delta_t


	der_data,append(der2)

	# For n=3, N-4
	stencil		= np.array([-1, 9, -45, 0, 45, -9, 1])/60

	data_vec	= data[:7]


	der3		= np.dot(stencil, data_vec) /delta_t

	data_vec	= data[-7:]

	derNm4		= np.dot(stencil, data_vec) /delta_t


	# For n=4, N-5
	stencil		= np.array([3, -32, 168, -672, 0, 672, -168, 32, 3])/840

	data_vec	= data[:9]


	der4		= np.dot(stencil, data_vec) /delta_t

	data_vec	= data[-19:]

	derNm5		= np.dot(stencil, data_vec) /delta_t

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

