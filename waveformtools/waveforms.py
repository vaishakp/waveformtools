''' Classes for handling the waveform modes or data defined on spheres.


Classes
-------
spherical_array:	A 2D data-type.
					Stores and manages two-dimensional data on surfaces of spherical topology.
modes_array: A data-type.
			 Handle and work with mode coefficients.
'''


import numpy as np
import h5py
from waveformtools.waveformtools import message
from qlmtools import Yslm_new

#from numba import jit, njit
#from numba import jitclass			 # import the decorator
#from numba import int32, float64, complex128	 # import the types
#import numba as nb
#from numba.experimental import jitclass

#spec_sp = { 'label' : nb.types.string,
#			'time_axis' : nb.double[:],
#			'frequency_axis' : nb.double[:],
#			'data' : nb.complex128[:, :, :],
#			'base_dir' : nb.types.string,
#			'file_name' : nb.types.string,
#			'spin_weight' : nb.int32

#}
#@jitclass(spec_sp)
class spherical_array:
	"""A class for handling waveforms on a sphere.


	Attributes
	----------

	label:	str
			The label of the waveform data.
	time_axis:	1d array
				The time axis of the data.
	frequency_axis:	1d array
					The frequency axis if the data
					is represented in frequency domain.
	grid_info:	spherical_grid
				An instance of the `spherical_grid` class.
	data_len:	int
				The length of the data along the time axis.

	Methods
	-------

	delta_t:
			Fetch the time stepping `delta_t`.
	to_modes_array:
					Find the waveform expressed in the
					SWSH basis.
	boost:
			Boost the waveform.
	supertranslate:
					Supertranslate the waveform.

	"""

	def __init__(
		self,
		label=None,
		time_axis=None,
		frequency_axis=None,
		data=None,
		data_dir=None,
		file_name=None,
		grid_info=None,
		spin_weight = 2,
	):

		self.label = label
		#self.base_dir = base_dir  # The base directory containing the
		self.data = data
		self.file_name = file_name
		self.data_dir = data_dir
		self.time_axis = time_axis
		self.frequency_axis = frequency_axis
		self.grid_info = grid_info
		self.spin_weight = spin_weight

	def delta_t(self, value=None):
		"""Sets and returns the value of time stepping :math:`dt`.

		Parameters
		----------
		value : float, optional
						The value of :math:`dt`
						to set to the attribute.

		Returns
		-------
		delta_t:	float
										 Sets the attribute.
		"""

		# if not self.delta_t:
		if not value:
			try:
				delta_t = self.time_axis[1] - self.time_axis[0]
			except:
				print("Please input the value of `delta_t` or supply the `time_axis` to the waveform.")
		else:
			delta_t = value

		return delta_t

	@property
	def data_len(self):
		"""Returns the length of the time/frequency axis.

		Returns
		-------
		data_len:	int
								The length of the time/frequency axis.
		"""

		try:
			data_len = len(self.time_axis)

		except:
			data_len = len(self.frequency_axis)

		return data_len

	def to_modes_array(self, grid_info=None, spin_weight=None, ell_max=8):
		"""Decompose a given spherical array function on a sphere
		into Spin Weighted Spherical Harmonic modes.

		Parameters
		----------
		spin_weight:	int, optional
						The spin weight of the waveform. It defaults to -2 for a gravitational waveform.
		ell_max:	int, optional
					The maximum value of the :math:`\\ell` polar quantum number. Defaults to 8.
		grid_info:	class instance
					The class instance that contains the properties of the spherical grid.

		Returns
		-------
		waveforms_modes:	modes_array
							An instance of the `modes_array` class containing the decomposed modes.

		Notes
		-----
		1. Assumes that the sphere on which this decomposition is carried out is so far out
		   that the coordinate system is spherical polar on a round sphere.
		2. Assumes that the poper area is the same as its co-ordinate area.
		3. Ensure that the label of the input spherical array indicates whether
		   it is a time domain data or frequency domain data.
		"""

		if grid_info is None:
			if self.grid_info is None:
				message('Please specify the grid specs. Assuming defaults.')
				grid_info = waveformtools.grids.spherical_array()
				self.grid_info = grid_info
			else:
				grid_info = self.grid_info

		if spin_weight is None:
			if self.spin_weight is None:
				message('Please specify spin weight. Assuming 2')
				spin_weight = 2
				self.spin_weight = spin_weight

			else:
				spin_weight = self.spin_weight



		spin_weight=abs(spin_weight)
		# Compute the meshgrid for theta and phi.
		theta, phi = grid_info.meshgrid

		# Create a modes array object

		# Create a modes list.
		modes_list = construct_mode_list(ell_max, spin_weight=spin_weight)

		if not self.label:
			label = "decomposed_time_domain"
		else:
			label = self.label

		# Create a mode array for the decomposed_waveform
		waveform_modes = modes_array(label=label, ell_max=ell_max, spin_weight=spin_weight)

		# Inherit the time or frequency axis.
		if "time" in label:
			axis = self.time_axis
			waveform_modes.time_axis = self.time_axis
		else:
			axis = self.frequency_axis
			waveform_modes.frequency_axis = self.frequency_axis

		# Create the modes_array
		waveform_modes._create_modes_array(ell_max=ell_max, data_len=self.data_len)
		waveform_modes.modes_list = modes_list
		# The area element on the sphere
		# Compute the meshgrid for theta and phi.
		theta, phi = grid_info.meshgrid

		#sqrt_met_det = np.sin(theta)
		sqrt_met_det = np.sqrt(np.power(np.sin(theta), 2))

		darea = sqrt_met_det * grid_info.dtheta * grid_info.dphi

		from qlmtools import Yslm_new

		modes_list = [item for item in modes_list if item[0]>=spin_weight]

		for mode in modes_list:
			ell_value, all_emm_values = mode

			for emm_value in all_emm_values:
				# m value.

				# Spin weighted spherical harmonic function at (theta, phi)

				Ybasis_fun = np.conj(Yslm_new(spin_weight, ell=ell_value, emm=emm_value, theta_grid=theta, phi_grid=phi))

				Ydarea = Ybasis_fun * darea
				#print(Ydarea.shape)
				#print(full_integrand)
				# Using quad
				multipole_ell_emm = np.tensordot(self.data, Ydarea, axes=((0, 1), (0, 1)))

				#print(f'l{ell_value}m{emm_value}', multipole_ell_emm)

				waveform_modes.set_mode_data(ell_value, emm_value, multipole_ell_emm)

		return waveform_modes

	def boost(self, conformal_factor):
		""" Boost the waveform given the unboosted waveform and the boost conformal factor.

		Parameters
		----------

		self:	  spherical_array
								A class instance of `spherical array`.

		conformal_factor:		2d array
								The conformal factor for the Lorentz transformation. It may be a single floating point number or an array on a spherical grid. The array will be of dimensions
								[ntheta, nphi]

		grid_info:		 class instance
						The class instance that contains the properties of the spherical grid.


		Returns
		-------

		boosted_waveform:	  sp_array
							  The class instance `sp_array` that
							  contains the boosted waveform.
		"""

		from waveformtools.waveforms import spherical_array

		# Compute the boosted waveform on the spherical grid on all the elements.

		boosted_waveform_data = np.transpose(self.data, (2, 0, 1)) * conformal_factor
		#boosted_waveform_data = np.multiply(self.data, conformal_factor[:, :, None])
		#boosted_waveform_data = boosted_waveform_item
		#boosted_waveform_data = np.array([np.transpose(item)*conformal_factor for item in np.transpose(self.data)])

		# Construct a 2d waveform array object
		boosted_waveform = spherical_array(grid_info=self.grid_info, data=np.transpose(np.array(boosted_waveform_data), (1, 2, 0)))

		# Assign other attributes.
		boosted_waveform.label = "boosted " + self.label
		boosted_waveform.time_axis = self.time_axis

		return boosted_waveform


	def supertranslate(self, supertransl_alpha_sp, order=1):

		"""Supertranslate the :math:`\\Psi_{4\\ell m}` waveform modes, give the,
		the supertranslation parameter and the order.

		Parameters
		----------
		supertransl_alpha_modes :  modes_array
														   The modes_array containing the
														   supertranslation mode coefficients.
		grid_info:		class instance
										The class instance that contains
										the properties of the spherical grid
										using which the computations are
										carried out.
		order:		int
								The number of terms to use to
								approximate the supertranslation.

		Returns
		-------
		Psi4_supertransl:	  modes_array
												  A class instance containg the
												  modes of the supertranslated
												  waveform :math:`\\Psi_4`.
		"""

		# Create a spherical_array to hold the supertranslated waveform
		Psi4_supertransl_sp = spherical_array(grid_info=self.grid_info, label='{} -> supertranslated time'.format(self.label))

		delta_t = float(self.delta_t())


		# Set the data.
		data = 0
		#data = self.data
		#Psi4_supertransl_sp.data = self.data
		delta = 0
		count = 0
		from waveformtools.differentiate import differentiate5_vec_numba
		for index in range(order):
			#print(f'{index} loop')
			dPsidu = self.data
			for dorder in range(index+1):
				#print(f'differentiating {dorder+1} times')
				dPsidu = differentiate5_vec_numba(dPsidu, delta_t)

			print('Incrementing...')
			#delta = np.power(supertransl_alpha_sp.data, index+1) * dPsidu / np.math.factorial(index+1)
			#print(delta/self.data)

			data += np.power(supertransl_alpha_sp.data, index+1) * dPsidu / np.math.factorial(index+1) #delta

		data+=self.data
		print('Equal to original waveform?', (data == self.data).all())

		Psi4_supertransl_sp.data = data
		Psi4_supertransl_sp.time_axis = self.time_axis
		print('Done.')
		return Psi4_supertransl_sp



	def load_shear_data(self, data_dir=None, grid_info=None, bh=0):
		''' Load the 2D shear data from h5 files.

		Parameters
		----------
		file_name: str
					The name of the file containing data.
		data_dir: str
					The name of the directory containing data.
		grid_info: class instance
					An instance of the grid_info class.
		bh: int
			The black hole number (0, 1 or 2)

		'''

		import sys
		import re
		import json

		#if file_name is None:
		#	if self.file_name is None:
		#		print('Please supply the file name!')
		#	else:
		#		file_name = self.file_name
		#else:
		#	if self.file_name is None:
		#		self.file_name = file_name

		if data_dir is None:
			if self.data_dir is None:
				print('Please supply the data directory!')
			else:
				data_dir = self.data_dir
		else:
			if self.data_dir is None:
				self.data_dir = data_dir


		if grid_info is None:
			if self.grid_info is None:
				print('Please supply the grid spec!')
			else:
				grid_info = self.grid_info
		else:
			if self.grid_info is None:
				self.grid_info = grid_info
		# get the full path.

		file_name=	f'qlm_npsigma[{bh}].xy.h5'

		full_path = self.data_dir + file_name

		cflag = 0

		nghosts = grid_info.nghosts
		ntheta = grid_info.ntheta
		nphi = grid_info.nphi

		# Open the modes file.
		with h5py.File(full_path, "r") as wfile:

			# Get all the mode keys.
			modes_keys_list = list(wfile.keys())
			#modes_keys_list = sorted(modes_keys_list)

			# Get the mode keys containing the data.
			modes_keys_list = [item for item in modes_keys_list if 'it=' in item]

			# Get the itaration numbers.
			iteration_numbers = sorted(get_iteration_numbers_from_keys(modes_keys_list))
			#sargs = np.argsort(iteration_numbers)
			#iteration_numbers = iteration_numbers[sargs]
			modes_keys_list = sort_keys(modes_keys_list)
			# Construct the data array.

			data_array = []

			for key in modes_keys_list:
				#data_item = np.array(wfile[key])
				#print(data_item.shape)
				data_item = np.array(wfile[key])[nghosts: nphi-nghosts, nghosts: ntheta-nghosts]
				data_item = data_item['real'] + 1j*data_item['imag']
				data_array.append(data_item)


		self.data = np.transpose(np.array(data_array), (2, 1, 0))

		self.iteration_axis = np.array(iteration_numbers)

		#########################################################
		# Load inv_coords data
		#########################################################

		inv_file_name = f'qlm_inv_z[{bh}].xy.h5'

		# get the full path.
		full_path = self.data_dir + inv_file_name

		# Open the modes file.
		with h5py.File(full_path, "r") as wfile:

			# Get all the mode keys.
			modes_keys_list = list(wfile.keys())
			#modes_keys_list = sorted(modes_keys_list)

			# Get the mode keys containing the data.
			modes_keys_list = [item for item in modes_keys_list if 'it=' in item]

			modes_keys_list = sort_keys(modes_keys_list)
			data_array = []

			for key in modes_keys_list:
				data_item = np.array(wfile[key])[nghosts: nphi-nghosts, nghosts: ntheta-nghosts]
				#data_item = data_item['real'] + 1j*data_item['imag']
				data_array.append(data_item)


		self.invariant_coordinates_data = np.transpose(np.array(data_array), (2, 1, 0))



		#########################################################
		# Load metric determinant  data
		#########################################################

		twometric_qtt_file_name = f'qlm_qtt[{bh}].xy.h5'
		twometric_qtp_file_name = f'qlm_qtp[{bh}].xy.h5'
		twometric_qpp_file_name = f'qlm_qpp[{bh}].xy.h5'


		# set the full path.
		full_path = self.data_dir + twometric_qtt_file_name

		# Open the modes file.
		with h5py.File(full_path, "r") as wfile:

			# Get all the mode keys.
			modes_keys_list = list(wfile.keys())
			#modes_keys_list = sorted(modes_keys_list)

			# Get the mode keys containing the data.
			modes_keys_list = [item for item in modes_keys_list if 'it=' in item]

			modes_keys_list = sort_keys(modes_keys_list)

			qtt_data_array = []

			for key in modes_keys_list:
				data_item = np.array(wfile[key])[nghosts: nphi-nghosts, nghosts: ntheta-nghosts]
				#data_item = data_item['real'] + 1j*data_item['imag']
				qtt_data_array.append(data_item)

		qtt_data_array = np.array(qtt_data_array)
		qtt_data_array = np.transpose(qtt_data_array, (2, 1, 0))

		# set the full path.
		full_path = self.data_dir + twometric_qtp_file_name

		# Open the modes file.
		with h5py.File(full_path, "r") as wfile:

			# Get all the mode keys.
			modes_keys_list = list(wfile.keys())
			#modes_keys_list = sorted(modes_keys_list)

			# Get the mode keys containing the data.
			modes_keys_list = [item for item in modes_keys_list if 'it=' in item]

			modes_keys_list = sort_keys(modes_keys_list)

			qtp_data_array = []

			for key in modes_keys_list:
				data_item = np.array(wfile[key])[nghosts: nphi-nghosts, nghosts: ntheta-nghosts]
				#data_item = data_item['real'] + 1j*data_item['imag']
				qtp_data_array.append(data_item)


		qtp_data_array = np.array(qtp_data_array)
		qtp_data_array = np.transpose(qtp_data_array, (2, 1, 0))

		# set the full path.
		full_path = self.data_dir + twometric_qpp_file_name

		# Open the modes file.
		with h5py.File(full_path, "r") as wfile:

			# Get all the mode keys.
			modes_keys_list = list(wfile.keys())
			#modes_keys_list = sorted(modes_keys_list)


			# Get the mode keys containing the data.
			modes_keys_list = [item for item in modes_keys_list if 'it=' in item]

			modes_keys_list = sort_keys(modes_keys_list)

			qpp_data_array = []

			for key in modes_keys_list:
				data_item = np.array(wfile[key])[nghosts: nphi-nghosts, nghosts: ntheta-nghosts]
				#data_item = data_item['real'] + 1j*data_item['imag']
				qpp_data_array.append(data_item)

		qpp_data_array = np.array(qpp_data_array)
		qpp_data_array = np.transpose(qpp_data_array, (2, 1, 0))

		sqrt_met_det = np.sqrt(np.linalg.det(np.transpose(np.array([[qtt_data_array, qtp_data_array], [qtp_data_array, qpp_data_array]]), (2, 3, 4, 0, 1))))


		self.sqrt_met_det_data = sqrt_met_det




	def to_shear_modes_array(self, grid_info=None, spin_weight=None, ell_max=8):
		"""Decompose a given spherical array function on a sphere
		into Spin Weighted Spherical Harmonic modes.

		Parameters
		----------
		spin_weight:	int, optional
						The spin weight of the waveform. It defaults to -2 for a gravitational waveform.
		ell_max:	int, optional
					The maximum value of the :math:`\\ell` polar quantum number. Defaults to 8.
		grid_info:	class instance
					The class instance that contains the properties of the spherical grid.

		Returns
		-------
		waveforms_modes:	modes_array
							An instance of the `modes_array` class containing the decomposed modes.

		Notes
		-----
		1. Assumes that the sphere on which this decomposition is carried out is so far out
		   that the coordinate system is spherical polar on a round sphere.
		2. Assumes that the poper area is the same as its co-ordinate area.
		3. Ensure that the label of the input spherical array indicates whether
		   it is a time domain data or frequency domain data.
		"""

		if grid_info is None:
			if self.grid_info is None:
				message('Please specify the grid specs. Assuming defaults.')
				grid_info = waveformtools.grids.spherical_array()
				self.grid_info = grid_info
			else:
				grid_info = self.grid_info

		if spin_weight is None:
			if self.spin_weight is None:
				message('Please specify spin weight. Assuming 2')
				spin_weight = 2
				self.spin_weight = spin_weight

			else:
				spin_weight = self.spin_weight



		spin_weight=abs(spin_weight)
		# Compute the meshgrid for theta and phi.
		theta, phi = grid_info.meshgrid

		# Create a modes array object

		# Create a modes list.
		modes_list = construct_mode_list(ell_max, spin_weight=spin_weight)

		if not self.label:
			label = "decomposed_time_domain"
		else:
			label = self.label

		# Create a mode array for the decomposed_waveform
		waveform_modes = modes_array(label=label, ell_max=ell_max, spin_weight=spin_weight)

		# Inherit the time or frequency axis.
		if "time" in label:
			axis = self.time_axis
			waveform_modes.time_axis = self.time_axis
		else:
			axis = self.frequency_axis
			waveform_modes.frequency_axis = self.frequency_axis

		# Create the modes_array
		waveform_modes.time_axis = self.time_axis[:]
		sargs = np.argsort(waveform_modes.time_axis)
		#print(sargs)
		waveform_modes.time_axis = waveform_modes.time_axis

		waveform_modes._create_modes_array(ell_max=ell_max, data_len=self.data_len)
		waveform_modes.modes_list = modes_list
		# The area element on the sphere
		# Compute the meshgrid for theta and phi.
		theta, phi = grid_info.meshgrid

		phi = np.transpose(np.array([phi for index in range(len(self.time_axis))]), (1, 2, 0))

		#sqrt_met_det = np.sin(theta)
		#sqrt_met_det = np.sqrt(np.power(np.sin(theta), 2))

		darea = self.sqrt_met_det_data * grid_info.dtheta * grid_info.dphi

		theta			   =   np.emath.arccos(self.invariant_coordinates_data)

		from qlmtools import Yslm_new

		modes_list = [item for item in modes_list if item[0]>=spin_weight]

		for mode in modes_list:
			ell_value, all_emm_values = mode

			for emm_value in all_emm_values:
				# m value.
				#print(f'Processing l{ell_value} m{emm_value}')
				# Spin weighted spherical harmonic function at (theta, phi)

				Ybasis_fun = np.conj(Yslm_new(spin_weight=spin_weight, ell=ell_value, emm=emm_value, theta_grid=theta, phi_grid=phi))
				#Ybasis_fun = np.array([np.conj(Yslm_new(spin_weight=spin_weight, ell=ell_value, emm=emm_value, theta_grid=theta[:, :, index], phi_grid=phi[:, :, index])) for index in range(self.data_len)])
				#Ybasis_fun = np.transpose(Ybasis_fun, (1, 2, 0))
				#print('Ybasis_fun', Ybasis_fun.shape)
				Ydarea = Ybasis_fun * darea
				#print('Ydarea', Ydarea.shape)
				#print(full_integrand)
				# Using quad
				#print('self.data', self.data.shape)
				#multipole_ell_emm = np.tensordot(self.data, Ydarea, axes=((0, 1), (0, 1)))
				multipole_ell_emm = np.sum(self.data * Ydarea, (0, 1))

				#print(f'l{ell_value}m{emm_value}', multipole_ell_emm)

				#print('multipole_ell_emm', multipole_ell_emm.shape)
				waveform_modes.set_mode_data(ell_value, emm_value, data=multipole_ell_emm)

		return waveform_modes

	# Construct the time axis




#	def reconstr_waveform():


#	def decompose_waaveform():
#


#	def resample_data():


#	def extrapolate_to_inf_per():


#	def extrapolate_to_inf_numeric():


#	def apply_CoM_correction():


#	def get_strain_from_psi4():


#	def get_news_from_psi4():


#	def get_psi4_from_news():


#	def get_strain_from_news():


#	def get_psi4_from_strain():


#	def get_news_from_strain():


#	def apply_supertranslation():


#	def apply_boost():


# @base_dir.setter
# def base_dir(self, base_dir):
#	 self.__base_dir = base_dir

################################################################
# Modes array class
################################################################

#spec_ma = { 'label' : nb.types.string,
#			'data_dir' : nb.types.string,
#			'key_format' : nb.types.string,
#			'key_ex' : nb.types.string,
#			'ell_max' : nb.int32,
#			'r_ext' :	nb.double,
#			'out_file_name' : nb.types.string,
#			'maxtime' :	nb.types.double,
#			'date' : nb.types.string,
#			'time' : nb.types.string,
#			 'time_axis' : nb.double[:],
#			 'frequency_axis' : nb.double[:],
#			 'modes_data' : nb.complex128[:, :, :],
#			 'base_dir' : nb.types.string,
#			 'file_name' : nb.types.string,
#			 'spin_weight' : nb.int32,
#			'modes_list' : nb.types.List(nb.int32)
#
#}
#@jitclass(spec_ma)
class modes_array:
	"""A class that holds mode array of waveforms

	Attributes
	----------
	label: str
					The label of the modes array.
	r_ext: float
					The extraction radius.
	modes_list: list
							 The list of available modes
							 in the format [l1, [m values], [l2, [m values], ...]]
	ell_max:	int
							The maximum :math:`\\ell`
							mode available.
	modes_data: 3d array
							 The three dimensional array
							 containing modes in time/frequency
							 space. The axis of the array is
							 (:math:`\\ell`, :math:`m`, time/freq).
	base_dir:	str
							The base directory containing the
							modes data.
	data_dir:	str
							The subdirectory in which to look
							for the data.
	filename: str
							The filename containg the modes data.


	Methods
	-------
	get_metadata:
					Get the metadata associated with the modes_array.
	mode:
			Get the data for the given :math:`\\ell, m` mode.
	_create_modes_array:
						A private method to create an empty modes_array of given shape.
	delta_t:
			Set the attribute `delta_t` and/ or return its value.
	load_modes:
				Load the waveform modes from a specified h5 file.
	save_modes:
				Save the waveform modes to a specified h5 file.
	set_mode_data:
					Set the `mode` data of specified modes.
	to_frequency_basis:
						Get the `modes_array` in frequency basis from its time basis representation.
	to_time_basis:
					Get the `modes_array` in temporal basis from its frequency basis representation.
	extrap_to_inf:
					Extrapolate the modes to infinity.
	supertranslate:
					Supertranslate the waveform modes.
	boost:
			Boost the waveform modes.

	"""

	def __init__(
		self,
		data_dir=None,
		file_name=None,
		modes_data=None,
		time_axis=None,
		frequency_axis=None,
		key_format=None,
		ell_max=None,
		modes_list=None,
		label=None,
		r_ext=500,
		out_file_name=None,
		maxtime=None,
		date=None,
		time=None,
		key_ex=None,
		spin_weight=None
	):

		self.label = label
		self.data_dir = data_dir
		self.file_name = file_name
		self.modes_data = modes_data
		self.key_format = key_format
		self.ell_max = ell_max
		self.modes_list = modes_list
		self.r_ext = r_ext
		self.time_axis = time_axis
		self.frequency_axis = frequency_axis
		self.out_file_name = out_file_name
		self.maxtime = maxtime
		self.date = date
		self.time = time
		self.key_ex = key_ex
		self.spin_weight = spin_weight
	def get_metadata(self):
		"""Get the metadata associated with the instance.

		Returns
		-------
		metadata:	dict
								A dictionary of metedata.
		"""
		# The metadata dict
		unnecessary_keys = ["time_axis", "modes_data", "freq_axis"]

		# Get all attributes
		# metadata = self.__dict__
		metadata = {}

		for key, val in self.__dict__.items():
			if key in unnecessary_keys:
				pass
			else:
				metadata.update({key: val})

		# for item in unnecessary_keys:
		#	metadata.pop(item, None)

		# self.metadata = metadata

		return metadata

	def mode(self, ell, emm):
		"""The modes array data structure to hold waveform modes.

		Parameters
		----------
		ell:		int
								The :math:`\\ell` index of the mode.
		emm:		int
								The :math:`m` index of the mode.

		Returns
		-------
		mode_data:		array
										The array of the requested mode.
		"""

		emm_index = ell + emm

		return self.modes_data[ell, emm_index, :]

	def _create_modes_array(self, ell_max, data_len):
		"""Create a modes array and initialize it with zeros.

		Parameters
		----------
		ell_max:	 int
								 The maximum :math:`\\ell` value of the modes.
		data_len:	int
								The number of points along the third (time / frequency) axis.

		Returns
		-------
		self.modes_array:		 modes_array
														 sets the `self.modes_array` attribute.
		"""
		import time
		import datetime

		#self.modes_data = np.zeros([ell_max + 1, 2 * (ell_max + 1) + 1, data_len], dtype=np.complex128)
		self.modes_data = np.zeros((ell_max + 1, 2 * (ell_max + 1) + 1, data_len), dtype=np.complex128)

		# Set the time metadata
		time_now = time.localtime()
		time_now = time.strftime("%H:%M:%S", time_now)

		date_now = str(datetime.date.today())

		if self.time is None:
			# Assign time and date stamp if it doesnt exist
			self.time = time_now
			self.date = date_now

	@property
	def data_len(self):
		"""Returns the length of the time/frequency axis.

		Returns
		-------
		data_len:	int
								THe length of the time/frequency axis.
		"""

		try:
			data_len = len(self.time_axis)

		except:
			data_len = len(self.frequency_axis)

		return data_len

	def delta_t(self, value=None):
		"""Sets and returns the value of time stepping :math:`dt`.

		Parameters
		----------
		value : float, optional
						The value of :math:`dt`
						to set to the attribute.
		Returns
		-------
		self.delta_t:	 float
										 Sets the attribute.
		"""

		# if not self.delta_t:
		if not value:
			try:
				delta_t = self.time_axis[1] - self.time_axis[0]
			except:
				print("Please input the value of `delta_t` or supply the `time_axis` to the waveform.")
		else:
			delta_t = value

		return delta_t

	@property
	def delta_f(self, value=None):
		"""Sets and returns the value of frequency stepping :math:`df`.

		Parameters
		----------
		value : float, optional
						The value of :math:`df`
						to set to the attribute.

		Returns
		-------
		delta_f:	 float
								 Sets the attribute.
		"""

		# if not self.delta_t:
		if not value:
			try:
				delta_f = self.frequency_axis[1] - self.frequency_axis[0]
			except:
				print("Please input the value of `delta_f` or supply the `frequency_axis` to the waveform.")
		else:
			delta_f = value

		return delta_f

	def load_modes(self, r_ext=None, ell_max=None, pre_key=None, modes_list=None, crop=False, centre=True, key_ex=None, r_ext_factor=1):
		"""Load the waveform mode data from an hdf file.

		Parameters
		----------

		pre_key:	str, optional
								A string containing the key of the group in
								the HDF file in which the modes` dataset exists.
								It defaults to `None`.
		mode_numbers:	list
										The mode numbers to load from the file.
										Each item in the list is a list that
										contains two integrer numbers, one for
										the mode index :math:`\\ell` and the
										other for the mode index :math:`m`.
		crop:	bool
						Whether or not to crop the beginning of the input
						waveform. If yes, the first :math:`t=r_{ext}`
						length of data will be discarded.

		Returns
		-------
		waveform_obj:	3d array
										Sets the three dimensional array `waveform.modes` that contains
										the required :math:`\\ell, m` modes.

		Examples
		--------
		>>> from waveformtools.waveforms import waveform
		>>> waveform.set_basedir('./')
		>>> waveform.set_datadir('data/')
		>>> mode_numbers = [[2, 2], [3, 3]]
		>>> waveform.load_data(mode_numbers=mode_numbers)
		"""
		import sys
		import re
		import json

		# get the full path.
		full_path = self.data_dir + self.file_name

		cflag = 0

		# Ext radius
		if r_ext is None:
			r_ext = self.r_ext

		# Open the modes file.
		with h5py.File(full_path, "r") as wfile:

			#################################
			# Get metadata
			###############################

			# Load metadata if present.
			try:
				metadata_bytes = bytes(np.void(wfile["metadata"])).decode()
				metadata = json.loads(metadata_bytes)
				self.__dict__.update(metadata)
				message("Metadata loaded")

			except:
				pass

			# data = np.array(wfile['l0_m0_r500.00'])
			# print(data)
			# Get the list of keys.
			modes_keys_list = list(wfile.keys())


			if key_ex is None:
				# Check attribute.
				key_ex = self.key_ex

			if key_ex is not None:
				# Filter the keys according to key_ex if specified.
				print(key_ex)
				self.key_ex=key_ex
				modes_keys_list=[item for item in modes_keys_list if key_ex in item]
				#print(modes_keys_list)

			else:
				print('key_ex is not given')
			modes_keys_list = sorted(modes_keys_list)

			#print('Modes keys', modes_keys_list)
			# self.mode_keys_list = modes_keys_list
			# Construct the list of modes if it doesnt exist.

			##########################
			# Construct modes list
			##########################

			if not modes_list:
				# Check if modes list is given for which mode to load.

				if not self.modes_list:
					# If the list of modes is not given and the attribute is
					# also not set, then construct the list of modes.

					if not ell_max:
						# If ell max is also not specified,
						# Get it from attr or construct.

						if not self.ell_max:
							# construct the list of modes using
							# the list of modes h5 file keys.
							modes_list = _get_modes_list_from_keys(modes_keys_list, r_ext)
							#print(modes_list)
							# Get the ell max
							ell_max = max([item[0] for item in modes_list])
							self.ell_max = ell_max

						else:
							# Get ell_max from attr.
							ell_max = self.ell_max

					else:
						self.ell_max = ell_max
						# If ell max is given, construct the
						# list of modes directly.
						modes_list = construct_mode_list(ell_max)

					# set the modes list attr.
					self.modes_list = modes_list

				else:
					# Assign to modes_list the attr.
					modes_list = self.modes_list
					# if modes list attr is present do nothing.

			else:
				self.modes_list = modes_list
				# If modes list is given, get ell_max from it.
				if not ell_max:
					# Get the ell max
					ell_max = max([item[0] for item in modes_list])

			# Set the ell_max attribute if not already.
			if not self.ell_max:
				self.ell_max = ell_max

			#################################################
			# Load modes
			#################################################

			# Load the modes listed in mode_numbers list
			for item in self.modes_list:
				# For every ell mode list in modes_list

				ell_value, emm_list = item

				for emm_index, emm_value in enumerate(emm_list):
					# For every (ell, emm) mode.

					# Find the key corresponding to the mode
					try:
						key = str(
							[
								item
								for item in modes_keys_list
								if re.search(f"l{ell_value}_m{emm_value}_r{r_ext}", item)
							][0]
						)
						# print('The loaded key is ', key, type(key))
						# print('The loaded key is ', key, type(key))
						# if key=='l0_m0_r500.00':
						# print('Its alright')
					except:
						message(f"Waveform dataset for l{ell_value}, m{emm_value} not found")
						sys.exit(0)

					# Get the data
					data = np.array(wfile[key])

					# set the time and data axis
					time_axis = data[:, 0]
					data_re = data[:, 1]
					data_im = data[:, 2]

					if not cflag:
						if not self.modes_data:

							if crop:
								# Crop the beginning portion.
								delta_t = time_axis[1] - time_axis[0]
								shift = int(self.r_ext / delta_t)

							else:
								shift = 0
							data_len = len(time_axis) - shift
							# self.data_len = data_len
							# Delete the attribute
							del self.modes_data
							# Create an array for the waveform mode object
							self._create_modes_array(self.ell_max, data_len)
							# self.modes_data = np.zeros([ell_max+1, 2*(ell_max+1) +1, data_len], dtype=np.complex128)
							#self.modes_data = np.zeros([ell_max+1, 2*(ell_max+1) +1, data_len], dtype=np.complex128)

							cflag = 1

							# set the time axis.
							# self.time_axis = time_axis[shift:]

					self.modes_data[ell_value, emm_index] = r_ext_factor*(data_re[shift:] + 1j * data_im[shift:])

			##############################
			# Recenter axis
			##############################
			maxloc = np.argmax(np.absolute(self.mode(2, 2)))
			maxtime = time_axis[shift + maxloc]
			if self.maxtime is None:
				self.maxtime = maxtime
			print("Max time is", maxtime)

			if centre:
				self.time_axis = time_axis[shift:] - maxtime

	def save_modes(
		self,
		ell_max=None,
		pre_key=None,
		key_format=None,
		modes_to_save=None,
		out_file_name="mp_new_modes.h5",
		r_ext_factor = None,
		compression_opts=0,
		r_ext=None
	):
		"""Save the waveform mode data to an hdf file.

		Parameters
		----------
		pre_key:	str, optional
								A string containing the key of the group in
								the HDF file in which the modes` dataset exists.
								It defaults to `None`.
		mode_numbers:	list
										The mode numbers to load from the file.
										Each item in the list is a list that
										contains two integrer numbers, one for
										the mode index :math:`\\ell` and the
										other for the mode index :math:`m`.

		Returns
		-------
		waveform_obj:	3d array
										Sets the three dimensional array `waveform.modes` that contains
										the required :math:`\\ell, m` modes.

		Examples
		--------
		>>> from waveformtools.waveforms import waveform
		>>> waveform.set_basedir('./')
		>>> waveform.set_datadir('data/')
		>>> mode_numbers = [[2, 2], [3, 3]]
		>>> waveform.load_data(mode_numbers=mode_numbers)
		"""

		import json

		#############################
		# I/O assignments.
		#############################

		self.out_file_name = self.label + '_' + out_file_name
		self.out_file_name.replace(' ', '_')

		# get the full path.
		full_path = self.data_dir + self.out_file_name

		if r_ext is None:
			if self.r_ext is None:
				r_ext = 500
			else:
				r_ext = self.r_ext

		if r_ext_factor is None:
			r_ext_factor = self.r_ext

		###################################
		# Identify the modes to save.
		###################################

		if not modes_to_save:

			if ell_max is not None:
				modes_to_save = self.modes_list[:ell_max]

			else:
				modes_to_save = self.modes_list

		##########################
		# Create the modes file.
		##########################

		with h5py.File(full_path, "w") as wfile:

			# Create the metadata dataset.
			metadata = self.get_metadata()

			metadata_bytes = json.dumps(metadata).encode()

			# dt = h5py.special_dtype(vlen=str)
			# metadata=np.asarray([metadata_bytes], dtype=dt)
			wfile.create_dataset("metadata", data=metadata_bytes, compression_opts=compression_opts)

			# Load the modes listed in mode_numbers list
			for item in modes_to_save:
				# For every ell mode list in modes_list

				ell_value, emm_list = item

				for emm_value in emm_list:
					# For every (ell, emm) mode.

					data = self.mode(ell_value, emm_value)
					# set the time and data axis
					data_re = data.real
					data_im = data.imag

					save_data = np.transpose(np.array([self.time_axis, data_re, data_im]))
					# Make the key
					key = _key_gen(ell_value, emm_value, extras=f"r{r_ext:.2f}")
					# print('Processing key', key)
					# Create data set
					wfile.create_dataset(key, data=save_data)

	def set_mode_data(self, ell_value, emm_value, data):
		"""Set the mode array data for the respective :math:`(\\ell, m)` mode.

		Parameters
		----------
		ell_value:		int
										The :math:`\\ell` polar mode number.
		emm_value:		int
										The :math:`emm` azimuthal mode number.
		data:		1d array
								The array consisting of mode data for the requested mode.

		Returns
		-------
		self.mode_data:	modes_data
												The updated modes data.
		"""
		# Compute the emm index given ell.
		emm_index = emm_value + ell_value

		# Set the mode data.
		self.modes_data[ell_value, emm_index] = data


	def to_spherical_array(self, grid_info, spin_weight=None):
		''' Obtain the spherical array from the modes array.

		Parameters
		----------

		grid_info:	cls instance
					An instance of the "spherical_grid" class
					to hold the grid info.

		Returns
		-------

		waveform_sp:	spherical_array
						A member of the "spherical_array" class
						constructed from the given "modes_rray".

		'''

		#from qlmtools import Yslm_new


		# Create a spherical array.
		waveform_sp = spherical_array(label=self.label, grid_info=grid_info)

		if spin_weight is None:
			if self.spin_weight is not None:
				spin_weight = self.spin_weight
			else:
				spin_weight = -2
				self.spin_weight = spin_weight

		spin_weight = abs(spin_weight)
		waveform_sp.spin_weight = spin_weight
		# Set the time-axis
		try:
			waveform_sp.time_axis = self.time_axis
		except:
			waveform_sp.frequency_axis = self.frequency_axis

		# Get the coordinate meshgrid
		theta, phi = grid_info.meshgrid

		s1, s2 = theta.shape
		s3 = self.data_len
		sp_data = np.zeros((s1, s2, s3), dtype=np.complex128)

		modes_list = [item for item in self.modes_list if item[0]>=spin_weight]
		for item in modes_list:

			# Get modes.
			ell, emm_list = item
			#if ell<spin_weight:
			#	continue

			for emm in emm_list:
				# For every l, m
				sp_data += np.multiply.outer(Yslm_new(spin_weight, ell=ell, emm=emm, theta_grid=theta, phi_grid=phi), self.mode(ell, emm))
				#print(sp_data)
		# Set the data of the spherical array.
		waveform_sp.data = sp_data
		try:
			waveform_sp.time_axis = self.time_axis
		except:
			waveform_sp.frequency_axis = self.frequency_axis
		return waveform_sp


	def trim(self, trim_upto_time=None):
		""" Trim the modes_array at the beginning.

		Parameters
		----------
		time:	float
				The time unit upto which to discard.

		Returns
		-------
		Re-sets the `time_axis` and `modes_array` data.

		"""
		if trim_upto_time is None:
			trim_upto_time = self.r_ext

		# Compute the start index
		start = int(trim_upto_time/self.delta_t())

		# Trim the time axis
		self.time_axis = self.time_axis[start:]

		# Trim the data
		self.modes_data = self.modes_data[:, :, start:]

		# Recenter the time axis
		max_ind = np.argmax(np.absolute(self.mode(2, 2)))
		self.time_axis = self.time_axis - self.time_axis[max_ind]

	def to_frequency_basis(self):
		"""Compute the modes in frequency basis.

		Returns
		-------
		waveform_tilde_modes:	modes_array
														A modes_array containing the modes
														in frequency basis.
		"""

		# Create a new modes array
		waveform_tilde_modes = modes_array(label=f"{self.label} -> frequency_domain")
		waveform_tilde_modes._create_modes_array(ell_max=self.ell_max, data_len=self.data_len)

		from waveformtools.transforms import compute_fft

		for mode in self.modes_list:
			# Extrapolate every mode

			# Ge the ell value
			ell_value, emm_list = mode

			for emm_value in emm_list:

				freq_axis, freq_data = compute_fft(self.mode(ell_value, emm_value), self.delta_t())

				waveform_tilde_modes.set_mode_data(ell_value, emm_value, freq_data)

		waveform_tilde_modes.frequency_axis = freq_axis
		waveform_tilde_modes.ell_max = self.ell_max
		waveform_tilde_modes.modes_list = self.modes_list
		return waveform_tilde_modes

	def to_time_basis(self):
		"""Compute the modes in time basis.

		Returns
		-------
		waveform_modes :  modes_array
										  A modes_array containing the modes
										  in frequency basis.
		"""

		# Create a new modes array
		waveform_modes = modes_array(label=f"{self.label} -> time_domain")
		waveform_modes._create_modes_array(ell_max=self.ell_max, data_len=self.data_len)

		from waveformtools.transforms import compute_ifft

		for mode in self.modes_list:
			# Extrapolate every mode

			# Ge the ell value
			ell_value, emm_list = mode

			for emm_value in emm_list:

				time_axis, time_data = compute_ifft(self.mode(ell_value, emm_value), self.delta_f)

				waveform_modes.set_mode_data(ell_value, emm_value, time_data)

		try:
			maxloc = np.argmax(np.absolute(waveform_modes.mode(2, 2)))
		except:
			maxloc = 0

		maxtime = time_axis[maxloc]

		waveform_modes.time_axis = time_axis - maxtime

		return waveform_modes

	def extrap_to_inf(self, mass=1, spin=None, modes_list="all", method="SIO", r_ext_factor=1):
		"""Extrapolate the :math:`\\Psi_4` modes to infinity
		using the perturbative improved second order method.

		Parameters
		----------
		mass:		float
								The effective total mass of the system.
		spin:		float
								The effective spin of the system.
		modes:		modes array, optional
					The modes to extrapolate. Defaults
					to `all` if not specified.
		method:	str
					The method to use for extrapolation. The available methods are:

		+------------+--------------------------------------+
		| Method str | Name									|
		+------------+--------------------------------------+
		|'FO'		 | First order							|
		|'SO'		 | Second order							|
		|'SIO'		 | Second improved order				|
		|'NM'		 | Numerical method (not ready yet)		|
		+------------+--------------------------------------+

		Returns
		-------
		waveform_inf_modes:	modes array
														A new modes array that contains
														the extrapolated modes.
		"""

		from functools import partial

		# Prepare the extrapolating method.
		if method == "SIO":

			from waveformtools.extrapolate import waveextract_to_inf_perturbative_twop5_order

			extrap_method = partial(
				waveextract_to_inf_perturbative_twop5_order,
				delta_t=self.delta_t(),
				areal_radius=self.r_ext,
				mass=mass,
				spin=spin,
			)

		if method == "SO":

			from waveformtools.extrapolate import waveextract_to_inf_perturbative_two_order

			extrap_method = partial(
				waveextract_to_inf_perturbative_two_order,
				delta_t=self.delta_t(),
				areal_radius=self.r_ext,
				mass=mass,
				spin=spin,
			)

		if method == "FO":

			from waveformtools.extrapolate import waveextract_to_inf_perturbative_one_order

			extrap_method = partial(
				waveextract_to_inf_perturbative_one_order, u_ret=self.time_axis, areal_radius=self.r_ext, mass=mass
			)

		if method == "NM":
			print("This method is not available yet! ")

		# Prepare the modes to be extrapolated.
		if modes_list == "all":
			modes_list = construct_mode_list(self.ell_max)

		# Create a mode array for the extrapolated waveform.
		extrap_wf = modes_array(label=f"{self.label} -> rPsi4_inf",  modes_list=self.modes_list, ell_max=self.ell_max, r_ext = self.r_ext)

		extrap_wf._create_modes_array(ell_max=self.ell_max, data_len=self.data_len)

		# Retain the time axis.
		extrap_wf.time_axis = self.time_axis
		for mode in modes_list:
			# Extrapolate every mode

			# Ge the ell value
			ell_value, emm_list = mode

			for emm_value in emm_list:
				# For every emm value
				print(f"Processing l{ell_value}, m{emm_value}")
				# Compute rPsi4_lm
				mode_data = r_ext_factor * self.mode(ell_value, emm_value)

				# Extrapolate
				# import ipdb; ipdb.set_trace()
				extrap_mode_data = extrap_method(rPsi4_rlm=mode_data)

				# Assign data to new modes array
				extrap_wf.set_mode_data(ell_value, emm_value, extrap_mode_data)

		print("Done!")
		return extrap_wf

	def supertranslate(self, supertransl_alpha_modes, grid_info, order=4):

		"""Supertranslate the :math:`\\Psi_{4\\ell m}` waveform modes, give the,
		the supertranslation parameter and the order.

		Parameters
		----------
		supertransl_alpha_modes :  modes_array
														   The modes_array containing the
														   supertranslation mode coefficients.
		grid_info:		class instance
										The class instance that contains
										the properties of the spherical grid
										using which the computations are
										carried out.
		order:		int
								The number of terms to use to
								approximate the supertranslation.

		Returns
		-------
		Psi4_supertransl:	  modes_array
												  A class instance containg the
												  modes of the supertranslated
												  waveform :math:`\\Psi_4`.
		"""

		import BMS

		ell_max = self.ell_max
		# Step 0: Get the grid properties for integrations

		# Compute the meshgrid for theta and phi.
		theta, phi = grid_info.meshgrid
		# theta
		# Step 1: get the grid function for supertranslation parameter
		supertransl_alpha_sphere = BMS.compute_supertransl_alpha(supertransl_alpha_modes, theta, phi)

		# The supertranslation is carried out in frequency space.
		# Step 2: get the FFT of the mode coefficients of Psi4
		Psi4_tilde_modes = self.to_frequency_basis()

		# Get the 2d angular frequency array
		omega_axis_2d = Psi4_tilde_modes.omega

		# Construct the supertranslation factor
		supertransl_factor = np.sum(
			np.array([np.power((-1j * omega_axis_2d * supertransl_alpha_sphere), index) for index in range(order)]),
			axis=0,
		)

		# Multiply with the fourier modes.
		supertransl_spherical_factor = Psi4_tilde_modes.multiply(supertransl_factor)

		from qlmtools import Yslm_new

		# Reconstruct the modes
		for ell_value in range(ell_max+1):
			for emm_value in range(-ell_value, ell_value + 1):
				# Multiply with the SWSH basis.
				supertransl_spherical_grid += supertransl_spherical_factor * Yslm_new(
					spin_weight=-2, ell=ell_value, emm=emm_value, theta=theta, phi=phi
				)

				# Step 3: Reconstruct the function on the sphere

		# Create a spherical_array to hold the supertranslated waveform
		supertransl_spherical_waveform = spherical_array(grid_info=grid_info)

		# Set the data.
		supertransl_spherical_waveform.data = supertransl_spherical_grid

		# Get modes_array from spherical_array
		Psi4_supertransl_modes = supertransl_spherical_waveform.to_modes_array(ell_max=ell_max)

		return Psi4_supertransl_modes

	def boost(self, conformal_factor, grid_info=None):
		"""Boost the waveform given the unboosted waveform and the boost conformal factor.

		Parameters
		----------
		conformal_factor:		2d array
								The conformal factor for the Lorentz transformation.
								It may be a single floating point number or an array
								on a spherical grid. The array will be of dimensions
								[ntheta, nphi]

		Returns
		-------
		boosted_waveform:	  spherical_array
							  The class instance `spherical_array`
							  that contains the boosted waveform.
		"""

		from waveformtools.grids import spherical_grid

		# Construct a spherical grid.
		if grid_info is None:
			grid_info = spherical_grid()

		# Get spherical array from modes.
		unboosted_waveform = self.to_spherical_array(grid_info)

		boosted_waveform_data = unboosted_waveform.boost(conformal_factor)

		# Construct a 2d waveform array object
		boosted_waveform = spherical_array(grid_info=unboosted_waveform.grid_info, data=np.array(boosted_waveform_data))
		boosted_waveform.label = "boosted"

		# Get modes from spherical data.
		#boosted_waveform_modes = boosted_waveform.to_modes_array()

		#return boosted_waveform_modes
		return boosted_waveform

	def get_strain_from_psi4(self, omega0='auto'):
		''' Get the strain `modes_array` from :math:`\\Psi_4` by
		fixed frequency integration.

		Parameters
		----------
		omega0:	float, optional
				The lower cutoff angular frequency for FFI.
				By default, it computes this from the mode
				data.

		Returns
		-------

		strain_waveform:	modes_array
							The computed strain modes.

		'''

		# Initialize a mode array for strain.
		#strain_waveform = modes_array(label=f'{self.label} strain from Psi4', r_ext=500, ell_max=8, modes_list=self.modes_list)
		strain_waveform = modes_array(label='{} strain from Psi4'.format(self.label), r_ext=self.r_ext, ell_max=8, modes_list=self.modes_list)

		strain_waveform.time_axis = self.time_axis
		strain_waveform.ell_max = self.ell_max

		data_len = self.data_len

		strain_waveform._create_modes_array(ell_max=self.ell_max, data_len=data_len)

		# Integrate,
		from waveformtools.integrate import fixed_frequency_integrator
		from waveformtools.waveformtools import get_starting_angular_frequency as sang_f

		omega_st = omega0
		for item in self.modes_list[:]:
			ell, emm_list = item
			for emm in emm_list:
				mode_data = self.mode(ell, emm)
				if omega0=='auto':
					omega_st = abs(sang_f(mode_data, delta_t=self.delta_t()))/10
				strain_mode_data, _ = fixed_frequency_integrator(udata_time=mode_data, delta_t=self.delta_t(), omega0=omega_st, order=2)
				strain_waveform.set_mode_data(ell, emm, data=strain_mode_data)

		return strain_waveform



	def get_news_from_psi4(self, omega0='auto'):
		''' Get the News `modes_array` from :math:`\\Psi_4` by
		fixed frequency integration.

		Parameters
		----------
		omega0:	float, optional
				The lower cutoff angular frequency for FFI.
				By default, it computes this as one tenth of
				the starting frequency of the mode data.

		Returns
		-------
		news_waveform:	modes_array
						The computed strain modes.

		'''

		# Initialize a mode array for strain.
		#news_waveform = modes_array(label=f'{self.label} news from Psi4', r_ext=500, ell_max=8, modes_list=self.modes_list)
		news_waveform = modes_array(label='{} news from Psi4'.format(self.label), r_ext=500, ell_max=8, modes_list=self.modes_list)

		news_waveform.time_axis = self.time_axis
		news_waveform.ell_max = self.ell_max

		data_len = self.data_len

		news_waveform._create_modes_array(ell_max=self.ell_max, data_len=data_len)

		# Integrate,
		from waveformtools.integrate import fixed_frequency_integrator
		from waveformtools.waveformtools import get_starting_angular_frequency as sang_f

		omega_st = omega0
		for item in self.modes_list[:]:
			ell, emm_list = item
			for emm in emm_list:
				mode_data = self.mode(ell, emm)
				if omega0=='auto':
					omega_st = abs(sang_f(mode_data, delta_t=self.delta_t()))/10
				news_mode_data, _ = fixed_frequency_integrator(udata_time=mode_data, delta_t=self.delta_t(), omega0=omega_st, order=1)
				news_waveform.set_mode_data(ell, emm, data=news_mode_data)

		return news_waveform

	def taper(self, zeros='auto'):
		''' Taper a waveform at both ends and insert zeros if needed

		Parameters
		----------

		zeros:	int
				The number of zeros to add at rach end

		Returns
		-------

		tapered_modes:	modes_array
						Modes array with tapered mode data.
		'''


		from waveformtools.waveformtools import taper

		if zeros=='auto':
			# Decide the number of zeros
			data_len = self.data_len

			nearest_power = int(np.log(data_len)/np.log(2))
			req_len = np.power(2, nearest_power+1)
			zeros = req_len - data_len
			print('num_zeros', zeros)

		# New modes array.

		tapered_modes= None

		for item in self.modes_list[:]:
			ell, emm_list = item
			for emm in emm_list:
				input_data_re = self.mode(ell, emm).real
				input_data_im = self.mode(ell, emm).imag

				tapered_data_re = taper(input_data_re, zeros=zeros)
				tapered_data_im = taper(input_data_im, zeros=zeros)

				#tapered_data_re = taper_tanh(input_data_re, delta_t=self.delta_t())
				#tapered_data_im = taper_tanh(input_data_im, delta_t=self.delta_t())

				new_data_len  = len(tapered_data_re)

				if tapered_modes is None:
					#tapered_modes = modes_array(label = f'tapered {self.label}', r_ext=self.r_ext, modes_list=self.modes_list, ell_max=self.ell_max)
					tapered_modes = modes_array(label = 'tapered {}'.format(self.label), r_ext=self.r_ext, modes_list=self.modes_list, ell_max=self.ell_max)

					tapered_modes._create_modes_array(ell_max=self.ell_max, data_len=new_data_len)
				tapered_data = tapered_data_re + 1j*tapered_data_im

				#print(len(tapered_data_re))
				tapered_modes.set_mode_data(ell, emm, data=tapered_data)

		# Set the time axis
		new_time_axis = np.arange(0, new_data_len*self.delta_t(), self.delta_t())

		tapered_modes.time_axis = new_time_axis

		# Recenter the modes.
		tapered_modes.trim(trim_upto_time=0)

		return tapered_modes

	def taper_tanh(self, time_axis=None, zeros='auto', duration=10, sides='both'):
		''' Taper a waveform at both ends and insert zeros if needed

		Parameters
		----------

		zeros:	int
				The number of zeros to add at rach end

		Returns
		-------

		tapered_modes:	modes_array
						Modes array with tapered mode data.
		'''


		from waveformtools.waveformtools import taper_tanh

		if zeros=='auto':
			# Decide the number of zeros
			data_len = self.data_len

			nearest_power = int(np.log(data_len)/np.log(2))
			req_len = np.power(2, nearest_power+1)
			zeros = req_len - data_len
			#print('num_zeros', zeros)

		# New modes array.

		tapered_modes= None

		for item in self.modes_list[:]:
			ell, emm_list = item
			for emm in emm_list:
				input_data_re = self.mode(ell, emm).real
				input_data_im = self.mode(ell, emm).imag

				#tapered_data_re = taper(input_data_re, zeros=zeros)
				#tapered_data_im = taper(input_data_im, zeros=zeros)

				_, tapered_data_re = taper_tanh(input_data_re, delta_t=self.delta_t(), duration=duration, sides=sides)
				_, tapered_data_im = taper_tanh(input_data_im, delta_t=self.delta_t(), duration=duration, sides=sides)

				new_data_len  = len(tapered_data_re)

				if tapered_modes is None:
					#tapered_modes = modes_array(label = f'tapered {self.label}', r_ext=self.r_ext, modes_list=self.modes_list, ell_max=self.ell_max)
					tapered_modes = modes_array(label = 'tapered {}'.format(self.label), r_ext=self.r_ext, modes_list=self.modes_list, ell_max=self.ell_max)

					tapered_modes._create_modes_array(ell_max=self.ell_max, data_len=new_data_len)
				tapered_data = tapered_data_re + 1j*tapered_data_im

				#print(len(tapered_data_re))
				tapered_modes.set_mode_data(ell, emm, data=tapered_data)

		# Set the time axis
		new_time_axis = np.arange(0, new_data_len*self.delta_t(), self.delta_t())

		tapered_modes.time_axis = new_time_axis

		# Recenter the modes.
		tapered_modes.trim(trim_upto_time=0)

		return tapered_modes


	def low_cut(self, omega0=0.03, order=2):
		''' Apply the low cut filter from waveformtools.low_cut_filter

		Parameters
		----------
		order:	   int, optional
					The order of the butterworth filter.
		omega0:		float, optional
					The cutoff frequency of the butterworth filter.

		Returns:
		--------
		filtered_modes:	modes_array
						A modes array object containing filtered modes.

		'''

		# modes_array for filtered data.
		filtered_modes=None

		# Import the filter
		from waveformtools.waveformtools import low_cut_filter

		for item in self.modes_list:
			# Iterate over available modes.
			ell, emm_list = item
			for emm in emm_list:

				if filtered_modes is None:
					# Create filtered_modes
					#filtered_modes = modes_array(label = f'lc filtered {self.label}', r_ext=self.r_ext, modes_list=self.modes_list, ell_max=self.ell_max)
					filtered_modes = modes_array(label = 'lc filtered {}'.format(self.label), r_ext=self.r_ext, modes_list=self.modes_list, ell_max=self.ell_max)

					filtered_modes._create_modes_array(ell_max=self.ell_max, data_len=self.data_len)

				# Get filtered mode data.
				filtered_data = low_cut_filter(self.mode(ell, emm), self.frequency_axis, omega0=omega0, order=order)

				# Set the mode data.
				filtered_modes.set_mode_data(ell, emm, data=filtered_data)

		# Set the f axis.
		filtered_modes.frequency_axis = self.frequency_axis

		return filtered_modes

#######################################################################################################
def _get_modes_list_from_keys(keys_list, r_ext):
	"""Get the modes list from the keys list
	of an hdf file.

	Parameters
	----------
	keys_list:		list
					The list containing all the keys
	r_ext:		float
				The extraction radius of the data.

	Returns
	-------
	modes_list:	list
					The list of modes.
	"""

	# Sort the keys to ensure a nice
	# modes list structure.
	keys_list_orig = sorted(keys_list)

	if r_ext!=-1:
		keys_list = [item for item in keys_list_orig if f"r{r_ext}" in item]


		if keys_list==[]:
			print('Got an empty list. Searching for r_ext value in key string')
			keys_list = [item for item in keys_list_orig if f"{r_ext}" in item]

	#print('List of keys received', keys_list)
	# The list of modes.
	modes_list = []

	# Initialize the emm modes sublist.
	emm_modes_for_ell = []

	# Present ell value to
	# initialize the mode concatenation.
	ell_old = 0

	for key in keys_list:
		# print(key)
		# Get the ell value
		ell_value, emm_value = _get_ell_emm_from_key(key)

		if ell_value != ell_old:
			# If the ell value has changed,
			# update the modes list before moving
			# onto the next ell value.
			modes_list.append([ell_old, emm_modes_for_ell])
			# The present ell value is the old
			# ell value.
			ell_old = ell_value

			# Reset the ell_mode list.
			emm_modes_for_ell = []

		# Update it with the new emm mode.
		emm_modes_for_ell.append(emm_value)

	modes_list.append([ell_value, emm_modes_for_ell])

	return modes_list

def _get_ell_emm_from_key(key):
	"""Get the :math:`\\ell` and :math:`m` values
	from a given key string of an hdf file.

	Parameters
	----------
	key:	str
			The input key string

	Returns
	-------
	ell_value:		int
					The :math:`\\ell` value
	emm_value:		int
					The :math:`m` value.

	Notes
	-----

	Assumes that the input string has :math:`\\ell` and :math:`m` values
	in the form `l{int}m{int}`.

	"""

	import re

	str_match = re.search("l\d*", key)
	ell_str_start = str_match.start()
	ell_str_end = str_match.end()
	ell_value = int(key[ell_str_start + 1 : ell_str_end])

	str_match = re.search("m-*\d*", key)
	emm_str_start = str_match.start()
	emm_str_end = str_match.end()
	emm_value = int(key[emm_str_start + 1 : emm_str_end])

	return ell_value, emm_value


def get_iteration_numbers_from_keys(keys_list):
	''' Get the iteration number from keys.

	Parameters
	----------
	keys_list: list
			   The list of keys.

	Returns
	-------
	iteration_numbers: list
						The list containing the iteration
						numbers.
	'''
	import re

	iteration_numbers = []

	for key in keys_list:
		str_match = re.search(" it=\d* ", key)
		it_str_start = str_match.start()
		it_str_end = str_match.end()
		it_value = int(key[it_str_start + 4 : it_str_end])
		iteration_numbers.append(it_value)

	return iteration_numbers


def construct_mode_list(ell_max, spin_weight=-2):
	"""
	Construct a modes list in the form [[ell1, [emm1, emm2, ...], [ell2, [emm..]],..]
	given the :math:`\\ell_{max}.`

	Parameters
	----------
	spin_weight	: int
				The spin weight of the modes.
	ell_max : int
			  The :math:`\\ell_{max}` of the modes list.

	Returns
	-------

	modes_list : list
				 A list containg the mode indices lists.

	Notes
	-----
	The modes list is the form which the `waveform` object understands.
	"""

	# The modes list.
	modes_list = []

	for ell_index in range(0, ell_max+1):
		# Append all emm modes for each ell mode.
		modes_list.append([ell_index, list(range(-ell_index, ell_index + 1))])

	return modes_list


def _key_gen(ell, emm, extras=None):
	"""Generates strings to be used as keys for
	managing h5 datasets.

	Parameters
	----------
	ell:	int
					The polar angular mode number
					:math:`\\ell`.
	emm : int
			  The aximuthal angular mode number
			  :math:`m`.
	extras:	str
				Any extra string to be appended
				to the end of the key.

	Returns
	-------
	key:	 str
					 A string key.
	"""

	key = f"l{ell}_m{emm}"

	if extras is not None:
		key += f"_{extras}"
		# print('adding rext')

	return key

def sort_keys(modes_keys_list):
    ''' Sort the keys in a list based on
        its iteration number

    Parameters
    ----------
    modes_keys_list: str
                     The list of keys.

    Returns
    -------
    sorted_modes_keys_list: str
                            The sorted list.
    '''

    iteration_numbers = get_iteration_numbers_from_keys(modes_keys_list)

    sargs = np.argsort(iteration_numbers)

    sorted_modes_keys_list = np.array(modes_keys_list)[sargs]

    return sorted_modes_keys_list

