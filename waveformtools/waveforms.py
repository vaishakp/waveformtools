##########################################################################
# A Class for handling the waveform data
##########################################################################

import numpy as np
import h5py
from waveformtools.waveformtools import message


class spherical_array:
	""" A class for handling waveforms on a sphere."""

	def __init__(
		self, label=None, time_axis=None, data=None, base_dir=None, file_name=None,
	):

		self.label = label
		self.base_dir = base_dir  # The base directory containing the
		self.file_name = file_name
		self.time_axis = time_axis
		self.delta_t = delta_t

	@property
	def delta_t(self, value=None):
		""" Sets and returns the value of time stepping :math:`dt`.

		Parameters
		----------

		value : float, optional
				The value of :math:`dt`
				to set to the attribute.

		Returns
		-------

		delta_t :	float
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
		""" Returns the length of the time/frequency axis.

		Returns
		-------

		data_len :	int
					THe length of the time/frequency axis.

		"""

		try:
			data_len = len(self.time_axis)

		except:
			data_len = len(self.frequency_axis)

		return data_len

	def to_modes_array(self, gridinfo, spin_weight=-2, ell_max=8):
		""" Decompose a given spherical array function on a sphere
			into Spin Weighted Spherical Harmonic modes.

		Parameters
		----------

		spin_weight :	 int, optional
						 The spin weight of the waveform. It defaults to -2 for a gravitational waveform.

		ell_max :	int, optional
					The maximum value of the :math:`\\ell' polar quantum number. Defaults to 6=8.

		gridinfo :		class instance
						The class instance that contains the properties of the spherical grid.


		Returns
		-------

		waveforms_modes :	modes_array
							An instance of the `modes_array` class.
							Containing the decomposed modes.


		Notes
		-----

		1. Assumes that the sphere on which this decomposition is carried out is so far out
		   that the coordinate system is spherical polar
		2. Assumes that the poper area is the same as its co-ordinate area.
		3. Ensure that the label of the input spherical array indicates whether
		   it is a time domain data or frequency domain data.

		"""

		# Compute the meshgrid for theta and phi.
		theta, phi = gridinfo.meshgrid

		# Create a modes array object

		# Create a modes list.
		modes_list = construct_mode_list(ell_max)

		if not self.label:
			label = "decomposed_time_domain"
		else:
			label = self.label

		# Create a mode array for the decomposed_waveform
		waveform_modes = modes_array(label=label)

		# Inherit the time or frequency axis.
		if "time" in label:
			axis = self.time_axis
			waveform_modes.time_axis = self.time_axis
		else:
			axis = self.freq_axis
			waveform_modes.freq_axis = self.freq_axis

		# Create the modes_array
		waveform_modes._create_modes_array(ell_max=ell_max, data_len=self.data_len)

		# m modes for decomposition.
		if m == "all":
			m = np.arange(-l, l + 1)

		# The area element on the sphere
		# Compute the meshgrid for theta and phi.
		theta, phi = gridinfo.meshgrid

		sqrt_met_det = np.sqrt(np.power(np.sin(theta), 2))

		darea = sqrt_metdet * info.dtheta * info.dphi

		for index, value in enumerate(axis):
			# Integrate on the sphere for
			# for each time/frequency

			# Convert input to arrays.
			integrand_data = self.data[:, :, index]

			for mode in modes_list:
				ell_value, all_emm_values = mode

				for emm_value in all_emm_values:
					# m value.

					# Spin weighted spherical harmonic function at (theta, phi)
					Ybasis_fun = Yslm(spin_weight, ell=ell_value, emm=emm_value, theta=theta, phi=phi)

					# Using quad
					multipole_ell_emm = integrand * Ybasis_fun * darea

					waveform_modes.set_mode_data(ell_value, emm_value, multipole_ell_emm)

			return waveform_modes


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


class modes_array:
	""" A class that holds mode array of waveforms

	Attributes
	----------

	label : str
			The label of the modes array.

	r_ext : float
			The extraction radius.

	modes_list : list
				 The list of available modes
				 in the format [l1, [m values], [l2, [m values], ...]]

	ell_max :	int
				The maximum :math:`\\ell`
				mode available.

	modes_data : 3d array
				 The three dimensional array
				 containing modes in time/frequency
				 space. The axis of the array is
				 (:math:`\\ell`, :math:`m`, time/freq).

	base_dir :	str
				The base directory containing the
				modes data.

	data_dir :	str
				The subdirectory in which to look
				for the data.

	filename : str
				The filename containg the modes data.



	"""

	def __init__(
		self,
		data_dir=None,
		file_name=None,
		modes_data=None,
		time_axis=None,
		freq_axis=None,
		file_path=None,
		key_format=None,
		ell_max=None,
		modes_list=None,
		label=None,
		r_ext=500,
	):

		self.label = label
		self.data_dir = data_dir
		self.file_name = file_name
		self.modes_data = modes_data
		self.key_format = key_format
		self.ell_max = ell_max
		self.modes_list = modes_list
		self.r_ext = r_ext

	def get_metadata(self):
		''' Get the metadata associated with the instance.

		Returns
		-------

		metadata :	dict
					A dictionary of metedata.

		'''
		# The metadata dict
		unnecessary_keys = ['time_axis', 'modes_data', 'freq_axis']

		# Get all attributes
		#metadata = self.__dict__
		metadata = {}

		for key, val in self.__dict__.items():
			if key in unnecessary_keys:
				pass
			else:
				metadata.update({key : val})

		#for item in unnecessary_keys:
		#	metadata.pop(item, None)

		#self.metadata = metadata

		return metadata

	def mode(self, ell, emm):
		""" The modes array data structure to hold waveform modes.

		Parameters
		----------

		ell :		int
					The :math:`\\ell` index of the mode.

		emm :		int
					The :math:`m` index of the mode.

		Returns
		-------

		mode_data :		array
						The array of the requested mode.


		"""

		emm_index = ell + emm

		return self.modes_data[ell, emm_index, :]

	def _create_modes_array(self, ell_max, data_len):
		""" Create a modes array and initialize it with zeros.

		Parameters
		----------

		ell_max :	 int
					 The maximum :math:`\\ell` value of the modes.

		data_len :	int
					The number of points along the third (time / frequency) axis.

		Returns
		-------

		self.modes_array :		 modes_array
								 sets the `self.modes_array` attribute.

		"""
		import time
		import datetime

		self.modes_data = np.zeros([ell_max + 1, 2 * (ell_max + 1) + 1, data_len], dtype=np.complex128)

		# Set the time metadata
		time_now = time.localtime()
		time_now = time.strftime("%H:%M:%S", time_now)

		date_now = str(datetime.date.today())

		self.time = time_now
		self.date = date_now

	@property
	def data_len(self):
		""" Returns the length of the time/frequency axis.

		Returns
		-------

		data_len :	int
					THe length of the time/frequency axis.

		"""

		try:
			data_len = len(self.time_axis)

		except:
			data_len = len(self.frequency_axis)

		return data_len

	@property
	def delta_t(self, value=None):
		""" Sets and returns the value of time stepping :math:`dt`.

		Parameters
		----------

		value : float, optional
				The value of :math:`dt`
				to set to the attribute.

		Returns
		-------

		self.delta_t :	 float
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
		""" Sets and returns the value of frequency stepping :math:`df`.

		Parameters
		----------

		value : float, optional
				The value of :math:`df`
				to set to the attribute.

		Returns
		-------

		delta_f :	 float
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

	def load_modes(self, r_ext=500, ell_max=None, pre_key=None, modes_list=None, crop=False, center=True):
		""" Load the waveform mode data from an hdf file.

			Parameters
			----------

			pre_key :	str, optional
						A string containing the key of the group in
						the HDF file in which the modes` dataset exists.
						It defaults to `None`.

			mode_numbers :	list
							The mode numbers to load from the file.
							Each item in the list is a list that
							contains two integrer numbers, one for
							the mode index :math:`\\ell` and the
							other for the mode index :math:`m`.

			crop :	bool
					Whether or not to crop the beginning of the input
					waveform. If yes, the first :math:`t=r_{ext}`
					length of data will be discarded.

			Returns
			-------

			waveform_obj :	3d array
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
		import datetime
		import json
		# get the full path.
		full_path = self.data_dir + self.file_name

		cflag = 0


		# Open the modes file.
		with h5py.File(full_path, "r") as wfile:


			# Load metadata if present.
			try:
				metadata_bytes = bytes(np.void(wfile['metadata'])).decode()
				metadata = json.loads(metadata_bytes)
				self.__dict__.update(metadata)
				message('Metadata loaded')

			except:
				pass
			# data = np.array(wfile['l0_m0_r500.00'])
			# print(data)
			# Get the list of keys.
			modes_keys_list = sorted(list(wfile.keys()))
			#self.mode_keys_list = modes_keys_list
			# Construct the list of modes if it doesnt exist.
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
					pass

			else:
				self.modes_list = modes_list
				# If modes list is given, get ell_max from it.
				if not ell_max:
					# Get the ell max
					ell_max = max([item[0] for item in modes_list])

			# Set the ell_max attribute if not already.
			if not self.ell_max:
				self.ell_max = ell_max

			# Load the modes listed in mode_numbers list
			for item in self.modes_list:
				# For every ell mode list in modes_list

				ell_value, emm_list = item

				# Iteration index
				index = 0
				# Key found token
				token = -1

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
							#self.data_len = data_len
							# Delete the attribute
							del self.modes_data
							# Create an array for the waveform mode object
							self._create_modes_array(self.ell_max, data_len)
							# self.modes_data = np.zeros([ell_max+1, 2*(ell_max+1) +1, data_len], dtype=np.complex128)
							cflag = 1

							# set the time axis.
							#self.time_axis = time_axis[shift:]

					self.modes_data[ell_value, emm_index] = data_re[shift:] + 1j * data_im[shift:]
			maxloc	= np.argmax(np.absolute(self.mode(2, 2)))
			maxtime = time_axis[shift+maxloc]
			self.maxtime = maxtime
			print('Max time is', maxtime)
			self.time_axis = time_axis[shift:] - maxtime


	def save_modes(self, r_ext=None, ell_max=None, pre_key = None, key_format=None, mode_numbers=None, out_file_name='mp_psi4_new_modes.h5'):
		''' Save the waveform mode data to an hdf file.

		Parameters
		----------

		pre_key :	str, optional
					A string containing the key of the group in
					the HDF file in which the modes` dataset exists.
					It defaults to `None`.

		mode_numbers :	list
						The mode numbers to load from the file.
						Each item in the list is a list that
						contains two integrer numbers, one for
						the mode index :math:`\\ell` and the
						other for the mode index :math:`m`.

		Returns
		-------

		waveform_obj :	3d array
						Sets the three dimensional array `waveform.modes` that contains
						the required :math:`\\ell, m` modes.

		Examples
		--------

		>>> from waveformtools.waveforms import waveform
		>>> waveform.set_basedir('./')
		>>> waveform.set_datadir('data/')
		>>> mode_numbers = [[2, 2], [3, 3]]
		>>> waveform.load_data(mode_numbers=mode_numbers)
		'''

		import json

		self.out_file_name = out_file_name
		# get the full path.
		full_path = self.data_dir + self.out_file_name

		if not r_ext:
			r_ext = self.r_ext

		# Identify the modes to save.
		if ell_max:
			try:
				modes_to_save = self.modes_list[:ell_max]
			except:
				pass
		else:
			modes_to_save = self.modes_list

		# Create the modes file.
		with h5py.File(full_path, "w") as wfile:

		# Create the metadata dataset.
			metadata = self.get_metadata()

			metadata_bytes = json.dumps(metadata).encode()

			#dt = h5py.special_dtype(vlen=str)
			#metadata=np.asarray([metadata_bytes], dtype=dt)
			wfile.create_dataset('metadata', data=metadata_bytes)

			# Load the modes listed in mode_numbers list
			for item in modes_to_save:
				# For every ell mode list in modes_list

				ell_value, emm_list = item

				for emm_value in emm_list:
					# For every (ell, emm) mode.

					data	  = self.mode(ell_value, emm_value)
					# set the time and data axis
					data_re   = data.real
					data_im   = data.imag

					save_data = np.transpose(np.array([self.time_axis, data_re, data_im]))
					# Make the key
					key = _key_gen(ell_value, emm_value, extras=f'r{r_ext:.2f}')
					#print('Processing key', key)
					# Create data set
					wfile.create_dataset(key, data=save_data)

	def set_mode_data(self, ell_value, emm_value, data):
		""" Set the mode array data for the respective :math:`(\\ell, m)` mode.


				Parameters
				----------

				ell_value :		int
								The :math:`\\ell` polar mode number.

				emm_value :		int
								The :math:`emm` azimuthal mode number.

				data :		1d array
							The array consisting of mode data for the requested mode.

				Returns
				-------

				self.mode_data :	modes_data
									The updated modes data.

				"""
		# Compute the emm index given ell.
		emm_index = emm_value + ell_value

		# Set the mode data.
		self.modes_data[ell_value, emm_index] = data

	def to_frequency_basis(self):
		""" Compute the modes in frequency basis.

		Returns
		-------

		waveform_tilde_modes :	modes_array
								A modes_array containing the modes
								in frequency basis.

		"""

		# Create a new modes array
		waveform_tilde_modes = modes_array(label="frequency_domain")
		waveform_tilde_modes._create_modes_array(ell_max=self.ell_max, data_len=self.data_len)

		from transforms import compute_fft

		for mode in modes_list:
			# Extrapolate every mode

			# Ge the ell value
			ell_value, emm_list = mode

			for emm_value in emm_list:

				freq_axis, freq_data = compute_fft(self.mode(ell_value, emm_value), self.delta_t)

				waveform_tilde_modes.set_mode_data(ell_value, emm_value, freq_data)

		waveform_tilde_modes.frequency_axis = freq_axis

		return waveform_tilde_modes

	def to_time_basis(self):
		""" Compute the modes in time basis.

		Returns
		-------

		waveform_modes :  modes_array
						  A modes_array containing the modes
						  in frequency basis.

		"""

		# Create a new modes array
		waveform_modes = modes_array(label="time_domain")
		waveform_modes._create_modes_array(ell_max=self.ell_max, data_len=self.data_len)

		from transforms import compute_ifft

		for mode in modes_list:
			# Extrapolate every mode

			# Ge the ell value
			ell_value, emm_list = mode

			for emm_value in emm_list:

				time_axis, time_data = compute_ifft(self.mode(ell_value, emm_value), self.delta_f)

				waveform_modes.set_mode_data(ell_value, emm_value, time_data)

		maxloc = np.argmax(np.absolute(waveform_modes.mode(2, 2)))
		maxtime = time_axis[maxloc]

		waveform_modes.time_axis = time_axis - maxtime

		return waveform_tilde_modes

	def extrap_to_inf(self, mass=1, spin=None, modes_list="all", method="SIO"):
		""" Extrapolate the :math:`\\Psi_4` modes to infinity
			using the perturbative improved second order method.

			Parameters
			----------

			mass :		float
						The effective total mass of the system.

			spin :		float
						The effective spin of the system.

			modes :		modes array, optional
						The modes to extrapolate. Defaults
						to `all` if not specified.

			method :	str
						The method to use for extrapolation. The available methods are:

						 Method str | Name									|
						------------+---------------------------------------+
						'FO'		| First order							|
						'SO'		| Second order							|
						'SIO'		| Second improved order					|
						'NM'		| Numerical method (not ready yet)		|

			Returns
			-------

			waveform_inf_modes :	modes array
									A new modes array that contains
									the extrapolated modes.

			"""

		from functools import partial
		from waveformtools import extrapolate

		from waveformtools.extrapolate import (
			waveextract_to_inf_perturbative_one_order,
			waveextract_to_inf_perturbative_two_order,
			waveextract_to_inf_perturbative_twop5_order,
		)

		# Prepare the extrapolating method.
		if method == "SIO":
			extrap_method = partial(
				waveextract_to_inf_perturbative_twop5_order,
				delta_t=self.delta_t,
				areal_radius=self.r_ext,
				Mass=mass,
				spin=spin,
			)

		if method == "SO":
			extrap_method = partial(
				waveextract_to_inf_perturbative_two_order,
				delta_t=self.delta_t,
				areal_radius=self.r_ext,
				Mass=mass,
				spin=spin,
			)

		if method == "FO":
			extrap_method = partial(
				waveextract_to_inf_perturbative_one_order, u_ret=self.time_axis, areal_radius=self.r_ext, Mass=mass
			)

		if method == "NM":
			print("This method is not available yet! ")

		# Prepare the modes to be extrapolated.
		if modes_list == "all":
			modes_list = construct_mode_list(self.ell_max)

		# Create a mode array for the extrapolated waveform.
		extrap_wf = modes_array(label="rPsi4_inf")

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
				mode_data = self.r_ext * self.mode(ell_value, emm_value)

				# Extrapolate
				# import ipdb; ipdb.set_trace()
				extrap_mode_data = extrap_method(rPsi4_rlm=mode_data)

				# Assign data to new modes array
				extrap_wf.set_mode_data(ell_value, emm_value, extrap_mode_data / self.r_ext)

		print("Done!")
		return extrap_wf

	def supertranslate(self, supertrans_alpha_modes, gridinfo, order=4):

		""" Supertranslate the :math:`\\Psi_{4\\ell m}` waveform modes, give the,
				the supertranslation parameter and the order.

				Parameters
				----------

				supertransl_alpha_modes :  modes_array
										   The modes_array containing the
										   supertranslation mode coefficients.


				gridinfo :		class instance
								The class instance that contains
								the properties of the spherical grid
								using which the computations are
								carried out.

				order :		int
							The number of terms to use to
							approximate the supertranslation.


				Returns
				-------

				Psi4_supertransl :	  modes_array
									  A class instance containg the
									  modes of the supertranslated
									  waveform :math:`\\Psi_4`.

				"""

		ell_max = self.ell_max
		# Step 0: Get the grid properties for integrations

		# Compute the meshgrid for theta and phi.
		theta, phi = gridinfo.meshgrid
		# theta
		# Step 1: get the grid function for supertranslation parameter
		supertransl_alpha_sphere = BMS.compute_supertransl_alpha(supertransl_alpha_modes, theta, phi)

		# The supertranslation is carried out in frequency space.
		# Step 2: get the FFT of the mode coefficients of Psi4
		Psi4_tilde_modes = self.to_frequency_space()

		# Get the 2d angular frequency array
		omega_axis_2d = Psi4_tilde_modes.omega

		# Construct the supertranslation factor
		supertransl_factor = np.sum(
			np.array([np.power((-1j * omega_axis_2d * supertransl_alpha_sphere), index) for index in range(order)]),
			axis=0,
		)

		# Multiply with the fourier modes.
		supertransl_spherical_factor = Psi4_tilde_modes.multiply(supertransl_factor)

		from transforms import Yslm

		# Reconstruct the modes
		for ell_value in range(ell_max):
			for emm_value in range(-ell_value, ell_value + 1):
				# Multiply with the SWSH basis.
				supertransl_spherical_grid += supertransl_spherical_factor * Yslm(
					spin_weight=-2, ell=ell_value, emm=emm_value, theta=theta, phi=phi
				)

				# Step 3: Reconstruct the function on the sphere

		# Create a spherical_array to hold the supertranslated waveform
		supertransl_spherical_waveform = spherical_array(gridinfo=gridinfo)

		# Set the data.
		supertransl_spherical_waveform.data = supertransl_spherical_grid

		# Get modes_array from spherical_array
		Psi4_supertransl_modes = supertransl_spherical_waveform.to_modes_array(ell_max=ell_max)

		return Psi4_supertransl_modes

	def boost(self, conformal_factor):
		""" Boost the waveform given the unboosted waveform and the boost conformal factor.

			Parameters
			----------

			conformal_factor :		2d array
									The conformal factor for the Lorentz transformation.
									It may be a single floating point number or an array
									on a spherical grid. The array will be of dimensions
									[ntheta, nphi]


			Returns
			-------

			boosted_waveform :	  spherical_array
								  The class instance `spherical_array`
								  that contains the boosted waveform.
			"""

		from grids import spherical_grid

		# Construct a spherical grid.
		info = spherical_grid()

		# Get spherical array from modes.
		unboosted_waveform = self.to_spherical_array(info)

		# Compute the meshgrid for theta and phi.
		theta, phi = unboosted_waveform.gridinfo.meshgrid

		# A list to store the boosted waveform.
		boosted_waveform_data = []

		for item in unboosted_waveform.data:
			# Compute the boosted waveform on the spherical grid on all the elements.

			conformal_k_on_sphere = compute_conformal_k(vec_v, theta, phi)
			boosted_waveform_item = conformal_k_on_sphere * item

			boosted_waveform_data.append(boosted_waveform_item)

		# Construct a 2d waveform array object
		boosted_waveform = spherical_array(gridinfo=unboosted_waveform.gridinfo, data=np.array(boosted_waveform_data))
		boosted_waveform.label = "boosted"

		# Get modes from spherical data.
		boosted_waveform_modes = boosted_waveform.to_modes_array()

		return boosted_waveform_modes






def _get_modes_list_from_keys(keys_list, r_ext):
	''' Get the modes list from the keys list
	of an hdf file.

	Parameters
	----------

	keys_list :		list
					The list containing all the keys

	r_ext :		float
				The extraction radius of the data.

	Returns
	-------

	modes_list :	list
					The list of modes.

	'''


	# Sort the keys to ensure a nice
	# modes list structure.
	keys_list = sorted(keys_list)
	keys_list = [item for item in keys_list if f'r{r_ext}' in item]

	#print('List of keys received', keys_list)
	# The list of modes.
	modes_list = []

	# Initialize the emm modes sublist.
	emm_modes_for_ell = []

	# Present ell value to
	# initialize the mode concatenation.
	ell_old = 0

	for key in keys_list:
		#print(key)
		# Get the ell value
		ell_value, emm_value = _get_ell_emm_from_key(key)


		if ell_value!=ell_old:
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


	return modes_list


def _get_ell_emm_from_key(key):
	''' Get the :math:`\\ell` and :math:`m` values
	from a given key string of an hdf file.

	Parameters
	----------

	key :	 str
			 The input key string

	Returns
	-------

	ell_value :		int
					The :math:`\\ell` value

	emm_value :		int
					The :math:`m` value.


	Notes
	-----

	Assumes that the input string has :math:`\\ell` and :math:`m` values
	in the form `l\{int\}m\{int\}`.

	'''

	import re

	mo = re.search('l\d*', key)
	ell_str_start = mo.start()
	ell_str_end   = mo.end()
	ell_value	  = int(key[ell_str_start+1 : ell_str_end])

	mo = re.search('m-*\d*', key)
	emm_str_start = mo.start()
	emm_str_end   = mo.end()
	emm_value	  = int(key[emm_str_start+1 : emm_str_end])

	return ell_value, emm_value


def construct_mode_list(ell_max, spin_weight=-2):
	''' Construct a modes list in the form [[ell1, [emm1, emm2, ...], [ell2, [emm..]],..]
	given the :math:`\\ell_{max}.`

	Parameters
	----------
	spin_weight :	 int
					 The spin weight of the modes.

	ell_max :	int
				The :math:`\\ell_{max}` of the modes list.

	Returns
	-------

	modes_list:		list
					A list containg the mode indices lists.

	Notes
	-----

	The modes list is the form which the `waveform` object understands.
	'''

	# The modes list.
	modes_list = []

	for ell_index in range(abs(spin_weight), ell_max):
		# Append all emm modes for each ell mode.
		modes_list.append([ell_index, [m_index for m_index in range(-ell_index, ell_index+1)]])

	return modes_list

def _key_gen(ell, emm, extras=None):
	''' Generates strings to be used as keys for
	managing h5 datasets.

	Parameters
	----------
	ell :	int
			The polar angular mode number
			:math:`\\ell`.

	emm : int
		  The aximuthal angular mode number
		  :math:`m`.


	extras :	str
				Any extra string to be appended
				to the end of the key.

	Returns
	-------

	key :	 str
			 A string key.

	'''

	key = f'l{ell}_m{emm}'

	if extras!=None:
		key += f'_{extras}'
		#print('adding rext')

	return key
