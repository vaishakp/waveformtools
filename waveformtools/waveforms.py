##########################################################################
# A Class for handling the waveform data
##########################################################################

import numpy as np
import h5py
from waveformtools.waveformtools import message

class waveform:
    """ A class for handling waveforms."""

    def __init__(self,	timeaxis=None,
						wavedata=None,
						base_dir=None,
						data_dir=None,
						filename=None
						frequency_series=None
						key_format=None
						mode=None
						ell_max = 8
						):

        self.basedir						= base_dir # The base directory containing the
        self.datadir						= data_dir
        self.filename						= filename
        self.time_axis						= timeaxis
        self.time_domain_waveform			= time_domain_waveform
		self.delta_t						= delta_t
		self.frequency_domain_waveform		= frequency_domain_waveform
		self.frequency_axis					= frequency_axis
		self.key_format						= key_format
		self.modes							= modes
		self.modes_data						= modes_data

    def load_modes(self, ell_max, pre_key = None, mode_numbers=None):
		''' Load the waveform mode data from an hdf file.

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

		# get the full path.
        full_path = self.base_dir + self.data_dir + self.filename

		# Load the modes listed in mode_numbers list
		for item in mode_numbers:
			ell, emm = item

			# Find the data key in the file
			with h5py.File(full_path, "r") as wfile:

				# List all dataset keys.
				keys = list(wfile.keys())

				# Iteration index
				index = 0
				# Key found token
				token = -1

				# Find the key corresponding to the mode
				while token < 0 and index < len(keys):
					# Get the current key
					key = keys[index]
					# Update the token if the required mode string is found
					token = key.find(f"l{ell}_m{emm}")
					# message(key)
					index += 1

				if token < 0:
					message("Waveform dataset not found")
				else:
					message(key)

				# Get the data
				data = np.array(wfile[key])

				# set the time and data axis
				time_axis = data[:, 0]
				data_re = data[:, 1]
				data_im = data[:, 2]

				data_len = len(timeaxis)

				if not waveform.modes:
					# Create an array for the waveform mode object
					self.modes_data	= np.zeros([ell_max, 2*ell_max +1, data_len])

				self.time_axis = time_axis
				self.modes_data[ell, emm] = data_re + 1j*data_im



	def set_delta_t(self, delta_t = None)
		''' Set the time step of the waveform. Sets the value from the timeaxis
			if the optional argument `delta_t` is not specified.

		Parameters
		----------

		delta_t :	float
					The time stepping `delta_t`

		Returns
		-------

		Sets the variable `waveform.delta_t`.`

		'''

        self.delta_t = self.timeaxis[1] - self.timeaxis[0]



	def get_frequency_domain_waveform(self, new_delta_f = None):
		''' Get the frequency domain waveform from the timedomain waveform.

		Parameters
		----------

		new_delta_f :	float, optional
						The sampling frequency. If set, then the frequency domain
						data is resampled at the requested sampling frequency.

		Returns
		-------

		frequency_axis :	1d array
							The frequency axis of the FFT. Also sets the
							variable `waveforms.frequency_axis`

		frequency_domain_waveform :	1d array
									The frequency domain waveform. Also sets
									the variable `waveforms.frequency_axis`


		'''




	def get_angular_frequencies(self, method='CS'):
		''' Get the instantaneous angular frequency of the waveform.


		Parameters
		----------

		method :	str
					The method to use for computing the derivative.
					The available methods are 'CS' (Chebyshev series)
					or 'FD' (Smoothened finite differences).
		Returns
		-------

		waveforms.omega :	1d array
							The instantaneous frequencies of the waveform.
							Also sets the corresponding variable.

		'''



	def reconstr_waveform():


	def decompose_waaveform():


	def resample_data():


	def extrapolate_to_inf_per():


	def extrapolate_to_inf_numeric():


	def apply_CoM_correction():


	def get_strain_from_psi4():


	def get_news_from_psi4():

	def get_psi4_from_news():

	def get_strain_from_news():


	def get_psi4_from_strain():

	def get_news_from_strain():


	def apply_supertranslation():

	def apply_boost():




    # @base_dir.setter
    # def base_dir(self, base_dir):
    # 	 self.__base_dir = base_dir
