'''  A Class for handling the waveform data '''
G

############################
# Imports
############################


import numpy as np
import h5py
from waveformtools.waveformtools import message

class waveform:
	""" A class for handling waveforms.




	Attributes
	----------

		Primary



"""

	def __init__(self,	timeaxis=None,
						wavedata=None,
						base_dir=None,
						data_dir=None,
						filename=None,
						frequency_series=None,
						key_format=None,
						mode=None,
						ell_max = 8
						):

		self.basedir						= base_dir # The base directory containing the.
		self.datadir						= data_dir # The directory inside the base directory containing the waveform data.
		self.filename						= filename # The name of the file containing the data.
		self.time_axis						= time_axis # The time axis of the data.
		self.time_domain_waveform			= time_domain_waveform # The 2D time domain waveform defined on a sphere.
		self.delta_t						= delta_t # The uniform time stepping delta_t
		self.frequency_domain_waveform		= frequency_domain_waveform # The frequency domain waveform
		self.frequency_axis					= frequency_axis # The frequency axis.
		self.key_format						= key_format # An example key to fetch data from h5 file.
		self.modes							= modes # The list of modes present in the waveform.
		self.modes_time_domain				= modes_time_domain # The time-domain waveform decomposed into modes.
		self.modes_frequency_domain			= modes_frequency_domain # The frequency-domain waveform decomposed into modes.


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


	@property
	def delta_t(self):
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

		return self._delta_t

	@delta_t.setter
	def delta_t(self, delta_t):
		''' Set the time steping delta_t

		delta_t :	float
					The time stepping `delta_t`

		Returns
		-------

		Sets the variable `waveform.delta_t`.`

		'''
		self._delta_t = delta_t

	@delta_t.deleter
	def delta_t(self):
		''' Delete the variable value '''

		del self._delta_t


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

		if self.modess_time






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



	def reconstr_waveform(delta_theta=0.1, delta_phi=0.1):
		''' Reconstruct the waveforms on a sphere given it`s modes.


		Parameters
		----------
		delta_theta :	float
						The polar angle step size.

		delta_phi :	float
					The aximuthal angle step size.


		Returns
		-------

		waveform_data :	func
						The waveform on a sphere.


		Notes
		-----

		The waveform is interpolated using `scipy.interpolate.interp1d` using third order
		method.

		'''

	def decompose_waaveform(self, spin_weight=-2, ell_max=8):
		''' Decompose the given waveform into SWSHs.

		Parameters
		----------

		spin_weight :	int, optional
						The spin weight of the SWSH basis.

		ell_max :	int
					The maximum angular mode number in the basis.

		Returns
		-------

		waveform_data :	1d array
						The one dimensional array of SWSH coefficients, one for each
						time/ frequency step.

		'''



	def resample_mode_data(self, new_delta_t=None, new_delta_f=None):
		''' Resample the given timeseries/ frequencyseries waveform mode data. Resamples at available `delta_t` or `delta_f`
			by default. If `new_delta_x` is given, the resampling happens at the supplied new value.

		Parameters
		----------

		new_delta_t :	float, optional
						The new time domain sampling width.

		new_delta_f :	float, optional
						The new frequency domain sampling width.

		Returns
		-------

		waveform_mode_data :	1d array
								The array of resampled waveform modes.

		Notes
		-----

		Resamples the timeseries and frequencyseries modes	if `new_delta_t` and `new_delta_f` are given, respectively.
		If none are given, then it resamples the timedomain modes at the default time stepping `delta_t`.


		'''




	def extrapolate_to_inf_per(self, mass=1, spin=0):
		''' Extrapolate the given set of waveform modes to null infinity using the 2.5 order perturbative method.

		Parameters
		----------

		mass :	float, optional
				The total mass of the system.

		spin :	float, optional
				The total spin of the system.

		Returns
		-------

		extrapolated_waveform_modes :	waveforms object
										A wavwforms object containig the extrapolated waveform.


		See Also
		--------

		waveformtools.extrapolate.extrapolate_to_inf_pert_2p5_order

		'''


	def extrapolate_to_inf_numeric(self, ext_radii, set_of_waveforms):
		''' Extrapolate a given set of waveform modes to null infinity using the numerical
			method.


		Parameters
		----------

		ext_radii :	list
					A list containing the values of the extraction radii of the waveforms.

		set_of_waveforms :	list
							A list of waveforms at the various extraction radii.

		Returns
		-------

		waveform_at_inf :	1d array
							A 1d array of modes of the approximate extrapolated waveform at future null infninity.


		See Also
		--------

		waveformtools.extrapolate.extrapolate_to_inf_numeric

		'''



	def apply_CoM_correction(self, alpha, beta):
		''' Apply the Centre of Mass corrections to the waveform modes, given its mean motion.

		Parameters
		----------

		alpha :	list
				The list containig the mean displacement of the CoM.
		beta :	list
				The list containig the mean drift oi the CoM.

		Returns
		-------

		com_corrected_modes :	waveforms obj
								A waveforms object containing the CoM corrected waveform modes.


		See Also
		--------

		waveformttools.

		'''


	def get_strain_from_psi4(self, omega0=None):
		''' Get the strain modes from :math:`\\Psi_4` modes.

		Parameters
		----------

		omega0 :	float, optional
					The cutoff angular frequency for fixed frequency integration.
					Defaults to one-tenth of the initial frequency of the mode if not
					given.

		Returns
		-------

		strain_modes :	waveforms obj
						The mode coefficients of the stain modes obtained by a double time
						integration of :math:`\\Psi_4`.
		'''

	def get_news_from_psi4(self, omega0=None):
		''' Get the news modes from :math:`\\Psi_4` modes.

		Parameters
		----------

		omega0 :	float, optional
					The cutoff angular frequency for fixed frequency integration.
					Defaults to one-tenth of the initial frequency of the mode if not
					given.

		Returns
		-------

		news_modes :	waveforms obj
						The mode coefficients of the News, obtained by a single time
						integration of :math:`\\Psi_4`.

		'''


	def get_psi4_from_news(self, method='CS'):
		''' Get the :math:`\\Psi_4` modes from the News function by differentiation.

		Parameters
		----------

		method :	str
					The method to use for computing the derivative.
					The available methods are 'CS' (Chebyshev series)
					or 'FD' (Smoothened finite differences).

		Returns
		-------

		Psi4 :	waveforms obj
				The modes coefficients of the :math:`\\Psi_4` waveform, obtained by differentiating the news
				function once in the time domain.


		See Also
		--------

		waveformtools.differentiate.Chebyshev_differential, wavefotmtools.differentiate.differentiate5


		'''


	def get_strain_from_news(self, omega0=None):
		''' Get the strain	modes from the News modes.

			Parameters
			----------

			omega0 :	float, optional
						The cutoff angular frequency for fixed frequency integration.
						Defaults to one-tenth of the initial frequency of the mode if not
						given.

			Returns
			-------

			strain_modes :	waveforms obj
							The mode coefficients of the stain modes obtained by a single time
							integration of the News.

			'''


	def get_psi4_from_strain(self, method='CS'):
		''' Get the :math:`\\Psi_4` modes from the strain by differentiation.

		Parameters
		----------

		method :	str
					The method to use for computing the derivative.
					The available methods are 'CS' (Chebyshev series)
					or 'FD' (Smoothened finite differences).

		Returns
		-------

		Psi4 :	waveforms obj
				The modes coefficients of the :math:`\\Psi_4` waveform, obtained by differentiating the strain
				function twice in the time domain.


		See Also
		--------

		waveformtools.differentiate.Chebyshev_differential, wavefotmtools.differentiate.differentiate5


		'''


	def get_news_from_strain(self):
		''' Get the :math:`\\Psi_4` modes from the strain by differentiation.

		Parameters
		----------

		method :	str
					The method to use for computing the derivative.
					The available methods are 'CS' (Chebyshev series)
					or 'FD' (Smoothened finite differences).

		Returns
		-------

		News :	waveforms obj
				The modes coefficients of the News waveform, obtained by differentiating the strain
				function once in the time domain.


		See Also
		--------

		waveformtools.differentiate.Chebyshev_differential, wavefotmtools.differentiate.differentiate5


		'''



	def apply_supertranslation(self, alpha):
		''' Apply the given supertranslation transfoamation :math:`\\alpha` approximately to the waveform.

		Parameters
		----------

		alpha :	list
				The list containing the supertranslation three-vector.

		Returns
		-------

		supertranslated_waveform :	waveforms obj
									The supertranslated waveform modes.

		Notes
		-----

		The supertranslation applied is approximate, given....

		'''

	def apply_boost():
		''' Apply the given boost transfoamation :math:`\\beta` to the waveform.

		Parameters
		----------

		beta :	list
				The list containing the boost three-vector.

		Returns
		-------

		supertranslated_waveform :	waveforms obj
									The boosted waveform modes.

		'''





	# @base_dir.setter
	# def base_dir(self, base_dir):
	#	 self.__base_dir = base_dir
