''' Classes to hold grid information '''

import numpy as np
#from numba import jit, njit, jitclass
#from numba.experimental import jitclass
from numba import jit, njit
#from numba import jitclass          # import the decorator
import numba as nb
from numba.experimental import jitclass

spec_sp = { 'ntheta' : nb.int32,
            'nphi' : nb.int32,
            'nghosts' : nb.int32,
            'nthetamax' : nb.int32,
            'nphimax' : nb.int32,
}

@jitclass(spec_sp)
class spherical_grid:
	''' A class to store the coordinate grid on a sphere.

	Attributes
	----------

	ntheta:	int
				The number of angular points in the :math:`\\theta`
				direction, including ghost zones.


	nphi:	int
			The number of angular points in the :math:`\\phi`
			direction, including ghost zones.

	nghosts:	int
				The number of ghost zones at the end of
				each direction.

	meshgrid:	tuple of 2d array
				The 2d array containing the meshgrid of
				(:math:`\\theta, \\phi`) angular points.


	theta_1d:	1d array
				The 1d array of angular points
				along the :math:`\\theta` axis.

	phi_1d:	1d array
				The 1d array of angular points
				along the :math:`\\phi` axis.

	dtheta:	float
				The angular step size in the :math:`\\theta`
				direction.

	dphi:	float
			The angular step size inthe :math:`\\phi`
			direction.

	npix_act:	int
				The total number of gridpoints on the sphere,
				excluding the ghost points.


	'''

	def __init__(self,
				 nphi		  = 80,
				 ntheta		  = 41,
				 nphimax	  = 124,
				 nthetamax	  = 66,
				 nghosts	  = 2):

		#Number of gridpoints along phi direction including ghost points.
		self.nphi	   = nphi
		#Number of gridpoints along theta direction including ghost points.
		self.ntheta    = ntheta
		#Total length of phi array used by ETK.
		self.nphimax   = nphimax
		#Total length of theta array used by ETK.
		self.nthetamax = nthetamax
		#Number of ghost points in theta/phi direction.
		self.nghosts   = nghosts

	@property
	def npix(self):
		''' Return the total number of pixels, including the ghost zones present at one iteration. '''
		return (self.ntheta)*(self.nphi)

	@property
	def npix_act(self):
		''' Return the actual number of pixels, excluding the ghost zones present at one iteration '''
		return (self.ntheta-2*self.nghosts)*(self.nphi - 2*self.nghosts)

	@property
	def npix_max(self):
		''' Return the (max) total number of pixels, including the ghost and buffer zones at one iteration.'''
		return (self.nthetamax)*(self.nphimax)

	@property
	def ntheta_act(self):
		''' Return the actual number of valid pixels, excluding the ghost and buffer zones, along the theta axis at one iteration.'''
		return self.ntheta - 2*self.nghosts

	@property
	def nphi_act(self):
		''' Return the actual number of valid pixels, excluding the ghost and buffer zones, along the phi axis at one iteration. '''
		return self.nphi - 2*self.nghosts

	@property
	def dtheta(self):
		''' Return the coodinate spacing :math:`d\\theta.` '''
		return np.pi/(self.ntheta - 2*self.nghosts)

	@property
	def dphi(self):
		''' Return the coordinate spacing :math:`d\\phi`. '''
		return 2*np.pi/(self.nphi - 2*self.nghosts)

	@property
	def nbuffer(self):
		''' Return the number of buffer zones (excluding ghosts) '''
		return self.nthetamax-self.ntheta

	def theta_1d(self, theta_index=None):
		''' Returns the coordinate value theta given the coordinate index. The coordinate index ranges from (0, ntheta).
			The actual indices without the ghost and extra zones is (nghosts, ntheta-nghosts).

		Parameters
		-----------

		theta_index:	int/ 1d array
						The theta coordinate index or axis.

		Returns
		-------

		theta_1d:	float
					The coordinate(s) :math:`\\theta` on the sphere.

		'''

		if not theta_index:
			theta_index = np.arange(self.nghosts, self.ntheta-self.nghosts)

		return (theta_index - self.nghosts + 0.5) * np.pi / (self.ntheta-2*self.nghosts)

	def phi_1d(self, phi_index=None):
		''' Returns the coordinate value theta given the coordinate index. The coordinate index lies in (0, nphi).
			The actual indices without the ghost and extra zones is (nghosts, nphi-nghosts).

		Parameters
		-----------

		phi_1d:	int / 1d array
					The phi coordinate index or axis.

		Returns
		-------

		phi_1d:	float or 1d array
					The coordinate(s) :math:`\\phi` on the sphere.

		'''

		if not phi_index:
			phi_index = np.arange(self.nghosts, self.nphi-self.nghosts)

		return (phi_index - self.nghosts) * 2*np.pi / (self.nphi-2*self.nghosts)



	@property
	def meshgrid(self):
		''' The (:math:`\\theta, \\phi)`: coordinate meshes.
			Excludes the ghost zones.


		Returns
		-------

		theta:	2d array
				The :math:`\\theta` coordinate matrix for vectorization.

		phi:	2d array
				The :math:`\\phi` coordinate matrix for vectorization.

		'''

		#theta, phi	  =   np.meshgrid(self.theta_1d(), self.phi_1d())

		theta = np.zeros((self.ntheta_act, self.nphi_act))
		phi   = np.zeros((self.ntheta_act, self.nphi_act))

		for theta_index, theta_val in enumerate(self.theta_1d()):
			for phi_index, phi_val in enumerate(self.phi_1d()):
				theta[theta_index, phi_index] = theta_val
				phi[theta_index, phi_index] = phi_val


		#return np.transpose(theta), np.transpose(phi)
		return theta, phi
