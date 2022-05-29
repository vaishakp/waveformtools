''' Classes to hold grid information '''

class spherical_grid:
    ''' A class to store the coordinate grid on a sphere. '''

    def __init__(self,
                 nphi         = 80,
                 ntheta       = 41,
                 nphimax      = 124,
                 nthetamax    = 66,
                 nghosts      = 2):

        #Number of gridpoints along phi direction including ghost points.
        self.nphi      = nphi
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
        #Return the total number of pixels, including the ghost zones present at one iteration.
        return (self.ntheta)*(self.nphi)

    @property
    def npix_act(self):
        #Return the actual number of pixels, excluding the ghost zones present at one iteration.
        return (self.ntheta-2*self.nghosts)*(self.nphi - 2*self.nghosts)

    @property
    def npix_max(self):
        #Return the (max) total number of pixels, including the ghost and buffer zones at one iteration.
        return (self.nthetamax)*(self.nphimax)

    @property
    def ntheta_act(self):
        #Return the actual number of valid pixels, excluding the ghost and buffer zones, along the theta axis at one iteration.
        return (self.ntheta - 2*self.nghosts)

    @property
    def nphi_act(self):
        #Return the actual number of valid pixels, excluding the ghost and buffer zones, along the phi axis at one iteration.
        return (self.nphi - 2*self.nghosts)

    @property
    def dtheta(self):
        #Return the coodinate spacing d\theta
        return np.pi/(self.ntheta - 2*self.nghosts)

    @property
    def dphi(self):
        #Return the coordinate spacing d\phi
        return 2*np.pi/(self.nphi - 2*self.nghosts)
    @property
    def nbuffer(self):
        #Return the coordinate spacing d\phi
        return self.nthetamax-self.ntheta

	@property
	def theta_1d(self, theta_index=None):
    ''' Returns the coordinate value theta given the coordinate index. The coordinate index ranges from (0, ntheta).
		The actual indices without the ghost and extra zones is (nghosts, ntheta-nghosts).

    Parameters
    -----------

    theta_index :	int/ 1d array
					The theta coordinate index or axis.

    Returns
    -------

    theta_1d :	float
				The coordinate(s) :math:`\\theta` on the sphere.

    '''

	if not theta_index:
		theta_index = np.linspace(self.nghosts, self.ntheta-self.nghosts)

    return (theta_index - info.nghosts + 0.5) * np.pi / (info.ntheta-2*info.nghosts)

	@property
	def phi_1d(self, phi_index=None):
		''' Returns the coordinate value theta given the coordinate index. The coordinate index lies in (0, nphi).
			The actual indices without the ghost and extra zones is (nghosts, nphi-nghosts).

		Parameters
		-----------

		phi_1d :	int / 1d array
					The phi coordinate index or axis.

		info :  class instance
				The instance of the class that contains infomation about the 2d grid properties.

		Returns
		-------

		phi_1d :	float or 1d array
					The coordinate(s) :math:`\\phi` on the sphere.

		'''

		if not phi_index:
			phi_index = np.linspace(self.nghosts, self.nphi-self.nghosts)

		return (phi_index - info.nghosts) * 2*np.pi / (info.nphi-2*info.nghosts)



	@property
	def meshgrid(self):
		''' The (:math:`\\theta, \\phi)`: coordinate meshes.
			Excludes the ghost zones.


		Returns
		-------

		theta :	2d array
				The :math:`\\theta` coordinate matrix for vectorization.

		phi :	2d array
				The :math:`\\phi` coordinate matrix for vectorization.

		'''

		theta, phi    =   np.meshgrid(self.theta_1d, self.phi_1d)

		return theta, phi

