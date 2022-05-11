''' Methods for waveform extrapolation. '''


###############################################
# Basic utilities
###############################################

def r_to_ra_conversion(coord_radius, Mass=1, spin=0):
	''' Convert the isotropic co-ordinate radius parameter `r' in the ETK simulations
		into the approximate areal radius.

	Inputs
	-------

	coord_radius :	float
					The coordinate radius in the Einstein toolkit

	Mass :	float, optional
			The sum of the quasi-local horizon (Christodolou) masses of the black holes.
			Defaults to 1


	spin :	float, optional
			The magnitude of the spin of the system, as approximated by a single Kerr black hole
			far away from the system. Defaults to 0.


	Retuens
	-------

	areal_radius :	float
					The appriximate areal radius of the sphere.


	Assumes
	--------

	The system interoir to the sphere at co-ordinate radius `r_coord' is well approximated by a
	Kerr black hole.



	'''





