

#############################################################################################
# Imports
#############################################################################################
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import seaborn as sns
import scipy 
import handles
import qlm_handles
import os
import scipy
from scipy.optimize import curve_fit
from pathlib import Path



#############################################################################################
# Set up paths
#############################################################################################
home = str(Path.home())
PRE1	  = home

#############################################################################################
# Retrieval of QNM data from Berti et. al.
#############################################################################################


qnmdatdir = PRE1 + '/Documents/IUCAA/Projects/projects/Post_merger/' + 'qnm_data'


def qnm_details_from_nlm_mass_spin(s, n, l, m, mass, spin, qnmdatdir, verbose='no'):
	''' Returns the QNM details given n, l, m , mass , spin and frequency. '''
	subdir_string = 's%dl%d'%(s, l)
	
	if m<0:
		m_string = 'mm'
	else:
		m_string = 'm'
		
	string = 'n'+str(n) + 'l'+str(l) + m_string+str(abs(m))

	qnmdat = np.genfromtxt(qnmdatdir + '/' + subdir_string + '/' + string+ '.dat')

	spins	 = qnmdat[:, 0]
	omega_re = mass*qnmdat[:, 1]
	omega_im = mass*qnmdat[:, 2]
	Alm_re	 = qnmdat[:, 3] 
	Alm_im	 = qnmdat[:, 4]

	qnms_from_spin			 = [omega_re, omega_im, Alm_re, Alm_im]

	qnms_interp_from_spin	 = []
	for item in qnms_from_spin:
		qnms_interp_from_spin.append(scipy.interpolate.interp1d(spins, item))
		
	qnm_det = [spin] + [item(spin) for item in qnms_interp_from_spin]
	
	
	spin	 = qnm_det[0]
	Omegare  = qnm_det[1]
	Omegaim  = qnm_det[2]
	Alm_Re	 = qnm_det[3]
	Alm_Im	 = qnm_det[4]
	
	if verbose=='yes':
		print('Mode s%d n%d l%d m%d'%(s, n, l, m))
		print('The QNM details are:\n Spin: %f \t  OmegaRe: %f \t OmegaIm: %f \t Alm_Re: %f \t Alm_Im: %f'%(spin, Omegare, Omegaim, Alm_Re, Alm_Im))
	return qnm_det














######################################################################################################
# Functions for fitting
######################################################################################################



def dsin_gen_m_some_free(s, N, l, mass, spin, qnmdatdir, qnm_details_from_nlm_mass_spin, nfree=2):
	
	omega0 = [abs(qnm_details_from_nlm_mass_spin(s, N_index+1, l, m_index, mass, spin, qnmdatdir)[1]) for N_index in range(N) for m_index in range(-l, l+1)]
	gamma0 = [abs(qnm_details_from_nlm_mass_spin(s, N_index+1, l, m_index, mass, spin, qnmdatdir)[2]) for N_index in range(N) for m_index in range(-l, l+1)]
	
	if N==1:
	
		if nfree==0:

			def dsin_model(x,  A1, omega1, gamma1, phase1,	\
							   A2, omega2, gamma2, phase2,	\
							   A3, omega3, gamma3, phase3,	\
							   A4, omega4, gamma4, phase4,	\
							   A5, omega5, gamma5, phase5,	\
							   c):
				return	A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) +\
						A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x) +\
						A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x) +\
						A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x) +\
						A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x) +\
						c
			
			
		if nfree==1:

			def dsin_model(x,  A1, omega1, gamma1, phase1,	\
							   A2, omega2, gamma2, phase2,	\
							   A3, omega3, gamma3, phase3,	\
							   A4, omega4, gamma4, phase4,	\
							   A5, omega5, gamma5, phase5,	\
							   A_free1, omega_free1, gamma_free1, phase_free1, \
							   c):
				return	A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) +\
						A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x) +\
						A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x) +\
						A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x) +\
						A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x) +\
						A_free1*np.sin((omega_free1)*x + phase_free1) * np.exp(-(gamma_free1)*x) +\
						c
		if nfree==2:

			def dsin_model(x,  A1, omega1, gamma1, phase1,	\
							   A2, omega2, gamma2, phase2,	\
							   A3, omega3, gamma3, phase3,	\
							   A4, omega4, gamma4, phase4,	\
							   A5, omega5, gamma5, phase5,	\
							   A_free1, omega_free1, gamma_free1, phase_free1, \
							   A_free2, omega_free2, gamma_free2, phase_free2, \
							   c):
				return	A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) +\
						A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x) +\
						A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x) +\
						A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x) +\
						A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x) +\
						A_free1*np.sin((omega_free1)*x + phase_free1) * np.exp(-(gamma_free1)*x) +\
						A_free2*np.sin((omega_free2)*x + phase_free2) * np.exp(-(gamma_free2)*x) +\
						c

	if N==2:
	
	
		if nfree==0:
	
			def dsin_model(x,  A1, omega1, gamma1, phase1,\
							   A2, omega2, gamma2, phase2,\
							   A3, omega3, gamma3, phase3,\
							   A4, omega4, gamma4, phase4,\
							   A5, omega5, gamma5, phase5,\
							   A6, omega6, gamma6, phase6,\
							   A7, omega7, gamma7, phase7,\
							   A8, omega8, gamma8, phase8,\
							   A9, omega9, gamma9, phase9,\
							   A10, omega10, gamma10, phase10,\
							   c):
				return	A1	   *np.sin((omega1	+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x)   +\
						A2	   *np.sin((omega2	+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)   +\
						A3	   *np.sin((omega3	+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)   +\
						A4	   *np.sin((omega4	+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x)   +\
						A5	   *np.sin((omega5	+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x)   +\
						A6	   *np.sin((omega6	+omega0[5])*x + phase6) * np.exp(-(gamma6+gamma0[5])*x)   +\
						A7	   *np.sin((omega7	+omega0[6])*x + phase7) * np.exp(-(gamma7+gamma0[6])*x)   +\
						A8	   *np.sin((omega8	+omega0[7])*x + phase8) * np.exp(-(gamma8+gamma0[7])*x)   +\
						A9	   *np.sin((omega9	+omega0[8])*x + phase9) * np.exp(-(gamma9+gamma0[8])*x)   +\
						A10    *np.sin((omega10 +omega0[9])*x + phase10) * np.exp(-(gamma10+gamma0[9])*x) +\
						c
		
		
		
		if nfree==1:
	
			def dsin_model(x,  A1, omega1, gamma1, phase1,\
							   A2, omega2, gamma2, phase2,\
							   A3, omega3, gamma3, phase3,\
							   A4, omega4, gamma4, phase4,\
							   A5, omega5, gamma5, phase5,\
							   A6, omega6, gamma6, phase6,\
							   A7, omega7, gamma7, phase7,\
							   A8, omega8, gamma8, phase8,\
							   A9, omega9, gamma9, phase9,\
							   A10, omega10, gamma10, phase10,\
							   A_free1, omega_free1, gamma_free1, phase_free1, \
							   c):
				return	A1	   *np.sin((omega1	+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x)   +\
						A2	   *np.sin((omega2	+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)   +\
						A3	   *np.sin((omega3	+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)   +\
						A4	   *np.sin((omega4	+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x)   +\
						A5	   *np.sin((omega5	+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x)   +\
						A6	   *np.sin((omega6	+omega0[5])*x + phase6) * np.exp(-(gamma6+gamma0[5])*x)   +\
						A7	   *np.sin((omega7	+omega0[6])*x + phase7) * np.exp(-(gamma7+gamma0[6])*x)   +\
						A8	   *np.sin((omega8	+omega0[7])*x + phase8) * np.exp(-(gamma8+gamma0[7])*x)   +\
						A9	   *np.sin((omega9	+omega0[8])*x + phase9) * np.exp(-(gamma9+gamma0[8])*x)   +\
						A10    *np.sin((omega10 +omega0[9])*x + phase10) * np.exp(-(gamma10+gamma0[9])*x) +\
						A_free1*np.sin((omega_free1)*x + phase_free1) * np.exp(-(gamma_free1)*x)   +\
						c
		
		if nfree==2:
	
			def dsin_model(x,  A1, omega1, gamma1, phase1,\
							   A2, omega2, gamma2, phase2,\
							   A3, omega3, gamma3, phase3,\
							   A4, omega4, gamma4, phase4,\
							   A5, omega5, gamma5, phase5,\
							   A6, omega6, gamma6, phase6,\
							   A7, omega7, gamma7, phase7,\
							   A8, omega8, gamma8, phase8,\
							   A9, omega9, gamma9, phase9,\
							   A10, omega10, gamma10, phase10,\
							   A_free1, omega_free1, gamma_free1, phase_free1, \
							   A_free2, omega_free2, gamma_free2, phase_free2, \
							   c):
				return	A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x)   +\
						A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)   +\
						A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)   +\
						A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x)   +\
						A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x)   +\
						A6*np.sin((omega6+omega0[5])*x + phase6) * np.exp(-(gamma6+gamma0[5])*x)   +\
						A7*np.sin((omega7+omega0[6])*x + phase7) * np.exp(-(gamma7+gamma0[6])*x)   +\
						A8*np.sin((omega8+omega0[7])*x + phase8) * np.exp(-(gamma8+gamma0[7])*x)   +\
						A9*np.sin((omega9+omega0[8])*x + phase9) * np.exp(-(gamma9+gamma0[8])*x)   +\
						A10*np.sin((omega10+omega0[9])*x + phase10) * np.exp(-(gamma10+gamma0[9])*x) +\
						A_free1*np.sin((omega_free1)*x + phase_free1) * np.exp(-(gamma_free1)*x)   +\
						A_free2*np.sin((omega_free2)*x + phase_free2) * np.exp(-(gamma_free2)*x)   +\
						c
	
	if N==3:
		
		
		if nfree==0:
	
			def dsin_model(x,  A1, omega1, gamma1, phase1,\
							   A2, omega2, gamma2, phase2,\
							   A3, omega3, gamma3, phase3,\
							   A4, omega4, gamma4, phase4,\
							   A5, omega5, gamma5, phase5,\
							   A6, omega6, gamma6, phase6,\
							   A7, omega7, gamma7, phase7,\
							   A8, omega8, gamma8, phase8,\
							   A9, omega9, gamma9, phase9,\
							   A10, omega10, gamma10, phase10,\
							   A11, omega11, gamma11, phase11,\
							   A12, omega12, gamma12, phase12,\
							   A13, omega13, gamma13, phase13,\
							   A14, omega14, gamma14, phase14,\
							   A15, omega15, gamma15, phase15,\
							   c):
				return	A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x)   +\
						A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)   +\
						A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)   +\
						A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x)   +\
						A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x)   +\
						A6*np.sin((omega6+omega0[5])*x + phase6) * np.exp(-(gamma6+gamma0[5])*x)   +\
						A7*np.sin((omega7+omega0[6])*x + phase7) * np.exp(-(gamma7+gamma0[6])*x)   +\
						A8*np.sin((omega8+omega0[7])*x + phase8) * np.exp(-(gamma8+gamma0[7])*x)   +\
						A9*np.sin((omega9+omega0[8])*x + phase9) * np.exp(-(gamma9+gamma0[8])*x)   +\
						A10*np.sin((omega10+omega0[9])*x + phase10) * np.exp(-(gamma10+gamma0[9])*x) +\
						A11*np.sin((omega11+omega0[10])*x + phase11) * np.exp(-(gamma11+gamma0[10])*x)	 +\
						A12*np.sin((omega12+omega0[11])*x + phase12) * np.exp(-(gamma12+gamma0[11])*x)	 +\
						A13*np.sin((omega13+omega0[12])*x + phase13) * np.exp(-(gamma13+gamma0[12])*x)	 +\
						A14*np.sin((omega14+omega0[13])*x + phase14) * np.exp(-(gamma14+gamma0[13])*x)	 +\
						A15*np.sin((omega15+omega0[14])*x + phase15) * np.exp(-(gamma15+gamma0[14])*x) +\
						c
			
			
		
		if nfree==1:
	
			def dsin_model(x,  A1, omega1, gamma1, phase1,\
							   A2, omega2, gamma2, phase2,\
							   A3, omega3, gamma3, phase3,\
							   A4, omega4, gamma4, phase4,\
							   A5, omega5, gamma5, phase5,\
							   A6, omega6, gamma6, phase6,\
							   A7, omega7, gamma7, phase7,\
							   A8, omega8, gamma8, phase8,\
							   A9, omega9, gamma9, phase9,\
							   A10, omega10, gamma10, phase10,\
							   A11, omega11, gamma11, phase11,\
							   A12, omega12, gamma12, phase12,\
							   A13, omega13, gamma13, phase13,\
							   A14, omega14, gamma14, phase14,\
							   A15, omega15, gamma15, phase15,\
							   A_free1, omega_free1, gamma_free1, phase_free1, \
							   c):
				return	A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x)   +\
						A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)   +\
						A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)   +\
						A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x)   +\
						A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x)   +\
						A6*np.sin((omega6+omega0[5])*x + phase6) * np.exp(-(gamma6+gamma0[5])*x)   +\
						A7*np.sin((omega7+omega0[6])*x + phase7) * np.exp(-(gamma7+gamma0[6])*x)   +\
						A8*np.sin((omega8+omega0[7])*x + phase8) * np.exp(-(gamma8+gamma0[7])*x)   +\
						A9*np.sin((omega9+omega0[8])*x + phase9) * np.exp(-(gamma9+gamma0[8])*x)   +\
						A10*np.sin((omega10+omega0[9])*x + phase10) * np.exp(-(gamma10+gamma0[9])*x) +\
						A11*np.sin((omega11+omega0[10])*x + phase11) * np.exp(-(gamma11+gamma0[10])*x)	 +\
						A12*np.sin((omega12+omega0[11])*x + phase12) * np.exp(-(gamma12+gamma0[11])*x)	 +\
						A13*np.sin((omega13+omega0[12])*x + phase13) * np.exp(-(gamma13+gamma0[12])*x)	 +\
						A14*np.sin((omega14+omega0[13])*x + phase14) * np.exp(-(gamma14+gamma0[13])*x)	 +\
						A15*np.sin((omega15+omega0[14])*x + phase15) * np.exp(-(gamma15+gamma0[14])*x) +\
						A_free1*np.sin((omega_free1)*x + phase_free1) * np.exp(-(gamma_free1)*x)   +\
						c
		
		if nfree==2:
	
			def dsin_model(x,  A1, omega1, gamma1, phase1,\
							   A2, omega2, gamma2, phase2,\
							   A3, omega3, gamma3, phase3,\
							   A4, omega4, gamma4, phase4,\
							   A5, omega5, gamma5, phase5,\
							   A6, omega6, gamma6, phase6,\
							   A7, omega7, gamma7, phase7,\
							   A8, omega8, gamma8, phase8,\
							   A9, omega9, gamma9, phase9,\
							   A10, omega10, gamma10, phase10,\
							   A11, omega11, gamma11, phase11,\
							   A12, omega12, gamma12, phase12,\
							   A13, omega13, gamma13, phase13,\
							   A14, omega14, gamma14, phase14,\
							   A15, omega15, gamma15, phase15,\
							   A_free1, omega_free1, gamma_free1, phase_free1, \
							   A_free2, omega_free2, gamma_free2, phase_free2, \
							   c):
				return	A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x)   +\
						A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)   +\
						A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)   +\
						A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x)   +\
						A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x)   +\
						A6*np.sin((omega6+omega0[5])*x + phase6) * np.exp(-(gamma6+gamma0[5])*x)   +\
						A7*np.sin((omega7+omega0[6])*x + phase7) * np.exp(-(gamma7+gamma0[6])*x)   +\
						A8*np.sin((omega8+omega0[7])*x + phase8) * np.exp(-(gamma8+gamma0[7])*x)   +\
						A9*np.sin((omega9+omega0[8])*x + phase9) * np.exp(-(gamma9+gamma0[8])*x)   +\
						A10*np.sin((omega10+omega0[9])*x + phase10) * np.exp(-(gamma10+gamma0[9])*x) +\
						A11*np.sin((omega11+omega0[10])*x + phase11) * np.exp(-(gamma11+gamma0[10])*x)	 +\
						A12*np.sin((omega12+omega0[11])*x + phase12) * np.exp(-(gamma12+gamma0[11])*x)	 +\
						A13*np.sin((omega13+omega0[12])*x + phase13) * np.exp(-(gamma13+gamma0[12])*x)	 +\
						A14*np.sin((omega14+omega0[13])*x + phase14) * np.exp(-(gamma14+gamma0[13])*x)	 +\
						A15*np.sin((omega15+omega0[14])*x + phase15) * np.exp(-(gamma15+gamma0[14])*x) +\
						A_free1*np.sin((omega_free1)*x + phase_free1) * np.exp(-(gamma_free1)*x)   +\
						A_free2*np.sin((omega_free2)*x + phase_free2) * np.exp(-(gamma_free2)*x)   +\
						c
		
		
	return dsin_model















def dsin_gen(s, N, l, m, mass, spin, qnmdatdir, qnm_details_from_nlm_mass_spin):
	
	omega0 = [abs(qnm_details_from_nlm_mass_spin(s, N_index+1, l, m, mass, spin, qnmdatdir)[1]) for N_index in range(N)]
	gamma0 = [abs(qnm_details_from_nlm_mass_spin(s, N_index+1, l, m, mass, spin, qnmdatdir)[2]) for N_index in range(N)]
	
	if N==1:
		def dsin_model(x, A1, omega1, gamma1, phase1, c):
			return A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) + c

	if N==2:
		def dsin_model(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, c):
			return A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) + A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)+ c

	if N==3:
		def dsin_model(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, A3, omega3, gamma3, phase3, c):
			return A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) + A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)+ A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)+ c

	if N==4:
		def dsin_model(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, A3, omega3, gamma3, phase3,  A4, omega4, gamma4, phase4, c):
			return A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) + A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)+ A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)+ A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x)+ c
	if N==5:
		def dsin_model(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, A3, omega3, gamma3, phase3,  A4, omega4, gamma4, phase4, A5, omega5, gamma5, phase5, c):
			return A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) + A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)+ A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)+  A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x)+ A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x) +c

	return dsin_model


def dsin_gen_m(s, N, l, mass, spin, qnmdatdir, qnm_details_from_nlm_mass_spin):
	
	omega0 = [abs(qnm_details_from_nlm_mass_spin(s, N_index+1, l, m_index, mass, spin, qnmdatdir)[1]) for N_index in range(N) for m_index in range(-l, l+1)]
	gamma0 = [abs(qnm_details_from_nlm_mass_spin(s, N_index+1, l, m_index, mass, spin, qnmdatdir)[2]) for N_index in range(N) for m_index in range(-l, l+1)]
	
	if N==1:

		def dsin_model(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, A3, omega3, gamma3, phase3,  A4, omega4, gamma4, phase4, A5, omega5, gamma5, phase5, c):
			return A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) + A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x)+ A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x)+  A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x)+ A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x) +c

	if N==2:
	
		def dsin_model(x,  A1, omega1, gamma1, phase1,\
						   A2, omega2, gamma2, phase2,\
						   A3, omega3, gamma3, phase3,\
						   A4, omega4, gamma4, phase4,\
						   A5, omega5, gamma5, phase5,\
						   A6, omega6, gamma6, phase6,\
						   A7, omega7, gamma7, phase7,\
						   A8, omega8, gamma8, phase8,\
						   A9, omega9, gamma9, phase9,\
						   A10, omega10, gamma10, phase10,\
						   c):
			return	A1*np.sin((omega1+omega0[0])*x + phase1) * np.exp(-(gamma1+gamma0[0])*x) +\
					A2*np.sin((omega2+omega0[1])*x + phase2) * np.exp(-(gamma2+gamma0[1])*x) +\
					A3*np.sin((omega3+omega0[2])*x + phase3) * np.exp(-(gamma3+gamma0[2])*x) +\
					A4*np.sin((omega4+omega0[3])*x + phase4) * np.exp(-(gamma4+gamma0[3])*x) +\
					A5*np.sin((omega5+omega0[4])*x + phase5) * np.exp(-(gamma5+gamma0[4])*x) +\
					A6*np.sin((omega1+omega0[5])*x + phase6) * np.exp(-(gamma1+gamma0[5])*x) +\
					A7*np.sin((omega2+omega0[6])*x + phase7) * np.exp(-(gamma2+gamma0[6])*x) +\
					A8*np.sin((omega3+omega0[7])*x + phase8) * np.exp(-(gamma3+gamma0[7])*x) +\
					A9*np.sin((omega4+omega0[8])*x + phase9) * np.exp(-(gamma4+gamma0[8])*x) +\
					A10*np.sin((omega5+omega0[9])*x + phase10) * np.exp(-(gamma5+gamma0[9])*x) +\
					c

	return dsin_model



def dsin_gen_m_fixed_omega(s, N, l, mass, spin, qnmdatdir, qnm_details_from_nlm_mass_spin):
	
	omega0 = [abs(qnm_details_from_nlm_mass_spin(s, N_index+1, l, m_index, mass, spin, qnmdatdir)[1]) for N_index in range(N) for m_index in range(-l, l+1)]
	gamma0 = [abs(qnm_details_from_nlm_mass_spin(s, N_index+1, l, m_index, mass, spin, qnmdatdir)[2]) for N_index in range(N) for m_index in range(-l, l+1)]
	
	if N==1:

		def dsin_model(x, A1, phase1, A2, phase2, A3, phase3,  A4, phase4, A5, phase5, c):
			return A1*np.sin((omega0[0])*x + phase1) * np.exp(-(gamma0[0])*x) + A2*np.sin((omega0[1])*x + phase2) * np.exp(-(gamma0[1])*x)+ A3*np.sin((omega0[2])*x + phase3) * np.exp(-(gamma0[2])*x)+  A4*np.sin((omega0[3])*x + phase4) * np.exp(-(gamma0[3])*x)+ A5*np.sin((omega0[4])*x + phase5) * np.exp(-(gamma0[4])*x) +c

	if N==2:
	
		def dsin_model(x,  A1, phase1,\
						   A2, phase2,\
						   A3, phase3,\
						   A4, phase4,\
						   A5, phase5,\
						   A6, phase6,\
						   A7, phase7,\
						   A8, phase8,\
						   A9, phase9,\
						   A10, phase10,\
						   c):
			return	A1*np.sin((omega0[0])*x + phase1) * np.exp(-(gamma0[0])*x) +\
					A2*np.sin((omega0[1])*x + phase2) * np.exp(-(gamma0[1])*x) +\
					A3*np.sin((omega0[2])*x + phase3) * np.exp(-(gamma0[2])*x) +\
					A4*np.sin((omega0[3])*x + phase4) * np.exp(-(gamma0[3])*x) +\
					A5*np.sin((omega0[4])*x + phase5) * np.exp(-(gamma0[4])*x) +\
					A6*np.sin((omega0[5])*x + phase6) * np.exp(-(gamma0[5])*x) +\
					A7*np.sin((omega0[6])*x + phase7) * np.exp(-(gamma0[6])*x) +\
					A8*np.sin((omega0[7])*x + phase8) * np.exp(-(gamma0[7])*x) +\
					A9*np.sin((omega0[8])*x + phase9) * np.exp(-(gamma0[8])*x) +\
					A10*np.sin((omega0[9])*x + phase10) * np.exp(-(gamma0[9])*x) +\
					c

	if N==3:
	
		def dsin_model(x,  A1, phase1,\
						   A2, phase2,\
						   A3, phase3,\
						   A4, phase4,\
						   A5, phase5,\
						   A6, phase6,\
						   A7, phase7,\
						   A8, phase8,\
						   A9, phase9,\
						   A10, phase10,\
						   A11, phase11,\
						   A12, phase12,\
						   A13, phase13,\
						   A14, phase14,\
						   A15, phase15,\
						   c):
			return	A1*np.sin((omega0[0])*x + phase1) * np.exp(-(gamma0[0])*x) +\
					A2*np.sin((omega0[1])*x + phase2) * np.exp(-(gamma0[1])*x) +\
					A3*np.sin((omega0[2])*x + phase3) * np.exp(-(gamma0[2])*x) +\
					A4*np.sin((omega0[3])*x + phase4) * np.exp(-(gamma0[3])*x) +\
					A5*np.sin((omega0[4])*x + phase5) * np.exp(-(gamma0[4])*x) +\
					A6*np.sin((omega0[5])*x + phase6) * np.exp(-(gamma0[5])*x) +\
					A7*np.sin((omega0[6])*x + phase7) * np.exp(-(gamma0[6])*x) +\
					A8*np.sin((omega0[7])*x + phase8) * np.exp(-(gamma0[7])*x) +\
					A9*np.sin((omega0[8])*x + phase9) * np.exp(-(gamma0[8])*x) +\
					A10*np.sin((omega0[9])*x + phase10) * np.exp(-(gamma0[9])*x) +\
					A11*np.sin((omega0[10])*x + phase11) * np.exp(-(gamma0[10])*x) +\
					A12*np.sin((omega0[11])*x + phase12) * np.exp(-(gamma0[11])*x) +\
					A13*np.sin((omega0[12])*x + phase13) * np.exp(-(gamma0[12])*x) +\
					A14*np.sin((omega0[13])*x + phase14) * np.exp(-(gamma0[13])*x) +\
					A15*np.sin((omega0[14])*x + phase15) * np.exp(-(gamma0[14])*x) +\
					c
		
		if N==4:
	
			def dsin_model(x,  A1, phase1,\
							   A2, phase2,\
							   A3, phase3,\
							   A4, phase4,\
							   A5, phase5,\
							   A6, phase6,\
							   A7, phase7,\
							   A8, phase8,\
							   A9, phase9,\
							   A10, phase10,\
							   A11, phase11,\
							   A12, phase12,\
							   A13, phase13,\
							   A14, phase14,\
							   A15, phase15,\
							   A16, phase16,\
							   A17, phase17,\
							   A18, phase18,\
							   A19, phase19,\
							   A20, phase20,\
							   c):
				return	A1*np.sin((omega0[0])*x + phase1) * np.exp(-(gamma0[0])*x) +\
						A2*np.sin((omega0[1])*x + phase2) * np.exp(-(gamma0[1])*x) +\
						A3*np.sin((omega0[2])*x + phase3) * np.exp(-(gamma0[2])*x) +\
						A4*np.sin((omega0[3])*x + phase4) * np.exp(-(gamma0[3])*x) +\
						A5*np.sin((omega0[4])*x + phase5) * np.exp(-(gamma0[4])*x) +\
						A6*np.sin((omega0[5])*x + phase6) * np.exp(-(gamma0[5])*x) +\
						A7*np.sin((omega0[6])*x + phase7) * np.exp(-(gamma0[6])*x) +\
						A8*np.sin((omega0[7])*x + phase8) * np.exp(-(gamma0[7])*x) +\
						A9*np.sin((omega0[8])*x + phase9) * np.exp(-(gamma0[8])*x) +\
						A10*np.sin((omega0[9])*x + phase10) * np.exp(-(gamma0[9])*x) +\
						A11*np.sin((omega0[10])*x + phase11) * np.exp(-(gamma0[10])*x) +\
						A12*np.sin((omega0[11])*x + phase12) * np.exp(-(gamma0[11])*x) +\
						A13*np.sin((omega0[12])*x + phase13) * np.exp(-(gamma0[12])*x) +\
						A14*np.sin((omega0[13])*x + phase14) * np.exp(-(gamma0[13])*x) +\
						A15*np.sin((omega0[14])*x + phase15) * np.exp(-(gamma0[14])*x) +\
						A16*np.sin((omega0[15])*x + phase16) * np.exp(-(gamma0[15])*x) +\
						A17*np.sin((omega0[16])*x + phase17) * np.exp(-(gamma0[16])*x) +\
						A18*np.sin((omega0[17])*x + phase18) * np.exp(-(gamma0[17])*x) +\
						A19*np.sin((omega0[18])*x + phase19) * np.exp(-(gamma0[18])*x) +\
						A20*np.sin((omega0[19])*x + phase20) * np.exp(-(gamma0[19])*x) +\
						c
		
		
		if N==5:
	
			def dsin_model(x,  A1, phase1,\
							   A2, phase2,\
							   A3, phase3,\
							   A4, phase4,\
							   A5, phase5,\
							   A6, phase6,\
							   A7, phase7,\
							   A8, phase8,\
							   A9, phase9,\
							   A10, phase10,\
							   A11, phase11,\
							   A12, phase12,\
							   A13, phase13,\
							   A14, phase14,\
							   A15, phase15,\
							   A16, phase16,\
							   A17, phase17,\
							   A18, phase18,\
							   A19, phase19,\
							   A20, phase20,\
							   A21, phase21,\
							   A22, phase22,\
							   A23, phase23,\
							   A24, phase24,\
							   A25, phase25,\
							   c):
				return	A1*np.sin((omega0[0])*x + phase1) * np.exp(-(gamma0[0])*x) +\
						A2*np.sin((omega0[1])*x + phase2) * np.exp(-(gamma0[1])*x) +\
						A3*np.sin((omega0[2])*x + phase3) * np.exp(-(gamma0[2])*x) +\
						A4*np.sin((omega0[3])*x + phase4) * np.exp(-(gamma0[3])*x) +\
						A5*np.sin((omega0[4])*x + phase5) * np.exp(-(gamma0[4])*x) +\
						A6*np.sin((omega0[5])*x + phase6) * np.exp(-(gamma0[5])*x) +\
						A7*np.sin((omega0[6])*x + phase7) * np.exp(-(gamma0[6])*x) +\
						A8*np.sin((omega0[7])*x + phase8) * np.exp(-(gamma0[7])*x) +\
						A9*np.sin((omega0[8])*x + phase9) * np.exp(-(gamma0[8])*x) +\
						A10*np.sin((omega0[9])*x + phase10) * np.exp(-(gamma0[9])*x) +\
						A11*np.sin((omega0[10])*x + phase11) * np.exp(-(gamma0[10])*x) +\
						A12*np.sin((omega0[11])*x + phase12) * np.exp(-(gamma0[11])*x) +\
						A13*np.sin((omega0[12])*x + phase13) * np.exp(-(gamma0[12])*x) +\
						A14*np.sin((omega0[13])*x + phase14) * np.exp(-(gamma0[13])*x) +\
						A15*np.sin((omega0[14])*x + phase15) * np.exp(-(gamma0[14])*x) +\
						A16*np.sin((omega0[15])*x + phase16) * np.exp(-(gamma0[15])*x) +\
						A17*np.sin((omega0[16])*x + phase17) * np.exp(-(gamma0[16])*x) +\
						A18*np.sin((omega0[17])*x + phase18) * np.exp(-(gamma0[17])*x) +\
						A19*np.sin((omega0[18])*x + phase19) * np.exp(-(gamma0[18])*x) +\
						A20*np.sin((omega0[19])*x + phase20) * np.exp(-(gamma0[19])*x) +\
						A21*np.sin((omega0[20])*x + phase21) * np.exp(-(gamma0[20])*x) +\
						A22*np.sin((omega0[21])*x + phase22) * np.exp(-(gamma0[21])*x) +\
						A23*np.sin((omega0[22])*x + phase23) * np.exp(-(gamma0[22])*x) +\
						A24*np.sin((omega0[23])*x + phase24) * np.exp(-(gamma0[23])*x) +\
						A25*np.sin((omega0[24])*x + phase25) * np.exp(-(gamma0[24])*x) +\
						c
		
		
		if N==6:
	
			def dsin_model(x,  A1, phase1,\
							   A2, phase2,\
							   A3, phase3,\
							   A4, phase4,\
							   A5, phase5,\
							   A6, phase6,\
							   A7, phase7,\
							   A8, phase8,\
							   A9, phase9,\
							   A10, phase10,\
							   A11, phase11,\
							   A12, phase12,\
							   A13, phase13,\
							   A14, phase14,\
							   A15, phase15,\
							   A16, phase16,\
							   A17, phase17,\
							   A18, phase18,\
							   A19, phase19,\
							   A20, phase20,\
							   A21, phase21,\
							   A22, phase22,\
							   A23, phase23,\
							   A24, phase24,\
							   A25, phase25,\
							   A26, phase26,\
							   A27, phase27,\
							   A28, phase28,\
							   A29, phase29,\
							   A30, phase30,\
							   c):
				return	A1*np.sin((omega0[0])*x + phase1) * np.exp(-(gamma0[0])*x) +\
						A2*np.sin((omega0[1])*x + phase2) * np.exp(-(gamma0[1])*x) +\
						A3*np.sin((omega0[2])*x + phase3) * np.exp(-(gamma0[2])*x) +\
						A4*np.sin((omega0[3])*x + phase4) * np.exp(-(gamma0[3])*x) +\
						A5*np.sin((omega0[4])*x + phase5) * np.exp(-(gamma0[4])*x) +\
						A6*np.sin((omega0[5])*x + phase6) * np.exp(-(gamma0[5])*x) +\
						A7*np.sin((omega0[6])*x + phase7) * np.exp(-(gamma0[6])*x) +\
						A8*np.sin((omega0[7])*x + phase8) * np.exp(-(gamma0[7])*x) +\
						A9*np.sin((omega0[8])*x + phase9) * np.exp(-(gamma0[8])*x) +\
						A10*np.sin((omega0[9])*x + phase10) * np.exp(-(gamma0[9])*x) +\
						A11*np.sin((omega0[10])*x + phase11) * np.exp(-(gamma0[10])*x) +\
						A12*np.sin((omega0[11])*x + phase12) * np.exp(-(gamma0[11])*x) +\
						A13*np.sin((omega0[12])*x + phase13) * np.exp(-(gamma0[12])*x) +\
						A14*np.sin((omega0[13])*x + phase14) * np.exp(-(gamma0[13])*x) +\
						A15*np.sin((omega0[14])*x + phase15) * np.exp(-(gamma0[14])*x) +\
						A16*np.sin((omega0[15])*x + phase16) * np.exp(-(gamma0[15])*x) +\
						A17*np.sin((omega0[16])*x + phase17) * np.exp(-(gamma0[16])*x) +\
						A18*np.sin((omega0[17])*x + phase18) * np.exp(-(gamma0[17])*x) +\
						A19*np.sin((omega0[18])*x + phase19) * np.exp(-(gamma0[18])*x) +\
						A20*np.sin((omega0[19])*x + phase20) * np.exp(-(gamma0[19])*x) +\
						A21*np.sin((omega0[20])*x + phase21) * np.exp(-(gamma0[20])*x) +\
						A22*np.sin((omega0[21])*x + phase22) * np.exp(-(gamma0[21])*x) +\
						A23*np.sin((omega0[22])*x + phase23) * np.exp(-(gamma0[22])*x) +\
						A24*np.sin((omega0[23])*x + phase24) * np.exp(-(gamma0[23])*x) +\
						A25*np.sin((omega0[24])*x + phase25) * np.exp(-(gamma0[24])*x) +\
						A26*np.sin((omega0[25])*x + phase26) * np.exp(-(gamma0[25])*x) +\
						A27*np.sin((omega0[26])*x + phase27) * np.exp(-(gamma0[26])*x) +\
						A28*np.sin((omega0[27])*x + phase28) * np.exp(-(gamma0[27])*x) +\
						A29*np.sin((omega0[28])*x + phase29) * np.exp(-(gamma0[28])*x) +\
						A30*np.sin((omega0[29])*x + phase30) * np.exp(-(gamma0[29])*x) +\
					c
		
		
		
		return dsin_model


def dsin_generator(n):
	parnames = make_par_names(n)
	
	for par_set_index in range(n):
		dsin_set 
																  
def Ndsin(x, A, omega_re, omega_im, phase, c):
	#print(len(A), len(omega_re), len(omega_im), len(phase))
	N = len(A)
	fun=c
	for index in range(N):
		#print(index)
		fun+=dsin(x, A[index], omega_re[index], omega_im[index], phase[index], 0)
	return fun

def wrapper_fit_func(x, N, *args):
	A, omega_re, omega_im, phase, c = list(args[0][:N]), list(args[0][N:2*N]), list(args[0][2*N:3*N]), list(args[0][2*N:3*N]), args[0][-1]
	return Ndsin(x, A, omega_re, omega_im, phase, c)

def dsin(x, A1, omega1, gamma1, phase1, c):
	#print(type(x), type(A1), type(omega1), type(gamma1), type(phase1), type(c))
	#print(A1)
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x) + c


def dsin2(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x) + A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x)+ c


def dsin3(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, A3, omega3, gamma3, phase3, c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x) + A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x)+ A3*np.sin(omega3*x + phase3) * np.exp(-gamma3*x)+ c


def dsin4(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, A3, omega3, gamma3, phase3,  A4, omega4, gamma4, phase4, c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x) + A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x)+ A3*np.sin(omega3*x + phase3) * np.exp(-gamma3*x)+ A4*np.sin(omega4*x + phase4) * np.exp(-gamma4*x)+ c

def dsin5(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, A3, omega3, gamma3, phase3,  A4, omega4, gamma4, phase4, A5, omega5, gamma5, phase5, c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x) + A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x)+ A3*np.sin(omega3*x + phase3) * np.exp(-gamma3*x)+ A4*np.sin(omega4*x + phase4) * np.exp(-gamma4*x)+ A5*np.sin(omega5*x + phase5) * np.exp(-gamma5*x) +c


def dsin6(x, A1, omega1, gamma1, phase1, A2, omega2, gamma2, phase2, A3, omega3, gamma3, phase3,  A4, omega4, gamma4, phase4, A5, omega5, gamma5, phase5, A6, omega6, gamma6, phase6, c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x) + A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x)+ A3*np.sin(omega3*x + phase3) * np.exp(-gamma3*x)+ A4*np.sin(omega4*x + phase4) * np.exp(-gamma4*x) \
		   + A5*np.sin(omega5*x + phase5) * np.exp(-gamma5*x)+ A6*np.sin(omega6*x + phase6) * np.exp(-gamma6*x) +c

def dsin7(x,  A1, omega1, gamma1, phase1,  \
			  A2, omega2, gamma2, phase2,  \
			  A3, omega3, gamma3, phase3,  \
			  A4, omega4, gamma4, phase4,  \
			  A5, omega5, gamma5, phase5,  \
			  A6, omega6, gamma6, phase6,  \
			  A7, omega7, gamma7, phase7,  \
			  c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x)\
			+ A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x) \
			+ A3*np.sin(omega3*x + phase3) * np.exp(-gamma3*x) \
			+ A4*np.sin(omega4*x + phase4) * np.exp(-gamma4*x) \
			+ A5*np.sin(omega5*x + phase5) * np.exp(-gamma5*x) \
			+ A6*np.sin(omega6*x + phase6) * np.exp(-gamma6*x) \
			+ A7*np.sin(omega7*x + phase7) * np.exp(-gamma7*x) \
			+c

def dsin8(x,  A1, omega1, gamma1, phase1,  \
			  A2, omega2, gamma2, phase2,  \
			  A3, omega3, gamma3, phase3,  \
			  A4, omega4, gamma4, phase4,  \
			  A5, omega5, gamma5, phase5,  \
			  A6, omega6, gamma6, phase6,  \
			  A7, omega7, gamma7, phase7,  \
			  A8, omega8, gamma8, phase8,  \
			  c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x)\
			+ A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x) \
			+ A3*np.sin(omega3*x + phase3) * np.exp(-gamma3*x) \
			+ A4*np.sin(omega4*x + phase4) * np.exp(-gamma4*x) \
			+ A5*np.sin(omega5*x + phase5) * np.exp(-gamma5*x) \
			+ A6*np.sin(omega6*x + phase6) * np.exp(-gamma6*x) \
			+ A7*np.sin(omega7*x + phase7) * np.exp(-gamma7*x) \
			+ A8*np.sin(omega8*x + phase8) * np.exp(-gamma8*x) \
			+c

def dsin9(x,  A1, omega1, gamma1, phase1,  \
			  A2, omega2, gamma2, phase2,  \
			  A3, omega3, gamma3, phase3,  \
			  A4, omega4, gamma4, phase4,  \
			  A5, omega5, gamma5, phase5,  \
			  A6, omega6, gamma6, phase6,  \
			  A7, omega7, gamma7, phase7,  \
			  A8, omega8, gamma8, phase8,  \
			  A9, omega9, gamma9, phase9,  \
			  c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x)\
			+ A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x) \
			+ A3*np.sin(omega3*x + phase3) * np.exp(-gamma3*x) \
			+ A4*np.sin(omega4*x + phase4) * np.exp(-gamma4*x) \
			+ A5*np.sin(omega5*x + phase5) * np.exp(-gamma5*x) \
			+ A6*np.sin(omega6*x + phase6) * np.exp(-gamma6*x) \
			+ A7*np.sin(omega7*x + phase7) * np.exp(-gamma7*x) \
			+ A8*np.sin(omega8*x + phase8) * np.exp(-gamma8*x) \
			+ A9*np.sin(omega9*x + phase9) * np.exp(-gamma9*x) \
			+c

def dsin10(x,  A1, omega1, gamma1, phase1,	\
			  A2, omega2, gamma2, phase2,  \
			  A3, omega3, gamma3, phase3,  \
			  A4, omega4, gamma4, phase4,  \
			  A5, omega5, gamma5, phase5,  \
			  A6, omega6, gamma6, phase6,  \
			  A7, omega7, gamma7, phase7,  \
			  A8, omega8, gamma8, phase8,  \
			  A9, omega9, gamma9, phase9,  \
			  A10, omega10, gamma10, phase10,  \
			  c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x)\
			+ A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x) \
			+ A3*np.sin(omega3*x + phase3) * np.exp(-gamma3*x) \
			+ A4*np.sin(omega4*x + phase4) * np.exp(-gamma4*x) \
			+ A5*np.sin(omega5*x + phase5) * np.exp(-gamma5*x) \
			+ A6*np.sin(omega6*x + phase6) * np.exp(-gamma6*x) \
			+ A7*np.sin(omega7*x + phase7) * np.exp(-gamma7*x) \
			+ A8*np.sin(omega8*x + phase8) * np.exp(-gamma8*x) \
			+ A9*np.sin(omega9*x + phase9) * np.exp(-gamma9*x) \
			+ A10*np.sin(omega10*x + phase10) * np.exp(-gamma10*x) \
			+c

def dsin15(x,  A1, omega1, gamma1, phase1,	\
			  A2, omega2, gamma2, phase2,  \
			  A3, omega3, gamma3, phase3,  \
			  A4, omega4, gamma4, phase4,  \
			  A5, omega5, gamma5, phase5,  \
			  A6, omega6, gamma6, phase6,  \
			  A7, omega7, gamma7, phase7,  \
			  A8, omega8, gamma8, phase8,  \
			  A9, omega9, gamma9, phase9,  \
			  A10, omega10, gamma10, phase10,  \
			  A11, omega11, gamma11, phase11,  \
			  A12, omega12, gamma12, phase12,  \
			  A13, omega13, gamma13, phase13,  \
			  A14, omega14, gamma14, phase14,  \
			  A15, omega15, gamma15, phase15,  \
			  c):
	return A1*np.sin(omega1*x + phase1) * np.exp(-gamma1*x)\
			+ A2*np.sin(omega2*x + phase2) * np.exp(-gamma2*x) \
			+ A3*np.sin(omega3*x + phase3) * np.exp(-gamma3*x) \
			+ A4*np.sin(omega4*x + phase4) * np.exp(-gamma4*x) \
			+ A5*np.sin(omega5*x + phase5) * np.exp(-gamma5*x) \
			+ A6*np.sin(omega6*x + phase6) * np.exp(-gamma6*x) \
			+ A7*np.sin(omega7*x + phase7) * np.exp(-gamma7*x) \
			+ A8*np.sin(omega8*x + phase8) * np.exp(-gamma8*x) \
			+ A9*np.sin(omega9*x + phase9) * np.exp(-gamma9*x) \
			+ A10*np.sin(omega10*x + phase10) * np.exp(-gamma10*x) \
			+ A11*np.sin(omega11*x + phase11) * np.exp(-gamma11*x) \
			+ A12*np.sin(omega12*x + phase12) * np.exp(-gamma12*x) \
			+ A13*np.sin(omega13*x + phase13) * np.exp(-gamma13*x) \
			+ A14*np.sin(omega14*x + phase14) * np.exp(-gamma14*x) \
			+ A15*np.sin(omega15*x + phase15) * np.exp(-gamma15*x) \
			+c



def make_par_names(n):
	names = []
	
	for index in range(1, n+1):
		names.append('A{}'.format(index))
		names.append('omega{}'.format(index))
		names.append('gamma{}'.format(index))
		names.append('phase{}'.format(index))
	names.append('c')
	return names

def make_par_names_d(n):
	names = []
	
	for index in range(1, n+1):
		names.append('A{}'.format(index))
		names.append('gamma{}'.format(index))
	names.append('c')
	return names

def make_par_names_with_free(n, nfree):
	names = []
	
	for index in range(1, n+1):
		names.append('A{}'.format(index))
		names.append('omega{}'.format(index))
		names.append('gamma{}'.format(index))
		names.append('phase{}'.format(index))
	names.append('c')
	
	for index in range(1, nfree+1):
		names.append('A_free{}'.format(index))
		names.append('omega_free{}'.format(index))
		names.append('gamma_free{}'.format(index))
		names.append('phase_free{}'.format(index))
	names.append('c')
	return names

def make_par_names_2(n):
	names = []
	
	for index in range(1, n+1):
		names.append('A{}'.format(index))
		#names.append('omega{}'.format(index))
		#names.append('gamma{}'.format(index))
		names.append('phase{}'.format(index))
	names.append('c')
	return names


def make_dict(pars):
	n = int(len(pars)/4)
	#print(n)
	par_names = make_par_names(n)
	c = pars[-1]
	pars_resized_array = np.resize(pars, [n, 4])
	pars_listof_dicts = []
	
	for arr_index in range(n):
		#print(arr_index)
	#for item in par_resized_array:
		item = pars_resized_array[arr_index]
		#print(pars[arr_index])
		one_pars_dict = {}
		for index in range(4):
			one_pars_dict.update({par_names[4*arr_index+index]: item[index]})
	
		pars_listof_dicts.append(one_pars_dict)
	pars_listof_dicts.append({'c' : c})
	
	
	return pars_listof_dicts
def make_dict_d(pars):
	num = 2
	n = int(len(pars)/num)
	#print(n)
	par_names = make_par_names_d(n)
	c = pars[-1]
	pars_resized_array = np.resize(pars, [n, num])
	pars_listof_dicts = []
	
	for arr_index in range(n):
		#print(arr_index)
	#for item in par_resized_array:
		item = pars_resized_array[arr_index]
		#print(pars[arr_index])
		one_pars_dict = {}
		for index in range(num):
			one_pars_dict.update({par_names[num*arr_index+index]: item[index]})
	
		pars_listof_dicts.append(one_pars_dict)
	pars_listof_dicts.append({'c' : c})
	
	
	return pars_listof_dicts


def make_dict_with_free(pars, nfree):
	n = int(len(pars)/4) - nfree
	
	#print(n)
	par_names = make_par_names_with_free(n, nfree)
	c = pars[-1]
	
	pars_resized_array = np.resize(pars, [n+nfree, 4])
	pars_listof_dicts = []
	
	for arr_index in range(n+nfree):
		#print(arr_index)
	#for item in par_resized_array:
		item = pars_resized_array[arr_index]
		#print(pars[arr_index])
		one_pars_dict = {}
		for index in range(4):
			one_pars_dict.update({par_names[4*arr_index+index]: item[index]})
	
		pars_listof_dicts.append(one_pars_dict)
	pars_listof_dicts.append({'c' : c})
	
	
	return pars_listof_dicts


def dsin_n_gen(n):
	pars = make_par_names(n)
	
	def dsin_n(x, *pars):
		
		num_dsin = n
		
		c  = pars[-1]

		par_array = np.reshape(np.array(pars), [4, num_dsin])

		#pars	  = list(zip(A, omega, gamma, phase))
		val = 0

		for item in par_array:
			par = np.concatenate(item, [c/num_dsin])
			val+=dsin(item)

		return val
def red_chisquared(model, x, y, *pars):
	''' Reduced chi squared computation for the 234 model'''

	# Compute the difference array.
	delta_array = (y-model(x, *pars))

	# DOF
	dof			= len(x) - len(pars)

	# Reduced chi2
	res = np.sum(np.power(delta_array, 2))/dof

	return res
