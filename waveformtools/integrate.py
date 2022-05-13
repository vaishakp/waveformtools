''' Methods to integrate functions '''


##################################################
# Fixed frequency integration
##################################################


def fixed_frequency_integrator(u_time, delta_t, omega0=0, order=1, zero_mode=0):
    ''' Fixed frequency integrator as presented in Reisswig


    Inputs
    -------
    u_time :	1d array
				The input data in time.
	delta_t :	float
				The time stepping.

    omega0 :	float
				The cutoff angular frequency in the integration. Must be lower than the starting angular frequency of the input waveform.
    order :	    int
				The number of times to integrate the integrand in time.
    zero_mode :	float
				The zero mode amplitude of the FFT required.

    Returns
    -------

    u_time :	1d array
				The input waveform in time-space, integrated in frequency space using FFI.

	u_integ_freq :	1d array
					The integrated u samples in Fourier space.

    '''

	if not utilde_conven:
        # Compute the FFT of data
        from numpy.fft import ifft
        from waveformtools.transforms import find_fft, unset_fft_conven
        from waveformtools.waveformtools import taper
        udata_x = taper(udata_x, delta_t = delta_x)
        x_axis = udata_x.sample_times
        udata_x = np.array(udata_x)
        freq_axis, utilde_conven        = find_fft(udata_x, delta_x)


        # Find the length of the input data.
        Nlen                    = len(udata_x)

    else:
        Nlen = len(utilde_conven)


    # Find the location of the zero index.
    if Nlen%2==0:
        zero_index = int(Nlen/2)
    else:
        zero_index = int((Nlen+1)/2)

    # Construct the angular frequency axis.
    omega_axis = 2*np.pi*freq_axis


    print('The chosen cutoff angular frequency is', omega0)

	if omega0>0:
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
                if abs(element) < omega0:
                    omega_axis[index] = sign*omega0

    # Set the zero frequency element separately.
	if not zero_mode:
		utilde[zero_index] = 0
	else:
		utilde[zero_index] = zero_mode

    # Integrate in frequency space
    utilde_integ_n      = np.power((-1j/omega_integ), n) * utilde

    # Get the inverse fft
    utilde_integ_n_orig = unset_fft_conven(utilde_integ_n)

    u_time              = ifft(utilde_integ_n_orig)

    return	u_time, utilde_integ_n
