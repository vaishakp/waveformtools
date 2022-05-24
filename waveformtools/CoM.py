''' Centre of mass correction for the waveforms. '''

#################
# Imports
#################



import numpy as np



def X_com_moments(time_axis, Xcom, order):
    ''' Compute the nth order temporal moment of the COM coordinates.

    Parameters
    ----------

    time_axis :     1d array
                    The time axis.
    Xcom :      list
                A list of three 1d arrays, each a 1d array containing the
                time series of the x, y and z co-ordinates in that order.
    order :     int
                The order of the moment.

    Returns
    -------

    moments :   list
                A list containing three real numbers, one each for the moment
                of x, y and z locations.

    '''
    # Initial and final times
    ti = time_axis[0]
    tf = time_axis[-1]

    dT = tf - ti
    # Split the data
    x_all, y_all, z_all = Xcom

    # Interpolate
    from scipy.interpolate import interp1d


    x_all_int_fun = interp1d(time_axis, np.power(time_axis, n) * x_all/dT, kind='quadratic')
    y_all_int_fun = interp1d(time_axis, np.power(time_axis, n) * y_all/dT, kind='quadratic')
    z_all_int_fun = interp1d(time_axis, np.power(time_axis, n) * z_all/dT, kind='quadratic')

    int_funcs = [x_all_int_fun, y_all_int_fun, z_all_int_fun]

    # Integrate
    from scipy.integrate import quad

    moments = {}
    labels = ['x', 'y', 'z']

    count = 0
    # Find the moments
    for item in int_funcs:
        moment, err = quad(item,  ti, tf)
        moments.update({labels[count] : [moment, err]})
        count+=1


    return moments


def compute_com_alpha(time_i, time_f, Xcom_0, Xcom_1):
    ''' Computes the CoM correction alpha parameter: the mean displacement of the system, of the COM correction as defined in Woodford et al. 2019 (Phys. Rev. D 100, 124010).

    Parameters
    ----------

    time_i :    float
                initial time
    time_f :    float
                final time
    Xcom_0 :    list
                A list containing the zeroth order moments of the COM.
    Xcom_1 :    list
                A list containing the first order moments of the COM.

    Returns
    -------

    com_alpha :     list
					The list containig the alpha parameter vector

    '''

    com_alpha = (4*(time_f**2 + time_f*ti + time_i**2)*np.array(Xcom_0) - 6*(time_f + time_i) * np.array(Xcom_1) )/ (time_f - time_i)**2

    return com_alpha


def compute_com_beta(time_i, time_f, Xcom_0, Xcom_1):
    ''' Computes the CoM beta parameter: the mean drift of the system, of the COM correction as defined in Woodford et al. 2019 (Phys. Rev. D 100, 124010).

    Parameters
    ----------

    time_i :     float
                 initial time
    time_f :    float
                final time
    Xcom_0 :    list
                A list containing the zeroth order moments of the COM.
    Xcom_1 :    list
                A list containing the first order moments of the COM.

    Returns
    -------

    com_beta :      list
                The list containig the alpha parameter vector

    '''

    com_beta = (12 * (Xcom_1) - 6 * (time_f + time_i) * Xcom_0 )/ (time_f - time_i)**2

    return com_beta


def compute_transl_alpha_modes(time_axis, com_alpha, com_beta):
    ''' Compute the translation scalar :math:`\\alpha` in its spherical harmonic components given the mean motion of the centre of mass.
        These are basically the quantities in Eq. (4-5d) in the reference Woodford et al. 2019.

        Parameters
        ----------

        time_axis :     1d array
                        The 1D array containing the time axis of the simulation.
        alpha :     1d array
                    The 1D array containing the mean co-ordinate displacement of the COM of the system.
        beta :      1d array
                    The 1D array containing the mean co-ordinate velocity of the COM.

        Returns
        -------

        modes :     dict
                    A dictionary of lists, with each sublist containing the SH decomposition of the 'Alpha' supertranslation variable for a particular ell.

        '''

    # Define the total displacement
    delta_t = 0
    delta_x = com_alpha[0] + com_beta[0]*time_axis
    delta_y = com_alpha[1] + com_beta[1]*time_axis
    delta_z = com_alpha[2] + com_beta[2]*time_axis


    Alpha_00  = np.sqrt(4*np.pi) * delta_t
    Alpha_1m1 = -2 * np.sqrt(2*np.pi/3) * (delta_x + 1j* delta_y)
    Alpha_10  = -np.sqrt(4*np.pi/3) * delta_z
    Alpha_11  = -2 * np.sqrt(2*np.pi/3) * (- delta_x + 1j* delta_y)

    # Combine into one list
    modes     = { 'l0' : [Alpha_00], 'l1' : [Alpha_1m1, Alpha_10, Alpha_11]}

    return modes
