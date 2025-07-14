# pylint: disable=C0301
"""  This module contains a set of tools to handle data
from numerical relativity simulations and gravitational waves.
It consists of a bunch of functions that were frequently used
for handling NR data.

Notes:
1. Some functions here display plots. If you intend to use these on the
cluster and if xterm is not adequately setup, you may have to
either comment the plot codes or use the`Agg` mode for plotting
and save the figures instead of showing by changing the appropriate
lines in the code.
2. These is a module consisting of functions, not completely optimized
for speed. This will happen in future.
3. These functions are not defined in classes as they mostly use and
operate on the objects of pycbc's builtin classes.
4. Any suggestions, comments, critisism invited to vaishak@gmail.com!
"""

import datetime
import os
import pickle

# import statistics
import sys
import traceback
from inspect import getframeinfo, stack

try:
    import vlconf
except Exception as ex:
    print(ex)
    import waveformtools.config as vlconf

import numpy as np

# from numba import njit
# from scipy import signal

try:
    import pycbc
except Exception as excep:
    print(excep)

import scipy

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from termcolor import colored


def message(
    *args,
    message_verbosity=2,
    print_verbosity=vlconf.print_verbosity,
    log_verbosity=vlconf.log_verbosity,
    **kwargs,
):
    """The print function with verbosity levels and logging facility.

    Notes
    -----
    Verbosity choices:

    * message_verbosity: Each message carries with it a verbosity level.
                         More the verbosity less the priority.
                         Default value is 2 i.e. informational.
    * print_verbosity: prints all messages above this level of verbosity.
    * log_verbosity: logs all messages above this level of verbosity.

    Verbosity levels:
    * 0: Errors
    * 1: Warnings
    * 2: Information

    Parameters
    ----------
    ``*args`: non-keyword arguments
              same arguments as to that of the print functions,
    message_verbosity: int
    print_verbosity: int
    log_verbosity: int
    ``**kwargs``: keyword arguments
                  Same as that of the print function.

    Returns
    -------
    1: int
       messages to stdout and logging of messages,
       while the function returns 1.
    """

    # If message verbosity matches the global verbosity level, then print
    if message_verbosity <= print_verbosity:
        print(*args, **kwargs)
    if log_verbosity <= message_verbosity:
        now = str(datetime.datetime.now())
        tstamp = (now[:10] + "_" + now[11:16]).replace(":", "-")
        caller = getframeinfo(stack()[1][0])
        # frameinfo = getframeinfo(currentframe())
        if not os.path.isdir("logs"):
            os.mkdir("logs")

        with open("logs/" + tstamp + ".log", "a") as log_file:
            if message_verbosity == -1:
                for line in traceback.format_stack():
                    log_file.write(line.strip())
            log_file.write("\n")
            log_file.write(
                "{}:{}\t{}".format(caller.filename, caller.lineno, *args)
            )
            log_file.write("\n")
    return 1


def unsort(sorted_array, args_order):
    message("Sorted array", sorted_array, message_verbosity=3)
    message("Args order", args_order, message_verbosity=3)

    original_array = np.zeros(len(sorted_array), dtype=object)

    for index, item in enumerate(args_order):
        message(f"Index {index} item {item}", message_verbosity=4)

        original_array[item] = sorted_array[index]

    return original_array


def stat_mode(a_list):
    """Find the mode of a list

    Parameters
    ----------
    a_list: list
                    The list whose mode is to be found

    Returns
    -------
    list_mode:  float
                            The mode of the list.

    """

    a_list = list(a_list)

    return max(set(a_list), key=a_list.count)


# Data I/O functions


def save_obj(obj, name, obj_dir="./", protocol=pickle.HIGHEST_PROTOCOL):
    """A function to save python objects to disk using pickle.

    Parameters
    ----------
    obj: object
         The python object to be saved.
    name: string
          The filename.
    obj_dir: string
             The path to directory to be saved in. Defaults to PWD
    protocol: int
              The protocol to be used to save. Default is binary.

    Notes
    -----
    Protocols:

    * 0 : Text
    * 5 : Binary

    See the man page of pickle for more details.

    Returns
    -------
    Nothing (other than saving the data to the disk).

    """

    # Create the directory dir if it doesn't exist.
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)

    # Pickle the file to disk.
    with open(obj_dir + name + ".pkl", "wb") as out_file:
        pickle.dump(obj, out_file, protocol)


def load_obj(name, obj_dir="./"):
    """A function to load python objects from the disk using pickle.

    Parameters
    ----------
    name: string
          The filename.
    obj_dir: string
             The path to directory in which file exists.
             Defaults to PWD.

    Returns
    -------
    obj: object
         A python object with the contents of the file.
    """

    # Load the pickled data
    with open(obj_dir + name + ".pkl", "rb") as data_file:
        return pickle.load(data_file)


def removeNans(xdata, ydata):
    """Remove Nans from (xdata,ydata) data pair.
    Removes Nans in xdata and ydata and the
    corresponding y and x entries.

    Parameters
    ----------
    xdata: 1d array
           The x axis of the data.
    ydata: 1d array
           The y axis of the data.

    Returns
    -------
    x_no_nan,y_no_nan:  1d,array
                        The data pair x,y with Nans removed.
    """

    # Find the location of x Nans to be removed.
    nan_locs = np.where(np.isnan(xdata))[0]
    # Remove the xdata and the corresponding y entries.
    xdata = np.delete(xdata, nan_locs)
    ydata = np.delete(ydata, nan_locs)
    # Find the location of the y Nans to be removed.
    nan_locs = np.where(np.isnan(ydata))[0]
    # Remove the y and the correspoinding x entries.
    xdata = np.delete(xdata, nan_locs)
    ydata = np.delete(ydata, nan_locs)
    # Find the location of the inifinities to be removed along x.
    inf_locs = np.where(np.isinf(xdata))[0]
    # Remove the x and the correspoiding y entries.
    xdata = np.delete(xdata, inf_locs)
    ydata = np.delete(ydata, inf_locs)
    # Find the location of the infinities to be removed along y.
    inf_locs = np.where(np.isinf(ydata))[0]
    # Remove the y and the corresponding x locations.
    x_no_nan = np.delete(xdata, inf_locs)
    y_no_nan = np.delete(ydata, inf_locs)

    # Return the reconditioned x and y data.
    return x_no_nan, y_no_nan


def differentiate(data, delta_t=None, TS=False):
    """Differentiate a timeseries in time domain
    using the Simple Euler method

    Parameters
    ----------
    data: 1d array
          a pycbc TimeSeries object
          The time series to be differentiated.
    delta_t: float
             The grid spacing delta_t.
             Supplying delta_t overrides
             the delta_t attribute of TimeSeries.

    Returns
    -------
    ddt_data: pycbc TimeSeries object
              The differentiated 1d data
              as pycbc TimeSeries"""

    if not delta_t:
        try:
            delta_t = data.delta_t
        except BaseException:
            message(
                "Input is not a TimeSeries."
                "Please supply gridspacing as delta_t",
                message_verbosity=0,
            )

    dydx = np.diff((np.array(data))) / delta_t

    if TS is True:
        dydx = pycbc.types.timeseries.TimeSeries(dydx, delta_t)

    return dydx


def integrate_first_order(
    data, t_start=None, t_end=None, delta_t=None, to_taper=False, TS=False
):
    """Integrate a timeseries using first order method.

    Notes
    -----
    Capabilities

    Simple Euler integrator, option to taper the ends.

    Parameters
    ----------
    data :  1d array or a pycbc TimeSeries object
                    The input function to be integrated.
    delta_t :   float
                            The grid spacing delta_t,
    to_taper :  bool
                            True or False. Whether or not
                            to taper the input series.

    Returns
    -------
    int_data :  a pycbc TimeSeries object
                            TimeSeries of the time integrated data"""

    # Check if object is pycbc timeseries.
    # Recover delta_t, t_start, t_end if yes.
    if not delta_t:
        try:
            # Find sampling time_step
            delta_t = data.delta_t
            data_time = data.sample_times
            t_start_dat, t_end_dat = data_time[0], data_time[-1]

        except BaseException:
            message(
                "Input is not a TimeSeries."
                "Please input a pycbc TimeSeries"
                "or supply gridspacing as delta_t",
                message_verbosity=0,
            )
    else:
        t_start_dat = 0
        t_end_dat = len(data) * delta_t

    if not t_start:
        t_start = t_start_dat

    if not t_end:
        t_end = t_end_dat

    data = np.array(data)

    # Revert t_start and t_end to t_start_dat and t_end_dat i.e. to the
    # starting time and  duration of the data respectively if user specified
    # t_start is shorter and t_end is longer than t_start + the duration of
    # the data.
    t_start = max(t_start, t_start_dat)
    t_end = min(t_end, t_end_dat)

    start_index = int((t_start - t_start_dat) / delta_t)
    end_index = int((t_end - t_start_dat) / delta_t)

    if to_taper:
        data = taper(data, delta_t)

    integdat = np.zeros([end_index - start_index])

    integdat[0] = 0.0  # dat[0]*delta_t
    # totinteg = np.sum(np.array(data)) * delta_t # commented on May 5 2022 to
    # conform with pep8 style. Not sure of it's use.
    # mean = np.mean(dat)
    data = np.array(data)

    for i in range(1, end_index - start_index):
        integdat[i] = integdat[i - 1] + (data[start_index + i - 1]) * delta_t

    if TS is True:
        integdat = pycbc.types.timeseries.TimeSeries(
            integdat, delta_t, epoch=t_start
        )

    return integdat


def compute_frequencies(t_coal, t_val, chirp_mass):
    """Compute the Newtonian instantaneous frequency
    of strain waveform from coalescence time and
    chirp mass, from the Finn-Chernoff model.

    Parameters
    ----------
    t_coal: float
            The coalescence time,
    t_val: 1d array
           The time,
    chirp_mass: float
                The chirpmass.

    Returns
    -------
    freqs: float
           The instantaneous frequency of the strain waveform.
    """
    freqs = (
        (1.0 / (np.pi * chirp_mass))
        * (5.0 / 256) ** (3.0 / 8)
        * (chirp_mass / (t_coal - t_val)) ** (3.0 / 8)
    )

    return freqs


def totalmass(mass_ratio, chirp_mass):
    """Find total mass from mass ratio and chirpmass.

    Parameters
    ----------
    mass_ratio: float
                The mass ratio of the system.
    mchirp: float
            The chirp mass of the system.

    Returns
    -------
    total_mass: float
                The total mass of the system.
    """

    return (chirp_mass * (1.0 + mass_ratio) ** (6.0 / 5)) / mass_ratio ** (
        3.0 / 5
    )


def massratio(chirp_mass):
    """Compute the mass ratio from chirpmass.
    Assumes total mass to be 1.

    Parameters
    ----------
    mchirp: float
            The chirp mass of the system.

    Returns
    -------
    mass_ratio: float
                The Mass ratio of the system
    """
    mass_ratio = (
        chirp_mass ** (1.0 / 3)
        - 2.0 * chirp_mass**2.0
        - np.sqrt(chirp_mass ** (2.0 / 3) - 4.0 * chirp_mass ** (7.0 / 3))
    ) / (2.0 * chirp_mass**2.0)

    return mass_ratio


def compute_masses_from_mass_ratio_and_total_mass(mass_ratio, M=1):
    """Compute the individual masses from mass-ratio and total mass"""

    mass2 = M / (mass_ratio + 1)
    mass1 = mass_ratio * mass2

    return mass1, mass2

def compute_masses_from_mass_ratio_and_chirp_mass(mass_ratio, chirp_mass):

    total_mass = compute_total_mass_from_mass_ratio_and_chirp_mass(mass_ratio, chirp_mass)
    mass1, mass2 = compute_masses_from_mass_ratio_and_total_mass(mass_ratio, total_mass)
    return mass1, mass2

def compute_chirp_mass_from_mass_ratio_and_total_mass(mass_ratio, M=1):
    return M*(mass_ratio/(1+mass_ratio)**2 )**(3/5)

# Defining function for calculating Chirpmass from a2

def compute_total_mass_from_mass_ratio_and_chirp_mass(mass_ratio, chirp_mass):
    Mtotal = chirp_mass/((mass_ratio/(1+mass_ratio)**2)**(3/5)) 
    return Mtotal

def compute_chirp_mass(a2_param):
    """Compute the chirpmass from a2, the coefficient
    of time of the Finn-Chernoff waform model.

    Parameters
    ----------
    a2_param: 1d array \\ float
              The a2 parameter in the Finn Chernoff model.

    Returns
    -------
    chirp_mass: float
                The chirp mass
    """
    chirp_mass = 2 ** (8.0 / 5) / (5 * np.array(a2_param) ** (8.0 / 5))
    return chirp_mass


def compute_chirp_mass_from_masses(m1, m2):
    return (m1*m2)**(3/5) / (m1+m2)**(1/5)

def compute_chi_eff_from_masses_and_spins(spin1, spin2, larger_mass_ratio):
    """Compute the effective z-spin parameter
    :math:`\\chi_{eff}`

    Parameters
    ----------
    spin1,spin2: tuple of floats
                 The spin vector of the two
                 black holes
    larger_mass_ratio: float, >1
                The SpEC convention mass-ratio
                :math:`\\dfrac{M_1}{M_2}`
    """

    _, _, s1z = spin1
    _, _, s2z = spin2

    chi_eff = (s1z * larger_mass_ratio + s2z) / (1 + larger_mass_ratio)

    return chi_eff


def compute_chi_prec_from_masses_and_spins(spin1, spin2, larger_mass_ratio):
    """Compute the effective spin-precession parameter
    :math:`\\chi_{prec}`


    Parameters
    ----------
    spin1,spin2: tuple of floats
                 The spin vector of the two
                 black holes
    larger_mass_ratio: float,
                The SpEC convention mass-ratio,
                usually greater than 1
    """

    mass1, mass2 = compute_masses_from_mass_ratio_and_total_mass(
        larger_mass_ratio
    )

    s1x, s1y, s1z = spin1
    s2x, s2y, s2z = spin2

    s1p = mass1**2 * np.sqrt(s1x**2 + s1y**2)
    s2p = mass2**2 * np.sqrt(s2x**2 + s2y**2)

    A1 = 2 + 3 / (2 * larger_mass_ratio)
    A2 = 2 + 3 * larger_mass_ratio / (2)

    chi_prec = max(A1 * s1p, A2 * s2p) / (A1 * mass1**2)

    return chi_prec


def lengtheq(data_a, data_b, delta_t=None, is_ts=False):
    """Equalize the length of two timeseries/array by
    appending zeros at the end of the array. No tapering.

    Procedure
    ---------

    1. Check if input data are timeseries.
       If not then construct timeseries using delta_t.
    2. Check data length.
       If they are already equal then skip and return the inputs.
    3. Check if data_a is smaller then data_b ( or vice versa).
       Then augment zeroes at either ends of array to data_a
       (data_b) and return.


    Parameters
    ----------
    data_a: list
            The input waveform A
    data_b: list
            The input waveform B.
    delta_t: float
             The time steping. Defaults to `None`.
    is_ts: bool
           To determine whether the given data
           is a pycbc TimeSeries.
    Returns
    -------
    equalized_signals: list
                       The Tapered, length equalized waveforms
                       data_a and data_b, and a flag denoting
                       which waveform was changed, `a` or `b`.

    Notes
    -----
    Recommended usage:

    Change length of waveform `a` to match with that of waveform `b`.

    """

    # Check if input data vectors are pycbc TimeSeries. If yes, create a copy
    # of data and extract delta_t.
    # ts = isinstance(data_a, pycbc.types.timeseries.Timeseries)
    # and isinstance(data_b, pycbc.types.timeseries.Timeseries)
    try:
        import pycbc

        if isinstance(data_a, pycbc.types.timeseries.TimeSeries) and isinstance(
            data_b, pycbc.types.timeseries.TimeSeries
        ):
            is_ts = True
    except ModuleNotFoundError as mnf:
        message(mnf, " Treating data as numpy array", message_verbosity=2)

    if is_ts:
        if not delta_t:
            signala = data_a
            signalb = data_b

            try:
                delta_t = signala.delta_t

            except AttributeError:
                try:
                    delta_t = signalb.delta_t
                except BaseException:
                    message(
                        "Input is not a TimeSeries."
                        "Please supply a pycbc TimeSeries object"
                        "or the gridspacing as delta_t",
                        message_verbosity=0,
                    )
    else:
        delta_t = 1

    # If not TimeSeries, then construct TimeSeries using delta_t.
    data_a = np.array(data_a)
    data_b = np.array(data_b)

    signala = data_a
    signalb = data_b
    # signala = pycbc.types.timeseries.TimeSeries(data_a, delta_t)
    # signalb = pycbc.types.timeseries.TimeSeries(data_b, delta_t)

    # If the length is already equal, then skip.
    if len(data_a) == len(data_b):
        lflag = "ab"

    # If data_a < data_b
    elif len(data_a) < len(data_b):
        # add zeros to data_a when a is smaller
        lflag = "a"
        zers = len(data_b) - len(data_a)
        signala = np.transpose(
            np.concatenate(
                (np.transpose(data_a), np.transpose(np.zeros([zers])))
            )
        )
        # signala = pycbc.types.timeseries.TimeSeries(signala, delta_t)
        # return pycbc.types.timeseries.TimeSeries(signalb,delta_t),lflag
    # If data_b < data_a
    else:
        # message("Error!")
        # add zeros to b when b is smaller
        lflag = "b"
        zers = len(data_a) - len(data_b)
        signalb = np.transpose(
            np.concatenate(
                (np.transpose(data_b), np.transpose(np.zeros([zers])))
            )
        )
        # signalb = pycbc.types.timeseries.TimeSeries(signalb, delta_t)
        # return pycbc.types.timeseries.TimeSeries(signala,delta_t),lflag

    # Returns a list containing the length equalized arrays and the flag.
    if not is_ts:
        signala = np.array(signala)
        signalb = np.array(signalb)
    else:
        signala = pycbc.types.timeseries.TimeSeries(signala, delta_t)
        signalb = pycbc.types.timeseries.TimeSeries(signalb, delta_t)

    return [signala, signalb, lflag]


def taperlengtheq(data_a, data_b, delta_t=None):
    """Taper and equalize the lengths of two arrays.

    Parameters
    ----------
    data_a: list
            The input waveform A.
    data_b: list
            The input waveform B.
    delta_t: float
             The time steping.

    Returns
    -------
    equalized_signals: list
                       The Tapered, length equalized waveforms
                       data_a and data_b, and a flag denoting
                       which waveform was changed, `a` or `b`.
    """

    # Check if input data is pycbc TimeSeries.
    # If yes, then ectract delta_t.
    if not delta_t:
        signala = data_a
        signalb = data_b

        try:
            delta_t = signala.delta_t

        except AttributeError:
            try:
                delta_t = signalb.delta_t
            except BaseException:
                message(
                    "Input is not a TimeSeries."
                    "Please supply a pycbc TimeSeries object "
                    "or the gridspacing as delta_t",
                    message_verbosity=0,
                )

    # Ensure data is numpy array.
    signalb = np.array(data_b)
    # Taper waveform A.
    signala = np.array(taper(data_a, delta_t))

    # equalize the length of a to match with b
    # and return the length equalized
    # arrays.
    return lengtheq(signala, signalb, delta_t)


def iscontinuous(time_axis, delta_t=None, toldt=1e-3):
    """Check if the data has discontinuities.
    This checks for repetitive time rows and jumps.

    Notes
    -----
    Types of discontunuities

    0: Continuous.
    1: Repetitive rows.
    2: Jumps in timeaxis.

    Parameters
    ----------
    data: list
          Input as a list of 1d arrays [time, data1, data2, ...].
          All the data share the common time axis `time`
    delta_t: float
             The time stepping.
    toldt: float
           The tolerance for error in checking. defaluts to toldt=1e-3.

    Returns
    -------
    discontinuity_details: list.
                           It contains:
                            1. A list. details of discontinuity.
                               index location of original array,
                               the corresponding discinbtinuity type.
                            2. A float. the global discontinuity type.
    """

    time_axis = np.sort(time_axis)

    dt_axis = np.diff(time_axis)

    if not delta_t:
        delta_t = scipy.stats.mode(dt_axis)[0]

    discontinuity_type = 0
    discontinuity_dict = {}
    discont_locs = None

    if ((dt_axis - delta_t) > (toldt) * delta_t).any():
        discont_locs = np.where((dt_axis - delta_t) > (toldt) * delta_t)[0]
        discontinuity_type = 1

    discontinuity_dict.update({"gaps": [discontinuity_type, discont_locs]})

    discontinuity_type = 0

    rep_locs = None
    if ((dt_axis - delta_t) < -toldt * delta_t).any():
        rep_locs = np.where((dt_axis - delta_t) < -(toldt) * delta_t)[0]
        discontinuity_type = 1

    discontinuity_dict.update({"repetitions": [discontinuity_type, rep_locs]})

    return discontinuity_dict


def sort_data(data):
    """Sort the data according to its
    time axis. The first axis is assumend to be
    the time axis.

    Parameters
    ----------
    data:   ndarray
                    The data array with shape (naxis, time_steps)
                    The first axis is assumed to be the time axis.

    Returns
    -------
    sorted_data:    ndarray
                                    The sorted data array
    """

    data = np.array(data)

    # message("Data shape:", (data.shape), message_verbosity=3)
    # Set axis along which to remove
    naxis = data.ndim - 1
    message("nAxis:%d" % naxis, message_verbosity=3)
    transposed = False
    # Associate data[0] as timeaxis
    if naxis > 0:
        shapes = data.shape
        if shapes[0] < shapes[1]:
            transposed = True
            data = np.transpose(data)
        time = data[:, 0]
        message("The time array:%s" % time, message_verbosity=3)
    else:
        time = data

    order = np.argsort(time)

    sorted_data = data[order]

    # if naxis > 0:
    #    sorted_data = data[order, :]
    # else:
    #    sorted_data =
    if transposed:
        sorted_data = sorted_data.T

    return sorted_data


def remove_repetitive_rows(data, delta_t=1, toldt=1e-3):
    """Remove repeated rows in the data

    Parameters
    ----------
    data: ndarray
          Data array where the first axis refers to different
          columns of data. The zeroth column is time axis.
          The second axis refers to entries at different time
          steps.

    delta_t: float
             The tim stepping.
    toldt: float
           The tolerance for error in checking.
           Defaluts to toldt=1e-3.

    Returns
    -------
    cleaned_data: list
                  The cleaned data array with repetitive
                  rows removed.

    """
    message("Checking data for repetative rows...\n", message_verbosity=2)
    data = sort_data(data)

    # message("Data shape:", (data.shape), message_verbosity=3)
    # Set axis along which to remove
    naxis = data.ndim - 1
    message("nAxis:%d" % naxis, message_verbosity=3)
    transposed = False
    # Associate data[0] as timeaxis
    if naxis > 0:
        shapes = data.shape
        if shapes[0] < shapes[1]:
            transposed = True
            data = np.transpose(data)
        time = data[:, 0]
        message("The time array:%s" % time, message_verbosity=3)
    else:
        time = data

    # Index of ros to delete
    dind = []

    discontinuities = iscontinuous(time)

    repetition = bool(discontinuities["repetitions"][0])

    if repetition:
        rep_rows = discontinuities["repetitions"][1]

        data = np.delete(data, rep_rows, axis=0)

    else:
        message("No points removed\n", message_verbosity=2)

    # Return the "cleaned" data matrix
    cleaned_data = np.array(data)

    if transposed:
        return cleaned_data.T
    else:
        return cleaned_data


def fill_gaps_in_data(data, k=5):
    """Fill gaps in data by interpolation

    Parameters
    ----------
    data: list
          Input as a list of 1d arrays [time, data1, data2, ...].
          All the data share the common time axis `time`
    k: int, optional
       The interpolation order. Defaults to 5

    Returns
    -------
    cleaned_data: list
                  The cleaned data array with repetitive
                  rows and gaps (if bridge=True) removed.

    See Also
    --------
    scipy.interpolate.InterpolatedUnivariateSpline

    """

    data = sort_data(data)
    time = data[0]

    s1, ell = data.shape

    from scipy.stats import mode

    delta_t = mode(np.diff(time))[0][0]

    discontinuities = iscontinuous(time)

    gaps = bool(discontinuities["gaps"][0])

    if gaps:
        gap_rows = discontinuities["gaps"][1]
        message(
            "The data will be interpolated to bridge the gaps",
            message_verbosity=2,
        )

        # Interpolate the data to fill in the discontinuities
        t_final = time[-1]
        t_initial = time[0]

        proper_timeaxis = np.arange(t_initial, t_final, delta_t)

        interp_data = []
        interp_data.append(proper_timeaxis)

        from scipy.interpolate import (
            InterpolatedUnivariateSpline as interpolator,
        )

        for index in range(1, s1):
            interp_data.append(
                interpolator(time, data[index, :], k=k)(proper_timeaxis)
            )

        cleaned_data = np.array(interp_data)
        message("The data has been interpolated", message_verbosity=3)
    else:
        message("No jumps found in the data", message_verbosity=2)

    return cleaned_data


def cleandata(data, toldt=1e-3, bridge=False, k=5):
    """Check the data (time,datar,datai) for repetetive rows and remove them.

    Parameters
    ----------
    data: list
          Input as a list of 1d arrays [time, data1, data2, ...].
          All the data share the common time axis `time`
    toldt: float
           The tolerance for error in checking. defaluts to toldt=1e-3.
    bridge: bool
            A bridge flag to decide whether or not to interpolate and
            resample to fill in jump discontinuities.
    k: int, optional
       The interpolation order. Defaults to 5

    Returns
    -------
    cleaned_data: list
                  The cleaned data array with repetitive
                  rows and gaps (if bridge=True) removed.

    """

    # Check the data (time,datar,datai) for repetetive rows and remove them.
    # Ensure data as numpy array
    data = np.array(data)

    # message("Data shape:", (data.shape), message_verbosity=3)
    # Set axis along which to remove
    naxis = data.ndim - 1
    message("nAxis:%d" % naxis, message_verbosity=3)

    # Associate data[0] as timeaxis
    if naxis > 0:
        shapes = data.shape
        if shapes[0] > shapes[1]:
            data = np.transpose(data)
        time = data[0, :]
        message("The time array:%s" % time, message_verbosity=3)
    else:
        time = data

    time = np.sort(time)
    # delta_t = statistics.mode(np.diff(time))
    delta_t = scipy.stats.mode(np.diff(time))[0]

    message("shape of data:", (data.shape), message_verbosity=3)

    cleaned_data = remove_repetitive_rows(data, delta_t=delta_t, toldt=toldt)

    if bridge:
        cleaned_data = fill_gaps_in_data(cleaned_data, k=k)

    if naxis > 0:
        if shapes[0] > shapes[1]:
            cleaned_data = np.transpose(cleaned_data)
    return np.array(cleaned_data)


def shiftmatched(hdat, ind, delta_t=None, is_ts=False):
    """Timeshift an array. IMP: After timeshifting, the original
    length of the array is retained by clipping last(first) when
    ind > 0(ind <0) `ind` number of points!!. Make sure the input array
    already has number of zeros z > ind (z<ind) initially at the end.

    Parameters
    ----------
    hdat: 1d array or a pycbc TimeSeries object
          The input waveform to be shifted in time.
    ind: int
         The numper of places to shift the input waveform.
    delta_t: int
             the grid spacing in time.

    Returns
    -------
    shifted_wf: a pycbc TimeSeries object
                The waveform array of same length timeshifted by
                `ind` units by prepending zeros.
    """

    if is_ts:
        if not delta_t:
            try:
                delta_t = hdat.delta_t
            except BaseException:
                message(
                    "Input is not a TimeSeries."
                    "Please supply gridspacing as delta_t",
                    message_verbosity=0,
                )

    if ind > 0:
        # ind>0 case for shifting array to the right
        # message hdat
        # Length of the data
        # l = len(hdat)
        # Array holding zeroes to be appended
        zeros = np.zeros([ind])
        # The shifted array
        shifted_wf = np.transpose(
            np.concatenate((np.transpose(zeros), np.transpose(hdat)))
        )[:-ind]
        # message msig
        # message msig[:-ind]
        # Return the clipped, shifted timeseries

    elif ind < 0:
        # ind <0 case for shifting array to the left
        # Length of the data
        # l = len(hdat)
        # Array holding zeroes to be appended
        zeros = np.zeros([ind])
        # The shifted array
        shifted_wf = np.transpose(
            np.concatenate((np.transpose(hdat), np.transpose(zeros)))
        )[ind:]
        # message msig
        # message msig[:-ind]
        # Return a timeseries

    else:
        shifted_wf = hdat

    if is_ts:
        shifted_wf = pycbc.types.timeseries.TimeSeries(shifted_wf, delta_t)

    return shifted_wf


# @njit
def unwrap_phase(phi0):
    """Unwrap the phase by finding turning points in phi0.
    Finding turning points for unwrapping arctan2 function

    Parameters
    ----------
    phi0:   1darray
                    The wrapped phase which takes values in the
                    range (0, 2pi).
    Returns
    -------
    phic:   1darray
                    The unwrapped phase.

    """

    # Bookkeeping for upper (tpu) and lower (tpd) turning points
    tpu = []
    tpd = []
    # j = 0
    # k = 0
    # Upper turning point criterion
    for i in range(0, len(phi0) - 2):
        if phi0[i] > 5 and phi0[i + 1] < 1:
            tpu.append(i)
            # j = j+1
        # Lower turning point
        if phi0[i] < 1 and phi0[i + 1] > 5:
            tpd.append(i)
        # k = k+1

    # Trim any zeros in the array( Note: Unecessary)
    # tpu = np.trim_zeros(tpu)
    # tpd = np.trim_zeros(tpd)
    # Calculate the timestamp of the turning points
    # tput = delta_t * np.array(tpu)
    # tpdelta_t = delta_t * np.array(tpd)

    # Unwrapping the phase: Unwrap using the turning points

    # Variable for unwrapped phase
    phic = phi0
    # Iteration varible for turning points
    j = 0

    # Unwrap the upper turning points by adding 2pi for every 2p
    for i in range(0, len(tpu)):
        for j in range(int(tpu[i]) + 1, len(phic)):
            phic[j] = phic[j] + 2.0 * np.pi
    # Unwrap lower turning points by subtracting 2*pi for every tp
    for i in range(0, len(tpd)):
        for j in range(int(tpd[i]) + 1, len(phic)):
            phic[j] = phic[j] - 2.0 * np.pi

    return phic


# Complex Phase-Amplitude representation of data
def xtract_cphase(tsdata_p, tsdata_x, delta_t=None, to_plot=False):
    """Given real and imaginary parts of a complex timeseries,
    extract the phase of the waveform :arctan_(Img(data)/Re(data))

    Parameters
    ----------
    tsdata_p, tsdata_x: 1d array / a pycbc TimeSeries
                        The plus and cross polarized components of
                        the waveforms.
    delta_t: float, optional
             The time step. Overrides
             the timestep from pycbc TS object if given.
    to_plot: bool, optional
             True or False. Whether to plot the data or not

    Returns
    -------
    phic: 1d array
          The 1d array of the phase of the waveform.
    """

    # Assign the timestep. Real and imaginary parts are
    # assumed to have same timestep.
    # if not delta_t:
    #   try:
    #       delta_t = tsdata_p.delta_t
    #   except AttributeError:
    #       try:
    #           delta_t = tsdata_x.delta_t
    #       except:
    #           message("Input is not a TimeSeries.
    #           Please supply gridspacing as delta_t", message_verbosity=0)

    # Assign the timestep. Real and imaginary parts
    # are assumed to have same timestep.
    # Convert the data in numpy arrays

    datap = np.array(tsdata_p)
    datax = np.array(tsdata_x)

    # Calculate the wrapped phase (phi0 -> (0,2Pi))
    phi0 = np.pi + np.arctan2(datax, datap)

    # phic = unwrap_phase(phi0) - np.pi

    phic = np.unwrap(phi0) - np.pi
    # Plots.
    # Phase vs time.
    if to_plot:
        timeaxis = np.linspace(0, len(phi0) * delta_t, len(phi0))
        message(len(timeaxis), len(phi0))
        plt.scatter(timeaxis, phi0, s=1)
        plt.title("Phase")
        plt.xlabel("cctk_time")
        plt.ylabel("Phase in radians")
        plt.grid()
        # plt.savefig('../graphs/waveform_phase_{}_q1a0.pdf'.format(name))
        plt.show()

        timeaxis = np.linspace(0, len(phic) * delta_t, len(phic))
        message(len(timeaxis), len(phic))
        # Unwrapped phase vs time
        plt.scatter(timeaxis, phic, s=1)
        plt.title("Phase (unwrpped)")
        plt.xlabel("cctk_time")
        plt.ylabel("Phase in radians")
        plt.grid()
        # plt.savefig('../graphs/waveform_phase_complete_{}_q1a0.pdf'
        # .format(name))
        plt.show()
        # Return a 1d list containing the unwrapped phase
    return phic


def xtract_camp(tsdata_p, tsdata_x, to_plot=False):
    """Given real and imaginary parts of a complex timeseries,
    extract the amplitude of the complex data vector
    : (tsdata_p + i * tsdata_x)

    Parameters
    ----------
    tsdata_p, tsdata_x: pycbc TimeSeries/1darray
                        The plus and cross polarized components
                        of the waveforms.
    to_plot: bool
             True or False. Whether to plot the data or not

    Returns
    -------
    camp: 1d array
          The 1d array of extracted amplitudes of the waveform.
    """

    # Assign the timestep. Real and imaginary parts
    # are assumed to have same timestep.
    # Complex modulous of the data
    camp = np.sqrt(np.array(tsdata_p) ** 2 + np.array(tsdata_x) ** 2)

    if to_plot:
        # Plot amplitude vs time
        plt.plot(camp)
        plt.title("Amplitude vs time")
        plt.xlabel("cctk_time")
        plt.ylabel("Amplitude")
        plt.grid()
        # plt.savefig('../graphs/waveform_phase_complete_{}_q1a0.pdf'
        # .format(name))
        plt.show()
    # Returns the 1d numpy array amplitude
    return camp


def xtract_camp_phase(tsdata_1, tsdata_2):
    """Wrapper for extracting the amplitude
     and the phase of the complex vector.

    Parameters
    ----------
    tsdata_1, tsdata_2: 1d array or pycbc TimeSeries object
                        The two input waveforms as timeseries
                        vectors, the plus and cross polarized
                        components.
    delta_t: float
             The timestepping delta_t.

    Returns
    -------
    amplitude: 1d array
               A list containing complex amplitude
               (list) and phase (list).
    """

    return xtract_camp(tsdata_1, tsdata_2), xtract_cphase(tsdata_1, tsdata_2)


def get_waveform_angular_frequency(
    waveform, delta_t, timeaxis=None, method="FD"
):
    """Get the angular frequency of the waveform given
    the complex waveform time step. The phase is
    extracted and is differentiated using one of
    the available methods to compute the angular
    frequency.

    Parameters
    ----------
    waveform: 1d array
              The 1d complex array of the input waveform.

    delta_t: float
             The time step.

    timeaxis: 1d array, optional
              The time axis of the waveform.
              Recommended especially when the sampling
              is non-uniform and Chebyshev method is chosen.
    method: str
            The method for computing the derivative.
            Can be `FD` or `CS`. See below for more
            information.

    Returns
    -------
    omega: 1d array
           The real instantaneous frequency of the waveform.



    Notes
    -----
    Available methods

    Chebyshev series (`CS`): The phase is expanded in
                             a Chebyshev series and
    Finite Differencing (`FD`): A 11 point finite difference
                                scheme is used to differentiate
                                the phase, and is then smoothened
                                using the Savgol filter.
    """
    # Get the real and imaginary parts.
    waveform_p = waveform.real
    waveform_x = waveform.imag
    # Get the phase
    phase = xtract_cphase(waveform_p, waveform_x)
    # Compute the derivative
    if method == "FD":
        # Finite differencing method
        from scipy.signal import savgol_filter

        from waveformtools.differentiate import differentiate5

        # Compute the derivative
        omega_rough = differentiate5(phase, delta_t)
        # Smoothen the derivative
        omega_sm = savgol_filter(omega_rough, 41, polyorder=3)
        # Duplicate the last elements to bring
        # the length to that of input waveform.
        len_omega_sm = len(omega_sm)
        while len_omega_sm != len(phase):
            omega_sm = np.concatenate((omega_sm, np.array([omega_sm[-1]])))
            len_omega_sm = len(omega_sm)

    if method == "CS":
        # Chebyshev spectools method
        # Chebyshev spectools method
        from waveformtools.differentiate import Chebyshev_differential

        if not timeaxis:
            # Construct the time axis
            timeaxis = np.arange(0, len(waveform) * delta_t, delta_t)

        omega_sm = Chebyshev_differential(timeaxis, phase, degree=25)

    return omega_sm


def get_starting_angular_frequency(waveform, delta_t, npoints=400):
    """Get the approximate starting frequency of the
    input data by averaging over the first `npoints` number of points.

    Parameters
    ----------
    waveform:   1d array
                            The 1d complex array of the input waveform.
    delta_t:    float
                            The time step.
    npoints:    int
                            The number of points to average over.

    Returns
    -------
    omega0:     float
                            The approximate starting angular frequency.


    Notes
    -----
    Please suppy a conditioned input waveform that is neatly clipped,
    and not tapered.
    """

    # Get angular frequencies
    omegas = get_waveform_angular_frequency(waveform, delta_t)

    # Compute the starting frequency as the mean of first npoints.
    omega0 = np.mean(omegas[100 : 100 + npoints])

    return omega0


# Simple overlap. #Error. Add frequency domain overlap computation.
def olap(data1, data2, psd=1):
    """Calcuate the overlap between two data vectors
    weighted by the given psd.

    Parameters
    ----------

    data1, data2:   1d array or a pycbc RimeSeries object
                                    The input waveforms
    psd:            1d array
                                    The power spectools density to weight.
                                    The power spectools density to weight.

    Returns
    -------
    overlap:        float
                                    The overlap divided by the psd.
    """

    if psd == 1:
        data1 = np.array(data1)
        data2 = np.array(data2)
        overlap = np.sum(data1 * data2) / psd

    else:
        1
        # Returns the overlap weighted by the psd if any
    return overlap


def norm(hdat, psd=1.0):
    """Calculate the norm of a vector.

    Parameters
    ----------
    hdat:   1d array or a pycbc TimeSeries object.
                    The input waveform.
    psa:    1d array
                    The noise power spectools density of the inner product.
                    The noise power spectools density of the inner product.

    Returns
    -------
    norm_f: float
                    The norm with weighting by the psd.
    """

    hdat = np.array(hdat)

    norm_f = np.sqrt(np.sum(hdat * hdat) / np.array(psd))

    # message("Norm is %f"%normfa)
    return norm_f


def flatten_3l(nflist):
    """Flatten a (3rd order) list of list of lists.
    i.e. a three tier list [[[],[]], [[],[]] ---> [].
    This is useful e.g. when combining the data from
    the list output of multiple MPI ranks.

    Parameters
    ----------
    nflist : a third tier list
             A list of list of lists (a list of depth three).

    Returns
    -------
    flattened_list : a list
                     The flattened list i.e. a tier one list.
    """

    flattened_list = []

    for item in nflist:
        for sub_item in item:
            flattened_list.append(sub_item)

    message("list length: (%d)" % (len(flattened_list)))
    # Return the 1d flattened list
    return flattened_list


def flatten_l(nflist):
    """Flatten a (3rd order) list of list of lists.
    i.e. a three tier list [[[],[]], [[],[]] ---> [].
    This is useful e.g. when combining the data from
    the list output of multiple MPI ranks.

    Parameters
    ----------
    nflist : a third tier list
             A list of list of lists (a list of depth three).

    Returns
    -------
    flattened_list : a list
                     The flattened list i.e. a tier one list.
    """

    if len(nflist) == 1:
        return nflist

    else:
        flattened_list = []

        for item in nflist:
            for sub_item in item:
                flattened_list.append(sub_item)

        message("list length: (%d)" % (len(flattened_list)))
        # Return the 1d flattened list
        return flattened_list


def startend(data):
    """Identify the start and endpoints of the data.

    Notes
    -----
    The starting and ending index of the non-zero part
    of the data is the identification criterion.
    Requires the data to be exactly zero outside
    a certain domain.

    Parameters
    ----------
    data : 1d array or a pycbc TimeSeries object
           The input waveform.

    Returns
    -------
    start_index, end_index : int (2)
                             The pair of indices denoting the start
                             and end points of an array
    """

    try:
        start_index = np.where(np.array(data) != 0)[0][0]
    except BaseException:
        message(
            colored("Warning! Start index not found!!", "red"),
            message_verbosity=1,
        )
        start_index = 0

    try:
        end_index = np.where(np.array(data) != 0)[0][-1] + 1
    except BaseException:
        message(
            colored("Warning! End index not found!!", "red"),
            message_verbosity=1,
        )
        end_index = 0
    return start_index, end_index


def apxstartend(data, tol=1e-5):
    """Identify the Approximate start and endpoints of the data.

    Notes
    -----
    The starting and ending index of the peak
    tol (default 1e-5)  part of the data is
    the identification criterion. Requires
    the data to fall off to tol*peak absolute
    value outside a certain range.

    Parameters
    ----------
    data : 1d array or a pycbc TimeSeries object
           The input waveform whose start and
           end points need to be identified based
           on a given tolerance value.
    tol : float
          The tolerance value below which
          the signal is assumed to be absent.

    Returns
    --------
    loc_pair : int (2)
               The pair of indices denoting
               the start and end points of an array
    """

    data = np.array(data)
    locs = np.where(data > np.amax(data) * tol)[0]

    loc_pair = locs[0], locs[-1] + 1

    # Return the beginning and ending indices
    return loc_pair


def addzeros(data, zeros):
    """Append zeros to an array without tapering.

    Parameters
    ----------
    data : 1d array or a pycbc TimeSeries
           The waveform data as list or numpy array or pycbc timeseries.
    zeros : int
            The number of zeros to be added.

    Returns
    -------
    data : 1darray
           data with `zeros` number of zeros concatenated
           at the end as numpy 1d array
    """

    return np.transpose(
        np.concatenate(
            (np.transpose(np.array(data)), np.transpose(np.zeros([zeros])))
        )
    )


def removezeros(data, delta_t):
    """Remove zeros from the input waveform from either sides.
    Similar to startend but return the truncated array.

    Parameters
    ----------
    data : 1d array or a pycbc TimeSeries
           The input waveform.
    delta_t : float, optional
              The time stepping.

    Returns
    -------
    short_ts : a list
               A list containing waveforms with zeros
               removed on either sides,
               the start and end indices in the format
               [short_ts, [start_index, end_index]]

    """

    # Assign the timestep.
    # Real and imaginary parts are assumed to have same
    # timestep.
    if not delta_t:
        try:
            delta_t = data.delta_t
        except BaseException:
            message(
                "Input is not a TimeSeries."
                " Please supply gridspacing as delta_t",
                message_verbosity=0,
            )

    starti, endi = startend(data)
    ret_data = pycbc.types.timeseries.TimeSeries(
        np.array(data)[starti:endi], delta_t
    )

    short_ts = [ret_data, [starti, endi]]

    return short_ts


def shorten(tsdata, start, end, delta_t=None):
    """Shorten an array given the start and end points.

    Parameters
    ----------
    tsdata : 1d array or a pycbc TimeSeries object
             The waveform data.
    start : int
            The start index of the data.
    end : int
          The end index of the data.
    delta_t : float, optional
              The time stepping.

    Returns
    -------
    short_ts : a pycbc TimeSeries object
               The shortened data,
               clipped before start and after end.
    """
    # Assign the timestep.
    # Real and imaginary parts
    # are assumed to have same

    # timestep.
    if not delta_t:
        try:
            delta_t = tsdata.delta_t
        except BaseException:
            message(
                "Input is not a TimeSeries."
                "Please supply gridspacing as delta_t",
                message_verbosity=0,
            )

    short_ts = pycbc.types.timeseries.TimeSeries(
        np.array(tsdata)[start:end], delta_t
    )

    return short_ts


def taper_tanh(
    waveform, time_axis=None, delta_t=None, duration=10, sides="both"
):
    """
    Taper a waveform with a :math:`tanh` function
    at either ends

    Parameters
    ----------
    waveform : 1d array
               A 1d array of waveform data.
    delta_t : float, optional
              The time stepping `delta_t`.
              Optional if `time_axis` is given.
    percent : int, optional
              The percent of data to taper. This
              is equally distributed on either
              sides of the array. Defaults to 10.
    sides : str
            A string indicating which sides to taper.
            `beg` tapers the beginning, `end` tapers the end
            and `both` tapers both the ends.

    Returns
    -------
    time_axis, waveform : 1d array
                          The timeaxis and the waveform data
                          of the tapered waveform.
    """

    data_len = len(waveform)

    if delta_t is None:
        try:
            delta_t = time_axis[1] - time_axis[0]
        except Exception as excep:
            delta_t = 1
            message("Assuming unity for time step", excep)

    #    delta_t = 1
    if time_axis is None:
        # Try to construct using `delta_t`
        try:
            time_axis = np.arange(0, data_len * delta_t, delta_t)
            # message('time axis', time_axis)
        except Exception as excep:
            message("Please suppy the time axis or delta_t!", excep)

    nearest_lower_power = int(np.log(data_len) / np.log(2))

    # message('nearest lower power', nearest_lower_power)
    # Change in length of the data.
    data_delta_len = np.power(2, nearest_lower_power + 2) - data_len
    # message('data delta len', data_delta_len)
    # nend_points = int((duration/2)/delta_t) + int(data_delta_len/2)
    nend_points = int(data_delta_len / 2)
    nstart_points = data_delta_len - nend_points

    # message('N startend points', nend_points, nstart_points)
    # message('New time axis', new_time_axis)
    tfinal = data_len * delta_t

    # start_axis = np.linspace(-1, 1, nstart_points)
    # end_axis   = np.linspace(1, -1, nend_points)

    # norm_time_axis = time_axis/max(time_axis)

    # n_delta_t = delta_t/data_len

    # from scipy.interpolate import interpolate
    waveform_widened = np.concatenate(
        (np.zeros([nstart_points]), waveform, np.zeros([nend_points - 1]))
    )
    new_time_axis = np.linspace(
        time_axis[0] - nstart_points * delta_t,
        time_axis[-1] + nend_points * delta_t,
        len(waveform_widened),
    )

    start_win = (
        np.tanh(3 * (new_time_axis - duration / 2) / (duration / 2)) + 1
    ) / 2
    end_win = (
        np.tanh(3 * (-new_time_axis + (tfinal - duration / 2)) / (duration / 2))
        + 1
    ) / 2

    # plt.scatter(new_time_axis, waveform_widened, s=1)
    # plt.show()

    if sides == "both":
        tapered_waveform = start_win * end_win * waveform_widened
    elif sides == "beg":
        tapered_waveform = start_win * waveform_widened
    elif sides == "end":
        tapered_waveform = end_win * waveform_widened
    else:
        message(
            "Please specify valid sides argument."
            "Sides can be beg, end or both."
        )

    # plt.scatter(new_time_axis, tapered_waveform, s=1)
    # plt.show()

    return new_time_axis, tapered_waveform


def taper(data, delta_t=1, zeros=150):
    """A method to taper and append additional zeros
    at either ends, using the `taper` function of
    the pycbc TimeSeries object.

    Parameters
    ----------
    data : 1d array or a pycbc TimeSeries
           The waveform data
    delta_t : float
              The timestepping.
    zeros : int
            The number of zeros to be added.

    Returns
    -------
    tapered_data : 1d array or a pycbc TimeSeries
                   The waveform data tapered and zero padded.

    Notes
    -----
    See `taper_timeseries` from pycbc.waveform.utils for more details."""

    import pycbc

    # Check if data is pycbc timeseries:
    if not isinstance(data, pycbc.types.timeseries.TimeSeries):
        # flag = 1
        # Convert to numpy array
        data = np.array(data)
        # First taper both sides of the data
        # i.e. the start and end of the data.
        # Convert to pycbc TimeSeries
        data = pycbc.types.timeseries.TimeSeries(data, delta_t)
    else:
        delta_t = data.delta_t

    # Taper the timeseries
    from pycbc.waveform.utils import taper_timeseries

    tapered_data = taper_timeseries(data, tapermethod="TAPER_STARTEND")

    # Append the zeros
    tapered_data = np.array(tapered_data)

    # Pad ends with extra zeros
    zeros = np.zeros([zeros])
    # Prepend with z zeros
    tapered_data = np.transpose(
        np.concatenate((np.transpose(zeros), np.transpose(tapered_data)))
    )
    # Append with extra zeros
    tapered_data = np.transpose(
        np.concatenate((np.transpose(tapered_data), np.transpose(zeros)))
    )

    # Convert back to timeseries if the input was a time series.
    tapered_data = pycbc.types.timeseries.TimeSeries(tapered_data, delta_t)

    # Return the timeseries
    return tapered_data


def low_cut_filter(utilde, freqs, order=2, omega0=0.03):
    """Apply low frequency cut filter using a butterworth filter.

    Parameters
    ----------
    utilde : 1d array
             The frequency domain data.
    freqs : 1d array
            The frequencies.
    order : int
            The order of the butterworth filter.
    omega0 : float
             The cutoff frequency of the butterworth filter.

    Returns
    -------
    utilde_lc : 1d array
                The filtered data.
    """

    from scipy import signal
    from scipy.interpolate import interp1d

    # import matplotlib.pyplot as plt

    b, a = signal.butter(order, omega0, "high", analog=True)

    w, h = signal.freqs(b, a)

    # Make negative axis and data.
    filter_freqs = np.concatenate((-w[::-1], w))
    filter_coeffs = np.concatenate((np.conjugate(h[::-1]), h))

    # Resample the filter at data freqs.
    filter_int = interp1d(filter_freqs, filter_coeffs)
    filter_resam = filter_int(freqs)

    filtered_signal = utilde * filter_resam

    return filtered_signal


def center(wvp, wvc=None, delta_t=None):
    """Center a waveform (wvp, wvc) at the peak
    of the complex modulous sqrt(wvp**2 + wvc**2).

    Parameters
    ----------
    wvp, wvc : 1d array or a pycbc TimeSeries object
               The one/two components of the waveforms
               as 1d list or numpy arrays or pycbc timeseries.
    delta_t : float
              The timestepping delta_t.

    Returns
    -------
    centered_wf : a pycbc TimeSeries object
                  The two 1d centered waveform(s)
                  as individual pycbc timeseries.

    """
    # Flag to find out if both polarizations are supplied or not.
    flag = 0
    # If only one waveform is provided, assume cross pol = plus pol.
    if not wvc:
        flag = 1
        wvc = wvp
    # Assign the timestep.
    # Real and imaginary parts are assumed to have same
    # timestep.
    if not delta_t:
        try:
            delta_t = wvp.delta_t
        except AttributeError:
            try:
                delta_t = wvc.delta_t
            except BaseException:
                message(
                    "Input is not a TimeSeries."
                    "Please supply gridspacing as delta_t",
                    message_verbosity=0,
                )

    datap = np.array(wvp)
    datac = np.array(wvc)

    # Assign the complex amplitude
    amp = np.power(datap, 2) + np.power(datac, 2)
    # Find the location of the max amplitude
    ind = np.where(amp == np.max(amp))[0][0]
    # Calculate the epoch
    tlim = [-ind * delta_t, (len(datap) - ind) * delta_t]
    # Returns the centered wvp and wvc
    # if wvc was provided else returns just
    # the former.

    if flag == 1:
        centered_wf = pycbc.types.timeseries.TimeSeries(
            datap, delta_t, epoch=tlim[0]
        )
    else:
        centered_wf = (
            pycbc.types.timeseries.TimeSeries(datap, delta_t, epoch=tlim[0]),
            pycbc.types.timeseries.TimeSeries(datac, delta_t, epoch=tlim[0]),
        )

    return centered_wf


def get_centered_taxis(time_ax, amps):
    """Get the time axis of the waveform centered
    at its maximum absolute amplitude.

    Parameters
    ----------
    time_ax : 1d array
              The 1d array containg the original
              (uncentered)time axis of the wveform.

    amps : 1d array
           The amplitude of the waveform.

    Returns
    -------
    time_centered : 1d array
                    The centered time axis of the waveform.
                    The maximum amplitude timestamp
                    will be at :math:`t=0`.
    """

    # Get the maximum amplitude
    amps_max = np.argmax(abs(amps))
    # Get the time stamp of the max amp location.
    time_max = time_ax[amps_max]
    # Center the time axis.
    time_cen = time_ax - time_max

    return time_cen


def plot(xdata, func_x, save="no"):
    """A Basic plotting function.

    Parameters
    ----------
    xdata : 1d array
            The x axis of the function,
    func_x : 1d array
             The y axis of the function f(x),
    save : bool.
           True or False. Whether the plot should be saved or not.

    Returns
    -------
    1 : int
    plots : figures to stdout and disk
            Displays the plot, and Saves with the filename provided.
    """

    plt.plot(np.array(xdata), np.array(func_x))
    plt.title("f(x) vs x")
    plt.grid(which="both", axis="both")
    plt.xlabel("xdata")
    plt.ylabel("f(x)")
    if save != "no":
        plt.savefig(save + ".pdf")
    plt.show()
    return 1


def coalignwfs(tsdata1, tsdata2, delta_t=None):
    """Coalign two timeseries. Wrapper and modification around
    pycbc functions with some additional functionalities.

    Parameters
    ----------
    tsdata1, tsdata2 : list, 1d array or a pycbc TimeSeries
                       The two data vectors as 1d lists or
                       numpy arrays or pycbc TimeSeries
    delta_t : float
              The time stepping.

    Returns
    -------
    ctsdata1, tsdata2 : a pycbc TimeSeries
                        A pair of pycbc TimeSeries objects
                        the aligned first waveform and the second.

    Notes
    -----
    The first waveform is changed.
    """

    # Lengths of the two input timeseries
    # len1 = len(tsdata1)
    # len2 = len(tsdata2)

    # Add zeros at the end of waveform 1 without tapering if len2>len1
    # if len2>len1:
    #                tsdata1,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
    # Add zeros at the end of waveform 2 without tapering if len1>len2
    # elif len1>len2:
    #                tsdata2,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)

    # Assign the timestep. Real and imaginary parts are assumed to have same
    # timestep.

    if not delta_t:
        try:
            delta_t = tsdata1.delta_t
        except AttributeError:
            try:
                delta_t = tsdata2.delta_t
            except BaseException:
                message(
                    "Input is not a TimeSeries."
                    "Please supply gridspacing as delta_t",
                    message_verbosity=0,
                )

    tsdata1, tsdata2, _ = lengtheq(tsdata1, tsdata2, delta_t, is_ts=True)
    from pycbc.types.timeseries import TimeSeries

    tsdata1 = TimeSeries(tsdata1, delta_t=delta_t)
    tsdata2 = TimeSeries(tsdata2, delta_t=delta_t)

    from pycbc.filter.matchedfilter import matched_filter

    # Calculate complex SNR using pycbc function. Note: This complex SNR is
    # actually the complex SNR * norm of the timeseries.
    csnr = matched_filter(tsdata1, tsdata2)
    # Find the absolute value of the complex SNR timeseries
    acsnr = np.array(np.abs(csnr))

    # message(acsnr,np.max(acsnr))
    # Find the location of the maximum element in acsnr
    maxloc = (np.where(acsnr == np.max(acsnr)))[0][0]
    message("Max location is %s, match is %s" % (maxloc, np.max(acsnr)))

    # Shift the waveform 1 in time using maxloc
    tsdata1 = roll(tsdata1, maxloc, is_ts=True)

    # tsdata1 = shiftmatched(tsdata1, maxloc, delta_t, is_ts=True)
    # Phase shift ( rotate) the waveform 1 by multipying
    # the frequency series of waveform 1
    # with the phase of the max element in acsnr
    # Calculate the rotation (as phase of the
    # max complex modulous element in
    # acsnr
    rotation = csnr[maxloc] / np.absolute(csnr[maxloc])
    # Rotate and take the inverse Fourier transform
    ctsdata1 = (rotation * tsdata1.to_frequencyseries()).to_timeseries()
    # Return the time and phase shifted waveform 1
    # to coalign with 2 and waveform 2.
    # Note that the max modulous element of acsnr
    # is only used to compute the time
    # shift and is not used to normalize the
    # waveforms. This therefore returns waveforms
    # with their original
    # normalization.

    return ctsdata1, tsdata2


def coalignwfs2(tsdata1, tsdata2, delta_t=None):
    """Coalign two waveforms function 2.

    Parameters
    ----------
    tsdata1, tsdata2 : 1d array or a pycbc TimeSeries
                       two data vectors as 1d lists or
                       numpy arrays or pycbc timeseries.

    Returns
    --------
    aligned_waveforms : list
                        The aligned waveforms in the format
                        [aligned_wf1,
                        aligned_wf2,
                        [norm1, norm2, location of maximum]].

    Notes
    -----
    See coalignwfs for more description. This algorithm does not use
    the builtin coalign function from pycbc. This involves normalization
    of the data vectors explicitly and identifiies the timeshift by computing
    the complex SNR and finding the maximum element.
    """
    # Lengths of the two input timeseries
    len1 = len(tsdata1)
    len2 = len(tsdata2)

    if not delta_t:
        try:
            delta_t = tsdata1.delta_t
        except AttributeError:
            try:
                delta_t = tsdata2.delta_t
            except Exception as excep:
                message("Please input delta_t or a valid TimeSeries!", excep)

    tsdata1, tsdata2, _ = lengtheq(tsdata2, tsdata1, delta_t, is_ts=True)

    # Find startend.
    if len1 == len2:
        try:
            start, end = startend(np.array(tsdata1))
        except BaseException:
            start, end = apxstartend(np.array(tsdata1))

    # Add zeros at the end of waveform 1 without tapering if len2>len1
    if len2 > len1:
        #       tsdata1,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
        try:
            start, end = startend(np.array(tsdata1))
        except BaseException:
            start, end = apxstartend(np.array(tsdata1))

    # Add zeros at the end of waveform 2 without tapering if len1>len2
    elif len1 > len2:
        # tsdata2,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
        try:
            start, end = startend(np.array(tsdata2))
        except BaseException:
            start, end = apxstartend(np.array(tsdata2))

    # match,shift=pycbc.filter.matchedfilter.match(tsdata1,tsdata2)

    # Normalize the waveforms
    norm1 = norm(np.array(tsdata1[start:end]))
    norm2 = norm(np.array(tsdata2[start:end]))

    tsdata1_cropped = pycbc.types.timeseries.TimeSeries(
        np.array(tsdata1[start:end]) / norm1, delta_t
    )
    tsdata2_cropped = pycbc.types.timeseries.TimeSeries(
        np.array(tsdata2[start:end]) / norm2, delta_t
    )

    from pycbc import filter

    max_match, max_shift = pycbc.filter.matchedfilter.match(
        tsdata1_cropped, tsdata2_cropped
    )

    tsdata1 = pycbc.types.timeseries.TimeSeries(
        np.array(tsdata1) / norm1, delta_t
    )
    tsdata2 = pycbc.types.timeseries.TimeSeries(
        np.array(tsdata2) / norm2, delta_t
    )

    # Calculate complex SNR using pycbc function. Note: This complex SNR is
    # actually the complex SNR * norm of the timeseries.
    csnr = pycbc.filter.matchedfilter.matched_filter(tsdata1, tsdata2)

    # Find the absolute value of the complex SNR timeseries
    acsnr = np.array(np.abs(csnr))
    # message(acsnr,np.max(acsnr))

    # Find the location of the maximum element in acsnr
    maxloc = (np.where(acsnr == np.max(acsnr)))[0][0]
    mmatch = np.amax(acsnr)
    message(
        f"Max location is {maxloc},"
        f"match is {mmatch}"
        f"max shift is {max_shift}"
    )

    # Shift the waveform 1 in time using maxloc
    tsdata1 = shiftmatched(tsdata1, maxloc, delta_t, is_ts=True)

    # Phase shift ( rotate) the waveform 1 by multipying
    # the frequency series of waveform 1 with the phase
    # of the max element in acsnr
    # Calculate the rotation (as phase of the max complex
    # modulous element in
    # acsnr
    rotation = csnr[maxloc] / np.absolute(csnr[maxloc])

    # Rotate and take the inverse Fourier transform
    ctsdata1 = (rotation * tsdata1.to_frequencyseries()).to_timeseries()

    # Recenter waveform 0 and assign the timeaxis of waveform 0 to waveform1
    ctsdata1, dummy = center(ctsdata1, ctsdata1)

    tsdata2 = pycbc.types.timeseries.TimeSeries(
        np.array(tsdata2), tsdata2.delta_t, epoch=ctsdata1.sample_times[0]
    )

    # Return the normalized, time and phase shifted waveform 1 to coalign with
    # 2 and waveform 2.
    aligned_waveforms = {
        "wf1": ctsdata1,
        "wf2": tsdata2,
        "norms": [norm1, norm2],
        "shift": maxloc,
        "match": max_match,
    }

    return aligned_waveforms


def resample_wfs(both_time_axes, both_waveforms, delta_t="auto", Plot=False):
    """Resample the waveform pairs.

    Parameters
    ----------
    both_time_axes : list
                     A list containing two 1d arrays
                     representing the time axes.
    both_waveforms : list
                     A list containing two 1d arrays
                     respresenting the waveforms.
    delta_t : string, float
              The time step to resample at.
              Auto uses the finest of the two available.
              A float value can be provided by the user as well.

    Returns
    -------
    both_time_axes_resam : 1d array
                           A 1d array representing
                           the resampled time axes.
    both_waveforms_resam : list
                           A list containing two 1d arrays
                           respresenting the resampled waveforms.
    """

    # from waveformtools.waveformtools import lengtheq
    # from scipy.interpolate import interp1d

    waveform1, waveform2 = both_waveforms
    time_axis1, time_axis2 = both_time_axes

    min_t = max(min(time_axis1), min(time_axis2))
    max_t = min(max(time_axis1), max(time_axis2))

    message("Taxis limits")
    message(f"WF1 tmin {min(time_axis1)} tmax {max(time_axis1)}")
    message(f"WF2 tmin {min(time_axis2)} tmax {max(time_axis2)}")

    delta_t1 = sorted(time_axis1)[1] - sorted(time_axis1)[0]
    delta_t2 = sorted(time_axis2)[1] - sorted(time_axis2)[0]

    if not isinstance(delta_t, float):
        if delta_t == "auto":
            delta_t = min(delta_t1, delta_t2)
        elif delta_t == "A":
            delta_t = delta_t1
        elif delta_t == "B":
            delta_t = delta_t2

    # time_axis = shear_1.time_axis
    # waveform1, waveform2, flag =
    # lengtheq(waveform1, waveform2, delta_t=delta_t)

    new_time_axis = np.arange(min_t, max_t, delta_t)

    wf1_amp, wf1_phase = xtract_camp_phase(waveform1.real, waveform1.imag)
    wf2_amp, wf2_phase = xtract_camp_phase(waveform2.real, waveform2.imag)

    # wf1_amp_int_fun     = interp1d(time_axis1, wf1_amp, kind='cubic')
    # wf1_phase_int_fun   = interp1d(time_axis1, wf1_phase, kind='cubic')

    # wf1_amp_resam       = wf1_amp_int_fun(new_time_axis)
    # wf1_phase_resam     = wf1_phase_int_fun(new_time_axis)

    wf1_amp_resam = interp_resam_wfs(wf1_amp, time_axis1, new_time_axis)
    wf1_phase_resam = interp_resam_wfs(wf1_phase, time_axis1, new_time_axis)

    wf1_resam = wf1_amp_resam * np.exp(1j * wf1_phase_resam)

    wf2_amp_resam = interp_resam_wfs(wf2_amp, time_axis2, new_time_axis)
    wf2_phase_resam = interp_resam_wfs(wf2_phase, time_axis2, new_time_axis)

    # wf2_amp_resam       = wf2_amp_int_fun(new_time_axis)
    # wf2_phase_resam     = wf2_phase_int_fun(new_time_axis)

    wf2_resam = wf2_amp_resam * np.exp(1j * wf2_phase_resam)

    return new_time_axis, wf1_resam, wf2_resam


def match_wfs(all_time_axes, all_waveforms, delta_t="auto"):
    """Match two waveforms and return the time shift,
    phase shift, normalized waveforms and match coefficient.

    Parameters
    ----------
    time_axes : list
                A list containing the time axes
                of the two waveforms
    waveforms : list
                A list of two waveforms.
                Each is a 1d array.
    delta_t : float, string, optional
              The time step of the resampled arrays.
              Can be A, B, auto or any float value.

    Returns
    -------
    match_details : dict
                    A dictionary containing the
                    i). match coeffient
                    ii). time_shift
                    iii). phase shift in radians
                    iv). normalized, resampled, waveforms
                         and their time-axes.

    Note
    ----
    The shifts give by how much
    the first waveform has to be shifted
    to match with the second.

    """

    time_axis1, time_axis2 = all_time_axes

    delta_t_A = sorted(time_axis1)[1] - sorted(time_axis1)[0]
    delta_t_B = sorted(time_axis2)[1] - sorted(time_axis2)[0]

    if isinstance(delta_t, str):
        if delta_t == "auto":
            from scipy.stats import mode

            # tsorted = sorted(time_axis1)
            # delta_t = tsorted[0] - tsorted[1]

            delta_t = mode(np.diff(sorted(time_axis1)))[0][0]

            # delta_t = min(delta_t_A, delta_t_B)
        elif delta_t == "A":
            delta_t = delta_t_A
        elif delta_t == "B":
            delta_t = delta_t_B
        else:
            raise ValueError(
                f"Did not understand speification for delta_t {delta_t}"
            )

    # message(type(time_axis1), type(time_axis2))
    # message(time_axis1-time_axis2)
    # message(time_axis1==time_axis2)

    # Don't interpolate if the time axis are identical
    Interp = True

    if len(time_axis1) == len(time_axis2):
        if (time_axis1 == time_axis2).all():
            Interp = False
            time_axis = time_axis1
            wf1, wf2 = all_waveforms
            message("waveforms have a common time axis")

            # from scipy.stat import mode
            # tsorted = sorted(time_axis1)
            # delta_t = tsorted[0] - tsorted[1]

            # delta_t = mode(np.diff(time_axis1))

    if Interp is True:
        message("Interpolating time axis")
        time_axis, wf1, wf2 = resample_wfs(
            all_time_axes, all_waveforms, delta_t
        )
        # message(time_axis, wf1, wf2)

    from spectools.fourier.fft import compute_fft, compute_ifft
    from spectools.fourier.fft import compute_fft, compute_ifft

    # The Fspace waveforms
    faxis, wf1_tilde = compute_fft(wf1, delta_t)
    _, wf2_tilde = compute_fft(wf2, delta_t)
    delta_f = faxis[1] - faxis[0]
    # Compute the Fspace complex SNR
    csnr_tilde = wf1_tilde * np.conjugate(wf2_tilde)
    # Tspace CSNR
    _, csnr = compute_ifft(csnr_tilde, delta_f)
    # T space CSNR amp and phase
    Acsnr = np.absolute(csnr)

    # Compute shift quantities for waveform 2 against 1
    # Time shift
    Tshift_rec_index = np.argmax(Acsnr)
    # max_snr = np.amax(Acsnr)

    Tshift_rec = (len(time_axis) - Tshift_rec_index) * delta_t
    # Phase shift
    Pshift_rec = csnr[Tshift_rec_index] / np.absolute(csnr[Tshift_rec_index])
    Pshift_rec_rad = np.log(Pshift_rec) / (1j)

    message(
        "-----------------------------------\n Shift information for"
        "waveform 2 against 1 \n"
    )
    message(f"Recovered Time shift: {Tshift_rec}")
    message(f"Recovered Phase shift: {Pshift_rec}, {Pshift_rec_rad} in radians")
    message("-----------------------------------")

    # Apply the time shift to the second waveform

    if Tshift_rec_index != 0:
        wf2_Trec = roll(wf2, Tshift_rec_index)
    else:
        wf2_Trec = wf2

    if Pshift_rec != 1:
        _, wf2_Trec_tilde = compute_fft(wf2_Trec, delta_t)
        Pshift2_rad = 2 * np.pi - Pshift_rec_rad
        Pshift2_fac = np.exp(1j * Pshift2_rad)
        wf2_TPrec_tilde = wf2_Trec_tilde * Pshift2_fac

        _, wf2_TPrec = compute_ifft(wf2_TPrec_tilde, delta_f)

    else:
        # Check error!
        wf2_TPrec = wf2_Trec

    norm1 = np.sqrt(np.sum(wf1 * np.conjugate(wf1)))
    norm2 = np.sqrt(np.sum(wf2_TPrec * np.conjugate(wf2_TPrec)))

    waveform1_aligned = wf1 / norm1
    waveform2_aligned = wf2_TPrec / norm2

    match_score = np.sum(
        waveform1_aligned * np.conjugate(waveform2_aligned)
    )  # max_snr/(norm1*norm2)

    match_details = {
        "match_score": match_score,
        "time_shift": Tshift_rec,
        "phase_shift": Pshift_rec_rad,
        "time": time_axis,
        "aligned_waveforms": [waveform1_aligned, waveform2_aligned],
    }

    return match_details


def match_wfs_pycbc(all_time_axes, all_waveforms):
    """Match two waveforms using pycbc subroutines and return the time shift,
    phase shift, normalized waveforms and match coefficient.

    Parameters
    ----------
    time_axes : list
                A list containing the time axes
                of the two waveforms

    waveforms : list
                A list of two waveforms.
                Each is a 1d array.

    change : int
             Which waveform to change, 1 or 2.

    Returns
    -------
    match_details : dict
                    A dictionary containing the
                    i). match coeffient
                    ii). time_shift
                    iii). phase shift
                    iv). normalized waveforms and their time-axes.
    """

    # Step 1: resample
    from scipy.interpolate import interp1d

    # from waveformtools.waveformtools import match_wfs
    # from waveformtools.waveformtools import lengtheq

    waveform1, waveform2 = all_waveforms
    time_axis1, time_axis2 = all_time_axes

    message("Taxis limits")
    message(f"Shear tmin {min(time_axis1)} tmax {max(time_axis1)}")
    message(f"News tmin {min(time_axis2)} tmax {max(time_axis2)}")

    delta_t_1 = time_axis1[1] - time_axis1[0]
    delta_t = time_axis2[1] - time_axis2[0]

    wf1_amp, wf1_phase = xtract_camp_phase(waveform1.real, waveform1.imag)
    wf2_amp, wf2_phase = xtract_camp_phase(waveform2.real, waveform2.imag)

    wf1_amp_int_fun = interp1d(time_axis1, wf1_amp)
    wf1_phase_int_fun = interp1d(time_axis1, wf1_phase)

    wf1_amp_resam = wf1_amp_int_fun(time_axis2)
    wf1_phase_resam = wf1_phase_int_fun(time_axis2)

    delta_phase = wf1_phase_resam - wf2_phase

    from waveformtools.waveformtools import roll

    corrs = []
    for index in range(len(time_axis2)):
        rwf2 = roll(wf2_amp, index)

        corrs.append(np.dot(rwf2, wf1_amp_resam))

    # maxloc = np.argmax(corrs)
    shift = np.argmax(np.array(corrs))
    message(f"The shift units is {shift}")

    # ii). apply the time shift to the second waveform

    if shift != 0:
        # wf2_amp_shifted = roll(wf2_amp, shift)
        # wf2_phase_shifted = roll(wf2_phase, shift)
        wf2_shifted = roll(waveform2, shift)
    else:
        # wf2_amp_shifted = wf2_amp
        # wf2_phase_shifted = wf2_phase
        # wf2_shifted =  wf2_amp_shifted * np.exp(1j*wf2_phase_shifted)
        wf2_shifted = waveform2
        # waveform2

    mid = int(len(time_axis2) / 2)
    phase_shift = np.mean(delta_phase[mid - 100 : mid + 100])

    from spectools.fourier.fft import compute_fft, compute_ifft
    from spectools.fourier.fft import compute_fft, compute_ifft

    faxis0, wf1_tilde = compute_fft(waveform1, delta_t_1)

    delta_f = faxis0[1] - faxis0[0]

    wf1_tilde_phase_shifted = wf1_tilde * np.exp(1j * phase_shift)
    taxis0, wf1_aligned = compute_ifft(wf1_tilde_phase_shifted, delta_f)

    maxloc = np.argmax(np.absolute(wf1_aligned))
    taxis0 = taxis0 - taxis0[maxloc]

    start_time = min(time_axis2)
    end_time = max(time_axis2)

    start_ind = int((start_time - taxis0[0]) / delta_t)
    end_ind = int((end_time - taxis0[0]) / delta_t)

    wf1_aligned_cropped = wf1_aligned[start_ind:end_ind]

    aligned_time_axis = taxis0[start_ind:end_ind]

    norm1 = np.sum(wf1_aligned_cropped * np.conjugate(wf1_aligned_cropped))
    norm2 = np.sum(waveform2 * np.conjugate(waveform2))

    mlen = min(len(wf1_aligned_cropped), len(wf2_shifted))

    waveform1_aligned = wf1_aligned_cropped[:mlen] / norm1
    waveform2_aligned = wf2_shifted[:mlen] / norm2

    aligned_time_axis = aligned_time_axis[:mlen]
    match_score = np.dot(
        wf1_aligned_cropped[:mlen], np.conjugate(wf2_shifted[:mlen])
    ) / (norm1 * norm2)

    match_details = {
        "match_score": match_score,
        "time_shift": shift * delta_t,
        "phase_shift": phase_shift,
        "time": aligned_time_axis,
        "aligned_waveforms": [waveform1_aligned, waveform2_aligned],
    }

    return match_details


def simplematch_wfs_old(waveforms, delta_t=None):
    """Simple match the given waveforms.
    Does not clip the waveforms at either ends.

    Parameters
    ----------
    waveforms : list
                A list of pairs
                [waveform A, waveform B] of waveforms.
    delta_t : float, optional
              The time stepping.

    Notes
    -----
    The time stepping delta_t is the same
    for each pair of waveforms in the list.

    Returns
    -------
    match : list
            A list of dicts [{ Aligned waveforms} ,
            {match score (float), shift (number)}]
            containing the match information
            for all the input waveform pairs.
    """

    match = []
    # Iterate over (signal,template) pairs in waveforms
    for waveformdat in waveforms:
        # Carryout the match
        if not delta_t:
            try:
                delta_t = waveformdat[0].delta_t
            except BaseException:
                message(
                    "Waveform is not a pycbc TimeSeries."
                    "Please provide the gridspacing delt"
                )
                sys.exit(0)
        # Match procedure
        # signaldat = lengtheq(waveformdat[0], waveformdat[1], delta_t)
        waveform1, waveform2, _ = lengtheq(
            waveformdat[0], waveformdat[1], delta_t
        )

        # waveform1 = signaldat[0]
        # waveform2 = signaldat[1]

        # alignedwvs = pycbc.waveform.utils.coalign_waveforms(signaldat,hpa)
        # Compute the match to calculate match and shift.
        # Note: The match function from pycbc returns the match of the
        # normalized templates
        (match_score, shift) = pycbc.filter.matchedfilter.match(
            waveform1, waveform2
        )

        # Coalign the waveforms using pycbc coalign.
        waveform1, waveform2 = pycbc.waveform.utils.coalign_waveforms(
            waveform1, waveform2
        )

        # Normalize the waveforms
        norm1 = norm(waveform1)
        norm2 = norm(waveform2)
        waveform1 = waveform1 / norm1
        waveform2 = waveform2 / norm2

        try:
            (match_score, shift) = pycbc.filter.matchedfilter.match(
                waveform1, waveform2
            )
        except BaseException:
            message("Final match couldn't be found!")
            match_score = None
            shift = None

        match.append(
            {
                "Waveforms": [waveform1, waveform2],
                "Match score": match_score,
                "Shift": shift,
                "Norms": [norm1, norm2],
            }
        )

    return match


def pmmatch_wfs(waveforms, offset=25, crop=None):
    """
    Match function for post merger waveforms.

    Procedure
    ---------
    1. Equalize the lengths
    2. Crop the waveforms if necessary.
    3. Normalize to their respective maximum amplitudes.
    4. Align the waveforms in phase.
    5. Compute the match score.

    Parameters
    ----------
    waveforms : a list of pairs
                The pairs of waveforms to match
                in the format
                [[wf1_pair1, wf2_pair2], [wf1_pair_2, wf2_pair2], ...].
    offset : int
             Number of indices to shift the data.
    crop : string
           A string to decide how to crop the waveforms.
           The available Options are 1. `signal` 2. `template` 3. `both`.

    Returns
    -------
    matchdet : a list of dicts
               A list of dictionaries.
               Each contains
                1. the waveform pair,
                2. the match score,
                3. the shift index. to maximize the match.

    """
    matchdet = []
    for waveformdat in waveforms:
        signal, template = waveformdat

        # message(type(signal), type(template))
        # message(type(signal), type(template), len(signal), len(template))

        signal, template, _ = lengtheq(signal, template)

        # message(type(signal), type(template))

        # Crop the template
        if crop == "both":
            signal = signal[np.argmax(np.array(signal)) + offset :]
            template = template[np.argmax(np.array(template)) + offset :]

        if crop == "signal":
            signal = signal[np.argmax(np.array(signal)) + offset :]

        if crop == "template":
            template = template[np.argmax(np.array(template)) + offset :]

        # message(type(signal), type(template))
        # message(np.amax(np.array(template_plus)),
        # np.amax(np.array(signal_plus)))

        # Normalize the waveforms to their peak amplitudes.
        signal = signal / norm(np.array(signal))
        template = template / norm(np.array(template))

        try:
            delta_t = signal.delta_t
        except BaseException:
            delta_t = 1
            signal = pycbc.types.timeseries.TimeSeries(signal, delta_t)

        try:
            delta_t = template.delta_t
        except BaseException:
            delta_t = 1
            template = pycbc.types.timeseries.TimeSeries(template, delta_t)

        # message(type(signal), type(template))
        # Align the waveforms in phase
        signal_al, template_al = pycbc.waveform.utils.coalign_waveforms(
            signal, template
        )

        # message(np.where(np.array(signalp_al)!=0))

        # Compute the match score
        (matchscore, finalshift) = pycbc.filter.matchedfilter.match(
            signal_al, template_al
        )

        # message('+ The match score, shift are %f,
        # %d'%(matchscore, finalshift))
        matchdet.append(
            {
                "Waveforms": [signal_al, template_al],
                "Match score": matchscore,
                "Shift": finalshift,
            }
        )

    return matchdet


def roll(tsdata, i_roll, is_ts=False):
    """Roll the data circularly. Circular counterpart
    of shiftmatched function.

    Parameters
    ----------
    tsdata: 1d array or pycbc TimeSeries
            1D data vector in the form of
            a list/ numpy array or timeseries.
    i_roll: int
            The number of indices to roll the array.

    Returns
    -------
    rolled_waveform: 1d array or(pycbc TimeSeries object
                     The rolled wavefrom.
    """

    flag = 0
    if is_ts:
        try:
            # Assign the time step.
            delta_t = tsdata.delta_t
            flag = 1
        except BaseException:
            flag = 0

    # Assign the data array
    tsdata = np.array(tsdata)
    # Break the array into two parts as last i + first i entries.
    arr1 = tsdata[-i_roll:]
    arr2 = tsdata[:-i_roll]
    # Join the two arrays and return them
    if flag == 1:
        rolled_waveform = pycbc.types.timeseries.TimeSeries(
            np.transpose(
                np.concatenate((np.transpose(arr1), np.transpose(arr2)))
            ),
            delta_t,
        )

    else:
        rolled_waveform = np.transpose(
            np.concatenate((np.transpose(arr1), np.transpose(arr2)))
        )

    return rolled_waveform


def smoothen(func_x, win, order, xdata=None, to_plot=False):
    """Use the Savitzky-Golay Filter to smoothen the data.
        Show the plots if plot=`yes`.

    Parameters
    ----------
    func_x: 1d array
            The y axis.
    win: int
         Window for smoothening. Must be odd.
    order: int
           The order of the polynomial used for interpolation.
    x: 1d array, optional.
       The 1D list or numpy array, to plot the smoothened function.
       Only required if to_plot=True.
    to_plot: bool
             True or False. Whether or not to display the plot.

    Returns
    -------
    ydata: 1d array
           The Savgol filtered list.

    """

    # Apply the filter
    ydata = scipy.signal.savgol_filter(func_x, win, order)
    # Show plots
    if to_plot:
        plt.plot(xdata, func_x, label="data")
        plt.plot(xdata, ydata, label="smoothened data")
        plt.title("Smoothened data using Savitzky-Golay Filter")
        plt.grid(which="both", axis="both")
        plt.legend()
        plt.show()
    # Returns the filtered data
    return ydata


def bintp(xdata, func_x, width, order, to_plot=True):
    """Function to bin the data and interpolate it
    at specified width and order.

    Parameters
    ----------
    xdata :  1d array
             1D list or numpy array.
    func_x : 1d array
                         The y axis.
    width : int
                        Window size for smoothening,
    order : int
                        Order of the polynomial used for interpolation.
    to_plot : bool
                          True or False. To plot or not plot the results.

    Returns
    -------
    hist : list
           [binloc, yvals], The location of the bins and
                   the y values associated with the bins.

    """

    # Interpolation orders
    kind = [0, "linear", "quadratic", "cubic"]
    # Parse width
    width = int(width)
    # Number of bins
    nbins = int(len(xdata) / width)
    # Location of the bins
    binloc = [
        np.mean(xdata[width * index : width * index + width])
        for index in range(0, nbins + 1)
    ]
    # message(binloc)
    # Assigning y values to the bins
    yvals = [
        np.mean(func_x[width * index : width * index + width])
        for index in range(0, nbins + 1)
    ]
    # Assigning x values to the smoothened data
    # xf=x[width:-(width)/2]
    y_final = yvals
    # Interpolate if specified order is more than 0
    if order != 0:
        y_interp_func = scipy.interpolate.interp1d(
            binloc, yvals, kind=kind[order]
        )
    # Reassign yf
    y_final = y_interp_func(binloc)
    # Set xf to binloc if order=0
    # if order==0:
    #       xf=binloc
    # y = signal.savgol_filter(func_x,win,order)
    if to_plot:
        # Plot the filtered data
        plt.plot(xdata, func_x, label="data")
        plt.plot(binloc, y_final, label="smoothened data")
        plt.title("Smoothened data by binning and interpolation")
        plt.grid(which="both", axis="both")
        plt.legend()
        plt.show()
    hist = [binloc, y_final]
    # Returns a list consisting of bin loacations
    # and the correspnding y values
    return hist


def mavg(func_x, width):
    """Function to smoothen data. Moving average over the window width.

    Parameters
    ----------
    func_x : 1d array
             A list or numpy array of y axis.
    Width : int
                        The width of the moving average window.

    Returns
    -------
    func_x_avgd : 1d array
                                  1D array of moving averaged y axis.

    """

    # message(len(func_x))
    # List to store smoothened data
    func_x_avgd = []
    # Calculate the moving-average upto last but width num of points
    for j in range(0, len(func_x) - width):
        func_x_avgd.append(np.mean(func_x[j : width + j]))
    # Calculate the moving-averaged values for the last width num of points
    for j in range(len(func_x) - width, len(func_x)):
        func_x_avgd.append(np.mean(func_x[j:]))
    # Returns the list containing the moving-averaged values
    return func_x_avgd


# <Interpola


def interpolate_wfs(ts_data, interp_func, delta_t=None, **kwargs):
    """Function to interpolate a list of timeseries data using
    the user specified interp_func function and the keyword arguments.

    Parameters
    ----------
    ts_data: list
             The 1d data. A list of waveforms
             as a list or numpy array or
             pycbc TimeSeries.
    interp_fun: function
                An interpolating function.
    delta_t: float
             Timestep.
    ``**kwargs``: keyword arguments
                  additional arguments to the user specified interp_func.

    Returns
    -------
    interp_data: list
                 A list containing interpolated data.

    """

    # List for storing the interpolated data function
    interp_data = []
    # Loop over items in input
    for wfs in ts_data:
        if not delta_t:
            try:
                # Find sampling time_step
                delta_t = wfs.delta_t
                timeaxis = wfs.sample_times

            except BaseException:
                message(
                    "Input is not a TimeSeries."
                    "Please supply gridspacing as delta_t",
                    message_verbosity=0,
                )
        else:
            timeaxis = np.arange(0, len(wfs) * delta_t, delta_t)

        # Interpolate using the supplied function. The keyword arguments
        # supplied to the fuction are the keyword arguments to be supplied
        # to the interpolating function)
        # Append to the list of interpolated data function
        interp_data.append(interp_func(timeaxis, np.array(wfs), **kwargs))
    # Return the interpolated data function list
    return interp_data


def resample(interp_data, new_delta_t, epoch, length, old_delta_t=None):
    """Function to generate timeseries out of the given interpolated data
    function, epoch,sampling frequency, length(duration).

    Parameters
    ----------
    interp_data : 1d array
                                  The yaxis to be interpolated.
    epoch : float
                        The starting point in time.
    delta_t : float
                          New grid spacing to be sampled at.
    length : int
             The duration of x axis.

    Returns
    -------

    data : list
                   A list containing resampled data as pycbc TimeSeries.
    """

    data = []
    # Loop over objects in interp_data
    for i in range(len(interp_data)):
        if not old_delta_t:
            try:
                old_delta_t = interp_data[i].delta_t

            except BaseException:
                message(
                    "Input is not a TimeSeries."
                    "Please supply gridspacing as delta_t",
                    message_verbosity=0,
                )
        else:
            interp_data[i] = pycbc.types.timeseries.TimeSeries(
                interp_data[i], old_delta_t
            )

        # Prepare timeaxis
        timeaxis = np.linspace(epoch, epoch + length, int(length / new_delta_t))
        # Append the timeseries to the data list
        ydata = interp_data[i](timeaxis)
        data.append(
            pycbc.types.timeseries.TimeSeries(ydata, new_delta_t, epoch=epoch)
        )
    # Return the list of samples timeseries
    return data


def interp_resam_wfs(wavf_data, old_taxis, new_taxis, kind="cubic", k=None):
    """Wrapper function for interpolation and resampling.

    Parameters
    ----------
    wavf_data : 1d array
                The yaxis to be interpolated,
    old_taxis, new_taxis : 1darray
                                                   Old and New time axis.

    Returns
    -------
    resam_wf_data : 1d array
                    Interpolated and resampled data.
    """
    # from scipy.interpolate import interp1d

    amp, phase = xtract_camp_phase(wavf_data.real, wavf_data.imag)

    # Interpolate
    if k is not None:
        from scipy.interpolate import (
            InterpolatedUnivariateSpline as interpolator,
        )

        interp_amp_data = interpolator(old_taxis, amp, k=k)
        amp_res = interp_amp_data.get_residual()
        interp_phase_data = interpolator(old_taxis, phase, k=k)
        phase_res = interp_phase_data.get_residual()
        message(
            f"Amplitude residue {amp_res} \t Phase residue {phase_res}",
            message_verbosity=2,
        )
    else:
        from scipy.interpolate import interp1d as interpolator

        interp_amp_data = interpolator(old_taxis, amp, kind=kind)
        interp_phase_data = interpolator(old_taxis, phase, kind=kind)

    # Resample
    resam_amp_data = interp_amp_data(new_taxis)
    resam_phase_data = interp_phase_data(new_taxis)

    resam_wf_data = resam_amp_data * np.exp(1j * resam_phase_data)

    return resam_wf_data


def progressbar(present_count, total_counts, normalize="yes"):
    """Display the progress bar to std out
    from present_count and total_count.

    Parameters
    ----------
    present_count : int
                    The present count state.
    total_counts : int
                   The final state.

    Returns
    -------

    1 : int
        The progress bar is messageed to stdout.
    """

    if normalize == "yes":
        final_progress = 98
        normalized_total_counts = final_progress * 10
        present_count = int(
            normalized_total_counts * present_count / total_counts
        )
        total_counts = normalized_total_counts

    # present_count = comm.gather(count,root=rank)
    else:
        final_progress = int(total_counts / 10)

    present_stage = int(present_count / 10)
    # message(present_stage)
    # message(total_counts)
    if present_stage != 0:
        if present_count == total_counts:
            present_semi_progress = "#"
        else:
            present_semi_progress = present_count % 10
    else:
        present_semi_progress = present_count
    offset = final_progress - present_stage
    per_cent_progress = 100 * present_count / total_counts

    sys.stdout.write(
        "\r"
        + "Progress:|"
        + "#" * present_stage
        + "%s" % present_semi_progress
        + " " * (offset)
        + "|"
        + "%.f%%" % per_cent_progress
    )
    sys.stdout.flush()
    return 1


def get_val_at_t_ref(time_axis, val_axis, time):
    """Interpolate and get the value at the requested time"""

    from scipy.interpolate import interp1d

    int_func = interp1d(time_axis, val_axis, kind="cubic")
    val_at_t_ref = float(int_func(time))
    # print(val_at_t_ref, type(val_at_t_ref))

    return round(val_at_t_ref, 5)
def get_nr_frame_angles_from_lal(inclination, phi_ref, tol=1e-3):
    ''' Convert the lalframe angles (inclination, phi_ref)
    to NR frame (theta, phi/psi, alpha) '''

    orb_phase = phi_ref
    #inclination = ref_params['inclination']

    # Define the LAL source frame vectors
    ln_hat_x = 0
    ln_hat_y = 0
    ln_hat_z = 1

    n_hat_x = 1
    n_hat_y = 0
    n_hat_z = 0

    ln_hat = np.array([ln_hat_x, ln_hat_y, ln_hat_z])
    n_hat = np.array([n_hat_x, n_hat_y, n_hat_z])

    # 2.3: Carryout vector math to get Zref in the lal wave frame
    corb_phase = np.cos(orb_phase)
    sorb_phase = np.sin(orb_phase)
    sinclination = np.sin(inclination)
    cinclination = np.cos(inclination)

    ln_cross_n = np.cross(ln_hat, n_hat)
    ln_cross_n_x, ln_cross_n_y, ln_cross_n_z = ln_cross_n

    z_wave_x = sinclination * (sorb_phase * n_hat_x + corb_phase * ln_cross_n_x)
    z_wave_y = sinclination * (sorb_phase * n_hat_y + corb_phase * ln_cross_n_y)
    z_wave_z = sinclination * (sorb_phase * n_hat_z + corb_phase * ln_cross_n_z)

    z_wave_x += cinclination * ln_hat_x
    z_wave_y += cinclination * ln_hat_y
    z_wave_z += cinclination * ln_hat_z

    z_wave = np.array([z_wave_x, z_wave_y, z_wave_z])
    z_wave = z_wave / np.linalg.norm(z_wave)

    #################################################################
    # Step 3.1: Extract theta and psi from Z in the lal wave frame
    # NOTE: Theta can only run between 0 and pi, so no problem with arccos here
    theta = np.arccos(z_wave_z)

    # Degenerate if Z_wave[2] == 1. In this case just choose psi randomly,
    # the choice will be cancelled out by alpha correction (I hope!)

    # If theta is very close to the poles
    # return a random value
    if abs(z_wave_z - 1.0) < tol:
        psi = 0.5

    else:
        # psi can run between 0 and 2pi, but only one solution works for x and y */
        # Possible numerical issues if z_wave_x = sin(theta) */
        if abs(z_wave_x / np.sin(theta)) > 1.0:
            if abs(z_wave_x / np.sin(theta)) < (1 + 10 * tol):
                # LAL tol retained.
                if (z_wave_x * np.sin(theta)) < 0.0:
                    psi = np.pi

                else:
                    psi = 0.0

            else:
                print(f"z_wave_x = {z_wave_x}")
                print(f"sin(theta) = {np.sin(theta)}")
                raise ValueError(
                    "Z_x cannot be bigger than sin(theta). Please contact the developers."
                )

        else:
            psi = np.arccos(z_wave_x / np.sin(theta))

        y_val = np.sin(psi) * np.sin(theta)

        # If z_wave[1] is negative, flip psi so that sin(psi) goes negative
        # while preserving cos(psi) */
        if z_wave_y < 0.0:
            psi = 2 * np.pi - psi
            y_val = np.sin(psi) * np.sin(theta)

        if abs(y_val - z_wave_y) > (5e3 * tol):
            # LAL tol retained.
            print(f"orb_phase = {orb_phase}")
            print(
                f"y_val = {y_val}, z_wave_y = {z_wave_y}, abs(y_val - z_wave_y) = {abs(y_val - z_wave_y)}"
            )
            raise ValueError("Math consistency failure! Please contact the developers.")

    # 3.2: Compute the vectors theta_hat and psi_hat
    # stheta = np.sin(theta)
    # ctheta = np.cos(theta)

    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    # theta_hat_x = cpsi * ctheta
    # theta_hat_y = spsi * ctheta
    # theta_hat_z = -stheta
    # theta_hat = np.array([theta_hat_x, theta_hat_y, theta_hat_z])

    psi_hat_x = -spsi
    psi_hat_y = cpsi
    psi_hat_z = 0.0
    psi_hat = np.array([psi_hat_x, psi_hat_y, psi_hat_z])

    # Step 4: Compute sin(alpha) and cos(alpha)
    # Rotation angles on the tangent plane
    # due to spin weight.

    # n_dot_theta = np.dot(n_hat, theta_hat)
    # ln_cross_n_dot_theta = np.dot(ln_cross_n, theta_hat)

    n_dot_psi = np.dot(n_hat, psi_hat)
    ln_cross_n_dot_psi = np.dot(ln_cross_n, psi_hat)

    # salpha = corb_phase * n_dot_theta - sorb_phase * ln_cross_n_dot_theta
    calpha = corb_phase * n_dot_psi - sorb_phase * ln_cross_n_dot_psi

    if abs(calpha) > 1:
        calpha_err = abs(calpha) - 1
        if calpha_err < tol:
            # This tol could have been much smaller.
            # Just resuing the default for now.
            print(
                f"Correcting the polarization angle for finite precision error {calpha_err}"
            )
            calpha = calpha / abs(calpha)
        else:
            raise ValueError(
                "Seems like something is wrong with the polarization angle. Please contact the developers!"
            )

    alpha = np.arccos(calpha)

    angles = {
        "theta": theta,
        "psi": psi,
        "alpha": alpha,
    }

    return angles



def find_maxloc_and_time(times, amp):

    from scipy.interpolate import interp1d

    amp_interp = interp1d(times, amp, kind='cubic')
    times_fine = np.linspace(times[0], times[-1], len(times)*1000)
    amp_fine = amp_interp(times_fine)

    maxloc = np.argmax(amp_fine)
    maxloc_0 = np.argmax(amp)
    t_maxloc = times_fine[maxloc]

    return t_maxloc, maxloc, maxloc_0

def load_lal_modes_to_modes_array(lal_modes, domain='fd'):
    ''' '''
    from waveformtools.modes_array import ModesArray

    nm = lal_modes
    ell_max = nm.l

    if domain=='fd':
        wfm = ModesArray(label=f'lal_{domain}',
                        ell_max=ell_max,
                        frequency_axis=lal_modes.fdata.data,
                        )
        print(len(lal_modes.fdata.data))
    if domain=='td':
        wfm = ModesArray(label=f'lal_{domain}',
                    ell_max=ell_max,
                    time_axis=lal_modes.tdata.data,
                    )
        print(len(lal_modes.tdata.data))

    wfm.create_modes_array()
    print(wfm.modes_data.shape)

    ell = ell_max
    emm = ell_max

    while ell!=2 and emm!=-2:
        ell = nm.l
        emm = nm.m
        print(ell, emm, nm.mode.data)
        wfm.set_mode_data(ell=ell, emm=emm, data=nm.mode.data.data)
        nm =  nm.next


    return wfm