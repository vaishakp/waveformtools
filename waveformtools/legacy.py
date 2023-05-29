""" Old codes and methods """
import sys

import numpy as np
import pycbc

from waveformtools.waveformtools import (
    apxstartend,
    lengtheq,
    message,
    roll,
    shiftmatched,
    startend,
)


def coalignwfs(tsdata1, tsdata2):
    """Coalign two timeseries. Adjust length of either
    waveforms if needed. Compute The complex SNR. Shift
    and roll the first. Returns normalized waveforms
    """

    len1 = len(tsdata1)
    len2 = len(tsdata2)
    if len2 > len1:
        tsdata1, lflag = lengtheq(tsdata2, tsdata1, tsdata1.delta_t)
    elif len1 > len2:
        tsdata2, lflag = lengtheq(tsdata2, tsdata1, tsdata1.delta_t)
    csnr = pycbc.filter.matchedfilter.matched_filter(tsdata1, tsdata2)
    acsnr = np.array(np.abs(csnr))
    # message(acsnr,np.max(acsnr))
    maxloc = (np.where(acsnr == np.max(acsnr)))[0][0]
    message("Max location is %s, match is %s" % (maxloc, np.max(acsnr)))
    tsdata1 = shiftmatched(tsdata1, maxloc, tsdata1.delta_t)
    rotation = csnr[maxloc] / np.absolute(csnr[maxloc])
    ctsdata1 = (rotation * tsdata1.to_frequencyseries()).to_timeseries()
    return [ctsdata1, tsdata2]


def match_wfs_pycbc_old(waveforms, delt=None):
    """Match given waveforms. Find the overlap.

    Procedure
    ---------
    1. For each pair of waveforms as a list:
                            a. Findout if delt has been specified.
                            b. Findout if the object has attribute delta_t
                            to discern whether it is a pycbc timeseries
                            ( not exactly. ). If not then exit.
                            c. Equalize the lengths.
                            d. Compute the match score and shift using the pycbc shift function.
                            e. Findout the start and the end of the waveform using handles.startend.
                            f. Reconstruct normalized and clipped pycbc.timeseries of the waveforms.
                            g. Confirm the equalization of the lengths of the waveoforms.
                            h. Append the match details to an array
                            [ waveform list, [ match score, shift, start_index, end_index]]
    2. Retun the match details for all the waveforms.


    Parameters
    ----------
    waveforms:  list
                                                    A List of pairs [waveform A, waveform B].

    Notes
    -----
    Assumes that the sampling rate is the  same for each pair.

    Returns
    -------
    match:  a list of dicts
                                    A list of dictionaries in the format
                                    {match score (float), shift (number), start_index, end_index}
    """
    # Iterate over (signal,template) pairs in waveforms
    for waveformdat in waveforms:
        # Carryout the match
        if not delt:
            try:
                delt = waveformdat[0].delta_t
            except BaseException:
                message("Waveform is not a pycbc TimeSeries. Please provide the gridspacing delt")
                sys.exit(0)
        # Match procedure

        # signaldat = lengtheq(waveformdat[0], waveformdat[1], delt)

        # waveform1 = signaldat[0]
        # waveform2 = signaldat[1]
        waveform1, waveform2, _ = lengtheq(waveformdat[0], waveformdat[1], delt, is_ts=True)

        # alignedwvs = pycbc.waveform.utils.coalign_waveforms(signaldat,hpa)
        # Compute the match to calculate match and shift.
        # Note: The match function from pycbc returns the match of the
        # normalized templates

        (match_score, shift) = pycbc.filter.matchedfilter.match(waveform1, waveform2)

        message("Priliminary match", match_score, shift)
        # Shift the matched data against the template using the shift obtained
        # above
        waveform1 = roll(np.array(waveform1), int(shift))
        # waveform1 = shiftmatched(np.array(waveform1), int(shift), delt)
        # Compute the start and end of the non-zero signal
        # First try with absolute startend. Then with approximate startend.
        # Note: The criterion that handles.startend() uses is that the signal
        # exists in non-zero portion of the data.
        try:
            starti, endi = startend(waveform1)
        except BaseException:
            message("Absolute startend not found. Fixing approximate startend")
        # Match procedure

        # signaldat = lengtheq(waveformdat[0], waveformdat[1], delt)

        # waveform1 = signaldat[0]
        # waveform2 = signaldat[1]
        waveform1, waveform2, _ = lengtheq(waveformdat[0], waveformdat[1], delt, is_ts=True)

        # alignedwvs = pycbc.waveform.utils.coalign_waveforms(signaldat,hpa)
        # Compute the match to calculate match and shift.
        # Note: The match function from pycbc returns the match of the
        # normalized templates

        (match_score, shift) = pycbc.filter.matchedfilter.match(waveform1, waveform2)

        message("Priliminary match", match_score, shift)
        # Shift the matched data against the template using the shift obtained
        # above
        waveform1 = roll(np.array(waveform1), int(shift))
        # waveform1 = shiftmatched(np.array(waveform1), int(shift), delt)
        # Compute the start and end of the non-zero signal
        # First try with absolute startend. Then with approximate startend.
        # Note: The criterion that handles.startend() uses is that the signal
        # exists in non-zero portion of the data.
        try:
            starti, endi = startend(waveform1)
        except BaseException:
            message("Absolute startend not found. Fixing approximate startend")
            starti, endi = apxstartend(waveform1)
            message("starti, endi")

        message("The approximate start and end indices are", starti, endi, "length", endi - starti)
        # Convert the non-zero portion of the signal and template to
        # time-series
        message("Converting shifted vectors to time series")
        signal = pycbc.types.timeseries.TimeSeries(
            np.array(waveform1)[starti:endi] / np.linalg.norm(np.array(waveform1)[starti:endi]), delt
        )
        template = pycbc.types.timeseries.TimeSeries(
            np.array(waveform2)[starti:endi] / np.linalg.norm(np.array(waveform2)[starti:endi]), delt
        )
        # Sanity check: The template and the signal must be of the same length
        # at this point in execution
        if len(signal) != len(template):
            message("Error\n")
            message("Length of data, template after truncation are %d,%d" % (len(signal), len(template)))
            sys.exit(0)
        # Compute the match, shift again on the truncated data
        # message("length of data %d, aligned data %d, template %d"%(len(signaldat),len(alignedwvs[0]),len(alignedwvs[1])))

        try:
            (match_score, shift) = pycbc.filter.matchedfilter.match(signal, template)
        except BaseException:
            message("Final match couldn't be found!")
            match_score = None
            shift = None

        match = {"Match score": match_score, "Shift": shift, "Start index": starti, "End index": endi}
    return match


def iscontinuous_old(timeaxis, delta_t=0):
    """Check if the data has discontinuities. This checks for repetitive time rows and jumps.

    Notes
    -----
    Types of discontunuities

    0: Continuous.
    1: Repetitive rows.
    2: Jumps in timeaxis.

    Parameters
    ----------

    timeaxis:   list
                            Input as a single 1d time axis
                            or a list of 1d arrays [time, data1, data2, ...].
                            All the data share the common time axis `time`
    delta_t:    float
                            The time stepping.
    toldt : float
                    The tolerance for error in checking.
                    Defaluts to toldt=1e-3.

    Returns
    -------

    discontinuity_details : a list.
                                                    It contains: [ the actual location of discontinuity along
                                                    the time axis,
                                                    value of time location of original array,
                                                    the type of discontinuity].

    """

    # If data array is supplied, assign first column as timeaxis
    if np.array(timeaxis).ndim > 1:
        timeaxis = timeaxis[:, 0]
    # Check data for continuity.
    # If not timeseries
    if delta_t == 0:
        delta_t = timeaxis[1] - timeaxis[0]
    # Set epoch to first element of timeaxis
    epoch = timeaxis[0]
    # List to hold discintinuity details
    discontinuity_details = []
    # List to hold indices
    # indices = []
    # Set start index,epoch_index to 0
    # index = 0
    epoch_index = 0
    # List to hold discontinuity type
    discontinuity_type = []

    for timestamp in timeaxis:
        # Iterate over every timestamp in timeaxis
        # Check for discontinuity
        if timestamp != timeaxis[0] + epoch_index * delta_t:
            # Check for type of discontinuity
            # Repetitive rows
            if timestamp < timeaxis[0] + epoch_index * delta_t:
                discontinuity_type = 1
            # Missing rows
            elif epoch > timeaxis[0] + epoch_index + delta_t:
                discontinuity_type = 2
            # Append [index location,correct timestamp
            discontinuity_details.append([epoch_index, timeaxis[epoch_index], discontinuity_type])
        epoch_index += 1
        # message("Progress: %f%%\r"%(epoch_index*100./len(timeaxis)))

    # Print the result
    # changed discontinuity_timestamps to discontinuity details May 5 2022
    if len(discontinuity_details) != 0:
        message("The data is discontinuous!")
    else:
        message("The data is continuous!")
    # Return the details of discontinuity
    return discontinuity_details

def cleandata_old(data, verbose=False):
    """Old version. Check the data (time,datar,datai) for repetetive rows and remove them.

    Parameters
    ----------

    data:   list
                                                                    Input as a list of 1d arrays [time, data1, data2, ...].
                                                                    All the data share the common time axis `time`
    verbose : bool
                                                                    A verbosity flag.

    Returns
    -------

    cleaned_data:   nd array
                                    The cleaned data array with repetitive
                                    rows and gaps (if bridge=True) removed
    """

    # Ensure data as numpy array
    data = np.array(data)
    if verbose == "yes":
        message("Data shape:", data.shape)
    # Set axis along which to remove
    axis = data.ndim - 1
    if verbose == "yes":
        message("Axis:", axis)
    # Associate data[0] as timeaxis
    if axis > 0:
        shapes = data.shape
        if shapes[0] > shapes[1]:
            data = np.transpose(data)
        time = data[0, :]
        if verbose == "yes":
            message("The time array:", time)
    else:
        time = data
    # Assign delta_t
    # delta_t = statistics.mode(np.diff(time))
    delta_t = mode(np.diff(time))

    if verbose:
        message("length,shape of data", len(data), data.shape)
    # Reassign data without time_array
    # data=data[:,1:]
    # message("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    # message("Checking data for repetative rows...\n")
    # Index of ros to delete
    dind = []
    # Initial data length
    ki_index = len(time)
    # message("length of old array is %d\n" %ki_index)
    # Row iteration variable
    ii_index = 0
    # Flag to identify if any repetetive rows were found
    rep_row_index = 0
    # ci=0
    # rowcounter=np.zeros([])
    # Iterate over rows
    while ii_index < ki_index - 1:
        # Repetition condition :if the successive time stamp is less than or
        # equal to the present, delete the row.
        if (time[ii_index + 1] - time[ii_index]) <= -0.1 * delta_t:
            # Set flag to 1 if repetition condition is met.
            rep_row_index = 1
            if verbose == "yes":
                message("found a repeating row at %d, time %f\n" % (ii_index + 1, time[ii_index + 1]))
                message("timei: %f timef %f\n" % (time[ii_index], time[ii_index + 1]))
            # ci = ci+1
            # Delete the entire row
            time = np.delete(time, ii_index + 1)
            # data = np.delete(data,ii_index,1)
            # data = [np.delete(item,ii_index+1) for item in data]
            data = np.delete(data, ii_index + 1, axis)
            # Append the deleted index to the bookkeeping array
            dind.append(ii_index + 1)
            # If a row is deleted, step back the iter variable by one step
            ii_index = ii_index - 1
        # Advance the iter variable
        ii_index = ii_index + 1
        # Recalculate the array length
        ki_index = len(time)
    if verbose == "yes":
        if rep_row_index == 1:
            message("No. of points removed = %d\n" % len(dind))
            message("length of new array is %d\n" % len(time))
        else:
            message("No points removed\n")
        # message("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    # Return the "cleaned" data matrix
    cleaned_data = data
    # cleaned_data=[time]
    # for item in data:
    #    cleaned_data.append(item)
    # cleaned_data.append(item for item in data)
    # Transpose the data back to original shape
    if shapes[0] > shapes[1]:
        cleaned_data = np.transpose(cleaned_data)
    return cleaned_data
