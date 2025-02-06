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

                d. Compute the match score and shift using
                    the pycbc shift function.

                e. Findout the start and the end of the waveform
                    using handles.startend.

                f. Reconstruct normalized and clipped
                    pycbc.timeseries of the waveforms.

                g. Confirm the equalization of the lengths of
                    the waveoforms.
                h. Append the match details to an array
                [ waveform list, [ match score, shift, start_index, end_index]]
    2. Retun the match details for all the waveforms.


    Parameters
    ----------
    waveforms:	list
                A List of pairs [waveform A, waveform B].

    Notes
    -----
    Assumes that the sampling rate is the  same for each pair.

    Returns
    -------
    match:	a list of dicts
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
                message(
                    "Waveform is not a pycbc TimeSeries."
                    "Please provide the gridspacing delt"
                )
                sys.exit(0)
        # Match procedure

        # signaldat = lengtheq(waveformdat[0], waveformdat[1], delt)

        # waveform1 = signaldat[0]
        # waveform2 = signaldat[1]
        waveform1, waveform2, _ = lengtheq(
            waveformdat[0], waveformdat[1], delt, is_ts=True
        )

        # alignedwvs = pycbc.waveform.utils.coalign_waveforms(signaldat,hpa)
        # Compute the match to calculate match and shift.
        # Note: The match function from pycbc returns the match of the
        # normalized templates

        (match_score, shift) = pycbc.filter.matchedfilter.match(
            waveform1, waveform2
        )

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
        waveform1, waveform2, _ = lengtheq(
            waveformdat[0], waveformdat[1], delt, is_ts=True
        )

        # alignedwvs = pycbc.waveform.utils.coalign_waveforms(signaldat,hpa)
        # Compute the match to calculate match and shift.
        # Note: The match function from pycbc returns the match of the
        # normalized templates

        (match_score, shift) = pycbc.filter.matchedfilter.match(
            waveform1, waveform2
        )

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

        message(
            "The approximate start and end indices are",
            starti,
            endi,
            "length",
            endi - starti,
        )
        # Convert the non-zero portion of the signal and template to
        # time-series
        message("Converting shifted vectors to time series")
        signal = pycbc.types.timeseries.TimeSeries(
            np.array(waveform1)[starti:endi]
            / np.linalg.norm(np.array(waveform1)[starti:endi]),
            delt,
        )
        template = pycbc.types.timeseries.TimeSeries(
            np.array(waveform2)[starti:endi]
            / np.linalg.norm(np.array(waveform2)[starti:endi]),
            delt,
        )
        # Sanity check: The template and the signal must be of the same length
        # at this point in execution
        if len(signal) != len(template):
            message("Error\n")
            message(
                "Length of data, template after truncation are %d,%d"
                % (len(signal), len(template))
            )
            sys.exit(0)
        # Compute the match, shift again on the truncated data
        # message("length of data %d, aligned data %d, template %d",
        # (len(signaldat),len(alignedwvs[0]),len(alignedwvs[1])))

        try:
            (match_score, shift) = pycbc.filter.matchedfilter.match(
                signal, template
            )
        except BaseException:
            message("Final match couldn't be found!")
            match_score = None
            shift = None

        match = {
            "Match score": match_score,
            "Shift": shift,
            "Start index": starti,
            "End index": endi,
        }
    return match


def iscontinuous_old(timeaxis, delta_t=0):
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
    timeaxis:	list
                Input as a single 1d time axis
                or a list of 1d arrays [time, data1, data2, ...].
                All the data share the common time axis `time`
    delta_t:	float
                The time stepping.
    toldt : float
            The tolerance for error in checking.
            Defaluts to toldt=1e-3.

    Returns
    -------
    discontinuity_details : a list.
                            It contains: [ the actual location of
                            discontinuity along the time axis,
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
            discontinuity_details.append(
                [epoch_index, timeaxis[epoch_index], discontinuity_type]
            )
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
    """Old version. Check the data (time,datar,datai)
    for repetetive rows and remove them.

    Parameters
    ----------
    data:	list
            Input as a list of 1d arrays [time, data1, data2, ...].
            All the data share the common time axis `time`
    verbose:   bool
                A verbosity flag.

    Returns
    -------
    cleaned_data:	ndarray
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
    from scipy.stats import modes as stat_mode

    delta_t = stat_mode(np.diff(time))

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
                message(
                    "found a repeating row at %d, time %f\n"
                    % (ii_index + 1, time[ii_index + 1])
                )
                message(
                    "timei: %f timef %f\n"
                    % (time[ii_index], time[ii_index + 1])
                )
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
    # 	 cleaned_data.append(item)
    # cleaned_data.append(item for item in data)
    # Transpose the data back to original shape
    if shapes[0] > shapes[1]:
        cleaned_data = np.transpose(cleaned_data)
    return cleaned_data


def cleandata(data, toldt=1e-3, bridge="no"):
    """Check the data (time,datar,datai) for repetetive rows and remove them.

    Parameters
    ----------
    data:	list
            Input as a list of 1d arrays [time, data1, data2, ...].
            All the data share the common time axis `time`
    toldt : float
            The tolerance for error in checking. defaluts to toldt=1e-3.
    bridge : bool
            A bridge flag to decide whether or not to interpolate and
            resample to fill in jump discontinuities.

    Returns
    -------
    cleaned_data:	list
                    The cleaned data array with repetitive
                    rows and gaps (if bridge=True) removed.
    """

    # Check the data (time,datar,datai) for repetetive rows and remove them.
    # Ensure data as numpy array
    data = np.array(data)
    # message("Data shape:", (data.shape), message_verbosity=3)
    # Set axis along which to remove
    axis = data.ndim - 1
    message("Axis:%d" % axis, message_verbosity=3)

    # Associate data[0] as timeaxis
    if axis > 0:
        shapes = data.shape
        if shapes[0] > shapes[1]:
            data = np.transpose(data)
        time = data[0, :]
        message("The time array:%s" % time, message_verbosity=3)
    else:
        time = data

    # delta_t = statistics.mode(np.diff(time))
    from scipy.stats import mode as stat_mode

    delta_t = stat_mode(np.diff(time))

    message("shape of data:", (data.shape), message_verbosity=3)

    # Reassign data without time_array
    # data=data[:,1:]
    # message("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    # message("Checking data for repetative rows...\n")
    # Index of ros to delete
    dind = []

    # Initial data length
    ki_index = len(time)

    # message("length of old array is %d\n" %ki)
    # Row iteration variable
    ii_index = 0

    # Flag to identify if any repetetive rows were found
    rep_row_index = 0

    # ci=0
    # rowcounter
    counter = 0

    # Iterate over rows
    while ii_index < ki_index - 1:
        # Repetition condition :if the successive time stamp is less than or
        # equal to the present, delete the row.
        if (time[ii_index + 1] == time[ii_index]) or (
            time[ii_index] - time[ii_index + 1]
        ) >= toldt * delta_t:
            # if time[ii_index]-time[ii_index+1]<=0.01*delta_t:
            # 		 message("Error!!",message_verbosity=0)
            # Set flag to 1 if repetition condition is met.
            rep_row_index = 1
            counter += 1
            message(
                "found a repeating row at %d, time %f\n"
                % (ii_index + 1, time[ii_index + 1]),
                message_verbosity=3,
            )
            message(
                "timei: %f timef %f\n" % (time[ii_index], time[ii_index + 1]),
                message_verbosity=3,
            )
            # ci = ci+1
            # Delete the entire row
            time = np.delete(time, ii_index + 1, 0)
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

    if rep_row_index == 1:
        message("Repetitive rows were removed", message_verbosity=2)
        message("No. of rows removed = %d\n" % counter, message_verbosity=2)
        message("Length of new array is %d\n" % len(time), message_verbosity=3)

    else:
        message("No points removed\n", message_verbosity=2)
        # message("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    # Return the "cleaned" data matrix
    cleaned_data = data

    if bridge and iscontinuous(cleaned_data)[-1] >= 2:
        message(
            "The data will be interpolated to bridge the gaps",
            message_verbosity=2,
        )

        # from scipy import interpolate

        # Interpolate the data to fill in the discontinuities
        t_final = time[-1]
        t_initial = time[0]

        # Find delta_t
        delta_t = time[1] - time[0]
        index = 1

        # If second row is repetitive (If its discontinuous, help!)
        while delta_t <= 0:
            delta_t = np.diff(time)[index]
            index += 1
        proper_timeaxis = np.arange(t_initial, t_final, delta_t)
        interp_data = []
        interp_data.append(proper_timeaxis)
        if axis > 0:
            from scipy.interpolate import interp1d

            for index in range(1, min(shapes[0], shapes[1])):
                interp_datai = interp1d(time, data[index, :])
                interp_data.append(interp_datai(proper_timeaxis))

        cleaned_data = np.array(interp_data)
        message("The data has been interpolated", message_verbosity=2)
    message("Cleaned!", message_verbosity=2)

    # cleaned_data=[time]
    # for item in data:
    # 	 cleaned_data.append(item)
    # cleaned_data.append(item for item in data)
    # Transpose the data back to original shape

    if axis > 0:
        if shapes[0] > shapes[1]:
            cleaned_data = np.transpose(cleaned_data)
    return cleaned_data


def iscontinuous(data, delta_t=0, toldt=1e-3):
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
    data:	list
            Input as a list of 1d arrays [time, data1, data2, ...].
            All the data share the common time axis `time`
    delta_t:	float
                The time stepping.
    toldt : float
            The tolerance for error in checking. defaluts to toldt=1e-3.

    Returns
    -------
    discontinuity_details:	a list.
                            It contains:
                                1. A list. details of discontinuity.
                                    index location of original array,
                                    the corresponding discinbtinuity type.
                                2. A float. the global discontinuity type.
    """

    # Check the data (time,datar,datai) for locations of repetition and
    # discontinuities.
    message("Checking continuity of data", message_verbosity=3)
    # Ensure data as numpy array
    data = np.array(data)
    message("Data shape:", (data.shape), message_verbosity=3)
    # Find if data contanins more than timeaxis
    axis = data.ndim - 1
    message("Axis:%d" % axis, message_verbosity=3)
    # Associate data[0] as timeaxis
    if axis > 0:
        shapes = data.shape
        if shapes[0] > shapes[1]:
            data = np.transpose(data)
        timeaxis = data[0, :]
        message("The time axis is :%s" % timeaxis, message_verbosity=3)
    else:
        timeaxis = data
    message("shape of data:", (data.shape), message_verbosity=3)

    # If data array is supplied, assign first column as timeaxis
    # if np.array(timeaxis).ndim>1:
    # 		 timeaxis=np.array([item[0] for item in timeaxis])
    # message('Timeaxis`,timeaxis,message_verbosity=3)
    # Check data for continuity.
    # If delta_t is not supplied

    if delta_t == 0:
        # delta_t = statistics.mode(np.diff(timeaxis))
        import scipy.stats.mode as stat_mode

        delta_t = stat_mode(np.diff(timeaxis))

    # Set epoch to first element of timeaxis
    epoch = timeaxis[0]
    # List to hold discintinuity details
    discontinuity_details = []
    # List to hold indices
    # indices = []
    # Initialize start index,epoch_index to 0
    index = 0
    # Initialize epoch to start of timeaxis
    epoch = timeaxis[0]
    # Initialize epoch_index to 0
    epoch_index = 0
    # Repetition flag
    repetition = 0
    # Discountinuity flag
    discont = 0
    # Discontinuity type
    discont_type = 0
    # List to hold discontinuity type
    discont_type_details = []
    # Counters
    repetition_counter = 0
    discont_counter = 0
    # Iterate over every timestamp in timeaxis.
    # Note the type of discontinuity.
    # Ignore repetition and check for actual discontinuity i.e. missing rows.

    while index < len(timeaxis):
        # Read original timestamp at the location of operation number
        original_timestamp = timeaxis[index]
        # Calculate the recentered timestamp starting at epoch where previous
        # discontinuity was found.
        recentered_timestamp = epoch + (index - epoch_index) * delta_t
        # message(original_timestamp,recentered_timestamp)
        # Check for next discontinuity
        if abs(original_timestamp - recentered_timestamp) >= toldt * delta_t:
            # Reset epoch, epoch_index
            epoch = original_timestamp
            epoch_index = index
            # Check for type of discontinuity
            # Repetitive rows
            if (original_timestamp == recentered_timestamp) or (
                recentered_timestamp - original_timestamp
            ) >= toldt * delta_t:
                # if recentered_timestamp-original_timestamp<=0.01*delta_t:
                # 		 message("Error!!",message_verbosity=0)
                repetition = 1
                discont_type = repetition
                message(
                    "Repetitive rows found at index: %d,timestamp: %f"
                    % (index, original_timestamp),
                    message_verbosity=3,
                )
                message(
                    "Repetition at timestamp original: %f correct %f\n"
                    % (original_timestamp, recentered_timestamp),
                    message_verbosity=3,
                )
                repetition_counter += 1
            # Missing rows
            if (original_timestamp - recentered_timestamp) >= (
                1.0 + toldt
            ) * delta_t:
                discont = 2
                discont_type = discont
                message(
                    "Jump discountinuity in data found at"
                    "index:%d,timestamp:%f" % (index, original_timestamp),
                    message_verbosity=2,
                )
                message("delta_t=%f" % delta_t, message_verbosity=1)
                message(
                    "Jump at timestamp original: %f correct %f\n Dt = %f"
                    % (
                        original_timestamp,
                        recentered_timestamp,
                        (original_timestamp - recentered_timestamp) / delta_t,
                    ),
                    message_verbosity=1,
                )
                discont_counter += 1

            # discont_type=repetition+discont
            discont_type_details.append([index, discont_type])
            # Append [index location,recentered timestamp
            # discontinuity_details.append([index,
            # recentered_timestamp,discontinuity_type])
        # Increment the no. of total operations
        index += 1
        # message("Progress: %f%%\r"%(epoch_index*100./len(timeaxis)))
    # discont_type=(repetition+discont)
    # Print the result
    if discont_type:
        message("The data is not clean!", message_verbosity=1)
        global_discont_type = repetition + discont
        message("Discontinuity type:", global_discont_type, message_verbosity=1)
        if global_discont_type == 1:
            message(
                "The data has repetitive rows at %d locations"
                % repetition_counter,
                message_verbosity=1,
            )
        elif global_discont_type == 2:
            message(
                "The data has %d discontinuities" % discont_counter,
                message_verbosity=1,
            )
        else:
            message(
                "The data has repetitive rows and is discontinious",
                message_verbosity=1,
            )

        discontinuity_details = [discont_type_details, global_discont_type]
    else:
        message("Data is continuous", message_verbosity=2)
        discontinuity_details = [[0, 0], 0]
    # Return the details of discontinuity

    return discontinuity_details


def remove_repeated_rows(data, delta_t, toldt=1e-3):
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
           The tolerance for error in checking. defaluts to toldt=1e-3.

    Returns
    -------
    cleaned_data: list
                  The cleaned data array with repetitive
                  rows removed.

    """
    message("Checking data for repetative rows...\n", message_verbosity=2)

    time = data[0]

    # Index of ros to delete
    dind = []

    # Initial data length
    ki_index = len(time)

    # message("length of old array is %d\n" %ki)
    # Row iteration variable
    ii_index = 0

    # Flag to identify if any repetetive rows were found
    rep_row_index = 0

    # ci=0
    # rowcounter
    counter = 0

    # Iterate over rows
    while ii_index < ki_index - 1:
        # Repetition condition :if the successive time stamp is less than or
        # equal to the present, delete the row.
        if (time[ii_index + 1] == time[ii_index]) or (
            time[ii_index] - time[ii_index + 1]
        ) >= toldt * delta_t:
            # if time[ii_index]-time[ii_index+1]<=0.01*delta_t:
            # 		 message("Error!!",message_verbosity=0)
            # Set flag to 1 if repetition condition is met.
            rep_row_index = 1
            counter += 1
            message(
                "found a repeating row at %d, time %f\n"
                % (ii_index + 1, time[ii_index + 1]),
                message_verbosity=3,
            )
            message(
                "timei: %f timef %f\n" % (time[ii_index], time[ii_index + 1]),
                message_verbosity=3,
            )
            # ci = ci+1
            # Delete the entire row
            time = np.delete(time, ii_index + 1, 0)
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

    if rep_row_index == 1:
        message("Repetitive rows were removed", message_verbosity=2)
        message("No. of rows removed = %d\n" % counter, message_verbosity=2)
        message("Length of new array is %d\n" % len(time), message_verbosity=3)

    else:
        message("No points removed\n", message_verbosity=2)
        # message("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    # Return the "cleaned" data matrix
    cleaned_data = data

    return cleaned_data
