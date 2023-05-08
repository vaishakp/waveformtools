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
