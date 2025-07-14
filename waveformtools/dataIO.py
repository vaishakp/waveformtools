"""Functions for handling data IO operations from waveforms class."""

#########################
# Imports
#########################

import json
import re
import sys
import os

import h5py, tarfile
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from scipy.interpolate import interp1d

from waveformtools.waveformtools import (
    cleandata,
    interp_resam_wfs,
    message,
    xtract_camp_phase,
)

from scipy.stats import mode as stats_mode

##########################
# RIT data
##########################


def _key_gen(ell, emm, extras=None):
    """Generates strings to be used as keys for.

    managing h5 datasets.

    Parameters
    ----------
    ell: int
         The polar angular mode number
         :math:`\\ell`.
    emm: int
         The aximuthal angular mode number
         :math:`m`.
    extras: str
            Any extra string to be appended
            to the end of the key.

    Returns
    -------
    key: str
         A string key.
    """

    key = f"l{ell}_m{emm}"

    if extras is not None:
        key += f"_{extras}"
        # message('adding rext')

    return key


def get_ell_max_from_keys(all_keys):
    """Get ell max from a list of keys.

    Parameters
    ----------
    all_keys: list
              A list of strings string containing the modes keys.
    Returns
    -------
    ell_max: int
             The maximum available ell.
    """

    # available mode numbers
    all_ell_modes = set({})

    # Get mode numbers
    for item in all_keys:
        this_match = re.search(r"\_l\d\_", item)

        if this_match is None:
            # message(f'Skipped file {item}')
            continue

        # message('Match found', this_match.string)
        s1, s2 = this_match.span()
        this_ell = int(this_match.string[s1 + 2 : s2 - 1])
        # message(this_ell)

        all_ell_modes.add(this_ell)

    ell_max = max(all_ell_modes)
    return ell_max


def get_ell_max_from_file(data_dir, var_type="Psi4", file_name="*.h5"):
    """Get the largest available mode number from available data in files.

    Parameters
    ----------
    data_dir: string
              A string containing the directory path where the mode files
              can be found.
    var_type: string, optional
              A string that denotes the variable that is being loaded.
              Options are Psi4 and strain.
              The former is the default.
    file_name: string, optional
               The h5 file that contains the modes data.
               It defaults to the only file in the directory.
               If there are multiple files, it throws an error.
    Returns
    -------
    ell_max: int
             The maximum available ell.
    keys_list: list
               A list of data access keys.

    Notes
    -----
    Reads in various ASCII dat files for RIT Psi4,
    h5 files for RIT strain and gen strain.
    """

    if var_type == "Psi4":
        import os

        # Get files
        all_fnames = os.listdir(data_dir)
        # Get only files
        all_fnames = [
            item for item in all_fnames if os.path.isfile(f"{data_dir}/{item}")
        ]

    elif var_type == "Strain":
        # import h5py
        message("Fetching all keys from H5 file", message_verbosity=3)
        data_file = h5py.File(f"{data_dir}/{file_name}")
        all_fnames = list(data_file.keys())
        data_file.close()
        # message(all_fnames)

    # Parse ell values
    # Filter the keys.
    all_fnames = [item for item in all_fnames if "_l" in item]

    message("All files found", all_fnames, message_verbosity=3)

    ell_max = get_ell_max_from_keys(all_fnames)

    return ell_max, all_fnames


def _get_modes_list_from_keys(keys_list, r_ext):
    """Get the modes list from the keys list.
    of an hdf file.

    Parameters
    ----------
    keys_list: list
               The list containing all the keys
    r_ext: float
           The extraction radius of the data.

    Returns
    -------
    modes_list: list
                The list of modes.
    """

    # Sort the keys to ensure a nice
    # modes list structure.
    keys_list_orig = sorted(keys_list)

    message("List of keys received", keys_list, message_verbosity=3)

    if r_ext != -1:
        keys_list = [item for item in keys_list_orig if f"r{r_ext}" in item]

        if keys_list == []:
            message(
                "Got an empty list. Searching for r_ext value in key string"
            )
            keys_list = [item for item in keys_list_orig if f"{r_ext}" in item]

    message("List of filtered keys", keys_list, message_verbosity=3)
    # The list of modes.
    modes_list = []

    # Initialize the emm modes sublist.
    emm_modes_for_ell = []

    # Present ell value to
    # initialize the mode concatenation.
    ell_old = 0

    for key in keys_list:
        # message(key)
        # Get the ell value
        ell, emm = _get_ell_emm_from_key(key)
        message(
            "ell value: ",
            ell,
            "emm value:",
            emm,
            message_verbosity=3,
        )

        if ell != ell_old:
            # If the ell value has changed,
            # update the modes list before moving
            # onto the next ell value.
            modes_list.append([ell_old, emm_modes_for_ell])
            # The present ell value is the old
            # ell value.
            ell_old = ell

            # Reset the ell_mode list.
            emm_modes_for_ell = []

        # Update it with the new emm mode.
        emm_modes_for_ell.append(emm)

    modes_list.append([ell, emm_modes_for_ell])

    return modes_list


def _get_ell_emm_from_key(key):
    """Get the :math:`\\ell` and :math:`m` values.

    from a given key string of an hdf file.

    Parameters
    ----------
    key: str
         The input key string

    Returns
    -------
    ell: int
               The :math:`\\ell` value
    emm: int
               The :math:`m` value.

    Notes
    -----
    Assumes that the input string has :math:`\\ell` and :math:`m` values
    in the form `l{int}m{int}`.
    """

    import re

    str_match = re.search(r"l\d*", key)
    ell_str_start = str_match.start()
    ell_str_end = str_match.end()
    ell = int(key[ell_str_start + 1 : ell_str_end])

    str_match = re.search(r"m-*\d*", key)
    emm_str_start = str_match.start()
    emm_str_end = str_match.end()
    emm = int(key[emm_str_start + 1 : emm_str_end])

    return ell, emm


def get_iteration_numbers_from_keys(keys_list):
    """Get the iteration number from keys.

    Parameters
    ----------
    keys_list: list
               The list of keys.

    Returns
    -------
    iteration_numbers: list
                       The list containing the iteration
                       numbers.
    """
    import re

    iteration_numbers = []

    for key in keys_list:
        str_match = re.search(r" it=\d* ", key)
        it_str_start = str_match.start()
        it_str_end = str_match.end()
        it_value = int(key[it_str_start + 4 : it_str_end])
        iteration_numbers.append(it_value)

    return iteration_numbers


def get_files_in_tar_gz(file_path):
    """Get the files in a tar.gz.

    Parameters
    ----------
    file_path: str,pathlib.Path
                The full path to the tar file
    """
    with tarfile.open(file_path, "r:gz") as tf:
        all_files = [one_file.name for one_file in tf.getmembers()]

    return all_files


def get_ell_max_from_tar_gz_file(file_path):
    """Get the ell_max from files in a tar.gz.

    Parameters
    ----------
    file_path: str,pathlib.Path
                The full path to the tar file

    Returns
    -------
    ell_max: int
             The l max of the modes available
    filtered_fnames: list of str
                     A list contaning the names of the modes files.
    """

    all_fnames = get_files_in_tar_gz(file_path)
    filtered_fnames = [item for item in all_fnames if "_l" in item]

    message("All files found", filtered_fnames, message_verbosity=3)

    ell_max = get_ell_max_from_keys(filtered_fnames)

    return ell_max, filtered_fnames


def read_data_from_tar_gz_subfile(tar_gz_file_handle, subfile_name):
    """Extract and read the data in a subfile inside the given open.

    tar gz file handle
    """

    with tar_gz_file_handle.extractfile(subfile_name) as df:
        data = np.genfromtxt(df)

    return data


def reorder_modes_list(modes_list):
    """Reorder a modes list."""

    modes_dict = {}
    ordered_modes_list = []

    for ell, emm_list in modes_list:
        modes_dict.update({ell: sorted(emm_list)})

    sorted_ells = sorted(list(modes_dict.keys()))

    for ell in sorted_ells:
        ordered_modes_list.append([ell, modes_dict[ell]])

    return ordered_modes_list


def construct_mode_list(ell_max, spin_weight):
    """
    Construct a modes list in the form.

    [[ell1, [emm1, emm2, ...], [ell2, [emm..]],..]
    given the :math:`\\ell_{max}.`

    Parameters
    ----------
    spin_weight: int
                 The spin weight of the modes.
    ell_max: int
             The :math:`\\ell_{max}` of the modes list.

    Returns
    -------
    modes_list: list
                A list containg the mode indices lists.

    Notes
    -----
    The modes list is the form which the `waveform` object understands.
    """

    # The modes list.
    modes_list = []

    message("Construct modes list in dataIO", message_verbosity=4)
    message(
        f"ell_max {ell_max}, spin weight {spin_weight}", message_verbosity=4
    )

    for ell_index in range(abs(spin_weight), ell_max + 1):
        # Append all emm modes for each ell mode.
        modes_list.append([ell_index, list(range(-ell_index, ell_index + 1))])

    message(
        "ell max of created modes list"
        f"{max([item[0] for item in modes_list])}",
        message_verbosity=4,
    )
    message("--------------------------------\n", message_verbosity=4)

    return modes_list


def sort_keys(modes_keys_list):
    """Sort the keys in a list based on its iteration number

    Parameters
    ----------
    modes_keys_list: str
                     The list of keys.

    Returns
    -------
    sorted_modes_keys_list: str
                            The sorted list.
    """

    iteration_numbers = get_iteration_numbers_from_keys(modes_keys_list)
    sargs = np.argsort(iteration_numbers)
    sorted_modes_keys_list = np.array(modes_keys_list)[sargs]

    return sorted_modes_keys_list


from matplotlib.style import available
from pathlib import Path


def load_RIT_Psi4_data_from_disk(
    data_file_path,
    wfa=None,
    label="RIT_rPsi4inf",
    ell_max=None,
    modes_list=None,
    save_as_ma=False,
    spin_weight=-2,
    resam_type="finest",
    interp_kind="cubic",
    crop=False,
    centre=False,
    output_modes_array=False,
):
    """Load the Psi4 waveforms from the RIT catalogue.

    from ASCII files from disk.

    Parameters
    ----------
    wfa: ModesArray
         An instance of the waveforms ModesArray class.
         Updates this instance if provided, else creates a new instance.
    data_file_path: string/pathlib.Path
                    A string containing the directory path where
                    the mode files can be found.
    label: string, optional
           The label of the modes_array object.
    ell_max: int, optional
             The maximum mode number to load. If not specified,
             then all available modes are loaded.
    save_as_ma: bool, optional
                Save to disk again as a modes_array h5 file?
    spin_weight: int, optional
                 The spin weight of the object. Used for filtering modes.
                 Defaults to -2.
    resam_type: string, float, optional
                The type of resampling to do. Options are finest and coarsest,
                and user input float.
    interp_kind: string, optional
                 The interpolation type to use. Default is cubic.

    Returns
    -------
    rit_modes_array: modes_array
                     A modes_array instance containing the loaded modes if
                     `output_modes_array` is True
    time_axis, modes_data: ndarray
                           Time axis and an array whose
                           first axis is time and second is
                           flatened modes index consistent with
                           ModesArray

    Notes
    -----
    It seems like the time axis of individual modes are identical to
    each other. Hence, one need not worry about choosing the time domain.
    This may change in the future.
    """
    tar_file_name_prefix = os.path.basename(data_file_path)[:-7]

    # For interpolation
    from scipy.interpolate import interp1d

    message("Loading RIT Psi4 type data.", message_verbosity=1)
    available_ell_max, available_modes_files = get_ell_max_from_tar_gz_file(
        data_file_path
    )

    # Create a modes array
    if modes_list is None:
        # Max available mode l.
        if ell_max is None:
            ell_max = available_ell_max

        else:
            if ell_max > available_ell_max:
                message(
                    f"ell_max {ell_max} requested but maximum"
                    "available ell_max is {available_ell_max}",
                    message_verbosity=1,
                )
                ell_max = available_ell_max

        # Construct a modes list
        wf_modes_list = construct_mode_list(
            ell_max=ell_max, spin_weight=spin_weight
        )

        message("The modes list is", wf_modes_list, message_verbosity=3)

    else:
        modes_list = reorder_modes_list(modes_list)
        ell_max = max([item[0] for item in modes_list])
        wf_modes_list = modes_list

    ##########################################
    # Read in the data
    #########################################
    message("Reading in modes...", message_verbosity=2)
    created = False
    with tarfile.open(data_file_path, "r:gz") as open_tar_file_handle:
        for ell, emm_list in wf_modes_list:
            for emm in emm_list:
                mode_idx = ell * 2 + ell + emm

                message("Loading", ell, emm, message_verbosity=2)

                # Construct file path
                wf_psi4_mode_data = read_data_from_tar_gz_subfile(
                    open_tar_file_handle,
                    f"{tar_file_name_prefix}/rPsi4_l{ell}_m{emm}_rInf.asc",
                )

                # Get time axis
                wf_psi4_time = wf_psi4_mode_data[:, 0]

                if not created:
                    message("\t Inferring time axis", message_verbosity=2)

                    min_dt = round(min(np.diff(wf_psi4_time)), 2)
                    max_dt = round(max(np.diff(wf_psi4_time)), 2)

                    message(
                        f"Min dt {min_dt} and Max dt {max_dt}",
                        message_verbosity=2,
                    )

                    if resam_type == "finest":
                        # Choose finest available timestep
                        # for upto 3 decimal digits.
                        m_dt = min_dt
                        message(
                            "\tResampling at the finest timestep",
                            m_dt,
                            message_verbosity=2,
                        )
                    elif resam_type == "coarsest":
                        m_dt = max_dt
                        message(
                            "\tResampling at the coarsest timestep",
                            m_dt,
                            message_verbosity=2,
                        )

                    elif isinstance(resam_type, float):
                        m_dt = resam_type
                        message(
                            "\tResampling at user defined timestep",
                            m_dt,
                            message_verbosity=2,
                        )

                    else:
                        raise KeyError(
                            f"Unrecognized resampling type {resam_type}"
                        )

                    # New (resampled) time axis
                    time_axis = np.arange(
                        wf_psi4_time[0], wf_psi4_time[-1] + m_dt, m_dt
                    )

                    # Length of data.
                    data_len = len(time_axis)
                    # To hold the loaded data
                    modes_data = np.zeros(
                        (data_len, (ell_max + 1) ** 2), dtype=np.complex128
                    )
                    created = True

                ###############################
                # Load the phase data
                ##############################
                Yphase = wf_psi4_mode_data[:, 4]
                Yphase_interp_fun = interp1d(
                    wf_psi4_time, Yphase, kind=interp_kind
                )

                # Resample
                Yphase_resam = Yphase_interp_fun(time_axis)

                ###########################
                # Load the amplitude data
                ###########################
                Yamp = wf_psi4_mode_data[:, 3]
                Yamp_interp_fun = interp1d(wf_psi4_time, Yamp, kind=interp_kind)

                # Resample
                Yamp_resam = Yamp_interp_fun(time_axis)
                wfmode = Yamp_resam * np.exp(1j * Yphase_resam)

                ###################################
                # Set the modes data
                ###################################
                modes_data[:, mode_idx] = wfmode

    if output_modes_array == True:
        from waveformtools.modes_array import ModesArray

        data_dir = os.path.dirname(data_file_path)

        if not isinstance(wfa, ModesArray):
            wfa = ModesArray(
                label=label,
                data_dir=data_dir,
                modes_list=modes_list,
                extra_mode_axes_shape=None,
            )

            # Create a modes array object
            wfa.create_modes_array(ell_max=ell_max, data_len=data_len)

        # Assign to it the time axis
        wfa.time_axis = time_axis
        wfa.ell_max = ell_max
        wfa.modes_list = wf_modes_list
        wfa.r_ext = np.inf
        wfa._actions += "->load_modes"
        wfa._modes_data = modes_data.T

        if crop is not False or centre is True:
            # Trim or recenter
            if crop is True or centre is True:
                wfa.trim(trim_upto_time=0)
                wfa.centered = True
                wfa._actions += "->recenter"

            elif isinstance(crop, float):
                wfa.trim(trim_upto_time=crop)
                wfa._actions += "->crop"

            if save_as_ma is True:
                # Save the modes array as waveforms hdf file
                wfa.save_modes(out_file_name="{label}_resam.h5")
                wfa._actions += "->save_as_wfh5"
        return wfa

    else:
        return time_axis, modes_data


def load_RIT_Strain_data_from_disk(
    wfa=None,
    data_dir="./",
    file_name="*",
    label="RIT_strain",
    spin_weight=-2,
    ell_max="auto",
    resam_type="auto",
    interp_kind="cubic",
    save_as_ma=False,
    modes_list=None,
    crop=False,
    centre=True,
    r_ext_factor=1,
    debug=False,
):
    """Load the RIT or strain waveforms from the RIT/ MAYA catalogue data,.

    from hdf5 files from disk.

    Parameters
    ----------
    wfa: waveforms
         An instance of the waveforms class. Creates a new one if
         not provided.
    data_dir: string
              A string containing the directory path
              where the mode files can be found.
    label: string, optional
           The label of the modes_array object.
    ell_max: int, optional
             The maximum mode number to load.
             If not specified, then all available modes are loaded.
    save_as_ma: bool, optional
                Save to disk again as a modes_array h5 file?
    spin_weight: int, optional
                 The spin weight of the object. Used for filtering modes.
                 Defaults to -2.
    resam_type: string, float, optional
                The type of resampling to do. Options are
                the finest and coarsest, and user input float.
    interp_kind: string, optional
                 The interpolation type to use. Default is cubic.

    Returns
    -------
    rit_modes_array: modes_array
                     A modes_array instance containing the loaded modes.

    Notes
    -----
    It seems like the time axis of individual modes are
    identical to each other. Hence, one need not worry about
    choosing the time domain. This may change in future.
    """
    message("Loading RIT strain data.", message_verbosity=1)

    from functools import partial

    # Initialize the interpolator
    if isinstance(interp_kind, int):
        message(
            "Interpolating using InterpolatedUnivariateSpline",
            message_verbosity=2,
        )
        interpolator = partial(interp, k=interp_kind)

    elif isinstance(interp_kind, str):
        from scipy.interpolate import interp1d

        message("Interpolating using interp1d", message_verbosity=2)
        interpolator = partial(interp1d, kind=interp_kind)

    from waveformtools.modes_array import ModesArray

    # Max available mode l.
    ell_max_act, keys_list = get_ell_max_from_file(
        data_dir=data_dir, var_type="Strain", file_name=file_name
    )

    ####################################
    # Set variables with priorities
    # Note: rework this in dictionaries
    ####################################

    if ell_max == "auto":
        ell_max = ell_max_act
    if ell_max is None:
        message("ell_max not provided.")

        if wfa is not None:
            wfa_ell_max = wfa.ell_max
        else:
            wfa_ell_max = None

        if wfa_ell_max is None:
            message("modes array not provided. Setting ell_max from file...")
            ell_max = ell_max_act
        else:
            message("Setting ell_max from given modes_array")
            ell_max = wfa.ell_max

    message("Chosen ell max", ell_max, "Available ell_max", ell_max_act)

    if not isinstance(wfa, ModesArray):
        # Create a modes array
        wfa = ModesArray(label=label, ell_max=ell_max, modes_list=modes_list)
    # wfa = modes_array(label=label, data_dir=data_dir, modes_list=modes_list)
    if debug is True:
        wf_nl = ModesArray(
            label=label + "_nl", ell_max=ell_max, modes_list=modes_list
        )

    if not data_dir:
        data_dir = wfa.data_dir
    else:
        wfa.data_dir = data_dir

    if not file_name:
        file_name = wfa.file_name
    else:
        wfa.file_name = file_name

    if not ell_max:
        ell_max = wfa.ell_max
    else:
        wfa.ell_max = ell_max

    # ell_max		 = 12
    if not modes_list:
        if not wfa.modes_list:
            message("Constructing the modes list")
            modes_list = construct_mode_list(
                ell_max=ell_max, spin_weight=wfa.spin_weight
            )
        else:
            modes_list = wfa.modes_list
    else:
        wfa.modes_list = modes_list

    # For interpolation
    from scipy.interpolate import interp1d

    # Alias of the modes_array
    # label = 'q1a0_a'
    # Enforce only l>abs(spin_Weight) modes.
    # wf_modes_list = [item for item in wf_modes_list if item[0]>=abs(spin_weight)]
    # tend = []
    # tstart = []
    ##########################################
    # Read in the data
    #########################################
    # message(file_name)
    # Get the time axis
    # import h5py
    data_file = h5py.File(f"{data_dir}/{file_name}")

    try:
        # For RIT data type
        time_axis = data_file["NRTimes"][...]
        # dt_auto = time_axis[1]-time_axis[0]
        from scipy.stats import mode as stats_mode

        dt_auto = stats_mode(np.diff(time_axis))[0]

    except Exception as excep:
        dt_auto = None
        message(
            "NRTimes not present. Will compute dt auto from mode time axis",
            excep,
            message_verbosity=2,
        )

    message("Reading in modes...")
    for ell, emm_list in modes_list:
        for emm in emm_list:
            this_amp_key = f"amp_l{ell}_m{emm}"
            this_phase_key = f"phase_l{ell}_m{emm}"

            message("Loading", ell, emm, message_verbosity=3)
            # Construct file path

            # Create modes_array on first run

            ###############################
            # Load the phase data
            ##############################
            Tphase = data_file[this_phase_key]["X"][...]

            Yphase = data_file[this_phase_key]["Y"][...]

            # message(wfa.modes_data)
            if wfa.modes_data.all() == np.array(None):
                message("Creating modes data")

                if dt_auto is None:
                    # For MAYA data type
                    time_axis = Tphase
                    # dt_auto = time_axis[1]-time_axis[0]
                    from scipy.stats import mode as stats_mode

                    dt_auto = stats_mode(np.diff(time_axis))[0]

                min_dt = round(min(np.diff(time_axis)), 2)
                max_dt = round(max(np.diff(time_axis)), 2)
                message(f"Default dt is {dt_auto}", message_verbosity=2)

                if resam_type == "auto":
                    # Choose finest available timestep
                    # for upto 3 decimal digits.
                    m_dt = dt_auto
                    message(
                        "Sampling at the default timestep",
                        m_dt,
                        message_verbosity=2,
                    )

                elif resam_type == "finest":
                    m_dt = min_dt
                    message(
                        "Sampling at the finest available timestep",
                        m_dt,
                        message_verbosity=2,
                    )

                elif resam_type == "coarsest":
                    m_dt = max_dt
                    message(
                        "Sampling at the coarsest available timestep",
                        m_dt,
                        message_verbosity=2,
                    )

                elif isinstance(resam_type, float):
                    m_dt = resam_type
                    message(
                        "Resampling at user defined timestep",
                        m_dt,
                        message_verbosity=2,
                    )

                    # New (resampled) time axis
                    time_axis = np.arange(time_axis[0], time_axis[-1], m_dt)

                else:
                    raise NotImplementedError(
                        f"Unknown resampling parameter {resam_type}"
                    )

                # Length of data.
                data_len = len(time_axis)

                # Create a modes array object
                wfa.create_modes_array(ell_max=ell_max, data_len=data_len)

                # Assign to it the time axis
                wfa.time_axis = time_axis
                # message(wfa.time_axis)
            # message(wfa.time_axis - wf_psi4_time)
            # continue
            ###################################
            # Uniform sampling
            ###################################
            # message('Wfa time axis', wfa.time_axis)

            Yphase_interp_fun = interpolator(Tphase, Yphase)
            # Yphase_interp_fun = interpolator(Tphase, Yphase, k=3)

            # Resample

            Yphase_resam = Yphase_interp_fun(time_axis)

            ###########################
            # Load the amplitude data
            ###########################
            Tamp = data_file[this_amp_key]["X"][...]

            Yamp = data_file[this_amp_key]["Y"][...]

            Yamp_interp_fun = interpolator(Tamp, Yamp)
            # Yamp_interp_fun = interpolator(Tamp, Yamp, k=3)

            # wf_c = Yamp*np.exp(1j*Yphase)

            # Resample

            Yamp_resam = Yamp_interp_fun(time_axis)

            wfmode = Yamp_resam * np.exp(1j * Yphase_resam)

            # if not (Tphase==Tamp).all():
            # 	 raise ValueError('The time axis of the amps and phase are different!')

            # wfmode = interp_resam_wfs(wf_c, Tphase, time_axis, k=4)

            ###################################
            # Load the modes data
            ###################################

            wfa.set_mode_data(ell=ell, emm=emm, data=r_ext_factor * wfmode)

    data_file.close()

    #####################
    # Finishing touches
    #####################
    wfa._actions += "->load_modes"

    # Trim or recenter
    if centre is True:
        wfa.trim(trim_upto_time=0)
        wfa._actions += "->center"

    if isinstance(crop, float):
        wfa.trim(trim_upto_time=crop)
        wfa._actions += "->crop"

    if save_as_ma is True:
        # Save the modes array as waveforms hdf file
        wfa.save_modes(out_file_name=f"{label}_resam.h5")
        wfa._actions += "->save_as_wfh5"

    if debug is True:
        return wfa, wf_nl
    else:
        return wfa


#################################################################
# Generic data type
#################################################################


def load_gen_data_from_disk(
    wfa=None,
    label="generic waveform",
    data_dir="./",
    file_name="*.h5",
    r_ext=None,
    ell_max=8,
    pre_key=None,
    modes_list=None,
    crop=False,
    centre=True,
    key_ex=None,
    r_ext_factor=1,
    *args,
    **kwargs,
):
    """Load the RIT strain waveforms from the RIT catalogue,.

    from hdf5 files from disk.

    Parameters
    ----------
    data_dir: string
              A string containing the directory path where
              the mode files can be found.
    label: string, optional
           The label of the modes_array object.
    ell_max: int, optional
             The maximum mode number to load. If not specified,
             then all available modes are loaded.
    save_as_ma: bool, optional
                Save to disk again as a modes_array h5 file?
    spin_weight: int, optional
                 The spin weight of the object. Used for filtering modes.
                 Defaults to -2.
    resam_type: string, float, optional
                The type of resampling to do.
                Options are finest and coarsest, and user input float.
    interp_kind: string, optional
                 The interpolation type to use. Default is cubic.

    Returns
    -------
    rit_modes_array: modes_array
                     A modes_array instance containing the loaded modes.

    Notes
    -----
    It seems like the time axis of individual modes are identical to
    each other. Hence, one need not worry about
    choosing the time domain. This may change in future.
    """
    message("Loading generic data.", message_verbosity=1)
    from waveformtools.modes_array import ModesArray

    # Max available mode l.
    if not isinstance(wfa, ModesArray):
        # Create a modes array
        wfa = ModesArray(
            label=label,
            data_dir=data_dir,
            modes_list=modes_list,
            ell_max=ell_max,
        )

    # if not data_dir:
    # 	data_dir = wfa.data_dir
    # else:
    # 	wfa.data_dir = data_dir

    # if not file_name:
    # 	file_name = wfa.file_name
    # else:
    # 	wfa.file_name = file_name

    # if not ell_max:
    # 	ell_max = wfa.ell_max
    # else:
    # 	wfa.ell_max = ell_max

    # if not label:
    # 	label = wfa.label
    # ell_max		 = 12
    # Max available mode l.

    full_path = f"{data_dir}/{file_name}"
    message(f"Loading data from {full_path}", message_verbosity=1)
    # Enforce only l>2 modes.
    # wf_modes_list = [item for item in wf_modes_list if
    # item[0]>=abs(spin_weight)]

    # Open the modes file.
    # import h5py, json
    full_path = wfa.data_dir/wfa.file_name

    # message(wfa.file_name)
    with h5py.File(full_path, "r") as wfile:
        #################################
        # Get metadata
        ###############################

        # Load metadata if present.
        try:
            # if 1:
            metadata_bytes = bytes(np.void(wfile["metadata"])).decode()
            metadata = json.loads(metadata_bytes)
            # Import the metadata.
            for key, val in metadata.items():
                if val is not None:
                    wfa.__dict__.update({key: val})
            message("Metadata loaded", message_verbosity=2)
            message(
                "Waveform meta data:", wfa.get_metadata(), message_verbosity=2
            )

        except Exception as ex:
            # If no metadata found, pass empty dict for updation.
            message("No metadata found!", ex)
            metadata = {}
            pass

        # Get the list of keys.
        modes_keys_list = list(wfile.keys())

        # message('Keys ', modes_keys_list)
        # message(wfa.get_metadata())

        # Check and filter for particular key string pattern
        if key_ex is not None:
            # Filter the keys according to key_ex if specified.
            message("Filtering as per", key_ex)
            wfa.key_ex = key_ex
            modes_keys_list = [
                item for item in modes_keys_list if key_ex in item
            ]
            # message(modes_keys_list)

        else:
            message(
                "key_ex is not specified. Proceeding without filtering..",
                message_verbosity=2,
            )

        modes_keys_list = sorted(modes_keys_list)
        # message('Modes keys', modes_keys_list)

        wfa.mode_keys_list = modes_keys_list

        # Construct the list of modes if it doesnt exist.
        ##########################
        # Construct modes list
        ##########################
        if r_ext:
            wfa.r_ext = r_ext
        elif wfa.r_ext is not None:
            r_ext = wfa.r_ext
        else:
            message("Unable to parse extraction radius!")
            sys.exit(0)
        if not modes_list:
            # Check if modes list is given for which mode to load.
            # If the list of modes is not given then construct the list of modes.

            if not ell_max:
                # If ell max is also not specified,
                # construct the list of modes using
                # the list of modes h5 file keys.
                modes_list = _get_modes_list_from_keys(modes_keys_list, r_ext)
                # message(modes_list)
                # Get the ell max
                ell_max = max([item[0] for item in modes_list])

                # metadata.update({ell_max : ell_max })
                wfa.ell_max = ell_max

            else:
                # self.ell_max = ell_max
                # If ell max is given, construct the
                # list of modes directly.
                modes_list = construct_mode_list(
                    ell_max, spin_weight=wfa.spin_weight
                )

                # set the modes list attr.
                wfa.modes_list = modes_list

        else:
            # self.modes_list = modes_list
            # If modes list is given, get ell_max from it.
            if not ell_max:
                # Get the ell max
                ell_max = max([item[0] for item in modes_list])

        # Set the ell_max attribute if not already.
        if not wfa.ell_max:
            wfa.ell_max = ell_max

        #################################################
        # Load modes
        #################################################
        # Modes array created flag
        cflag = 0
        # Load the modes listed in mode_numbers list
        for item in modes_list:
            # For every ell mode list in modes_list

            ell, emm_list = item

            for emm_index, emm in enumerate(emm_list):
                # For every (ell, emm) mode.

                # Find the key corresponding to the mode
                try:
                    key = str(
                        [
                            item
                            for item in modes_keys_list
                            if re.search(
                                f"l{ell}_m{emm}_r{r_ext}", item
                            )
                        ][0]
                    )
                    # message('The loaded key is ', key, type(key))
                    # message('The loaded key is ', key, type(key))
                    # if key=='l0_m0_r500.00':
                    # message('Its alright')
                except Exception as ex:
                    message(
                        f"Waveform dataset for l{ell}, m{emm} not found",
                        ex,
                    )
                    sys.exit(0)

                # Get the data
                data = np.array(wfile[key])

                # set the time and data axis
                time_axis, data_re = cleandata([data[:, 0], data[:, 1]])
                time_axis, data_im = cleandata([data[:, 0], data[:, 2]])

                # create flag
                if not cflag:
                    if (wfa.modes_data == np.array(None)).all():
                        if crop:
                            # Crop the beginning portion.
                            # delta_t = time_axis[1] - time_axis[0]
                            # shift = int(wfa.r_ext / delta_t)
                            raise NotImplementedError(
                                "Not implemented! Please contact the developers!"
                            )

                        else:
                            shift = 0
                        data_len = len(time_axis) - shift
                        wfa._data_len = data_len
                        # Delete the attribute
                        # del self.modes_data
                        # Create an array for the waveform mode object
                        wfa.create_modes_array(wfa.ell_max, data_len)
                        # self.modes_data = np.zeros([ell_max+1, 2*(ell_max+1) +1, data_len],
                        # dtype=np.complex128)
                        # self.modes_data = np.zeros([ell_max+1, 2*(ell_max+1) +1, data_len],
                        # dtype=np.complex128)

                        cflag = 1

                        # set the time axis.
                        # wfa.time_axis = time_axis[shift:]
                        wfa._time_axis = time_axis

                # wfa.set_mode_data(ell, emm,
                # r_ext_factor*(data_re[shift:] + 1j * data_im[shift:]))

                wfa.set_mode_data(
                    ell=ell,
                    emm=emm,
                    data=r_ext_factor * (data_re + 1j * data_im),
                )

        ##############################
        # Recenter axis
        ##############################
        # Trim or recenter
        if centre:
            crop_time = 0
            if crop:
                crop_time = wfa.r_ext

            wfa.trim(trim_upto_time=crop_time)

        # maxloc = np.argmax(np.absolute(self.mode(2, 2)))
        # maxtime = time_axis[shift + maxloc]

        # if wfa.maxtime is None:
        # 	 wfa.maxtime = maxtime
        # message("Max time is", maxtime)

        # if centre:
        # 	 wfa.time_axis = time_axis[shift:] - maxtime
        # message(wfa.file_name)
        return wfa


###################################################################
# SpEC
###################################################################


def load_SpEC_data_from_disk(
    wfa=None,
    label="SXS Strain",
    data_dir="./",
    file_name="rhOverM_Extrapolated_N5_CoM_Mem.h5",
    extrap_order=None,
    r_ext=None,
    ell_max=None,
    centre=True,
    modes_list=None,
    save_as_ma="False",
    resam_type="auto",
    interp_kind="cubic",
    compression_opts=0,
    r_ext_factor=1,
    debug=False,
):
    """Load the SpEC waveform to modes_array,from hdf5 files from disk.

    Parameters
    ----------
    wfa: modes_array, optional
          The modes array to which to store the loaded waveform to. A new modes array will be returned
          if not provided.
    data_dir: string
              A string containing the directory path where the mode files can be found.
    file_name: string
               The name of the file containing the waveform data.
    label: string, optional
           The label of the modes_array object.
    ell_max: int, optional
             The maximum mode number to load. If not specified,
             then all available modes are loaded.
    save_as_ma: bool, optional
                Save to disk again as a modes_array h5 file?
    resam_type: string, float, optional
                The type of resampling to do. Options are finest and coarsest, and user input float.
    interp_kind: string, optional
                 The interpolation type to use. Default is cubic.

    Returns
    -------
    modes_array: modes_array
                 A modes_array instance containing the loaded modes.
    """
    message(f"Loading SpEC data N{extrap_order}", message_verbosity=1)

    from waveformtools.modes_array import ModesArray

    # Load SXS waveforms to modes_array.
    # Spectra infinty

    full_path = f"{data_dir}/{file_name}"

    wf_f0 = h5py.File(full_path)

    if extrap_order is not None:
        # Extrap Key pattern
        gkey = f"Extrapolated_N{extrap_order}.dir"
        try:
            wf_file = wf_f0[gkey]
        except KeyError as ke:
            message(
                ke,
                ":\n Reading as SpEC file in external extrap mode",
                message_verbosity=2,
            )

            wf_file = wf_f0
    else:
        wf_file = wf_f0

    all_keys = list(wf_file.keys())
    print(all_keys)

    # Max available mode l.
    ell_max_act = get_ell_max_from_keys(all_keys)
    # message(ell_max_act)

    ####################################
    # Set variables with priorities
    # Note: rework this in dictionaries
    ####################################

    if ell_max == "auto":
        ell_max = ell_max_act

    if ell_max is None:
        message("ell_max not provided.")

        if wfa is not None:
            wfa_ell_max = wfa.ell_max
        else:
            wfa_ell_max = None

        if wfa_ell_max is None:
            message("modes array not provided. Setting ell_max from file...")
            ell_max = ell_max_act
        else:
            message("Setting ell_max from given modes_array")
            ell_max = wfa.ell_max

    message("Chosen ell max", ell_max, "Available ell_max", ell_max_act)

    if not isinstance(wfa, ModesArray):
        # Create a modes array
        wfa = ModesArray(label=label, ell_max=ell_max, modes_list=modes_list)
    # wfa = modes_array(label=label, data_dir=data_dir, modes_list=modes_list)
    if debug is True:
        wf_nl = ModesArray(
            label=label + "_nl", ell_max=ell_max, modes_list=modes_list
        )

    wfa.extrap_order = extrap_order
    message(f"Using extrap order {extrap_order}")

    if not data_dir:
        data_dir = wfa.data_dir
    else:
        wfa.data_dir = data_dir

    if not file_name:
        file_name = wfa.file_name
    else:
        wfa.file_name = file_name

    if not ell_max:
        ell_max = wfa.ell_max
    else:
        wfa.ell_max = ell_max

    # ell_max		 = 12
    if not modes_list:
        if not wfa.modes_list:
            message("Constructing the modes list")
            # sys.exit(0)
            modes_list = construct_mode_list(
                ell_max=ell_max, spin_weight=wfa.spin_weight
            )
        else:
            modes_list = wfa.modes_list
    else:
        wfa.modes_list = modes_list

    # Filter
    modes_list = [item for item in modes_list if item[0] >= 2]
    ############################################################

    # create flag
    # flag = None

    if (wfa.modes_data == np.array(None)).any():
        wfa = initialize_modes_array(wf_time, wfa, resam_type=resam_type, ell_max=ell_max)

    # Load modes
    for ell, emm_list in modes_list:
        for emm in emm_list:
            # message(ell, emm)

            this_key = f"Y_l{ell}_m{emm}.dat"

            # Input waveform from disk
            wf_data = wf_file[this_key][...]
            wf_time = wf_data[:, 0]
            # print('wf time', wf_time)

            wf_data_re = wf_data[:, 1]
            wf_data_im = wf_data[:, 2]
            wf_data_c = wf_data_re + 1j * wf_data_im

            # wf_amp, wf_phase = xtract_camp_phase(wf_data_re, wf_data_im)
            # Interpolate and resamplea
            # Note
            # Interpolating in amplitude and phase is better
            # and has lower interpolation errors
            # but is slower due to unwrapping of phases.

            wf_int = interp_resam_wfs(
                wf_data_c, wf_time, time_axis, kind="cubic", k=None
            )

            # amp_int = interp_resam_wfs(wf_amp, wf_time, time_axis)
            # phase_int = interp_resam_wfs(wf_phase, wf_time, time_axis)

            # re_int = interp1d(wf_time, wf_data_re)
            # message(wf_time[0], wf_time[-1], time_axis[0], time_axis[-1])
            # re_dat = re_int(time_axis)

            # im_int = interp1d(wf_time, wf_data_im)
            # im_dat = im_int(time_axis)

            # wfa.set_mode_data(ell, emm, data=re_dat + 1j * im_dat)

            # message(f"Setting mode Data {ell}, {emm} \n {wf_int} \n" )

            wfa.set_mode_data(ell=ell, emm=emm, data=wf_int)

    if debug is True:
        for ell, emm_list in modes_list:
            for emm in emm_list:
                this_key = f"Y_l{ell}_m{emm}.dat"

                # Input waveform from disk
                wf_data = wf_file[this_key]
                wf_time = wf_data[:, 0]
                wf_data_re = wf_data[:, 1]
                wf_data_im = wf_data[:, 2]

                if wf_nl.modes_data.all() == np.array(None):
                    wf_nl.create_modes_array(
                        ell_max=ell_max, data_len=len(wf_time)
                    )
                    wf_nl.time_axis = wf_time
                    wf_nl.data_len = len(wf_time)

                wf_nl.set_mode_data(ell=ell, emm=emm, data=wf_data_re + 1j * wf_data_im)

    if centre:
        wfa.trim(trim_upto_time=0)

    if save_as_ma is True:
        # Save the modes array as waveforms hdf file
        wfa.save_modes(
            out_file_name=f"{label}_resam.h5",
            compression_opts=compression_opts,
        )

    wf_f0.close()

    if debug is True:
        return wfa, wf_nl
    else:
        return wfa


def load_SpEC_non_extrap_data_from_disk(
    wfa=None,
    label="SXS Strain",
    data_dir="./",
    file_name="rh_FiniteRadii_CodeUnits.h5",
    r_ext=None,
    ell_max=None,
    centre=True,
    modes_list=None,
    save_as_ma="False",
    resam_type="auto",
    interp_kind="cubic",
    compression_opts=0,
    r_ext_factor=1,
    debug=False,
):
    """Load the SpEC waveform at finite radii.

    to modes_array, from hdf5 files from disk.

    Parameters
    ----------
    wfa: modes_array, optional
         The modes array to which to store
         the loaded waveform to. A new
         modes array will be returned
         if not provided.
    data_dir: string
              A string containing
              the directory path where
              the mode files can be found.
    file_name: string
               The name of the file
               containing the waveform data.
    label: string, optional
           The label of the modes_array object.
    ell_max: int, optional
             The maximum mode number to load.
             If not specified,
             then all available modes are loaded.
    save_as_ma: bool, optional
                Save to disk again
                as a modes_array h5 file?
    resam_type: string, float, optional
                The type of resampling to do.
                Options are finest and coarsest,
                and user input float.
    interp_kind: string, optional
                 The interpolation type to use.
                 Default is cubic.

    Returns
    -------
    modes_array: modes_array
                 A modes_array instance
                 containing the loaded modes.
    """
    message("Loading SpEC data.", message_verbosity=1)

    from waveformtools.modes_array import ModesArray

    # Load SXS waveforms to modes_array.
    # Spectra infinty

    full_path = f"{data_dir}/{file_name}"

    wf_file = h5py.File(full_path)
    all_keys = list(wf_file.keys())
    # Key pattern
    gkey = all_keys[0]

    # Search for detector
    if r_ext is None:
        raise ValueError("Please provide r_ext")

    all_radii = np.array([int(item[1:-4]) for item in all_keys])

    message("All available radii:", all_radii, message_verbosity=1)

    req_key_loc = np.where(all_radii == r_ext)[0][0]
    # print(req_key_loc)

    req_key = all_keys[req_key_loc]

    dset = wf_file[req_key]

    # one_Rkey = all_keys[0]
    ell_keys = list(dset.keys())

    # Max available mode l.
    ell_max_act = get_ell_max_from_keys(ell_keys)

    # message(ell_max_act)

    ####################################
    # Set variables with priorities
    # Note: rework this in dictionaries
    ####################################

    if ell_max == "auto":
        ell_max = ell_max_act
    if ell_max is None:
        message("ell_max not provided.")

        if wfa is not None:
            wfa_ell_max = wfa.ell_max
        else:
            wfa_ell_max = None

        if wfa_ell_max is None:
            message("modes array not provided. Setting ell_max from file...")
            ell_max = ell_max_act
        else:
            message("Setting ell_max from given modes_array")
            ell_max = wfa.ell_max

    message("Chosen ell max", ell_max, "Available ell_max", ell_max_act)

    if not isinstance(wfa, ModesArray):
        # Create a modes array
        wfa = ModesArray(label=label, ell_max=ell_max, modes_list=modes_list)
    # wfa = modes_array(label=label, data_dir=data_dir, modes_list=modes_list)

    wfa._areal_radii = dset["ArealRadius.dat"][...]

    if debug is True:
        wf_nl = ModesArray(
            label=label + "_nl", ell_max=ell_max, modes_list=modes_list
        )

    wf_nl._areal_radii = dset["ArealRadius.dat"][...]

    wfa.extrap_order = "None"

    message(f"Using detector radius {r_ext}")

    if not data_dir:
        data_dir = wfa.data_dir
    else:
        wfa.data_dir = data_dir

    if not file_name:
        file_name = wfa.file_name
    else:
        wfa.file_name = file_name

    if not ell_max:
        ell_max = wfa.ell_max
    else:
        wfa.ell_max = ell_max

    # ell_max		 = 12
    if not modes_list:
        if not wfa.modes_list:
            message("Constructing the modes list")
            # sys.exit(0)
            modes_list = construct_mode_list(
                ell_max=ell_max, spin_weight=wfa.spin_weight
            )
        else:
            modes_list = wfa.modes_list
    else:
        wfa.modes_list = modes_list

    # Filter
    modes_list = [item for item in modes_list if item[0] >= 2]
    ############################################################

    # create flag
    # flag = None

    if (wfa.modes_data == np.array(None)).any():
        wfa = initialize_modes_array(wf_time, wfa, resam_type=resam_type, ell_max=ell_max)

    # Load modes
    for ell, emm_list in modes_list:
        for emm in emm_list:
            # message(ell, emm)

            this_key = f"Y_l{ell}_m{emm}.dat"

            # Input waveform from disk
            wf_data = dset[this_key][...]
            wf_time = wf_data[:, 0]
            wf_data_re = wf_data[:, 1]
            wf_data_im = wf_data[:, 2]
            wf_data_c = wf_data_re + 1j * wf_data_im

            # wf_amp, wf_phase = xtract_camp_phase(wf_data_re, wf_data_im)

            # message(type(wfa.modes_data))


            # Interpolate and resamplea
            # Note
            # Interpolating in amplitude and phase is better
            # and has lower interpolation errors
            # but is slower due to unwrapping of phases.

            wf_int = interp_resam_wfs(
                wf_data_c, wf_time, time_axis, kind="cubic", k=None
            )

            # amp_int = interp_resam_wfs(wf_amp, wf_time, time_axis)
            # phase_int = interp_resam_wfs(wf_phase, wf_time, time_axis)

            # re_int = interp1d(wf_time, wf_data_re)
            # message(wf_time[0], wf_time[-1], time_axis[0], time_axis[-1])
            # re_dat = re_int(time_axis)

            # im_int = interp1d(wf_time, wf_data_im)
            # im_dat = im_int(time_axis)

            # wfa.set_mode_data(ell, emm, data=re_dat + 1j * im_dat)
            wfa.set_mode_data(ell=ell, emm=emm, data=wf_int)

    if debug is True:
        for ell, emm_list in modes_list:
            for emm in emm_list:
                this_key = f"Y_l{ell}_m{emm}.dat"

                # Input waveform from disk
                wf_data = dset[this_key]
                wf_time = wf_data[:, 0]
                wf_data_re = wf_data[:, 1]
                wf_data_im = wf_data[:, 2]

                if wf_nl.modes_data.all() == np.array(None):
                    wf_nl.create_modes_array(
                        ell_max=ell_max, data_len=len(wf_time)
                    )
                    wf_nl.time_axis = wf_time
                    wf_nl.data_len = len(wf_time)

                wf_nl.set_mode_data(ell=ell, emm=emm, data=wf_data_re + 1j * wf_data_im)

    if centre:
        wfa.trim(trim_upto_time=0)

    if save_as_ma is True:
        # Save the modes array as waveforms hdf file
        wfa.save_modes(
            out_file_name=f"{label}_resam.h5",
            compression_opts=compression_opts,
        )

    wf_file.close()

    if debug is True:
        return wfa, wf_nl

    else:
        return wfa


########################################################################################################################
# SpECTRE
########################################################################################################################


def load_SpECTRE_data_from_disk(
    wfa=None,
    label="SpECTRE Strain",
    data_dir="./",
    file_name="rhOverM_Extrapolated_N5_CoM_Mem.h5",
    r_ext=None,
    ell_max=12,
    centre=True,
    modes_list=None,
    r_ext_factor=1,
    save_as_ma="False",
    resam_type="auto",
    kind="cubic",
    compression_opts=0,
    spin_weight=None,
):
    """Load the SpECTRE or SpEC CCE waveform to modes_array,
    from hdf5 files from disk.

    Parameters
    ----------
    wfa: modes_array, optional
         The modes array to which to store
         the loaded waveform to. A new modes array will be returned
         if not provided.
    data_dir: string
              A string containing the directory path
              where the mode files can be found.
    file_name: string
               The name of the file containing the waveform data.
    label: string, optional
           The label of the modes_array object.
    ell_max: int, optional
             The maximum mode number to load. If not specified,
             then all available modes are loaded.
    save_as_ma: bool, optional
                Save to disk again as a modes_array h5 file?
    resam_type: string, float, optional
                The type of resampling to do.
                Options are finest and coarsest, and user input float.
    interp_kind: string, optional
                 The interpolation type to use. Default is cubic.

    Returns
    -------
    modes_array: modes_array
                 A modes_array instance containing the loaded modes.
    """
    #spin_weight = -2
    if spin_weight is None:
        if wfa is None:
            spin_weight=-2
        else:
            if wfa.spin_weight is None:
                spin_weight = -2
            else:
                spin_weight = wfa.spin_weight

    message("Loading SpECTRE data.", message_verbosity=1)
    from waveformtools.waveforms import ModesArray
    import sxs
    # Load SXS waveforms to modes_array.
    # Spectre infinty
    full_path = f"{data_dir}/{file_name}"

    try:
        import scri
    except Exception as ex:
        message(
            "scri module is required for reading in SXS waveforms."
            "Please install and try again",
            ex,
        )
        sys.exit(0)

    wf_file = sxs.load("/home/vaishakprasad/.cache/sxs/SXS:BBH:2526v3.0/Lev3:Strain_N2.h5")
    #wf_file = scri.rpxmb.load(full_path)[0].to_inertial_frame()
    ell_max_act = int(wf_file.ell_max)
    #import pdb
    #pdb.set_trace()

    # Add readinf ell max from file
    if ell_max is None:
        if wfa is None:
            wfa_ell_max = None
        else:
            wfa_ell_max = wfa.ell_max

        if wfa_ell_max is None:
            ell_max = ell_max_act
        else:
            ell_max = wfa_ell_max

    if not isinstance(wfa, ModesArray):
        # Create a modes array
        wfa = ModesArray(label=label, ell_max=ell_max, modes_list=modes_list)


    if not data_dir:
        data_dir = wfa.data_dir
    else:
        wfa.data_dir = data_dir

    if not file_name:
        file_name = wfa.file_name
    else:
        wfa.file_name = file_name

    if not ell_max:
        ell_max = wfa.ell_max
    else:
        wfa.ell_max = ell_max

    if not modes_list:
        if not wfa.modes_list:
            message("Constructing the modes list")
            modes_list = construct_mode_list(
                ell_max=ell_max, spin_weight=spin_weight
            )
        else:
            modes_list = wfa.modes_list
    else:
        wfa.modes_list = modes_list

    wf_time = wf_file.t

    if (wfa.modes_data == np.array(None)).any():
        wfa = initialize_modes_array(wf_time, wfa, resam_type=resam_type, ell_max=ell_max)

    time_axis = wfa.time_axis

    for ell, emm_list in modes_list:
        for emm in emm_list:
            wf_data = wf_file.data[:, wf_file.index(ell, emm)]
            wf_data_re = wf_data.real
            wf_data_im = wf_data.imag
            
            re_int = interp1d(wf_time, wf_data_re, kind=kind)
            re_dat = re_int(time_axis)
            im_int = interp1d(wf_time, wf_data_im, kind=kind)
            im_dat = im_int(time_axis)
            wfa.set_mode_data(ell=ell, emm=emm, data=re_dat + 1j * im_dat)

    if centre:
        wfa.trim(trim_upto_time=0)

    if save_as_ma is True:
        # Save the modes array as waveforms hdf file
        wfa.save_modes(
            out_file_name=f"{label}_resam.h5",
            compression_opts=compression_opts,
        )

    return wfa


def initialize_modes_array(time_axis, modes_array, resam_type='finest', ell_max=8):
    """ Initialize a modes array given the time axis """

    message("Creating modes data")
    dt_auto = stats_mode(np.diff(time_axis))[0]
    message(f'Default dt is {dt_auto}', message_verbosity=3)
    min_dt = min(np.diff(time_axis))
    max_dt = max(np.diff(time_axis))

    message(
        f"Min dt {min_dt} and Max dt {max_dt}", message_verbosity=2
    )
    if resam_type == "finest":
        # Choose finest available timestep
        # for upto 3 decimal digits.
        m_dt = min_dt
        message(
            "Resampling at the finest timestep",
            m_dt,
            message_verbosity=1,
        )
    if resam_type == "coarsest":
        m_dt = max_dt
        message(
            "Resampling at the coarsest timestep",
            m_dt,
            message_verbosity=1,
        )
    if isinstance(resam_type, float):
        m_dt = resam_type
        message(
            "Resampling at user defined timestep",
            m_dt,
            message_verbosity=1,
        )
    if resam_type == "auto":
        # Choose finest available timestep
        # for upto 3 decimal digits.
        m_dt = dt_auto
        message(
            "Resampling at the default timestep",
            m_dt,
            message_verbosity=1,
        )

    # New (resampled) time axis
    time_axis = np.arange(time_axis[0], time_axis[-1], m_dt)
    # Length of data.
    data_len = len(time_axis)
    modes_array.create_modes_array(ell_max=ell_max, data_len=data_len)
    modes_array.time_axis = time_axis

    return modes_array



##################################################################
# Output
##################################################################


def save_modes_data_to_gen(
    wfa,
    ell_max=None,
    pre_key=None,
    key_format=None,
    modes_to_save=None,
    out_file_name="mp_new_modes.h5",
    r_ext_factor=None,
    compression_opts=0,
    r_ext=None,
):
    """Save the waveform mode data to an hdf file.

    Parameters
    ----------
    pre_key: str, optional
             A string containing the key of the group in
             the HDF file in which the modes` dataset exists.
             It defaults to `None`.
    mode_numbers: list
                  The mode numbers to load from the file.
                  Each item in the list is a list that
                  contains two integrer numbers, one for
                  the mode index :math:`\\ell` and the
                  other for the mode index :math:`m`.

    Returns
    -------
    waveform_obj: 3d array
                  Sets the three dimensional array `waveform.modes`
                  that contains the required :math:`\\ell, m` modes.

    Examples
    --------
    >>> from waveformtools.modes_array import ModesArray
    >>> wf = modes_array()
    >>> wf.data_dir = './'
    >>> wf.filename = 'data_file.h5'
    >>> wf.modes_list = [[2, 2], [3, 3]]
    >>> wf.load_gen_data()
    """
    # from waveformtools.modes_array import ModesArray

    #############################
    # I/O assignments.
    #############################

    wfa.out_file_name = wfa.label + "_" + out_file_name
    wfa.out_file_name = wfa.out_file_name.replace(" ", "_")
    wfa.out_file_name = wfa.out_file_name.replace("->", "_")

    # get the full path.
    full_path = wfa.data_dir/wfa.out_file_name

    if r_ext is None:
        if wfa.r_ext is None:
            r_ext = 500
        else:
            r_ext = wfa.r_ext

    if r_ext_factor is None:
        r_ext_factor = wfa.r_ext

    ###################################
    # Identify the modes to save.
    ###################################

    if not modes_to_save:
        if ell_max is not None:
            modes_to_save = wfa.modes_list[:ell_max]

        else:
            modes_to_save = wfa.modes_list

    ##########################
    # Create the modes file.
    ##########################
    message("Saving waveform", wfa.label, message_verbosity=2)
    with h5py.File(full_path, "w") as wfile:
        # Create the metadata dataset.
        metadata = wfa.get_metadata()

        message(metadata, message_verbosity=1)

        metadata_bytes = json.dumps(metadata).encode()

        # dt = h5py.special_dtype(vlen=str)
        # metadata=np.asarray([metadata_bytes], dtype=dt)
        wfile.create_dataset(
            "metadata", data=metadata_bytes, compression_opts=compression_opts
        )

        # Load the modes listed in mode_numbers list
        for item in modes_to_save:
            # For every ell mode list in modes_list

            ell, emm_list = item

            for emm in emm_list:
                # For every (ell, emm) mode.

                data = wfa.mode(ell, emm)
                # set the time and data axis
                data_re = data.real
                data_im = data.imag

                save_data = np.transpose(
                    np.array([wfa.time_axis, data_re, data_im])
                )
                # Make the key
                key = _key_gen(ell, emm, extras=f"r{r_ext:.2f}")
                # message('Processing key', key)
                # Create data set
                wfile.create_dataset(key, data=save_data)

    return 1
