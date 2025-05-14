import os
import re
from pathlib import Path
import scri

import numpy as np

from waveformtools.sxs.prepare_waveforms import PrepareSXSWaveform
from waveformtools.waveformtools import (
    compute_chi_eff_from_masses_and_spins,
    compute_chi_prec_from_masses_and_spins,
    compute_masses_from_mass_ratio_and_total_mass,
    message,
)
import subprocess
import h5py
import pandas as pd
from sxstools.coordinate_transform import CoordinateTransform
import json
from sxstools.quantities import get_dynamics_from_h5, get_t_ref_from_dynamics_and_freq, get_NR_ref_quantities_at_t_ref

class SimulationExplorer:
    """Find and load simulations in a given directory.
    Use this to

    1. find available, running, and failed simulations,
    2. parse their parameters,
    3. compute secondary parameters
    4. export parameter tables to Markdown
    5. Keep a track of already processed waveforms
    6. process waveforms in batch if not i.r. extrapolate and CoM correct

    Attributes
    ----------
    search_dir: str/Path
                The directory containing simulations
    available_sim_names: list of str
                     All discovered simulation names
    available_sim_paths: dict of str/Path
                     The full paths of all found simulations
    highest_ecc_nums: dict of str/Path
                      The highest available Ecc dir across
                      all simulations
    all_sim_params: dict
                    All parsed and available parameters
                    across all discovered simulations.
    all_sim_params_df: DataFrame
                       Same as `all_sim_params` but a sorted data frame
    prepared_waveforms: list of str
                        Simulations whose waveforms had already
                        been processed, that were found in the
                        `prepared_waveforms_dir`
    available_sim_levels: dict of int
                          The available levels within each simulation.
    waveforms_to_prepare: list of str
                          The waveforms that are waiting to be processed.
    processed_waveforms: list of str
                         Simulations whose waveforms have been processed
                         using methods in this class.
    prepared_waveforms_dir: str/Path
                            The full path to the directory
                            where all processed waveforms are
                            being saved.
    bfi_home_dir: str/Path
                  The full path to the BFI home dir
    bfi_project_name: str
                      The name of the BFI project
    bfi_sim_params: dict
                    The simulation parameters as defined in
                    the bfi Project
    inspiral_segments, ringdown_segments: dict
                                          The inspiral and ringdown segements
                                          of all available simulations
    sim_status: dict
                The current status of all sims

    Methods
    -------
    check_for_ecc_dir
        Check if  given directory is an Ecc directory
    search_simulations
        Search for simulations
    find_highest_ecc
        Given the full path to a sim,
        find the highest available Ecc directory
    discover_levels
        find all available levels given the full path
        to a simulation
    get_all_segments
        Given s sim, find all available segments
        across all available levs
    concatenate_ascii_data
        given a sim, lev and a file name, concatenate
        the file across segments.
    get_ecc_dirs
        get all ecc dirs given the full path to a simulation
    find_highest_ecc_dirs
        Discover the highest ecc dirs across all available simulations
    parse_sim_params_input_file
        Load parameters of all the discovered simulations from their
        Params.input file
    parse_sim_target_params_input_file
        Load parameters of all the discovered simulations from their
        TargetParams.input file
    fetch_sim_params
        Fetch the simulation parameters across the discovered
        simulations.
    strip_sim_name_from_waveform_dir
        Given the name of a directory in `prepared_waveforms_dir`,
        parse the simulation name.
    discover_extrapolated_sims
        Given the already-prepared waveforms in `prepared_waveforms_dir`,
        discover the corresponding simulations.
    get_sim_names_from_prepared_waveform_dirs
        Given a list of directories in `prepared_waveforms_dir`,
        get their corresponding simulation names.
    prepare_list_of_sims
        Find simulations that need waveform processing.
    prepare_waveforms
        Process the waveforms
    compute_masses
        COmpute the masses of all BHs from their mass ratios.
    compute_chi_eff
        Compute the :math:`\\chi_{eff}` parameters of all the
        simulations
    compute_chi_prec
        Compute the :math:`\\chi_{prec}` parameters of all the
        simulations
    get_all_chi_eff
        Fetch the :math:`\\chi_{eff}` parameter of all simulations
    get_all_chi_prec
        Fetch the :math:`\\chi_{prec}` parameter of all simulations
    get_all_mass_ratios
        Fetch the mass-ratio :math:`q` parameter of all simulations
    write_history
        Write processed waveforms in this session to a history file.
    """

    def __init__(
        self,
        search_dir="./",
        prepared_waveforms_dir=None,
        bfi_project_name=None,
        bfi_home_dir=None,
        ref_time=None,
        filter_items=[
            "Sl. No",
            "Status",
            "ReqRefTime",
            "Omega0",
            "Omega_ref",
            "Ncycles",
            "Ncycles (2,2)",
            "D0",
            "MassRatio",
            "ChiA",
            "ChiA_ref",
            "ChiB",
            "ChiB_ref",
            "ChiEff",
            "ChiPrec",
        ],
    ):
        self._search_dir = Path(search_dir)
        self._prepared_waveforms_dir = Path(prepared_waveforms_dir)
        self._ref_time = ref_time
        self._filter_items = filter_items
        self._bfi_project_name = bfi_project_name
        self._bfi_home_dir = Path(bfi_home_dir)
        self._sim_basename = None
        self._bfi_sim_params = {}
        self._all_req_ref_times = {}
        self._all_req_ref_time_segments = []
        if bfi_project_name is None:
            self.parse_bfi=False
        else:
            self.parse_bfi=True

    @property
    def search_dir(self):
        return self._search_dir

    @property
    def filter_items(self):
        return self._filter_items

    @property
    def sim_basename(self):
        return self._sim_basename

    @property
    def bfi_project_name(self):
        return self._bfi_project_name

    @property
    def bfi_home_dir(self):
        return self._bfi_home_dir

    @property
    def bfi_sim_params(self):
        return self._bfi_sim_params

    @property
    def available_sim_names(self):
        return self._available_sim_names

    @property
    def available_sim_paths(self):
        return self._available_sim_paths

    @property
    def highest_ecc_nums(self):
        return self._highest_ecc_nums

    @property
    def all_sim_params(self):
        return self._all_sim_params

    @property
    def all_sim_params_df(self):
        return self._all_sim_params_df

    @property
    def prepared_waveforms(self):
        return self._prepared_waveforms

    @property
    def available_sim_levs(self):
        return self._available_sim_levs

    @property
    def processed_waveforms(self):
        return self._processed_waveforms

    @property
    def prepared_waveforms_dir(self):
        return self._prepared_waveforms_dir

    @property
    def waveforms_to_prepare(self):
        return self._waveforms_to_prepare

    @property
    def inspiral_segments(self):
        return self._inspiral_segments

    @property
    def ringdown_segments(self):
        return self._ringdown_segments

    @property
    def sim_status(self):
        return self._sim_status

    @property
    def ncycles(self):
        return self._ncycles

    @property
    def ncycles_wf(self):
        return self._ncycles_wf

    @property
    def ref_time(self):
        return self._ref_time

    @property
    def all_req_ref_times(self):
        return self._all_req_ref_times

    @property
    def all_req_ref_time_segments(self):
        return self._all_req_ref_time_segments

    @property
    def deleted_sims(self):
        return self._deleted_sims

    def check_for_ecc_dir(self, path_of_dir_to_check):
        """Check if the given directory is an Ecc directory"""

        flag = True

        message(f"Checking {path_of_dir_to_check}", message_verbosity=3)

        if not os.path.isdir(path_of_dir_to_check):
            flag = False
            message(
                f"{path_of_dir_to_check} is not a directory",
                message_verbosity=3,
            )

            return flag

        subsubdirs = os.listdir(path_of_dir_to_check)

        if "Ecc0" not in subsubdirs:
            flag = False
            message(
                f"{path_of_dir_to_check} does not contain Ecc0",
                message_verbosity=3,
            )
            return flag

        ecc0_dir = os.listdir(path_of_dir_to_check.joinpath("Ecc0"))

        if "DoMultipleRuns.input" not in ecc0_dir:
            flag = False
            message(
                f"{path_of_dir_to_check} does not contain"
                "DoMultipleRuns.input",
                message_verbosity=3,
            )

        if flag:
            message(
                f"\t {path_of_dir_to_check} is a simulation directory",
                message_verbosity=3,
            )

        return flag

    def search_simulations(self):
        """Search for simulations within the search directories"""

        available_sim_paths = {}
        available_sim_names = []
        one_search_dir = self.search_dir
        message(f"Searching in {one_search_dir}", message_verbosity=1)
        subdirs = os.listdir(one_search_dir)
        message(f"All subdirs {subdirs}", message_verbosity=2)

        for possible_sim_name in subdirs:
            possible_sim_path = one_search_dir.joinpath(possible_sim_name)
            message(f"\tSearching in {possible_sim_path}", message_verbosity=2)

            flag = self.check_for_ecc_dir(possible_sim_path)

            if flag:
                available_sim_paths.update(
                    {possible_sim_name: possible_sim_path}
                )

                available_sim_names.append(possible_sim_name)

        self._available_sim_names = available_sim_names
        self._available_sim_paths = available_sim_paths

        self.compute_sim_basename()
        self.find_highest_ecc_dirs()

    def find_highest_ecc(self, sim_path):
        """Find the highest ecc level in a given sim dir"""

        ecc_dirs = self.get_ecc_dirs(sim_path)
        available_ecc_nums = [item.strip("Ecc") for item in ecc_dirs]
        available_ecc_nums = [
            int(item) for item in available_ecc_nums if item.isdigit()
        ]

        # print(available_eccs)

        highest_ecc_num = np.argmax(available_ecc_nums)

        return highest_ecc_num

    def discover_levels_old(self):
        """Discover all avaiable levels across all simulations"""

        all_levs = {}

        for sim_name, sim_path in self.available_sim_paths.items():
            sim_levs = []

            path_to_highest_ecc_dir = sim_path.joinpath(
                f"Ecc{self.highest_ecc_nums[sim_name]}/Ev"
            )

            subdirs = os.listdir(path_to_highest_ecc_dir)

            subdirs = [item for item in subdirs if "Ringdown" in item]

            if subdirs is None:
                continue

            levs = [item for item in subdirs if "Lev" in item]
            levs = [item for item in levs if "SysInit" not in item]

            for item in levs:
                result = re.search("Lev[0-9]_Ringdown", item)

                lev_num = int(result.group()[3:4])

                sim_levs.append(lev_num)

            sim_levs = list(set(sim_levs))

            all_levs.update({sim_name: sim_levs})

        self._available_sim_levs = all_levs

    def discover_levels(self):
        """Discover all avaiable levels across all simulations"""

        all_levs = {}

        for sim_name in self.available_sim_names:
            sim_levs = []

            for sim_lev in self.inspiral_segments[sim_name].keys():

                this_lev = int(sim_lev[3:])
                sim_levs.append(this_lev)

            all_levs.update({sim_name: sim_levs})

        self._available_sim_levs = all_levs

    def discover_segments(self):
        """Find all the segments across all simulations"""
        import os

        all_inspiral_segments = {}
        all_ringdown_segments = {}

        for sim_name in self.available_sim_names:

            message(f"Loading segments in {sim_name}", message_verbosity=2)
            ecc_ev_dir = self.available_sim_paths[sim_name].joinpath(
                f"Ecc{self.highest_ecc_nums[sim_name]}/Ev"
            )
            inspiral_segments, ringdown_segments = (
                self.get_all_one_sim_segments(ecc_ev_dir)
            )
            all_inspiral_segments.update({sim_name: inspiral_segments})
            all_ringdown_segments.update({sim_name: ringdown_segments})

        self._inspiral_segments = all_inspiral_segments
        self._ringdown_segments = all_ringdown_segments

    def get_all_one_sim_segments(self, ecc_ev_dir):
        """Get the segments within one Ecc dir"""
        import os

        inspiral_segments = {}
        ringdown_segments = {}

        available_dirs = os.listdir(ecc_ev_dir)
        available_dirs = [item for item in available_dirs if ".bak" not in item]
        message("Available dirs", available_dirs, message_verbosity=2)
        available_lev_dirs = [item for item in available_dirs if "Lev" in item]
        available_lev_dirs = [
            item for item in available_lev_dirs if "SysInit" not in item
        ]
        available_lev_dirs = [
            item for item in available_lev_dirs if "Ringdown" not in item
        ]
        available_lev_dirs = [
            item
            for item in available_lev_dirs
            if re.search("Lev[0-9]_[A-Z]+", item) is not None
        ]
        message("Available lev dirs", available_lev_dirs, message_verbosity=2)
        available_ringdown_lev_dirs = [
            item for item in available_dirs if "_Ringdown" in item
        ]
        available_ringdown_lev_dirs = [
            item
            for item in available_ringdown_lev_dirs
            if re.search("Lev[0-9]_Ringdown", item) is not None
        ]
        message(
            "Available ringdown lev dirs",
            available_ringdown_lev_dirs,
            message_verbosity=2,
        )

        available_levs = set([int(item[3:-3]) for item in available_lev_dirs])
        message("Available levs", available_levs, message_verbosity=2)

        available_ringdown_levs = set(
            [int(item[3:-9]) for item in available_ringdown_lev_dirs]
        )
        message(
            "Available ringdown levs",
            available_ringdown_levs,
            message_verbosity=2,
        )

        # Fetch segments
        for lev in available_levs:

            this_lev_segments = [
                item for item in available_lev_dirs if f"Lev{lev}" in item
            ]
            this_lev_segments = sorted([item[5:] for item in this_lev_segments])
            inspiral_segments.update({f"Lev{lev}": this_lev_segments})

            if lev in available_ringdown_levs:
                inside_ringdown_dirs = os.listdir(
                    ecc_ev_dir.joinpath(f"Lev{lev}_Ringdown")
                )
                inside_ringdown_segments = [
                    item for item in inside_ringdown_dirs if "Lev" in item
                ]
                inside_ringdown_segments = [
                    item
                    for item in inside_ringdown_segments
                    if "SysInit" not in item
                ]
                inside_ringdown_segments = [
                    item
                    for item in inside_ringdown_segments
                    if re.search("Lev[0-9]_[A-Z]+", item) is not None
                ]
                this_lev_ringdown_segments = sorted(
                    [item[5:] for item in inside_ringdown_segments]
                )
                ringdown_segments.update(
                    {f"Lev{lev}": this_lev_ringdown_segments}
                )

        return inspiral_segments, ringdown_segments

    def concatenate_ascii_data(self, fname, ecc, lev, sim_dir):
        """Concatenate the data in the given file name, lev,
        ecc dir"""

        full_ecc_path = f"{sim_dir}/Ecc{ecc}/Ev/"

        # get segments
        all_super_segments = self.get_all_segments(full_ecc_path)

        all_segments = all_super_segments[f"Lev{lev}"]

        all_data = []

        for item in all_segments:
            file = f"{full_ecc_path}/Lev{lev}_{item}/Run/{fname}"

            one_segment_data = np.genfromtxt(file)
            all_data.append(one_segment_data)

        all_data = np.concatenate((all_data))

        return all_data

    def get_ecc_dirs(self, sim_path):
        """Get all ecc directories within a sim dir"""
        subdirs = os.listdir(sim_path)

        ecc_dirs = [item for item in subdirs if "Ecc" in item]

        return ecc_dirs

    def find_highest_ecc_dirs(self):
        """Find and load the highest ecc level available"""

        highest_ecc_nums = {}

        for sim_name, sim_path in self.available_sim_paths.items():
            highest_ecc_num = self.find_highest_ecc(sim_path)

            highest_ecc_nums.update({sim_name: highest_ecc_num})

        self._highest_ecc_nums = highest_ecc_nums

    def compute_sim_basename(self):
        """Compute the sim basename from the dir names"""

        one_sim_name = self.available_sim_names[-1]

        r1 = re.search("[a-zA-Z][0-9]+", one_sim_name)

        end_ind = r1.start() + 1

        self._sim_basename = one_sim_name[:end_ind]

    def parse_sim_params_input_file(self, path_to_params_input_file):
        """Parse all sim params given path to the Params.input file"""

        sim_params = {}

        message(
            "Parsing simulation parameters.input file"
            f"for {path_to_params_input_file}",
            message_verbosity=2,
        )

        with open(path_to_params_input_file, "r") as pfile:
            eof = False

            while not eof:
                line = pfile.readline()

                if "Omega0" in line:
                    val = re_fetch_float(line)

                    sim_params.update({"Omega0": val})

                if "adot0" in line:
                    val = re_fetch_float(line)

                    sim_params.update({"adot0": val})

                if "D0" in line:
                    val = re_fetch_float(line)

                    sim_params.update({"D0": val})

                if "MassRatio" in line:
                    val = re_fetch_float(line)

                    sim_params.update({"MassRatio": val})

                if "SpinA" in line:
                    val = re_fetch_vector(line)

                    sim_params.update({"ChiA": val})

                if "SpinB" in line:
                    val = re_fetch_vector(line)

                    sim_params.update({"ChiB": val})

                if "$IDType" in line:
                    val = re_fetch_string(line)

                    sim_params.update({"IDType": val})

                    eof = True

        return sim_params

    def parse_sim_target_params_input_file(
        self, path_to_target_params_input_file
    ):
        """Parse a TargetParams.input file given a sim"""

        if not os.path.exists(path_to_target_params_input_file):
            return None

        else:
            target_params = {}

            with open(path_to_target_params_input_file, "r") as tpfile:
                eof = False

                while not eof:
                    line = tpfile.readline()

                    if "$ReferenceTime" in line:
                        val = re_fetch_float(line)

                        target_params.update({"ReferenceTime": val})

                    if "$SemiMajorAxis" in line:
                        val = re_fetch_float(line)

                        target_params.update({"SemiMajorAxis": val})

                    if "$Eccentricity" in line:
                        val = re_fetch_float(line)

                        target_params.update({"Eccentricity": val})

                    if "$AnomalyAngle" in line:
                        val = re_fetch_float(line)

                        target_params.update({"AnomalyAngle": val})

                    if "$IDType" in line:
                        eof = True

            return target_params

    def parse_bfi_params_file(self):
        """Parse parameters from a BFI params file"""

        bfi_sim_params = {}
        bfi_sim_basename = self.bfi_project_name

        if self.sim_basename != bfi_sim_basename:
            message(
                f"sim base name: {self.sim_basename} \t"
                f"bfi project name: {bfi_sim_basename}"
            )
            raise KeyError(
                "The BFI project name "
                "do not match sim basenames found in search dir"
            )

        else:
            projects_rel_dir = Path(
                f"ParamFiles/{self.bfi_project_name}/params"
            )
            bfi_params_file = self.bfi_home_dir.joinpath(projects_rel_dir)

            with open(bfi_params_file, "r", encoding="utf-8") as pf:
                for line in pf:
                    if line is not None:
                        # Get params
                        results = re.findall("[a-zA-Z]+=[0-9]+[.]?[0-9]*", line)
                        fresults = [
                            item for item in results if "chi" not in item
                        ]
                        fresults = [
                            item for item in fresults if "q" not in item
                        ]
                
                        if len(fresults) == 0:
                            continue

                        message("fresults", fresults, message_verbosity=3)
                        # Assuming that ID is the first element
                        sim_name = self.sim_basename + fresults[0].split("=")[1]
                        bfi_sim_params.update({sim_name: {}})

                        for item in fresults[1:]:

                            key, value = item.split("=")
                            value = float(value)
                            bfi_sim_params[sim_name].update({key: value})
        self._bfi_sim_params = bfi_sim_params

        # Delete empty sims from bfi file
        for item in self.deleted_sims:
            self._bfi_sim_params.pop(item)

        self.append_bfi_sim_params_to_all_params()

    def append_bfi_sim_params_to_all_params(self):
        """Append the contents of the BFI sim params to the master
        params dict"""

        for sim_name, sim_params in self.bfi_sim_params.items():
            # for sim_name in self.available_sim_names:

            sim_params = self.bfi_sim_params[sim_name]
            available_sim_param_keys = self.all_sim_params[sim_name].keys()

            for key, value in sim_params.items():

                if key not in available_sim_param_keys:
                    self._all_sim_params[sim_name].update({key: value})

    def fetch_sim_params(self):
        """Get all the simulations' parameters"""

        all_sim_params = {}

        for sim_name, sim_path in self.available_sim_paths.items():
            ecc_num = self.highest_ecc_nums[sim_name]
            path_to_params_input_file = sim_path.joinpath(
                f"Ecc{ecc_num}"
            ).joinpath("Params.input")
            path_to_target_params_input_file = sim_path.joinpath(
                f"Ecc{ecc_num}"
            ).joinpath("TargetParams.input")

            one_sim_params_dict = self.parse_sim_params_input_file(
                path_to_params_input_file
            )
            one_sim_target_params_dict = (
                self.parse_sim_target_params_input_file(
                    path_to_target_params_input_file
                )
            )
            one_sim_params = one_sim_params_dict

            if one_sim_target_params_dict is not None:
                one_sim_params.update(one_sim_target_params_dict)

            all_sim_params.update({sim_name: one_sim_params})

        self._all_sim_params = all_sim_params
        if self.parse_bfi:
            self.parse_bfi_params_file()
        self.compute_chi_eff()
        self.compute_chi_prec()
        self.compute_ncycles()
        self.compute_ncycles_from_waveform()
        self.prepare_pandas_dataframe()

    def prepare_pandas_dataframe(self):
        """Prepare a pandas dataframe from the sim params
        found"""

        import pandas as pd

        all_params_df = pd.DataFrame(self.all_sim_params)
        all_params_df = all_params_df.transpose()
        all_params_df = all_params_df.sort_index()
        all_params_df.insert(0, "Sl. No.", range(1, 1 + len(all_params_df)))
        all_params_df.insert(1, "Status", self.get_basic_sim_status())
        all_params_df.insert(3, "Ncycles", self.ncycles)
        all_params_df.insert(4, "Ncycles (2,2)", self.ncycles_wf)


        self._all_sim_params_df = all_params_df

    def get_basic_sim_status(self):
        """Insert status of individual sim levs
        into the dataframe"""

        basic_sim_status_dict = {}

        for sim_name_with_lev, lev_status in self.sim_status["Status"].items():
        #for sim_name_with_lev in self.sim_status
            print(sim_name_with_lev)
            sim_name, lev_name = sim_name_with_lev.split("_")
            lev_num = int(lev_name[3:])

            if not sim_name in basic_sim_status_dict.keys():
                basic_sim_status_dict.update({sim_name: {lev_num: lev_status}})
            else:
                basic_sim_status_dict[sim_name].update({lev_num: lev_status})

        return basic_sim_status_dict

    def update_pandas_dataframe(self):
        import pandas as pd

        self._all_sim_params_df = pd.DataFrame(
            self.all_sim_params
        ).T.sort_index()

        self.reference_parameters_df = pd.DataFrame(
            self.ref_parameters
        ).T.sort_index()

        self.prepare_pandas_dataframe()

    def strip_sim_name_from_waveform_dir(self, wdir):
        """Extract the sim name from its processed waveform
        dir name"""
        result = re.search("[A-Za-z]+[0-9]+_", wdir)

        if result is not None:
            return result.group()[:-1]

        else:
            return None

    def discover_extrapolated_sims(self):
        """Read in all available waveform directories
        in the given directory and create a list of sims
        whose waveforms have already been prepared"""

        waveforms_dir = self.prepared_waveforms_dir

        all_wfs = os.listdir(waveforms_dir)

        message("Contents of waveforms dir", message_verbosity=2)

        all_wfs = [
            item
            for item in all_wfs
            if os.path.isdir(waveforms_dir.joinpath(item))
        ]

        message("Dirs of waveforms dir", message_verbosity=2)

        # Strip suffix
        sim_names = self.get_sim_names_from_prepared_waveform_dirs(all_wfs)

        message("sims in waveforms dir", message_verbosity=2)
        self._prepared_waveforms = sim_names

    def get_sim_names_from_prepared_waveform_dirs(self, prepared_waveform_dirs):
        """Get all sim names from prepared waveform dirs"""
        prepared_waveform_sim_names = []

        for item in prepared_waveform_dirs:
            one_sim_name = self.strip_sim_name_from_waveform_dir(item)

            if one_sim_name is not None:
                prepared_waveform_sim_names.append(one_sim_name)

        prepared_waveform_sim_names = list(set(prepared_waveform_sim_names))

        return prepared_waveform_sim_names

    def prepare_list_of_sims(self):
        """Prepare a non-redundant list of simulations
        to run waveform preps for"""

        self.discover_extrapolated_sims()

        waveforms_to_prep = []

        for sim_name in self.available_sim_names:
            if (sim_name not in self.prepared_waveforms) and (
                self.available_sim_levs[sim_name] is not None
            ):
                waveforms_to_prep.append(sim_name)

        self._waveforms_to_prepare = waveforms_to_prep

    def prepare_waveforms(self, simulations_to_prepare=None):
        """Prepare waveforms that havent been prepared yet
        by extrapolating, CoM correcting"""

        self.prepare_list_of_sims()

        processed_waveforms = {}

        if (np.array(simulations_to_prepare) == np.array(None)).any():
            simulations_to_prepare = self.waveforms_to_prepare

        else:
            simulations_to_prepare = [
                item
                for item in simulations_to_prepare
                if item not in self.prepared_waveforms
            ]

        for sim_name in simulations_to_prepare:
            processed_levs = []

            available_sim_levs = self.available_sim_levs[sim_name]

            ecc = [self.highest_ecc_nums[sim_name]]

            success_sims = []

            for lev in available_sim_levs:
                message(
                    "Preparing waveform for sim ",
                    sim_name,
                    "Lev ",
                    lev,
                    message_verbosity=2,
                )

                try:
                    wfp = PrepareSXSWaveform(
                        sim_name=sim_name,
                        sim_dir=self.available_sim_paths[sim_name],
                        lev=lev,
                        out_dir=self.prepared_waveforms_dir\
                        / f"{sim_name}_waveforms_Lev{lev}",
                        ecc=self.highest_ecc_nums[sim_name]
                    )

                    flag = wfp.prepare_waveform()

                    if flag:
                        processed_levs.append(lev)

                except Exception as excep:
                    message(excep)
                    message(
                        "Waveform processing failed sim"
                        f"{sim_name}, lev {lev}"
                    )

            processed_waveforms.update({sim_name: processed_levs})

        self._processed_waveforms = processed_waveforms

    def compute_masses(self):
        """Compute the individual masses of all sims"""
        for sim_name in self.available_sim_names:
            mass1, mass2 = compute_masses_from_mass_ratio_and_total_mass(
                self.all_sim_params[sim_name]["MassRatio"]
            )

            self.all_sim_params[sim_name].update(
                {"IndividualMasses": (mass1, mass2)}
            )

    def compute_chi_eff(self):
        """Compute the :math:`\\chi_{eff} parameter` across all sims"""
        for sim_name in self.available_sim_names:
            mass_ratio = self.all_sim_params[sim_name]["MassRatio"]

            spin1 = self.all_sim_params[sim_name]["ChiA"]
            spin2 = self.all_sim_params[sim_name]["ChiB"]

            chi_eff = compute_chi_eff_from_masses_and_spins(
                spin1, spin2, mass_ratio
            )

            self.all_sim_params[sim_name].update({"ChiEff": chi_eff})

    def compute_chi_prec(self):
        """Compute the :math:`\\chi_{prec}` parameters across all sims"""
        for sim_name in self.available_sim_names:
            mass_ratio = self.all_sim_params[sim_name]["MassRatio"]

            spin1 = self.all_sim_params[sim_name]["ChiA"]
            spin2 = self.all_sim_params[sim_name]["ChiB"]

            chi_prec = compute_chi_prec_from_masses_and_spins(
                spin1, spin2, mass_ratio
            )

            self.all_sim_params[sim_name].update({"ChiPrec": chi_prec})

    def get_all_chi_eff(self):
        """Fetch all :math:`\\chi_{eff}` parameters"""
        all_chi_eff = []

        for sim_name in self.available_sim_names:
            all_chi_eff.append(self.all_sim_params[sim_name]["ChiEff"])

        return all_chi_eff

    def get_all_mass_ratios(self):
        """Fetch all mass-ratios"""

        all_mass_ratios = []

        for sim_name in self.available_sim_names:
            all_mass_ratios.append(self.all_sim_params[sim_name]["MassRatio"])

        return all_mass_ratios

    def write_history(self):
        raise NotImplementedError

        with open(self.history_file, "a") as th:
            for item in dirs:
                th.write(item)
                th.write("\n")

    def get_last_segment(self, sim_name, lev):
        """Get the last available segment of a sim"""

        message(
            f"Fetching last segment for sim {sim_name}, lev{lev}",
            message_verbosity=2,
        )

        RINGDOWN_INITIATED = False

        ringdown_segments = self.ringdown_segments[sim_name]

        if ringdown_segments:

            lev_key = f"Lev{lev}"

            if (
                lev_key in ringdown_segments.keys()
                and len(ringdown_segments[lev_key]) > 0
            ):
                RINGDOWN_INITIATED = True
                last_segment = sorted(ringdown_segments[lev_key])[-1]
                last_segment_path = self.available_sim_paths[sim_name].joinpath(
                    f"Ecc{self.highest_ecc_nums[sim_name]}"
                    f"/Ev/Lev{lev}_Ringdown/Lev{lev}_{last_segment}"
                )

            else:
                RINGDOWN_INITIATED = False

        if RINGDOWN_INITIATED is False:
            inspiral_segments = self.inspiral_segments[sim_name]

            if inspiral_segments:
                last_segment = sorted(inspiral_segments[f"Lev{lev}"])[-1]
                last_segment_path = self.available_sim_paths[sim_name].joinpath(
                    f"Ecc{self.highest_ecc_nums[sim_name]}"
                    f"/Ev/Lev{lev}_{last_segment}"
                )

            else:
                raise KeyError(f"No segments in sim {sim_name}")

        return last_segment, last_segment_path, RINGDOWN_INITIATED

    def get_segment_jobid(self, last_segment_path):
        # Check if segment is running
        # Get Jobid

        all_jobids = []

        spec_jobid_file = last_segment_path.joinpath("SpEC.jobid")

        if not os.path.exists(spec_jobid_file):
            return -1

        with open(spec_jobid_file, "r", encoding="utf-8") as sjf:
            for line in sjf:
                one_jobid = line.split()[0]
                all_jobids.append(int(one_jobid))

        latest_jobid = all_jobids[-1]
        return latest_jobid

    def check_if_job_is_running(self, jobid):
        """Check if a segment is runnning"""

        message(
            f"Checking if the job {jobid} is running...", message_verbosity=2
        )

        # Get running jobs
        cmd_stdout = subprocess.run(
            ["squeue -u ${USER} | awk '{print $1}'"],
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout

        cmd_stdout_status = subprocess.run(
            ["squeue -u ${USER} | awk '{print $5}'"],
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout

        cmd_stdout_nodes = subprocess.run(
            ["squeue -u ${USER} | awk '{print $8}'"],
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout

        running_jobs = cmd_stdout.decode().split("\n")[1:-1]
        running_jobs = [int(item) for item in running_jobs]

        # Get status
        jobs_status = cmd_stdout_status.decode().split("\n")[1:-1]

        nodes = cmd_stdout_nodes.decode().split("\n")[1:-1]

        jobs = {}

        for index in range(len(running_jobs)):
            one_jobid = running_jobs[index]
            one_status = jobs_status[index]
            one_node = nodes[index]

            jobs.update({one_jobid: {"Status": one_status, "Node": one_node}})

        # jobs = dict(zip(running_jobs, jobs_status))

        message("Running jobs ", running_jobs, message_verbosity=2)
        message("sim job id", jobid, message_verbosity=2)

        if jobid in running_jobs:
            status = jobs[jobid]["Status"]
            message("Segment running", message_verbosity=2)

            if status == "R":
                SEGMENT_STATUS = "Running"
            elif status == "PD":
                SEGMENT_STATUS = "Queued"
            else:
                SEGMENT_STATUS = "Unknown"
        else:
            SEGMENT_STATUS = "Not Running"

        try:
            jobs[jobid]["Status"] = SEGMENT_STATUS
            return jobs[jobid]
        except:
            return None

    def check_for_errors(self, last_segment_path):
        """Check for error files to learn about the status"""

        # Check for errors
        run_dir = last_segment_path.joinpath("Run")

        files_in_run_dir = os.listdir(run_dir)

        error_files = [item for item in files_in_run_dir if "Errors000" in item]

        message("Error files", error_files, message_verbosity=2)

        if len(error_files) > 0:
            SEGMENT_ERROR = True
        else:
            SEGMENT_ERROR = False
            message("Sim status is unknown", message_verbosity=2)
            status = "Unknown"

        return SEGMENT_ERROR

    def read_last_segment_status(self, sim_name, lev):
        """Find out the status of the last segment of a particular sim"""

        one_sim_status = {}

        last_segment, last_segment_path, RINGDOWN_INITIATED = (
            self.get_last_segment(sim_name, lev)
        )

        if RINGDOWN_INITIATED:
            last_segment = "RD_" + last_segment

        sim_key = f"{sim_name}_Lev{lev}"
        one_sim_status.update({sim_key: {"Segment": last_segment}})

        SIM_COMPLETE = False
        SEGMENT_ERROR = False
        status = None
        SEGMENT_TO_SUBMIT = False
        # SEGMENT_RUNNING=False
        allocation = None
        # Check not submitted
        spec_run_dir = last_segment_path.joinpath("Run")
        spec_out_file = last_segment_path.joinpath("Run/SpEC.out")

        if not os.path.exists(spec_run_dir):
            SEGMENT_COMPLETE = False
            SIM_COMPLETE = False
            # SEGMENT_RUNNING = SEGMENT_RUNNING_STATUS
            SEGMENT_TO_SUBMIT = True

            SEGMENT_RUNNING_STATUS = "Not Running"
            status = "To be submitted"
            # else:
            # status = SEGMENT_RUNNING_STATUS
            segment_status_comment = "No Run dir"
        elif not os.path.exists(spec_out_file):
            SEGMENT_COMPLETE = False
            SIM_COMPLETE = False
            SEGMENT_RUNNING = "Not Running"

            segment_status_comment = "SpEC.out not found"
            status = "Halted"

        # Check if terminated
        else:

            segment_jobid = self.get_segment_jobid(last_segment_path)

            if segment_jobid < 0:
                SEGMENT_RUNNING_STATUS = "Not Running"
            else:
                seg_status = self.check_if_job_is_running(segment_jobid)

                if seg_status is not None:
                    SEGMENT_RUNNING_STATUS = seg_status["Status"]
                    allocation = seg_status["Node"]
                else:
                    SEGMENT_RUNNING_STATUS = "Not Running"

            with open(spec_out_file, "r", encoding="utf-8") as sof:
                lines = sof.readlines()[-3:]

                message("Lines from SpEC.out", lines, message_verbosity=2)

                message(
                    "Segment runnint 2",
                    SEGMENT_RUNNING_STATUS,
                    message_verbosity=2,
                )
                SEGMENT_ERROR = self.check_for_errors(last_segment_path)

                if "To be continued...\n" in lines:
                    SEGMENT_COMPLETE = True
                    SIM_COMPLETE = False
                    assert (
                        SEGMENT_RUNNING_STATUS == "Not Running"
                    ), "Segment was found to be completed. It cant be running!"
                    assert (
                        SEGMENT_ERROR == False
                    ), "Segment was found to have errors but also complete!"

                    segment_status_comment = [
                        item
                        for item in lines
                        if "Termination condition" in item
                    ][-1]

                    status = "Halted"

                elif "Termination condition FinalTime\n" in lines:
                    SEGMENT_COMPLETE = True
                    SIM_COMPLETE = True
                    assert (
                        SEGMENT_RUNNING_STATUS == "Not Running"
                    ), "Segment was found to be completed. It cant be running!"

                    # assert SEGMENT_ERROR == False, "Segment was found
                    # to have errors but also complete!"
                    if SEGMENT_ERROR == True:
                        message(
                            "Segment was found to have errors but also complete!"
                        )

                    segment_status_comment = "Termination condition FinalTime"
                    status = "Completed"

                elif "t=" in lines[-1]:
                    SEGMENT_COMPLETE = False
                    SIM_COMPLETE = False

                    if SEGMENT_RUNNING_STATUS == "Running":
                        status = "Running"
                    elif SEGMENT_ERROR:
                        status = "Error"
                    else:
                        status = "Halted"

                    segment_status_comment = status
                else:
                    if SEGMENT_ERROR:
                        status = "Error"

                    else:
                        status = "Unknown"
                segment_status_comment = status

        one_sim_status[sim_key].update({"Status": status})

        one_sim_status[sim_key].update({"Comment": segment_status_comment})

        one_sim_status[sim_key].update({"Allocation": allocation})

        return one_sim_status

    def construct_simulation_status(self):
        """Construct the simulation status of all the sims found

        Check if Ringdown segment
            Check TerminationCondition
                If finished
                If not CheckRunning
        Elseif Check Runnung
        Else NotRunning

        NotRunning:
            CheckFailed
        """

        sim_status = {}

        for sim_name in self.available_sim_names:
            # one_sim_status = {}
            for lev in self.available_sim_levs[sim_name]:
                one_status = self.read_last_segment_status(sim_name, lev)
                print(one_status)
                sim_status.update(one_status)

            # sim_status.update({sim_name: one_sim_status})

        self._sim_status = pd.DataFrame(sim_status).transpose().sort_index()

    def compute_ncycles(self):
        """Compute the number of cycles using the orbital
        revolutions"""

        import h5py
        all_sim_ncycles = {}

        for sim_name in self.available_sim_names:
            one_sim_ncycles = {}

            for lev in self.available_sim_levs[sim_name]:

                message(
                    f"Computing ncycles for {sim_name} Lev{lev}",
                    message_verbosity=2,
                )
                path_to_joined_h5 = self.join_horizons_file(sim_name, lev)

                try:
                    hf = h5py.File(path_to_joined_h5)
                except Exception as excep:
                    message(excep, f"Skipping {sim_name} Lev{lev}")
                    continue

                CA = hf["AhA.dir"]["CoordCenterInertial.dat"][...]
                CB = hf["AhB.dir"]["CoordCenterInertial.dat"][...]
                tA = CA[:, 0]
                xA = CA[:, 1]
                yA = CA[:, 2]
                zA = CA[:, 3]
                tB = CB[:, 0]
                xB = CB[:, 1]
                yB = CB[:, 2]
                zB = CB[:, 3]
                dx = xA - xB
                dy = yA - yB
                dz = zA - zB
                # Assuming low precession of orbital plane
                dRxy = dx + 1j * dy
                phase = np.unwrap(np.angle(dRxy))

                if np.mean(np.diff(phase)) < 0:
                    phase = -phase

                assert (
                    min(phase) == phase[0]
                ), "Phase must be min at the beginning"
                assert max(phase) == phase[-1], "Phase must be max at the end"
                phase = phase - min(phase)
                ncycles = max(phase / (2 * np.pi))
                one_sim_ncycles.update({lev: round(ncycles * 2,1)})
                hf.close()

            all_sim_ncycles.update({sim_name: one_sim_ncycles})

        self._ncycles = all_sim_ncycles

    def compute_ncycles_from_waveform(self):
        """Compute the ncycles from the 2,2 mode waveform"""

        import h5py

        all_wf_ncycles = {}

        for sim_name in self.available_sim_names:
            one_wf_ncycles = {}

            for lev in self.available_sim_levs[sim_name]:
                message(
                    f"Computing ncycles for {sim_name} Lev{lev}",
                    message_verbosity=2,
                )
                path_to_extrap_waveform_h5 = self.get_waveform_out_path(
                    sim_name, lev
                )

                try:
                    hf = h5py.File(path_to_extrap_waveform_h5)

                except Exception as excep:
                    message(excep, f"Skipping {sim_name} Lev{lev}")
                    continue

                h22 = hf["Y_l2_m2.dat"][...]
                h22_re = h22[:, 1]
                h22_im = h22[:, 2]
                h22_cmplx = h22_re + 1j * h22_im
                h22_maxloc = np.argmax(abs(h22_cmplx))
                phase_22 = np.unwrap(np.angle(h22_cmplx[:h22_maxloc]))

                if np.mean(np.diff(phase_22)) < 0:
                    phase_22 = -phase_22

                phase_22 = phase_22 - min(phase_22)
                ncycles = max(phase_22 / (2 * np.pi))
                one_wf_ncycles.update({lev: round(ncycles,1)})
                hf.close()

            all_wf_ncycles.update({sim_name: one_wf_ncycles})

        self._ncycles_wf = all_wf_ncycles

    def get_waveform_out_path(self, sim_name, lev, N=2):
        """Fetch the waveform output path"""

        wpath = (
            self.prepared_waveforms_dir / f"{sim_name}_waveforms_Lev{lev}/"
            f"extrapolated/rhOverM_Extrapolated_N{N}_CoM.h5"
        )
        
        return wpath

    def join_horizons_file(self, sim_name, lev):
        """Join the horizons.h5 file across segments for
        a simulation and return the path"""

        path_to_joined_h5 = (
            self.prepared_waveforms_dir
            / f"{sim_name}_waveforms_Lev{lev}/joined/"
            f"{sim_name}Lev{lev}JoinedHorizons.h5"
        )

        if os.path.exists(path_to_joined_h5):
            message(
                f"Joined Horizons.h5 file "
                "already found in {path_to_joined_h5}",
                message_verbosity=2,
            )

        else:
            wfp = PrepareSXSWaveform(
                sim_name=sim_name,
                sim_dir=self.available_sim_paths[sim_name],
                lev=lev,
                out_dir=self.prepared_waveforms_dir,
                ecc=self.highest_ecc_nums[sim_name]
            )

            wfp.join_horizons()
            path_to_joined_h5 = wfp.joined_horizons_outfile_path

        return path_to_joined_h5

    def discover_ref_time(self):
        """Discover ref time as the light travel time to the
        innermost waveform extraction sphere"""

        all_req_ref_times = {}

        for sim_name, sim_path in self.available_sim_paths.items():
            try:
                one_sim_lev = self.available_sim_levs[sim_name][0]
            except IndexError:
                continue

            one_sim_ecc = self.get_ecc_dirs(sim_path)[-1]
            sub_path = f"{one_sim_ecc}/Ev/Lev{one_sim_lev}_AA/Run/GW2/"\
            "rh_FiniteRadii_CodeUnits.h5"
            wf_file_path = sim_path.joinpath(sub_path)

            with h5py.File(wf_file_path) as wf:

                print(sim_name)
                print(wf.keys())
                rkeys = [item for item in wf.keys() if ".dir" in item]
                ref_time = 2 * min([int(item[1:-4]) for item in rkeys])
                print(ref_time)
                all_req_ref_times.update({sim_name: ref_time})
                self._all_sim_params[sim_name].update(
                    {"RequestedReferenceTime": ref_time}
                )

        self._all_req_ref_times = all_req_ref_times

    def get_segment_at_req_ref_time(self):
        """Get the segments corresponding
        to the requested reference time"""

        all_req_ref_time_segments = {}

        for sim_name, sim_path in self.available_sim_paths.items():
            
            all_req_ref_time_segments.update({sim_name : []})
            message(f"Looking for ref segment in sim {sim_name}", message_verbosity=2)
            highest_sim_ecc = max(self.get_ecc_dirs(sim_path))

            for sim_lev in self.available_sim_levs[sim_name]:
                seg_count = 0
                for seg in self.inspiral_segments[sim_name][f"Lev{sim_lev}"]:

                    message(f"Searching in Lev{sim_lev}_{seg}", message_verbosity=2)
                    path_to_traj = f"{highest_sim_ecc}/Ev/Lev{sim_lev}_{seg}"\
                    "/Run/ApparentHorizons/Trajectory_AhA.dat"

                    try:
                        traj_dat = np.genfromtxt(
                            self.available_sim_paths[sim_name].joinpath(
                                path_to_traj
                            )
                        )
                        avail_seg_times = traj_dat[:, 0]
                    except (FileNotFoundError, IndexError) as fe:
                        message(f"Skipping segment {seg} as Traj file is missing")
                        seg_count += 1
                        continue

                    if (self.all_req_ref_times[sim_name] < max(avail_seg_times)) and (self.all_req_ref_times[sim_name] > min(avail_seg_times)):
                        prev_seg = self.inspiral_segments[sim_name][f"Lev{sim_lev}"][seg_count]
                        #all_req_ref_time_segments.update({sim_name: f"Lev{sim_lev}_{prev_seg}"})
                        all_req_ref_time_segments[sim_name].append(f"Lev{sim_lev}_{prev_seg}")
                        message(f"Segment found for {sim_name} Lev{sim_lev}", message_verbosity=2)
                        break

                    seg_count += 1

        self._all_req_ref_time_segments = all_req_ref_time_segments


    def load_ID_sim_params_from_joined_horizons_file(self, sim_name, sim_lev, req_t_ref):

        hdir = self.prepared_waveforms_dir / f"{sim_name}_waveforms_Lev{sim_lev}/"f"joined"
        
        files = os.listdir(hdir)
        filtered_files = [item for item in files if "Horizons_" in item]
        if len(filtered_files)>1:
            raise KeyError("More than one horizons file found!")
        else:
            file_name = filtered_files[0]

        full_path_to_h = hdir/file_name

        # parameters_to_load = ["ChristodoulouMass.dat", "DimensionfulInertialSpin.dat", "chiInertial.dat", 'CoordCenterInertial.dat']
        parameters = get_horizon_quantities_from_h5(full_path_to_h, req_t_ref)
            
        return parameters

    def load_dynamics_from_joined_horizons_file(self, sim_name, sim_lev, req_t_ref):

        hdir = self.prepared_waveforms_dir / f"{sim_name}_waveforms_Lev{sim_lev}/"f"joined"
        
        files = os.listdir(hdir)
        filtered_files = [item for item in files if "Horizons_" in item]
        if len(filtered_files)>1:
            raise KeyError("More than one horizons file found!")
        else:
            file_name = filtered_files[0]

        full_path_to_h = hdir/file_name

        dynamics = get_dynamics_from_h5(full_path_to_h)
        
        return dynamics
    
    def load_waveform_modes(self, sim_name, sim_lev):
        ''' Load processed waveform modes '''

        path_to_extrap_waveform_h5 = self.get_waveform_out_path(sim_name, sim_lev)
        return scri.SpEC.read_from_h5(str(path_to_extrap_waveform_h5))

    def save_waveform_modes_v0(self, waveform_modes, sim_name, sim_lev, Next):

        wdir = self.prepared_waveforms_dir / f"{sim_name}_waveforms_Lev{sim_lev}/"f"transformed"
        if not os.path.isdir(wdir):
            os.mkdir(wdir)
        wpath = wdir/f"rhOverM_Extrapolated_N{Next}_CoM.h5"
        if not os.path.isfile(wpath):
            scri.SpEC.file_io.write_to_h5(waveform_modes, str(wpath))

    def save_waveform_modes(self, waveform_modes, file_dir, file_path):

        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)

        if not os.path.isfile(file_path):
            scri.SpEC.file_io.write_to_h5(waveform_modes, str(file_path))

    def transform_parameters_to_t_ref(self, 
                                      sim_name, 
                                      sim_lev, 
                                      dynamics, 
                                      NR_ref_params, 
                                     ):
        ''' Transform the parameters to the frame at t_ref '''
        
        for N_ext in [-1, 2, 3, 4, 5, 6]:
            wdir   = self.prepared_waveforms_dir / f"{sim_name}_waveforms_Lev{sim_lev}/transformed"
            wpath  = wdir/f"extrapolated_CoM_transformed_N{N_ext}.h5"
            mdpath = wdir/"reference_metadata.json"

            if not os.path.isfile(mdpath):
                print(f"metadata does not exist for {sim_name} {sim_lev}")
                metadata_exists=False
            else:
                print(f"metadata exists for {sim_name} {sim_lev}")
                metadata_exists=True

            if not os.path.isfile(wpath):
                print(f"TWF does not exist for {sim_name} {sim_lev}")
                waveform_exists = False
            else:
                print(f"TWF exists for {sim_name} {sim_lev}")
                waveform_exists = True

            # if transformed waveform and metadata do not exist
            if not (waveform_exists and metadata_exists):
                print(f"{sim_name}: Creating Waveform and metadata")
                waveform_modes = self.load_waveform_modes(sim_name, sim_lev)
                t_operator = CoordinateTransform(NR_ref_parames=NR_ref_params,
                                                dynamics=dynamics,
                                                waveform_modes=waveform_modes,
                                                )
                
                t_operator.transform()
                self.save_waveform_modes(t_operator.waveform_modes_rot_xyz, 
                                        wdir,
                                        wpath)

                # recompute metadata only if metadata is not saved
                #if not metadata_exists:
                #print(f"{sim_name}: Creating metadata ")
                #import pdb
                #pdb.set_trace()
                reference_parameters = t_operator.reference_parameters

                with open(mdpath, "w") as f:
                    json.dump(reference_parameters, f)
                    metadata=True
            else:
                print(f"{sim_name} Waveform and metadata exists ")
     
            with open(mdpath, "r") as f:
                reference_parameters = json.load(f)

        return reference_parameters

    def parse_sim_params_at_req_ref_times(self):
        """Parse the masses, spins and Orbital freq at requested ref times
        
        1. Read in the masses and spins arrays
        2. Do a quaternionic transform
        3. Evaluate the interpolated values at requested ref time
        4. update the data frames 
        
        """

        from waveformtools.differentiate import derivative
        self.ref_parameters = {}

        for sim_name, all_sim_segs in self.all_req_ref_time_segments.items():
            for sim_seg in all_sim_segs:
                sim_lev = int(sim_seg.split('_')[0][3:])
                sim_name_lev = sim_name + f"_Lev{sim_lev}"
                status = self.sim_status.loc[sim_name_lev].Status
                
                if status=='Completed':
                    print(sim_name_lev)
                    self.ref_parameters.update({ sim_name_lev: self.all_sim_params[sim_name]})
                    req_tref = self.all_req_ref_times[sim_name]
                    #IDparams = self.load_ID_sim_params_from_joined_horizons_file(sim_name, sim_lev, req_tref)
                    dynamics = self.load_dynamics_from_joined_horizons_file(sim_name, sim_lev, req_tref)
                    reference_NR_parameters = get_NR_ref_quantities_at_t_ref(dynamics=dynamics, t_ref=req_tref)
                    reference_parameters = self.transform_parameters_to_t_ref(sim_name, 
                                                                              sim_lev, 
                                                                              dynamics, 
                                                                              reference_NR_parameters, 
                                                                             )

                    message(f"Reference parameters at tref={req_tref} are")
                    message("------------------------")
                    message(reference_parameters)
                    message("------------------------")
                    
                    omega_ref  = float(reference_parameters['omega_ref'])
                    chiA_ref   = reference_parameters['chiA_ref']
                    chiB_ref   = reference_parameters['chiB_ref']
                    chiC_final = tuple(reference_parameters['chiC_final_ref'])
                    v_kick     = tuple(reference_parameters['v_kick_ref'])
                    phi_ref    = reference_parameters['phi_ref']
                    massA      = reference_parameters["massA"]
                    massB      = reference_parameters["massB"]
                    massC      = reference_parameters["massC"]

                    self.ref_parameters[sim_name_lev].update(
                        {"ReqRefTime": self.all_req_ref_times[sim_name]}
                    )
                    self.ref_parameters[sim_name_lev].update({"Omega_ref": omega_ref})
                    self.ref_parameters[sim_name_lev].update(
                        {"ChiA_ref": (chiA_ref[0], chiA_ref[1], chiA_ref[2])}
                    )
                    self.ref_parameters[sim_name_lev].update(
                        {"ChiB_ref": (chiB_ref[0], chiB_ref[1], chiB_ref[2])}
                    )
                    self.ref_parameters[sim_name_lev].update({"MA_ref": massA})
                    self.ref_parameters[sim_name_lev].update({"MB_ref": massB})
                    self.ref_parameters[sim_name_lev].update({"MC_final": massC})
                    self.ref_parameters[sim_name_lev].update({"v_kick": v_kick})
                    self.ref_parameters[sim_name_lev].update({"phi_ref": phi_ref})
                    self.ref_parameters[sim_name_lev].update({"ChiC" : chiC_final})

    def delete_empty_sims(self):
        """Delete sims that have no levels in them"""

        to_delete = []

        for sim_name, avail_levs in self.available_sim_levs.items():

            if avail_levs == []:

                to_delete.append(sim_name)

                try:
                    self._available_sim_names.remove(sim_name)
                    self._available_sim_paths.pop(sim_name)
                    # self._available_sim_levs.pop(sim_name)
                    self._all_sim_params.pop(sim_name)
                    self._all_sim_params_df.pop(sim_name)

                except Exception as ex:
                    message(ex, "Data not loaded")

        for item in to_delete:
            self._available_sim_levs.pop(item)

        self._deleted_sims = to_delete

    def run(self):
        """Run the pipeline"""
        self.search_simulations()
        self.discover_segments()
        self.discover_levels()
        self.delete_empty_sims()
        self.construct_simulation_status()
        self.fetch_sim_params()
        self.discover_ref_time()
        self.get_segment_at_req_ref_time()
        self.parse_sim_params_at_req_ref_times()
        self.update_pandas_dataframe()

    def get_filtered_dataframe(self):
        """Get only a subset of the data"""

        return self.all_sim_params_df.filter(items=self.filter_items)


def re_fetch_float(line):
    """Fetch a float value from .input file"""
    result = re.search("= -?[0-9]*[.]?[0-9]*", line)

    val = float(result.group()[2:])

    return val


def re_fetch_string(line):
    """Fetch a string value from .input file"""
    result = re.search('".*";', line)

    val = str(result.group()[1:-2])

    return val


def re_fetch_vector(line):
    """Fetch a vector value from a .input file"""
    # result = re.search('\(-?[0-9]*[.]?[0-9]*,
    # -?[0-9]*[.]?[0-9]*,-?[0-9]*[.]?[0-9]*\)' ,line)
    result1 = re.search("\(-?[0-9]*[.]?[0-9]*,", line)
    result2 = re.search(",-?[0-9]*[.]?[0-9]*,", line)
    result3 = re.search(",-?[0-9]*[.]?[0-9]*\)", line)

    # val = str(result.group())
    val1 = float(result1.group()[1:-1])
    val2 = float(result2.group()[1:-1])
    val3 = float(result3.group()[1:-1])

    return (val1, val2, val3)


def re_fetch_ext_rad(line):
    """Fetch the ext radius from the key"""


def get_val_at_t_ref(time_axis, val_axis, time):
    """Interpolate and get the value at the requested time"""

    from scipy.interpolate import interp1d

    int_func = interp1d(time_axis, val_axis, kind="cubic")

    val_at_t_ref = float(int_func(time))

    # print(val_at_t_ref, type(val_at_t_ref))

    return round(val_at_t_ref, 5)
