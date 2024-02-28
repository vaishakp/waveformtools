import numpy as np
import os
from pathlib import Path
from waveformtools.waveformtools import (
    message,
    compute_masses_from_mass_ratio_and_total_mass,
)
from waveformtools.sxs.prepare_waveforms import PrepareSXSWaveform

import re


class SimulationExplorer:
    def __init__(
        self,
        search_dir="./",
        prepared_waveforms_dir=None,
    ):
        self._search_dir = Path(search_dir)

        self._prepared_waveforms_dir = prepared_waveforms_dir

    @property
    def search_dir(self):
        return self._search_dir

    @property
    def found_sim_names(self):
        return self._found_sim_names

    @property
    def found_sim_paths(self):
        return self._found_sim_paths

    @property
    def highest_ecc_nums(self):
        return self._highest_ecc_nums

    @property
    def all_sim_params(self):
        return self._all_sim_params

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
                f"{path_of_dir_to_check} does not contain DoMultipleRuns.input",
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

        found_sim_paths = {}
        found_sim_names = []

        one_search_dir = self.search_dir

        message(f"Searching in {one_search_dir}", message_verbosity=1)

        subdirs = os.listdir(one_search_dir)

        message(f"All subdirs {subdirs}", message_verbosity=2)

        for possible_sim_name in subdirs:
            possible_sim_path = one_search_dir.joinpath(possible_sim_name)

            message(f"\tSearching in {possible_sim_path}", message_verbosity=2)

            flag = self.check_for_ecc_dir(possible_sim_path)

            if flag:
                found_sim_paths.update({possible_sim_name: possible_sim_path})

                found_sim_names.append(possible_sim_name)

        self._found_sim_names = found_sim_names
        self._found_sim_paths = found_sim_paths

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

    def discover_levels(self):
        """Discover all avaiable levels across all simulations"""

        all_levs = {}

        for sim_name, sim_path in self.found_sim_paths.items():
            sim_levs = []
            path_to_highest_ecc_dir = sim_path.joinpath(
                f"Ecc{self.highest_ecc_nums[sim_name]}/Ev"
            )

            subdirs = os.listdir(path_to_highest_ecc_dir)

            subdirs = [item for item in subdirs if "Ringdown" in item]

            if subdirs is None:
                continue

            levs = [item for item in subdirs if "Lev" in item]

            for item in levs:
                result = re.search("Lev[0-9]_Ringdown", item)

                lev_num = int(result.group()[3:4])

                sim_levs.append(lev_num)

            sim_levs = list(set(sim_levs))

            all_levs.update({sim_name: sim_levs})

        self._available_sim_levs = all_levs

    def get_all_segments(self, ecc_dir):
        """Get the segments within one Ecc dir"""
        import os

        all_segments = {}

        available_dirs = os.listdir(ecc_dir)

        available_lev_dirs = [item for item in available_dirs if "Lev" in item]

        available_levs = set([item[3:-3] for item in available_lev_dirs])

        print("Available dirs", available_dirs)
        print("Available lev dirs", available_lev_dirs)
        print("Available levs", available_levs)

        for lev in available_levs:
            this_lev_dirs = [
                item for item in available_lev_dirs if f"Lev{lev}" in item
            ]

            this_lev_segments = sorted([item[5:] for item in this_lev_dirs])

            all_segments.update({f"Lev{lev}": this_lev_segments})

        return all_segments

    def concatenate_ascii_data(self, fname, ecc, lev, sim_dir):
        """Concatenate the data in the given file name, lev,
        ecc dir"""

        full_ecc_path = f"{sim_dir}/Ecc{ecc}/Ev/"

        # get segments
        all_super_segments = get_all_segments(full_ecc_path)

        all_segments = all_super_segments[f"Lev{lev}"]

        all_data = []

        for item in all_segments:
            file = f"{full_ecc_path}/Lev{lev}_{item}/Run/{fname}"

            one_segment_data = np.genfromtxt(file)
            all_data.append(one_segment_data)

        all_data = np.concatenate((all_data))

        return all_data

    def get_ecc_dirs(self, sim_path):
        subdirs = os.listdir(sim_path)

        ecc_dirs = [item for item in subdirs if "Ecc" in item]

        return ecc_dirs

    def find_highest_ecc_dirs(self):
        """Find and load the highest ecc level available"""

        highest_ecc_nums = {}

        for sim_name, sim_path in self.found_sim_paths.items():
            highest_ecc_num = self.find_highest_ecc(sim_path)

            highest_ecc_nums.update({sim_name: highest_ecc_num})

        self._highest_ecc_nums = highest_ecc_nums

    def parse_sim_params_input_file(self, path_to_params_input_file):
        """Parse all sim params given path to the Params.input file"""

        sim_params = {}

        message(
            f"Parsing simulation parameters.input file for {path_to_params_input_file}",
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

    def fetch_sim_params(self):
        """Get all the simulations' parameters"""

        all_sim_params = {}

        for sim_name, sim_path in self.found_sim_paths.items():
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

    def strip_sim_name_from_waveform_dir(self, wdir):
        result = re.search("[A-Za-z]+[0-9]+_", wdir)

        if result is not None:
            return result.group()[:-1]

        else:
            return None

    def discover_extrapolated_sims(self):
        """Read in all available waveform directories in the given directory
        and create a list of sims whose waveforms have already been prepared"""

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

        for sim_name in self.found_sim_names:
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
                message("Preparing waveform for sim ", sim_name, "Lev ", lev)

                try:
                    wfp = PrepareSXSWaveform(
                        sim_name=sim_name,
                        sim_dir=self.found_sim_paths[sim_name],
                        lev=lev,
                        out_dir=self.prepared_waveforms_dir,
                    )

                    flag = wfp.prepare_waveform()

                    if flag:
                        processed_levs.append(lev)

                except Exception as excep:
                    message(excep)
                    message(
                        f"Waveform processing failed sim {sim_name}, lev {lev}"
                    )

            processed_waveforms.update({sim_name: processed_levs})

        self._processed_waveforms = processed_waveforms

    def compute_masses(self):
        for sim_name in self.found_sim_names:
            mass1, mass2 = compute_masses_from_mass_ratio_and_total_mass(
                self.all_sim_params[sim_name]["MassRatio"]
            )

            self.all_sim_params[sim_name].update(
                {"IndividualMasses": (mass1, mass2)}
            )

    def compute_chi_eff(self):
        for sim_name in self.found_sim_names:
            mass_ratio = 1 / self.all_sim_params[sim_name]["MassRatio"]

            _, _, spin1z = self.all_sim_params[sim_name]["ChiA"]
            _, _, spin2z = self.all_sim_params[sim_name]["ChiB"]

            chi_eff = (spin1z * mass_ratio + spin2z) / (1 + mass_ratio)

            self.all_sim_params[sim_name].update({"ChiEff": chi_eff})

    def compute_chi_prec(self):
        for sim_name in self.found_sim_names:
            mass_ratio = self.all_sim_params[sim_name]["MassRatio"]

            mass1, mass2 = compute_masses_from_mass_ratio_and_total_mass(
                self.all_sim_params[sim_name]["MassRatio"]
            )

            s1x, s1y, _ = self.all_sim_params[sim_name]["ChiA"]
            s2x, s2y, _ = self.all_sim_params[sim_name]["ChiB"]

            s1p = mass1**2 * np.sqrt(s1x**2 + s1y**2)
            s2p = mass2**2 * np.sqrt(s2x**2 + s2y**2)

            A1 = 2 + 3 / (2 * mass_ratio)
            A2 = 2 + 3 * mass_ratio / (2)

            chi_prec = max(A1 * s1p, A2 * s2p) / (A1 * mass1**2)

            self.all_sim_params[sim_name].update({"ChiPrec": chi_prec})

    def get_all_chi_eff(self):
        all_chi_eff = []

        for sim_name in self.found_sim_names:
            all_chi_eff.append(self.all_sim_params[sim_name]["ChiEff"])

        return all_chi_eff

    def get_all_mass_ratios(self):
        all_mass_ratios = []

        for sim_name in self.found_sim_names:
            all_mass_ratios.append(self.all_sim_params[sim_name]["MassRatio"])

        return all_mass_ratios

    def write_history(self):
        with open(self.history_file, "a") as th:
            for item in dirs:
                th.write(item)
                th.write("\n")


def re_fetch_float(line):
    result = re.search("= -?[0-9]*[.]?[0-9]*", line)

    val = float(result.group()[2:])

    return val


def re_fetch_string(line):
    result = re.search('".*";', line)

    val = str(result.group()[1:-2])

    return val


def re_fetch_vector(line):
    # result = re.search('\(-?[0-9]*[.]?[0-9]*,-?[0-9]*[.]?[0-9]*,-?[0-9]*[.]?[0-9]*\)' ,line)

    result1 = re.search("\(-?[0-9]*[.]?[0-9]*,", line)
    result2 = re.search(",-?[0-9]*[.]?[0-9]*,", line)
    result3 = re.search(",-?[0-9]*[.]?[0-9]*\)", line)

    # val = str(result.group())

    val1 = float(result1.group()[1:-1])
    val2 = float(result2.group()[1:-1])
    val3 = float(result3.group()[1:-1])

    return (val1, val2, val3)
