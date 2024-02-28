import numpy as np
import os
from pathlib import Path
from waveformtools.waveformtools import message
import re


class SimulationExplorer:
    def __init__(
        self,
        search_dir=[],
    ):
        self._search_dir = Path(search_dir)

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


def re_fetch_float(line):
    result = re.search("= -?[0-9]*[.]?[0-9]*", line)

    val = float(result.group()[2:])

    return val


def re_fetch_string(line):
    result = re.search('".*";', line)

    val = str(result.group()[1:-2])

    return val


def re_fetch_vector(line):
    result = re.search(
        "\(-?[0-9]*[.]?[0-9]*,-?[0-9]*[.]?[0-9]*,-?[0-9]*[.]?[0-9]*\)", line
    )

    val = str(result.group())

    return val
