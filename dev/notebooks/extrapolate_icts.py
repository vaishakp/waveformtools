import numpy as np
import os
from pathlib import Path
import config
import subprocess

import matplotlib.pyplot as plt

config.conf_matplolib()


from waveformtools.sxs.discover_sims import SimulationExplorer

list_of_search_dirs = ["/mnt/pfs/vaishak.p/sims/SpEC/gcc/bfi/ICTSEccParallel"]

se = SimulationExplorer(
    search_dir=list_of_search_dirs[0],
    prepared_waveforms_dir=Path("/mnt/pfs/vaishak.p/Projects/Codes/waveforms"),
)

se.search_simulations()
se.find_highest_ecc_dirs()
se.discover_levels()

# se.prepare_waveforms()


for sim_name, sim_path in se.found_sim_paths.items():

    inside_ev = sim_path.joinpath(f"Ecc{se.highest_ecc_nums[sim_name]}/Ev")

    out_file = sim_path.joinpath("WaveformsJoinedLev3.h5")

    cmd = f"JoinH5 -v -o {out_file} -l {inside_ev}/Lev3*/Run/GW2/rh_FiniteRadii_CodeUnits.h5"

    print(cmd)

    (cmd)
