""" 
MIT License

Copyright (c) 2023 Vaishak Prasad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Script to 

    a) join h5 files
    b) extrapolate to infinity
    c) correct for CoM drift
    
    from SXS simulations
    
"""

import os
from pathlib import Path

import numpy as np
import scri
from waveformtools.waveformtools import message


class PrepareSXSWaveform:
    """Prepare waveforms from a particular Lev and ECC of
    an SXS NR run by

    1) Joining all segments
    2) Extrapolating to infinity
    3) Transforming to CoM frame
    4) Upload to cloud


    Attributes
    ----------
    sim_dir: str/POSIXPath
              The root directory containing
              the simulation directory
    sim_name: str
              The alias of the simulation whose
              waveform is to be processed
    out_dir: str/Path
             The directory in which to save extrapolated
             waveforms
    joined_outfile_dir: str/Path
                        The directory in which to save joined
                        waveforms from across segments
    joined_waveform_outfile_name: str/Path
                                  The filename of the processed
                                  waveform.
    joined_waveform_outfile_path: The full path to the processed
                                  waveform file.
    joined_horizons_outfile_name: str/Path
                                  The filename of the joined horizon file
    lev: int
         The sim resolution level to process
    ecc: int
         The Ecc dir number to process
    extrap_out_dir: str/Path
                    The full path to the output dir.

    Methods
    -------
    join_waveform_h5_files
        Join the waveform h5 files from across segments
    extrapolate
        Extrapolate a joined waveform to infinity
    join_horizons
        Join the Horizons.h5 files from across segments
    transform_to_CoM_frame
        Apply CoM correction
    upload_output_dir
        Upload the output directory to a cloud
    prepare_waveform
        Process the waveform by extrapolating and
        applying CoM correction
    """

    def __init__(
        self,
        sim_name,
        history_file=Path("./history_file.txt"),
        sim_dir=Path("./"),
        out_dir=None,
        joined_waveform_outfile_name=None,
        joined_horizons_outfile_name=None,
        lev=2,
        ecc=0,
    ):
        if not os.path.isabs(sim_dir):
            raise ValueError("Please provide the full sim path!")

        self._sim_dir = Path(sim_dir)

        self._sim_name = sim_name
        self._lev = lev
        self._ecc = ecc
        self._history_file = history_file

        if joined_waveform_outfile_name is None:
            message("Choosing default directory for output...", message_verbosity=2)
            joined_waveform_outfile_name = ('rhOverM_'+
                sim_name + f"Lev{self.lev}JoinedWaveform.h5"
            )

        self._joined_waveform_outfile_name = joined_waveform_outfile_name

        if out_dir is None:

            cwd = Path(os.getcwd())
            message(f"Current working directory {cwd}", message_verbosity=2)
            self._out_dir = (
                cwd / f"processed_waveforms/{sim_name}_waveforms_Lev{self.lev}"
            )
            message(f"Setting out-directory to ({self.out_dir})...", message_verbosity=2)
        else:
            self._out_dir = out_dir

        message(f"Out directory is set to {self.out_dir}", message_verbosity=2)

        if not os.path.isdir(self.out_dir):
            message(f"Creating directory {self.out_dir}", message_verbosity=2)
            os.makedirs(self.out_dir)

        joined_outfile_dir = os.path.join(self.out_dir, Path(f"joined"))

        self._joined_outfile_dir = joined_outfile_dir

        if not os.path.isdir(joined_outfile_dir):
            os.mkdir(joined_outfile_dir)

        self._joined_outfile_path = os.path.join(
            joined_outfile_dir, self.joined_waveform_outfile_name
        )

        if joined_horizons_outfile_name is None:
            joined_horizons_outfile_name = ('Horizons_'+
                sim_name + f"Lev{self.lev}JoinedHorizons.h5"
            )
        
        self._joined_horizons_outfile_name = joined_horizons_outfile_name

        self.setup_env()

    @property
    def sim_dir(self):
        """The full path to the directory containing the
        simulation"""
        return self._sim_dir

    @property
    def sim_name(self):
        return self._sim_name

    @property
    def out_dir(self):
        return self._out_dir

    @property
    def joined_outfile_dir(self):
        """The directory containing the
        joined files"""
        return self._joined_outfile_dir

    @property
    def joined_waveform_outfile_name(self):
        return self._joined_waveform_outfile_name

    @property
    def joined_waveform_outfile_path(self):
        return os.path.join(
            self.joined_outfile_dir,
            Path(f"{self.joined_waveform_outfile_name}"),
        )

    @property
    def joined_horizons_outfile_name(self):
        return self._joined_horizons_outfile_name

    @property
    def joined_horizons_outfile_path(self):
        return os.path.join(
            self.joined_outfile_dir,
            Path(f"{self.joined_horizons_outfile_name}"),
        )

    @property
    def lev(self):
        return self._lev

    @property
    def ecc(self):
        return self._ecc

    @property
    def extrap_out_dir(self):
        return os.path.join(self.out_dir, Path("extrapolated"))

    @property
    def history_file(self):
        return self._history_file

    def setup_env(self):
        """Setup the environment"""

        # exec(open('/mnt/pfs/vaishak.p/soft/modules-5.2.0/init/python.py').read())

        # module('unload', 'gcc/11.1.0')

        # module('load', 'gcc/11.1.0')

    def join_waveform_h5_files(self, verbose=False):
        """Join the waveform h5 files"""

        if Path(self.joined_horizons_outfile_path).exists():
            message("File already exists. Skipping operation.", message_verbosity=2)

        else:
            message("Joining waveform h5 files...", message_verbosity=2)

            data_paths_insp = os.path.join(
                self.sim_dir,
                Path(
                    f"Ecc{self.ecc}"
                    f"/Ev/Lev{self.lev}*/Run/GW2/"
                    "rh_FiniteRadii_CodeUnits.h5"
                ),
            )

            data_paths_rdown = os.path.join(
                self.sim_dir,
                Path(
                    f"Ecc{self.ecc}"
                    f"/Ev/Lev{self.lev}_Ringdown/"
                    f"Lev{self.lev}*/Run/GW2/"
                    "rh_FiniteRadii_CodeUnits.h5"
                ),
            )

            if verbose:
                run_cmd = "JoinH5 -v"

            else:
                run_cmd = "JoinH5"

            try:
                run_cmd += (
                    f" -o {self.joined_waveform_outfile_path}"
                    f" {data_paths_insp} "
                    f" {data_paths_rdown}"
                )

                message(f"Running command\n {run_cmd}", message_verbosity=2)

                # with open('join_waveforms_output.txt', "wb") as fout:
                # subprocess.check_call('dir',stdout=f)
                # cmd_out = subprocess.Popen(run_cmd, stdout=subprocess.PIPE)

                # for cline in iter(lambda: process.stdout.read(1), b""):
                # sys.stdout.buffer.write(cline)
                # f.buffer.write(cline)

                cmd = os.popen(run_cmd)

                out = cmd.read()

                message("Command output \n", out, message_verbosity=2)
            except Exception as ex:
                run_cmd += (
                    f" -o {self.joined_waveform_outfile_path}"
                    f" {data_paths_insp}"
                )

                message(f"Running command\n {run_cmd}", message_verbosity=2)

                # with open('join_waveforms_output.txt', "wb") as fout:
                # subprocess.check_call('dir',stdout=f)
                # cmd_out = subprocess.Popen(run_cmd, stdout=subprocess.PIPE)

                # for cline in iter(lambda: process.stdout.read(1), b""):
                # sys.stdout.buffer.write(cline)
                # f.buffer.write(cline)

                cmd = os.popen(run_cmd)

                out = cmd.read()

                message("Command output \n", out, message_verbosity=2)

            message("Command completed. Please check Errors.txt for details", message_verbosity=2)

    def extrapolate(self, ChMass=1.0, UseStupidNRARFormat=True):
        """Extrapolate the waveform"""

        try:
            message(
                "\tChecking if extrapolated files from a previous run exists..."
            )
            files = os.listdir(self.extrap_out_dir)

            exists = np.array(
                [item for item in files if "Extrapolated" in item]
            )
        # message(exists)
        except Exception as excep:
            message("\t", excep, message_verbosity=2)
            message("\tNo extrapolated files from previous run found", message_verbosity=2)
            exists = []

        if len(exists) > 0:
            message("\tSkipping extrapolation", message_verbosity=2)

        else:
            message("Extrapolating...", message_verbosity=2)

            wf = scri.extrapolate(
                InputDirectory=self.joined_outfile_dir,
                OutputDirectory=self.extrap_out_dir,
                DataFile=self.joined_waveform_outfile_name,
                ChMass=ChMass,
                UseStupidNRARFormat=UseStupidNRARFormat,
                DifferenceFiles="",
                PlotFormat="",
            )

    def join_horizons(self, verbose=False):
        """Join horizons file and save to the joined
        file dir"""

        if Path(self.joined_horizons_outfile_path).exists():
            message("File already exists. Skipping join horizons operation.", message_verbosity=2)

        else:
            message("Joining Horizon h5 files...", message_verbosity=2)

            input_insp_dat_rel_loc = (
                f"Ecc{self.ecc}"
                f"/Ev/Lev{self.lev}*/Run/"
                "ApparentHorizons/Horizons.h5"
            )

            input_rdown_dat_rel_loc = (
                f"Ecc{self.ecc}"
                f"/Ev/Lev{self.lev}_Ringdown/"
                f"Lev{self.lev}*/Run/"
                "ApparentHorizons/Horizons.h5"
            )

            # data_paths_insp = os.path.join(self.sim_dir, input_insp_dat_rel_loc)

            data_paths_insp = self.sim_dir.joinpath(input_insp_dat_rel_loc)

            message("data paths insp", data_paths_insp, message_verbosity=2)

            # data_paths_rdown = os.path.join(
            #    self.sim_dir, input_rdown_dat_rel_loc
            # )

            data_paths_rdown = self.sim_dir.joinpath(input_rdown_dat_rel_loc)

            message("data paths rdown", data_paths_rdown, message_verbosity=2)

            if verbose:
                run_cmd = "JoinH5 -v"

            else:
                run_cmd = "JoinH5"

            try:
                run_cmd += (
                    f" -o {self.joined_horizons_outfile_path}"
                    f" {data_paths_insp} {data_paths_rdown}"
                )

                message(f"Running command\n {run_cmd}", message_verbosity=2)

                cmd = os.popen(run_cmd)

                out = cmd.read()

                message("Command output \n", out, message_verbosity=2)

            except Exception as ex:
                run_cmd += (
                    f" -o {self.joined_horizons_outfile_path}"
                    f" {data_paths_insp}"
                )

                message(f"Running command\n {run_cmd}", message_verbosity=2)

                cmd = os.popen(run_cmd)

                out = cmd.read()

                message("Command output \n", out, message_verbosity=2)

            message("Command completed. Please check Errors.txt for details.", message_verbosity=2)

    def transform_to_CoM_frame(
        self,
        skip_beginning_fraction=0.01,
        skip_ending_fraction=0.10,
        file_format="NRAR",
        extrap_enn_list=[-1, 2, 3, 4, 5, 6],
    ):
        """Apply CoM correction to a waveform"""
        from scri.SpEC.com_motion import remove_avg_com_motion

        try:
            files = os.listdir(self.extrap_out_dir)

            exists = np.array([item for item in files if "CoM" in item])

        except Exception as excep:
            message(excep)
            message("Continuing with transformation", message_verbosity=2)
            exists = []

        if len(exists) > 0:
            message("Skipping CoM transformation", message_verbosity=2)
        else:
            message("Transforming to CoM frame...", message_verbosity=2)

            for extrap_enn in extrap_enn_list:
                message(f"Working on Extrapolated N_{extrap_enn}", message_verbosity=2)

                path_to_waveform_h5 = os.path.join(
                    self.extrap_out_dir,
                    f"rhOverM_Extrapolated_N{extrap_enn}.h5",
                )

                path_to_horizons_h5 = self.joined_horizons_outfile_path

                remove_avg_com_motion(
                    w_m=None,
                    path_to_waveform_h5=path_to_waveform_h5,
                    path_to_horizons_h5=path_to_horizons_h5,
                    skip_beginning_fraction=skip_beginning_fraction,
                    skip_ending_fraction=skip_ending_fraction,
                    file_write_mode="w",
                    m_A=None,
                    m_B=None,
                    file_format=file_format,
                    write_corrected_file=True,
                )

    def upload_output_dir(self):
        """Upload the outut directory to a cloud"""
        raise NotImplementedError

    def prepare_waveform(
        self,
        verbose=False,
        ChMass=1.0,
        UseStupidNRARFormat=True,
        skip_beginning_fraction=0.01,
        skip_ending_fraction=0.10,
        file_format="NRAR",
        extrap_enn_list=[-1, 2, 3, 4, 5, 6],
        upload=False,
    ):
        """Carry out extrapolation + CoM correction"""
        self.join_waveform_h5_files(verbose=verbose)

        self.extrapolate(ChMass=ChMass, UseStupidNRARFormat=UseStupidNRARFormat)

        self.join_horizons(verbose=verbose)

        self.transform_to_CoM_frame(
            skip_beginning_fraction=skip_beginning_fraction,
            skip_ending_fraction=skip_ending_fraction,
            file_format=file_format,
            extrap_enn_list=extrap_enn_list,
        )

        if upload:
            self.upload_output_dir()

            pass

        message("\n--------------------------------------------------------\n", message_verbosity=2)

        return True

    def write_history(self):
        """Write processing history to a file"""
        with open(self.history_file, "a") as th:
            for item in dirs:
                th.write(item)
                th.write("\n")

        # th.write('EOF')
