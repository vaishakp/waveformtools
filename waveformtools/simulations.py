"""Data container for Numerical Relativity data."""

import vlconf

vlconf.conf_matplolib()
import numpy as np
import scipy

from waveformtools.waveformtools import cleandata, message

# import waveformtools


class sim:
    """A data container for simulation data.

    Arrtibutes
    ----------
    ROOTDIR: string
             Root directory as a string containing the simulation folders.
    WAVDIR: string
            Root directory as a string containing the simulation directies
            containing the wavefom data.
    data_dir: string
              The path of the folder containing data relative to
              the simulation direcory.
    strain_dir: string
                The path of the folder containing the waveform data
                relative to the strain directory.
    aliases: a list of strings
             The names/aliases for the simulations.
    multipoles: dict of lists
                The multipole moments of the simulation as a dictionary.
                Each entry is a list of width 4 with axis 0 the timeaxis
                of multipoles.
    mass1: dict of floats
           The BH1 horizon mass.
    mass2: dict of floats
           The BH2 horizon mass.
    mass3: dict of floats
           The BH3 horizon mass.
    delta_t: dict of floats
             The time stepping in simulation units (delta_t/M).
    timeaxis: dict of 1d arrays
              The timeaxis of the simulations.
    distance: dict of 1d arrays
              The distances of simulations.
    merger_ind: dict of ints
                The merger index/ common horizon formation index
                of simulations.
    dinit: dict of floats
           The initial distances.
    multipoles: dict of lists
                The two sets of  mass multipoles of the three horizons.
                Axis 0 is usuall the time array.
    mass_multipoles: dict of lists
                     The mass multipoles upto (ell=8).
    spin_multipoles: dict of lists
                     The spin multipoles upto (l=8).
    data_length: dict of float
                 The data length of the multipole simulation data loaded.
    dist_data_length: dict of ints
                      The data length of distances of simulations.
    merger_distance: dict of floats
                     The distance between the blackholes at the merger index.
    true_merger_dist: dict of floats
                      The true merger distance (non-normalized).
    sampling_f: dict of floats
                The sampling frequency of simulations (1/delta_t).
    merger_time: dict of floats
                 The cctk_time stamp at merger.
    massratio: dict of floats
               The massratio of the simulations.
    chirpmass: dict of floats
               The chirpmass of the simulations.
    totalmass: dict of floats
               The total mass of the simulations.
    log_multipoles: dict of lists
                    The natural logarithm of the negative
                    of the multipole moments as
                    a list [time, multipole1, multipole2,
                    multipole3(if exists)].
    data_duration: dict of floats
                   The total cctk_time units of simulations present.
    BH_locations: dict of lists.
                  A dictionary of values containing BH
                  locations of every simulation. Each
                  simulation has three lists, one for
                  each black hole.
    CoM_locations: dict of lists
                   A dictionary of lists containing the
                   X, Y and Z locations of the CoM of the simulations.
    NP_1d: dict
           A dict of dicts of lists containig
           the 1d Newman Penrose data from simulations.

    Methods
    -------
    calc_ref_multipoles
            Calculate the reference multipoles
            as time average of first few timesclices
            of (2) multipole moment data.

            Assignes/Updates:
                    * ref_multipoles.

    calc_log_multipoles
            Calculate the natural logarithm
            of the negative of the two sets
            of multipole moment.
            Assignes/ Updates:
                    * log_multipoles.

    calc_delta_multipoles
            Calculate the two delta multipoles - ref multipoles.
            Assignes/ Updates:
                    * delta_multipoles.

    calc_amp_phase
            Extract the amplitudes and phases of the strain waveforms.
            Assignes/Updates:
                    * strain_amp
                    * strain_phase.

    load_data
            Load the multipole and distance data of the simulations.
            Assignes/Updates:
                    * distance.
                    * dinit.
                    * mass1.
                    * mass2.
                    * merger_index.
                    * merger_time.
                    * merger_distance.
                    * true_merger_distance.
                    * data_length.
                    * data_duration.
                    * multipoles.
                    * mass_multipoles.
                    * spin_multipoles.

    load_strain
            Load the strain data of the simulations from waveform directories.
            Assignes/Updates:
                    * strain.
                    * strain_amp.
                    * strain_phase.
                    * strain_shift.

    load_shears
            Load the shear data at a pole of
            respective horizons of the simulations
            from waveform directories.
            Assignes/Updates:
                    * shear.
                    * shear_amp.
                    * shear_phase.
                    * shear_shift.

    ret_horizon_radii
            A method to retrieve the areal radii of the horizons.
            Assigns/Updates
                    * areal_radii

    _resize_multipoles
            Private method to resize the
            sim.multipoles after retrieving
            the length of distance.

    _ifreversal
            Private method to reverse the data
            of BH 1 and 2 if mass2 > mass1.
    """

    def __init__(
        self,
        # Variables for initialization.
        aliases=None,
        multipoles=None,
        mass_multipoles=None,
        spin_multipoles=None,
        mass1=None,
        mass2=None,
        mass3=None,
        delta_t=None,
        merger_ind=None,
        actmerger_time=None,
        timeaxis=None,
        dinit=None,
        distance=None,
        ROOTDIR=None,
        WAVDIR=None,
        simdir=None,
        data_dir="data/primary/",
        data_length=None,
        dist_data_length=None,
        comm_data_length=None,
        strain_dir="output/",
        strain=None,
        strain_phase=None,
        strain_frequency=None,
        strain_amplitude=None,
        strain_indexshifts=None,
        indexjn=None,
        distjn=None,
        areal_radii={},
        log_deltamultipoles2=None,
        log_multipoles2=None,
        log_deltamultipoles=None,
        log_multipoles=None,
        ref_multipoles=None,
        indjn=None,
        NP_1d=None,
    ):
        # Load the variables at initialization.
        self.aliases = aliases or {}
        self.multipoles = multipoles or {}
        self.spin_multipoles = spin_multipoles or {}
        self.mass_multipoles = mass_multipoles or {}
        self.mass1 = mass1 or {}
        self.mass2 = mass2 or {}
        self.mass3 = mass3 or {}
        self.delta_t = delta_t or {}
        self.timeaxis = timeaxis or {}
        self.distance = distance or {}
        self.merger_ind = merger_ind or {}
        self.actmerger_time = actmerger_time or {}
        self.dinit = dinit or {}
        self.data_length = data_length or {}
        self.dist_data_length = dist_data_length or {}
        self.comm_data_length = comm_data_length or {}
        self.strain = strain or {}
        self.strain_phase = strain_phase or {}
        self.strain_frequency = strain_frequency or {}
        self.strain_amplitude = strain_amplitude or {}
        self.strain_indexshifts = strain_indexshifts or {}
        self.indexjn = indexjn or {}
        self.distjn = distjn or {}
        self.ROOTDIR = ROOTDIR
        self.WAVDIR = WAVDIR
        self.simdir = simdir
        self.strain_dir = strain_dir
        self.data_dir = data_dir
        self.areal_radii = areal_radii
        self.log_deltamultipoles2 = log_deltamultipoles2
        self.log_multipoles2 = log_multipoles2
        self.log_deltamultipoles = log_deltamultipoles
        self.log_multipoles = log_multipoles
        self.ref_multipoles = ref_multipoles
        self.indjn = indjn
        self.NP_1d = NP_1d

    @property
    def data_duration(self):
        """Compute and return the data duration of the simulations."""
        return {
            alias: self.data_length[alias] * self.delta_t[alias]
            for alias in self.aliases
        }

    @property
    def comm_data_duration(self):
        """Compute and return the duration of.

        the common data (multipole and Bhdiag/distance)
        of the simulations.
        """
        return {
            alias: self.comm_data_length[alias] * self.delta_t[alias]
            for alias in self.aliases
        }

    @property
    def pm_data_duration(self):
        """Compute and return the duration.

        of post-merger data present in the simulations.
        """
        return {
            alias: self.data_duration[alias] - self.merger_time[alias]
            for alias in self.aliases
        }

    @property
    def pm_data_length(self):
        """Compute and return the post merger.

        data length avaialable for all simulations.
        """
        return {
            alias: self.data_length[alias] - self.merger_ind[alias]
            for alias in self.aliases
        }

    @property
    def merger_distance(self):
        """Compute and return the distance.

        at the merger index of the simulations.
        """
        merger_dist = {}
        for alias in self.aliases:
            try:
                merger_dist.update(
                    {alias: self.distance[alias][self.merger_ind[alias]]}
                )
            except IndexError:
                if (self.merger_ind[alias] - len(self.distance[alias])) == 1:
                    merger_dist.update(
                        {
                            alias: self.distance[alias][
                                self.merger_ind[alias] - 1
                            ]
                        }
                    )
                else:
                    message(
                        "%s :Distance upto merger not defined."
                        "Setting final value" % alias
                    )
                    merger_dist.update({alias: self.distance[alias][-1]})
        return merger_dist

    @property
    def true_merger_distance(self):
        """Compute and return the true.

        (i.e. non-normalized) distance at merger for the simulations.
        """
        return {
            alias: self.dinit[alias] * self.merger_distance[alias]
            for alias in self.aliases
        }

    @property
    def sampling_f(self):
        """Compute and return the sampling frequency of the simulations."""
        return {alias: 1.0 / self.delta_t[alias] for alias in self.aliases}

    @property
    def merger_time(self):
        """Compute and return the.

        cctk_time stamp at the merger
        for the simulations.
        """
        return {
            alias: self.merger_ind[alias] * self.delta_t[alias]
            for alias in self.aliases
        }

    @property
    def massratio(self):
        """Compute and return the massratio of the simulations."""
        return {
            alias: self.mass2[alias] / self.mass1[alias]
            for alias in self.aliases
        }

    @property
    def chirpmass(self):
        """Compute and return the chirp mass of the simulations."""
        return {
            alias: ((self.mass1[alias] * self.mass2[alias]) ** (3.0 / 5))
            / (self.mass1[alias] + self.mass2[alias]) ** (1.0 / 5)
            for alias in self.aliases
        }

    @property
    def totalmass(self):
        """Compute and return the total mass of the simulations."""
        return {
            alias: self.mass1[alias] + self.mass2[alias]
            for alias in self.aliases
        }

    def calc_junkend(self, tjn=200.0):
        """Compute the indices and the starting distances.

        of the system at timestamp t = 200.

        Parameters
        ----------
        tjn: float
             The definition of time end of junk radiation. Default is 200.

        Notes
        -----
        Computes:

        self.indjn: dict
                    A dictionary containing the index location
                    corresponding to timestamp tjn.
        self.distjn: dict
                     A dictionary containing the normalized co-ordinate
                     distance between the two BHs at tjn.
        """

        # Find the starting distance at t = 200M.
        # Initialize the directory.
        self.indjn = {}
        for alias in self.aliases:
            # Iterate over simulations.
            # Find the index corresponding to the timestamp tjn.
            indjn = np.argmin(np.absolute(self.timeaxis[alias] - tjn))
            # Update the sjn dictionary.
            self.indjn.update({alias: indjn})
            # message(len(sim_A.distance[alias]), ind)
            # Update th djn dictionary.
            self.distjn.update({alias: self.distance[alias][1][indjn]})
        # Return 1
        return 1

    def calc_ref_multipoles(self):
        """Compute and assign the reference (l=2) multuipoles
        to sim.ref_multipoles of the simulations."""
        refmult = {}
        # index = 0
        for alias in self.aliases:
            message(alias)
            item = np.transpose(self.multipoles[alias])
            ml_length = len(item[:, 0])
            if ml_length < self.comm_data_length[alias]:
                self.comm_data_length[alias] = ml_length
                self.distance[alias] = self.distance[alias][:ml_length]
            message(item.shape)
            """Unpack data as 1d arrays."""
            _, Ml12, Ml22, Ml32 = (
                item[: self.comm_data_length[alias], 0],
                item[: self.comm_data_length[alias], 1],
                item[: self.comm_data_length[alias], 2],
                item[self.merger_ind[alias] :, 3],
            )
            """Compute delta multipoles."""
            Ml12_ref = np.mean(Ml12[30:100])
            Ml22_ref = np.mean(Ml22[30:100])
            Ml32_ref = np.mean(Ml32[-100:])

            refmult.update({alias: [Ml12_ref, Ml22_ref, Ml32_ref]})
            self.ref_multipoles = refmult

    def calc_log_multipoles(self):
        """Compute and assign the natural logarithm of the (l=2)
        multipoles to sim.log_multipoles of the simulations."""
        log_mult = {}
        for alias in self.aliases:
            message(alias)
            item = np.transpose(self.multipoles[alias])
            ml_length = len(item[:, 0])
            if ml_length < self.comm_data_length[alias]:
                self.comm_data_length[alias] = ml_length
                self.distance[alias] = self.distance[alias][:ml_length]
            message(item.shape)
            log_mult.update(
                {
                    alias: [
                        item[: self.comm_data_length[alias], 0],
                        np.log(
                            np.absolute(
                                -item[: self.comm_data_length[alias], 1]
                            )
                        ),
                        np.log(
                            np.absolute(
                                -item[: self.comm_data_length[alias], 2]
                            )
                        ),
                        np.log(np.absolute(-item[self.merger_ind[alias] :, 3])),
                    ]
                }
            )
        self.log_multipoles = log_mult

    def calc_delta_multipoles(self):
        """Compute and return the delta multipoles (w.r.t.

        reference (l=2) multipoles)
        """
        log_deltmult = {}
        for alias in self.aliases:
            message(alias)
            # Multipole data
            item = np.transpose(self.multipoles[alias])
            """Length of multipoles."""
            ml_length = len(item[:, 0])
            """Check the lengths of multipole and distance data."""
            if ml_length < self.comm_data_length[alias]:
                """Reset the datalengths."""
                self.comm_data_length[alias] = ml_length
                """Resize the distance array."""
                self.distance[alias] = self.distance[alias][:ml_length]
            message(item.shape)
            """Unpack data as 1d arrays."""
            time, Ml12, Ml22, Ml32 = (
                item[: self.comm_data_length[alias], 0],
                item[: self.comm_data_length[alias], 1],
                item[: self.comm_data_length[alias], 2],
                item[self.merger_ind[alias] :, 3],
            )
            """Compute delta multipoles."""
            Ml12_ref = np.mean(Ml12[30:100])
            Ml22_ref = np.mean(Ml22[30:100])
            Ml32_ref = np.mean(Ml32[-100:])
            dMl12 = Ml12 - Ml12_ref
            dMl22 = Ml22 - Ml22_ref
            dMl32 = Ml32 - Ml32_ref

            log_deltmult.update(
                {
                    alias: [
                        time,
                        np.log(np.absolute(-dMl12)),
                        np.log(np.absolute(-dMl22)),
                        np.log(np.absolute(-dMl32)),
                    ]
                }
            )

        self.log_deltamultipoles = log_deltmult

    def calc_log_multipoles2(self):
        """Compute and assign the natural logarithm of the (l=2)
        multipoles to sim.log_multipoles of the simulations."""
        log_mult = {}
        for alias in self.aliases:
            message(alias)
            item = np.transpose(self.multipoles[alias])
            ml_length = len(item[:, 0])
            if ml_length < self.comm_data_length[alias]:
                self.comm_data_length[alias] = ml_length
                self.distance[alias] = self.distance[alias][:ml_length]
            message(item.shape)
            log_mult.update(
                {
                    alias: [
                        item[: self.comm_data_length[alias], 0],
                        np.log(-item[: self.comm_data_length[alias], 1]),
                        np.log(-item[: self.comm_data_length[alias], 2]),
                        np.log(-item[self.merger_ind[alias] :, 3]),
                    ]
                }
            )
        self.log_multipoles2 = log_mult

    def calc_delta_multipoles2(self):
        """Compute and return the delta multipoles (w.r.t.

        reference (l=2) multipoles)
        """
        log_deltmult = {}
        for alias in self.aliases:
            message(alias)
            # Multipole data
            item = np.transpose(self.multipoles[alias])
            """Length of multipoles."""
            ml_length = len(item[:, 0])
            """Check the lengths of multipole and distance data."""
            if ml_length < self.comm_data_length[alias]:
                """Reset the datalengths."""
                self.comm_data_length[alias] = ml_length
                """Resize the distance array."""
                self.distance[alias] = self.distance[alias][:ml_length]
            message(item.shape)
            """Unpack data as 1d arrays."""
            time, Ml12, Ml22, Ml32 = (
                item[: self.comm_data_length[alias], 0],
                item[: self.comm_data_length[alias], 1],
                item[: self.comm_data_length[alias], 2],
                item[self.merger_ind[alias] :, 3],
            )
            """Compute delta multipoles."""
            Ml12_ref = np.mean(Ml12[30:100])
            Ml22_ref = np.mean(Ml22[30:100])
            Ml32_ref = np.mean(Ml32[-100:])
            dMl12 = Ml12 - Ml12_ref
            dMl22 = Ml22 - Ml22_ref
            dMl32 = Ml32 - Ml32_ref

            log_deltmult.update(
                {alias: [time, np.log(-dMl12), np.log(-dMl22), np.log(dMl32)]}
            )

        self.log_deltamultipoles2 = log_deltmult

    def load_data(self):
        """Load data of the simulations.

        Notes
        -----

        Data is assigned to

                        multipoles: list
                        mass_multipoles:    list
                        spin_multipoles:    list
                        timeaxis:   list
                        mass1:  float
                        mass2:  float
                        mass3:  float
                        delta_t:    float
                        distance:   list
                        merger_ind: int
                        actmerger_time: float
                        dinit:  float
                        data_length:    int
                        dist_data_length:   int
        """
        """Common variables."""
        multipoles_one = []
        masses_one = []
        masses_one_all = []
        M1_one = []
        M2_one = []
        M3_one = []
        delta_t_one = []
        timeaxis_one = []
        # Multipole variables

        all_mass_multipoles_one = []
        all_spin_multipoles_one = []

        # Load Distances
        d_one = []
        timeaxis_one = []
        act_shape_one = []
        d0_one = []
        merger_time_one = []

        ml_data_length = []

        dist_data_length = []
        all_mass_multipoles_1 = []
        all_mass_multipoles_2 = []
        all_mass_multipoles_3 = []

        all_spin_multipoles_1 = []
        all_spin_multipoles_2 = []
        all_spin_multipoles_3 = []

        # simulation directories.
        sim1 = [item + "/" for item in self.aliases]
        # Array for merger index.
        ind_merger_one = np.zeros(len(self.aliases), dtype=np.int32) - 1

        for sim_index, sim_name in enumerate(self.aliases):
            # Loop over simulations.

            message(sim_name)
            message("------------------")

            # Load the qlm_multipole moment data.
            file0 = np.genfromtxt(
                self.ROOTDIR
                / sim_name
                / self.data_dir
                / "quasilocalmeasures-qlm_multipole_moments..asc"
            )

            # Assign the timeaxis.
            cctk_time = file0[:, 8]

            # Assign the multipole data lengths to a list.
            ml_data_length.append(len(cctk_time))

            # Assign the timeaxis to a list.
            timeaxis_one.append(cctk_time)
            # Assign the time stepping to a list.
            delta_t_one.append(cctk_time[1] - cctk_time[0])
            # Load the BH horizon masses.
            M1_one_all = file0[:, 12]
            M2_one_all = file0[:, 13]
            M3_one_all = file0[:, 14]

            ###################################################################
            # I.
            #   1. Load multipole data.
            #   2. Mass data.
            #   3. Find merger index.
            ###################################################################

            # Lists to hold mass multipole data of the three horizons of a
            # simulation.
            amm1 = []
            amm2 = []
            amm3 = []

            # Lists to hold spin multipole data of the three horizons of a
            # simulation.
            asm1 = []
            asm2 = []
            asm3 = []

            for col in range(0, 9):
                # Extract the mass multipole data.
                amm1.append(file0[:, 12 + col * 3])
                amm2.append(file0[:, 13 + col * 3])
                amm3.append(file0[:, 14 + col * 3])

                # Extract the spin multipole data.
                asm1.append(file0[:, 39 + col * 3])
                asm2.append(file0[:, 40 + col * 3])
                asm3.append(file0[:, 41 + col * 3])

            # check for data continuity
            # Load a set of data to chek into a list.
            data0 = [cctk_time, M1_one_all, M2_one_all, M3_one_all]

            # Prepend the timeaxis to the mass multipoles.
            data1 = [cctk_time] + amm1
            data2 = [cctk_time] + amm2
            data3 = [cctk_time] + amm3

            # Prepend the timeaxis to the spin multipoles
            # message(len(data1), len(amm1))

            data4 = [cctk_time] + asm1
            data5 = [cctk_time] + asm2
            data6 = [cctk_time] + asm3

            # Clean the data using handles.cleandata.
            # Collect the data set 1.
            cctk_time, M1_one_all, M2_one_all, M3_one_all = cleandata(data0)

            # Collect the cleaned data set 2 and 3.
            cctk_time, *amm1 = cleandata(data1)
            cctk_time, *amm2 = cleandata(data2)
            cctk_time, *amm3 = cleandata(data3)

            cctk_time, *asm1 = cleandata(data4)
            cctk_time, *asm2 = cleandata(data5)
            cctk_time, *asm3 = cleandata(data6)

            # Find the merger index.
            # Note: This may not work if no merger has happened. This captures
            # the length of data in that case.

            # Use the first non-zero index of mass3 array common horizon as the
            # merger index.

            # Append to merger_one_a.
            try:
                # Try to find the non-zero index of horizon mass3.
                merger_one_a = np.where(M3_one_all != 0)[0]

            except BaseException:
                # Else set merger_one_a to length of mass3.
                merger_one_a = len(M3_one_all)
                message(
                    "Mass3 not non-zero anywhere in the data."
                    "Merger has probably not happened yet."
                    " Reverting to data length."
                )
            # When merger happens, the BH horizon masses
            # mass1 and mass2 acquire the same mass as the
            # common horizon. Find the index where mass1
            # jumps by more than 0.1.
            # Assign to merger_one.

            try:
                # Try to find the jump location of the horizon mass1 data.
                merger_one = np.where(np.diff(M1_one_all) > 0.1)[0]

            except BaseException:
                merger_one = len(M1_one_all)
                message(
                    "Mass1 has no jumps. Merger has probably not happened."
                    "Reverting to data length."
                )

            message("Merger index (through jump in mass1):", merger_one)

            if merger_one_a.any() and merger_one.any():
                # If both indices have been found, select the least of the two
                # as merger index.
                merger_one = min(merger_one_a[0], merger_one[0])

                if merger_one == merger_one_a[0]:
                    message("Merger data set from Mass3.")

                else:
                    message("Merger data set from Mass1 jump.")

            elif not merger_one_a.any() and not merger_one.any():
                # If index of non-zero mass3 and mass1 jump was not found,
                # declare no merger data exists.
                message("No merger data exists")
                merger_one = -1

            else:
                # If both indices are not equal, declare inconsistency but
                # accept the result from either.

                message(
                    "Merger index inconsistency found."
                    "Mergertime may not be correct"
                )

                try:
                    # First try with non-zero mass3 index.
                    merger_one = merger_one[0]
                    message("Merger time set from Mass3")
                except BaseException:
                    # Else assign from mass1 jump.
                    merger_one = merger_one_a[0]
                    message("Merger time set from Mass1 jump")

            # Declare merger time.
            message("Merger time:", delta_t_one[sim_index] * merger_one)
            message("Merger index:", merger_one)

            # Update the merger index of the simulation.
            ind_merger_one[sim_index] = merger_one

            # Load the masses.

            M1_one.append(np.mean(M1_one_all[200:300]))
            M2_one.append(np.mean(M2_one_all[200:300]))
            M3_one.append(np.mean(M3_one_all[-100:]))

            # Load the (l=2) multipoles.
            Ml12_one = file0[:, 18]
            Ml22_one = file0[:, 19]
            Ml32_one = file0[:, 20]

            # message('Ml32_one', len(Ml32_one))

            # Update the mass multipoles for all l.
            all_mass_multipoles_1.append(amm1)
            all_mass_multipoles_2.append(amm2)
            all_mass_multipoles_3.append(amm3)

            # Update the spin multipoles for all l.
            all_spin_multipoles_1.append(asm1)
            all_spin_multipoles_2.append(asm2)
            all_spin_multipoles_3.append(asm3)

            # Gather the mass and spin multipoles together.
            all_mass_multipoles_one.append([amm1, amm2, amm3])
            all_spin_multipoles_one.append([asm1, asm2, asm3])

            # message(len(cctk_time))

            # Gather the l=2 mass multipoles.
            multipoles_one.append(
                np.array([cctk_time, Ml12_one, Ml22_one, Ml32_one])
            )

            # Gather the masses.
            masses_one.append([M1_one, M2_one, M3_one])
            masses_one_all.append([M1_one_all, M2_one_all, M3_one_all])

            ################################################################
            # II . Load the distance data.
            # Compute the distance using
            # the coordinate positions in BHdiagnostics files.
            ################################################################
            temp0 = np.genfromtxt(
                self.ROOTDIR
                / sim1[sim_index]
                / self.data_dir
                / "BH_diagnostics.ah1.gp"
            )
            temp1 = np.genfromtxt(
                self.ROOTDIR
                / sim1[sim_index]
                / self.data_dir
                / "BH_diagnostics.ah2.gp"
            )

            t_coord_0 = temp0[:, 1]
            x_coord_0_locs = temp0[:, 2]
            y_coord_0_locs = temp0[:, 3]

            message("Coord len", len(t_coord_0))
            t_coord_1 = temp1[:, 1]
            x_coord_1_locs = temp1[:, 2]
            y_coord_1_locs = temp1[:, 3]

            temp0 = np.transpose(
                np.array([t_coord_0, x_coord_0_locs, y_coord_0_locs])
            )
            temp1 = np.transpose(
                np.array([t_coord_1, x_coord_1_locs, y_coord_1_locs])
            )

            message("Dist shapes", temp0.shape, temp1.shape)
            # check for data continuity.
            # Load data.
            data0 = temp0
            data1 = temp1

            print(data0)
            # Clean the data.
            temp0 = cleandata(data0)
            temp1 = cleandata(data1)

            message("Dist shapes after cleaning", temp0.shape, temp1.shape)
            print(temp0)
            # Retrieve shape.
            shape0 = temp0.shape
            shape1 = temp1.shape

            message(shape0, shape1)
            # Retrieve timestepping.
            delta_t = scipy.stats.mode(np.diff(temp0[:, 0]))[0]
            # delta_t = np.diff(temp0[:, 0])[0]

            # message('Time step',delta_t)

            # Retrieve merger index.
            mergerind = ind_merger_one[sim_index]

            # Update the merger time list.
            merger_time_one.append(
                ind_merger_one[sim_index] * delta_t_one[sim_index]
            )

            # Find the shorter data. BHdiag1,2 or multipole data.
            act_len = min(shape0[0], shape1[0], ml_data_length[sim_index])

            # If BHdiag 1 and 2 are equal in length and equal to multipole
            # length:
            if (
                shape0[0] == shape1[0]
                and shape0[0] == ml_data_length[sim_index]
            ):
                message(
                    "BH_data length is consistent with multipole data length"
                )

            # If the shortest data length is different from multipole length.
            if act_len == shape0[0] or act_len == shape1[0]:
                message("BH_diagnostics data is shorter")

            # If multipole data is shorter.
            else:
                message("Multipole data is shorter")

            # Update the actual length array.
            act_shape_one.append(act_len)

            # message('Data length',act_len)

            # act_shape_one.append()#
            # act_shape_one.append(min(shape1[0],shape2[0]))
            # message(temp0.shape,temp1.shape)

            # Try to load BHdiag3 file if present.
            try:
                # Try to load the BH_diagnostics file for BH3.
                temp2 = np.genfromtxt(
                    self.ROOTDIR
                    + sim1[sim_index]
                    + self.data_dir
                    + "BH_diagnostics.ah3.gp"
                )[:, np.r_[1, 2, 3]]

                # If merger index is not found from masses set using BH_diag3.

                # If the merger_ind from masses is greater than the BHdiag3
                # data length.

                if mergerind * delta_t >= int(temp2[0, 0]):
                    merger_time_one[sim_index] = temp2[0, 0]
                    mergerind = int(temp2[0, 0] / delta_t)
                    ind_merger_one[sim_index] = mergerind
                    message(
                        "Merger time acquired from BHdiag3",
                        mergerind * delta_t,
                    )
                    message(
                        "Merger index has been updated with info from BHdiag3"
                    )
            except BaseException:
                message("Merger time acquired from masses", mergerind * delta_t)

            message("Merger index", mergerind)

            # FInd shape of data.
            shape0 = temp0.shape
            shape1 = temp1.shape

            # if shape0[0] < shape0[1]:
            #    temp1 = np.transpose(temp1)

            # if shape1[0] < shape1[1]:
            #    temp2 = np.transpose(temp2)

            # Assign the centroid locations to variables.

            # t_coord_0 = t_coord_0  # temp0[:, 0]
            x_coord_0 = x_coord_0_locs  # temp0[:, 1]
            y_coord_0 = y_coord_0_locs  # temp0[:, 2]

            # t_coord_1 = t_coord_1  # temp1[:, 0]
            x_coord_1 = x_coord_1_locs  # temp1[:, 1]
            y_coord_1 = y_coord_1_locs  # temp1[:, 2]

            # t_coord_0      =   temp0[:, 0]
            # x_coord_0      =   temp0[:, 1]
            # y_coord_0      =   temp0[:, 2]
            #
            # t_coord_1      =   temp1[:, 0]
            # x1         =   temp1[:, 1]
            # y1         =   temp1[:, 2]

            # if int((x_coord_0-x_coord_0_locs)[0])!=0:
            #    message('ERRORRRRRRR!')
            #    sys.exit(0)
            # Compute lengths

            len_0 = len(t_coord_0)
            len_1 = len(t_coord_1)

            # Find minimum
            lmin = min(len_0, len_1)
            timeaxis_one.append(t_coord_0)

            # Crop data

            if len_0 < len_1:
                t_coord_1 = t_coord_1[:lmin]
                x_coord_1 = x_coord_1[:lmin]
                y_coord_1 = y_coord_1[:lmin]

            elif len_1 < len_0:
                t_coord_0 = t_coord_0[:lmin]
                x_coord_0 = x_coord_0[:lmin]
                y_coord_0 = y_coord_0[:lmin]

            # Compute differences

            delta_x = x_coord_0 - x_coord_1
            delta_y = y_coord_0 - y_coord_1

            # if sim_index==3:
            # d.append(np.sqrt((temp0[:3014,1]-temp1[:ind_merger[sim_index],1])**2
            # + (temp0[:3014,2]-temp1[:ind_merger[sim_index],2])**2))

            # Compute the Eucledian distance.
            d_sim = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
            d_sim = np.array(d_sim)
            d0_sim = d_sim[0]
            message("Initial true distance", d0_sim)
            d0_one.append(d0_sim)
            d_sim = d_sim / d0_sim
            # d_one.append(np.sqrt((temp0[:act_shape_one[sim_index],1]-
            # temp1[:act_shape_one[sim_index],1])**2
            # + (temp0[:act_shape_one[sim_index],2]-
            # temp1[:act_shape_one[sim_index],2])**2))
            # d_one.append([t_coord_0, d_sim])

            assert len(t_coord_0) == len(
                t_coord_1
            ), "Check time and data axis of distance data"
            d_one.append([t_coord_0, d_sim])

            message(
                "ml_timeaxis_length: %d, Multipole data length: %d,"
                "BHdiag length: %d, Distance length: %d"
                % (
                    len(timeaxis_one[sim_index]),
                    ml_data_length[sim_index],
                    shape1[0],
                    len(d_one[sim_index][0]),
                )
            )
            dist_data_length.append(len(d_sim))

        # multipoles = np.array(multipoles)
        # Convert the acquired data lists into numpy arrays.
        masses_one = np.array(masses_one)
        M1_one = np.array(M1_one)
        M2_one = np.array(M2_one)
        M3_one = np.array(M3_one)

        # message('Multipoles shape',multipoles.shape)

        # message('Masses shape:',masses_one.shape)
        # print("d1", d_one)
        # try:
        #    d_one = np.array(d_one)
        # except Exception as ex:
        #    print(ex)
        #    return d_one

        # d_one.shape
        # d_one.append(np.sqrt((temp0[:act_shape_one[sim_index],1]-
        # temp1[:act_shape_one[sim_index],1])**2
        # + (temp0[:act_shape_one[sim_index],2]
        # -temp1[:act_shape_one[sim_index],2])**2))
        # d_one=np.array(d_one)

        # d_one.shape

        # Plot the distance
        # message('multipoles length',len(multipoles_one))

        # Assign data to sim variables.
        for sim_index, sim_name in enumerate(self.aliases):
            self.multipoles.update({sim_name: multipoles_one[sim_index]})
            self.mass_multipoles.update(
                {sim_name: all_mass_multipoles_one[sim_index]}
            )
            self.spin_multipoles.update(
                {sim_name: all_spin_multipoles_one[sim_index]}
            )
            self.timeaxis.update({sim_name: timeaxis_one[sim_index]})
            self.mass1.update({sim_name: M1_one[sim_index]})
            self.mass2.update({sim_name: M2_one[sim_index]})
            self.mass3.update({sim_name: M3_one[sim_index]})
            self.delta_t.update({sim_name: delta_t_one[sim_index]})
            self.distance.update({sim_name: d_one[sim_index]})
            self.merger_ind.update({sim_name: ind_merger_one[sim_index]})
            self.actmerger_time.update({sim_name: merger_time_one[sim_index]})
            self.dinit.update({sim_name: d0_one[sim_index]})
            self.data_length.update({sim_name: ml_data_length[sim_index]})
            self.comm_data_length.update({sim_name: act_shape_one[sim_index]})
            self.dist_data_length.update(
                {sim_name: dist_data_length[sim_index]}
            )
        # message(self.multipoles)

        # Resize the multipoles data if merger index was updated from BHdiag3.
        self._resize_multipoles()
        # Reverse the BH1 and BH2 data if BH mass2>mass1.
        self._ifreversal()

        import matplotlib.pyplot as plt

        # Plot the distances.
        for alias in self.aliases:
            x_coord = self.distance[alias][0]
            y_coord = self.distance[alias][1]
            length = min(len(x_coord), len(y_coord))
            x_coord = x_coord[:length]
            y_coord = y_coord[:length]
            # message(x,y)

            fig, ax = plt.subplots()
            ax.plot(x_coord, y_coord)
            ax.set_title("Distance vs t/M " + alias)
            # ax.grid(which="both", axis="both")
            ax.set_xlabel("t")
            ax.set_ylabel(r"$d/d_{init}$")
            plt.show()

    def _resize_multipoles(self):
        """Private method to resize the (l=2) multipole data.

        Useful when merger index was updated from BHdiag3.
        """
        for alias in self.aliases:
            # Loop over simulations.
            # message(alias)
            # self.timeaxis[alias] = [item[:self.dist_data_length[alias]]
            # for item in self.timeaxis[alias]]
            # Resize the lengths of the data.
            self.multipoles[alias] = [
                item[: self.dist_data_length[alias]]
                for item in self.multipoles[alias]
            ]
            # self.data_length.update({alias : len(self.multipoles[alias][0])})
            # self.mass_multipoles[alias] = [item[:self.dist_data_length[alias]
            # for item in self.mass_multipoles[alias]]]
            # self.spin_multipoles[alias] = [item[:self.dist_data_length[alias]
            # for item in self.spin_multipoles[alias]]]

    def _ifreversal(self):
        """Private method to reverse the (l=2) multipole data if mass2>mass1.

        Notes
        -----
        Updates:

                sim.multipoles : Resized (l=2) multipole moment data.
        """

        # Flag to identify if reversal is required.
        flag = 0

        for alias in self.aliases:
            # Loop over simulations.
            message("Check for puncture reversal, %s" % alias)
            if alias != "q1a0_a" and alias != "q1a0_b":
                # Condition for reversal decision.
                if self.mass1[alias] < self.mass2[alias]:
                    # Toggle the flag.
                    flag = 1
                    message(
                        "**************************************************"
                    )
                    message("BH 1 and 2 reversal found!!! \n Reversing data...")
                    message(
                        "**************************************************"
                    )
                    message(
                        "original mass1:%f, mass2: %f"
                        % (self.mass1[alias], self.mass2[alias])
                    )
                    # Reverse the data.
                    self.mass1[alias], self.mass2[alias] = (
                        self.mass2[alias],
                        self.mass1[alias],
                    )
                    # Unpack the multipoles data.
                    time, multipole1, multipole2, multipole3 = self.multipoles[
                        alias
                    ]
                    # Reverse the multipole data.
                    multipole1, multipole2 = multipole2, multipole1
                    # Repack the data.
                    self.multipoles[alias] = np.array(
                        [time, multipole1, multipole2, multipole3]
                    )
            if not flag:
                message("Data O.K.")
            return 1.0

    def load_strain(self, start_index=0):
        """Method to load the shear data of simulations."""

        for sim_index, sim_name in enumerate(self.aliases):
            # Loop over simulations.
            # alias = sim_name
            # Set the starting index for data.
            start_index = 0  # start_index = 0#int(190/deltat[j])
            # Load the strain data.
            sim_strain_data = np.genfromtxt(
                self.WAVDIR
                + sim_name
                + "/"
                + self.strain_dir
                + "/strain_"
                + sim_name
                + "_wavextcpm.dat"
            )

            # Load the timeaxis, plus and cross polarized data.
            htdat = sim_strain_data[start_index:, 0]
            hpdat = sim_strain_data[start_index:, 1]
            hxdat = sim_strain_data[start_index:, 2]

            message("The strain file is")
            message(
                self.WAVDIR
                + sim_name
                + "/"
                + self.strain_dir
                + "/strain_"
                + str(alias)
                + "_wavextcpm.dat"
            )

            # Align the peak of the strain with the formation of the common
            # horizon (merger index).

            Lpeak_loc = np.argmax(np.diff(hpdat) ** 2 + np.diff(hxdat) ** 2)

            # Load the common horizon location.
            commhor_loc = self.merger_ind[alias]
            # Compute the shift betweeen the Luminosity peak index and common
            # horizon formation location.
            shift = Lpeak_loc - commhor_loc

            # Update the strain_indexshifts.
            self.strain_indexshifts.update({alias: shift})

            # Load the time stepping.
            # delta_t = self.delta_t[alias]

            # Shift the timeaxis and clip the beginning of data.
            htdat = htdat[start_index:-shift]
            hpdat = hpdat[start_index + shift :]
            hxdat = hxdat[start_index + shift :]

            # Update the sim.strain
            self.strain.update({alias: [htdat, hpdat, hxdat]})

            if vlconf.print_verbosity > 1:

                fig, ax = plt.subplots()

                # Plot the strains.
                ax.plot(htdat, hpdat, label="data " + alias)
                ax.set_ylabel("Strain")
                ax.set_xlabel("Time (s)")
                # plt.grid(which="both", axis="both")
                # plt.xlim(-600,400)
                plt.legend()
                plt.show()

        return 1

    def calc_amp_phase(self):
        """Extract the amplitude and the phase from strain data."""
        from waveformtools.waveformtools import xtract_camp, xtract_cphase

        for alias in self.aliases:
            # Loop over simulations.

            # Load the plus and cross polarized strain data.
            hpdat = self.strain[alias][1]
            hxdat = self.strain[alias][2]
            # Load the time stepping.
            delta_t = self.delta_t[alias]
            # Extract and update the amplitude and phases.
            self.strain_phase.update(
                {
                    alias: (
                        xtract_cphase(
                            hpdat, hxdat, delta_t=delta_t, to_plot="yes"
                        )
                    )
                }
            )
            self.strain_amplitude.update({alias: xtract_camp(hpdat, hxdat)})
            self.strain_frequency.update(
                {alias: np.diff(self.strain_phase[alias]) / delta_t}
            )
        return 1

    def ret_horizon_radii(self):
        """Retrieve the radius of the common horizon
        at the time of formation."""

        # Dictionary to hold the areal radii of the horizons.
        self.areal_radii = {}
        for alias in self.aliases:
            # Loop over simulations.

            # Load the BHdiagnostics file to load the radius.
            ar_rad0 = np.genfromtxt(
                self.ROOTDIR
                + alias
                + "/"
                + self.data_dir
                + "BH_diagnostics.ah1.gp"
            )[:, 27]
            ar_rad1 = np.genfromtxt(
                self.ROOTDIR
                + alias
                + "/"
                + self.data_dir
                + "BH_diagnostics.ah2.gp"
            )[:, 27]
            ar_rad2 = 1.75
            try:
                ar_rad2 = np.genfromtxt(
                    self.ROOTDIR
                    + alias
                    + "/"
                    + self.data_dir
                    + "BH_diagnostics.ah3.gp"
                )[:, 27]
            except BaseException:
                message("No BHdiagnostics 3 file found for %s" % alias)

            # Load the data for this simulation in to a dictionary.
            self.areal_radii.update({alias: [ar_rad0, ar_rad1, ar_rad2]})

        return 1

    def get_BH_locations(self, alias=None):
        """Get the co-ordinate locations of the BHs.

        Parameters
        ----------

        alias  :    str, optional.
                                The simulation label. If not specified, then
                                all available simulationswill be processed.

        Returns
        -------
        """

        # A dictionary to store BH location data
        self.BH_locations = {}

        if not alias:
            list_of_aliases = self.aliases

        else:
            list_of_aliases = [alias]

        for alias in list_of_aliases:
            # Load the data
            # alias = 'q1a0_a'
            # sim_index = 0

            # Masses of the horizons
            m1 = self.mass1[alias]
            m2 = self.mass2[alias]
            # M = m1 + m2

            flag = 1
            bh1 = np.genfromtxt(
                self._get_file_path_from_str(string="*.ah1.gp", alias=alias)
            )
            bh2 = np.genfromtxt(
                self._get_file_path_from_str(string="*.ah2.gp", alias=alias)
            )
            try:
                bh3 = np.genfromtxt(
                    self._get_file_path_from_str(string="*.ah3.gp", alias=alias)
                )
            except Exception as excep:
                message("BH3 file not found!", excep)
                flag = -1

            # Read co-ordinates.

            bhd1_time = bh1[:, 1]
            bhd1_x = bh1[:, 2]
            bhd1_y = bh1[:, 3]
            bhd1_z = bh1[:, 4]

            bhd2_time = bh2[:, 1]
            bhd2_x = bh2[:, 2]
            bhd2_y = bh2[:, 3]
            bhd2_z = bh2[:, 4]

            if flag != -1:
                bhd3_time = bh3[:, 1]
                bhd3_x = bh3[:, 2]
                bhd3_y = bh3[:, 3]
                bhd3_z = bh3[:, 4]
            else:
                bhd3_time = None
                bhd3_x = None
                bhd3_y = None
                bhd3_z = None

            bh1_loc = [bhd1_time, bhd1_x, bhd1_y, bhd1_z]
            bh2_loc = [bhd2_time, bhd2_x, bhd2_y, bhd2_z]
            bh3_loc = [bhd3_time, bhd3_x, bhd3_y, bhd3_z]

            self.BH_locations.update({alias: [bh1_loc, bh2_loc, bh3_loc]})

    def get_CoM_locations(self, alias=None):
        """Get the CoM location of the given simulation.

        Parameters
        ----------
        alias: str, optional.
               The simulation label. If not specified, then
               all available simulationswill be processed.

        Returns
        -------
        self.CoM_locations: dict
                            A dictionary of lists containing
                            CoM locations of the simulations.

        Notes
        -----
        This fetches the location of the BHs from data files.
        """

        # Storage for CoM locations
        self.CoM_locations = {}

        # Aliases to run over.
        if not alias:
            list_of_aliases = self.aliases

        else:
            list_of_aliases = [alias]

        for alias in list_of_aliases:
            try:
                BH_locs_sim = self.BH_locations[alias]
            except Exception as excep:
                message(excep)
                self.get_BH_locations(alias)
                BH_locs_sim = self.BH_locations[alias]

            # Unpack the location data.
            BH1_loc, BH2_loc, _ = BH_locs_sim

            bh1_time, bh1_x, bh1_y, bh1_z = BH1_loc
            bh2_time, bh2_x, bh2_y, bh2_z = BH2_loc
            # bh3_time, bh3_x, bh3_y, bh3_z = BH3_loc

            # The masses.
            mass1 = self.mass1[alias]
            mass2 = self.mass2[alias]
            total_mass = mass1 + mass2

            # End time
            max_len = min(self.merger_ind[alias], len(bh1_time), len(bh2_time))

            # CoM location
            T_com = bh1_time[:max_len]
            X_com = (mass1 * bh1_x[:max_len] + mass2 * bh2_x[:max_len]) / (
                total_mass
            )
            Y_com = (mass1 * bh1_y[:max_len] + mass2 * bh2_y[:max_len]) / (
                total_mass
            )
            Z_com = (mass1 * bh1_z[:max_len] + mass2 * bh2_z[:max_len]) / (
                total_mass
            )

            self.CoM_locations.update({alias: [T_com, X_com, Y_com, Z_com]})

    def get_CoM_mean_motion(self, alias=None):
        """Get the mean motion of the CoM.

        Parameters
        ----------
        alias: str, optional.
               The simulation label. If not specified, then
               all available simulationswill be processed.

        Returns
        -------
        alpha: dict
               A dictionary containing the
               mean CoM displacement array.

        beta: dict
              A dictionary containing the
              mean CoM velocity array.
        """
        CoM_motion_params = {}
        # Aliases to run over.

        from waveformtools.CoM import (
            X_com_moments,
            compute_com_alpha,
            compute_com_beta,
        )

        if not alias:
            list_of_aliases = self.aliases

        else:
            list_of_aliases = [alias]

        for alias in list_of_aliases:
            try:
                taxis, X_com, Y_com, Z_com = self.CoM_locations[alias]
            except Exception as excep:
                message(excep)
                self.get_CoM_locations(alias)
                taxis, X_com, Y_com, Z_com = self.CoM_locations[alias]

            all_coords = [X_com, Y_com, Z_com]

            zeroth_moments = X_com_moments(taxis, all_coords, 0)
            first_moments = X_com_moments(taxis, all_coords, 1)

            Xcom_0 = np.array(
                [zeroth_moments[label][0] for label in zeroth_moments.keys()]
            )
            Xcom_1 = np.array(
                [first_moments[label][0] for label in zeroth_moments.keys()]
            )

            ti = taxis[0]
            tf = taxis[-1]

            alpha = compute_com_alpha(ti, tf, Xcom_0, Xcom_1)
            beta = compute_com_beta(ti, tf, Xcom_0, Xcom_1)

            CoM_motion_params.update({alias: [alpha, beta]})
        return CoM_motion_params

    def _get_file_path_from_str(self, alias, string=None):
        """Get the path of a file that contains.

        the given string in its name.

        Parameters
        ----------
        alias: str, optional.
               The simulation label. If not specified, then
               all available simulationswill be processed.

        string: str
                The string of a part of a file name.

        Returns
        -------
        file_path: str
                   The full path of the file

        Notes
        -----
        The first occuring instance of the file is returned
        if there are multiple files found.
        """

        from pathlib import Path

        # The directory to look into.
        path = self.ROOTDIR + alias

        dir = Path(path)
        # The path of the file.
        # message(list(dir.rglob(string)))
        try:
            file_path = list(dir.rglob(string))[0]
        except Exception as excep:
            message("File not found!", excep)

        return file_path

    def load_NP_1d_data(self, np_qty="sigma", source="qlm"):
        """Load the 1d NP quantities."""

        import re

        np_data_dict = {}

        if self.NP_1d is None:
            self.NP_1d = {}

        np_data_alias_dict = {}

        for alias in self.aliases:
            if source == "qlm":
                file_string = (
                    f"{self.data_dir}quasilocalmeasures"
                    "-qlm_newman_penrose..asc"
                )
            elif source == "ih":
                file_string = (
                    f"{self.data_dir}isolatedhorizon" "-ih_newman_penrose..asc"
                )

            full_file_path = self._get_file_path_from_str(
                alias, string=file_string
            )

            with open(full_file_path, "r") as file:
                for line_index in range(15):
                    fline = next(file)
                    # message(fline)
                    str_match = re.search(f"\\d\\d:qlm_np{np_qty}", fline)
                    # message(str_match)
                    if str_match is not None:
                        start_col = int(str_match[0][:2])
                        # message(start_col)
                        break

            NP_all_data = np.genfromtxt(full_file_path)[
                :,
                np.r_[
                    8,
                    start_col - 1,
                    start_col,
                    start_col + 1,
                    start_col + 2,
                    start_col + 3,
                    start_col + 4,
                ],
            ]

            np_data_alias_dict.update({alias: NP_all_data})

        self.NP_1d.update({np_qty: np_data_alias_dict})
