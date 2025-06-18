import numpy as np
from spectools.spherical.swsh import Yslm_vec
from spectools.spherical.swsh import Yslm_vec
from waveformtools.dataIO import construct_mode_list
from waveformtools.waveformtools import message
from waveformtools.diagnostics import MethodInfo

# from numba import int32, float64, complex128, uint16, types, typed,
# from numba.types import List, Dict


# key and value types
# emode_list = [[2, 2], [0, 0]]
# emode_dict = {''}

# spec =[
#    ('modes_data', complex128[:]),
#    ('zero_modes', List),
#    ('non_zero_modes', List),
#    ('spin_weight', int32),
#    ('ell_max', uint16),
#    ('modes_list', list),
#    ('modes_dict', )
# ]


class SingleMode:
    """Storage and handling of a single mode"""

    def __init__(
        self,
        modes_data=None,
        zero_modes=None,
        non_zero_modes=None,
        spin_weight=0,
        ell_max=8,
        modes_list=None,
        modes_dict=None,
        tol=1e-8,
        extra_mode_axes_shape=None,
        Grid=None,
        vec_modes=None,
        func=None,
        label=None,
        sine_power=0,
    ):
        self._label = label
        self._modes_data = modes_data
        self._zero_modes = zero_modes
        self._non_zero_modes = non_zero_modes
        self._spin_weight = spin_weight
        self._ell_max = ell_max
        self._modes_list = modes_list
        self._tol = tol
        self._extra_mode_axes_shape = extra_mode_axes_shape
        self._Grid = Grid
        self._vec_modes = vec_modes
        self._func = func
        self._sine_power = sine_power

        if (np.array(modes_data) == np.array(None)).all():
            created = False
        else:
            created = True

        if isinstance(modes_dict, dict):
            message(
                "Creating SingleMode obj from modes dict...",
                message_verbosity=2,
            )

            ell_keys_str = list(modes_dict.keys())

            ell_keys_dict = [int(item[1:]) for item in ell_keys_str]

            ell_max_dict = max(ell_keys_dict)

            message("Parsed dict ell_max as", ell_max_dict, message_verbosity=2)

            # if self._ell_max < ell_max:
            self._ell_max = ell_max_dict

            if not self._modes_list:
                self._modes_list = construct_mode_list(
                    ell_max=self.ell_max, spin_weight=self.spin_weight
                )

            self.create_modes_array()
            created = True
            for ell, emm_list in self.modes_list:
                for emm in emm_list:
                    value = modes_dict[f"l{ell}"][f"m{emm}"]
                    self.set_mode_data(ell=ell, emm=emm, data=value)

        if not created:
            self.create_modes_array()

        if Grid is None:
            from spectools.spherical.grids import GLGrid
            from spectools.spherical.grids import GLGrid

            self._Grid = GLGrid(L=ell_max)

        self._modes_spherepack = None

        if not (np.array(self.vec_modes) == np.array(None)).all():
            self.construct_from_vec_modes()

        self.St, _ = self.Grid.meshgrid

    @property
    def label(self):
        return self._label

    @property
    def vec_modes(self):
        return self._vec_modes

    @property
    def extra_mode_axes_shape(self):
        return self._extra_mode_axes_shape

    @property
    def tol(self):
        return self._tol

    @property
    def Grid(self):
        return self._Grid

    @property
    def func(self):
        return self._func

    @property
    def sine_power(self):
        """This is just kept for
        bookkeeping purposes."""
        return self._sine_power

    @property
    def n_modes(self):
        return (self.ell_max+1)**2 - self.spin_weight**2
    
    @property
    def modes_data(self):
        return self._modes_data

    @property
    def modes_spherepack(self):
        """Get the modes in spherepack convention. Note that
        this assumes expansion of a real function in real spherical
        harmonics"""
        if (np.array(self._modes_spherepack) == np.array(None)).any():

            self.get_in_spherepack_convention()

        return self._modes_spherepack

    def get_modes_dict(self, modes_list=None):
        """Get a dictionary representation of the
        modes requested"""

        modes_dict = {}

        if not modes_list:
            message("Create SingleMode modes list", message_verbosity=2)
            modes_list = self.modes_list
            message(f"mode list {modes_list}", message_verbosity=3)

        for ell, emm_list in modes_list:
            modes_dict.update({f"l{ell}": {}})

            for emm in emm_list:
                message(f"Get modes dict l{ell} m{emm}", message_verbosity=3)
                value = self.mode(ell, emm)

                modes_dict[f"l{ell}"].update({f"m{emm}": value})

        return modes_dict

    @property
    def modes_list(self):
        message(f"ell_max of modes list {self.ell_max}", message_verbosity=3)
        message(f"spin weight {self.spin_weight}", message_verbosity=3)

        if not self._modes_list:
            self._modes_list = construct_mode_list(
                self.ell_max, self.spin_weight
            )

        # message(f"ell max of created modes list {max([item[0]
        # for item in self._modes_list])}",
        #        message_verbosity=4)

        return self._modes_list

    @property
    def ell_max(self):
        if self.Grid is not None:
            if self.Grid.grid_type == "GL":
                if self._ell_max is None:
                    self._ell_max = self.Grid.L

                assert self._ell_max <= self.Grid.L, (
                    "The grid L must be"
                    "greater than or equal to ell_max"
                    "of this SingleMode obj"
                )

        message(f"Accessing attr ell max {self._ell_max}", message_verbosity=4)

        return self._ell_max

    @property
    def spin_weight(self):
        """The spin weight of the modes"""
        return self._spin_weight

    def zero_modes(self, tol=1e-8):
        re_eval = False

        if not self._zero_modes:
            message("Zero modes will be computed afresh", message_verbosity=2)
            re_eval = True

        if self.tol is not None:
            if self.tol != tol:
                message(
                    "Tol has changed. Recalculating zero modes",
                    message_verbosity=2,
                )
                re_eval = True
                self._tol = tol

        if re_eval:
            self.compute_zero_modes(tol=self.tol)

        return self._zero_modes

    def non_zero_modes(self, tol=None):
        re_eval = False

        if not self._non_zero_modes:
            message("Zero modes will be computed afresh", message_verbosity=2)
            re_eval = True

        if tol is None:
            tol = self.tol

        if self.tol != tol:
            message(
                "Tol has changed. Recalculating zero modes",
                message_verbosity=2,
            )
            re_eval = True
            self._tol = tol

        if re_eval:
            self.compute_zero_modes(tol=tol)

        return self._non_zero_modes

    def mode(self, ell, emm):
        """Return a particular mode

        Parameters
        ----------
        ell : int
              The :math:`\\ell` index of the mode.
        emm : int
              The :math:`m` index of the mode.

        Returns
        -------
        mode_data : float
                    The value of the requested mode.
        """
        if abs(emm) > ell:
            raise ValueError(
                "Please request a valid mode ( abs(emm) > abs(ell) here)"
            )

        elif ell < abs(self.spin_weight):
            raise ValueError(
                "Please request a valid mode with ell >= abs(spin_weight)"
            )
        
        vec_idx = ell**2 + emm + ell #- self.spin_weight**2

        return self._modes_data[vec_idx]

    def create_modes_array(self):
        """Create a modes array and initialize it with zeros.

        Parameters
        ----------
        ell_max : int
                  The maximum :math:`\\ell` value of the modes.

        Returns
        -------
        self.modes_array : modes_array
                           sets the `self.modes_array` attribute.

        Notes
        -----
        The ordering of axis is (ell, emm, extra axis)

        """

        message("Creating single mode data array", message_verbosity=3)

        ell_max = self.ell_max

        # Check ell_max
        if ell_max is None:
            try:
                ell_max = self.ell_max
            except Exception as ex:
                message(ex)
                raise NameError("Please supply ell_max")

        if self.modes_list is None:
            self._modes_list = construct_mode_list(
                ell_max=ell_max, spin_weight=self.spin_weight
            )

        if type(self.extra_mode_axes_shape) is tuple:
            self._modes_data = np.zeros(
                ((ell_max + 1) ** 2, *self.extra_mode_axes_shape),
                dtype=np.complex128,
            )
        else:
            self._modes_data = np.zeros((ell_max + 1) ** 2, dtype=np.complex128)

    def construct_from_vec_modes(self, vec_modes):
        """Load the single modes object using the modes vector"""

        ell_max_vec_modes = int(np.sqrt(len(vec_modes)) - 1)

        if (ell_max_vec_modes + 1) ** 2 != len(vec_modes):
            raise ValueError(
                "The length of vec_modes does not match the filled modes array"
            )

        self._modes_data = vec_modes

    def set_mode_data(self, data, ell=None, emm=None):
        """Set the mode array data for the respective :math:`(\\ell, m)` mode.

        Parameters
        ----------
        ell : int
              The :math:`\\ell` polar mode number.
        emm_value : int
                    The :math:`m` azimuthal mode number.
        data : float
               The value of the mode data of the corresponding mode.

        """
        if (ell is None) and (emm is None):
            self._modes_data = np.array(data)
            # if len(value.shape)>1:
            #    self._extra_mode_axes_shape =
        else:
            # elif int(ell)==ell and int(emm)==emm:
            # Compute the linear vector index
            vec_idx = (ell) ** 2 + emm + ell - self.spin_weight**2
            message(f"Setting l{ell} m{emm} data {data}", message_verbosity=4)

            # Set the mode data.
            self._modes_data[vec_idx] = data
            message(f"Set mode data {self.mode(ell, emm)}", message_verbosity=4)
        # else:
        #    raise TypeError("Please provide integer values for ell and emm")

    def compute_zero_modes(self, tol=1e-8):
        """Get the details of the zero modes in
        the data below a given tolerance"""

        zero_modes_list = []
        non_zero_modes_list = []

        for ell, emm_list in self.modes_list:
            z_emm_list = []
            nz_emm_list = []

            for emm in emm_list:
                if abs(self.mode(ell, emm)) < tol:
                    z_emm_list.append(emm)
                else:
                    nz_emm_list.append(emm)

            if len(z_emm_list) > 0:
                zero_modes_list.append([ell, z_emm_list])

            if len(nz_emm_list) > 0:
                non_zero_modes_list.append([ell, nz_emm_list])

        self._zero_modes = zero_modes_list
        self._non_zero_modes = non_zero_modes_list

    def contract(
        self,
        ell_max=None,
        all_available_modes=True,
        only_modes_above_tol=False,
        tol=None,
    ):
        """Contract the modes to get co-ordinate space
        representation of the function.

        Parameters
        ----------
        ell_max: int, optional
                 The max :math:`\\ell` upto which to contract
                 the modes to. If None, then the available
                 `ell_max` will be chosen.

        all_available_modes: bool,optional
                             Wheter or not to use all available modes.

        Returns
        -------
        func: 2darray
              The function on the sphere in coordinate space
              representation.
        """

        if ell_max is None:
            ell_max = self.ell_max

        if self.Grid is not None:
            if self.Grid.grid_type == "GL":
                assert self.Grid.L >= ell_max, (
                    "The grid L must be greater than"
                    "or equal to ell_max of the requested SH expansion"
                )

        if all_available_modes:
            modes = self  # self.get_modes_dict()
        else:
            raise NotImplementedError

        if only_modes_above_tol:
            raise NotImplementedError
            modes = self.non_zero_modes(tol=tol)

        from spectools.spherical.transforms import SHContract
        from spectools.spherical.transforms import SHContract
        from waveformtools.diagnostics import method_info

        minfo = method_info()

        func = SHContract(
            modes, self.Grid, ell_max, method_info=minfo, vectorize=True
        )

        return func

    def calculate_power_monitor_ell(self, ell_max=None):
        """Get the power in each ell mode

        Parameters
        ----------
        single_mode : modes dict / single_mode
                      A dictionary of modes or an instance
                      of the single mode class, that contains
                      the modes of an SH expansion.

        Returns
        -------
        power : list
                    A list containing the power
                    in each :math:`\\ell` mode
                    of the system.

        cumulative_power
        cumulative_even_power
        cumulative_even_power_axis
        cumulative_power_axis
        power_ell
        """

        if ell_max is None:
            ell_max = self.ell_max

        power = []

        for ell in range(ell_max + 1):
            power_this_ell = 0

            for emm in range(-ell, ell + 1):
                power_this_ell += np.absolute(self.mode(ell, emm)) ** 2

            power.append(power_this_ell)

        even_power = [
            power[index] + power[index + 1] for index in range(0, ell_max, 2)
        ]

        if ell_max % 2 == 1:
            even_power += [power[-1]]

        cumulative_even_power = [
            np.sum(even_power[:index]) for index in range(len(even_power))
        ]
        cumulative_power = [
            np.sum(power[:index]) for index in range(len(power))
        ]

        cumulative_even_power_axis = list(np.arange(0, ell_max, 2))

        if ell_max % 2 == 1:
            cumulative_even_power_axis += [ell_max]

        self.cumulative_power = cumulative_power
        self.cumulative_even_power = cumulative_even_power

        self.cumulative_even_power_axis = cumulative_even_power_axis
        self.power_axis = np.arange(ell_max + 1)

        self.power_ell = power

    def compare_modes(self, other_modes, prec=18):
        ell_max1 = self.ell_max
        ell_max2 = other_modes.ell_max

        ell_max = min(ell_max1, ell_max2)

        # result = False

        for ell in range(ell_max + 1):
            for emm in range(-ell, ell + 1):
                mode1 = other_modes.mode(ell, emm)
                mode2 = self.mode(ell, emm)

                np.testing.assert_almost_equal(
                    mode1, mode2, prec, f"The l{ell} m{emm} mode must equal"
                )

    def compare_modes_dict(self, other_modes_dict, prec=18):
        """Compare the Single modes object against a given modes
        dictionary"""

        for ell_key, ell_modes in other_modes_dict.items():
            ell = int(ell_key[1:])
            for emm_key, other_mode in ell_modes.items():
                emm = int(emm_key[1:])
                np.testing.assert_almost_equal(
                    self.mode(ell, emm),
                    other_mode,
                    prec,
                    f"The l{ell} m{emm} mode must equal at precision {prec}",
                )

    def truncate_modes(self, ell_max_choice):
        """Returns a new SingleModes object containing
        only modes upto the given `ell_max_choice`"""

        trunc_modes = SingleMode(
            ell_max=ell_max_choice,
            spin_weight=self.spin_weight,
            Grid=self.Grid,
            extra_mode_axes_shape=self.extra_mode_axes_shape,
            tol=self.tol,
        )

        trunc_modes._modes_data = self._modes_data[
            : (ell_max_choice + 1)**2 
        ]

        return trunc_modes

    def get_expansion_residues(self, func=None):
        """Get the expansion residues"""

        if (np.array(func) == np.array(None)).all():
            func = self.func

        residues = [np.sum(func**2)]
        modes_list = construct_mode_list(ell_max=self.ell_max, spin_weight=self.spin_weight)
        theta_grid, phi_grid = self.Grid.meshgrid
        recon_func = np.zeros(theta_grid.shape, dtype=np.complex128)

        for ell, emm_list in modes_list:
            for emm in emm_list:
                Clm = self.mode(ell, emm)
                message(
                    f"Clm shape in SHContract {Clm.shape}", message_verbosity=4
                )
                recon_func += Clm * Yslm_vec(
                    spin_weight=self.spin_weight,
                    ell=ell,
                    emm=emm,
                    theta_grid=theta_grid,
                    phi_grid=phi_grid,
                )

            res = np.sum(np.absolute(recon_func - func) ** 2)
            residues.append(res)

        return residues

    def get_in_spherepack_convention(self):
        """Convert the modes into spherepack convention. This
        returns a set of complex modes but only the real part
         is to be used. The imaginary part may be useful for
         error checking and should be small"""

        Rlm = []

        for emm in range(self.ell_max + 1):
            for ell in range(self.ell_max + 1):

                if abs(emm) > ell:
                    continue

                val_pm = self.mode(ell, emm)
                val_mm = self.mode(ell, -emm)

                if emm == 0:
                    factor = 1 / np.sqrt(2 * np.pi)
                    # val_sp*=np.sqrt(2*np.pi)
                    re_Ylm_val = val_pm.real

                else:
                    # val_sp*=np.sqrt(4*np.pi)
                    factor = 1 / np.sqrt(
                        4 * np.pi
                    )  # (-1)**emm * 5/2 #(-1)**emm * np.sqrt(4*np.pi)

                    re_Ylm_val = (
                        1 * (val_mm + (-1) ** emm * val_pm) / np.sqrt(2)
                    )

                re_Ylm_val *= factor

                Rlm.append(re_Ylm_val)

        self._modes_spherepack = Rlm

    def vector(self):
        """return the vector of modes in ell - emm format
        (opposite to spherepack's emm - ell format"""

        vec = self._modes_data.copy()
        # for ell in range(self.ell_max+1):
        #    for emm in range(-ell, ell+1):
        #        vec.append(self.mode(ell, emm))

        return np.array(vec)

    def evaluate_old(self, theta, phi, ell_max=None):
        """Evaluate the expansion at requested angular coordinates
         by looping over SWSH and summing. SWSHs are generated using
        `Yslm_vec`
        """

        if ell_max is None:
            ell_max = self.ell_max

        val = 0

        for ell in range(ell_max + 1):
            for emm in range(-ell, ell + 1):

                Ylm = Yslm_vec(
                    spin_weight=self.spin_weight,
                    theta_grid=theta,
                    phi_grid=phi,
                    ell=ell,
                    emm=emm,
                    cache=False,
                )

                val += self.mode(ell, emm) * Ylm

        # val = np.sum(self._modes_data * sYlm._modes_data)

        return val

    def evaluate_old_1(self, theta, phi, ell_max=None):
        """Evaluate the expansion at requested angular coordinates
        by generating SWSHs in serial and vectorizing the
        summation"""

        if ell_max is None:
            ell_max = self.ell_max

        from spectools.spherical.swsh import create_Yslm_modes_array
        from spectools.spherical.swsh import create_Yslm_modes_array

        Yslm = create_Yslm_modes_array(
            theta=float(theta),
            phi=float(phi),
            ell_max=ell_max,
            spin_weight=self.spin_weight,
        )

        val = np.sum(self._modes_data * Yslm.sYlm_modes._modes_data)

        return val

    def evaluate_angular(self, theta=None, phi=None, ell_max=None):
        """Evaluate the expansion at requested angular coordinates
        by generating SWSHs in parallel and vectorizing the
        summation"""
        from spectools.spherical.Yslm_mp import Yslm_mp
        from spectools.spherical.Yslm_mp import Yslm_mp

        if ell_max is None:
            ell_max = self.ell_max

        if (np.array(theta) == np.array(None)).all() or (
            np.array(phi) == np.array(None)
        ).all():
            theta, phi = self.Grid.meshgrid

        sYlm = Yslm_mp(
            ell_max=ell_max, spin_weight=self.spin_weight, theta=theta, phi=phi
        )
        sYlm.run()
        Ylm_vec = sYlm.sYlm_modes._modes_data
        modes_data_len = len(Ylm_vec)

        val = np.tensordot(
            self._modes_data[:modes_data_len], Ylm_vec, axes=((0), (0))
        )

        return val

    def evaluate_sp(self, theta, phi, ell_max=None):
        """Evaluate the expansion at requested angular coordinates
        by computing SWSHs using the spherical package"""

        if ell_max is None:
            ell_max = self.ell_max

        from spectools.spherical.swsh import create_spherical_Yslm_modes_array
        from spectools.spherical.swsh import create_spherical_Yslm_modes_array

        sYlm = create_spherical_Yslm_modes_array(
            theta=float(theta),
            phi=float(phi),
            ell_max=ell_max,
            spin_weight=self.spin_weight,
        )

        val = np.sum(self._modes_data * sYlm._modes_data)

        return val

    def compute_spatial_detivatives(self):
        """Given the modes, compute its spatial derivatives"""

        assert self.spin_weight == 0, (
            "Derivatives have only been implemented"
            "for spin weight zero harmonics "
        )

        from qlmtools.differentiation import DerivSHFromSpec

        minfo = method_info(
            diff_method="SH", ell_max=self.ell_max, int_method="GL", reg=False
        )

        return DerivSHFromSpec(self, minfo)

    def plot_residues(self, orig_func=None, *args, **kwargs):
        """Plot the residues of this expanion"""

        if (np.array(orig_func) == np.array(None)).all():
            orig_func = self.func

        residues = self.get_expansion_residues(orig_func)

        ell_axis = np.arange(abs(self.spin_weight)-1, self.ell_max + 1)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.scatter(ell_axis, residues, *args, **kwargs)
        ax.set_title(r"Residues vs $\ell$")
        ax.set_ylabel("Residues")
        ax.set_xlabel(r"$\ell$")
        ax.set_xticks(ell_axis)
        plt.show()
