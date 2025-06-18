import numpy as np
from spectools.spherical.transforms import SHExpand
from spectools.spherical.transforms import SHExpand
from spectools.spherical.grids import GLGrid
from spectools.spherical.grids import GLGrid
from waveformtools.diagnostics import MethodInfo as method_info
from waveformtools.waveformtools import message
import unittest, pytest
from vlconf.verbosity import levels
from waveformtools.single_mode import SingleMode

lv = levels()
lv.set_print_verbosity(2)


class TestGLGridYlm(unittest.TestCase):
    def test_ylm_single_mode_recovery(self):
        """Test single mode recovery for every mode
        upto a given L"""

        info = GLGrid(L=11)
        minfo = method_info(ell_max=11, int_method="GL")

        from spectools.spherical.swsh import Yslm_vec
        from spectools.spherical.swsh import Yslm_vec

        theta_grid, phi_grid = info.meshgrid

        for ell in range(info.L + 1):
            for emm in range(-ell, ell + 1):
                message("Testing", ell, emm, message_verbosity=2)

                Ylm = Yslm_vec(
                    spin_weight=0,
                    ell=ell,
                    emm=emm,
                    theta_grid=theta_grid,
                    phi_grid=phi_grid,
                )
                # Ylm = Yslm_prec(spin_weight=0, ell=ell, emm=emm, theta=theta_grid, phi=phi_grid)

                test_dist = 3.12345 * Ylm

                test_modes = SHExpand(
                    func=test_dist, method_info=minfo, info=info
                )

                # Test mode coefficient
                np.testing.assert_almost_equal(
                    test_modes.mode(ell, emm),
                    3.12345,
                    13,
                    f"The l{ell} m{emm} mode"
                    "must be 3.12345 upto"
                    "12 digits",
                )

        def test_ylm_multiple_mode_recovery(self):
            """Test multi mode recovery starting with
            a superposition of modes upoto L"""

            info = GLGrid(L=11)
            minfo = method_info(ell_max=11, int_method="GL")

            from spectools.spherical.swsh import Yslm_vec
            from spectools.spherical.swsh import Yslm_vec

            theta_grid, phi_grid = info.meshgrid

            Yl0m0 = Yslm_vec(
                spin_weight=0,
                ell=0,
                emm=0,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )
            Yl1mm1 = Yslm_vec(
                spin_weight=0,
                ell=1,
                emm=-1,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )
            Yl2m2 = Yslm_vec(
                spin_weight=0,
                ell=2,
                emm=2,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )
            Yl7m6 = Yslm_vec(
                spin_weight=0,
                ell=7,
                emm=6,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )
            Yl9m5 = Yslm_vec(
                spin_weight=0,
                ell=9,
                emm=5,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )
            Yl11mm11 = Yslm_vec(
                spin_weight=0,
                ell=11,
                emm=-11,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )

            mode_amps = np.random.randn(6)

            mode_amps_dict = {
                "l0m0": mode_amps[0],
                "l1m-1": mode_amps[1],
                "l2m2": mode_amps[2],
                "l7m6": mode_amps[3],
                "l9m5": mode_amps[4],
                "l11m-11": mode_amps[5],
            }

            message(
                "Mode amplitides assumed", mode_amps_dict, message_verbosity=2
            )

            test_data = (
                mode_amps_dict["l0m0"] * Yl0m0
                + mode_amps_dict["l1m-1"] * Yl1mm1
                + mode_amps_dict["l2m2"] * Yl2m2
                + mode_amps_dict["l7m6"] * Yl7m6
                + mode_amps_dict["l9m5"] * Yl9m5
                + mode_amps_dict["l11m-11"] * Yl11mm11
            )

            test_modes = SHExpand(func=test_data, method_info=minfo, info=info)

            # Test the mode coeffs

            rec_amps_dict = {
                "l0m0": test_modes.mode(0, 0),
                "l1m-1": test_modes.mode(1, -1),
                "l2m2": test_modes.mode(2, 2),
                "l7m6": test_modes.mode(7, 6),
                "l9m5": test_modes.mode(9, 5),
                "l11m-11": test_modes.mode(11, -11),
            }

            np.testing.assert_almost_equal(
                rec_amps_dict["l0m0"],
                mode_amps_dict["l0m0"],
                14,
                "The l0m0 mode must be recovered" "to 14 digits",
            )

            np.testing.assert_almost_equal(
                rec_amps_dict["l1m-1"],
                mode_amps_dict["l1m-1"],
                14,
                "The l1m-1 mode must be recovered" "to 14 digits",
            )

            np.testing.assert_almost_equal(
                rec_amps_dict["l2m2"],
                mode_amps_dict["l2m2"],
                14,
                "The l2m2 mode must be recovered" "to 14 digits",
            )

            np.testing.assert_almost_equal(
                rec_amps_dict["l7m6"],
                mode_amps_dict["l7m6"],
                14,
                "The l7m6 mode must be recovered" "to 14 digits",
            )

            np.testing.assert_almost_equal(
                rec_amps_dict["l9m5"],
                mode_amps_dict["l9m5"],
                14,
                "The l9m5 mode must be recovered" "to 14 digits",
            )

            np.testing.assert_almost_equal(
                rec_amps_dict["l11m-11"],
                mode_amps_dict["l11m-11"],
                14,
                "The l11m-11 mode must be recovered" "to 14 digits",
            )

    def test_ylm_contraction(self):
        """Test the data reconstructed from the
        SH expansion against the original data"""

        info = GLGrid(L=11)
        minfo = method_info(ell_max=11, int_method="GL")

        from spectools.spherical.swsh import Yslm_vec
        from spectools.spherical.swsh import Yslm_vec

        theta_grid, phi_grid = info.meshgrid

        Yl0m0 = Yslm_vec(
            spin_weight=0,
            ell=0,
            emm=0,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl1mm1 = Yslm_vec(
            spin_weight=0,
            ell=1,
            emm=-1,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl2m2 = Yslm_vec(
            spin_weight=0,
            ell=2,
            emm=2,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl7m6 = Yslm_vec(
            spin_weight=0,
            ell=7,
            emm=6,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl9m5 = Yslm_vec(
            spin_weight=0,
            ell=9,
            emm=5,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl11mm11 = Yslm_vec(
            spin_weight=0,
            ell=11,
            emm=-11,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )

        mode_amps = np.random.randn(6)

        mode_amps_dict = {
            "l0m0": mode_amps[0],
            "l1m-1": mode_amps[1],
            "l2m2": mode_amps[2],
            "l7m6": mode_amps[3],
            "l9m5": mode_amps[4],
            "l11m-11": mode_amps[5],
        }

        message("Mode amplitides assumed", mode_amps_dict, message_verbosity=2)

        test_data = (
            mode_amps_dict["l0m0"] * Yl0m0
            + mode_amps_dict["l1m-1"] * Yl1mm1
            + mode_amps_dict["l2m2"] * Yl2m2
            + mode_amps_dict["l7m6"] * Yl7m6
            + mode_amps_dict["l9m5"] * Yl9m5
            + mode_amps_dict["l11m-11"] * Yl11mm11
        )

        test_modes = SHExpand(func=test_data, method_info=minfo, info=info)

        test_data_back = test_modes.evaluate_angular()

        # Test contracted modes
        np.testing.assert_array_almost_equal(
            test_data_back,
            test_data,
            12,
            "The contracted modes must equal"
            " the original input data"
            " upto 12 decimals",
        )

    def test_ylm_reexpand(self):
        """Test the re expansion of the SH modes
        after reconstruction from expanded modes
        against the original mode amplitudes used
        to construct the test data
        (expansion - reconstruction - expansion)"""

        info = GLGrid(L=11)
        minfo = method_info(ell_max=11, int_method="GL")

        from spectools.spherical.swsh import Yslm_vec
        from spectools.spherical.swsh import Yslm_vec

        theta_grid, phi_grid = info.meshgrid

        Yl0m0 = Yslm_vec(
            spin_weight=0,
            ell=0,
            emm=0,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl1mm1 = Yslm_vec(
            spin_weight=0,
            ell=1,
            emm=-1,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl2m2 = Yslm_vec(
            spin_weight=0,
            ell=2,
            emm=2,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl7m6 = Yslm_vec(
            spin_weight=0,
            ell=7,
            emm=6,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl9m5 = Yslm_vec(
            spin_weight=0,
            ell=9,
            emm=5,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        Yl11mm11 = Yslm_vec(
            spin_weight=0,
            ell=11,
            emm=-11,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )

        mode_amps = np.random.randn(6)

        mode_amps_dict = {
            "l0m0": mode_amps[0],
            "l1m-1": mode_amps[1],
            "l2m2": mode_amps[2],
            "l7m6": mode_amps[3],
            "l9m5": mode_amps[4],
            "l11m-11": mode_amps[5],
        }

        message("Mode amplitides assumed", mode_amps_dict, message_verbosity=2)

        test_data = (
            mode_amps_dict["l0m0"] * Yl0m0
            + mode_amps_dict["l1m-1"] * Yl1mm1
            + mode_amps_dict["l2m2"] * Yl2m2
            + mode_amps_dict["l7m6"] * Yl7m6
            + mode_amps_dict["l9m5"] * Yl9m5
            + mode_amps_dict["l11m-11"] * Yl11mm11
        )

        test_modes = SHExpand(func=test_data, method_info=minfo, info=info)

        test_data_back = test_modes.evaluate_angular()

        # Test re expansion
        # Re-expand contracted modes
        test_modes_back = SHExpand(
            func=test_data_back, method_info=minfo, info=info
        )

        test_modes.compare_modes(test_modes_back, prec=12)

    def test_ylm_vs_scipy(self):
        """Test the SH modes upto a given L
        against the scipy package for computing
        SH modes.

        Notes
        -----
        It was observed that the
        scipy package is accurate
        upto 9 digits at :math:`\\ell\\sim 24`

        """

        info = GLGrid(L=24)
        minfo = method_info(ell_max=24, int_method="GL")

        from spectools.spherical.swsh import Yslm_vec
        from spectools.spherical.swsh import Yslm_vec

        theta_grid, phi_grid = info.meshgrid

        #from scipy.special import sph_harm_y as sph_harm
        from scipy.special import sph_harm

        for ell in range(info.L + 1):
            for emm in range(-ell, ell + 1):
                Ylm_this_module = Yslm_vec(
                    spin_weight=0,
                    ell=ell,
                    emm=emm,
                    theta_grid=theta_grid,
                    phi_grid=phi_grid,
                )

                Ylm_scipy = sph_harm(emm, ell, phi_grid, theta_grid)

                # Note: scipy is not accurate. Any disagreement
                # and low precision
                # is due to short comings of scipy's sph_harm

                np.testing.assert_array_almost_equal(
                    Ylm_this_module,
                    Ylm_scipy,
                    9,
                    "The spherical harmonic"
                    f" modes (l{ell}, m{emm}"
                    " from this module"
                    " and scipy must equal"
                    " upto 9 decimals",
                )

    def test_ylm_vs_exact(self):
        """Test the SH modes computed using
        the fast method against the exact
        symbolic computation"""

        # info = GLGrid(L=24)
        # minfo = method_info(ell_max=24, int_method='GL')

        ell_max = 8  # Warning: Very expensive. Tested upto 84

        from spectools.spherical.swsh import Yslm_vec, Yslm_prec
        from spectools.spherical.swsh import Yslm_vec, Yslm_prec

        # theta_grid, phi_grid = info.meshgrid

        theta = np.pi / 21
        phi = np.pi / 6

        for ell in range(ell_max + 1):
            for emm in range(ell, ell + 1):
                message(f"Testing l{ell} m{emm}", message_verbosity=2)

                Ylm_this_module = Yslm_vec(
                    spin_weight=0,
                    ell=ell,
                    emm=emm,
                    theta_grid=theta,
                    phi_grid=phi,
                )

                Ylm_exact = Yslm_prec(
                    spin_weight=0,
                    ell=ell,
                    emm=emm,
                    theta=theta,
                    phi=phi,
                    prec=14,
                )

                np.testing.assert_array_almost_equal(
                    Ylm_this_module,
                    np.complex128(Ylm_exact),
                    12,
                    "The spherical harmonic"
                    "modes from this module"
                    "must agree with exact"
                    "upto 12 digits",
                )

    def test_ylm_vs_exact_grid(self):
        """Test the SH modes computed using
        the fast method against the exact
        symbolic computation"""

        info = GLGrid(L=24)
        # minfo = method_info(ell_max=24, int_method='GL')

        ell_max = 8  # Warning: very expensive. Tested upto 84

        from spectools.spherical.swsh import Yslm_vec, Yslm_prec_grid
        from spectools.spherical.swsh import Yslm_vec, Yslm_prec_grid

        theta_grid, phi_grid = info.meshgrid

        # theta = np.pi/21
        # phi = np.pi/6

        for ell in range(ell_max + 1):
            for emm in range(ell, ell + 1):
                message(f"Testing l{ell} m{emm}", message_verbosity=2)

                Ylm_this_module = Yslm_vec(
                    spin_weight=0,
                    ell=ell,
                    emm=emm,
                    theta_grid=theta_grid,
                    phi_grid=phi_grid,
                )

                Ylm_exact = Yslm_prec_grid(
                    spin_weight=0,
                    ell=ell,
                    emm=emm,
                    theta_grid=theta_grid,
                    phi_grid=phi_grid,
                    prec=14,
                )

                np.testing.assert_array_almost_equal(
                    Ylm_this_module,
                    np.complex128(Ylm_exact),
                    14,
                    "The spherical harmonic"
                    "modes from this module"
                    "must agree with exact"
                    "upto 14 digits",
                )

    def test_ylm_vs_spherical(self):
        """Test the vectorized SH mode
        computation against the spherical
        package modes

        Notes
        -----
        It was observed that the
        spherical package is accurate
        upto 13 digits at :math:`\\l`=100
        """

        def get_index(ell, emm):
            ind = 0
            for ell_ind in range(ell + 1):
                ind += 2 * ell_ind + 1

            return ind + emm - ell_ind - 1

        # info = GLGrid(L=24)
        # minfo = method_info(ell_max=24, int_method='GL')

        ell_max = 24

        from spectools.spherical.swsh import Yslm_vec
        from spectools.spherical.swsh import Yslm_vec

        # theta_grid, phi_grid = info.meshgrid

        theta = np.pi / 21
        phi = np.pi / 6
        spin_weight = 0

        import quaternionic, spherical

        R = quaternionic.array.from_spherical_coordinates(theta, phi)
        # ell_max = ell
        wigner = spherical.Wigner(ell_max)
        Y2 = wigner.sYlm(spin_weight, R)

        for ell in range(ell_max + 1):
            for emm in range(ell, ell + 1):
                message(f"Testing l{ell} m{emm}", message_verbosity=2)

                Ylm_this_module = Yslm_vec(
                    spin_weight=0,
                    ell=ell,
                    emm=emm,
                    theta_grid=theta,
                    phi_grid=phi,
                )

                Ylm_spherical = Y2[get_index(ell, emm)]

                np.testing.assert_array_almost_equal(
                    Ylm_this_module,
                    Ylm_spherical,
                    15,
                    "The spherical harmonic"
                    f" l{ell}m{emm} mode"
                    " from the spherical module"
                    " must agree with this"
                    " module upto 14 digits",
                )

    def test_ylm_vs_spherical_grid(self):
        """Test the vectorized SH mode
        computation against the spherical
        package modes

        Notes
        -----
        It was observed that the
        spherical package is accurate
        upto 13 digits at :math:`\\l`=100
        """

        def get_index(ell, emm):
            ind = 0
            for ell_ind in range(ell + 1):
                ind += 2 * ell_ind + 1

            return ind + emm - ell_ind - 1

        info = GLGrid(L=24)
        # minfo = method_info(ell_max=24, int_method='GL')

        ell_max = 24

        from spectools.spherical.swsh import Yslm_vec
        from spectools.spherical.swsh import Yslm_vec

        theta_grid, phi_grid = info.meshgrid

        # theta = np.pi/21
        # phi = np.pi/6
        spin_weight = 0

        import quaternionic, spherical

        R = quaternionic.array.from_spherical_coordinates(theta_grid, phi_grid)
        # ell_max = ell
        wigner = spherical.Wigner(ell_max)
        Y2 = wigner.sYlm(spin_weight, R)

        for ell in range(ell_max + 1):
            for emm in range(ell, ell + 1):
                message(f"Testing l{ell} m{emm}", message_verbosity=2)

                Ylm_this_module = Yslm_vec(
                    spin_weight=0,
                    ell=ell,
                    emm=emm,
                    theta_grid=theta_grid,
                    phi_grid=phi_grid,
                )

                Ylm_spherical = Y2.T[get_index(ell, emm)].T

                np.testing.assert_array_almost_equal(
                    Ylm_this_module,
                    Ylm_spherical,
                    14,
                    "The spherical harmonic"
                    f" l{ell}m{emm} mode"
                    " from the spherical module"
                    " must agree with this"
                    " module upto 14 digits",
                )

    @pytest.mark.skip(reason="This tests spherical code. Takes long.")
    def test_spherical_vs_exact_grid(self):
        """Test the vectorized SH mode
        computation against the spherical
        package modes

        Notes
        -----
        It was observed that the
        spherical package is accurate
        upto 13 digits at :math:`\\l`=100
        """

        def get_index(ell, emm):
            ind = 0
            for ell_ind in range(ell + 1):
                ind += 2 * ell_ind + 1

            return ind + emm - ell_ind - 1

        info = GLGrid(L=24)
        # minfo = method_info(ell_max=24, int_method='GL')

        ell_max = 6  # Warning: very expensive. Tested upto 84

        from spectools.spherical.swsh import Yslm_prec_grid
        from spectools.spherical.swsh import Yslm_prec_grid

        theta_grid, phi_grid = info.meshgrid

        # theta = np.pi/21
        # phi = np.pi/6
        spin_weight = 0

        import quaternionic, spherical

        R = quaternionic.array.from_spherical_coordinates(theta_grid, phi_grid)
        # ell_max = ell
        wigner = spherical.Wigner(ell_max)
        Y2 = wigner.sYlm(spin_weight, R)

        for ell in range(ell_max + 1):
            for emm in range(ell, ell + 1):
                message(f"Testing l{ell} m{emm}", message_verbosity=2)

                Ylm_exact = Yslm_prec_grid(
                    spin_weight=0,
                    ell=ell,
                    emm=emm,
                    theta_grid=theta_grid,
                    phi_grid=phi_grid,
                    prec=16,
                )

                Ylm_spherical = Y2.T[get_index(ell, emm)].T

                np.testing.assert_array_almost_equal(
                    Ylm_spherical,
                    Ylm_exact,
                    14,
                    "The spherical harmonic"
                    f" l{ell}m{emm} mode"
                    " from the spherical module"
                    " must agree with this"
                    " module upto 14 digits",
                )


# if __name__ == '__main__':

#    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)
