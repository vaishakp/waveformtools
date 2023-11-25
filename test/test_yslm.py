import numpy as np
import unittest

from waveformtools.transforms import Yslm_vec
from waveformtools.grids import GLGrid

info = GLGrid()
theta_grid, phi_grid = info.meshgrid


fun1_ana_110 = np.sqrt(3 / (8 * np.pi)) * np.sin(theta_grid)
fun1_ana_11p1 = (
    -np.sqrt(3 / (16 * np.pi))
    * (1 - np.cos(theta_grid))
    * np.exp(1j * phi_grid)
)
fun1_ana_11m1 = (
    -np.sqrt(3 / (16 * np.pi))
    * (1 + np.cos(theta_grid))
    * np.exp(-1j * phi_grid)
)


class TestYslm(unittest.TestCase):
    def test_s1l1m0(self):
        """Test :math:`Y_{slm}` for :math:`s=1, \\ell=1, m=0`"""

        spin_weight = 1

        ell = 1

        emm1 = 0

        fun1_110 = Yslm_vec(
            spin_weight=spin_weight,
            ell=ell,
            emm=emm1,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )

        np.testing.assert_almost_equal(
            fun1_110,
            fun1_ana_110,
            15,
            "Yslm must agree upto 15 decimals for" "s=1, l=1, m=0",
        )

    def test_s1l1mp1(self):
        """Test :math:`Y_{slm}` for :math:`s=1, \\ell=1, m=1`"""

        spin_weight = 1

        ell = 1

        emm2 = 1

        fun1_11p1 = Yslm_vec(
            spin_weight=spin_weight,
            ell=ell,
            emm=emm2,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )

        np.testing.assert_almost_equal(
            fun1_11p1,
            fun1_ana_11p1,
            15,
            "Yslm must agree upto 15 decimals for" "s=1, l=1, m=1",
        )

    def test_s1l1mm1(self):
        """Test :math:`Y_{slm}` for :math:`s=1, \\ell=1, m=-1`"""

        spin_weight = 1

        ell = 1

        emm3 = -1

        fun1_11m1 = Yslm_vec(
            spin_weight=spin_weight,
            ell=ell,
            emm=emm3,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )

        np.testing.assert_almost_equal(
            fun1_11m1,
            fun1_ana_11m1,
            15,
            "Yslm must agree upto 15 decimals for" "s=1, l=1, m=-1",
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False, verbosity=2)
