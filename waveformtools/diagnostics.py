""" Diagnostic tools """

import numpy as np


class MethodInfo:
    """The methods for integration ,differential to be passed on
    for operations."""

    def __init__(
        self,
        int_method="MP",
        diff_method="SH",
        ell_max=8,
        eval_ell_max=8,
        degree=8,
        reg=True,
        swsh_routine="waveformtools",
        atol=1e-4,
        rtol=1e-2,
        ode_solver="solve_ivp",
        ivp_method="RK45",
        vectorized=True,
        error_info=False,
        spin_weight=None,
    ):

        self.int_method = int_method
        self.diff_method = diff_method
        self.ell_max = ell_max
        self.eval_ell_max = eval_ell_max
        self.reg = reg
        self.degree = degree
        self.swsh_routine = swsh_routine
        self.atol = atol
        self.rtol = rtol
        self.ode_solver = ode_solver
        self.vectorized = vectorized
        self.ivp_method = ivp_method
        self.error_info = error_info
        self.spin_weight = spin_weight


def RMSerrs(func1, func2, info):
    """Compute and return the RMS error between two arrays

    Parameters
    ----------
    func1, func2 : ndarray
                   Arrays of same shape to compare with.
    info : sphericalarray
           Grid info

    Returns
    -------
    RMS : float
          The RMS error
    Amax : float
           The max diff
    Amin : float the min diff
    """
    diff = func1 - func2

    Amax = np.amax(diff)
    Amin = np.amin(diff)

    RMS = np.sqrt(np.sum(np.absolute(diff) ** 2, axis=(-2, -1)) / info.npix_act)

    return RMS, Amin, Amax


def IsModesEqual(modes1, modes2, modes_list):
    """Check if the mode amplitudes are equal

    Parameters
    ----------
    modes1, modes2: dict
                    A dictionary of modes.
                    For each mode, the data could be an array.

    modes_list: list
                A list of list of mode numbers
    Returns
    -------
    modes_err: float
               The error of each mode as a dict
    """

    modes_err = {}

    for ell, emm_list in modes_list:
        for emm in emm_list:
            A1 = modes1[f"l{ell}"][f"m{emm}"]
            A2 = modes2[f"l{ell}"][f"m{emm}"]

            delta = A1 - A2

            label = f"l{ell}m{emm}"

            modes_err.update({label: delta})

    return modes_err
