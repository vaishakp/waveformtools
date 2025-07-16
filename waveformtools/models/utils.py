import numpy as np
from waveformtools.modes_array import ModesArray
from spectools.spherical.grids import GLGrid


def get_modes_array_from_eob_modes_dict(time_axis, modes_dict, L):

    ell_max =4
    Grid = GLGrid(L=L)
    wf_modes = ModesArray(ell_max=ell_max+1,
                           time_axis=time_axis,
                           spin_weight=-2,
                           Grid=Grid)
    
    wf_modes.create_modes_array()

    for ell in range(2, ell_max+1):
        for emm in range(-ell, ell+1):
            wf_modes.set_mode_data(ell=ell, emm=emm, data=modes_dict[f'{ell},{emm}'])


    return wf_modes