import numpy as np
from waveformtools.modes_array import ModesArray



def get_modes_array_from_eob_modes_dict(time_axis, modes_dict):

    ell_max =4

    wf_modes = ModesArray(ell_max=ell_max+1,
                           time_axis=time_axis,
                           spin_weight=-2)
    
    wf_modes.create_modes_array()

    for ell in range(2, ell_max+1):
        for emm in range(-ell, ell+1):
            wf_modes.set_mode_data(ell=ell, emm=emm, data=modes_dict[f'{ell},{emm}'])


    return wf_modes