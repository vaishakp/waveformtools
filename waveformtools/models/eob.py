import numpy as np
from waveformtools.models.waveform_models import WaveformModel
from pyseobnr.generate_waveform import generate_modes_opt, generate_prec_hpc_opt,  GenerateWaveform
from waveformtools.models.utils import get_modes_array_from_eob_modes_dict
from pycbc.waveform import get_td_waveform

class EOBWaveformModel(WaveformModel):
        
    def __init__(self, *args, **kwargs):
            
        super().__init__(*args, **kwargs)

        self.chi_1 = np.array([self.parameters_dict['spin1x'], self.parameters_dict['spin1y'], self.parameters_dict['spin1z']])
        self.chi_2 = np.array([self.parameters_dict['spin2x'], self.parameters_dict['spin2y'], self.parameters_dict['spin2z']])

        # Greater than 1, with m1 > m2
        self.mass_ratio = self.parameters_dict['mass1']/self.parameters_dict['mass2']
        self.td_waveform_modes = None

    def compute_model(self, L):
        self.time_axis, self.modes_dict, self.model = generate_modes_opt(self.mass_ratio,
                                                                    self.chi_1,
                                                                    self.chi_2,
                                                                    self.dimless_omega0,
                                                                    debug=True,
                                                                    approximant=self.approximant)
        
        self.td_waveform_modes = get_modes_array_from_eob_modes_dict(self.time_axis, self.modes_dict, L=L)


    def get_td_waveform_modes(self, dimensionless=True, L=29, **parameters_dict):

        if self.td_waveform_modes is None:
            self.compute_model(L)

        wfm_td = self.td_waveform_modes

        if not dimensionless:
            wfm_td = self.dimensionalize_td_waveform_modes(wfm_td, **parameters_dict)
            
        return wfm_td
    
    def get_td_waveform(self, **kwargs):

        parameters = self.parameters_dict.copy()
        parameters.update(kwargs)
        self.compute_model()

        #return self.td_waveform_modes.to_td_waveform(**parameters)
        return get_td_waveform(**parameters)
