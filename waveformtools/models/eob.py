import numpy as np
from waveformtools.models.waveform_models import WaveformModel

class EOBWaveformModel(WaveformModel):
        
    def __init__(self, parameters_dict=None, deviation_dict=None, *args, **kwargs):
        ''' The deviation dict is to be passes as a dict of dicts, following 
        https://waveforms.docs.ligo.org/software/pyseobnr/source/notebooks/pseob_example.html

        These can be the dicts domega_dict, dtau_dict, dA_dict, dw_dict, da6, ddSO,
        each a dict with fractional deciations in specific modes.

        '''
        if parameters_dict is None:
            parameters_dict = {}
        else:
            parameters_dict = dict(parameters_dict)

        if deviation_dict is None:
            deviation_dict = {}
        else:
            deviation_dict = dict(deviation_dict)

        super().__init__(parameters_dict=parameters_dict, *args, **kwargs)

        self.chi_1 = np.array([self.parameters_dict['spin1x'], self.parameters_dict['spin1y'], self.parameters_dict['spin1z']])
        self.chi_2 = np.array([self.parameters_dict['spin2x'], self.parameters_dict['spin2y'], self.parameters_dict['spin2z']])
        self.settings = deviation_dict
        
        # Greater than 1, with m1 > m2
        self.mass_ratio = self.parameters_dict['mass1']/self.parameters_dict['mass2']
        self.delta_t_dimless = self.parameters_dict["delta_t"]/(self.Mtotal * self.MTSUN_SI)
        self.settings.update({"dt" : self.parameters_dict["delta_t"]})
        self.td_waveform_modes = None

    def capabilities(self):
        """Return the output capabilities advertised by this backend."""
        return {
            "td_modes": True,
            "fd_modes": False,
            "fd_modes_as_td": False,
            "td_polarizations": "partial",
            "fd_polarizations": False,
            "td_projection": True,
            "fd_projection": False,
            "nr_hdf5": False,
        }

    def compute_model(self, L=28, **parameters_dict):
        from pyseobnr.generate_waveform import generate_modes_opt
        from waveformtools.models.utils import get_modes_array_from_eob_modes_dict
        
        self.update_parameters(parameters_dict)
        self.time_axis, self.modes_dict, self.model = generate_modes_opt(self.mass_ratio,
                                                                    self.chi_1,
                                                                    self.chi_2,
                                                                    self.dimless_omega0,
                                                                    self.dimless_omega_ref,
                                                                    debug=True,
                                                                    approximant=self.approximant,
                                                                    settings=self.settings,
                                                                    )
        
        self.td_waveform_modes = get_modes_array_from_eob_modes_dict(self.time_axis, self.modes_dict, L=L)


    def get_td_waveform_modes(self, 
                              dimensionless=True, 
                              L=29,
                              **parameters_dict
                              ):

        self.update_parameters(parameters_dict)
        
        #if self.td_waveform_modes is None:
        self.compute_model(L, **parameters_dict)

        wfm_td = self.td_waveform_modes

        if not dimensionless:
            wfm_td = self.dimensionalize_td_waveform_modes(
                wfm_td,
                **self.parameters_dict,
            )
            
        return self._standardize_generated_modes(
            wfm_td,
            domain='td',
            dimensionless=dimensionless,
            generator='EOBWaveformModel.get_td_waveform_modes',
        )

    def get_td_modes(self, **parameters_dict):
        return self.get_td_waveform_modes(**parameters_dict)
    
    def get_td_waveform(self, **kwargs):
        from pycbc.waveform import get_td_waveform as pycbc_get_td_waveform

        parameters = self.parameters_dict.copy()
        parameters.update(kwargs)
        parameters.update({'coa_phase' : parameters['phi_ref']})
        self.compute_model()

        #return self.td_waveform_modes.to_td_waveform(**parameters)
        return pycbc_get_td_waveform(**parameters)
