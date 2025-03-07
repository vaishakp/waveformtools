from lal import CreateDict


class WaveformModel:

    def __init__(self,
                 parameters_dict,
                 ):
        
        print("Super init")
        #if self.delta_t is None:
        #    if self.delta_f is None:
        #        raise KeyError("Please provide delta_t or delta_f")
        #    else:
        #        self.domain='frequency'
        #else:
        #    self.domain = 'time'

        self.parameters_dict = parameters_dict
        self.parameters_dict_keys = [
                                        'mass_1',
                                        'mass_2',
                                        'spin1x',
                                        'spin1y',
                                        'spin1z',
                                        'spin2x',
                                        'spin2y',
                                        'spin2z',
                                        'phi_ref',
                                        'inclination',
                                        'distance',
                                        'f_lower',
                                        'f_ref',
                                        #'sampling_frequency',
                                        'delta_t',
                                        #'delta_f',
                                        'omega0',
                                        'approximant',
                                        'PhenomXHMReleaseVersion',
                                        'PhenomXPrecVersion',
                                        'debug',

                                    ]

        self.parameters_dict['model']=None
        self.parameters_dict['lal_dict'] = CreateDict()
        self.set_parameters()

    @property
    def phi_ref(self):
        return self.coa_phase
    
    def set_parameters(self):

        for key in self.parameters_dict.keys():
            #try:
            setattr(self, key, self.parameters_dict[key])
            #except KeyError:
            #    setattr(self, key, None)
            

    def get_td_waveform_modes(self):
        raise NotImplementedError
    
    def get_fd_waveform_modes(self):
        raise NotImplementedError
    
    def get_td_waveform(self, **parameters_dict):
        raise NotImplementedError
    
    def get_fd_waveform(self):
        raise NotImplementedError
    
    def get_model(self):

        if self.model is not None:
            self.compute_model()
            
        return self.model

    def compute_model(self):
        raise NotImplementedError
    







        



