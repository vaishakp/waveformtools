from lal import CreateDict
import numpy as np
from pycbc.detector import Detector
from waveformtools.waveformtools import find_maxloc_and_time

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
                                        'sampling_frequency',
                                        'delta_t',
                                        'delta_f',
                                        'omega0',
                                        'approximant',
                                        'PhenomXHMReleaseVersion',
                                        'PhenomXPrecVersion',
                                        'debug',
                                        'lvcnr_file_path'
                                    ]

        self.parameters_dict['model']=None
        self.parameters_dict['lal_dict'] = CreateDict()
        self.set_parameters()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['lal_dict']
        del self.parameters_dict['lal_dict']
        return state

    def __setstate__(self, state):
        
        self.__dict__.update(state)
        self.parameters_dict['lal_dict'] = CreateDict()
        self.set_parameters()

    #@property
    #def phi_ref(self):
    #    return self.coa_phase
    
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

    def project_polarizations(self, 
                         hp, 
                         hc,
                         extrinsic_parameters, 
                         detector_string):
        ''' t_coal is defined here to be the location of the peak of the 
        polarization '''
        
        ifo = Detector(detector_string)
        amp = np.absolute(hp+1j*hc)
        times = np.arange(0, len(hp)*self.delta_t, self.delta_t)
        t_maxloc, _, _ = find_maxloc_and_time(times, amp)
        #maxloc = np.argmax()
        #pmax = len(hp) - maxloc
        #tmax = times[maxloc]
        det_times = (times - t_maxloc)
        det_times += extrinsic_parameters['t_coal']
        Fp, Fc = ifo.antenna_pattern(extrinsic_parameters['ra'], 
                                     extrinsic_parameters['dec'], 
                                     extrinsic_parameters['psi'], 
                                     det_times)
        h_inj = Fp*hp + Fc*hc

        return det_times, h_inj







        



