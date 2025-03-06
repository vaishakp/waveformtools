from abc import ABC

from waveformtools.models.utils import get_eob_modes_array


class WaveformModel(ABC):

    def __init__(self,
                 mass_1,
                 mass_2,
                 spin1x,
                 spin1y,
                 spin1z,
                 spin2x,
                 spin2y,
                 spin2z,
                 phi_ref,
                 inclination,
                 distance,
                 delta_t=None,
                 delta_f=None,
                 *args,
                 f_low=20,
                 omega0=None,
                 debug=True,
                 approximant='SEOBNRv5PHM',
                 **kwargs,
                 ):
        

        if omega0 is None:
            self.omega0 = f_low/2
        
        self.delta_t = delta_t
        self.delta_f = delta_f

        if self.delta_t is None:
            if self.delta_f is None:
                raise KeyError("Please provide delta_t or delta_f")
            else:
                self.domain='frequency'
        else:
            self.domain = 'time'


        self.parameters_dict = { 'mass_1' : mass_1,
                                'mass_2' : mass_2,
                                'spin1x' : spin1x,
                                'spin1y' : spin1y,
                                'spin1z' : spin1z,
                                'spin2x' : spin2x,
                                'spin2y' : spin2y,
                                'spin2z' : spin2z,
                                'phi_ref' : phi_ref,
                                'inclination' : inclination,
                                'distance' : distance,
                                'f_low' : f_low,

        }

        self.debug=debug
        self.approximant = approximant


    def get_td_waveform_modes(self):
        raise NotImplementedError
    
    def get_fd_waveform_modes(self):
        raise NotImplementedError
    
    def get_td_waveform(self):
        raise NotImplementedError
    
    def get_fd_waveform(self):
        raise NotImplementedError
    
    def get_model(self):

        if self.model is not None:
            self.compute_model()
            
        return self.model

    def compute_model(self):
        raise NotImplementedError
    







        



