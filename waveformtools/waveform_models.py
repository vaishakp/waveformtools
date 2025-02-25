from abc import ABC

from waveformtools.waveform_models_utils import get_eob_modes_array


class WaveformModel(ABC):

    def __init__(self,
                 mass_ratio,
                 chi_1,
                 chi_2,
                 omega0,
                 debug=True,
                 approximant='SEOBNRv5PHM',
                 ):
        

        self.mass_ratio = mass_ratio
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.omega0 = omega0
        self.debug=debug
        self.approximant = approximant



    def get_waveform_modes():

        if self.waveform_modes is not None:
            self.compute_model()

        return self.waveform_modes
    
    def get_model():

        if self.model is not None:
            self.compute_model()
            
        return self.model
    

    def compute_model(self):
        raise NotImplementedError
    



class EOBWaveformModel(WaveformModel):
        
    def __init__(self, mass_ratio, chi_1, chi_2, omega0):
            
            super().__init__(mass_ratio, chi_1, chi_2, omega0, approximant)


    def compute_model(self):
        self.time_axis, modes_dict, self.model = generate_modes_opt(self.mass_ratio,
                                            self.chi_1,
                                            self.chi_2,
                                            self.omega0,
                                            debug=self.debug,
                                            approximant=self.approximant)
        
        self.waveform_modes = get_modes_array_from_eob_modes_dict(self.time_axis, modes_dict)


class IMRWaveformModel(WaveformModel):
     
    def __init__(self, mass_ratio, chi_1, chi_2, omega0):
        super().__init__(mass_ratio, chi_1, chi_2, omega0, approximant)



    def compute_model(self):
        self.EOBModel = EOBWaveformModel(self.mass_ratio, self.chi_1, self.chi_2, self.omega0m self.approximant)
        self.model = self.EOBModel.get_model()

        self.waveform_modes = self.compute_waveform_modes()


    def compute_waveform_modes(self):
        