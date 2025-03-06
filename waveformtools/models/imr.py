class IMRWaveformModel(WaveformModel):
     
    def __init__(self, mass_ratio, chi_1, chi_2, omega0):
        super().__init__(mass_ratio, chi_1, chi_2, omega0, approximant)



    def compute_model(self):
        self.EOBModel = EOBWaveformModel(self.mass_ratio, self.chi_1, self.chi_2, self.omega0, self.approximant)
        self.model = self.EOBModel.get_model()

        self.waveform_modes = self.compute_waveform_modes()


    def compute_waveform_modes(self):