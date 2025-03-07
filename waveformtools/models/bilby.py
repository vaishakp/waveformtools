class BilbyWaveformGenerator(WaveformModel):

    def __int__(self, 
                duration=4,
                *args, 
                **kwargs):

        super.__init__(*args, **kwargs)

        self.parameters_dict.update({'duration' : duration})
        self.sampling_frequency = int(1/self.delta_t)


        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                                    duration = duration,
                                    sampling_frequency=self.sampling_frequency,
                                    #frequency_domain_source_model=bilby.gw.source.gwsignal_binary_black_hole
                                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                    # parameter_conversion=bilby.gw.conversion.generate_all_bbh_parameters,
                                    waveform_arguments={    
                                                            'minimum_frequency' : self.f_low,
                                                            'reference_frequency': self.f_ref,
                                                            'waveform_approximant': self.approximant,
                                                            'PhenomXHMReleaseVersion': self.PhenomXHMReleaseVersion, 
                                                            'PhenomXPrecVersion': self.PhenomXPrecVersion, # IMRPhenomXPHMSpin-Taylor
                                                            'catch_waveform_errors' : True
                                                        }
                                    )
        
        bilby_parameter_mapping = { 
                                    'distance' : 'luminosity_distance',
                                    'f_low' : 'minimum_frequency',
                                    'phi_ref': 'phase',
                                    'f_ref' : 'reference_frequency',
        }

        

    def get_bilby_parameters_dict(self):
        pass

    def get_polarizations(self):

        return self.waveform_generator.frequency_domain_strain()



    def get_td_waveform(self):
        pass