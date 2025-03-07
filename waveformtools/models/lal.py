from waveformtools.models.waveform_models import WaveformModel
#import bilby
from lalsimulation import SimInspiralChooseTDWaveform, SimInspiralGetApproximantFromString
from lalsimulation import SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion, SimInspiralWaveformParamsInsertPhenomXPrecVersion
from lal import MSUN_SI, PC_SI, CreateDict


class LALWaveformModel(WaveformModel):
    
    def __init__(self, PhenomXHMReleaseVersion=122022, PhenomXPrecVersion=320, parameters_dict={}):

        #print("Init")
        super().__init__(parameters_dict)

        #print("Local init")
        self.parameters_dict['lal_approximant'] = SimInspiralGetApproximantFromString(self.approximant)

        if self.approximant == "IMRPhenomXPHM":
            if PhenomXHMReleaseVersion is not None:
                SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(self.lal_dict, PhenomXHMReleaseVersion)
            if PhenomXPrecVersion is not None:
                SimInspiralWaveformParamsInsertPhenomXPrecVersion(self.lal_dict, 320)

        #print("A")
        self.parameters_dict['longAscNodes']=0
        self.parameters_dict['eccentricity'] = 0
        self.parameters_dict['meanPerAno'] = 0
        self.set_parameters()


    def get_td_waveform(self, **parameters_dict):

        self.update_parameters(parameters_dict)

        hp, hc = SimInspiralChooseTDWaveform(   self.mass1*MSUN_SI,
                                                self.mass2*MSUN_SI,
                                                self.spin1x,
                                                self.spin1y,
                                                self.spin1z,
                                                self.spin2x,
                                                self.spin2y,
                                                self.spin2z,
                                                self.distance*1e6*PC_SI,
                                                self.inclination,
                                                self.phi_ref,
                                                self.longAscNodes,
                                                self.eccentricity,
                                                self.meanPerAno,
                                                self.delta_t,
                                                self.f_lower,
                                                self.f_ref,
                                                self.lal_dict,
                                                self.lal_approximant
                                            )
        
        return hp.data.data, hc.data.data
    

    def get_td_waveform_dict(self, **parameters_dict):

        self.update_parameters(parameters_dict)

        hp, hc = SimInspiralChooseTDWaveform(   parameters_dict['mass1']*MSUN_SI,
                                                parameters_dict['mass2']*MSUN_SI,
                                                parameters_dict['spin1x'],
                                                parameters_dict['spin1y'],
                                                parameters_dict['spin1z'],
                                                parameters_dict['spin2x'],
                                                parameters_dict['spin2y'],
                                                parameters_dict['spin2z'],
                                                parameters_dict['distance']*1e6*PC_SI,
                                                parameters_dict['inclination'],
                                                parameters_dict['phi_ref'],
                                                parameters_dict['longAscNodes'],
                                                parameters_dict['eccentricity'],
                                                parameters_dict['meanPerAno'],
                                                parameters_dict['delta_t'],
                                                parameters_dict['f_lower'],
                                                parameters_dict['f_ref'],
                                                parameters_dict['lal_dict'],
                                                parameters_dict['lal_approximant']
                                            )
        
        return hp.data.data, hc.data.data
    

    def update_parameters(self, parameters_dict):
        

        self.parameters_dict.update(parameters_dict)
        #self.parameters_dict['phi_ref'] = self.parameters_dict['coa_phase']
        self.set_parameters()
        
        
