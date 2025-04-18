from waveformtools.models.waveform_models import WaveformModel
#import bilby
import lalsimulation
from lalsimulation import SimInspiralChooseTDWaveform, SimInspiralGetApproximantFromString
from lalsimulation import SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion, SimInspiralWaveformParamsInsertPhenomXPrecVersion
from lal import MSUN_SI, PC_SI, CreateDict


class LALWaveformModel(WaveformModel):
    
    def __init__(self, 
                 PhenomXHMReleaseVersion=122022, 
                 PhenomXPrecVersion=320, 
                 parameters_dict={}):

        #print("Init")
        super().__init__(parameters_dict)

        #print("Local init")
        self.parameters_dict['lal_approximant'] = SimInspiralGetApproximantFromString(self.approximant)
        self.parameters_dict['PhenomXHMReleaseVersion'] = PhenomXHMReleaseVersion
        self.parameters_dict['PhenomXPrecVersion'] = PhenomXPrecVersion
                    
        #print("A")
        self.parameters_dict['longAscNodes']=0
        self.parameters_dict['eccentricity'] = 0
        self.parameters_dict['meanPerAno'] = 0
        self.set_parameters()

        if self.approximant == "IMRPhenomXPHM":
            self.add_waveform_generation_arguments_to_lal_dict()
        if self.parameters_dict["approximant"] == "NR_hdf5":
            lalsimulation.SimInspiralWaveformParamsInsertNumRelData(
                self.parameters_dict['lal_dict'], 
                str(self.parameters_dict['lvcnr_file_path']))
            self.set_spins_from_NR_data()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['lal_dict']
        del self.parameters_dict['lal_dict']
        return state

    def __setstate__(self, state):

        self.__dict__.update(state)
        self.parameters_dict['lal_dict'] = CreateDict()
        self.add_waveform_generation_arguments_to_lal_dict()
        self.set_parameters()

    def add_waveform_generation_arguments_to_lal_dict(self):

        if self.PhenomXHMReleaseVersion is not None:
            SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(self.lal_dict, self.PhenomXHMReleaseVersion)
        if self.PhenomXPrecVersion is not None:
            SimInspiralWaveformParamsInsertPhenomXPrecVersion(self.lal_dict, self.PhenomXPrecVersion)

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
        if self.approximant == "NR_hdf5":
            return -hp.data.data, hc.data.data
        else:
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
        
        if self.approximant == "NR_hdf5":
            return -hp.data.data, hc.data.data
        else:
            return hp.data.data, hc.data.data
    

    def update_parameters(self, parameters_dict):
        
        self.parameters_dict.update(parameters_dict)
        #self.parameters_dict['phi_ref'] = self.parameters_dict['coa_phase']
        self.set_parameters()
        
    def set_spins_from_NR_data(self):
        Mtotal_NR = self.parameters_dict["mass1"] + self.parameters_dict["mass2"]
        spins = lalsimulation.SimInspiralNRWaveformGetSpinsFromHDF5File(
            self.f_ref, 
            Mtotal_NR, 
            str(self.lvcnr_file_path))
        
        spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = spins

        self.parameters_dict.update({'spin1x' : spin1x,
                                     'spin1y' : spin1y,
                                     'spin1z' : spin1z,
                                     'spin2x' : spin2x,
                                     'spin2y' : spin2y,
                                     'spin2z' : spin2z,
                                    })
        
        self.set_parameters()
        return self.parameters_dict