from waveformtools.models.waveform_models import WaveformModel
#import bilby
import lalsimulation
from lalsimulation import SimInspiralChooseTDWaveform, SimInspiralGetApproximantFromString, SimInspiralChooseTDModes, SimInspiralChooseFDModes
from lalsimulation import SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion, SimInspiralWaveformParamsInsertPhenomXPrecVersion
from lal import MSUN_SI, MTSUN_SI, PC_SI, CreateDict
from pycbc.waveform import td_approximants, fd_approximants
from waveformtools.waveformtools import load_lal_modes_to_modes_array, get_starting_angular_frequency
from waveformtools.models.eob import EOBWaveformModel
from scipy.interpolate import InterpolatedUnivariateSpline

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
        self.parameters_dict['longAscNodes']= 0
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

        if 'ell_max' not in self.parameters_dict:
            self.parameters_dict.update({"ell_max" : 4})

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
    
    def get_approximant_type(self, apx):

        if apx in td_approximants():
            apx_domain = 'td'
        elif apx in fd_approximants():
            apx_domain = 'fd'
        else:
            raise KeyError(f"Unknown apx type {apx}")
        
        return apx_domain


    def get_td_waveform_modes(self, dimensionless=True, **parameters_dict):
        ''' Return the waveform modes object as a 
        `waveformtools.waveform_modes.WaveformModes` object.
         
        Tapering conventions: default lal 
        '''
        print(parameters_dict)

        self.update_parameters(parameters_dict)
        apx_domain = self.get_approximant_type(self.approximant)
        print(apx_domain)

        if apx_domain == 'td':

            try:
                waveform_modes_list = SimInspiralChooseTDModes(             
                                                                            parameters_dict['phi_ref'],
                                                                            parameters_dict['delta_t'],
                                                                            parameters_dict['mass1']*MSUN_SI,
                                                                            parameters_dict['mass2']*MSUN_SI,
                                                                            parameters_dict['spin1x'],
                                                                            parameters_dict['spin1y'],
                                                                            parameters_dict['spin1z'],
                                                                            parameters_dict['spin2x'],
                                                                            parameters_dict['spin2y'],
                                                                            parameters_dict['spin2z'],
                                                                            parameters_dict['f_lower'],
                                                                            parameters_dict['f_ref'],
                                                                            parameters_dict['distance']*1e6*PC_SI,
                                                                            parameters_dict['lal_dict'],
                                                                            parameters_dict['ell_max'],
                                                                            parameters_dict['lal_approximant']
                                                            )
            except Exception as ex:
                print(ex)
                apx_domain='fd'

        if apx_domain == 'fd':
            waveform_modes_list = SimInspiralChooseFDModes( 
                                                                        parameters_dict['mass1']*MSUN_SI,
                                                                        parameters_dict['mass2']*MSUN_SI,
                                                                        parameters_dict['spin1x'],
                                                                        parameters_dict['spin1y'],
                                                                        parameters_dict['spin1z'],
                                                                        parameters_dict['spin2x'],
                                                                        parameters_dict['spin2y'],
                                                                        parameters_dict['spin2z'],
                                                                        parameters_dict['delta_f'],
                                                                        parameters_dict['f_lower'],
                                                                        parameters_dict['f_max'],
                                                                        parameters_dict['f_ref'],
                                                                        parameters_dict['phi_ref'],
                                                                        parameters_dict['distance']*1e6*PC_SI,
                                                                        parameters_dict['inclination'],
                                                                        parameters_dict['lal_dict'],
                                                                        parameters_dict['lal_approximant']
                                                            )

        #else:
        #    raise KeyError(f"Unknown apx domain {apx_domain}")
        

        wfm = load_lal_modes_to_modes_array(waveform_modes_list, domain=apx_domain)

        if 'fd' in wfm.label:
            wfm_td = wfm.to_time_basis()
        elif 'td' in wfm.label:
            wfm_td = wfm
        else:
            raise KeyError("The modes array is not correctly representing the lal modes.")
        
        if dimensionless:
            Mtotal = (parameters_dict["mass1"] + parameters_dict["mass2"])
            wfm_td.time_axis = wfm_td.time_axis/(Mtotal*MTSUN_SI)
            wfm_td.modes_data = wfm_td.modes_data*parameters_dict['distance']*PC_SI*1e6/(Mtotal*MSUN_SI)

        return wfm_td


    def get_corresponding_eob_hamiltonian(self, waveform_modes, **parameters_dict):
        ''' Get the corresponding EoB Hamiltonian at the starting frequency of the 
        waveform '''

        mass1 = parameters_dict['mass1']
        mass2 = parameters_dict['mass2']

        mu = mass1*mass2/(mass1 + mass2)
        #omega_init = get_starting_angular_frequency(waveform_modes.mode(2,2).real, waveform_modes.delta_t)

        eob_parameters_dict = parameters_dict.copy()

        omega0_dimless = parameters_dict['omega0']*(mass1+mass2)*MTSUN_SI
        
        eob_parameters_dict.update({"omega0" : omega0_dimless})
        eob_generator = EOBWaveformModel(eob_parameters_dict)
        eob_generator.compute_model()

        eob_waveform_modes = eob_generator.get_td_waveform_modes()
        E0 = eob_generator.model[0, 5]*mu

        #eob_omega = eob_generator.model[:, 6]

        #eob_omega_interp = InterpolatedUnivariateSpline(eob_waveform_modes.time_axis, 
        #                                                eob_omega)
        
        #eob_fine_time_axis = np.arange(eob_waveform_modes.time_axis[0], 
        #                               eob_waveform_modes.time_axis[-1], 1e-3)
        
        #eob_omega_fine = eob_omega_interp(eob_fine_time_axis)

        #targ = np.argmin(eob_omega_fine-omega0_dimless

        return E0







