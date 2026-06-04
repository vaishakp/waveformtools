from waveformtools.models.waveform_models import WaveformModel
#import bilby
import numpy as np
import lalsimulation
from lalsimulation import SimInspiralChooseTDWaveform, SimInspiralFD, SimInspiralGetApproximantFromString, SimInspiralChooseTDModes, SimInspiralChooseFDModes
from lalsimulation import SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion, SimInspiralWaveformParamsInsertPhenomXPrecVersion
from lal import MSUN_SI, MTSUN_SI, PC_SI, CreateDict, G_SI, C_SI
from pycbc.waveform import td_approximants, fd_approximants
from waveformtools.waveformtools import load_lal_modes_to_modes_array, get_starting_angular_frequency, message
from waveformtools.models.eob import EOBWaveformModel
from waveformtools.fd_to_td import (
    lal_fd_modes_to_td_modes,
    prepare_physical_td_window,
)
from scipy.interpolate import InterpolatedUnivariateSpline


# Explicit overrides for approximants whose preferred generation domain is known
# from current waveformtools usage. Unknown approximants are resolved against the
# PyCBC/LAL registries in ``get_approximant_domain`` rather than silently being
# treated as time-domain models.
APPROXIMANT_DOMAINS = {
    "NRSur7dq4": "td",
    "SEOBNRv5PHM": "td",
    "IMRPhenomXPHM": "fd",
}


class LALWaveformModel(WaveformModel):
    
    def __init__(self, 
                 PhenomXHMReleaseVersion=122022, 
                 PhenomXPrecVersion=320, 
                 parameters_dict=None):

        if parameters_dict is None:
            parameters_dict = {}
        else:
            parameters_dict = dict(parameters_dict)

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
            self.set_parameters()


    def __getstate__(self):
        state = self.__dict__.copy()
        if 'lal_dict' in state.keys():
            del state['lal_dict']
        if 'lal_dict' in self.parameters_dict.keys():
            del self.parameters_dict['lal_dict']
        return state

    def __setstate__(self, state):

        self.__dict__.update(state)
        self.parameters_dict['lal_dict'] = CreateDict()
        self.add_waveform_generation_arguments_to_lal_dict()
        self.set_parameters()

    def add_waveform_generation_arguments_to_lal_dict(self):

        if self.parameters_dict["approximant"]=="IMRPhenomXPHM":
            if self.PhenomXHMReleaseVersion is not None:
                SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(self.lal_dict, self.PhenomXHMReleaseVersion)
            if self.PhenomXPrecVersion is not None:
                SimInspiralWaveformParamsInsertPhenomXPrecVersion(self.lal_dict, self.PhenomXPrecVersion)

    def capabilities(self):
        """Return the output capabilities advertised by this backend."""
        return {
            "td_modes": True,
            "fd_modes": True,
            "fd_modes_as_td": True,
            "fd_modes_as_td_physical_window": True,
            "td_polarizations": True,
            "fd_polarizations": True,
            "td_projection": True,
            "fd_projection": False,
            "nr_hdf5": True,
        }

    def get_td_waveform(self, **parameters_dict):

        self.update_parameters(parameters_dict)

        hp, hc = SimInspiralChooseTDWaveform(   
                                                self.mass1*MSUN_SI,
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

    def get_fd_waveform(self, approximant=None, **parameters_dict):

        self.update_parameters(parameters_dict)

        hp, hc = SimInspiralFD(   
                                self.mass1*MSUN_SI,
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
                                self.delta_f,
                                self.f_lower,
                                self.f_max,
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
    
    def get_approximant_domain(self, apx=None):
        """Return the preferred generation domain for an approximant.

        The explicit ``APPROXIMANT_DOMAINS`` registry records choices needed by
        waveformtools. For any other approximant, fall back to PyCBC's LAL-backed
        TD/FD registries. Unknown approximants raise instead of silently being
        treated as time-domain models.
        """

        if apx is None:
            apx = self.approximant

        if apx in APPROXIMANT_DOMAINS:
            apx_domain = APPROXIMANT_DOMAINS[apx]
        elif apx in fd_approximants():
            apx_domain = 'fd'
        elif apx in td_approximants():
            apx_domain = 'td'
        else:
            raise KeyError(f"Unknown approximant/domain for {apx}")
        
        message(f"Apx type {apx_domain}", message_verbosity=2)
        return apx_domain

    def get_approximant_type_auto(self, apx):
        return self.get_approximant_domain(apx)

    def get_approximant_type(self, apx):
        return self.get_approximant_domain(apx)

    def _choose_td_modes(self):
        return SimInspiralChooseTDModes(             
                                                    self.phi_ref,
                                                    self.delta_t,
                                                    self.mass1*MSUN_SI,
                                                    self.mass2*MSUN_SI,
                                                    self.spin1x,
                                                    self.spin1y,
                                                    self.spin1z,
                                                    self.spin2x,
                                                    self.spin2y,
                                                    self.spin2z,
                                                    self.f_lower,
                                                    self.f_ref,
                                                    self.distance*1e6*PC_SI,
                                                    self.lal_dict,
                                                    self.ell_max,
                                                    self.lal_approximant
                                                )

    def _choose_fd_modes(self):
        return SimInspiralChooseFDModes( 
                                                    self.mass1*MSUN_SI,
                                                    self.mass2*MSUN_SI,
                                                    self.spin1x,
                                                    self.spin1y,
                                                    self.spin1z,
                                                    self.spin2x,
                                                    self.spin2y,
                                                    self.spin2z,
                                                    self.delta_f,
                                                    self.f_lower,
                                                    self.f_max,
                                                    self.f_ref,
                                                    self.phi_ref,
                                                    self.distance*1e6*PC_SI,
                                                    self.inclination,
                                                    self.lal_dict,
                                                    self.lal_approximant
                                                )

    def _load_lal_modes(self, waveform_modes_list, domain):
        return load_lal_modes_to_modes_array(lal_modes=waveform_modes_list, 
                                             domain=domain,
                                             Mtotal=self.Mtotal)

    def get_td_waveform_modes(self, dimensionless=True, **parameters_dict):
        ''' Return the waveform modes object as a 
        `waveformtools.waveform_modes.WaveformModes` object.
         
        Tapering conventions: default lal 
        '''
        self.update_parameters(parameters_dict)
        apx_domain = self.get_approximant_domain(self.approximant)

        if apx_domain == 'fd':
            return self.get_fd_waveform_modes_as_td(dimensionless=dimensionless)

        try:
            waveform_modes_list = self._choose_td_modes()
            wfm_td = self._load_lal_modes(waveform_modes_list, domain='td')
        except Exception as ex:
            message(
                "TD mode generation failed; trying FD modes and converting to TD. "
                f"Original exception: {ex}",
                message_verbosity=1,
            )
            return self.get_fd_waveform_modes_as_td(dimensionless=dimensionless)

        if self.approximant == 'NRSur7dq4':
            _, maxtime = wfm_td.find_max_intensity_loc()
            wfm_td._modes_data/=1
            wfm_td._time_axis -= maxtime
        
        if dimensionless:
            wfm_td = self.non_dimensionalize_td_waveform_modes(wfm_td, **self.parameters_dict)

        return wfm_td
    

    def get_fd_waveform_modes(self, dimensionless=True, **parameters_dict):
        ''' Return the FD waveform modes object as a 
        `waveformtools.waveform_modes.WaveformModes` object.
         
        Tapering conventions: default lal 
        '''
        self.update_parameters(parameters_dict)
        waveform_modes_list = self._choose_fd_modes()

        wfm_fd = self._load_lal_modes(waveform_modes_list, domain='fd')
        return wfm_fd

    def get_fd_waveform_modes_as_td(self, dimensionless=True, undo_warp=True, **parameters_dict):
        """Generate FD modes and return them in the current TD convention.

        This is the explicit version of the existing FD-approximant path in
        ``get_td_waveform_modes``: generate modes with
        ``SimInspiralChooseFDModes``, load them as an FD ``ModesArray``, convert
        to the time basis, and optionally non-dimensionalize the resulting TD
        modes.

        Parameters
        ----------
        undo_warp : bool, optional
            Preserve the historical ``ModesArray.to_time_basis()`` behavior by
            applying ``ModesArray.undo_warp()`` after the inverse FFT. Set this
            to ``False`` when you want the raw circular FFT buffer and will
            perform explicit physical windowing with
            ``get_fd_waveform_modes_as_td_physical_window``.
        """

        self.update_parameters(parameters_dict)
        wfm_fd = self.get_fd_waveform_modes(dimensionless=False)
        wfm_td = lal_fd_modes_to_td_modes(wfm_fd, undo_warp=undo_warp)

        if dimensionless:
            wfm_td = self.non_dimensionalize_td_waveform_modes(wfm_td, **self.parameters_dict)

        return wfm_td

    def get_fd_waveform_modes_as_td_physical_window(
        self,
        dimensionless=True,
        t_min=None,
        t_max=None,
        peak_target_frac=0.5,
        taper_width=None,
        taper_frac=None,
        taper_sides='both',
        set_peak_time_to_zero=True,
        **parameters_dict,
    ):
        """Generate FD modes and return a centered physical TD window.

        This path is intended for analyses, such as balance-law integrals, where
        wrapped circular-FFT content should not be interpreted as a physical
        post-merger tail. It differs from the legacy path by inverse-FFTing the
        FD modes without the heuristic ``undo_warp`` roll, applying one common
        shift to all modes so the total-intensity peak is at the requested target
        index, assigning a fresh time axis, optionally cropping, and optionally
        tapering the crop edges.

        If ``dimensionless=True`` then ``t_min``, ``t_max``, and ``taper_width``
        are interpreted in geometric units ``t/M``. If ``dimensionless=False``
        they are interpreted in seconds.
        """

        self.update_parameters(parameters_dict)
        wfm_fd = self.get_fd_waveform_modes(dimensionless=False)
        wfm_td = lal_fd_modes_to_td_modes(wfm_fd, undo_warp=False)

        if dimensionless:
            wfm_td = self.non_dimensionalize_td_waveform_modes(wfm_td, **self.parameters_dict)

        wfm_td = prepare_physical_td_window(
            wfm_td,
            t_min=t_min,
            t_max=t_max,
            peak_target_frac=peak_target_frac,
            taper_width=taper_width,
            taper_frac=taper_frac,
            taper_sides=taper_sides,
            set_peak_time_to_zero=set_peak_time_to_zero,
        )

        return wfm_td

    # Explicit public aliases. These make the output type unambiguous while
    # preserving the historical method names above.
    def get_td_modes(self, **parameters_dict):
        return self.get_td_waveform_modes(**parameters_dict)

    def get_fd_modes(self, **parameters_dict):
        return self.get_fd_waveform_modes(**parameters_dict)

    def get_fd_modes_as_td(self, **parameters_dict):
        return self.get_fd_waveform_modes_as_td(**parameters_dict)

    def get_fd_modes_as_td_physical_window(self, **parameters_dict):
        return self.get_fd_waveform_modes_as_td_physical_window(**parameters_dict)

    def get_td_polarizations(self, **parameters_dict):
        return self.get_td_waveform(**parameters_dict)

    def get_fd_polarizations(self, **parameters_dict):
        return self.get_fd_waveform(**parameters_dict)

    def project_td_polarizations(self, hp, hc, extrinsic_parameters, detector_string):
        return self.project_polarizations(hp, hc, extrinsic_parameters, detector_string)
