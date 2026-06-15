from lal import CreateDict
import numpy as np
from pathlib import Path
from lal import MSUN_SI, MTSUN_SI, PC_SI, G_SI, C_SI


def _message(*args, **kwargs):
    from waveformtools.waveformtools import message

    return message(*args, **kwargs)


class WaveformModel:

    def __init__(self,
                 parameters_dict,
                 ):
        
        # print("Super init")
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
                                        'f_max',
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
        self._ensure_omega0_parameter()
        self.set_parameters()
        self.MSUN_SI = MSUN_SI
        self.MTSUN_SI = MTSUN_SI
        self.G_SI = G_SI
        self.PC_SI = PC_SI
        self.C_SI = C_SI


    def __getstate__(self):
        state = self.__dict__.copy()
        del state['lal_dict']
        del self.parameters_dict['lal_dict']
        return state

    def __setstate__(self, state):
        
        self.__dict__.update(state)
        self.parameters_dict['lal_dict'] = CreateDict()
        self._ensure_omega0_parameter()
        self.set_parameters()

    def _ensure_omega0_parameter(self, force_from_f_lower=False):
        if "omega0" in self.parameters_dict and not force_from_f_lower:
            return
        if "f_lower" not in self.parameters_dict:
            raise KeyError("Please supply omega0 or f_lower")
        self.parameters_dict.update({"omega0": np.pi * self.parameters_dict["f_lower"]})
    
    def set_parameters(self):

        for key in self.parameters_dict.keys():
            #try:
            setattr(self, key, self.parameters_dict[key])
            #except KeyError:
            #    setattr(self, key, None)
    
    def update_parameters(self, parameters_dict):
        
        force_omega0_from_f_lower = (
            "f_lower" in parameters_dict and "omega0" not in parameters_dict
        )
        self.parameters_dict.update(parameters_dict)
        self._ensure_omega0_parameter(force_from_f_lower=force_omega0_from_f_lower)
        #self.parameters_dict['phi_ref'] = self.parameters_dict['coa_phase']
        self.set_parameters()

    def _standardize_generated_modes(
        self,
        wfm,
        *,
        domain=None,
        dimensionless=None,
        generator=None,
    ):
        from waveformtools.comparison.conventions import (
            standardize_generated_modes_in_place,
        )
        from waveformtools.comparison.metadata import (
            WaveformMetadata,
            attach_comparison_metadata,
            get_comparison_metadata,
        )

        existing_metadata = get_comparison_metadata(wfm).to_dict()
        safe_parameters = self._metadata_safe_parameters()
        safe_parameters.update(
            {
                "generation_domain": domain,
                "dimensionless_output": dimensionless,
            }
        )
        existing_parameters = existing_metadata.get("parameters") or {}
        metadata_parameters = {**existing_parameters, **safe_parameters}
        metadata_dict = {
            **existing_metadata,
            "approximant": self.parameters_dict.get("approximant"),
            "parameters": metadata_parameters,
            "mass1": self.parameters_dict.get("mass1"),
            "mass2": self.parameters_dict.get("mass2"),
            "total_mass": self.Mtotal,
            "mass_ratio": self._safe_mass_ratio(),
            "spin1": self._spin_tuple_from_parameters("spin1"),
            "spin2": self._spin_tuple_from_parameters("spin2"),
            "f_ref": self.parameters_dict.get("f_ref"),
            "f_lower": self.parameters_dict.get("f_lower"),
            "mode_output_frame": self.parameters_dict.get(
                "mode_output_frame",
            ),
            "reference_time_or_frequency": domain,
            "generator": generator or self.__class__.__name__,
        }
        attach_comparison_metadata(
            wfm,
            WaveformMetadata.from_mapping(metadata_dict),
        )
        _, diagnostics = standardize_generated_modes_in_place(wfm)
        setattr(wfm, "mode_convention_diagnostics", diagnostics)
        return wfm

    def _metadata_safe_parameters(self):
        excluded = {"lal_dict", "model", "eob_generator"}
        return {
            key: self._metadata_safe_value(value)
            for key, value in self.parameters_dict.items()
            if key not in excluded
        }

    def _metadata_safe_value(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [self._metadata_safe_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): self._metadata_safe_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _safe_mass_ratio(self):
        mass1 = self.parameters_dict.get("mass1")
        mass2 = self.parameters_dict.get("mass2")
        if mass1 is None or mass2 in (None, 0):
            return None
        return mass1 / mass2

    def _spin_tuple_from_parameters(self, prefix):
        keys = (f"{prefix}x", f"{prefix}y", f"{prefix}z")
        if not all(key in self.parameters_dict for key in keys):
            return None
        return tuple(float(self.parameters_dict[key]) for key in keys)

    
    @property
    def dimless_omega_ref(self):
        return np.pi*self.f_ref*self.Mtotal*MTSUN_SI
    
    @property
    def Mtotal(self):
        return self.parameters_dict['mass1'] + self.parameters_dict['mass2']
    
    @property
    def dimless_omega0(self):
        return self.omega0*(self.Mtotal*MTSUN_SI)
    
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
        from pycbc.detector import Detector

        ifo = Detector(detector_string)
        amp = np.absolute(hp+1j*hc)
        times = np.arange(0, len(hp)*self.delta_t, self.delta_t)
        from waveformtools.waveformtools import find_maxloc_and_time

        t_maxloc, _, _ = find_maxloc_and_time(times, amp)
        det_times = (times - t_maxloc)
        det_times += extrinsic_parameters['t_coal']
        Fp, Fc = ifo.antenna_pattern(extrinsic_parameters['ra'], 
                                     extrinsic_parameters['dec'], 
                                     extrinsic_parameters['psi'], 
                                     det_times)
        h_inj = Fp*hp + Fc*hc

        return det_times, h_inj

    def non_dimensionalize_td_waveform_modes(self, wfm_td, **parameters_dict):

        Mtotal = parameters_dict["mass1"] + parameters_dict["mass2"]
        wfm_td.time_axis = wfm_td.time_axis/(Mtotal*MTSUN_SI)
        wfm_td._modes_data *= (
            parameters_dict['distance']
            * PC_SI
            * 1e6
            * (C_SI**2)
            / (G_SI * Mtotal * MSUN_SI)
        )
        return wfm_td

    def dimensionalize_td_waveform_modes(self, wfm_td, **parameters_dict):

        Mtotal = parameters_dict["mass1"] + parameters_dict["mass2"]
        wfm_td.time_axis = wfm_td.time_axis*(Mtotal*MTSUN_SI)
        wfm_td._modes_data *= (
            (G_SI * Mtotal * MSUN_SI)
            / (parameters_dict['distance'] * PC_SI * 1e6 * (C_SI**2))
        )
        return wfm_td

    def compute_infinite_time_balance_laws(
        self,
        chunk_size=None,
        grid_L=28,
        **parameters_dict,
    ):
        ''' Compute the infinite time version of the balance laws 
        by fetching the waveform modes and generating an equivalent 
        EoB hamiltonian.

        Parameters
        ----------
        chunk_size : int or None
            If None, use the legacy full-angular-time-array balance-law path.
            If an integer, use the chunked RHS implementation in
            ``waveformtools.chunked_balance_laws``. This reduces peak memory
            during the angular RHS evaluation.
        grid_L : int
            Gauss-Legendre angular resolution.
        '''
        from spectools.spherical.grids import GLGrid
        Grid = GLGrid(L=grid_L)

        wfm = self.get_td_waveform_modes(dimensionless=True, **parameters_dict)
        E0 = self.get_corresponding_eob_hamiltonian(**parameters_dict)
        # message(f"EoB Hamiltonain {E0}", message_verbosity=1)

        news_modes = wfm.get_news_from_strain()
        Erad = wfm.compute_energy_radiated(news_modes=news_modes)
        # Mfinal_rad = E0 - Erad
        # message(f"Energy radiated {Erad}", message_verbosity=1)
        # message(f"Final mass from energy radiated {Mfinal_rad}", message_verbosity=1)
        # Mfinal_eob = self.eob_generator.model.final_mass

        if 'EOB' in parameters_dict['approximant']:
            Mfinal_eob = self.eob_generator.model.final_mass
            Mfinal = Mfinal_eob
            # error = 100*(Mfinal_eob/Mfinal_rad -1)

        else:
            Mfinal_rad = E0 - Erad
            Mfinal = Mfinal_rad

        # message(f"Final mass from EoB {Mfinal_eob}", message_verbosity=1)
        # message(f"%error {error}", message_verbosity=1)

        v_kick = wfm.compute_kick(Mfinal=Mfinal)
        # message(f"Kick velocity {v_kick}", message_verbosity=1)
        # message(f"Length", news_modes.data_len, message_verbosity=1)

        if chunk_size is None:
            violations = wfm.compute_waveform_balance_law(M_adm=E0,
                                             M_final=Mfinal,
                                             v_kick=v_kick,
                                             Grid=Grid,
                                             )
        else:
            from waveformtools.chunked_balance_laws import balance_law_chunked
            violations = balance_law_chunked(
                strain_modes=wfm,
                ginfo=Grid,
                M_adm=E0,
                M_final=Mfinal,
                v_kick=v_kick,
                chunk_size=chunk_size,
            )

        return violations
    
    def compute_infinite_time_balance_laws_debug(
        self,
        chunk_size=None,
        grid_L=28,
        **parameters_dict,
    ):
        ''' Compute the infinite time version of the balance laws 
        by fetching the waveform modes and generating an equivalent 
        EoB hamiltonian '''
        from spectools.spherical.grids import GLGrid
        Grid = GLGrid(L=grid_L)

        
        wfm = self.get_td_waveform_modes(dimensionless=True, **parameters_dict)
        E0 = self.get_corresponding_eob_hamiltonian(**parameters_dict)
        _message(f"EoB Hamiltonain {E0}", message_verbosity=1)

        news_modes = wfm.get_news_from_strain()
        _message(f"Length", news_modes.data_len, message_verbosity=1)
        Erad = wfm.compute_energy_radiated(news_modes=news_modes)
        Mfinal_rad = E0 - Erad
        _message(f"Energy radiated {Erad}", message_verbosity=1)
        _message(f"Final mass from energy radiated {Mfinal_rad}", message_verbosity=1)
        Mfinal_eob = self.eob_generator.model.final_mass

        if 'EOB' in parameters_dict['approximant']:
            Mfinal_eob = self.eob_generator.model.final_mass
            Mfinal = Mfinal_eob
        

        else:
            Mfinal_rad = E0 - Erad
            Mfinal = Mfinal_rad

        error = 100*(Mfinal_eob/Mfinal_rad -1)
        _message(f"Final mass from EoB {Mfinal_eob}", message_verbosity=1)
        _message(f"%error {error}", message_verbosity=1)

        v_kick = wfm.compute_kick(Mfinal=Mfinal)
        _message(f"Kick velocity {v_kick}", message_verbosity=1)
        
        if chunk_size is None:
            violations = wfm.compute_waveform_balance_law_debug(M_adm=E0, 
                                             M_final=Mfinal,
                                             v_kick=v_kick,
                                             Grid=Grid,
                                             debug=True
                                             )
        else:
            from waveformtools.chunked_balance_laws import balance_law_chunked
            violations = balance_law_chunked(
                strain_modes=wfm,
                ginfo=Grid,
                M_adm=E0,
                M_final=Mfinal,
                v_kick=v_kick,
                chunk_size=chunk_size,
                debug=True,
            )

        return violations
    

    def get_corresponding_eob_hamiltonian(self,  L=29, **parameters_dict):
        ''' Get the corresponding EoB Hamiltonian at the starting frequency of the 
        waveform '''

        from waveformtools.models.eob import EOBWaveformModel

        mass1 = parameters_dict['mass1']
        mass2 = parameters_dict['mass2']
        Mtotal = mass1 + mass2
        # message(f"Total mass {Mtotal}", message_verbosity=1)
        m1 = mass1/Mtotal
        m2 = mass2/Mtotal
        mu = m1*m2
        eob_parameters_dict = parameters_dict.copy()

        explicit_omega0 = eob_parameters_dict.get("omega0")
        explicit_f_lower = eob_parameters_dict.get("f_lower")
        has_explicit_omega0 = (
            explicit_omega0 is not None
            and np.isfinite(float(explicit_omega0))
            and float(explicit_omega0) > 0.0
        )
        has_explicit_f_lower = (
            explicit_f_lower is not None
            and np.isfinite(float(explicit_f_lower))
            and float(explicit_f_lower) > 0.0
        )
        if has_explicit_f_lower and not has_explicit_omega0:
            eob_parameters_dict.update({"omega0": np.pi * float(explicit_f_lower)})
            has_explicit_omega0 = True

        if not (has_explicit_omega0 or has_explicit_f_lower) and (
            self.f_lower==0 or self.omega0==0
        ):
            wfm    = self.get_td_waveform_modes(**parameters_dict)
            from waveformtools.waveformtools import get_starting_angular_frequency

            omega0_dimless = get_starting_angular_frequency(wfm.mode(2,2), 
                                                    wfm.delta_t(),
                                                    npoints=250
                                                    )
            omega0 = omega0_dimless/(Mtotal*MTSUN_SI)
            # message(f"omega0_dimless {omega0_dimless}", message_verbosity=1)
            # message(f"omega0 {omega0}", message_verbosity=1)
            eob_parameters_dict.update({"omega0" : omega0})
        
        eob_parameters_dict.update({"approximant" : "SEOBNRv5PHM" })
        eob_generator = EOBWaveformModel(parameters_dict=eob_parameters_dict)
        eob_generator.compute_model(L=L)

        #self.eob_model = eob_generator.model
        self.eob_generator = eob_generator
        E0 = eob_generator.model.dynamics[0, 5]*mu

        return E0
