from copy import deepcopy
import sys

import h5py
import numpy as np
import math

from sympy import comp

from waveformtools import dataIO
from waveformtools.dataIO import (
    construct_mode_list,
    get_iteration_numbers_from_keys,
    sort_keys,
)
from spectools.spherical.grids import UniformGrid
from spectools.spherical.grids import UniformGrid
from spectools.spherical.swsh import Yslm_vec
from spectools.spherical.swsh import Yslm_vec
from waveformtools.waveformtools import interp_resam_wfs, message, get_nr_frame_angles_from_lal
from pathlib import Path
from waveformtools.integrate import fixed_frequency_integrator
from waveformtools.waveformtools import (
    get_starting_angular_frequency as sang_f,
)

from waveformtools.single_mode import SingleMode
from scipy.interpolate import InterpolatedUnivariateSpline
from waveformtools.BMS import compute_linear_momentum_contribution_from_news, compute_impulse_from_force, compute_angular_momentum

class ModesArray:
    """A class that holds mode array of waveforms

    This can handle two index and three index modes.

    Attributes
    ----------
    label: str
            The label of the modes array.
    r_ext: float
            The extraction radius.
    modes_list: list
                 The list of available modes
                 in the format [l1, [m values], [l2, [m values], ...]]
    ell_max: int
              The maximum :math:`\\ell` mode available.
    modes_data: 3d array
                 The three dimensional array
                 containing modes in time/frequency
                 space. The axis of the array is
                 (:math:`\\ell`, :math:`m`, time/freq).
    base_dir: str
               The base directory containing the
               modes data.
    data_dir: str
               The subdirectory in which to look
               for the data.
    filename: str
               The filename containg the modes data.

    Methods
    -------
    get_metadata:
                   Get the metadata associated with the ModesArray.
    mode:
           Get the data for the given :math:`\\ell, m` mode.
    create_modes_array:
                         A private method to create an
                         empty `ModesArray` of given shape.
    delta_t:
              Set the attribute `delta_t` and/ or return its value.
    load_modes:
                Load the waveform modes from a specified h5 file.
    save_modes:
                Save the waveform modes to a specified h5 file.
    set_mode_data:
                   Set the `mode` data of specified modes.
    to_frequency_basis:
                        Get the `ModesArray` in frequency basis
                        from its time basis representation.
    to_time_basis:
                   Get the `ModesArray` in temporal basis
                   from its frequency basis representation.
    extrap_to_inf:
                   Extrapolate the modes to infinity.
    supertranslate:
                    Supertranslate the waveform modes.
    boost:
           Boost the waveform modes.
    """

    def __init__(
        self,
        data_dir=None,
        file_name=None,
        extra_mode_axes_shape=None,
        modes_data=None,
        time_axis=None,
        frequency_axis=None,
        key_format=None,
        ell_max=None,
        data_len=None,
        modes_list=None,
        label=None,
        r_ext=np.inf,
        out_file_name=None,
        maxtime=None,
        date=None,
        time=None,
        key_ex=None,
        spin_weight=-2,
        actions="empty",
        areal_radii=[],
        Grid=None,
    ):
        self.label = label
        self.data_dir = data_dir
        self.file_name = file_name
        self.modes_data = modes_data
        self.key_format = key_format
        self.ell_max = ell_max
        self.modes_list = modes_list
        self.r_ext = r_ext
        self.time_axis = time_axis
        self.frequency_axis = frequency_axis
        self.out_file_name = out_file_name
        self.maxtime = maxtime
        self.date = date
        self.time = time
        self.key_ex = key_ex
        self._data_len = data_len
        self._spin_weight = spin_weight
        self._actions = actions
        self._extra_mode_axes_shape = extra_mode_axes_shape
        self._areal_radii = areal_radii
        self._Grid = Grid
        if (np.array(self.extra_mode_axes_shape) == np.array(None)).all():
            self.extra_mode_axes = False
        else:
            self.extra_mode_axes = True

    #def __getstate__(self):

    #    if self.data_dir is not None:
    #        self.data_dir = str(self.data_dir)

    #    return self
    
    #def __setstate__(self, other):

    #    self.__dict__.update(other)
    #    self.data_dir = Path(self.data_dir)

    @property
    def extra_mode_axes_shape(self):
        return self._extra_mode_axes_shape

    @property
    def actions(self):
        return self._actions

    @property
    def spin_weight(self):
        return self._spin_weight

    @property
    def Grid(self):
        return self._Grid

    @property
    def n_modes(self):
        return (self.ell_max+1)**2 - self.spin_weight**2
    
    def deepcopy(self):
        return deepcopy(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.conjugate:
            cma = self.deepcopy()
            cma._modes_data = np.conjugate(self._modes_data)
            cma._spin_weight=int(np.sign(cma.spin_weight)*cma.spin_weight)
            return cma
        
        return NotImplemented
    
    def __add__(self, obj):

        obj2 = self.deepcopy() 
        if isinstance(obj, self.__class__):
            obj2._modes_data = self.modes_data + obj.modes_data
        else:
            obj2._modes_data = self.modes_data + obj

        return obj2

    def __radd__(self, obj):

        obj2 = self.deepcopy()
        if isinstance(obj, self.__class__):
            obj2._modes_data = self.modes_data + obj.modes_data
        else:
            obj2._modes_data = self.modes_data + obj

        return obj2

    def __sub__(self, obj):

        obj2 = self.deepcopy()
        if isinstance(obj, self.__class__):
            obj2._modes_data = self.modes_data - obj.modes_data
        else:
            obj2._modes_data = self.modes_data - obj

        return obj2

    def __rsub__(self, obj):

        obj2 = self.deepcopy() 
        if isinstance(obj, self.__class__):
            obj2._modes_data = obj.modes_data - obj.modes_data
        else:
            obj2._modes_data = obj - self.modes_data

        return obj2
    
    def __mul__(self, obj):

        obj2 = self.deepcopy()
        if isinstance(obj, self.__class__):
            obj2._modes_data = self.modes_data * obj.modes_data
            obj2._spin_weight = self.spin_weight + obj.spin_weight
        else:
            obj2._modes_data = self.modes_data*obj

        return obj2
    
    def __rmul__(self, obj):

        obj2 = self.deepcopy() 
        if isinstance(obj, self.__class__):
            obj2._modes_data = self.modes_data * obj.modes_data
            obj2._spin_weight = self.spin_weight + obj.spin_weight
        else:
            obj2._modes_data = obj*self.modes_data

        return obj2
    
    def __truediv__(self, obj):

        obj2 = self.deepcopy() 
        if isinstance(obj, self.__class__):
            obj2._modes_data = self.modes_data / obj.modes_data
            obj2._spin_weight = self.spin_weight - obj.spin_weight
        else:
            obj2._modes_data= self.modes_data /obj
        return obj2
    
    def __rtruediv__(self, obj):

        obj2 = self.deepcopy() 
        if isinstance(obj, self.__class__):
            obj2._modes_data = obj.modes_data/self.modes_data
            obj2._spin_weight = obj.spin_weight - self.spin_weight
        else:
            obj2._modes_data = obj/self.modes_data

        return obj2
    
    def __getitem__(self, index):

        aslice = self.deepcopy()
        aslice._modes_data = self.modes_data[..., index]
        aslice._time_axis = np.array([self.time_axis[index]])

        return aslice
    
    def __len__(self):
        if not (np.array(self.time_axis) == np.array(None)).any():
            return len(self.time_axis)
        elif not (np.array(self.frequency_axis) == np.array(None)).any():
            return len(self.frequency_axis)
        else:
            return False
    
    def get_metadata(self):
        """Get the metadata associated with the instance.

        Returns
        -------
        metadata: dict
                   A dictionary of metedata.
        """
        # The metadata dict
        unnecessary_keys = ["_time_axis", "_modes_data", "freq_axis"]

        # Get all attributes
        # metadata = self.__dict__
        metadata = {}

        for key, val in self.__dict__.items():
            if key in unnecessary_keys:
                pass
            else:
                metadata.update({key: val})

        if metadata['data_dir'] is not None:
             metadata['data_dir'] = str(metadata['data_dir'])

        self.ell_max = int(self.ell_max)

        # for item in unnecessary_keys:
        # metadata.pop(item, None)

        # self.metadata = metadata

        return metadata

    def mode(self, ell, emm, extra_indices=None):
        """Return the time series of a particular mode.

        Parameters
        ----------
        ell: int
              The :math:`\\ell` index of the mode.
        emm: int
              The :math:`m` index of the mode.
        r_index: int, optional
                  The third mode axis index, if any

        Returns
        -------
        mode_data: array
                    The array of the requested mode.
        """
        vec_index = ell**2 + emm + ell

        ang_mode_data = self.modes_data[vec_index]

        if self.extra_mode_axes:
            if not (np.array(extra_indices) == np.array(None)).all():
                message(f"Extra indices supplied {extra_indices}...")
                return ang_mode_data[*extra_indices]

        return ang_mode_data

    @property
    def time_axis(self):
        """The time axis"""
        return np.array(self._time_axis)

    @time_axis.setter
    def time_axis(self, time_axis):
        self._time_axis = time_axis

    @property
    def modes_data(self):
        """The modes array"""
        return np.array(self._modes_data)

    @modes_data.setter
    def modes_data(self, modes_data):
        self._modes_data = modes_data

    def create_time_axis(self, data_len):
        message(f"Created time axis of length {data_len}", message_verbosity=3)
        self._time_axis = np.zeros(data_len)

    def create_modes_array(self, ell_max=None, data_len=None):
        """Create a modes array and initialize it with zeros.

        Parameters
        ----------
        ell_max: int
                  The maximum :math:`\\ell` value of the modes.
        data_len: int
                   The number of points along
                   the third (time / frequency) axis.

        Returns
        -------
        self.modes_data: modes_data
                           sets the `self.ModesArray` attribute.
        """
        import datetime
        import time

        # Check ell_max
        if ell_max is None:
            try:
                ell_max = self.ell_max
            except Exception as ex:
                message(ex)
                raise NameError("Please supply ell_max")

        if data_len is None:
            try:
                data_len = self.data_len
            except Exception as ex:
                message(ex)
                raise NameError("Please supply data_len")

        if (np.array(self.time_axis) == np.array(None)).all():
            self.create_time_axis(data_len)

        if self.modes_list is None:
            self.modes_list = construct_mode_list(
                ell_max=ell_max, spin_weight=self.spin_weight
            )

        if self.extra_mode_axes:
            message(
                "Creating modes array"
                " with extra mode axis"
                f" of shape {self.extra_mode_axes_shape}",
                message_verbosity=3,
            )

            self._modes_data = np.zeros(
                (
                    (ell_max + 1) ** 2,
                    *self.extra_mode_axes_shape,
                    data_len,
                ),
                dtype=np.complex128,
            )

        else:
            message("Creating modes array", message_verbosity=3)

            self._modes_data = np.zeros(
                (
                    (ell_max + 1) ** 2,
                    data_len,
                ),
                dtype=np.complex128,
            )

        # Set the time metadata
        time_now = time.localtime()
        time_now = time.strftime("%H:%M:%S", time_now)

        date_now = str(datetime.date.today())

        if self.time is None:
            # Assign time and date stamp if it doesnt exist
            self.time = time_now
            self.date = date_now

    @property
    def data_len(self):
        """Returns the length of the time/frequency axis.

        Returns
        -------
        data_len: int
                   The length of the time/frequency axis.
        """

        #if self._data_len is None:
        #    try:
        #        data_len = len(self.time_axis)

        #    except Exception as ex:
        #        message(ex)
        #        data_len = len(self.frequency_axis)

        #    self._data_len = data_len
        if self._data_len is not None:
            return self._data_len
        else:
            return len(self)

    #@data_len.setter
    #def data_len(self, data_len):
    #    self._data_len = data_len

    def delta_t(self, value=None):
        """Sets and returns the value of time stepping :math:`dt`.

        Parameters
        ----------
        value: float, optional
                The value of :math:`dt`
                to set to the attribute.

        Returns
        -------
        self.delta_t: float
                       Sets the attribute.
        """

        if not value:
            try:
                delta_t = self.time_axis[1] - self.time_axis[0]
            except Exception as ex:
                message(
                    "Please input the value of `delta_t`"
                    "or supply the `time_axis` to the waveform.",
                    ex,
                )
        else:
            delta_t = value

        return delta_t

    @property
    def delta_f(self, value=None):
        """Sets and returns the value of frequency stepping :math:`df`.

        Parameters
        ----------
        value: float, optional
                The value of :math:`df`
                to set to the attribute.

        Returns
        -------
        delta_f: float
                  Sets the attribute.
        """

        # if not self.delta_t:
        if not value:
            try:
                delta_f = self.frequency_axis[1] - self.frequency_axis[0]
            except Exception as ex:
                message(
                    "Please input the value of `delta_f`"
                    "or supply the `frequency_axis` to the waveform.",
                    ex,
                )
        else:
            delta_f = value

        return delta_f

    def load_modes(
        self,
        label=None,
        data_dir=None,
        file_name=None,
        ftype="generic",
        var_type="Psi4",
        resam_type="finest",
        interp_kind="cubic",
        extrap_order=4,
        r_ext=None,
        ell_max=None,
        pre_key=None,
        modes_list=None,
        crop=False,
        centre=True,
        key_ex=None,
        save_as_ma=False,
        compression_opts=None,
        r_ext_factor=1,
        debug=False,
    ):
        """Load the waveform mode data from an hdf file.

        Parameters
        ----------
        extrap_order: int, optional
                       For SpEC waveforms only.
                       This is the order of extrapolation to use.

        pre_key: str, optional
                  A string containing the key of the group in
                  the HDF file in which the modes` dataset exists.
                  It defaults to `None`.
        mode_numbers: list
                       The mode numbers to load from the file.
                       Each item in the list is a list that
                       contains two integrer numbers, one for
                       the mode index :math:`\\ell` and the
                       other for the mode index :math:`m`.
        crop: bool
               Whether or not to crop the beginning of the input
               waveform. If yes, the first :math:`t=r_{ext}`
               length of data will be discarded.

        Returns
        -------
        waveform_obj: 3d array
                       Sets the three dimensional array
                       `waveform.modes` that contains
                       the required :math:`\\ell, m` modes.

        Examples
        --------
        # Note
        # Update this!
        #>>> from waveformtools.waveforms import waveform
        #>>> waveform.set_basedir('./')
        #>>> waveform.set_datadir('data/')
        #>>> mode_numbers = [[2, 2], [3, 3]]
        #>>> waveform.load_data(mode_numbers=mode_numbers)
        """

        if debug is False:
            wfs_nl = 1

        if not data_dir:
            data_dir = self.data_dir
        else:
            self.data_dir = data_dir

        data_dir = Path(data_dir)

        if not file_name:
            file_name = self.file_name
        else:
            self.file_name = file_name

        if not ell_max:
            ell_max = self.ell_max
        else:
            self.ell_max = ell_max

        if not label:
            label = self.label

        # if self.data_dir is not None:
        # data_dir = self.data_dir

        # if self.file_name is not None:
        # file_name = self.file_name
        message(f"Passing {data_dir}/{file_name}", message_verbosity=3)

        if ftype == "generic":
            dataIO.load_gen_data_from_disk(
                wfa=self,
                label=label,
                data_dir=data_dir,
                file_name=file_name,
                r_ext=r_ext,
                ell_max=ell_max,
                pre_key=pre_key,
                modes_list=modes_list,
                crop=crop,
                centre=centre,
                key_ex=key_ex,
                r_ext_factor=r_ext_factor,
            )

        elif (ftype) == "RIT" or (ftype == "GT"):
            if var_type == "Psi4":
                data_file_path = data_dir / file_name

                dataIO.load_RIT_Psi4_data_from_disk(
                    wfa=self,
                    data_file_path=data_file_path,
                    resam_type=resam_type,
                    interp_kind=interp_kind,
                    ell_max=ell_max,
                    modes_list=modes_list,
                    crop=crop,
                    centre=centre,
                )

            elif var_type == "Strain":
                # message(file_name)
                dataIO.load_RIT_Strain_data_from_disk(
                    self,
                    data_dir=data_dir,
                    file_name=file_name,
                    label=label,
                    resam_type=resam_type,
                    interp_kind=interp_kind,
                    ell_max=ell_max,
                    save_as_ma=save_as_ma,
                    modes_list=modes_list,
                    crop=crop,
                    centre=centre,
                    r_ext_factor=r_ext_factor,
                    debug=debug,
                )
            else:
                message(f"Data {ftype} {var_type} not supported yet!")
                sys.exit(0)

        elif ftype == "SpEC":
            wfs_nl = dataIO.load_SpEC_data_from_disk(
                self,
                label=label,
                data_dir=data_dir,
                file_name=file_name,
                extrap_order=extrap_order,
                r_ext=r_ext,
                ell_max=ell_max,
                centre=centre,
                modes_list=modes_list,
                save_as_ma=save_as_ma,
                resam_type=resam_type,
                interp_kind=interp_kind,
                compression_opts=compression_opts,
                r_ext_factor=r_ext_factor,
                debug=debug,
            )

        elif ftype == "SpEC_raw":
            wfs_nl = dataIO.load_SpEC_non_extrap_data_from_disk(
                self,
                label,
                data_dir,
                file_name,
                r_ext,
                ell_max,
                centre,
                modes_list,
                save_as_ma,
                resam_type,
                interp_kind,
                compression_opts,
                r_ext_factor,
                debug,
            )

        elif ftype == "SpECTRE":
            dataIO.load_SpECTRE_data_from_disk(
                self,
                label=label,
                data_dir=data_dir,
                file_name=file_name,
                r_ext=r_ext,
                ell_max=ell_max,
                centre=centre,
                modes_list=modes_list,
                save_as_ma=save_as_ma,
                resam_type=resam_type,
                kind=interp_kind,
                compression_opts=compression_opts,
                r_ext_factor=r_ext_factor,
            )
        else:
            message(f"Data {ftype} {var_type} not supported yet!")
            sys.exit(0)

        return wfs_nl

    def save_modes(
        self,
        ell_max=None,
        pre_key=None,
        key_format=None,
        modes_to_save=None,
        out_file_name="mp_new_modes.h5",
        r_ext_factor=None,
        compression_opts=0,
        r_ext=None,
    ):
        """Save the waveform mode data to an hdf file.

        Parameters
        ----------
        pre_key: str, optional
                  A string containing the key of the group in
                  the HDF file in which the modes` dataset exists.
                  It defaults to `None`.
        mode_numbers: list
                       The mode numbers to load from the file.
                       Each item in the list is a list that
                       contains two integrer numbers, one for
                       the mode index :math:`\\ell` and the
                       other for the mode index :math:`m`.

        Returns
        -------
        waveform_obj: 3d array
                       Sets the three dimensional array `waveform.modes`
                       that contains the required :math:`\\ell, m` modes.

        Examples
        --------
        >>> from waveformtools.waveforms import waveform
        >>> waveform.set_basedir('./')
        >>> waveform.set_datadir('data/')
        >>> mode_numbers = [[2, 2], [3, 3]]
        >>> waveform.load_data(mode_numbers=mode_numbers)
        """
        # import dataIO
        from waveformtools import dataIO

        dataIO.save_modes_data_to_gen(
            self,
            ell_max=None,
            pre_key=None,
            key_format=None,
            modes_to_save=None,
            out_file_name="mp_new_modes.h5",
            r_ext_factor=None,
            compression_opts=0,
            r_ext=None,
        )

    def resample(self, new_time_axis=None, new_delta_t=None):
        """Resample all the waveform modes in time"""

        raise NotImplementedError

    def set_mode_data(self,
                      data, 
                      ell=None, 
                      emm=None, 
                      extra_mode_indices=None
    ):
        """Set the mode array data
        for the respective :math:`(\\ell, m)` mode. If the modes array
        has a mode axis of length more then one e.g. if one is dealing
        with not two but three index modes, them one needs to specify
        the third (r) axis index to which this data corresponds to.

        If r_index is not given, then it is assumed that
        the supplied `data` is 2 dimensional (ell, emm, all r)

        Else, only the (ell, emm, one r element) is updated.

        If mode axis length is 1, then it corresponds to usual
        two index modes like (l, m) and they are updated accordingly.

        Parameters
        ----------
        ell: int
                    The :math:`\\ell` polar mode number.
        emm: int
                    The :math:`emm` azimuthal mode number.
        data: 1d array
               The array consisting of
               mode data for the requested mode.
        extra_mode_indices: tuple
                            A tuple containing the additional
                            mode indices locating the data to
                            be set.

        Returns
        -------
        self.mode_data: modes_data
                         The updated modes data.
        """

        if ell is None and emm is None:
            self._modes_data = data

        elif isinstance(ell, int) and isinstance(emm, int):
            # Compute the emm index given ell.
            emm_index = emm + ell
            vec_idx = (ell) ** 2 + emm + ell

            if self.extra_mode_axes:
                if extra_mode_indices is not None:
                    self._modes_data[vec_idx, *extra_mode_indices] = data
            else:
                # Set the mode data.
                self._modes_data[vec_idx] = data
                # self._modes_data[ell, emm_index] = data
        else:
            raise KeyError
        
    def set_mode_data_at_t_step(
        self, t_step, time_stamp, ell, emm, data, extra_mode_indices=None
    ):
        """Set the mode array data
        for the respective :math:`(\\ell, m)` mode.
        at a single time step.

        Parameters
        ----------
        t_step: int
                 The time step of the mode
        ell: int
                    The :math:`\\ell` polar mode number.
        emm: int
                    The :math:`emm` azimuthal mode number.
        r_index: int, optional
                  The index of the third mode axis, if any
        data: 1d array
               The array consisting of
               mode data for the requested mode.
        extra_mode_indices: tuple
                            A tuple containing the additional
                            mode indices locating the data to
                            be set.
        Returns
        -------
        self.mode_data: modes_data
                         The updated modes data.
        """
        # Compute the emm index given ell.
        # emm_index = emm + ell
        vec_idx = (ell) ** 2 + emm + ell

        if self.extra_mode_axes:
            if extra_mode_indices is not None:
                message(
                    f"Setting mode data at a mode indices {extra_mode_indices}",
                    message_verbosity=4,
                )

                self._modes_data[vec_idx, *extra_mode_indices, ..., t_step] = (
                    data
                )

        else:
            message(
                "Setting mode data at just ell, emm (no r axis found)",
                message_verbosity=4,
            )
            self._modes_data[vec_idx, ..., t_step] = data

    def to_spherical_array(self, Grid, meth_info, spin_weight=None):
        """Obtain the spherical array from the modes array.

        Parameters
        ----------
        Grid: cls instance
                    An instance of the "UniformGrid" class
                    to hold the grid info.
        meth_info: cls instance
                    An instance of the class
                    `waveformtools.diagnostics.method_info` that
                    provides information on
                    what methods to use for integration.

        Returns
        -------
        waveform_sp: spherical_array
                      A member of the "spherical_array" class
                      constructed from the given "modes_rray".
        """

        # Create a spherical array.
        waveform_sp = spherical_array(label=self.label, Grid=Grid)

        if spin_weight is None:
            if self.spin_weight is not None:
                spin_weight = self.spin_weight
            else:
                spin_weight = -2
                self.spin_weight = spin_weight

        spin_weight = abs(spin_weight)
        waveform_sp._spin_weight = spin_weight
        # Set the time-axis
        try:
            waveform_sp._time_axis = self.time_axis

        except Exception as ex:
            message(ex)
            waveform_sp._frequency_axis = self.frequency_axis

        # Get the coordinate meshgrid
        theta, phi = Grid.meshgrid

        s1, s2 = theta.shape
        s3 = self.data_len
        sp_data = np.zeros((s1, s2, s3), dtype=np.complex128)

        modes_list = [
            item for item in self.modes_list if item[0] >= spin_weight
        ]
        for item in modes_list:
            # Get modes.
            ell, emm_list = item
            # if ell<spin_weight:
            # continue

            for emm in emm_list:
                # For every l, m
                sp_data += np.multiply.outer(
                    Yslm_vec(
                        spin_weight,
                        ell=ell,
                        emm=emm,
                        theta_grid=theta,
                        phi_grid=phi,
                    ),
                    self.mode(ell, emm),
                )
                # message(sp_data)
        # Set the data of the spherical array.
        waveform_sp._data = sp_data

        waveform_sp._areal_radii = self._areal_radii

        # try:
        #    waveform_sp._time_axis = self.time_axis
        # except Exception as ex:
        #    message(ex)
        #    waveform_sp._frequency_axis = self.frequency_axis

        return waveform_sp

    def trim(self, trim_upto_time=None):
        """Trim the ModesArray at the beginning and center about
        the peak of the 2,2 mode.

        Parameters
        ----------
        time: float
               The time unit upto which to discard.

        Returns
        -------
        Re-sets the `time_axis` and `ModesArray` data.
        """
        if trim_upto_time is None:
            trim_upto_time = self.r_ext

        # Compute the start index
        start = int(trim_upto_time / self.delta_t())

        # Trim the time axis
        self._time_axis = self.time_axis[start:]

        # Trim the data
        self._modes_data = self.modes_data[..., start:]

        # Recenter the time axis
        max_ind = np.argmax(np.absolute(self.mode(2, 2)))
        self._time_axis = self.time_axis - self.time_axis[max_ind]

    def to_frequency_basis(self):
        """Compute the modes in frequency basis.

        Returns
        -------
        waveform_tilde_modes: ModesArray
                               A ModesArray containing the modes
                               in frequency basis.
        """

        # Create a new modes array
        waveform_tilde_modes = ModesArray(
            label=f"{self.label} -> frequency_domain"
        )
        waveform_tilde_modes.create_modes_array(
            ell_max=self.ell_max, data_len=self.data_len
        )

        from spectools.fourier.transforms import compute_fft
        from spectools.fourier.transforms import compute_fft

        for mode in self.modes_list:
            # Ge the ell value
            ell, emm_list = mode

            for emm in emm_list:
                freq_axis, freq_data = compute_fft(
                    self.mode(ell, emm), self.delta_t()
                )

                waveform_tilde_modes.set_mode_data(
                    ell=ell, emm=emm, data=freq_data
                )

        waveform_tilde_modes.frequency_axis = freq_axis
        waveform_tilde_modes.ell_max = self.ell_max
        waveform_tilde_modes.modes_list = self.modes_list
        return waveform_tilde_modes

    def to_time_basis(self):
        """Compute the modes in time basis.

        Returns
        -------
        waveform_modes: ModesArray
                         A ModesArray containing the modes
                         in frequency basis.
        """

        # Create a new modes array
        waveform_modes = ModesArray(label=f"{self.label} -> time_domain")
        waveform_modes.create_modes_array(
            ell_max=self.ell_max, data_len=self.data_len
        )

        from spectools.fourier.transforms import compute_ifft
        from spectools.fourier.transforms import compute_ifft

        for mode in self.modes_list:
            # Extrapolate every mode

            # Ge the ell value
            ell, emm_list = mode

            for emm in emm_list:
                time_axis, time_data = compute_ifft(
                    self.mode(ell, emm), self.delta_f
                )

                waveform_modes.set_mode_data(ell=ell, emm=emm, data=time_data)

        try:
            maxloc = np.argmax(np.absolute(waveform_modes.mode(2, 2)))
        except Exception as ex:
            message(ex)
            maxloc = 0

        maxtime = time_axis[maxloc]

        waveform_modes.time_axis = time_axis - maxtime

        return waveform_modes

    def extrap_to_inf(
        self,
        mass=1,
        spin=None,
        modes_list="all",
        method="SIO",
        r_ext_factor=1,
        diff_method="CS",
        diff_degree=24,
    ):
        """Extrapolate the :math:`\\Psi_4` modes to infinity
        using the perturbative improved second order method.

        Parameters
        ----------
        mass: float
               The effective total mass of the system.
        spin: float
               The effective spin of the system.
        modes: modes array, optional
                The modes to extrapolate. Defaults
                to `all` if not specified.
        method: str
                 The method to use for extrapolation.
                 The available methods are:

        +------------+--------------------------------------+
        | Method str | Name                                 |
        +------------+--------------------------------------+
        |'FO'        | First order                          |
        |'SO'        | Second order                         |
        |'SIO'       | Second improved order                |
        |'NM'        | Numerical method (not ready yet)     |
        +------------+--------------------------------------+

        Returns
        -------
        waveform_inf_modes: modes array
                             A new modes array
                             that contains
                             the extrapolated modes.
        """

        from functools import partial

        # Prepare the extrapolating method.
        if method == "SIO":
            from waveformtools.extrapolate import (
                waveextract_to_inf_perturbative_twop5_order,
            )

            extrap_method = partial(
                waveextract_to_inf_perturbative_twop5_order,
                delta_t=self.delta_t(),
                areal_radius=self.r_ext,
                mass=mass,
                spin=spin,
                method=diff_method,
                degree=diff_degree,
            )

        if method == "SO":
            from waveformtools.extrapolate import (
                waveextract_to_inf_perturbative_two_order,
            )

            extrap_method = partial(
                waveextract_to_inf_perturbative_two_order,
                delta_t=self.delta_t(),
                areal_radius=self.r_ext,
                mass=mass,
                spin=spin,
            )

        if method == "FO":
            from waveformtools.extrapolate import (
                waveextract_to_inf_perturbative_one_order,
            )

            extrap_method = partial(
                waveextract_to_inf_perturbative_one_order,
                u_ret=self.time_axis,
                areal_radius=self.r_ext,
                mass=mass,
            )

        if method == "NM":
            message("This method is not available yet! ")

        # Prepare the modes to be extrapolated.
        if modes_list == "all":
            modes_list = construct_mode_list(self.ell_max, self.spin_weight)

        # Create a mode array for the extrapolated waveform.
        extrap_wf = ModesArray(
            label=f"{self.label} -> rPsi4_inf",
            modes_list=self.modes_list,
            ell_max=self.ell_max,
            r_ext=self.r_ext,
        )

        extrap_wf.create_modes_array(
            ell_max=self.ell_max, data_len=self.data_len
        )

        # Retain the time axis.
        extrap_wf.time_axis = self.time_axis
        for mode in modes_list:
            # Extrapolate every mode

            # Ge the ell value
            ell, emm_list = mode

            for emm in emm_list:
                # For every emm value
                message(f"Processing l{ell}, m{emm}")
                # Compute rPsi4_lm
                mode_data = r_ext_factor * self.mode(ell, emm)

                # Extrapolate
                # import ipdb; ipdb.set_trace()
                extrap_mode_data = extrap_method(rPsi4_rlm=mode_data)

                # Assign data to new modes array
                extrap_wf.set_mode_data(ell=ell, emm=emm, data=extrap_mode_data)

        message("Done!")
        return extrap_wf

    def supertranslate(self, supertransl_alpha_modes, Grid, order=4):
        """Supertranslate the :math:`\\Psi_{4\\ell m}`
        waveform modes, give then, the supertranslation parameter
        and the order.

        Parameters
        ----------
        supertransl_alpha_modes: ModesArray
                                  The ModesArray containing the
                                  supertranslation mode coefficients.
        Grid: class instance
                    The class instance that contains
                    the properties of the spherical grid
                    using which the computations are carried out.
        order: int
                The number of terms to use
                to approximate the supertranslation.

        Returns
        -------
        Psi4_supertransl: ModesArray
                           A class instance containg the
                           modes of the supertranslated
                           waveform :math:`\\Psi_4`.
        """

        import BMS

        ell_max = self.ell_max
        # Step 0: Get the grid properties for integrations

        # Compute the meshgrid for theta and phi.
        theta, phi = Grid.meshgrid
        # theta
        # Step 1: get the grid function for supertranslation parameter
        supertransl_alpha_sphere = BMS.compute_supertransl_alpha(
            supertransl_alpha_modes, theta, phi
        )

        # The supertranslation is carried out in frequency space.
        # Step 2: get the FFT of the mode coefficients of Psi4
        Psi4_tilde_modes = self.to_frequency_basis()

        # Get the 2d angular frequency array
        omega_axis_2d = Psi4_tilde_modes.omega

        # Construct the supertranslation factor
        supertransl_factor = np.sum(
            np.array(
                [
                    np.power(
                        (-1j * omega_axis_2d * supertransl_alpha_sphere), index
                    )
                    for index in range(order)
                ]
            ),
            axis=0,
        )

        # Multiply with the fourier modes.
        supertransl_spherical_factor = Psi4_tilde_modes.multiply(
            supertransl_factor
        )

        # Reconstruct the modes

        # Check!
        supertransl_spherical_grid = np.zeros(
            supertransl_spherical_factor.shape, dtype=np.complex128
        )

        for ell in range(ell_max + 1):
            for emm in range(-ell, ell + 1):
                # Multiply with the SWSH basis.
                supertransl_spherical_grid += (
                    supertransl_spherical_factor
                    * Yslm_vec(
                        spin_weight=-2,
                        ell=ell,
                        emm=emm,
                        theta=theta,
                        phi=phi,
                    )
                )

                # Step 3: Reconstruct the function on the sphere

        # Create a spherical_array to hold the supertranslated waveform
        supertransl_spherical_waveform = spherical_array(Grid=Grid)

        # Set the data.
        supertransl_spherical_waveform.data = supertransl_spherical_grid

        # Get ModesArray from spherical_array
        Psi4_supertransl_modes = supertransl_spherical_waveform.to_modes_array(
            ell_max=ell_max
        )

        return Psi4_supertransl_modes

    def boost(self, conformal_factor, Grid=None):
        """Boost the waveform given the unboosted waveform
        and the boost conformal factor.

        Parameters
        ----------
        conformal_factor: 2d array
                           The conformal factor for
                           the Lorentz transformation.
                           It may be a single floating point number
                           on a spherical grid. The array will be
                           of dimensions [ntheta, nphi].

        Returns
        -------
        boosted_waveform: spherical_array
                           The class instance `spherical_array`
                           that contains the boosted waveform.
        """

        from spectools.spherical.grids import UniformGrid
        from spectools.spherical.grids import UniformGrid

        # Construct a spherical grid.
        if Grid is None:
            Grid = UniformGrid()

        # Get spherical array from modes.
        unboosted_waveform = self.to_spherical_array(Grid)

        boosted_waveform_data = unboosted_waveform.boost(conformal_factor)

        # Construct a 2d waveform array object
        boosted_waveform = spherical_array(
            Grid=unboosted_waveform.Grid,
            data=np.array(boosted_waveform_data),
        )
        boosted_waveform.label = "boosted"

        # Get modes from spherical data.
        # boosted_waveform_modes = boosted_waveform.to_modes_array()

        # return boosted_waveform_modes
        return boosted_waveform

    def get_strain_from_psi4(self, omega0="auto"):
        """Get the strain `ModesArray` from :math:`\\Psi_4` by
        fixed frequency integration.

        Parameters
        ----------
        omega0: float, optional
                The lower cutoff angular frequency for FFI.
                By default, it computes this from the mode data.

        Returns
        -------
        strain_waveform: ModesArray
                          The computed strain modes.
        """

        # Initialize a mode array for strain.
        strain_waveform = ModesArray(
            label="{} strain from Psi4".format(self.label),
            r_ext=self.r_ext,
            ell_max=8,
            modes_list=self.modes_list,
        )

        strain_waveform.ell_max = self.ell_max

        data_len = self.data_len

        strain_waveform.create_modes_array(
            ell_max=self.ell_max, data_len=data_len
        )

        # Integrate,
        from waveformtools.integrate import fixed_frequency_integrator
        from waveformtools.waveformtools import (
            get_starting_angular_frequency as sang_f,
        )

        omega_st = omega0
        for item in self.modes_list:
            ell, emm_list = item
            for emm in emm_list:
                mode_data = self.mode(ell, emm)
                if omega0 == "auto":
                    omega_st = (
                        abs(sang_f(mode_data, delta_t=self.delta_t())) / 10
                    )
                strain_time, strain_mode_data = fixed_frequency_integrator(
                    udata_time=mode_data,
                    delta_t=self.delta_t(),
                    omega0=omega_st,
                    order=2,
                )
                strain_waveform.set_mode_data(ell=ell, emm=emm, data=strain_mode_data)

        strain_waveform.time_axis = strain_time

        strain_waveform.trim(trim_upto_time=0)

        return strain_waveform

    def get_news_from_psi4(self, omega0="auto"):
        """Get the News `ModesArray` from :math:`\\Psi_4` by
        fixed frequency integration.

        Parameters
        ----------
        omega0: float, optional
                 The lower cutoff angular frequency for FFI.
                 By default, it computes this as one tenth of
                 the starting frequency of the respective mode data.

        Returns
        -------
        news_waveform: ModesArray
                        The computed strain modes.
        """

        # Initialize a mode array for strain.
        # news_waveform = ModesArray(label=f'{self.label} news from
        # Psi4', r_ext=500, ell_max=8, modes_list=self.modes_list)
        news_waveform = ModesArray(
            label="{} news from Psi4".format(self.label),
            r_ext=500,
            ell_max=8,
            modes_list=self.modes_list,
        )

        # news_waveform.time_axis = self.time_axis
        news_waveform.ell_max = self.ell_max

        data_len = self.data_len

        news_waveform.create_modes_array(
            ell_max=self.ell_max, data_len=data_len
        )

        # Integrate,
        omega_st = omega0
        for item in self.modes_list:
            ell, emm_list = item
            for emm in emm_list:
                mode_data = self.mode(ell, emm)
                if omega0 == "auto":
                    omega_st = (
                        abs(sang_f(mode_data, delta_t=self.delta_t())) / 10
                    )
                news_time_axis, news_mode_data = fixed_frequency_integrator(
                    udata_time=mode_data,
                    delta_t=self.delta_t(),
                    omega0=omega_st,
                    order=1,
                )
                news_waveform.set_mode_data(
                    ell=ell, emm=emm, data=news_mode_data
                )

        news_waveform._time_axis = news_time_axis

        news_waveform.trim(trim_upto_time=0)

        return news_waveform

    def taper(self, zeros="auto"):
        """Taper a waveform at both ends and insert zeros if needed

        Parameters
        ----------
        zeros: int
                The number of zeros to add at rach end

        Returns
        -------
        tapered_modes: ModesArray
                        Modes array with tapered mode data.
        """

        from waveformtools.waveformtools import taper

        if zeros == "auto":
            # Decide the number of zeros
            data_len = self.data_len

            nearest_power = int(np.log(data_len) / np.log(2))
            req_len = np.power(2, nearest_power + 1)
            zeros = req_len - data_len
            message("num_zeros", zeros)

        # New modes array.

        tapered_modes = None

        for item in self.modes_list[:]:
            ell, emm_list = item
            for emm in emm_list:
                input_data_re = self.mode(ell, emm).real
                input_data_im = self.mode(ell, emm).imag

                tapered_data_re = taper(input_data_re, zeros=zeros)
                tapered_data_im = taper(input_data_im, zeros=zeros)

                # tapered_data_re = taper_tanh(input_data_re,
                # delta_t=self.delta_t())
                # tapered_data_im = taper_tanh(input_data_im,
                # delta_t=self.delta_t())

                new_data_len = len(tapered_data_re)

                if tapered_modes is None:
                    tapered_modes = ModesArray(
                        label="tapered {}".format(self.label),
                        r_ext=self.r_ext,
                        modes_list=self.modes_list,
                        ell_max=self.ell_max,
                    )

                    tapered_modes.create_modes_array(
                        ell_max=self.ell_max, data_len=new_data_len
                    )
                tapered_data = tapered_data_re + 1j * tapered_data_im

                # message(len(tapered_data_re))
                tapered_modes.set_mode_data(ell=ell, emm=emm, data=tapered_data)

        # Set the time axis
        new_time_axis = np.arange(
            0, new_data_len * self.delta_t(), self.delta_t()
        )

        tapered_modes.time_axis = new_time_axis

        # Recenter the modes.
        tapered_modes.trim(trim_upto_time=0)

        return tapered_modes

    def taper_tanh(
        self, time_axis=None, zeros="auto", duration=10, sides="both"
    ):
        """Taper a waveform at both ends and insert zeros if needed

        Parameters
        ----------
        zeros: int
                The number of zeros to add at rach end

        Returns
        -------
        tapered_modes: ModesArray
                        Modes array with tapered mode data.
        """

        from waveformtools.waveformtools import taper_tanh

        if zeros == "auto":
            # Decide the number of zeros
            data_len = self.data_len

            nearest_power = int(np.log(data_len) / np.log(2))
            req_len = np.power(2, nearest_power + 1)
            zeros = req_len - data_len
            # message('num_zeros', zeros)

        # New modes array.

        tapered_modes = None

        for item in self.modes_list[:]:
            ell, emm_list = item
            for emm in emm_list:
                input_data_re = self.mode(ell, emm).real
                input_data_im = self.mode(ell, emm).imag

                # tapered_data_re = taper(input_data_re, zeros=zeros)
                # tapered_data_im = taper(input_data_im, zeros=zeros)

                _, tapered_data_re = taper_tanh(
                    input_data_re,
                    delta_t=self.delta_t(),
                    duration=duration,
                    sides=sides,
                )
                _, tapered_data_im = taper_tanh(
                    input_data_im,
                    delta_t=self.delta_t(),
                    duration=duration,
                    sides=sides,
                )

                new_data_len = len(tapered_data_re)

                if tapered_modes is None:
                    tapered_modes = ModesArray(
                        label="tapered {}".format(self.label),
                        r_ext=self.r_ext,
                        modes_list=self.modes_list,
                        ell_max=self.ell_max,
                    )

                    tapered_modes.create_modes_array(
                        ell_max=self.ell_max, data_len=new_data_len
                    )
                tapered_data = tapered_data_re + 1j * tapered_data_im

                # message(len(tapered_data_re))
                tapered_modes.set_mode_data(ell=ell, emm=emm, data=tapered_data)

        # Set the time axis
        new_time_axis = np.arange(
            0, new_data_len * self.delta_t(), self.delta_t()
        )

        tapered_modes.time_axis = new_time_axis

        # Recenter the modes.
        tapered_modes.trim(trim_upto_time=0)

        return tapered_modes

    def low_cut(self, omega0=0.03, order=2):
        """Apply the low cut filter from `waveformtools.low_cut_filter`

        Parameters
        ----------
        order: int, optional
                The order of the butterworth filter.
        omega0: float, optional
                 The cutoff frequency of the butterworth filter.

        Returns:
        --------
        filtered_modes: ModesArray
                         A modes array object containing filtered modes.
        """

        # ModesArray for filtered data.
        filtered_modes = None

        # Import the filter
        from waveformtools.waveformtools import low_cut_filter

        for item in self.modes_list:
            # Iterate over available modes.
            ell, emm_list = item
            for emm in emm_list:
                if filtered_modes is None:
                    # Create filtered_modes
                    filtered_modes = ModesArray(
                        label="lc filtered {}".format(self.label),
                        r_ext=self.r_ext,
                        modes_list=self.modes_list,
                        ell_max=self.ell_max,
                    )

                    filtered_modes.create_modes_array(
                        ell_max=self.ell_max, data_len=self.data_len
                    )

                # Get filtered mode data.
                filtered_data = low_cut_filter(
                    self.mode(ell, emm),
                    self.frequency_axis,
                    omega0=omega0,
                    order=order,
                )

                # Set the mode data.
                filtered_modes.set_mode_data(ell=ell, emm=emm, data=filtered_data)

        # Set the f axis.
        filtered_modes.frequency_axis = self.frequency_axis

        return filtered_modes

    def to_td_waveform(
        self,
        Mtotal=1,
        inclination=0,
        phi_ref=0,
        distance=1,
        delta_t=None,
        k=None,
    ):
        """Get the plus and cross polarizations of
        of the waveform time series by summing the modes.

        Parameters
        ----------
        theta, phi: float
                    The inclination and the azimuthal
                    angular position of the observer
                    in the NR coodinate system.
        distance: float
                   The distance to the source

        method: str
                 The method to use to generate
                 the SWSH basis. This can be
                 `precise` or `fast`.

        Returns
        -------
        taxis, hp, hc: 1d arrays
                        The 1d arrays of the time axis and
                        the polarizations of the waveforms.

        Notes
        -----
        This does not rotate the polarizations. The rotation
        must be done separately using the `rotate_polarizations`
        function of `waveformtools.transforms`

        For precessing systems and to obtain the waveform
        in the LAL convention, one should use the
        nrcatalogtools package to obtain the correct
        angles first.
        """
        import lal

        angles = get_nr_frame_angles_from_lal(inclination, phi_ref)
        
        theta = angles["theta"]
        phi = angles["psi"]
        ialpha = angles["alpha"]

        polarizations = self.evaluate_angular(theta=theta, phi=phi)

        #import matplotlib.pyplot as plt
        #plt.plot(polarizations.real)
        #plt.plot(polarizations.imag)
        #plt.show()

        # Rotate polarizations
        #if alpha is not None:
        from waveformtools.transforms import rotate_polarizations
        polarizations = rotate_polarizations(polarizations, ialpha)
        
        #plt.plot(polarizations.real)
        #plt.plot(polarizations.imag)
        #plt.show()

        # Rescale the time axis
        taxis = self.time_axis * lal.MTSUN_SI * Mtotal

        if isinstance(delta_t, float):
            # Resample the waveform
            new_taxis = np.arange(taxis[0], taxis[-1], delta_t)
            polarizations = interp_resam_wfs(polarizations, taxis, new_taxis, k=k)
            taxis = new_taxis

        amp_factor = lal.G_SI * Mtotal * lal.MSUN_SI / (lal.C_SI**2 * distance * 1e6 * lal.PC_SI)

        # Rescale the magnitude of the waveform
        polarizations *= amp_factor

        #import matplotlib.pyplot as plt

        #plt.plot(polarizations.real, label='+')
        #plt.plot(polarizations.imag, label='x')
        #plt.legend()
        #plt.show()
        
        return taxis, polarizations.real, polarizations.imag

    def evaluate_angular(
        self, theta=None, phi=None, ell_max=None, max_t_steps=None
    ):
        """Evaluate the expansion at requested angular coordinates
        by generating SWSHs in parallel and vectorizing the
        summation"""
        from spectools.spherical.Yslm_mp import Yslm_mp
        from spectools.spherical.Yslm_mp import Yslm_mp

        if ell_max is None:
            ell_max = self.ell_max

        if max_t_steps is None:
            max_t_steps = self.data_len

        if (np.array(theta) == np.array(None)).all() or (
            np.array(phi) == np.array(None)
        ).all():
            theta, phi = self.Grid.meshgrid

        sYlm = Yslm_mp(
            ell_max=ell_max, spin_weight=self.spin_weight, theta=theta, phi=phi,
            Grid=self.Grid)

        sYlm.run()
        sYlm_funcs_vec = sYlm.sYlm_modes._modes_data
        modes_data_len = sYlm_funcs_vec.shape[0]

        val = np.tensordot(
            self._modes_data[:modes_data_len, ..., :max_t_steps],
            sYlm_funcs_vec,
            axes=((0), (0)),
        )

        return val

    def plot_modes(self, modes_to_plot='auto', threshold=1e-4, threshold_n_points=10):
        """Plot the requested set of modes"""

        import matplotlib.pyplot as plt

        try:
            import vlconf
            vlconf.conf_matplolib()
        except Exception as excep:
            excep("Unable to conf matplotlib")

        
        if isinstance(modes_to_plot, str):
            if modes_to_plot == 'auto':
                modes_to_plot = self.get_non_zero_modes(threshold, threshold_n_points)

        fig, ax = plt.subplots(figsize=(12, 6))

        nmodes = len(modes_to_plot)
        nrows = int(np.sqrt(nmodes))
        ncols = nrows+1

        for one_mode in modes_to_plot:
            ell, emm = one_mode

            ax.plot(
                self.time_axis,
                self.mode(ell, emm).real,
                label=f"Re( ({ell}, {emm}) )",
            )

        ax.set_xlabel("t/M")
        ax.set_ylabel(r"$\Psi_{(\ell, m)}$")
        plt.legend(ncol=ncols)
        plt.show()
        return fig, ax
        

    def plot_strongest_modes(self, 
                             nmodes=3,
                             save_fig=False,
                             xlim=[-1200, 100],
                             ylim="auto",
                             nstop=1,
                             plot22=False,):
        
        from waveformtools.compare import plot_modes

        return plot_modes(self, nmodes,
                         save_fig,
                         xlim,
                         ylim,
                         nstop,
                         plot22)
        
    def get_non_zero_modes(self, threshold=1e-4, threshold_n_points=10):

        non_zero_modes = []
        for ell, emm_list in self.modes_list:
            for emm in emm_list:
                amp_lm = np.absolute(self.mode(ell, emm))

                locs = np.where(amp_lm > threshold)[0]
                if len(locs) > threshold_n_points:
                    non_zero_modes.append((ell, emm))

        return non_zero_modes
    
    def time_derivative(self, mode=None, method="spline"):

        #if mode is None:
        #    # modes_list = self.modes_list
        #    data = self.modes_data
        #else:
        #    data = self.mode(*mode)

        d_wfm = deepcopy(self)
        from waveformtools.differentiate import derivative
        # dt = self.delta_t
        # for ell, emm_list in modes_list:
        #    for emm in range(-ell, ell+1):
        # data = self.mode(ell, emm)
        if "FD" in method:
            d_data = derivative(self.time_axis, self.modes_data, method=method)
            d_wfm._modes_data = d_data
        else:
            modes_list = self.modes_list

            for ell, emm_list in modes_list:
                for emm in emm_list:
                    y_data = self.mode(ell, emm)
                    d_data_lm_re = derivative(
                        self.time_axis, y_data.real, method=method
                    )
                    d_data_lm_im = derivative(
                        self.time_axis, y_data.imag, method=method
                    )
                    d_data_lm = d_data_lm_re +1j*d_data_lm_im
                    d_wfm.set_mode_data(
                        ell=ell, emm=emm, data=d_data_lm
                    )

        return d_wfm

    def time_integral(self, a=None, b=None, method='SP'):
        modes_list = self.modes_list
        int_wf_modes = SingleMode(spin_weight=self.spin_weight,
                                  ell_max=self.ell_max,
                                  label='BLawRHS',
                                  Grid=self.Grid)
        xdata = self.time_axis

        if a is None:
            a = xdata[0]
        if b is None:
            b = xdata[-1]

        for ell, emm_list in modes_list:
            for emm in emm_list:
                ydata = self.mode(ell, emm)
                #xdata = self.time_axis
                yinterp_re = InterpolatedUnivariateSpline(xdata, ydata.real, k=5)
                yinterp_im = InterpolatedUnivariateSpline(xdata, ydata.imag, k=5)
                mode_integral_re = yinterp_re.integral(a=a,
                                                    b=b)
                mode_integral_im = yinterp_im.integral(a=a,
                                                    b=b)
                mode_integral  = mode_integral_re + 1j*mode_integral_im
                int_wf_modes.set_mode_data(ell=ell,
                                           emm=emm,
                                           data=mode_integral
                                          )

        return int_wf_modes
    
    def compute_waveform_balance_law(self, 
                                     M_adm, 
                                     M_final, 
                                     v_kick, 
                                     Grid=None,
                                     debug=False):

        if Grid is None:
            Grid = self.Grid

        from waveform_balance_laws.laws import balance_law
        
        violations = balance_law(strain_modes=self,
                                 ginfo=Grid,
                                 M_adm=M_adm,
                                 M_final=M_final,
                                 v_kick=v_kick,
                                 debug=debug
                                )
        return violations
    

    def compute_waveform_balance_law_finite_time(self, 
                                                 psi2_modes, 
                                                 Grid=None):

        if Grid is None:
            Grid = self.Grid

        from waveform_balance_laws.laws import balance_law_finite_time
        violations = balance_law_finite_time(strain_modes=self,
                                             psi2_modes=psi2_modes,
                                             ginfo=Grid,
                                            )
        
        return violations
    
    def get_news_from_strain(self, method='spline'):
        ''' '''
        news = deepcopy(self)
        #news._time_axis = deepcopy(news.time_axis)
        news = self.time_derivative(method='spline')

        return news

    def compute_momentum_flux(self, news_modes):

        dPxdt = np.zeros(self.data_len, dtype=np.float64)
        dPydt = np.zeros(self.data_len, dtype=np.float64)
        dPzdt = np.zeros(self.data_len, dtype=np.complex128)

        # ell-1 because the linear momentum contrib involves higher order
        # terms ell+1
        modes_list = construct_mode_list(ell_max=self.ell_max-1, spin_weight=-2)

        for ell, emm_list in modes_list:
            for emm in emm_list:
                f_lm = compute_linear_momentum_contribution_from_news(news_modes, ell, emm)
                #print(f"f_{ell, emm}", f_lm[2])
                dPxdt += f_lm[0]
                dPydt += f_lm[1]
                dPzdt += f_lm[2]

        #print("Force", dPzdt)
        return dPxdt, dPydt, dPzdt
    
    def compute_kick(self, Mfinal=1):
        news_modes = self.get_news_from_strain()
        dPxdt, dPydt, dPzdt = self.compute_momentum_flux(news_modes)
        p_kick = compute_impulse_from_force(news_modes.time_axis, 
                                            dPxdt, 
                                            dPydt, 
                                            dPzdt)
        v_kick = p_kick/Mfinal
        
        return v_kick
    
    def compute_kick_direct(self, Mfinal=1):
        from waveformtools.integrate import TwoDIntegral
        news_modes = self.get_news_from_strain()
        news = news_modes.evaluate_angular()
        intensity = np.absolute(news)**2
        theta, phi = self.Grid.meshgrid
        Y10 = Yslm_vec(spin_weight=0, ell=1, emm=0, theta_grid=theta, phi_grid=phi)
        Y11 = Yslm_vec(spin_weight=0, ell=1, emm=1, theta_grid=theta, phi_grid=phi)
        P10 = intensity*Y10
        factor_10 = 2*np.sqrt(np.pi/3)
        factor_11 = 2*np.sqrt(2*np.pi/3)
        dPzdt = factor_10*TwoDIntegral(P10, self.Grid, 'GL')/(16*np.pi)
        del P10
        P11 = intensity*Y11
        dPxydt = factor_11*TwoDIntegral(P11, self.Grid, 'GL')/(16*np.pi)
        del P11
        del intensity
        dPxdt = dPxydt.imag
        dPydt = dPxydt.real
        p_kick = compute_impulse_from_force(news_modes.time_axis, dPxdt, dPydt, dPzdt)
        v_kick = p_kick/Mfinal
        
        return v_kick.real
    
    def crop(self, start_idx, end_idx):

        cropped_wfm = self.deepcopy()
        cropped_wfm._time_axis = self.time_axis[start_idx:end_idx]
        cropped_wfm._modes_data = self.modes_data[..., start_idx:end_idx]

        return cropped_wfm

    def get_power_from_news_modes(self, news_modes):

        power = np.sum(np.absolute(news_modes.modes_data )**2, axis=0)/(16*np.pi)

        return power
    
    def compute_energy_radiated(self, 
                                news_modes=None, 
                                t_start=None,
                                t_end=None, 
                                since_peak=False,
                                inspiral_only=False):
    
        if news_modes is None:
            news_modes = self.get_news_from_strain()

        if since_peak or inspiral_only:
            power = self.get_power_from_news_modes(news_modes)
            if since_peak:
                t_start = self.time_axis[np.argmax(power)]

            if inspiral_only:
                t_end = self.time_axis[np.argmax(power)]

        if t_start is None:
            t_start = self.time_axis[0]

        if t_end is None:
            t_end = self.time_axis[-1]

        power = self.get_power_from_news_modes(news_modes)
        print(t_start, t_end)
        energy_loss = InterpolatedUnivariateSpline(self.time_axis, 
                                                   power, 
                                                   k=5).integral(a=t_start, 
                                                                  b=t_end)

        return energy_loss
    
    def compute_angular_momentum_radiated(self,
                                          news_modes=None,
                                          t_start=None,
                                          t_end=None,
                                          since_peak=False,
                                          inspiral_only=False):
        
        if news_modes is None:
            news_modes = self.get_news_from_strain()

        d_ang_momentum = compute_angular_momentum(self, 
                                                  news_modes,
                                                  t_start=t_start,
                                                  t_end=t_end,
                                                  since_peak=since_peak,
                                                  inspiral_only=inspiral_only)
        
        return d_ang_momentum
    
    def bar(self):
        ''' Only complex conjugate the modes - don't change the 
        spin weight '''

        bar_modes = deepcopy(self)
        bar_modes._modes_data = np.conjugate(self.modes_data)
        return bar_modes
