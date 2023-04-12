*********
Waveforms
*********

The `waveformtools.waveforms` is a class and a set of functions to conveniently load, transform, handle and save waveforms data from Numerical Relativity smulations. 

Introduction 
------------

Waveforms data from NR simulations are routinely described in terms of modes, from expansion in a spectral basis on spherical surfaces. In gravitational wave astronomy, one is mainly conerned with the Weyl scalar component :math:`\Psi_4` and the strain :math:`h`.


The `modes_array` class
-----------------------

The `modes_array` class provides a basic container for holding modes data of such spectral expansions of any spin weight, and carrying out various transformations on them. Some of its basic features/capabilities are:

1. Time domain integration and differentiation using Fixed frequency integration. e.g. Integrate :math:`\Psi_4` twice to obtain the strain :math:`h`. 
2. Convenient access of modes data using :math:`(l, m)` notation.
3. Data output to custom simple data format (explained below) called `gen`.
4. Automatically load strain :math:`h` or :math:`\Psi_4` modes data from various catalogues, given the source file path.
	1. RIT
	2. SpEC
	3. SpECTRE CCE
	4. simple/generic
5. Transformation to `spherical_array` (see below).
6. Sample the waveform time series at any angular location :math:`(\theta, \phi)`
6. Conversion to frequency domain modes.
7. All the modes data are saved as an ndarray and mostly vectorized operations are implemented.
8. Expensive operations are accelerated using numba, which also provides GPU interface and caching. 
9. Extrapolation of waveforms to infinity using perturbative methods.
10. BMS transformations.
11. Centre of mass corrections.


The simple "gen" data format
----------------------------

1. This is based on h5 files with optional compression. 
2. All the attributes of the `modes_array` class, barring the main numerical modes data values are treated as metadata data and saved into the h5 file. 
3. The data, when loaded into a modes_array, automatically loads the metadata and associates to to the attributes.
4. The data  of individual modes is saved as data sets of the h5 file.
5. The time axis is common across the modes and is uniformly sampled. 
6. If the input data loaded from a catalogue is non uniformly sampled, the user can specify the resampled time stepping, or default to finest/ coarsest available.
7. Examples of default metadata attributes:
	1. spin_weight
	2. time stepping.
	3. Last modified.
	4. Source file path
	5. label / alias

`spherical_array` class 
-----------------------

The `spherical_array` class is the coordinate space (:math:`\theta, \phi`) representation of waveforms time series data. This contains data represented on a spherical grid and has the following main features

1. The grid is staggered across the poles to circumvent dealing with coordinate singularities.
2. Transformation/ expansion to spectral basis i.e. `modes_array` given the spin weight.




Interface to `waveformtools.waveformtools`
------------------------------------------

The `waveformtools.waveformtools` module is a toolkit to carryout various transformations of the waveforms. Please see its description file for further details.


