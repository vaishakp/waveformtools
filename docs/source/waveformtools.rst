************
Waveforms
************

This is a class and a set of functions to conveniently load, transform, handle and save waveforms data from Numerical Relativity smulations. 

# Introduction 

Waveforms data from NR simulations are routinely described in terms of modes, from expansion in a spectral basis on spherical surfaces. In gravitational wave physics, one is mainly conerned with the Weyl scalar component $`r\Psi_4/M`$ and the strain $`r h/M`$.


The `modes_array' class provides a container for holding modes data of such spectral expansions of any spin weight, and carrying out various transformations on them. Some of the basic features/capabilities are:

1. Time domain integration and differentiation using Fixed frequency integration. e.g. integrate $`\Psi_4`$ twice to obtain the strain $`h`$. 
2. Convenient access of modes data using (l, m) notation. E.g. wf(2, 2) is used to access the (l=2, m=2) mode time series.
3. Data output to custom simple data format (explained below) called "gen".
4. Automatically load strain $`h`$ or $`\Psi_4`$ modes data from various catalogues, given the source file path.
	1. RIT
	2. SpEC
	3. SpECTRE CCE
	4. simple/generic
5. Transformation to `spherical_array` (see below)
6. Conversion to frequency domain.
7. All the modes data are saved as an ndarray and mostly vectorized operations are implemented.
8. Expensive operations are accelerated using numba, which also provides GPU interface and caching. 
9. Extrapolation of waveforms to infinity using perturbative methods.
10. BMS transformations.
11. Centre of mass corrections.


## The simple "gen" data format
1. All the attributes of the `modes_array` class, barring the main numerical modes data values, are treated as metadata and saved into the h5 file. 
2. Based on h5 files with optional compression. 
3. The data, when loaded into a modes_array, automatically loads all available metadata and assigns them to its attributes at once.
4. The data of individual modes is saved as data sets of the h5 file.
5. The time axis is common across the modes and is uniformly sampled. 
6. If the input data loaded from a catalogue is non uniformly sampled, the user can specify the resampled time stepping, or it defaults to the finest/ coarsest available.
7. Examples of default metadata attributes:
	1. spin_weight
	2. time stepping.
	3. Last modified.
	4. Source file path
	5. label / alias
	6. Extraction radius

## `spherical_array` class is the coordinate space ($`\theta, \phi`$) representation of waveforms data. This contains data represented on a spherical grid and has the following main features

1. The grid is staggered across the poles to circumvent dealing with coordinate singularities.
2. Transformation/ expansion to spectral basis i.e. `modes_array` given the spin weight.






