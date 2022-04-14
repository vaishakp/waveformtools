.. waveformtools documentation master file, created by
   sphinx-quickstart on Thu Jul  9 21:06:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


===================================

Welcome to waveformtools documentation!

Browse through these pages for details on the functions in this module. Please contact vaishak@iucaa.in if you want to report any bugs or have comments/questions. 

waveformtools is a numerical relativity data handling package that was written to aid the handling and analysis of numerical relativity data.

This package contains implementations of customized algorithms and techniques.  Some of these contain the usage of existing python based library functions from pycbc, scipy, etc but effort has been made to keep these to a minimum.
 
1.  Handling of numerical relativity data, and retreiving specific information about the physical system.
    
    The class container and methods "sim" can load NR data into convenient lists and dictionaries, which can be used to retrieve specific data/ information about the numerical simulation. This offers the following functionality, like retreiving:
    
    a). Horizon masses, mass-ratios, and areas.
    
    b). Horizon multipole moments.
    
    c). Merger time/ formation time of common horizon.
    
    d). The strain waveform.
    
    e). The shear data of the dynamical horizons.
	
    f). And computing/ extracting the frequency, amplitude and phase of waveforms, etc.

2.  Handling of waveforms.
    
    A basic class container for handling numerical waveforms. This offers functionality like:
    
    a). loading data from hdf5 files.
    
    b). basic information such as time step, time axis, data axis, etc.
    
    c). extracting a numerical waveform recorded at a finite radius to null infinity.


3.  Tools for preparation/handling of numerical relativity data, like:
    
    a). checking for discontinuity, removal of duplicated rows, and interpolating for missing rows in the data.
    
    b). integration, differentiation of numerical data.
    
    c). equalizing lengths of waveforms.
    
    d). resampling
    
    e). computing the norm, shifting/ rolling the data, etc.
    
    f). smoothening, tapering.

4.  Tools for basic data analysis on waveforms.
    
    a). match algorithms for matching two waveforms.
    
    b). binning and interpolation of data. 

5.  Miscellaneous tools:
    
    a). Progress bar to display progress of loops.
    
    b). A custom print function with message prioritization.
    
    c). Saving data to disk with protocol support (binary, text, etc.)   



.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


waveformtools main
============
.. automodule:: waveformtools
   :members:



