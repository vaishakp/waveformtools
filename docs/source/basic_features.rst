*********************
Basic Features
*********************

This is a python module for the handling and analysis of waveforms and data from Numerical Relativity codes, and for carrying out gravitational wave data analysis.  

waveformtools is a numerical relativity data handling package that was written to aid the handling and analysis of numerical relativity data.

This package contains implementations of customized algorithms and techniques.  Some of these contain the usage of existing python based library functions from pycbc, scipy, etc but effort has been made to keep these to a minimum.

 

* Handling of numerical relativity data, and retreiving specific information about the physical system.

    The class container and methods "sim" can load NR data into convenient lists and dictionaries, which can be used to     retrieve specific data/ information about the numerical simulation. 

    This offers the following functionality, like retreiving:

    * Horizon masses, mass-ratios, and areas.

    * Horizon multipole moments.

    * Merger time/ formation time of common horizon.

    * The strain waveform.

    * The shear data of the dynamical horizons.

	* And computing/ extracting the Frequency, amplitude and phase of waveforms, etc.

* Handling of waveforms.

    A  basic class container for handling numerical waveforms. 

    This offers functionality like:

    * loading data from hdf5 files.

    * basic information such as time step, time axis, data axis, etc.

    * extrapolating a numerical waveform recorded at a finite radius to null infinity (to be added soon).

    * integrating and differentiating waveforms in the frequency domain (to be added soon). 



* Tools for preparation/handling of numerical relativity data, like:

    * checking for discontinuity, removal of duplicated rows, and interpolating for missing rows in the data.

    * integration, differentiation of numerical data.

    * equalizing lengths of waveforms.

    * resampling

    * computing the norm, shifting/ rolling the data, etc.

    * Smoothening, tapering.


* Tools for basic data analysis.

    * A match algorithms for matching two waveforms.

    * Binning and interpolation of data. 

* Miscellaneous tools:

    * Progress bar to display progress of loops.

    * A custom print function with message prioritization.

    * Saving data to disk with protocol support (binary, text, etc.)   
