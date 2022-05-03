[![pipeline status](https://gitlab.com/vaishakp/waveformtools/badges/main/pipeline.svg)](https://gitlab.com/vaishakp/waveformtools/commits/main)

Waveformtools 
===============


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



## Citing this code

Please cite the latest version of this code if used in your work.




## Dependencies

This module has the following dependencies:

* Standard packages (come with full anaconda installation)
    * [`numpy`](http://www.numpy.org/)
    * [`scipy`](http://scipy.org/)
    * [`matplotlib`](http://matplotlib.org/)
    * [`h5py`](http://www.h5py.org/)
* The pyCBC module
    * [`pyCBC`](https://pycbc.org/).


## Installing this module


### Recommended method

I recommend installing this module simply by cloning it and adding the path to the $PYTHONPATH variable

```sh
git clone https://gitlab.com/vaishakp/waveformtools.git
PYTHONPATH="/path/to/this/cloned/repo":$PYTHONPATH
export PYTHONPATH
```

And resolving the dependencies manually by using the anaconda distribution.

```sh
conda install ....
```

### Manual method

This is not recommended. One can also install this using the pip commands on this git repository:

```sh
pip install git+git://gitlab.com/vaishakp/waveformtools.git
```
or by running python setup file on the cloned repository:


```sh
python setyp.py install
```

This is not recommended because of various reasons:

* The commands are better run on user privelages, and using virtual environments, so as to 
    * not cause system version conflicts 
    * Avoid dependency issues


## Documentation

The documentation for this module is available at [Link to the Documentation](https://vaishakp.gitlab.io/waveformtools/). This was built automatically using Read the Docs.

## Acknowledgements

This project has been hosted, as you can see, on gitlab. Several gitlab tools are used in the deployment of the code, its testing, version control.

The work of this was developed in aiding my PhD work at Inter-University Centre for Astronomy and Astrophysics (IUCAA, Pune, India)](https://www.iucaa.in/). The PhD is in part supported by the [Shyama Prasad Nukherjee Fellowship](https://csirhrdg.res.in/Home/Index/1/Default/2006/59) awarded to me by the [Council of Scientific and Industrial Research (CSIR, India)](https://csirhrdg.res.in/). Resources of the [Inter-University Centre for Astronomy and Astrophysics (IUCAA, Pune, India)](https://www.iucaa.in/) were are used in part.
