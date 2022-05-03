[![pipeline status](https://gitlab.com/vaishakp/waveformtools/badges/main/pipeline.svg)](https://gitlab.com/vaishakp/waveformtools/commits/main) [![Crowdin](https://d322cqt584bo4o.cloudfront.net/screencam/localized.svg)](https://crowdin.com/project/screencam)

Waveformtools 
===============


This is a python module for the handling and analysis of waveforms and data from Numerical Relativity codes, and for carrying out gravitational wave data analysis.  


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

The work of this code was in part supported by the Shyama Prasad Nukherjee Fellowship awarded to me by the Council of Scientific and Industrial Research (CSIR, India). Resources of the Inter-University Centre for Astronomy and Astrophysics (IUCAA, Pune, India) were are used in part.
