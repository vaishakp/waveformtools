Transforms
==========

The ``wavefortools.transforms`` module provides capabilities to transform the waveforms in the following ways:

1. The time series to frequency domain and vice-versa.
2. The modes data into coordinate representation on the sphere and vice-versa.


Fourier transforms
------------------

One can carry out FFT of waveform data either presesented in coordinate space (as ``spherical_array``) or Fourier space (of ``modes_arrays``) using the methods contained in the respective classes.


Spin-weighted spherical harmonic transforms
-------------------------------------------

For changing the representation between the coordinate space and SWSH basis, one needs to reconstruct/ find the mode coefficients. These are carried out using Gauss-Legendre grids or uniform grids. The 2D integration can be performed using one of the several available methods (like Gauss-Legendre, Midpoint, Simpson's, Driscoll-Healy, etc.).

Computation of SWSH basis functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two choised for computing the SWSH basis functions required in the transformations. One is named `fast` and is implemented using numpy, and the other is a `precise`. The latter method uses sympy (and optionally gmpy) and is capable of computing the SWSHs at arbitrary accuracy, upto any number of digits as requested by the user. In the former method, the computations are fast and accurate upto about 14 decimals for :math:`\\ell \\leq 25`. In the latter, the SWSH can be computed to any precision required at the cost of execution time.





