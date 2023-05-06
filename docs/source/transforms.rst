Transforms
==========

The ``wavefortools.transforms`` module provides capabilities to transform the waveforms in the fllowing ways:

1. The time series to frequency domain and vice-versa.
2. The modes data into coordinate representation on the sphere.


Fourier transforms
------------------

One can carry out FFT of waveform data either presesented in coordinate space (as ``spherical_array``) or Fourier space (of ``modes_arrays``) using the methods contained in the respective classes.


Spin-weighted spherical harmonic transforms
-------------------------------------------

For changing the representation between the coordinate space and SWSH basis, one needs to reconstruct/ find the mode coefficients. These are carried out using Gauss-Legendre grids or uniform grids. The 2D integration can be performed using one of the several available methods (like Gauss-Legendre, Midpoint, Simpson's, Driscoll-Healy, etc.).

Computation of SWSH basis functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two choised for computing the SWSH basis functions required in the transformations. One using numpy implementation and the other using sympy. In the former method, the computations are only accurate upto about 12 decimals for :math:`\\ell \\leq 20` but they are fast. In the latter, the SWSH can be computed to any precision required at the cost of speed.





