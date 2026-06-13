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
7. Conversion to frequency domain modes.
8. All the modes data are saved as an ndarray and mostly vectorized operations are implemented.
9. Expensive operations are accelerated using numba, which also provides GPU interface and caching.
10. Extrapolation of waveforms to infinity using perturbative methods.
11. BMS transformations.
12. Centre of mass corrections.


LAL mode conventions
--------------------

LAL-backed waveform modes loaded through
``waveformtools.waveformtools.load_lal_modes_to_modes_array`` are stored in a
waveformtools convention, not as raw LAL arrays.  The loader walks the raw LAL
linked list and keeps both positive and negative :math:`m` modes.

For time-domain LAL modes, waveformtools stores
``conj(raw_lal_mode(t))``.  For frequency-domain LAL modes, waveformtools
stores ``conj(raw_lal_mode(f)) / N``, where ``N`` is the number of samples on
the two-sided LAL frequency axis.  Code that uses these stored FD modes as
raw LAL frequency-domain modes must undo the conjugation and ``1/N`` scaling.

For FD approximants such as ``IMRPhenomXPHM``, the LAL
``SimInspiralChooseFDModes`` linked-list data use a sorted two-sided frequency
axis running from negative to positive frequencies.  Positive-:math:`m` modes
carry their physical inspiral content on the negative-frequency side, while
negative-:math:`m` modes carry it on the positive-frequency side.  This is why
waveformtools reads the linked list directly: helper APIs that extract a single
mode onto a nonnegative-frequency LAL grid can miss positive-:math:`m` inspiral
content and make the mode look as if it starts near the merger/ringdown
frequency rather than at ``f_lower``.

When ``get_td_waveform_modes()`` is called for an FD approximant,
waveformtools routes through ``get_fd_waveform_modes_as_td()``.  That path
recovers raw LAL FD samples from the stored convention, applies the inverse
FFT, and conjugates the result so the returned time-domain ``ModesArray``
follows the same waveformtools TD mode convention.


Fitting factors
---------------

Importing ``waveformtools.comparison`` attaches comparison helpers to
``ModesArray`` objects, including ``match``, ``mismatch`` and
``fitting_factor``.  The fitting-factor interface accepts any waveform
generator that returns a ``ModesArray``.  Users choose the parameters to
optimize by naming them in ``FittingFactorConfig.variable_parameters`` and
supplying bounds:

.. code-block:: python

    from waveformtools.comparison import (
        AlignmentSpec,
        FittingFactorConfig,
        ModeComparisonConfig,
    )

    def candidate_generator(**parameters):
        # Convert these user-chosen parameters into the model/backend call.
        return make_candidate_modes(**parameters)

    result = target_modes.fitting_factor(
        candidate_generator,
        config=FittingFactorConfig(
            comparison=ModeComparisonConfig(
                modes=[(2, -2), (2, 2)],
                alignment=AlignmentSpec(
                    time_alignment="peak_total_news_power",
                    time_domain_policy="resample_to_reference",
                    phase_alignment="orbital_phase_and_global",
                ),
            ),
            fixed_parameters={"approximant": "IMRPhenomXPHM"},
            variable_parameters={
                "q": (1.0, 6.0),
                "chi1z": (-0.8, 0.8),
                "phi_ref": (-3.14159, 3.14159),
            },
            initial_parameters={"q": 2.0, "chi1z": 0.2, "phi_ref": 0.0},
            optimizer="scipy_minimize",
        ),
    )

``fixed_parameters`` and each optimizer trial are passed to the generator.
The parameter names are not interpreted by waveformtools, so they can describe
intrinsic parameters, extrinsic parameters, backend flags, spin-angle
coordinates, phase conventions, or any other model-specific settings.  If the
generator naturally accepts one dictionary instead of keyword arguments, set
``generator_call_style="dict"``.

Alignment and frame nuisance parameters are configured separately in
``ModeComparisonConfig``.  For example, use ``AlignmentSpec`` for common time
and phase alignment, and ``RotationSpec`` for fixed or optimized z-axis/Wigner
rotations.  The result stores the best generator parameters under
``result.best_parameters["generator"]`` and the best alignment/rotation
parameters under ``result.best_parameters["alignment"]``.


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

Presently, the supported grids are

1. Gauss-Legendre grid.
2. Uniformly spaced grid in :math:`\theta, \phi`.



Interface to `waveformtools.waveformtools`
------------------------------------------

The `waveformtools.waveformtools` module is a toolkit to carryout various transformations of the waveforms. Please see its description file for further details.
