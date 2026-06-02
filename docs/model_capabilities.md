# Waveform model capabilities

This document summarizes the current waveform-generation capabilities exposed through
`waveformtools.models`, based on the implementation in:

- `waveformtools/models/waveform_models.py`
- `waveformtools/models/lal.py`
- `waveformtools/models/eob.py`
- `waveformtools/models/utils.py`

The summary is intentionally implementation-facing: it records what the code appears
to support today, and where the API should be tightened.

## Implemented model backends

### `LALWaveformModel`

`LALWaveformModel` wraps LALSimulation waveform-generation routines. It initializes a
LAL approximant from the string stored in `parameters_dict["approximant"]`, constructs
and stores a LAL dictionary, and applies special handling for selected models such as
`IMRPhenomXPHM` and `NR_hdf5`.

Supported LAL-related functionality:

- time-domain plus/cross polarizations via `SimInspiralChooseTDWaveform`;
- frequency-domain plus/cross polarizations via `SimInspiralFD`;
- time-domain modes via `SimInspiralChooseTDModes` when the approximant is treated as
  time-domain;
- frequency-domain modes via `SimInspiralChooseFDModes` when the approximant is
  treated as frequency-domain;
- loading LAL mode linked lists into `ModesArray` through
  `load_lal_modes_to_modes_array`;
- limited special-case support for `NR_hdf5`, including insertion of the NR file path
  into the LAL dictionary and spin loading from the HDF5 metadata;
- special PhenomX configuration knobs for `IMRPhenomXPHM`, namely
  `PhenomXHMReleaseVersion` and `PhenomXPrecVersion`.

The explicit approximant-domain classifier currently treats:

- `NRSur7dq4` and `SEOBNRv5PHM` as time-domain;
- `IMRPhenomXPHM` as frequency-domain;
- all other approximants as time-domain by default.

There is also an automatic classifier using PyCBC's `td_approximants()` and
`fd_approximants()`, but the mode-generation methods currently use the explicit
classifier.

### `EOBWaveformModel`

`EOBWaveformModel` wraps `pyseobnr` mode generation. It supports:

- time-domain mode generation through `pyseobnr.generate_waveform.generate_modes_opt`;
- conversion of the returned EOB mode dictionary into a `ModesArray`;
- passing a deviation/settings dictionary to pySEOBNR for parameterized-deviation or
  pseudo-EOB studies;
- returning dimensional or dimensionless time-domain modes;
- a `get_td_waveform` helper that currently delegates to PyCBC's
  `get_td_waveform` rather than projecting the internally generated EOB modes.

The EOB backend currently does not expose a native frequency-domain mode generator.

## Output capability matrix

| Capability | `LALWaveformModel` | `EOBWaveformModel` | Notes |
|---|---:|---:|---|
| TD plus/cross polarizations | Yes | Partial | LAL uses `SimInspiralChooseTDWaveform`. EOB has `get_td_waveform`, but it delegates to PyCBC rather than using `td_waveform_modes.to_td_waveform`. |
| FD plus/cross polarizations | Yes | No | LAL uses `SimInspiralFD`. EOB has no native FD polarization method. |
| TD modes | Yes | Yes | LAL uses `SimInspiralChooseTDModes`; EOB uses `generate_modes_opt` and converts to `ModesArray`. |
| FD modes | Yes, for FD LAL models | No | LAL uses `SimInspiralChooseFDModes`. EOB does not provide native FD modes. |
| Convert FD modes to TD modes | Yes, in `get_td_waveform_modes` | No direct FD path | For FD-labelled LAL modes, `get_td_waveform_modes` converts the `ModesArray` to the time basis via `to_time_basis()`. |
| Detector projections | Yes, TD only | Inherited, TD only | The base class provides `project_polarizations(hp, hc, extrinsic_parameters, detector_string)` using PyCBC `Detector`. It expects TD `hp`, `hc` arrays and uses `delta_t`. |
| NR HDF5 model path | Yes | No | LAL supports `NR_hdf5` with `lvcnr_file_path` inserted into the LAL dictionary. |
| Dimensionless modes | Yes, TD route | Yes | The base class can non-dimensionalize/dimensionalize TD modes using total mass and distance. |
| Balance-law diagnostics | Inherited | Inherited | The base class has infinite-time balance-law utilities that consume TD modes and construct a corresponding EOB Hamiltonian. |

## Answers to the specific questions

### Can it generate modes?

Yes.

- `LALWaveformModel.get_td_waveform_modes()` generates modes using LALSimulation.
  Depending on the approximant classification, it calls either
  `SimInspiralChooseTDModes` or `SimInspiralChooseFDModes`, then loads the result into
  a `ModesArray`.
- `LALWaveformModel.get_fd_waveform_modes()` directly calls
  `SimInspiralChooseFDModes`.
- `EOBWaveformModel.get_td_waveform_modes()` generates TD modes through pySEOBNR and
  returns a `ModesArray`.

### Can it generate FD and TD modes?

For LAL-backed models: yes, with caveats.

- TD modes are supported through `get_td_waveform_modes()`.
- FD modes are supported through `get_fd_waveform_modes()`.
- For FD approximants, the TD-mode route can generate FD modes and convert them into a
  time-basis `ModesArray` using `to_time_basis()`.

For EOB-backed models: TD modes are supported; FD modes are not currently exposed.

### Can it generate projections?

Yes, but only for time-domain plus/cross arrays at present.

The base `WaveformModel.project_polarizations()` method computes the detector antenna
response using PyCBC's `Detector`, aligns the time array to the polarization-amplitude
peak, applies `t_coal`, and returns `(det_times, h_inj)` with

```python
h_inj = Fp * hp + Fc * hc
```

The current projection method does not directly consume `ModesArray` objects, does not
return a PyCBC `TimeSeries`, and does not implement an FD projection path.

### Can it generate polarizations?

Yes for LAL-backed models.

- `LALWaveformModel.get_td_waveform()` returns TD `(hp, hc)` arrays.
- `LALWaveformModel.get_fd_waveform()` returns FD `(hp, hc)` arrays.

For EOB-backed models, polarization support is incomplete/indirect:

- `EOBWaveformModel.get_td_waveform_modes()` provides modes.
- `EOBWaveformModel.get_td_waveform()` currently delegates to PyCBC
  `get_td_waveform()` with the parameters, instead of converting the internally
  generated EOB modes into plus/cross polarizations.
- There is no EOB `get_fd_waveform()` implementation.

### Can it generate TD and FD polarizations?

For LAL-backed models: yes.

For EOB-backed models: TD polarizations are only partial/indirect, and FD
polarizations are not currently implemented.

## Suggested improvements

### 1. Make the domain model explicit

The code currently has both `get_approximant_type_auto()` and a hard-coded
`get_approximant_type()` list. The hard-coded version is used by the mode-generation
methods and defaults unknown approximants to `td`.

Recommended change:

- make domain detection explicit and centralized;
- prefer PyCBC/LAL approximant registries when possible;
- fail loudly for unknown approximants instead of silently defaulting to TD;
- store the chosen domain as `self.domain` or derive it from a small registry.

### 2. Separate four output concepts cleanly

A clean public API would expose exactly these methods with consistent return types:

```python
get_td_modes(...)
get_fd_modes(...)
get_td_polarizations(...)
get_fd_polarizations(...)
project_td_polarizations(...)
project_fd_polarizations(...)
```

The present names are close, but the behavior is uneven across backends. In particular,
`EOBWaveformModel.get_td_waveform()` does not currently use the internally generated EOB
modes.

### 3. Convert EOB modes to polarizations directly

For the EOB backend, implement a direct route:

```python
td_modes -> h_plus(t), h_cross(t)
```

using the existing `ModesArray` machinery. This would make EOB behavior parallel to the
LAL backend and avoid delegating `EOBWaveformModel.get_td_waveform()` to PyCBC.

### 4. Add FD projection support

The base projection method is TD-only. A useful extension would be:

```python
project_fd_polarizations(hp_f, hc_f, extrinsic_parameters, detector_string)
```

using frequency-domain antenna factors when the sky location and polarization angle are
constant over the segment, or explicitly documenting the assumptions.

### 5. Normalize return types

At present, low-level LAL calls return raw NumPy arrays, while mode-generation routes
return `ModesArray`. It would be useful to define and document return-type conventions,
for example:

- raw arrays for low-level wrappers;
- `ModesArray` for mode outputs;
- optionally PyCBC `TimeSeries` / `FrequencySeries` wrappers for direct data-analysis
  use.

### 6. Improve parameter validation

The base class directly sets every key in `parameters_dict` as an attribute. This is
convenient, but fragile.

Recommended additions:

- required-parameter checks for TD, FD, modes, polarizations, and projections;
- clearer error messages for missing `delta_t`, `delta_f`, `f_lower`, `f_max`,
  `f_ref`, `distance`, and angular parameters;
- validation of mass naming (`mass1`/`mass2` vs `mass_1`/`mass_2`).

### 7. Avoid mutable default arguments

Several constructors use default dictionaries such as `parameters_dict={}` and
`deviation_dict={}`. These should be replaced with `None` defaults and initialized
inside the function body.

Recommended pattern:

```python
def __init__(self, parameters_dict=None, deviation_dict=None):
    if parameters_dict is None:
        parameters_dict = {}
    if deviation_dict is None:
        deviation_dict = {}
```

### 8. Add smoke tests for the capability matrix

Add small tests that verify:

- LAL TD polarizations are generated for a lightweight TD approximant;
- LAL FD polarizations are generated for a lightweight FD approximant;
- LAL TD modes return a populated `ModesArray`;
- LAL FD modes return a populated `ModesArray`;
- EOB TD modes return a populated `ModesArray` when pySEOBNR is installed;
- TD projection returns an array with the expected length.

The EOB tests should be optional/skipped when `pyseobnr` is unavailable.

### 9. Document backend dependencies

The model layer depends on optional heavy packages:

- `lalsimulation` / `lal`;
- `pycbc`;
- `pyseobnr`;
- `spectools` for some mode-grid utilities and balance-law computations.

The docs should state which capabilities require which backend dependencies.
