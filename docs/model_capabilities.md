# Waveform model capabilities

This document summarizes the current waveform-generation capabilities exposed through
`waveformtools.models`, based on the implementation in:

- `waveformtools/models/waveform_models.py`
- `waveformtools/models/lal.py`
- `waveformtools/models/eob.py`
- `waveformtools/models/utils.py`

The summary is implementation-facing: it records what the code supports today, the
public methods that expose each output type, and remaining API gaps.

## Implemented model backends

### `LALWaveformModel`

`LALWaveformModel` wraps LALSimulation waveform-generation routines. It initializes a
LAL approximant from the string stored in `parameters_dict["approximant"]`, constructs
and stores a LAL dictionary, and applies special handling for selected models such as
`IMRPhenomXPHM` and `NR_hdf5`.

Supported LAL-related functionality:

- time-domain plus/cross polarizations via `SimInspiralChooseTDWaveform`;
- frequency-domain plus/cross polarizations via `SimInspiralFD`;
- time-domain modes via `SimInspiralChooseTDModes`;
- frequency-domain modes via `SimInspiralChooseFDModes`;
- an explicit FD-modes-as-TD route through `get_fd_waveform_modes_as_td()`;
- public aliases with clearer output names:
  - `get_td_modes()`;
  - `get_fd_modes()`;
  - `get_fd_modes_as_td()`;
  - `get_td_polarizations()`;
  - `get_fd_polarizations()`;
  - `project_td_polarizations()`;
- loading LAL mode linked lists into `ModesArray` through
  `load_lal_modes_to_modes_array`;
- limited special-case support for `NR_hdf5`, including insertion of the NR file path
  into the LAL dictionary and spin loading from the HDF5 metadata;
- special PhenomX configuration knobs for `IMRPhenomXPHM`, namely
  `PhenomXHMReleaseVersion` and `PhenomXPrecVersion`.

Approximant-domain handling is centralized in `get_approximant_domain()`. The explicit
registry currently records:

- `NRSur7dq4` and `SEOBNRv5PHM` as time-domain;
- `IMRPhenomXPHM` as frequency-domain.

For other approximants, the code falls back to PyCBC's `td_approximants()` and
`fd_approximants()` registries. Unknown approximants raise a `KeyError` rather than
silently defaulting to the time domain.

### `EOBWaveformModel`

`EOBWaveformModel` wraps `pyseobnr` mode generation. It supports:

- time-domain mode generation through `pyseobnr.generate_waveform.generate_modes_opt`;
- conversion of the returned EOB mode dictionary into a `ModesArray`;
- passing a deviation/settings dictionary to pySEOBNR for parameterized-deviation or
  pseudo-EOB studies;
- returning dimensional or dimensionless time-domain modes;
- a `capabilities()` method;
- a `get_td_modes()` alias for `get_td_waveform_modes()`;
- a `get_td_waveform` helper that currently delegates to PyCBC's
  `get_td_waveform` rather than projecting the internally generated EOB modes.

The EOB backend currently does not expose a native frequency-domain mode generator.

## Output capability matrix

| Capability | `LALWaveformModel` | `EOBWaveformModel` | Notes |
|---|---:|---:|---|
| TD plus/cross polarizations | Yes | Partial | LAL uses `SimInspiralChooseTDWaveform`. EOB has `get_td_waveform`, but it delegates to PyCBC rather than using the internally generated modes. |
| FD plus/cross polarizations | Yes | No | LAL uses `SimInspiralFD`. EOB has no native FD polarization method. |
| TD modes | Yes | Yes | LAL uses `SimInspiralChooseTDModes`; EOB uses `generate_modes_opt` and converts to `ModesArray`. |
| FD modes | Yes | No | LAL uses `SimInspiralChooseFDModes`. EOB does not provide native FD modes. |
| FD modes returned as TD modes | Yes | No | LAL exposes `get_fd_waveform_modes_as_td()` and alias `get_fd_modes_as_td()`. This generates FD modes, loads them as a frequency-basis `ModesArray`, converts with `to_time_basis()`, and optionally non-dimensionalizes the TD result. |
| Detector projections | Yes, TD only | Inherited, TD only | The base class provides `project_polarizations(hp, hc, extrinsic_parameters, detector_string)` using PyCBC `Detector`. LAL also exposes `project_td_polarizations()`. |
| NR HDF5 model path | Yes | No | LAL supports `NR_hdf5` with `lvcnr_file_path` inserted into the LAL dictionary. |
| Dimensionless modes | Yes, TD route | Yes | The base class can non-dimensionalize/dimensionalize TD modes using total mass and distance. |
| Capability introspection | Yes | Yes | Both backends expose `capabilities()`. |
| Balance-law diagnostics | Inherited | Inherited | The base class has infinite-time balance-law utilities that consume TD modes and construct a corresponding EOB Hamiltonian. |

## Current public methods by output type

### LAL backend

```python
model.get_td_waveform_modes(...)
model.get_fd_waveform_modes(...)
model.get_fd_waveform_modes_as_td(...)

model.get_td_modes(...)
model.get_fd_modes(...)
model.get_fd_modes_as_td(...)

model.get_td_waveform(...)
model.get_fd_waveform(...)
model.get_td_polarizations(...)
model.get_fd_polarizations(...)

model.project_td_polarizations(hp, hc, extrinsic_parameters, detector_string)
model.capabilities()
```

### EOB backend

```python
model.get_td_waveform_modes(...)
model.get_td_modes(...)
model.get_td_waveform(...)
model.capabilities()
```

## Answers to common capability questions

### Can it generate modes?

Yes.

- `LALWaveformModel.get_td_waveform_modes()` generates TD modes for TD approximants.
- `LALWaveformModel.get_fd_waveform_modes()` generates FD modes.
- `LALWaveformModel.get_fd_waveform_modes_as_td()` generates FD modes and returns them
  in the same TD `ModesArray` convention used by the existing FD-to-TD path.
- `EOBWaveformModel.get_td_waveform_modes()` generates TD modes through pySEOBNR and
  returns a `ModesArray`.

### Can it generate FD and TD modes?

For LAL-backed models: yes.

- TD modes are supported through `get_td_waveform_modes()` and `get_td_modes()`.
- FD modes are supported through `get_fd_waveform_modes()` and `get_fd_modes()`.
- FD modes returned as TD modes are supported through `get_fd_waveform_modes_as_td()`
  and `get_fd_modes_as_td()`.

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

- `LALWaveformModel.get_td_waveform()` and `get_td_polarizations()` return TD
  `(hp, hc)` arrays.
- `LALWaveformModel.get_fd_waveform()` and `get_fd_polarizations()` return FD
  `(hp, hc)` arrays.

For EOB-backed models, polarization support is incomplete/indirect:

- `EOBWaveformModel.get_td_waveform_modes()` provides modes.
- `EOBWaveformModel.get_td_waveform()` currently delegates to PyCBC
  `get_td_waveform()` with the parameters, instead of converting the internally
  generated EOB modes into plus/cross polarizations.
- There is no EOB `get_fd_waveform()` implementation.

## Remaining suggested improvements

### 1. Convert EOB modes to polarizations directly

For the EOB backend, implement a direct route:

```python
td_modes -> h_plus(t), h_cross(t)
```

using the existing `ModesArray` machinery. This would make EOB behavior parallel to the
LAL backend and avoid delegating `EOBWaveformModel.get_td_waveform()` to PyCBC.

### 2. Add FD projection support

The base projection method is TD-only. A useful extension would be:

```python
project_fd_polarizations(hp_f, hc_f, extrinsic_parameters, detector_string)
```

using frequency-domain antenna factors when the sky location and polarization angle are
constant over the segment, or explicitly documenting the assumptions.

### 3. Normalize return types

At present, low-level LAL calls return raw NumPy arrays, while mode-generation routes
return `ModesArray`. It would be useful to define and document return-type conventions,
for example:

- raw arrays for low-level wrappers;
- `ModesArray` for mode outputs;
- optionally PyCBC `TimeSeries` / `FrequencySeries` wrappers for direct data-analysis
  use.

### 4. Improve parameter validation

The base class directly sets every key in `parameters_dict` as an attribute. This is
convenient, but fragile.

Recommended additions:

- required-parameter checks for TD, FD, modes, polarizations, and projections;
- clearer error messages for missing `delta_t`, `delta_f`, `f_lower`, `f_max`,
  `f_ref`, `distance`, and angular parameters;
- validation of mass naming (`mass1`/`mass2` vs `mass_1`/`mass_2`).

### 5. Add smoke tests for the capability matrix

Add small tests that verify:

- LAL TD polarizations are generated for a lightweight TD approximant;
- LAL FD polarizations are generated for a lightweight FD approximant;
- LAL TD modes return a populated `ModesArray`;
- LAL FD modes return a populated `ModesArray`;
- LAL FD modes returned as TD modes return a populated TD `ModesArray`;
- EOB TD modes return a populated `ModesArray` when pySEOBNR is installed;
- TD projection returns an array with the expected length.

The EOB tests should be optional/skipped when `pyseobnr` is unavailable.

### 6. Document backend dependencies

The model layer depends on optional heavy packages:

- `lalsimulation` / `lal`;
- `pycbc`;
- `pyseobnr`;
- `spectools` for some mode-grid utilities and balance-law computations.

The docs should state which capabilities require which backend dependencies.
