# Comparison Alignment Manifest

This note records the intended semantics of waveformtools mode-space
comparison and fitting-factor alignment.  It is meant to be stable enough for
other repositories to review against.

## Inner Product And Match

For selected modes on a common time axis, waveformtools uses the unweighted
mode-space time-domain inner product

```text
<a,b> = integral dt sum_lm conj(a_lm(t)) b_lm(t).
```

The normalized match is

```text
match = phase_aligned_inner / (norm_a * norm_b),
norm_a = sqrt(<a,a>),
norm_b = sqrt(<b,b>).
```

This is not a detector PSD-weighted gravitational-wave overlap.  It is a
mode-space comparison objective.

## Replay Convention

The candidate waveform alignment parameters returned by `mode_match` must be
replayable.  After any configured rotation, the candidate is interpreted as

```text
t_candidate_aligned =
    t_candidate - reference_time_candidate + candidate_time_shift

h_lm_candidate_aligned =
    exp(i * global_phase) * exp(i * m * orbital_phase) * h_lm_candidate.
```

The sign convention is:

- positive `candidate_time_shift` shifts the candidate time axis forward after
  reference-time alignment;
- `orbital_phase` applies `exp(i * m * orbital_phase)` to each candidate mode;
- `global_phase` applies `exp(i * global_phase)` to every candidate mode.

Any plotting, residue, or downstream diagnostic code should use
`prepare_aligned_mode_data` or an equivalent replay implementation rather than
manually applying only one of the returned parameters.

## Phase Alignment Modes

The value of `phase_aligned_inner` depends on `phase_alignment`.

### `phase_alignment="none"`

No phase freedom is optimized.

```text
orbital_phase = 0
global_phase = 0
phase_aligned_inner = real(raw_inner)
```

### `phase_alignment="global_complex"`

Only one global complex phase is optimized.

```text
orbital_phase = 0
global_phase = -arg(raw_inner)
phase_aligned_inner = abs(raw_inner)
```

This is equivalent to applying `global_phase` to the candidate and then taking
the real inner product.

### `phase_alignment="orbital_phase"`

The orbital phase is optimized, but no additional global phase is applied.

```text
orbital_phase = argmax_phi real(inner(phi))
global_phase = 0
phase_aligned_inner = real(inner(orbital_phase))
```

The objective is the real inner product after the orbital-phase transform.

### `phase_alignment="orbital_phase_and_global"`

The orbital phase is optimized while also allowing a global complex phase.

```text
orbital_phase = argmax_phi abs(inner(phi))
global_phase = -arg(inner(orbital_phase))
phase_aligned_inner = abs(inner(orbital_phase))
```

This is equivalent to applying both returned phases to the candidate and then
taking the real inner product.  Because the global phase is part of the
alignment freedom, any outer optimizer over time shift or waveform parameters
must use `abs(inner)` after orbital-phase optimization for this mode, not
`real(inner)` before applying `global_phase`.

## Degeneracies

For a single dominant mode, especially an `h22`-only comparison,
`orbital_phase` and `global_phase` can be degenerate because

```text
exp(i * global_phase) * exp(i * m * orbital_phase)
```

can represent the same total phase in many ways.  In degenerate cases,
`orbital_phase_and_global` should prefer a replay-stable convention:

```text
orbital_phase = 0
global_phase = -arg(raw_inner)
```

and record a diagnostic flag.

Z-axis rotations are also m-dependent phase transformations.  Simultaneously
optimizing z-axis-like rotations and `orbital_phase` is degenerate unless the
caller explicitly opts into that redundancy.

## Time-Shift Optimization

The exact/default time-shift optimizer should preserve the same mathematical
objective as `mode_match` at a fixed `candidate_time_shift`.

For each candidate shift, it should:

1. apply the documented time alignment convention;
2. form the common time grid according to the configured time-domain policy;
3. resample or crop according to that policy;
4. evaluate the selected phase-alignment objective;
5. return `1 - clipped_match` as the scalar minimization objective.

For `phase_alignment="orbital_phase_and_global"`, the time-shift objective must
use the absolute inner product after orbital-phase optimization, because global
phase is also optimized.  For `phase_alignment="orbital_phase"`, it must use
the real inner product after orbital-phase optimization.

Any faster time-shift method must be tested against the exact/default method
for:

- best match and mismatch;
- best `candidate_time_shift`;
- best `orbital_phase` and `global_phase`;
- replayed aligned arrays from `prepare_aligned_mode_data`;
- normalized RMS residue before and after alignment.

Approximate methods such as roll-based or FFT-correlation based shifts should
be clearly labeled as approximate unless they are proven to reproduce the exact
configured resampling/cropping objective within documented tolerances.
