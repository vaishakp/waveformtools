# Performance Notes

This file records small, curated profiling results that are useful for future
optimization work. Do not commit large profiler outputs or flamegraph artifacts
unless they are intentionally reduced and documented.

## Mode-Comparison Phase Optimization

Commit: `44089bc` (`Vectorize mode overlap and hoist phase-independent integrals`)

Context:

- Nonspinning waveform comparison/profile run.
- Specs referenced in the profiling run: `spec1.0`, `spec1.1`, `spec1.2`.
- The profile was used to understand bottlenecks in fixed-frame mismatch and
  fitting-factor evaluations.

Bottleneck:

- The orbital-phase optimizer repeatedly recomputed time-domain inner products.
- The expensive part was repeatedly evaluating trapezoidal integrals of
  `conj(a_lm(t)) * b_lm(t)` for every trial orbital phase.

Optimization:

- Precompute per-mode overlaps once:

  ```text
  J_lm = integral conj(a_lm(t)) b_lm(t) dt
  ```

- Reuse them during orbital-phase search:

  ```text
  <a,b>(phi) = sum_lm exp(i m phi) J_lm
  ```

- Vectorize the per-mode overlap and norm integrations over modes.

Observed result:

- Evaluation time: `12.23 s -> 5.20 s`.
- Overall speedup: about `2.35x`.
- `spec1.0` and `spec1.1` speedup: about `3x`.
- Mismatches and objective-evaluation counts were unchanged.

## Expensive Tests

The expensive real-waveform tests are skipped by default. Run them explicitly
when profiling or validating changes that affect waveform generation,
comparison, memory, or balance-law code.

Real waveform comparison tests:

```bash
NUMBA_CACHE_DIR=/tmp/waveformtools_numba_cache \
WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 \
pytest test/test_comparison_real_waveforms.py -vv
```

Real waveform balance-law tests:

```bash
NUMBA_CACHE_DIR=/tmp/waveformtools_numba_cache \
WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 \
pytest test/test_memory_real_balance_law.py -vv -s
```

Large balance-law cases:

```bash
NUMBA_CACHE_DIR=/tmp/waveformtools_numba_cache \
WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1 \
WAVEFORMTOOLS_RUN_LARGE_MEMORY_TESTS=1 \
pytest test/test_memory_real_balance_law.py -vv -s
```
