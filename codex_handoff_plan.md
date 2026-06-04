# Codex Handoff Plan

Branch: `config-optimization-fitting-factors`

Do not commit `AGENTS.md`. It is user-provided local instruction context.

## Current Goal

Implement config-based optimization and fitting-factor support for mode-space
waveform comparisons, then fix high-priority correctness issues found during a
local audit in small batches.

## Completed PRs

- PR #4 merged: fixed-frame mode comparison core API.
- PR #5 merged: explicit alignment choices.
- PR #6 merged: news-power default alignment and shared fine time-shift
  optimization.

## Current Uncommitted Work

Implemented but not committed:

- `waveformtools/comparison/config.py`
  - `ModeComparisonConfig`
  - `FittingFactorConfig`
- `waveformtools/comparison/fitting_factor.py`
  - fixed-candidate fitting factor
  - generator-backed fitting factor
  - supports `optimizer="none"`, `"scipy_minimize"`, and
    `"differential_evolution"`
  - omitted generator parameters are not passed, so model defaults apply
- `waveformtools/comparison/modes_api.py`
  - `ModesArray.fitting_factor(...)` now calls the implementation
- `waveformtools/comparison/__init__.py`
  - exports config and fitting-factor APIs
- `test/test_comparison.py`
  - fitting-factor/config tests
  - reflected subtraction regression test
- `waveformtools/modes_array.py`
  - fixed `ModesArray.__rsub__`
- `waveformtools/transforms.py`
  - fixed local `compute_ifft` time-axis construction
- `waveformtools/integrate.py`
  - made `fixed_frequency_integrator` robust for supplied FFT data
- `test/test_transforms.py`
  - local FFT/IFFT round-trip and time-axis test
- `test/test_integrate.py`
  - supplied FFT/frequency-axis tests for FFI

Untracked local files present:

- `AGENTS.md` - do not commit.
- `sess1.txt` - user/local artifact; do not touch unless asked.
- new test/config modules listed above.

## Verification Already Run

Targeted tests only; do not run expensive Ylm/Yslm tests unless explicitly
approved.

- `conda run -n codex python -m pytest test/test_comparison.py`
  - latest result: `19 passed`
- `conda run -n codex python -m pytest test/test_transforms.py`
  - latest result: `1 passed`
- `conda run -n codex python -m pytest test/test_integrate.py`
  - latest result: `3 passed`
- targeted `py_compile` checks passed for changed files.

Local lint/format tools were installed into `codex`:

- `black`
- `isort`
- `flake8`
- `pylint`

For changed comparison files, the following passed:

- `isort --check-only`
- `flake8 --max-line-length=120 --ignore=E203,W503`
- `pylint` returned `10.00/10`; it may warn that it cannot write cache under
  read-only `~/.cache`.

Do not use broad `flake8 waveformtools/modes_array.py` as a batch gate; that
file has many pre-existing lint failures unrelated to the small fixes.

## Completed Audit Batches

### Batch 1: `ModesArray.__rsub__`

Fixed:

```python
obj2._modes_data = obj.modes_data - self.modes_data
```

Added regression test:

- `test_modes_array_reflected_subtraction_uses_left_operand`

### Batch 2: local `compute_ifft`

Fixed `waveformtools.transforms.compute_ifft`:

```python
time_axis = np.arange(Nlen) * delta_t
```

Added:

- `test/test_transforms.py`

### Batch 3: `fixed_frequency_integrator`

Fixed:

- explicit `utilde_conven is None`
- require `freq_axis` when supplied FFT data is used
- copy supplied FFT data before modifying zero bin
- only set `zero_mode` if a zero-frequency bin exists

Added:

- `test/test_integrate.py`

### Batch 4: BMS angular argument bug

Fixed:

- `waveformtools/BMS.py::compute_supertransl_alpha` accepts `theta, phi` but
  previously overwrote them internally with `theta = np.pi / 2`, `phi = 0.0`.
- Removed those assignments.
- Added `test/test_bms.py`.

### Batch 5: fitting-factor result namespace

Fixed:

- `FittingFactorResult.best_parameters` currently flattens alignment nuisance
  parameters and generator parameters into one dict.
- This can collide with physical generator parameters such as
  `candidate_time_shift`, `orbital_phase`, or `phase_alignment`.
- Stores nested best parameters:

```python
best_parameters = {
    "alignment": {...},
    "generator": {...},
}
```

- Keep `candidate_generation_parameters` populated for backward readability.
- Updated tests accordingly, including a collision regression for a generator
  parameter named `candidate_time_shift`.

### Batch 6: import hygiene

Fixed:

- `waveformtools/__init__.py` prints `package_directory` on import.
- Remove import-time `print(package_directory)`.
- Added `test/test_import_hygiene.py`.

### Batch 7: displacement-memory source and strain kernel

Implemented:

- `waveformtools/memory.py`
  - opt-in displacement-memory API now computes a scalar memory source
    `|news|^2/(16*pi)` and projects it through `SphericalArray.to_modes_array`.
  - `compute_displacement_memory_from_news` maps the integrated scalar source
    to spin `-2` memory modes using a `bar_eth^2` spectral inverse.
  - The spin coefficient is taken from
    `qlmtools.spin_coefficient.analytic_spin_raise_basis_factor` so the ladder
    coefficient convention is centralized in `qlmtools`.
- `waveformtools/modes_array.py`
  - added `ModesArray.compute_displacement_memory_source`.
- `waveformtools/spherical_array.py`
  - fixed `to_modes_array` for time-dependent angular data by normalizing to
    `(time, theta, phi)` before calling `TwoDIntegral`.
  - fixed a missing local `ModesArray` import in `to_modes_array`.
- `test/test_memory.py`
  - source projection, zero-news, and spectral-inverse tests.
- `test/test_spherical_array.py`
  - regression test for projecting time-dependent angular data.
- `test/test_memory_real_balance_law.py`
  - gated real-waveform balance-law diagnostics. Skipped by default unless
    `WAVEFORMTOOLS_RUN_REAL_WAVEFORM_TESTS=1`.
  - larger diagnostics additionally require
    `WAVEFORMTOOLS_RUN_LARGE_MEMORY_TESTS=1`.

Local `bhive` diagnostics:

- Small `NRSur7dq4`, `ell_max=2`:
  - RMS residual original: `2.09180243e-02`
  - RMS residual with memory: `2.08370051e-02`
  - RMS ratio: `9.96126822e-01`
  - `l00` original: `-4.48324105e-07 - 1.54249528e-09j`
  - `l00` with memory: `-4.48324581e-07 - 1.54249613e-09j`
  - `|l00|` ratio: `1.00000106e+00`
- Full `NRSur7dq4`, `f_lower=0`, `f_ref=20Hz`, `ell_max=4`:
  - `n_times`: `6658`
  - RMS residual original: `1.77555029e-02`
  - RMS residual with memory: `1.76524554e-02`
  - RMS ratio: `9.94196309e-01`
  - `l00` original: `-4.51012650e-07 - 1.57822590e-09j`
  - `l00` with memory: `-4.51013154e-07 - 1.57822724e-09j`
  - `|l00|` ratio: `1.00000112e+00`
- `SEOBNRv5PHM`, `f_lower=15Hz`, `ell_max=5`:
  - `n_times`: `3942`
  - RMS residual original: `1.82081653e-02`
  - RMS residual with memory: `1.81103263e-02`
  - RMS ratio: `9.94626640e-01`
  - `l00` original: `-2.06979920e-08 + 2.48634526e-25j`
  - `l00` with memory: `-2.06976950e-08 - 3.76313718e-22j`
  - `|l00|` ratio: `9.99985649e-01`

Interpretation:

- Memory lowers the full RMS balance-law residual by about `0.5%` in all
  tested real-waveform cases.
- The `l=0,m=0` residual component improves for `SEOBNRv5PHM`, but is slightly
  larger for `NRSur7dq4` at the `~1e-6` relative level after the conjugation
  fix. Keep `l00` as a printed diagnostic, not a hard assertion, until the
  memory normalization and finite-time/infinite-time conventions are audited
  more deeply.
- Operator audit: the balance-law RHS applies
  `qlmtools.spin_coefficient.eth_n_modes_from_modes(news.bar(), times=2)`.
  The memory inverse must therefore assign spin `-2` memory modes from the
  complex conjugate of the integrated scalar source coefficient. A focused
  synthetic operator-response test now checks this convention.
- Source normalization is now explicit in `DisplacementMemoryConfig`:
  - `source_normalization="news_squared"` is the default memory convention and
    uses the `|N|^2` source.
  - `source_normalization="balance_law_16pi"` keeps the historical diagnostic
    `|N|^2/(16*pi)` source.
  - `source_scale` provides an additional finite multiplicative factor for
    convention diagnostics.
- Public SXS/scri comparison: SXS computes the energy-flux memory contribution
  with `0.25 * (hdot * hdot.bar).int` inside their inverse operator and an
  outer `0.5`, which algebraically gives the same spin-lowering inverse but
  with an effective `|N|^2` source rather than `|N|^2/(16*pi)`.
- Local convention diagnostics after switching the default to `news_squared`:
  - Small `NRSur7dq4`:
    - original RMS: `2.0918024325e-02`
    - `balance_law_16pi` RMS ratio: `9.9612682222e-01`
    - `news_squared` RMS ratio: `8.9601280069e-01`
  - Full `NRSur7dq4`, `f_lower=0`, `f_ref=20Hz`:
    - original RMS: `1.7755502877e-02`
    - `balance_law_16pi` RMS ratio: `9.9419630923e-01`
    - `news_squared` RMS ratio: `8.4067520326e-01`
  - `SEOBNRv5PHM`:
    - original RMS: `1.8208165332e-02`
    - `balance_law_16pi` RMS ratio: `9.9462663998e-01`
    - `news_squared` RMS ratio: `8.5339730789e-01`

## Deferred Balance-Law Minimization Design

Goal: allow a user to improve a given `ModesArray` by minimizing balance-law
violations with physically constrained perturbative waveform corrections.

Recommended parameterization:

- Use a small correction
  `h_corrected = h + delta_h`.
- Represent the correction as an electric-parity, memory-like field:
  `delta_h = bar_eth^{-2} alpha`, where `alpha(u, theta, phi)` is a real
  scalar field.
- Restrict `alpha` to low angular modes first, e.g. `ell <= 2` or `ell <= 4`.
- Use smooth low-frequency time basis functions, endpoint-controlled cumulative
  profiles, or spline knots rather than arbitrary per-sample corrections.

Recommended objective:

```text
minimize || balance_law_residual(h + delta_h) ||^2
       + lambda_size * ||delta_h||^2
       + lambda_power * max(0, Delta E_rad)^2
       + lambda_high_l * high_l_power(delta_h)
       + lambda_osc * high_frequency_power(delta_h)
```

Physical constraints:

- Do not increase radiated power unless explicitly allowed.
- Do not inject high-ell or high-frequency structure to hide balance-law
  defects.
- Keep the correction small compared with the original modes.
- Use zero-at-start or otherwise explicit endpoint/integration constants.
- Treat uncertain charges (`M_adm`, `M_final`, kick) as nuisance parameters
  only with priors, because charge errors can masquerade as waveform-memory
  defects.

Implementation sketch:

- First implement a diagnostic optimizer over low scalar source coefficients
  `alpha_lm` only, using the same `bar_eth^{-2}` inverse as the memory kernel.
- Evaluate the objective through `balance_law_chunked` to keep memory bounded.
- Start with linearized/perturbative solves around the current waveform before
  adding nonlinear optimization.
- Report the correction norm, power change, residual change per spectral mode,
  and whether any high-mode power increased.

## Later Work After Audit Batches

- Re-run targeted tests:
  - `test/test_comparison.py`
  - `test/test_transforms.py`
  - `test/test_integrate.py`
  - any new BMS/import-hygiene tests
- Show full `git diff`.
- Do not commit until explicitly asked.
- If pushing, use:

```bash
git push git@github.com-vaishakp:vaishakp/waveformtools.git <branch>:<branch>
```

because the default SSH identity previously resolved to the wrong GitHub user.

## Deferred Design Work

- BMS alignment for waveform comparisons is intentionally postponed.
  Add a design note/API sketch for a future `BMSAlignmentSpec` before
  implementing anything. The design should separate preprocessing into a common
  BMS/CoM frame from small residual nuisance optimization, and should avoid
  free supertranslation optimization as a default fitting-factor option.

## Current Design Notes

- Keep `spectools` and `qlmtools` as black-box dependencies.
- Current fitting-factor implementation stays in mode space and accepts a
  user-supplied candidate generator returning `ModesArray`.
- Omitted generator parameters are intentionally not passed, so waveform-model
  defaults can apply naturally.
- No frequency-domain matching yet. If added later, use existing
  `compute_fft`/`compute_ifft` convention helpers rather than hand-rolling
  conventions inside comparison code.
