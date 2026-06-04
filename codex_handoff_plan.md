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

## Remaining Batches

### Batch 4: BMS angular argument bug

Issue:

- `waveformtools/BMS.py::compute_supertransl_alpha` accepts `theta, phi` but
  overwrites them with:

```python
theta = np.pi / 2
phi = 0.0
```

Plan:

- Remove those assignments.
- Add a focused test proving different angular inputs produce expected values.
- Use existing public function behavior only; do not inspect black-box deps.

### Batch 5: fitting-factor result namespace

Issue:

- `FittingFactorResult.best_parameters` currently flattens alignment nuisance
  parameters and generator parameters into one dict.
- This can collide with physical generator parameters such as
  `candidate_time_shift`, `orbital_phase`, or `phase_alignment`.

Plan:

- Store nested best parameters:

```python
best_parameters = {
    "alignment": {...},
    "generator": {...},
}
```

- Keep `candidate_generation_parameters` populated for backward readability.
- Update tests accordingly.

### Batch 6: import hygiene

Issue:

- `waveformtools/__init__.py` prints `package_directory` on import.

Plan:

- Remove import-time `print(package_directory)`.
- Consider converting `read_git_version()` fallback print to a warning or quiet
  fallback, but keep that as a separate tiny edit if possible.

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

## Current Design Notes

- Keep `spectools` and `qlmtools` as black-box dependencies.
- Current fitting-factor implementation stays in mode space and accepts a
  user-supplied candidate generator returning `ModesArray`.
- Omitted generator parameters are intentionally not passed, so waveform-model
  defaults can apply naturally.
- No frequency-domain matching yet. If added later, use existing
  `compute_fft`/`compute_ifft` convention helpers rather than hand-rolling
  conventions inside comparison code.
