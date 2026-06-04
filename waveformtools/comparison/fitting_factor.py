"""Fitting-factor helpers built on top of mode-space comparisons."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np

from waveformtools.comparison.config import (
    FittingFactorConfig,
    ModeComparisonConfig,
)
from waveformtools.comparison.core import mode_match
from waveformtools.comparison.results import (
    ComparisonResult,
    FittingFactorResult,
)

CandidateGenerator = Callable[..., Any]


def fixed_candidate_fitting_factor(
    target_modes: Any,
    candidate_modes: Any,
    *,
    config: ModeComparisonConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> FittingFactorResult:
    """Evaluate the fitting factor for one already-generated candidate."""

    comparison = ModeComparisonConfig.from_value(config, **overrides)
    match_result = mode_match(
        target_modes,
        candidate_modes,
        ell_min=comparison.ell_min,
        ell_max=comparison.ell_max,
        modes=comparison.modes,
        alignment=comparison.alignment,
    )
    return _result_from_match(
        match_result,
        objective_name="fixed_candidate_fitting_factor",
        optimizer=match_result.optimizer,
        fixed_parameters={},
        candidate_generation_parameters={},
        n_waveform_generations=0,
        diagnostics={
            **match_result.diagnostics,
            "comparison_config": comparison.to_dict(),
        },
    )


def fitting_factor(  # pylint: disable=too-many-locals
    target_modes: Any,
    candidate_generator: CandidateGenerator,
    *,
    config: FittingFactorConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> FittingFactorResult:
    """Optimize a generator-backed candidate family against target modes.

    The generator is called with ``fixed_parameters`` plus the current optimizer
    trial values.  Omitted parameters are not passed, so normal Python/model
    defaults apply naturally.
    """

    t0 = time.perf_counter()
    ff_config = FittingFactorConfig.from_value(config, **overrides)
    parameter_names = list(ff_config.variable_parameters)
    n_generations = 0
    best_match: ComparisonResult | None = None
    best_parameters: dict[str, Any] | None = None

    def evaluate(parameters: Mapping[str, Any]) -> ComparisonResult:
        nonlocal n_generations, best_match, best_parameters
        candidate_parameters = _candidate_parameters(ff_config, parameters)
        candidate_modes = _call_generator(
            candidate_generator, candidate_parameters, ff_config
        )
        n_generations += 1
        result = mode_match(
            target_modes,
            candidate_modes,
            ell_min=ff_config.comparison.ell_min,
            ell_max=ff_config.comparison.ell_max,
            modes=ff_config.comparison.modes,
            alignment=ff_config.comparison.alignment,
        )
        if best_match is None or _match_value(result) > _match_value(
            best_match
        ):
            best_match = result
            best_parameters = dict(candidate_parameters)
        return result

    if not parameter_names:
        final_match = evaluate({})
        optimizer_name = "none"
        optimizer_status = "ok"
        optimizer_diagnostics: dict[str, Any] = {
            "message": "No variable parameters."
        }
        n_objective_evaluations = 1
    elif ff_config.optimizer == "none":
        trial = _initial_parameter_values(ff_config)
        final_match = evaluate(trial)
        optimizer_name = "none"
        optimizer_status = "ok"
        optimizer_diagnostics = {
            "message": "Evaluated initial parameter values only."
        }
        n_objective_evaluations = 1
    elif ff_config.optimizer == "scipy_minimize":
        final_match, optimizer_diagnostics, n_objective_evaluations = (
            _run_scipy_minimize(
                evaluate,
                ff_config,
                parameter_names,
            )
        )
        optimizer_name = "scipy_minimize"
        optimizer_status = (
            "ok" if optimizer_diagnostics["success"] else "failed"
        )
    else:
        final_match, optimizer_diagnostics, n_objective_evaluations = (
            _run_differential_evolution(
                evaluate,
                ff_config,
                parameter_names,
            )
        )
        optimizer_name = "differential_evolution"
        optimizer_status = (
            "ok" if optimizer_diagnostics["success"] else "failed"
        )

    if best_match is None or best_parameters is None:
        best_match = final_match
        best_parameters = _candidate_parameters(
            ff_config, _initial_parameter_values(ff_config)
        )

    diagnostics = {
        **best_match.diagnostics,
        "fitting_factor_config": ff_config.to_dict(),
        "optimizer_diagnostics": optimizer_diagnostics,
    }
    elapsed = time.perf_counter() - t0
    return _result_from_match(
        best_match,
        objective_name="fitting_factor",
        optimizer=optimizer_name,
        optimizer_status=optimizer_status,
        fixed_parameters=dict(ff_config.fixed_parameters),
        candidate_generation_parameters=best_parameters,
        n_objective_evaluations=n_objective_evaluations,
        n_waveform_generations=n_generations,
        elapsed_s=elapsed,
        diagnostics=diagnostics,
    )


def _run_scipy_minimize(  # pylint: disable=import-outside-toplevel
    evaluate: Callable[[Mapping[str, Any]], ComparisonResult],
    config: FittingFactorConfig,
    parameter_names: list[str],
) -> tuple[ComparisonResult, dict[str, Any], int]:
    try:
        from scipy.optimize import (
            minimize,
        )
    except Exception as exc:  # pragma: no cover - scipy is normally present
        raise ImportError(
            "optimizer='scipy_minimize' requires scipy.optimize.minimize"
        ) from exc

    bounds = [config.variable_parameters[name] for name in parameter_names]
    x0 = np.array(
        [_initial_value(config, name) for name in parameter_names], dtype=float
    )
    n_evaluations = 0
    last_result: ComparisonResult | None = None

    def objective(values: np.ndarray) -> float:
        nonlocal n_evaluations, last_result
        n_evaluations += 1
        parameters = _array_to_parameters(parameter_names, values)
        try:
            last_result = evaluate(parameters)
        except ValueError:
            return np.inf
        match = _match_value(last_result)
        if not np.isfinite(match):
            return np.inf
        return 1.0 - float(np.clip(match, -1.0, 1.0))

    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method=str(config.optimizer_options.get("method", "L-BFGS-B")),
        options=dict(config.optimizer_options.get("options", {})),
    )
    if last_result is None:
        last_result = evaluate(_array_to_parameters(parameter_names, result.x))
        n_evaluations += 1
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "best_mismatch": float(result.fun),
        "parameter_names": parameter_names,
        "bounds": bounds,
        "x": [float(value) for value in result.x],
        "n_evaluations": int(n_evaluations),
    }
    return last_result, diagnostics, n_evaluations


def _run_differential_evolution(  # pylint: disable=import-outside-toplevel
    evaluate: Callable[[Mapping[str, Any]], ComparisonResult],
    config: FittingFactorConfig,
    parameter_names: list[str],
) -> tuple[ComparisonResult, dict[str, Any], int]:
    try:
        from scipy.optimize import (
            differential_evolution,
        )
    except Exception as exc:  # pragma: no cover - scipy is normally present
        raise ImportError(
            "optimizer='differential_evolution' requires scipy.optimize.differential_evolution"
        ) from exc

    bounds = [config.variable_parameters[name] for name in parameter_names]
    n_evaluations = 0
    last_result: ComparisonResult | None = None

    def objective(values: np.ndarray) -> float:
        nonlocal n_evaluations, last_result
        n_evaluations += 1
        parameters = _array_to_parameters(parameter_names, values)
        try:
            last_result = evaluate(parameters)
        except ValueError:
            return np.inf
        match = _match_value(last_result)
        if not np.isfinite(match):
            return np.inf
        return 1.0 - float(np.clip(match, -1.0, 1.0))

    options = dict(config.optimizer_options)
    result = differential_evolution(objective, bounds=bounds, **options)
    if last_result is None:
        last_result = evaluate(_array_to_parameters(parameter_names, result.x))
        n_evaluations += 1
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "best_mismatch": float(result.fun),
        "parameter_names": parameter_names,
        "bounds": bounds,
        "x": [float(value) for value in result.x],
        "n_evaluations": int(n_evaluations),
    }
    return last_result, diagnostics, n_evaluations


def _result_from_match(  # pylint: disable=too-many-arguments
    match_result: ComparisonResult,
    *,
    objective_name: str,
    optimizer: str | None,
    fixed_parameters: dict[str, Any],
    candidate_generation_parameters: dict[str, Any],
    n_waveform_generations: int,
    diagnostics: dict[str, Any],
    optimizer_status: str = "ok",
    n_objective_evaluations: int | None = None,
    elapsed_s: float | None = None,
) -> FittingFactorResult:
    best_parameters = {
        "alignment": dict(match_result.best_parameters),
        "generator": dict(candidate_generation_parameters),
    }
    return FittingFactorResult(
        objective_name=objective_name,
        match=match_result.match,
        mismatch=match_result.mismatch,
        best_parameters=best_parameters,
        fixed_parameters=fixed_parameters,
        optimizer=optimizer,
        optimizer_status=optimizer_status,
        n_objective_evaluations=(
            match_result.n_objective_evaluations
            if n_objective_evaluations is None
            else n_objective_evaluations
        ),
        n_waveform_generations=n_waveform_generations,
        elapsed_s=match_result.elapsed_s if elapsed_s is None else elapsed_s,
        diagnostics=diagnostics,
        target_metadata=match_result.target_metadata,
        candidate_metadata=match_result.candidate_metadata,
        candidate_approximant=getattr(
            match_result.candidate_metadata, "approximant", None
        ),
        candidate_generation_parameters=candidate_generation_parameters,
    )


def _candidate_parameters(
    config: FittingFactorConfig,
    variable_values: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = dict(config.fixed_parameters)
    parameters.update(
        {str(name): value for name, value in variable_values.items()}
    )
    return parameters


def _call_generator(
    generator: CandidateGenerator,
    parameters: Mapping[str, Any],
    config: FittingFactorConfig,
) -> Any:
    if config.generator_call_style == "dict":
        return generator(dict(parameters))
    return generator(**parameters)


def _initial_parameter_values(config: FittingFactorConfig) -> dict[str, float]:
    return {
        name: _initial_value(config, name)
        for name in config.variable_parameters
    }


def _initial_value(config: FittingFactorConfig, name: str) -> float:
    if name in config.initial_parameters:
        return float(config.initial_parameters[name])
    lo, hi = config.variable_parameters[name]
    return 0.5 * (lo + hi)


def _array_to_parameters(
    parameter_names: list[str], values: np.ndarray
) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(parameter_names, values, strict=True)
    }


def _match_value(result: ComparisonResult) -> float:
    if result.match is None:
        return -np.inf
    return float(result.match)
