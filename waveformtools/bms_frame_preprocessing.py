"""Preprocessing reports for approximate BMS-frame compatibility.

This module provides a conservative entry point for comparing two waveforms
before fitting-factor or balance-law repair workflows.  The first
implementation computes diagnostics and compatibility flags, but does not
silently transform the input waveforms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from waveformtools.bms_frame_diagnostics import (
    BMSFrameDiagnosticsConfig,
    BMSFrameDiagnosticsResult,
    compute_bms_frame_diagnostics,
)

FrameBoostMethod = Literal["none"]
FrameSupertranslationMethod = Literal["none"]


@dataclass(slots=True)
class BMSFramePreprocessingSpec:
    """Configuration for approximate BMS-frame preprocessing."""

    diagnose: bool = True
    diagnostics: BMSFrameDiagnosticsConfig | Mapping[str, Any] = field(
        default_factory=BMSFrameDiagnosticsConfig
    )
    boost: FrameBoostMethod = "none"
    supertranslation: FrameSupertranslationMethod = "none"
    require_compatible_diagnostics: bool = False
    relative_energy_tolerance: float = 0.05
    relative_angular_momentum_tolerance: float = 0.1
    relative_linear_momentum_tolerance: float = 0.1

    def __post_init__(self) -> None:
        self.diagnostics = BMSFrameDiagnosticsConfig.from_value(
            self.diagnostics
        )
        if self.boost != "none":
            raise NotImplementedError(
                "Only boost='none' is currently supported."
            )
        if self.supertranslation != "none":
            raise NotImplementedError(
                "Only supertranslation='none' is currently supported."
            )
        self.relative_energy_tolerance = _nonnegative_float(
            self.relative_energy_tolerance,
            "relative_energy_tolerance",
        )
        self.relative_angular_momentum_tolerance = _nonnegative_float(
            self.relative_angular_momentum_tolerance,
            "relative_angular_momentum_tolerance",
        )
        self.relative_linear_momentum_tolerance = _nonnegative_float(
            self.relative_linear_momentum_tolerance,
            "relative_linear_momentum_tolerance",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        data = asdict(self)
        data["diagnostics"] = self.diagnostics.to_dict()
        return data

    @classmethod
    def from_value(
        cls,
        value: "BMSFramePreprocessingSpec | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "BMSFramePreprocessingSpec":
        """Construct preprocessing spec from a dataclass, mapping, or none."""

        if value is None:
            data: dict[str, Any] = {}
        elif isinstance(value, cls):
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "BMS frame preprocessing spec must be a "
                "BMSFramePreprocessingSpec, mapping, or None; "
                f"got {type(value)!r}."
            )
        data.update(
            {key: val for key, val in overrides.items() if val is not None}
        )
        return cls(**data)


@dataclass(slots=True)
class BMSFramePreprocessingReport:
    """Diagnostics and compatibility report for two waveforms."""

    target_diagnostics: BMSFrameDiagnosticsResult | None
    candidate_diagnostics: BMSFrameDiagnosticsResult | None
    compatibility: dict[str, Any]
    charge_diagnostics: dict[str, Any]
    transforms: dict[str, Any]
    assumptions: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""

        return {
            "target_diagnostics": (
                None
                if self.target_diagnostics is None
                else self.target_diagnostics.to_dict()
            ),
            "candidate_diagnostics": (
                None
                if self.candidate_diagnostics is None
                else self.candidate_diagnostics.to_dict()
            ),
            "compatibility": self.compatibility,
            "charge_diagnostics": self.charge_diagnostics,
            "transforms": self.transforms,
            "assumptions": self.assumptions,
            "config": self.config,
        }


def preprocess_bms_frame(
    target_modes: Any,
    candidate_modes: Any,
    spec: BMSFramePreprocessingSpec | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> tuple[Any, Any, BMSFramePreprocessingReport]:
    """Return preprocessed copies plus a BMS-frame compatibility report.

    No BMS transformation is applied; the returned modes are copies of the
    inputs.
    """

    preprocessing_spec = BMSFramePreprocessingSpec.from_value(
        value=spec,
        **overrides,
    )
    target_diagnostics = None
    candidate_diagnostics = None
    if preprocessing_spec.diagnose:
        target_diagnostics = compute_bms_frame_diagnostics(
            target_modes,
            preprocessing_spec.diagnostics,
        )
        candidate_diagnostics = compute_bms_frame_diagnostics(
            candidate_modes,
            preprocessing_spec.diagnostics,
        )
    target_pre = _copy_modes(target_modes)
    candidate_pre = _copy_modes(candidate_modes)
    transforms = _initial_transform_report(preprocessing_spec)

    compatibility = _compatibility_report(
        target_diagnostics,
        candidate_diagnostics,
        preprocessing_spec,
    )
    if (
        preprocessing_spec.require_compatible_diagnostics
        and not compatibility["compatible"]
    ):
        raise ValueError(
            "BMS frame diagnostics are outside requested tolerances: "
            f"{compatibility['failed_checks']}."
        )
    report = BMSFramePreprocessingReport(
        target_diagnostics=target_diagnostics,
        candidate_diagnostics=candidate_diagnostics,
        compatibility=compatibility,
        charge_diagnostics=_paired_charge_diagnostics_report(
            target_diagnostics,
            candidate_diagnostics,
        ),
        transforms=transforms,
        assumptions={
            "pn_eob_phenom_frame_comment": (
                "Analytic approximants are usually supplied in conventional "
                "asymptotic source frames, but full BMS-frame equivalence is "
                "not guaranteed by waveform modes alone."
            ),
            "superrest_frame_fixed": False,
            "absolute_bms_charges_computed": False,
        },
        config=preprocessing_spec.to_dict(),
    )
    return target_pre, candidate_pre, report


def _initial_transform_report(
    spec: BMSFramePreprocessingSpec,
) -> dict[str, Any]:
    return {
        "boost": spec.boost,
        "supertranslation": spec.supertranslation,
        "applied": False,
        "applied_operations": [],
        "notes": "No BMS transform is applied.",
    }


def _direction_residual(target: np.ndarray, candidate: np.ndarray) -> float:
    target_unit = target / max(float(np.linalg.norm(target)), 1e-300)
    candidate_unit = candidate / max(float(np.linalg.norm(candidate)), 1e-300)
    return float(np.linalg.norm(target_unit - candidate_unit))


def _paired_charge_diagnostics_report(
    target: BMSFrameDiagnosticsResult | None,
    candidate: BMSFrameDiagnosticsResult | None,
) -> dict[str, Any]:
    if target is None or candidate is None:
        return {
            "available": False,
            "diagnostics_available": False,
            "notes": "BMS-frame diagnostics were not computed.",
        }
    return {
        "available": True,
        "diagnostics_available": True,
        "diagnostic_type": "paired_low_order_bms_flux_proxy_summary",
        "target": target.charge_diagnostics,
        "candidate": candidate.charge_diagnostics,
        "comparison": {
            "supermomentum_l0_energy": _relative_difference_summary(
                target.energy_radiated,
                candidate.energy_radiated,
            ),
            "supermomentum_l1_linear_momentum": _vector_difference_summary(
                target.radiated_linear_momentum,
                candidate.radiated_linear_momentum,
            ),
            "angular_momentum": _vector_difference_summary(
                target.angular_momentum_radiated,
                candidate.angular_momentum_radiated,
            ),
            "kick_velocity": _vector_difference_summary(
                target.kick_velocity,
                candidate.kick_velocity,
            ),
        },
        "notes": (
            "This is a diagnostics-only comparison of low-order radiated-flux "
            "proxies. It does not equalize charges or modify waveform modes."
        ),
    }


def _relative_difference_summary(
    target_value: float | None,
    candidate_value: float | None,
) -> dict[str, Any]:
    if target_value is None or candidate_value is None:
        return {"available": False}
    target_float = float(target_value)
    candidate_float = float(candidate_value)
    absolute_difference = candidate_float - target_float
    scale = max(abs(target_float), abs(candidate_float), 1e-300)
    return {
        "available": True,
        "target": target_float,
        "candidate": candidate_float,
        "absolute_difference": absolute_difference,
        "relative_difference": abs(absolute_difference) / scale,
    }


def _vector_difference_summary(
    target_value: np.ndarray | None,
    candidate_value: np.ndarray | None,
) -> dict[str, Any]:
    if target_value is None or candidate_value is None:
        return {"available": False}
    target_vector, target_imaginary_norm = _summary_real_vector(target_value)
    candidate_vector, candidate_imaginary_norm = _summary_real_vector(
        candidate_value
    )
    target_norm = float(np.linalg.norm(target_vector))
    candidate_norm = float(np.linalg.norm(candidate_vector))
    difference_norm = float(np.linalg.norm(candidate_vector - target_vector))
    scale = max(target_norm, candidate_norm, 1e-300)
    direction_residual = None
    if target_norm > 0.0 and candidate_norm > 0.0:
        direction_residual = _direction_residual(
            target_vector,
            candidate_vector,
        )
    return {
        "available": True,
        "target_vector": [float(item) for item in target_vector],
        "candidate_vector": [float(item) for item in candidate_vector],
        "target_imaginary_norm": target_imaginary_norm,
        "candidate_imaginary_norm": candidate_imaginary_norm,
        "target_norm": target_norm,
        "candidate_norm": candidate_norm,
        "difference_norm": difference_norm,
        "relative_norm_difference": abs(candidate_norm - target_norm) / scale,
        "relative_vector_difference": difference_norm / scale,
        "direction_residual": direction_residual,
    }


def _summary_real_vector(value: np.ndarray) -> tuple[np.ndarray, float]:
    array = np.asarray(value).reshape(3)
    vector = np.asarray(array.real, dtype=float)
    imaginary_norm = float(np.linalg.norm(np.asarray(array.imag, dtype=float)))
    return vector, imaginary_norm


def _compatibility_report(
    target: BMSFrameDiagnosticsResult | None,
    candidate: BMSFrameDiagnosticsResult | None,
    spec: BMSFramePreprocessingSpec,
) -> dict[str, Any]:
    if target is None or candidate is None:
        return {
            "compatible": True,
            "diagnostics_available": False,
            "checks": {},
            "failed_checks": [],
        }
    checks = {
        "energy_radiated": _relative_difference_check(
            target.energy_radiated,
            candidate.energy_radiated,
            spec.relative_energy_tolerance,
        ),
        "angular_momentum_radiated": _vector_relative_difference_check(
            target.angular_momentum_radiated,
            candidate.angular_momentum_radiated,
            spec.relative_angular_momentum_tolerance,
        ),
        "radiated_linear_momentum": _vector_relative_difference_check(
            target.radiated_linear_momentum,
            candidate.radiated_linear_momentum,
            spec.relative_linear_momentum_tolerance,
        ),
        "omitted_inspiral": _omitted_inspiral_check(
            target.omitted_inspiral,
            candidate.omitted_inspiral,
        ),
    }
    failed_checks = [
        name
        for name, check in checks.items()
        if check["available"] and not check["within_tolerance"]
    ]
    return {
        "compatible": not failed_checks,
        "diagnostics_available": True,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _relative_difference_check(
    target_value: float | None,
    candidate_value: float | None,
    tolerance: float,
) -> dict[str, Any]:
    if target_value is None or candidate_value is None:
        return {"available": False, "within_tolerance": True}
    target_float = float(target_value)
    candidate_float = float(candidate_value)
    scale = max(abs(target_float), abs(candidate_float), 1e-300)
    relative_difference = abs(target_float - candidate_float) / scale
    return {
        "available": True,
        "target": float(target_value),
        "candidate": float(candidate_value),
        "relative_difference": float(relative_difference),
        "tolerance": float(tolerance),
        "within_tolerance": bool(relative_difference <= tolerance),
    }


def _vector_relative_difference_check(
    target_value: np.ndarray | None,
    candidate_value: np.ndarray | None,
    tolerance: float,
) -> dict[str, Any]:
    if target_value is None or candidate_value is None:
        return {"available": False, "within_tolerance": True}
    target_array = np.asarray(target_value, dtype=np.complex128)
    candidate_array = np.asarray(candidate_value, dtype=np.complex128)
    scale = max(
        float(np.linalg.norm(target_array)),
        float(np.linalg.norm(candidate_array)),
        1e-300,
    )
    vector_difference = np.linalg.norm(target_array - candidate_array)
    relative_difference = vector_difference / scale
    return {
        "available": True,
        "relative_difference": float(relative_difference),
        "tolerance": float(tolerance),
        "within_tolerance": bool(relative_difference <= tolerance),
    }


def _omitted_inspiral_check(
    target_value: Mapping[str, Any] | None,
    candidate_value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if target_value is None or candidate_value is None:
        return {"available": False, "within_tolerance": True}
    target_flag = bool(target_value.get("omitted_inspiral_likely", False))
    candidate_omitted = candidate_value.get("omitted_inspiral_likely", False)
    candidate_flag = bool(candidate_omitted)
    return {
        "available": True,
        "target_omitted_inspiral_likely": target_flag,
        "candidate_omitted_inspiral_likely": candidate_flag,
        "within_tolerance": target_flag == candidate_flag,
    }


def _copy_modes(modes_obj: Any) -> Any:
    if hasattr(modes_obj, "deepcopy"):
        return modes_obj.deepcopy()
    raise TypeError("modes objects must provide deepcopy().")


def _nonnegative_float(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return numeric
