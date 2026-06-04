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
from waveformtools.comparison.rotation import RotationSpec, rotate_modes
from waveformtools.rotation_math import (
    quaternion_from_two_vectors,
    quaternion_rotate_vector,
    quaternion_to_euler_zyz,
)

FrameRotationMethod = Literal[
    "none",
    "align_radiated_linear_momentum",
    "align_kick",
]
FrameBoostMethod = Literal["none"]
FrameSupertranslationMethod = Literal["none"]
_MOMENTUM_ALIGNMENT = "align_radiated_linear_momentum"


@dataclass(slots=True)
class BMSFramePreprocessingSpec:
    """Configuration for approximate BMS-frame preprocessing."""

    diagnose: bool = True
    diagnostics: BMSFrameDiagnosticsConfig | Mapping[str, Any] = field(
        default_factory=BMSFrameDiagnosticsConfig
    )
    rotation: FrameRotationMethod = "none"
    boost: FrameBoostMethod = "none"
    supertranslation: FrameSupertranslationMethod = "none"
    require_compatible_diagnostics: bool = False
    relative_energy_tolerance: float = 0.05
    relative_angular_momentum_tolerance: float = 0.1
    relative_linear_momentum_tolerance: float = 0.1
    minimum_rotation_vector_norm: float = 1e-12
    vector_imaginary_relative_tolerance: float = 1e-2

    def __post_init__(self) -> None:
        self.diagnostics = BMSFrameDiagnosticsConfig.from_value(
            self.diagnostics
        )
        self.rotation = _normalize_rotation_method(self.rotation)
        if self.rotation not in {"none", _MOMENTUM_ALIGNMENT}:
            raise NotImplementedError(
                "Only rotation='none' is currently supported."
            )
        if self.rotation != "none" and not self.diagnose:
            raise ValueError(
                "diagnose=True is required for diagnostic-driven rotation."
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
        self.minimum_rotation_vector_norm = _nonnegative_float(
            self.minimum_rotation_vector_norm,
            "minimum_rotation_vector_norm",
        )
        self.vector_imaginary_relative_tolerance = _nonnegative_float(
            self.vector_imaginary_relative_tolerance,
            "vector_imaginary_relative_tolerance",
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

    This conservative first implementation applies no BMS transformation.  It
    computes diagnostics, records transform choices, and returns deep copies so
    callers can use the result as a preprocessing step before fitting-factor
    comparisons.
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
    if preprocessing_spec.rotation == _MOMENTUM_ALIGNMENT:
        candidate_pre, candidate_diagnostics, rotation_report = (
            _align_candidate_momentum_direction(
                candidate_pre,
                target_diagnostics,
                candidate_diagnostics,
                preprocessing_spec,
            )
        )
        transforms["rotation_details"] = rotation_report
        transforms["applied"] = True
        transforms["applied_operations"].append("rotation")

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


def _normalize_rotation_method(value: str) -> str:
    method = str(value)
    if method == "align_kick":
        return _MOMENTUM_ALIGNMENT
    return method


def _initial_transform_report(
    spec: BMSFramePreprocessingSpec,
) -> dict[str, Any]:
    return {
        "rotation": spec.rotation,
        "boost": spec.boost,
        "supertranslation": spec.supertranslation,
        "applied": False,
        "applied_operations": [],
        "notes": (
            "No BMS transform is applied unless an explicit synchronization "
            "method is requested."
        ),
    }


def _align_candidate_momentum_direction(
    candidate_modes: Any,
    target_diagnostics: BMSFrameDiagnosticsResult | None,
    candidate_diagnostics: BMSFrameDiagnosticsResult | None,
    spec: BMSFramePreprocessingSpec,
) -> tuple[Any, BMSFrameDiagnosticsResult, dict[str, Any]]:
    if target_diagnostics is None or candidate_diagnostics is None:
        raise ValueError(
            "radiated-linear-momentum alignment requires diagnostics."
        )
    target_vector = _real_diagnostic_vector(
        target_diagnostics.radiated_linear_momentum,
        "target radiated_linear_momentum",
        spec.vector_imaginary_relative_tolerance,
    )
    candidate_vector = _real_diagnostic_vector(
        candidate_diagnostics.radiated_linear_momentum,
        "candidate radiated_linear_momentum",
        spec.vector_imaginary_relative_tolerance,
    )
    target_norm = float(np.linalg.norm(target_vector))
    candidate_norm = float(np.linalg.norm(candidate_vector))
    minimum_norm = spec.minimum_rotation_vector_norm
    if target_norm <= minimum_norm or candidate_norm <= minimum_norm:
        raise ValueError(
            "radiated-linear-momentum alignment requires nonzero momentum "
            f"vectors above {minimum_norm}."
        )

    quat = quaternion_from_two_vectors(candidate_vector, target_vector)
    alpha, beta, gamma = quaternion_to_euler_zyz(quat)
    rotation = RotationSpec(
        kind="wigner",
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    rotated_modes = rotate_modes(candidate_modes, rotation)
    rotated_candidate_vector = quaternion_rotate_vector(quat, candidate_vector)
    rotated_candidate_diagnostics = _rotate_diagnostics_vectors(
        candidate_diagnostics,
        quat,
    )
    report = {
        "method": _MOMENTUM_ALIGNMENT,
        "source_quantity": "radiated_linear_momentum",
        "rotation": rotation.to_dict(),
        "quaternion": [float(item) for item in quat],
        "target_vector": [float(item) for item in target_vector],
        "candidate_vector_before": [float(item) for item in candidate_vector],
        "candidate_vector_after": [
            float(item) for item in rotated_candidate_vector
        ],
        "target_norm": target_norm,
        "candidate_norm": candidate_norm,
        "direction_residual": _direction_residual(
            target_vector,
            rotated_candidate_vector,
        ),
    }
    return rotated_modes, rotated_candidate_diagnostics, report


def _rotate_diagnostics_vectors(
    diagnostics: BMSFrameDiagnosticsResult,
    quat: np.ndarray,
) -> BMSFrameDiagnosticsResult:
    assumptions = dict(diagnostics.assumptions)
    assumptions["rotated_by_bms_frame_preprocessing"] = True
    metadata = dict(diagnostics.diagnostics)
    metadata["rotated_by_bms_frame_preprocessing"] = True
    return BMSFrameDiagnosticsResult(
        energy_radiated=diagnostics.energy_radiated,
        radiated_linear_momentum=_rotate_optional_vector(
            diagnostics.radiated_linear_momentum,
            quat,
            "radiated_linear_momentum",
            np.inf,
        ),
        kick_velocity=_rotate_optional_vector(
            diagnostics.kick_velocity,
            quat,
            "kick_velocity",
            np.inf,
        ),
        angular_momentum_radiated=_rotate_optional_vector(
            diagnostics.angular_momentum_radiated,
            quat,
            "angular_momentum_radiated",
            np.inf,
        ),
        memory_finite_time=diagnostics.memory_finite_time,
        omitted_inspiral=diagnostics.omitted_inspiral,
        assumptions=assumptions,
        diagnostics=metadata,
    )


def _rotate_optional_vector(
    value: np.ndarray | None,
    quat: np.ndarray,
    name: str,
    imaginary_relative_tolerance: float,
) -> np.ndarray | None:
    if value is None:
        return None
    vector = _real_diagnostic_vector(
        value,
        name,
        imaginary_relative_tolerance,
    )
    return quaternion_rotate_vector(quat, vector)


def _real_diagnostic_vector(
    value: np.ndarray | None,
    name: str,
    imaginary_relative_tolerance: float,
) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is unavailable.")
    array = np.asarray(value)
    if array.size != 3:
        raise ValueError(f"{name} must contain three components.")
    vector = array.reshape(3)
    real_part = np.asarray(vector.real, dtype=float)
    imag_part = np.asarray(vector.imag, dtype=float)
    imag_norm = float(np.linalg.norm(imag_part))
    real_norm = float(np.linalg.norm(real_part))
    tolerance = imaginary_relative_tolerance * max(real_norm, 1e-300)
    if imag_norm > tolerance:
        raise ValueError(f"{name} has a significant imaginary component.")
    if not np.all(np.isfinite(real_part)):
        raise ValueError(f"{name} must be finite.")
    return real_part


def _direction_residual(target: np.ndarray, candidate: np.ndarray) -> float:
    target_unit = target / max(float(np.linalg.norm(target)), 1e-300)
    candidate_unit = candidate / max(float(np.linalg.norm(candidate)), 1e-300)
    return float(np.linalg.norm(target_unit - candidate_unit))


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
