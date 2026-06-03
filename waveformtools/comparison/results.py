"""Structured result objects for waveform comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from waveformtools.comparison.metadata import WaveformMetadata


@dataclass(slots=True)
class ComparisonResult:
    """Result returned by mode-comparison objectives.

    The object is intentionally richer than a bare float so optimization runs
    can later record diagnostics, best-fit nuisance parameters, and provenance.
    """

    objective_name: str
    match: float | None = None
    mismatch: float | None = None
    distance: float | None = None
    normalized_distance: float | None = None
    best_parameters: dict[str, Any] = field(default_factory=dict)
    fixed_parameters: dict[str, Any] = field(default_factory=dict)
    optimizer: str | None = None
    optimizer_status: str = "not_run"
    n_objective_evaluations: int = 1
    n_waveform_generations: int = 0
    elapsed_s: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    target_metadata: WaveformMetadata | None = None
    candidate_metadata: WaveformMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly dictionary representation."""

        data = asdict(self)
        if self.target_metadata is not None:
            data["target_metadata"] = self.target_metadata.to_dict()
        if self.candidate_metadata is not None:
            data["candidate_metadata"] = self.candidate_metadata.to_dict()
        return data


@dataclass(slots=True)
class FittingFactorResult(ComparisonResult):
    """Result for future waveform-family fitting-factor searches."""

    candidate_approximant: str | None = None
    candidate_generation_parameters: dict[str, Any] = field(default_factory=dict)
