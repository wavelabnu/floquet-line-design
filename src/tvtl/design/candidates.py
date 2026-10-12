"""Generic candidate data structures for TVTL design search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    component_id: str
    component_kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    terminals: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("ComponentSpec component_id must be non-empty.")
        if not self.component_kind.strip():
            raise ValueError("ComponentSpec component_kind must be non-empty.")


@dataclass(frozen=True, slots=True)
class BranchSpec:
    branch_id: str
    attachment_nodes: tuple[str, ...]
    components: tuple[ComponentSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.branch_id.strip():
            raise ValueError("BranchSpec branch_id must be non-empty.")
        if not self.attachment_nodes:
            raise ValueError("BranchSpec requires at least one attachment node.")


@dataclass(frozen=True, slots=True)
class UnitCellGraph:
    graph_id: str
    through_components: tuple[ComponentSpec, ...]
    branches: tuple[BranchSpec, ...] = ()
    ports: tuple[str, str] = ("input", "output")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.graph_id.strip():
            raise ValueError("UnitCellGraph graph_id must be non-empty.")
        if len(self.ports) != 2:
            raise ValueError("UnitCellGraph currently represents generic two-port unit cells.")


@dataclass(frozen=True, slots=True)
class DesignCandidate:
    candidate_id: str
    unit_cell: UnitCellGraph
    variable_values: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("DesignCandidate candidate_id must be non-empty.")

    def value(self, name: str, default: Any | None = None) -> Any:
        return self.variable_values.get(name, default)


@dataclass(frozen=True, slots=True)
class CandidateScorecard:
    candidate_id: str
    proxy_metrics: Mapping[str, float]
    penalties: Mapping[str, float] = field(default_factory=dict)
    full_device_gain_available: bool = False
    full_device_gain_db: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        return float(sum(self.proxy_metrics.values()) - sum(self.penalties.values()))

    def __post_init__(self) -> None:
        if self.full_device_gain_available and self.full_device_gain_db is None:
            raise ValueError("full_device_gain_db is required when full_device_gain_available=True.")


