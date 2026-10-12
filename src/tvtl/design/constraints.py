"""Generic constraints for design candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class ConstraintStatus(StrEnum):
    PASSING = "passing"
    FAILING = "failing"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    name: str
    metric_name: str
    minimum: float | None = None
    maximum: float | None = None
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConstraintEvaluation:
    constraint: ConstraintSpec
    status: ConstraintStatus
    value: float | None
    reason: str = ""


def evaluate_constraints(
    metrics: Mapping[str, float],
    constraints: tuple[ConstraintSpec, ...],
) -> tuple[ConstraintEvaluation, ...]:
    evaluations: list[ConstraintEvaluation] = []
    for constraint in constraints:
        if constraint.metric_name not in metrics:
            status = ConstraintStatus.FAILING if constraint.required else ConstraintStatus.UNAVAILABLE
            evaluations.append(ConstraintEvaluation(constraint, status, None, "metric unavailable"))
            continue
        value = float(metrics[constraint.metric_name])
        passing = True
        if constraint.minimum is not None and value < float(constraint.minimum):
            passing = False
        if constraint.maximum is not None and value > float(constraint.maximum):
            passing = False
        evaluations.append(
            ConstraintEvaluation(
                constraint=constraint,
                status=ConstraintStatus.PASSING if passing else ConstraintStatus.FAILING,
                value=value,
            )
        )
    return tuple(evaluations)


def constraint_from_mapping(payload: Mapping[str, Any]) -> ConstraintSpec:
    return ConstraintSpec(
        name=str(payload["name"]),
        metric_name=str(payload.get("metric_name", payload["name"])),
        minimum=None if payload.get("minimum") is None else float(payload.get("minimum")),
        maximum=None if payload.get("maximum") is None else float(payload.get("maximum")),
        required=bool(payload.get("required", True)),
        metadata=dict(payload.get("metadata", {})),
    )


