"""Generic design-variable contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class VariableKind(StrEnum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass(frozen=True, slots=True)
class DesignVariable:
    name: str
    kind: VariableKind
    units: str | None = None
    lower: float | None = None
    upper: float | None = None
    choices: tuple[Any, ...] = ()
    default: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("DesignVariable name must be non-empty.")
        if self.kind == VariableKind.CONTINUOUS:
            if self.lower is None or self.upper is None:
                raise ValueError("Continuous variables require lower and upper bounds.")
            if float(self.lower) >= float(self.upper):
                raise ValueError("Continuous variables require lower < upper.")
        elif self.kind == VariableKind.DISCRETE:
            if not self.choices:
                raise ValueError("Discrete variables require at least one choice.")
        else:
            raise ValueError(f"Unsupported variable kind {self.kind!r}.")


def sample_variable_midpoint(variable: DesignVariable) -> Any:
    if variable.default is not None:
        return variable.default
    if variable.kind == VariableKind.CONTINUOUS:
        assert variable.lower is not None and variable.upper is not None
        return 0.5 * (float(variable.lower) + float(variable.upper))
    return variable.choices[0]


def design_variable_from_mapping(payload: Mapping[str, Any]) -> DesignVariable:
    kind = VariableKind(str(payload.get("kind", VariableKind.CONTINUOUS)))
    return DesignVariable(
        name=str(payload["name"]),
        kind=kind,
        units=None if payload.get("units") is None else str(payload.get("units")),
        lower=None if payload.get("lower") is None else float(payload.get("lower")),
        upper=None if payload.get("upper") is None else float(payload.get("upper")),
        choices=tuple(payload.get("choices", ())),
        default=payload.get("default"),
        metadata=dict(payload.get("metadata", {})),
    )


def deterministic_variable_value(variable: DesignVariable, index: int, count: int, *, seed: int = 0) -> Any:
    if variable.kind == VariableKind.DISCRETE:
        choices = tuple(variable.choices)
        return choices[(int(index) + int(seed)) % len(choices)]
    assert variable.lower is not None and variable.upper is not None
    if count <= 1:
        return sample_variable_midpoint(variable)
    stride = ((int(index) * (2 * int(seed) + 1)) + int(seed)) % int(count)
    fraction = (float(stride) + 0.5) / float(count)
    return float(variable.lower) + fraction * (float(variable.upper) - float(variable.lower))


