"""The chain model registry: three model axes, YAML-selectable, one source -> 3 legs.

Three independent registries -- one per axis
(varactor / via / open-end) -- each mapping a string ``kind`` to a builder. A
:class:`ChainModelConfig` (typically loaded from YAML, see :mod:`.io`) names the
``kind`` + params for each axis; :func:`build_chain_models` constructs the three model
objects the emitter, analytical leg, and optimizer all consume.

**Extensibility (the PT-paper hook).** A new variant on any axis registers with a
single decorator -- ``@register_varactor("my_kind")`` / ``@register_via("my_kind")`` /
``@register_open_end("my_kind")`` -- and drops into every call site unchanged, because
the legs only ever see the built model object's interface (the varactor duck-type, the
:class:`~.via.ViaModel` / :class:`~.open_end.OpenEndModel` ABCs), never a ``kind`` string.

The reduced config (``power_law`` Varactor A / ``lumped_L`` / ``ideal``) rebuilds the
exact objects the legacy workflow uses, so it reproduces the canonical anchor
byte-for-byte (the recovery gate). It is retained for that byte-identity, but its
``lumped_L`` via and ``ideal`` open are **deprecated**.

**A4 standardization.** The standard terminations are the distributed/physical ADS
elements -- ``viagnd`` (:class:`~.via.ViagndVia`) for SHORT ends and ``mlef``
(:class:`~.open_end.MlefOpen`) for OPEN ends (see ``configs/model_registry/standard.yaml``
and :func:`standard_chain_models`). The simpler, non-distributed terminations
(``lumped_L`` / ``rlc`` vias, ``ideal`` / ``mloc`` opens) are deprecated and warn on
instantiation, but stay functional (see :mod:`.deprecation`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .deprecation import (
    DEPRECATED_OPEN_END_KINDS,
    DEPRECATED_VIA_KINDS,
    STANDARD_OPEN_END_KIND,
    STANDARD_VIA_KIND,
)
from .open_end import IdealOpen, MlefOpen, MlocOpen, OpenEndModel
from .varactor import (
    build_effective_c_varactor,
    build_polynomial_varactor,
    build_power_law_varactor,
)
from .via import LumpedLVia, RlcVia, ViagndVia, ViaModel

__all__ = [
    "ChainModelConfig",
    "ChainModels",
    "AxisSpec",
    "build_chain_models",
    "standard_chain_models",
    "build_varactor",
    "build_via",
    "build_open_end",
    "register_varactor",
    "register_via",
    "register_open_end",
    "VARACTOR_REGISTRY",
    "VIA_REGISTRY",
    "OPEN_END_REGISTRY",
    "STANDARD_VIA_KIND",
    "STANDARD_OPEN_END_KIND",
    "DEPRECATED_VIA_KINDS",
    "DEPRECATED_OPEN_END_KINDS",
]

# --------------------------------------------------------------------------- #
# The three registries                                                         #
# --------------------------------------------------------------------------- #
VARACTOR_REGISTRY: dict[str, Callable[[dict], object]] = {}
VIA_REGISTRY: dict[str, Callable[[dict], ViaModel]] = {}
OPEN_END_REGISTRY: dict[str, Callable[[dict], OpenEndModel]] = {}


def _register(reg: dict, kind: str):
    def deco(builder):
        if kind in reg:
            raise ValueError(f"model kind {kind!r} already registered.")
        reg[kind] = builder
        return builder
    return deco


def register_varactor(kind: str):
    """Register a varactor builder ``(params: dict) -> <varactor object>``."""
    return _register(VARACTOR_REGISTRY, kind)


def register_via(kind: str):
    """Register a via builder ``(params: dict) -> ViaModel``."""
    return _register(VIA_REGISTRY, kind)


def register_open_end(kind: str):
    """Register an open-end builder ``(params: dict) -> OpenEndModel``."""
    return _register(OPEN_END_REGISTRY, kind)


# --------------------------------------------------------------------------- #
# Built-in variants                                                            #
# --------------------------------------------------------------------------- #
register_varactor("power_law")(build_power_law_varactor)
register_varactor("effective_c")(build_effective_c_varactor)
register_varactor("polynomial")(build_polynomial_varactor)


@register_via("lumped_L")
def _build_lumped_L_via(params: dict) -> ViaModel:
    return LumpedLVia()


@register_via("rlc")
def _build_rlc_via(params: dict) -> ViaModel:
    return RlcVia(
        Lvia_h=float(params["Lvia_h"]),
        Rvia_ohm=float(params.get("Rvia_ohm", 0.0)),
        Cvia_f=float(params.get("Cvia_f", 0.0)),
        Z0via_ohm=float(params.get("Z0via_ohm", 0.0)),
    )


@register_via("viagnd")
def _build_viagnd_via(params: dict) -> ViaModel:
    kwargs: dict = {}
    if "via_radius_m" in params:
        kwargs["via_radius_m"] = float(params["via_radius_m"])
    if "metal_thickness_m" in params:
        kwargs["metal_thickness_m"] = float(params["metal_thickness_m"])
    if "rho_relative" in params:
        rho = params["rho_relative"]
        kwargs["rho_relative"] = None if rho is None else float(rho)
    if "analytical_mode" in params:
        kwargs["analytical_mode"] = str(params["analytical_mode"])
    if "effective_L_h" in params:
        kwargs["effective_L_h"] = float(params["effective_L_h"])
    if "effective_R_ohm" in params:
        kwargs["effective_R_ohm"] = float(params["effective_R_ohm"])
    return ViagndVia(**kwargs)


@register_open_end("ideal")
def _build_ideal_open(params: dict) -> OpenEndModel:
    return IdealOpen()


@register_open_end("mlef")
def _build_mlef_open(params: dict) -> OpenEndModel:
    return MlefOpen()


@register_open_end("mloc")
def _build_mloc_open(params: dict) -> OpenEndModel:
    return MlocOpen()


# --------------------------------------------------------------------------- #
# Config + builder                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AxisSpec:
    """One axis selection: a registered ``kind`` + that kind's params."""

    kind: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChainModelConfig:
    """A chain device/parasitic model config (the YAML the design selects).

    Three independent axes. ``from_dict`` accepts the schema the
    ``configs/model_registry/*.yaml`` files use.
    """

    name: str
    varactor: AxisSpec
    via: AxisSpec
    open_end: AxisSpec

    @classmethod
    def from_dict(cls, d: dict) -> "ChainModelConfig":
        def _axis(key: str) -> AxisSpec:
            if key not in d:
                raise ValueError(f"config is missing the {key!r} axis.")
            block = dict(d[key])
            if "kind" not in block:
                raise ValueError(f"config {key!r} axis is missing 'kind'.")
            kind = str(block.pop("kind"))
            return AxisSpec(kind=kind, params=block)

        return cls(
            name=str(d.get("name", "unnamed")),
            varactor=_axis("varactor"),
            via=_axis("via"),
            open_end=_axis("open_end"),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "varactor": {"kind": self.varactor.kind, **self.varactor.params},
            "via": {"kind": self.via.kind, **self.via.params},
            "open_end": {"kind": self.open_end.kind, **self.open_end.params},
        }


@dataclass(frozen=True)
class ChainModels:
    """The three built model objects the emitter / analytical / optimizer consume."""

    varactor: object
    via: ViaModel
    open_end: OpenEndModel
    name: str = ""


def _lookup(reg: dict, kind: str, axis: str):
    if kind not in reg:
        raise ValueError(
            f"unknown {axis} kind {kind!r}; registered: {sorted(reg)}."
        )
    return reg[kind]


def build_varactor(spec: AxisSpec) -> object:
    return _lookup(VARACTOR_REGISTRY, spec.kind, "varactor")(spec.params)


def build_via(spec: AxisSpec) -> ViaModel:
    return _lookup(VIA_REGISTRY, spec.kind, "via")(spec.params)


def build_open_end(spec: AxisSpec) -> OpenEndModel:
    return _lookup(OPEN_END_REGISTRY, spec.kind, "open_end")(spec.params)


def build_chain_models(config: ChainModelConfig) -> ChainModels:
    """Build ``(varactor, via, open_end)`` from a :class:`ChainModelConfig`."""

    return ChainModels(
        varactor=build_varactor(config.varactor),
        via=build_via(config.via),
        open_end=build_open_end(config.open_end),
        name=config.name,
    )


def standard_chain_models(varactor: object, *, name: str = "standard") -> ChainModels:
    """The A4-standard chain models for ``varactor``: ``viagnd`` via + ``mlef`` open-end.

    A programmatic shortcut for the standardized distributed terminations
    (:data:`STANDARD_VIA_KIND` / :data:`STANDARD_OPEN_END_KIND`), equivalent to
    building ``configs/model_registry/standard.yaml`` but taking the varactor object
    directly. The simpler lumped/ideal terminations are deprecated (see
    :mod:`.deprecation`); new designs should pass these models to the cell / spec
    builder instead of relying on the legacy inline default.
    """

    return ChainModels(
        varactor=varactor,
        via=build_via(AxisSpec(kind=STANDARD_VIA_KIND)),
        open_end=build_open_end(AxisSpec(kind=STANDARD_OPEN_END_KIND)),
        name=name,
    )


