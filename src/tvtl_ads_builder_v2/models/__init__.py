"""Chain device/parasitic model registry.

A YAML-selectable registry for the three model axes the TVTL chain depends on --
the **varactor** (device C), the **via** (stub-to-ground termination), and the
**open-end** (stub far end) -- consumed identically by the ADS emitter, the
analytical leg, and the optimizer. The reduced default (``power_law`` Varactor A
/ ``lumped_L`` via / ``ideal`` open) reproduces the canonical anchor
byte-for-byte; the REFERENCE convention (``polynomial`` / ``rlc`` / ``mlef``) is
a named config that drops in; enriched PT-paper cell models register with one
decorator (see :mod:`.registry`).

See ``configs/model_registry/*.yaml`` for the schema.
"""

from __future__ import annotations

from .open_end import IdealOpen, MlefOpen, MlocOpen, OpenEndModel
from .registry import (
    AxisSpec,
    ChainModelConfig,
    ChainModels,
    DEPRECATED_OPEN_END_KINDS,
    DEPRECATED_VIA_KINDS,
    OPEN_END_REGISTRY,
    STANDARD_OPEN_END_KIND,
    STANDARD_VIA_KIND,
    VARACTOR_REGISTRY,
    VIA_REGISTRY,
    build_chain_models,
    build_open_end,
    build_varactor,
    build_via,
    register_open_end,
    register_varactor,
    register_via,
    standard_chain_models,
)
from .varactor import PolynomialVaractor, smv1405
from .via import LumpedLVia, RlcVia, ViagndVia, ViaModel

__all__ = [
    "AxisSpec",
    "ChainModelConfig",
    "ChainModels",
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
    "ViaModel",
    "LumpedLVia",
    "RlcVia",
    "ViagndVia",
    "OpenEndModel",
    "IdealOpen",
    "MlefOpen",
    "MlocOpen",
    "PolynomialVaractor",
    "smv1405",
]


