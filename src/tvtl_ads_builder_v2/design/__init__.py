"""Topology-driven TVTL design API.

A :class:`TVTLDesign` declares a TVTL design as data -- one unit cell, a
cell count, the design parameters, the sources and the analysis. The
expander :func:`to_device_spec` turns it into a fully-explicit
:class:`~tvtl_ads_builder_v2.device_spec.DeviceSpec`, generating per-cell
element instances and per-cell net names mechanically.

This layer sits ON TOP of the existing v2 platform. The dependency
direction is strict and one-way: ``design`` imports from ``device_spec``
(and, for the charge-polynomial registry, the analytical ``tvtl`` stack);
nothing outside ``design`` imports from ``design``. If platform code
needs design-layer functionality, the abstraction is wrong.

The HB-controller dataclasses (:class:`HBSingleToneSpec`,
:class:`HBTwoToneSpec`) are re-exported from the device-spec schema so a
caller can build an :class:`AnalysisSpec` from this package alone.
"""

from ..device_spec import HBSingleToneSpec, HBTwoToneSpec
from .elements import (
    AnalysisSpec,
    BranchSpec,
    CellSpec,
    ElementSpec,
    MSUBSpec,
    MicrostripSpec,
    SParamSpec,
    SourceSpec,
    TVTLDesign,
)
from .expand import to_device_spec
from .validator import DesignError, validate_design

__all__ = [
    "AnalysisSpec",
    "BranchSpec",
    "CellSpec",
    "DesignError",
    "ElementSpec",
    "HBSingleToneSpec",
    "HBTwoToneSpec",
    "MSUBSpec",
    "MicrostripSpec",
    "SParamSpec",
    "SourceSpec",
    "TVTLDesign",
    "to_device_spec",
    "validate_design",
]


