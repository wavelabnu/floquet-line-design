"""Topology registry for TVTL cascade-Floquet design optimisation.

A *topology* is a unit-cell architecture: a specific arrangement of microstrip
segments, varactor position, and terminations. Multiple physical designs
share a topology if they differ only in dimensional parameters.

This module exposes:

* :class:`DesignVariableSpec` -- one scalar design variable with bounds.
* :class:`TopologyDescriptor` -- a complete unit-cell topology specification.
* :func:`register_topology`, :func:`get_topology`, :func:`list_topologies`
  -- access helpers for the global registry.

The registry is populated by importing topology modules. The default
``via_stub_3segment`` descriptor is registered by :mod:`tvtl.floquet.uc_model`;
the additional :mod:`tvtl.floquet.topologies` submodules register the open-stub
and lumped-shunt variants. All numeric outputs produced by topology builders
are analytical proxies only; callers must propagate the
``"analytical_proxy_only"`` label into downstream artifacts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Tuple


class DesignVariableSpec(NamedTuple):
    """One scalar design variable in a topology's design space."""

    name: str
    lower_bound: float
    upper_bound: float
    units: str
    description: str


class TopologyDescriptor(NamedTuple):
    """Complete specification of a unit-cell topology.

    Attributes
    ----------
    name : str
        Unique snake_case identifier (e.g. ``'via_stub_3segment'``).
    version : str
        Topology schema version string.
    description : str
        One-paragraph description of the unit cell architecture.
    design_variables : tuple of DesignVariableSpec
        Ordered list of named scalar variables and their bounds.
    build_unit_cell_block_abcd : Callable
        Function with signature
        ``(design, varactor_params, modulation_params, substrate, ws, wm, N) -> (A, B, C, D)``
        returning the harmonic-basis block ABCD of one unit cell.
    period : Callable
        Function ``design -> float`` returning the unit-cell period in metres.
    static_check : Callable | None
        Optional ``design -> bool`` predicate that returns ``True`` iff the
        design is physically realisable (positive lengths, widths inside the
        microstrip model's validity, etc.).
    pump_voltage_extractor : Callable | None
        Optional ``(design, varactor_params, substrate, n_cells, pump_frequency_Hz,
        pump_amplitude_V=1.0, source_z_ohm=50.0, load_z_ohm=50.0)
        -> ndarray (n_cells,) complex`` function returning the complex
        voltage across the varactor element of each cell at the pump
        frequency under static (``dS = 0``) linear network analysis. When
        ``None``, the pump-coherence filter is unavailable for this
        topology and the optimisation rejects designs from such topologies
        as infeasible (it does not silently treat the missing extractor
        as a pass).
    """

    name: str
    version: str
    description: str
    design_variables: Tuple[DesignVariableSpec, ...]
    build_unit_cell_block_abcd: Callable[..., Any]
    period: Callable[..., float]
    static_check: Callable[..., bool] | None = None
    pump_voltage_extractor: Callable[..., Any] | None = None


_TOPOLOGY_REGISTRY: Dict[str, TopologyDescriptor] = {}


def register_topology(descriptor: TopologyDescriptor) -> None:
    """Register a topology in the global registry.

    Re-registering an already-known name with an identical descriptor is a
    no-op (this keeps module-level registration idempotent under repeated
    imports). Re-registering with a *different* descriptor raises
    ``ValueError`` to prevent silent override.
    """

    if not isinstance(descriptor, TopologyDescriptor):
        raise TypeError("descriptor must be a TopologyDescriptor.")
    existing = _TOPOLOGY_REGISTRY.get(descriptor.name)
    if existing is None:
        _TOPOLOGY_REGISTRY[descriptor.name] = descriptor
        return
    if existing is descriptor:
        return
    raise ValueError(
        f"Topology '{descriptor.name}' is already registered with a different descriptor."
    )


def get_topology(name: str) -> TopologyDescriptor:
    """Look up a registered topology by name."""

    try:
        return _TOPOLOGY_REGISTRY[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(_TOPOLOGY_REGISTRY)) or "<none registered>"
        raise KeyError(f"Unknown topology '{name}'. Registered topologies: {known}") from exc


def list_topologies() -> list[str]:
    """Return sorted names of all currently registered topologies."""

    return sorted(_TOPOLOGY_REGISTRY)


def design_bounds_dict(descriptor: TopologyDescriptor) -> Dict[str, Tuple[float, float]]:
    """Return ``{variable_name: (low, high)}`` for the topology's design space."""

    return {spec.name: (float(spec.lower_bound), float(spec.upper_bound)) for spec in descriptor.design_variables}


