"""Emit an ``MLEF`` (microstrip open-end effect) element line for hpeesofsim.

``MLEF`` is the ADS Libra open-end-effect element: a 1-port appended at the open far
end of a stub ``MLIN`` that supplies the open-end fringing capacitance (the
emitter-accuracy pass verified that ``MLOC`` is an *ideal* open and ``MLEF`` is the
element carrying the fringe). Syntax (1-port, verified on ADS 2026 U1.2):

    MLEF:<id> <n1> Subst="<name>" W=<mm> mm L=<mm> mm

The companion stub ``MLIN`` carries the line length; the appended ``MLEF`` (typically
``L=0``) adds the fringe from its width ``W``.
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import Element


def emit_mlef(
    element: Element,
    substrate_name: str,
    resolve_node: Callable[[str], str] = lambda node: node,
) -> str:
    """Return a single-line ``MLEF`` directive for ``element`` (1 pin)."""

    if element.type != "MLEF":
        raise ValueError(f"emit_mlef called on element of type '{element.type}'.")
    if len(element.pins) != 1:
        raise ValueError(
            f"MLEF '{element.id}' must declare exactly one pin (got {len(element.pins)})."
        )
    substrate_ref = str(element.parameters.get("substrate_ref", substrate_name))
    if substrate_ref != substrate_name:
        raise ValueError(
            f"MLEF '{element.id}' references substrate '{substrate_ref}' but the "
            f"netlist declares only '{substrate_name}'."
        )
    width_mm = float(element.parameters["w_m"]) * 1.0e3
    length_mm = float(element.parameters.get("l_m", 0.0)) * 1.0e3
    node1 = resolve_node(element.pins[0].node)
    return (
        f"MLEF:{element.id} {node1} "
        f'Subst="{substrate_name}" '
        f"W={width_mm:.12g} mm "
        f"L={length_mm:.12g} mm"
    )


