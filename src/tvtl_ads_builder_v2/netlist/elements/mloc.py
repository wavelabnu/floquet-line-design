"""Emit an ``MLOC`` (microstrip open-circuited stub) element line for hpeesofsim.

``MLOC`` is the ADS Libra open-circuited-stub element. An emitter-accuracy
probe verified on ADS
2026 U1.2 that flat-deck ``MLOC`` is an **ideal open** -- it adds *no* open-end
fringing capacitance over a dangling-pin ``MLIN`` (S11 phase delta 0.0000 deg, 2-11
GHz); the open-end fringe lives in the separate ``MLEF`` element. Syntax (1-port):

    MLOC:<id> <n1> Subst="<name>" W=<mm> mm L=<mm> mm

Appended at a stub's open far end with ``L=0`` it is electrically inert (an ideal
open at the connection point) -- the companion ``MLIN`` already carries the stub
length. The element is emitted anyway so the ADS-side deck for the *VIAGND + MLOC*
convention literally contains the ``MLOC`` the REFERENCE ADS workflow uses (the
model-convention comparison). This makes ``MLOC`` vs ``MLEF`` a clean
ideal-open-vs-fringe swap at the same node.
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import Element


def emit_mloc(
    element: Element,
    substrate_name: str,
    resolve_node: Callable[[str], str] = lambda node: node,
) -> str:
    """Return a single-line ``MLOC`` directive for ``element`` (1 pin)."""

    if element.type != "MLOC":
        raise ValueError(f"emit_mloc called on element of type '{element.type}'.")
    if len(element.pins) != 1:
        raise ValueError(
            f"MLOC '{element.id}' must declare exactly one pin (got {len(element.pins)})."
        )
    substrate_ref = str(element.parameters.get("substrate_ref", substrate_name))
    if substrate_ref != substrate_name:
        raise ValueError(
            f"MLOC '{element.id}' references substrate '{substrate_ref}' but the "
            f"netlist declares only '{substrate_name}'."
        )
    width_mm = float(element.parameters["w_m"]) * 1.0e3
    length_mm = float(element.parameters.get("l_m", 0.0)) * 1.0e3
    node1 = resolve_node(element.pins[0].node)
    return (
        f"MLOC:{element.id} {node1} "
        f'Subst="{substrate_name}" '
        f"W={width_mm:.12g} mm "
        f"L={length_mm:.12g} mm"
    )


