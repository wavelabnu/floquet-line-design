"""Emit a ``VIAGND`` (microstrip via-to-ground) element line for hpeesofsim.

``VIAGND`` is the ADS Libra physical via-to-ground element: a **1-port** that
grounds internally (no second pin to a ground net). An emitter-accuracy
probe verified on ADS
2026 U1.2 that ``D`` (via barrel diameter) and ``T`` (metal thickness) are
required; ``W`` (the connecting conductor width) and ``Rho`` (resistivity relative
to gold) are optional. Syntax:

    VIAGND:<id> <n1> Subst="<name>" D=<mm> mm W=<mm> mm T=<mm> mm Rho=<x>

On Rogers 4003C (h 1.54 mm, via D 1.6 mm) it presents ~0.19 nH -- ~3x *below* the
lumped Bahl 0.569 nH the reduced model uses. It is the physical-via half of the
*VIAGND + MLOC* convention in the model-convention comparison; the matching
analytical termination is :func:`tvtl.microstrip.discontinuities.via_impedance_ohm`
(Goldfarb-Pucel), which presents the same ~0.19 nH equivalent.

``W`` / ``Rho`` are emitted only when present in ``element.parameters`` so the
element degenerates to the required-only ``D`` / ``T`` form when those physical
estimates are not supplied.
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import Element


def emit_viagnd(
    element: Element,
    substrate_name: str,
    resolve_node: Callable[[str], str] = lambda node: node,
) -> str:
    """Return a single-line ``VIAGND`` directive for ``element`` (1 pin, grounds internally)."""

    if element.type != "VIAGND":
        raise ValueError(
            f"emit_viagnd called on element of type '{element.type}'."
        )
    if len(element.pins) != 1:
        raise ValueError(
            f"VIAGND '{element.id}' must declare exactly one pin "
            f"(got {len(element.pins)})."
        )
    substrate_ref = str(element.parameters.get("substrate_ref", substrate_name))
    if substrate_ref != substrate_name:
        raise ValueError(
            f"VIAGND '{element.id}' references substrate '{substrate_ref}' but the "
            f"netlist declares only '{substrate_name}'."
        )
    diameter_mm = float(element.parameters["d_m"]) * 1.0e3
    thickness_mm = float(element.parameters["t_m"]) * 1.0e3
    node1 = resolve_node(element.pins[0].node)
    parts = [
        f"VIAGND:{element.id} {node1} ",
        f'Subst="{substrate_name}" ',
        f"D={diameter_mm:.12g} mm ",
    ]
    if element.parameters.get("w_m") is not None:
        width_mm = float(element.parameters["w_m"]) * 1.0e3
        parts.append(f"W={width_mm:.12g} mm ")
    parts.append(f"T={thickness_mm:.12g} mm")
    if element.parameters.get("rho") is not None:
        rho = float(element.parameters["rho"])
        parts.append(f" Rho={rho:.12g}")
    return "".join(parts)


