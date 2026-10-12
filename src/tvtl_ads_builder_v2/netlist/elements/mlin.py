"""Emit an ``MLIN`` element line for hpeesofsim.

The MLIN model is the ADS native ``MLIN`` (alias ``MLIN2`` in some
versions) at the documented parameter list: substrate by name reference,
strip width ``W``, length ``L``. Units are stated explicitly (``mm``).
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import Element


def emit_mlin(
    element: Element,
    substrate_name: str,
    resolve_node: Callable[[str], str] = lambda node: node,
) -> str:
    """Return a single-line MLIN directive for ``element``.

    Parameters
    ----------
    element : Element
        Must have ``type == "MLIN"`` and two pins (1, 2) declared.
    substrate_name : str
        Name of the substrate the netlist's MSUB line declared. The
        emitter checks ``element.parameters["substrate_ref"]`` matches
        this name (the validator has already done the same check; the
        defensive recheck here protects against direct calls).
    resolve_node : callable
        Maps a device-spec node id to the netlist token (translates
        ground nodes to the literal ``0``). Defaults to the identity.
    """

    if element.type != "MLIN":
        raise ValueError(f"emit_mlin called on element of type '{element.type}'.")
    if len(element.pins) != 2:
        raise ValueError(
            f"MLIN '{element.id}' must declare exactly two pins (got {len(element.pins)})."
        )
    substrate_ref = str(element.parameters.get("substrate_ref", substrate_name))
    if substrate_ref != substrate_name:
        raise ValueError(
            f"MLIN '{element.id}' references substrate '{substrate_ref}' but the "
            f"netlist declares only '{substrate_name}'."
        )
    width_mm = float(element.parameters["w_m"]) * 1.0e3
    length_mm = float(element.parameters["l_m"]) * 1.0e3
    pins = {pin.pin: resolve_node(pin.node) for pin in element.pins}
    return (
        f"MLIN:{element.id} {pins[1]} {pins[2]} "
        f'Subst="{substrate_name}" '
        f"W={width_mm:.12g} mm "
        f"L={length_mm:.12g} mm"
    )


