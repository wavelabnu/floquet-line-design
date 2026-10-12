"""Emit a single-terminal port line for hpeesofsim.

The device-spec uses ``TermG`` as the canonical element type name (it
is the matching ADS schematic symbol). In the flat-deck syntax
hpeesofsim accepts, the corresponding directive is ``Port:Name <node>
0 Num=<port#> Z=<ohms> Ohm Noise=no``; the second node is the implicit
simulator ground (``0``).
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import Element


def emit_termg(
    element: Element,
    resolve_node: Callable[[str], str] = lambda node: node,
) -> str:
    """Return a single-line port directive for a TermG element.

    Parameters
    ----------
    element : Element
        Must have ``type == "TermG"`` and exactly one pin.
    resolve_node : callable
        Maps a device-spec node id to the netlist token. Defaults to
        the identity. (A port node is never a ground node -- the
        validator forbids it -- but the resolver is threaded through
        uniformly with the other element emitters.)
    """

    if element.type != "TermG":
        raise ValueError(f"emit_termg called on element of type '{element.type}'.")
    if len(element.pins) != 1:
        raise ValueError(
            f"TermG '{element.id}' must declare exactly one pin (got {len(element.pins)})."
        )
    pin = element.pins[0]
    z_ohm = float(element.parameters["z_ohm"])
    num = int(element.parameters["num"])
    return (
        f"Port:{element.id} {resolve_node(pin.node)} 0 "
        f"Num={num} Z={z_ohm:.12g} Ohm Noise=no"
    )


