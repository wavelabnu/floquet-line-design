"""Emit lumped ``R``, ``L``, ``C`` and composite ``RLC_SERIES`` lines for hpeesofsim.

``R``, ``L`` and ``C`` are the standard ADS two-terminal lumped
elements. Each emits a single netlist line; the value carries an
explicit unit (``Ohm`` / ``nH`` / ``pF``) so the simulator never has to
guess, and so the OA emitter -- which sets the identical value string
on the ADS instance -- yields a netlist-equivalent deck.

``RLC_SERIES`` has no single ADS primitive. It expands to a series
``R -- L -- C`` chain through two auto-generated internal nodes,
``<id>_RL_node`` and ``<id>_LC_node``. :func:`expand_rlc_series` is the
single source of truth for that expansion; both this emitter and the OA
emitter call it, so the two backends place an identical R-L-C topology
with identical internal-node names and the netlist cross-check stays
meaningful.
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import Element, ElementPin


# Internal-node name suffixes for an expanded RLC_SERIES. Deterministic
# and derived from the element id so the netlist comparator can pair the
# hpeesofsim and OA intermediate nets across the two backends.
RLC_RL_NODE_SUFFIX = "_RL_node"
RLC_LC_NODE_SUFFIX = "_LC_node"


def _ordered_two_terminal_nodes(element: Element) -> tuple[str, str]:
    """Return ``(pin-1 node, pin-2 node)`` for a two-terminal element."""

    if len(element.pins) != 2:
        raise ValueError(
            f"{element.type} '{element.id}' must declare exactly two pins "
            f"(got {len(element.pins)})."
        )
    by_pin = {pin.pin: pin.node for pin in element.pins}
    if set(by_pin) != {1, 2}:
        raise ValueError(
            f"{element.type} '{element.id}' must declare pins 1 and 2 "
            f"(got {sorted(by_pin)})."
        )
    return by_pin[1], by_pin[2]


def format_r_value(element: Element) -> str:
    """Return the ADS value string for an ``R`` element, e.g. ``'50 Ohm'``."""

    return f"{float(element.parameters['r_ohm']):.12g} Ohm"


def format_l_value(element: Element) -> str:
    """Return the ADS value string for an ``L`` element, e.g. ``'1 nH'``.

    The device spec stores inductance in henries (``l_h``); ADS and the
    hpeesofsim deck use nanohenries, so the stored value is scaled by 1e9.
    """

    return f"{float(element.parameters['l_h']) * 1.0e9:.12g} nH"


def format_c_value(element: Element) -> str:
    """Return the ADS value string for a ``C`` element, e.g. ``'1 pF'``.

    The device spec stores capacitance in farads (``c_f``); ADS and the
    hpeesofsim deck use picofarads, so the stored value is scaled by 1e12.
    """

    return f"{float(element.parameters['c_f']) * 1.0e12:.12g} pF"


def emit_r(
    element: Element, resolve_node: Callable[[str], str] = lambda node: node
) -> str:
    """Return a single-line ``R`` (resistor) directive for ``element``.

    Parameters
    ----------
    element : Element
        Must have ``type == "R"`` and exactly two pins.
    resolve_node : callable
        Maps a device-spec node id to the netlist token (translates a
        ground node to the literal ``0``). Defaults to the identity.
    """

    if element.type != "R":
        raise ValueError(f"emit_r called on element of type '{element.type}'.")
    node1, node2 = _ordered_two_terminal_nodes(element)
    return (
        f"R:{element.id} {resolve_node(node1)} {resolve_node(node2)} "
        f"R={format_r_value(element)}"
    )


def emit_l(
    element: Element, resolve_node: Callable[[str], str] = lambda node: node
) -> str:
    """Return a single-line ``L`` (inductor) directive for ``element``."""

    if element.type != "L":
        raise ValueError(f"emit_l called on element of type '{element.type}'.")
    node1, node2 = _ordered_two_terminal_nodes(element)
    return (
        f"L:{element.id} {resolve_node(node1)} {resolve_node(node2)} "
        f"L={format_l_value(element)}"
    )


def emit_c(
    element: Element, resolve_node: Callable[[str], str] = lambda node: node
) -> str:
    """Return a single-line ``C`` (capacitor) directive for ``element``."""

    if element.type != "C":
        raise ValueError(f"emit_c called on element of type '{element.type}'.")
    node1, node2 = _ordered_two_terminal_nodes(element)
    return (
        f"C:{element.id} {resolve_node(node1)} {resolve_node(node2)} "
        f"C={format_c_value(element)}"
    )


def expand_rlc_series(element: Element) -> tuple[Element, Element, Element]:
    """Expand an ``RLC_SERIES`` element into its ``(R, L, C)`` sub-elements.

    The series chain runs ``pin 1 -> R -> L -> C -> pin 2`` through two
    auto-generated internal nodes, ``<id>_RL_node`` and ``<id>_LC_node``.
    The three returned elements have ids ``<id>_R``, ``<id>_L`` and
    ``<id>_C`` and carry only their own one parameter each.

    Both the hpeesofsim emitter (:func:`emit_rlc_series`) and the OA
    emitter call this, so the two backends place an identical R-L-C
    topology with identical internal-node names; the netlist comparator
    can then pair the intermediate nets deterministically.
    """

    if element.type != "RLC_SERIES":
        raise ValueError(
            f"expand_rlc_series called on element of type '{element.type}'."
        )
    node_a, node_b = _ordered_two_terminal_nodes(element)
    rl_node = f"{element.id}{RLC_RL_NODE_SUFFIX}"
    lc_node = f"{element.id}{RLC_LC_NODE_SUFFIX}"
    params = element.parameters
    resistor = Element(
        id=f"{element.id}_R",
        type="R",
        pins=(ElementPin(pin=1, node=node_a), ElementPin(pin=2, node=rl_node)),
        parameters={"r_ohm": float(params["r_ohm"])},
    )
    inductor = Element(
        id=f"{element.id}_L",
        type="L",
        pins=(ElementPin(pin=1, node=rl_node), ElementPin(pin=2, node=lc_node)),
        parameters={"l_h": float(params["l_h"])},
    )
    capacitor = Element(
        id=f"{element.id}_C",
        type="C",
        pins=(ElementPin(pin=1, node=lc_node), ElementPin(pin=2, node=node_b)),
        parameters={"c_f": float(params["c_f"])},
    )
    return (resistor, inductor, capacitor)


def emit_rlc_series(
    element: Element, resolve_node: Callable[[str], str] = lambda node: node
) -> str:
    """Return the three-line ``R`` / ``L`` / ``C`` expansion of an RLC_SERIES.

    The lines are joined by a single ``\\n``; the caller in
    :mod:`tvtl_ads_builder_v2.netlist.emit` appends the trailing newline
    just as it does for any other element.
    """

    if element.type != "RLC_SERIES":
        raise ValueError(
            f"emit_rlc_series called on element of type '{element.type}'."
        )
    resistor, inductor, capacitor = expand_rlc_series(element)
    return "\n".join(
        (
            emit_r(resistor, resolve_node),
            emit_l(inductor, resolve_node),
            emit_c(capacitor, resolve_node),
        )
    )


