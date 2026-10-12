"""DC return of every varactor terminal (checked before a harmonic-balance deck is emitted).

A varactor junction is biased through DC paths from both of its terminals to ground. A terminal that
reaches ground only through capacitors or open ends (e.g. the anode on an open stub with no bias-return
element) floats: harmonic balance then solves a circuit with an undefined DC operating point. Such a
spec is refused with the reason ``no_dc_return_for_varactor``.

DC conduction per element type (``device_spec`` recognised types): ``MLIN``, ``MTEE``, ``L``, ``R``,
``I_Probe`` connect all their pins; ``VIAGND`` and ``TermG`` connect their pin to ground; ``C``,
``RLC_SERIES``, ``MLEF``, ``MLOC``, ``SDD2P`` and ``NAME`` do not conduct. Ground is ``0`` / ``gnd``, the
spec's ``ground_nodes`` and every source port node (a 50-ohm port is a DC path). The varactor terminals
are pins 1 and 2 of each ``SDD2P``.
"""
from __future__ import annotations

REASON = "no_dc_return_for_varactor"
_CONDUCTING = frozenset({"MLIN", "MTEE", "L", "R", "I_Probe"})
_TO_GROUND = frozenset({"VIAGND", "TermG"})
_GROUND = "__ground__"


class NoDCReturn(ValueError):
    """A varactor terminal has no DC path to ground. The message starts with the reason."""

    reason = REASON


def floating_varactor_terminals(spec) -> list[tuple[str, int, str]]:
    """(element id, pin, node) of every SDD2P terminal without a DC path to ground."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for g in ("0", "gnd", *spec.ground_nodes):
        union(g, _GROUND)
    for s in spec.sources:
        if getattr(s, "port_node", None):
            union(s.port_node, _GROUND)
    for e in spec.elements:
        nodes = [p.node for p in e.pins]
        if e.type in _CONDUCTING:
            for n in nodes[1:]:
                union(nodes[0], n)
        elif e.type in _TO_GROUND:
            for n in nodes:
                union(n, _GROUND)
    ground = find(_GROUND)
    out = []
    for e in spec.elements:
        if e.type == "SDD2P":
            for p in e.pins:
                if p.pin in (1, 2) and find(p.node) != ground:
                    out.append((e.id, p.pin, p.node))
    return out


def check_dc_return(spec) -> None:
    """Raise :class:`NoDCReturn` if a varactor terminal floats at DC."""
    bad = floating_varactor_terminals(spec)
    if bad:
        shown = ", ".join(f"{eid} pin {pin} ({node})" for eid, pin, node in bad[:4])
        more = f" and {len(bad) - 4} more" if len(bad) > 4 else ""
        raise NoDCReturn(
            f"{REASON}: {shown}{more} has no DC path to ground (e.g. an open stub without a bias-return "
            "element); add one (bias_return_ohm) or use a via-terminated topology")
