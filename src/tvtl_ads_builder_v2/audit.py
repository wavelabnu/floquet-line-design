"""Per-circuit netlist adjacency audit -- reusable, ADS-free.

The cross-check verifies that the hpeesofsim and OA-Python backends
emit *equivalent* netlists, but not that each component's pins land on
*distinct, sensible* nets. This module audits a single circuit's
hpeesofsim flat netlist on a per-component basis and flags suspicious
adjacency patterns:

* INTERNAL SHORT    -- two or more pins of one component on one net,
  *unless* that net is a pure self-tie: a net every one of whose pins
  belongs to that single component is an intentional self-tie (e.g. an
  SDD2P's port-2 terminals tied to one shared return net) and is not
  flagged;
* SHORTED TO GROUND -- a two-terminal component with both pins on the
  ground net;
* PIN COUNT         -- a component emitted with the wrong number of net
  tokens for its type;
* DANGLING NET      -- a net with exactly one pin attached that is
  neither the ground net, a declared open termination, nor a node
  driven by a ``P_1Tone`` source.

``spec.ground_nodes``, ``spec.open_terminations`` and ``spec.sources``
are honoured: the ground net, any declared open termination, and any
node a ``P_1Tone`` source drives are exempt from the DANGLING NET
flag. A source's ``P_1Tone:`` line is not parsed as a routed component
(it is not one of :data:`_EXPECTED_TERMINALS`' types), so a
source-driven node would otherwise look one-pin-short to the audit.

:func:`audit_circuit` is the single entry point. It is used by the v2
build pipeline, which writes a per-circuit ``*_adjacency.txt`` on every
build.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .device_spec import DeviceSpec
from .oa.netlist_compare import parse_netlist_text

# hpeesofsim ground-reference net.
_GROUND_NET = "0"

# Expected terminal count per parsed component type. Used only to flag a
# component emitted with the wrong number of net tokens. SDD2P is a
# 2-port symbolically-defined device -> 4 terminals.
_EXPECTED_TERMINALS = {
    "PORT": 2, "MLIN": 2, "MTEE": 3, "R": 2, "L": 2, "C": 2, "SDD2P": 4,
}

# Parsed entries that are model declarations, not routed components.
_MODEL_CARD_TYPES = {"MSUB"}


@dataclass(frozen=True)
class CircuitAuditReport:
    """The adjacency-audit result for one circuit.

    ``flags`` is ordered most-severe first (internal shorts, then
    ground shorts, pin-count anomalies, dangling nets). ``report_text``
    is the full human-readable per-circuit report, including the
    annotated nets section.
    """

    circuit_name: str
    flags: tuple[str, ...]
    report_text: str

    @property
    def is_clean(self) -> bool:
        """True when the circuit raised no flag."""

        return not self.flags


def audit_circuit(
    spec: DeviceSpec,
    netlist_text: str,
    *,
    circuit_name: str | None = None,
) -> CircuitAuditReport:
    """Run the adjacency audit on one circuit.

    Parameters
    ----------
    spec : DeviceSpec
        The device spec; ``ground_nodes`` and ``open_terminations`` are
        read so their nets are exempt from the dangling-net flag.
    netlist_text : str
        The circuit's hpeesofsim flat netlist (the caller emits it once,
        e.g. via ``emit_hpeesofsim_netlist``).
    circuit_name : str, optional
        Name shown in the report header; defaults to ``spec.label``.

    Returns
    -------
    CircuitAuditReport
        The flags and the rendered per-circuit report.
    """

    name = circuit_name or spec.label
    parsed = parse_netlist_text(netlist_text)
    components = [c for c in parsed.elements if c.ctype not in _MODEL_CARD_TYPES]
    net_pins = _net_to_pins(components)
    open_terminations = set(spec.open_terminations)
    source_nodes = {source.port_node for source in spec.sources}
    exempt_one_pin = open_terminations | source_nodes
    flags = _compute_flags(components, net_pins, exempt_one_pin)
    report_text = _render_report(
        name, components, net_pins, flags, open_terminations, source_nodes
    )
    return CircuitAuditReport(
        circuit_name=name, flags=tuple(flags), report_text=report_text
    )


def _net_to_pins(components):
    """Map net name -> [(component_name, pin_index), ...] (pin_index 1-based)."""

    net_pins = defaultdict(list)
    for comp in components:
        for pin_index, net in enumerate(comp.terminals, start=1):
            net_pins[net].append((comp.name, pin_index))
    return net_pins


def _compute_flags(components, net_pins, exempt_one_pin):
    """Return a circuit's flag list, ordered most-severe first.

    ``exempt_one_pin`` is the set of net names a one-pin net may
    legitimately carry without being flagged as dangling: the spec's
    declared open terminations plus any node a ``P_1Tone`` source
    drives (a source line is not parsed as a routed component, so a
    source-only node would otherwise look one-pin-short).
    """

    shorts = []
    ground_shorts = []
    pin_counts = []
    danglings = []

    for comp in components:
        terminals = comp.terminals

        # Flag 1 -- two or more pins of THIS component on the same net.
        # A net whose pins ALL belong to this one component is an
        # intentional self-tie (e.g. an SDD2P port-2 tied to one shared
        # return net), not an internal short -- it is skipped. A net
        # also carrying another component's pin is still flagged.
        pins_by_net = defaultdict(list)
        for pin_index, net in enumerate(terminals, start=1):
            pins_by_net[net].append(pin_index)
        for net, pin_indices in pins_by_net.items():
            if len(pin_indices) < 2:
                continue
            net_owners = {cname for cname, _ in net_pins.get(net, [])}
            if net_owners == {comp.name}:
                # Pure self-tie -- every pin on this net is this
                # component's; an intentional internal tie, not a short.
                continue
            shorts.append(
                "INTERNAL SHORT: %s (%s) pins %s are all on net '%s'"
                % (
                    comp.name,
                    comp.ctype,
                    ", ".join("t%d" % i for i in pin_indices),
                    net,
                )
            )

        # Flag 4 -- a two-terminal component with BOTH pins on ground.
        if len(terminals) == 2 and all(t == _GROUND_NET for t in terminals):
            ground_shorts.append(
                "SHORTED TO GROUND: %s (%s) has both pins on ground net '%s'"
                % (comp.name, comp.ctype, _GROUND_NET)
            )

        # Flag 2 -- wrong number of net tokens for the component type.
        expected = _EXPECTED_TERMINALS.get(comp.ctype)
        if expected is not None and len(terminals) != expected:
            detail = (
                "unconnected pin(s)"
                if len(terminals) < expected
                else "extra terminal(s)"
            )
            pin_counts.append(
                "PIN COUNT: %s (%s) has %d net token(s), expected %d -- %s"
                % (comp.name, comp.ctype, len(terminals), expected, detail)
            )

    # Flag 3 -- a net with exactly one pin attached. Ground is excluded,
    # and so is any net in ``exempt_one_pin`` (a declared open-circuit
    # termination such as an open-stub end, or a P_1Tone-source-driven
    # node).
    for net in sorted(net_pins):
        if net == _GROUND_NET or net in exempt_one_pin:
            continue
        attached = net_pins[net]
        if len(attached) == 1:
            cname, cidx = attached[0]
            danglings.append(
                "DANGLING NET: net '%s' connects only %s.t%d -- a one-pin "
                "net not declared as an open termination"
                % (net, cname, cidx)
            )

    return shorts + ground_shorts + pin_counts + danglings


def _render_report(
    name, components, net_pins, flags, open_terminations, source_nodes
):
    """Render one circuit's per-component adjacency report."""

    lines = []
    lines.append("Circuit: %s" % name)
    lines.append("Backend: hpeesofsim")
    lines.append("Source : emit_hpeesofsim_netlist(<canonical DeviceSpec>)")
    lines.append("")
    lines.append("%-22s %-6s %s" % ("Component", "Pin", "Net"))
    lines.append("-" * 52)
    for comp in components:
        for pin_index, net in enumerate(comp.terminals, start=1):
            lines.append("%-22s %-6s %s" % (comp.name, "t%d" % pin_index, net))
    lines.append("")
    lines.append("Nets (net -> connected pins):")
    for net in sorted(net_pins):
        attached = ", ".join("%s.t%d" % (c, i) for c, i in net_pins[net])
        if net in open_terminations:
            note = "    [declared open termination]"
        elif net in source_nodes:
            note = "    [P_1Tone source-driven node]"
        else:
            note = ""
        lines.append("  %-18s %s%s" % (net, attached, note))
    lines.append("")
    lines.append("Flags:")
    if flags:
        for flag in flags:
            lines.append("  - %s" % flag)
    else:
        lines.append("  (none)")
    lines.append("")
    return "\n".join(lines)


