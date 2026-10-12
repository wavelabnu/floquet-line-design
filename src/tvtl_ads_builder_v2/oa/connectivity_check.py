"""Headless CONNECTIVITY check on an OA-generated (schematic) netlist.

This is the check that was MISSING when earlier OA "send" workspaces
shipped electrically broken: the export checks verified instance counts, the
flat-deck hash, and the presence of NAME wire-labels / MeasEqn cells -- none
of which can see net/terminal connectivity. The schematic was shorted
(``Port:SIGNAL_SRC n_input n_input`` -- both source terminals merged onto
``n_input``) and it went undetected.

This module parses the netlist the SCHEMATIC produces
(``design.generate_netlist()`` -- NOT the flat HB deck) and verifies,
programmatically:

* every P_1Tone source ``Port`` is present and carries its correct ``Num``
  (Port 1 = signal, Port 2 = pump);
* no source is SHORTED -- its two terminals are on DIFFERENT nets and its
  return terminal is global ground;
* no element has a NEW FLOATING terminal (a net touched by exactly one
  terminal) beyond the design's intrinsic open-stub / controller baseline;
* the NAME-annotated nets are present and connected (degree >= 2).

Pure / ADS-free: it takes the netlist TEXT, so it runs in base Python for
unit tests with a captured string AND under ADSPY against a live
``generate_netlist()``. The ADSPY regression test
``tests/parity/test_oa_schematic_connectivity.py`` drives it on the
freshly-emitted anchor; the base-Python unit test
``tests/test_oa_connectivity_check.py`` exercises the parser/checker on
captured good/broken netlists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Net names ADS uses for global ground (the source return must land here).
GROUND_NETS = {"0", "gnd", "gnd!"}

# Controller / non-topological models that carry NO circuit nets (all params),
# so a degree-1 net touched only by these is a controller pseudo-net, not a
# floating circuit terminal.
_NON_TOPO_MODELS = {"S_Param", "HarmonicBalance", "HB", "DC", "AC", "Tran",
                    "MeasEqn", "VarEqn", "Var", "ParamSweep", "Options"}


@dataclass(frozen=True)
class Instance:
    model: str
    name: str
    nets: tuple[str, ...]


def parse_oa_netlist(text: str) -> list[Instance]:
    r"""Parse an ADS/hpeesofsim netlist into ``Instance(model, name, nets)``.

    Continuation lines (trailing ``\``) are joined first. A circuit line is
    ``Model:Inst  net net ...  key=val ...``; the nets are the leading
    whitespace-separated tokens BEFORE the first token that contains ``=``
    (so an SDD body ``I[1,1]=...`` is treated as a param and the 4 SDD nets
    parse correctly). ``model``/``Options``/comment/blank lines are skipped.
    """

    joined: list[str] = []
    buf = ""
    for raw in text.splitlines():
        s = raw.rstrip()
        if s.endswith("\\"):
            buf += s[:-1] + " "
            continue
        buf += s
        joined.append(buf)
        buf = ""
    if buf:
        joined.append(buf)

    out: list[Instance] = []
    for line in joined:
        st = line.strip()
        if not st or st.startswith(";"):
            continue
        if st.startswith("model ") or st.startswith("Options"):
            continue
        head = st.split(None, 1)
        token0 = head[0]
        if ":" not in token0:
            continue
        model, _, name = token0.partition(":")
        rest = head[1] if len(head) > 1 else ""
        nets: list[str] = []
        for tok in rest.split():
            if "=" in tok:
                break
            nets.append(tok)
        out.append(Instance(model=model, name=name, nets=tuple(nets)))
    return out


def _net_degree(insts: list[Instance]) -> dict[str, int]:
    """Number of terminal appearances per net across all instances.

    A net an instance places on two of its OWN terminals (the SDD pin 3/4
    self-tie return) counts twice, so it reads as degree>=2 and is not
    mistaken for a floating terminal.
    """

    deg: dict[str, int] = {}
    for inst in insts:
        for net in inst.nets:
            deg[net] = deg.get(net, 0) + 1
    return deg


def topology_floating_nets(insts: list[Instance]) -> set[str]:
    """Degree-1 nets that sit on a real circuit element (not a controller).

    Controller pseudo-nets (a degree-1 net whose sole touch is a non-topo
    model such as ``S_Param``) are excluded. Genuine open-stub far ends ARE
    degree-1 by design, so callers compare the returned set against a
    known-good baseline (``allowed_floating``) rather than failing on every
    degree-1 net.
    """

    deg = _net_degree(insts)
    net_models: dict[str, set[str]] = {}
    for inst in insts:
        for net in inst.nets:
            net_models.setdefault(net, set()).add(inst.model)
    floating: set[str] = set()
    for net, d in deg.items():
        if net in GROUND_NETS or d != 1:
            continue
        models = net_models.get(net, set())
        if models and models <= _NON_TOPO_MODELS:
            continue
        floating.add(net)
    return floating


@dataclass
class ConnReport:
    ok: bool = True
    failures: list[str] = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    output_port: dict = field(default_factory=dict)
    floating_terminals: list[str] = field(default_factory=list)
    named_nets_present: dict = field(default_factory=dict)
    n_instances: int = 0
    element_kinds: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.failures.append(msg)


def source_nums_from_spec(spec) -> dict[str, int]:
    """``{source_name: Num}`` for every P_1Tone source on the spec."""

    return {s.name: int(s.num) for s in getattr(spec, "sources", ())}


def named_nets_from_spec(spec) -> set[str]:
    """The spec's NAME-annotated nets, minus ground nodes."""

    nets = {e.pins[0].node for e in spec.elements
            if e.type == "NAME" and len(e.pins) == 1}
    return nets - set(getattr(spec, "ground_nodes", ()))


def check_connectivity(
    netlist_text: str,
    *,
    source_nums: dict[str, int],
    named_nets: set[str] | None = None,
    output_port_name: str | None = None,
    output_port_num: int | None = None,
    allowed_floating: set[str] | None = None,
) -> ConnReport:
    """Verify the schematic netlist's CONNECTIVITY.

    Parameters
    ----------
    netlist_text
        Text from ``design.generate_netlist()`` (the schematic netlist).
    source_nums
        ``{source instance name: expected Num}`` (e.g.
        ``{"SIGNAL_SRC": 1, "PUMP_SRC": 2}``). Each must appear as a
        ``Port:<name>`` with two DISTINCT terminals, the return on ground.
    named_nets
        The spec's NAME-annotated nets; each must appear with degree >= 2.
    output_port_name / output_port_num
        The output termination ``Port`` (e.g. ``("P_OUT", 3)``); checked for
        two distinct terminals when given.
    allowed_floating
        Degree-1 nets that are floating BY DESIGN (open-stub far ends, etc.).
        Only degree-1 nets OUTSIDE this set fail. When ``None``, floating
        terminals are recorded but do not fail (baseline-discovery mode).
    """

    rep = ConnReport()
    insts = parse_oa_netlist(netlist_text)
    rep.n_instances = len(insts)
    kinds: dict[str, int] = {}
    for inst in insts:
        kinds[inst.model] = kinds.get(inst.model, 0) + 1
    rep.element_kinds = kinds

    by_name = {inst.name: inst for inst in insts}
    deg = _net_degree(insts)

    # --- sources: present, numbered, not shorted, return on ground --------
    for src_name, num in source_nums.items():
        inst = by_name.get(src_name)
        if inst is None:
            rep.sources[src_name] = {"present": False, "num": num}
            rep.fail(f"source {src_name!r} (Port {num}) MISSING from netlist "
                     f"(dropped -- its terminals floated/were unconnected)")
            continue
        nets = inst.nets
        shorted = len(nets) >= 2 and nets[0] == nets[1]
        rep.sources[src_name] = {"present": True, "num": num,
                                 "nets": list(nets), "model": inst.model,
                                 "shorted": shorted}
        if inst.model != "Port":
            rep.fail(f"source {src_name!r} netlists as {inst.model!r}, expected Port")
        if len(nets) < 2:
            rep.fail(f"source {src_name!r} has {len(nets)} terminal(s); a source "
                     f"is two-terminal")
        elif shorted:
            rep.fail(f"source {src_name!r} (Port {num}) SHORTED: both terminals "
                     f"on net {nets[0]!r}")
        elif nets[1] not in GROUND_NETS:
            rep.fail(f"source {src_name!r} (Port {num}) return terminal on "
                     f"{nets[1]!r}, expected ground {sorted(GROUND_NETS)}")

    # --- output termination port -----------------------------------------
    if output_port_name is not None:
        inst = by_name.get(output_port_name)
        if inst is None:
            rep.fail(f"output port {output_port_name!r} MISSING")
        else:
            rep.output_port[output_port_name] = {
                "num": output_port_num, "nets": list(inst.nets)}
            if len(inst.nets) >= 2 and inst.nets[0] == inst.nets[1]:
                rep.fail(f"output port {output_port_name!r} SHORTED: both "
                         f"terminals on {inst.nets[0]!r}")

    # --- floating terminals (vs the known-good open-stub baseline) --------
    floating = topology_floating_nets(insts)
    rep.floating_terminals = sorted(floating)
    if allowed_floating is not None:
        new_floaters = sorted(floating - set(allowed_floating))
        if new_floaters:
            rep.fail(f"{len(new_floaters)} NEW floating terminal net(s) beyond "
                     f"the known-good open-stub/controller baseline: "
                     f"{new_floaters[:12]}"
                     + (" ..." if len(new_floaters) > 12 else ""))

    # --- named nets present & connected ----------------------------------
    if named_nets:
        for net in sorted(named_nets):
            if net in GROUND_NETS:
                continue
            d = deg.get(net, 0)
            rep.named_nets_present[net] = d
            if d == 0:
                rep.fail(f"named net {net!r} ABSENT from netlist")
            elif d < 2:
                rep.fail(f"named net {net!r} has degree {d} (floating)")

    return rep


def summarize(rep: ConnReport) -> str:
    lines = [f"  connectivity: {'PASS' if rep.ok else 'FAIL'}  "
             f"({rep.n_instances} instances)"]
    for name, info in rep.sources.items():
        if not info.get("present"):
            lines.append(f"    {name}: MISSING (Port {info['num']})")
        else:
            tag = "SHORTED" if info.get("shorted") else "ok"
            lines.append(f"    {name} (Port {info['num']}): nets={info['nets']} [{tag}]")
    for name, info in rep.output_port.items():
        lines.append(f"    {name} (Port {info['num']}): nets={info['nets']}")
    if rep.floating_terminals:
        lines.append(f"    floating terminals: {rep.floating_terminals[:12]}")
    if not rep.ok:
        for f in rep.failures:
            lines.append(f"    FAIL: {f}")
    return "\n".join(lines)


