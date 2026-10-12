"""Netlist-level equivalence cross-check for the OA-Python backend.

Every workspace the OA-Python emitter produces must be electrically
equivalent to the hpeesofsim flat netlist built from the same device
spec. This module parses both an ADS-generated netlist and an
hpeesofsim ``.ckt`` deck into one normalized representation and
compares them. A divergence is an emitter bug, caught at build time.

The two formats share a grammar:

* ``;`` comment lines;
* ``model <name> <TYPE> <params...>``      -- a model declaration;
* ``<TYPE>:<name> <nets...> <params...>``  -- an instance line;
* a trailing ``\\`` continues a logical line.

The ADS netlist additionally carries ``Options`` / ``#load`` /
``Component`` / ``OutputPlan`` directives that do not describe the
circuit; those are collected as ``ignored`` and never affect
equivalence.

A P_1Tone power source is likewise collected as ``ignored``: the
hpeesofsim emitter writes it as a ``P_1Tone:`` line, ADS 2026
normalises the OA cell to a ``Port:`` line with a tone-power
parameter, and either way it is an analysis stimulus rather than a
routed element (the same status ``HB:`` has). Excluding both forms
keeps the two backends' element inventories equal despite the token
asymmetry; ``p1tone_content_check`` is the source's authoritative
cross-check. A *sink* ``Port:`` -- an output-port termination with no
tone-power parameter -- stays a counted element.

An ``I_Probe`` current probe reaches a netlist as a ``Short:`` line
(ADS normalises the ``ads_rflib/I_Probe`` cell to its multi-mode
``Short`` element); the cross-check keys on the ``Mode`` parameter, not
the bare ``Short`` token, to tell the modes apart. ``Short ... Mode=0``
is the current probe and normalizes to the counted element type
``I_PROBE``; a ``Short`` of any other mode (e.g. ``Mode=1`` = a DC
block) is out of scope and collected as ``ignored``. Both backends emit
the same ``Short ... Mode=0`` line, so the two are paired directly.

Equivalence is topological and parametric, tolerant only of
differences that *provably* mean the same circuit:

* type aliases   -- ADS ``MLIN2`` == hpeesofsim ``MLIN``;
* terminal order -- a two-terminal element's nets may be listed in
                    either order;
* net relabeling -- net *names* may differ (``n_in`` vs an auto-named
                    ``N__0``) as long as the port-anchored net
                    topology is isomorphic;
* sweep form     -- a ``SweepPlan`` ``Lin=N`` point count and a
                    ``Step=`` increment over the same ``Start``/``Stop``
                    describe one frequency grid;
* extra params   -- a parameter present on only one side (ADS emits
                    many model defaults the flat deck omits) is
                    reported as a note, not a mismatch.

A *shared* parameter whose value differs, an element-count difference,
or a topology difference IS a mismatch.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# Component types that are real circuit elements (vs analysis controllers
# or structural references). GROUND is deliberately NOT here: a ground
# symbol is a net reference, not a device -- the hpeesofsim deck emits no
# GROUND line at all (ground is just net "0"), so a GROUND in the ADS
# netlist is routed to `ignored` and never counted.
_ELEMENT_TYPES = {
    "MSUB", "PORT", "MLIN", "MTEE", "MLOC", "VIAGND",
    "R", "L", "C", "SDD2P", "I_PROBE",
}
_CONTROLLER_TYPES = {"S_PARAM", "SWEEPPLAN"}

# Normalized type aliases: the ADS netlister's simulator-model name on
# the left, the canonical device-spec name on the right. The MTEE
# aliases are best-effort -- the exact token ADS emits for an MTEE_ADS
# instance is confirmed when D3/D4 are first cross-checked.
_TYPE_ALIASES = {
    "MLIN2": "MLIN",
    "MTEE_ADS": "MTEE",
    "MTEE2": "MTEE",
    "S_PARAM": "S_PARAM",
    # The ADS netlister emits a Symbolically-Defined Device with the
    # element token "SDD" (the hpeesofsim emitter matches it); both
    # normalize to the device-spec element type "SDD2P".
    "SDD": "SDD2P",
}

# Bare tokens that are units belonging to the preceding ``key=value``.
_KNOWN_UNITS = {
    "mm", "mil", "um", "cm", "m", "nm",
    "ohm", "kohm", "mohm",
    "ghz", "mhz", "khz", "hz", "thz",
    "pf", "nf", "uf", "ff", "ph", "nh", "uh", "mh",
    "s", "sec", "v", "a", "db", "dbm", "deg", "rad",
}

# Frequency-unit scale factors to Hz, for SweepPlan grid normalization.
_FREQ_SCALE = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}

# A tone-power parameter key -- ``P`` or an array-indexed ``P[1]``. Its
# presence on a ``Port:`` line marks the line as a P_1Tone *source*
# (ADS 2026 normalises the ads_sources/P_1Tone OA cell to a ``Port:``
# line), as opposed to a sink output-port termination.
_TONE_POWER_KEY_RE = re.compile(r"P(\[\d+\])?\Z")


@dataclass(frozen=True)
class Component:
    """One parsed netlist entry (a model, an element, or a controller)."""

    ctype: str                      # normalized type, e.g. "MLIN", "PORT"
    name: str                       # instance / model name
    terminals: tuple[str, ...]      # net tokens, in listed order
    params: dict[str, str]          # raw "key -> value" (value may carry a unit)


@dataclass
class NormalizedCircuit:
    """A netlist normalized into elements, controllers and ignored lines."""

    elements: list[Component] = field(default_factory=list)
    controllers: list[Component] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Outcome of :func:`compare_circuits`."""

    match: bool
    differences: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'[^\s"]*"[^"]*"[^\s"]*|\S+')


def _logical_lines(text: str) -> list[str]:
    """Join ``\\``-continued physical lines into logical statements."""

    statements: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buffer += line[:-1].rstrip() + " "
            continue
        buffer += line
        statements.append(buffer.strip())
        buffer = ""
    if buffer.strip():
        statements.append(buffer.strip())
    return statements


def _tokenize(statement: str) -> list[str]:
    """Split a statement into tokens, keeping ``key="quoted value"`` whole."""

    return _TOKEN_RE.findall(statement)


def _normalize_type(raw: str) -> str:
    upper = raw.strip().upper()
    return _TYPE_ALIASES.get(upper, upper)


def _split_nets_and_params(tokens: list[str]) -> tuple[tuple[str, ...], dict[str, str]]:
    """Leading bare tokens are nets; from the first ``key=`` on, params."""

    nets: list[str] = []
    first_param = len(tokens)
    for index, token in enumerate(tokens):
        if "=" in token:
            first_param = index
            break
        nets.append(token)
    params = _parse_params(tokens[first_param:])
    return tuple(nets), params


def _parse_params(tokens: list[str]) -> dict[str, str]:
    """Parse ``key=value`` tokens, attaching a trailing bare unit token."""

    params: dict[str, str] = {}
    last_key: str | None = None
    for token in tokens:
        if "=" in token:
            key, _, value = token.partition("=")
            params[key] = value
            last_key = key
        elif last_key is not None and token.lower() in _KNOWN_UNITS:
            params[last_key] = f"{params[last_key]} {token}"
            last_key = None
        else:
            last_key = None
    return params


def _classify_short(params: dict[str, str]) -> str | None:
    """Classify a ``Short:`` netlist line by its ``Mode`` parameter.

    ADS netlists several distinct components through one multi-mode
    ``Short`` element; the ``Mode`` parameter selects which. The
    classification keys on ``Mode``, NOT on the bare ``Short`` token, so
    the cross-check distinguishes them:

    * ``Mode=0`` -- an ideal 0-ohm short used as a current probe. This
      is the device-spec ``I_Probe`` element; it
      normalizes to the canonical type ``I_PROBE`` and is a counted
      element.
    * ``Mode=1`` (carrying a ``C``) -- a DC block. Out of scope this
      round; returns ``None`` so the line is collected as an ignored
      directive. A future round adds ``"DC_BLOCK"`` here and to
      :data:`_ELEMENT_TYPES` -- a one-line change, no re-architecting.

    Returns the canonical element type, or ``None`` when the ``Short``
    line is not an in-scope circuit element.
    """

    mode = params.get("Mode")
    if mode is None:
        return None
    value = _norm_value(mode)
    if value[0] == "n" and value[1] == 0.0:
        return "I_PROBE"
    return None


def _is_p1tone_source(ctype: str, params: dict[str, str]) -> bool:
    """True if a parsed line is a P_1Tone power-source emission.

    A P_1Tone source reaches a netlist two ways: the hpeesofsim flat
    emitter writes it as a ``P_1Tone:`` line; ADS 2026 normalises the OA
    ``ads_sources/P_1Tone`` cell to a ``Port:`` line carrying a
    tone-power parameter (``P[1]``). Both are analysis *stimuli*, not
    routed circuit elements -- the same status the ``HB:`` controller
    line has -- so the structural comparator excludes them from the
    element inventory and the dedicated ``p1tone_content_check`` is
    their authoritative cross-check. Excluding both forms keeps the two
    backends' element counts equal despite the ``P_1Tone:`` vs ``Port:``
    token asymmetry.

    A *sink* ``Port:`` -- an output-port termination, ``Num`` / ``Z`` /
    ``Noise`` with no tone-power parameter -- is a real element and is
    NOT matched here. The source/sink decision is by parameter
    signature, not the element token, because ``Port:`` serves both
    roles.
    """

    if ctype == "P_1TONE":
        return True
    if ctype == "PORT":
        return any(_TONE_POWER_KEY_RE.fullmatch(key) for key in params)
    return False


def parse_netlist_text(text: str) -> NormalizedCircuit:
    """Parse ADS-style or hpeesofsim-style netlist text into a circuit.

    Both :func:`parse_ads_netlist` and :func:`parse_hpeesofsim_deck`
    delegate here -- the two formats share one grammar.
    """

    circuit = NormalizedCircuit()
    for statement in _logical_lines(text):
        if not statement or statement.startswith(";"):
            continue
        tokens = _tokenize(statement)
        if not tokens:
            continue
        head = tokens[0]

        if head == "model" and len(tokens) >= 3:
            comp = Component(
                ctype=_normalize_type(tokens[2]),
                name=tokens[1],
                terminals=(),
                params=_parse_params(tokens[3:]),
            )
            circuit.elements.append(comp)
            continue

        if ":" in head:
            type_token, _, name_token = head.partition(":")
            rest = tokens[1:]
            if not name_token:  # "SweepPlan: SP1_stim" -- name is the next token
                if not rest:
                    circuit.ignored.append(statement)
                    continue
                name_token, rest = rest[0], rest[1:]
            ctype = _normalize_type(type_token)
            nets, params = _split_nets_and_params(rest)
            if ctype == "SHORT":
                # ADS netlists the multi-mode "Short" element for several
                # components; key on Mode to recognize the Mode=0 form as
                # an I_Probe (-> "I_PROBE", a counted element). A "Short"
                # that is not an in-scope element keeps ctype "SHORT",
                # which is in neither _ELEMENT_TYPES nor _CONTROLLER_TYPES,
                # so it falls through to the ignored bucket below.
                ctype = _classify_short(params) or "SHORT"
            comp = Component(ctype=ctype, name=name_token, terminals=nets, params=params)
            if _is_p1tone_source(ctype, params):
                # A P_1Tone source emission -- hpeesofsim writes it as a
                # ``P_1Tone:`` line, ADS normalises the OA cell to a
                # ``Port:`` line with a tone-power parameter. It is an
                # analysis stimulus, not a routed element (the same
                # status ``HB:`` has); the p1tone_content_check is its
                # authoritative cross-check. Excluding both forms keeps
                # the backends' element counts equal.
                circuit.ignored.append(statement)
            elif ctype in _CONTROLLER_TYPES:
                circuit.controllers.append(comp)
            elif ctype in _ELEMENT_TYPES:
                circuit.elements.append(comp)
            else:
                circuit.ignored.append(statement)
            continue

        # Bare directives: Options, Component, #load, etc. -- not circuit.
        circuit.ignored.append(statement)
    return circuit


def parse_ads_netlist(path: str | Path) -> NormalizedCircuit:
    """Parse an ADS-generated netlist (Tools -> Generate Netlist output)."""

    return parse_netlist_text(Path(path).read_text(encoding="utf-8"))


def parse_hpeesofsim_deck(path: str | Path) -> NormalizedCircuit:
    """Parse an hpeesofsim flat ``.ckt`` deck."""

    return parse_netlist_text(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Value normalization
# --------------------------------------------------------------------------

def _norm_value(raw: str):
    """Normalize a raw parameter value to a comparable tagged tuple.

    Returns one of ``("n", float)``, ``("q", float, unit)``,
    ``("s", str)`` (a quoted string) or ``("w", str)`` (a bare word).
    """

    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return ("s", raw[1:-1].strip().lower())
    parts = raw.split()
    if len(parts) == 2:
        try:
            return ("q", float(parts[0]), parts[1].lower())
        except ValueError:
            pass
    try:
        return ("n", float(raw))
    except ValueError:
        return ("w", raw.lower())


def _values_equal(a, b) -> bool:
    """Two normalized values are equal modulo representation."""

    def _close(x: float, y: float) -> bool:
        return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-12)

    if a[0] in ("n", "q") and b[0] in ("n", "q"):
        if not _close(a[1], b[1]):
            return False
        if a[0] == "q" and b[0] == "q":
            # 0 is unit-agnostic (0 mm == 0 mil); otherwise units must match.
            return a[2] == b[2] or (a[1] == 0.0 and b[1] == 0.0)
        return True
    if a[0] in ("s", "w") and b[0] in ("s", "w"):
        return str(a[1]).lower() == str(b[1]).lower()
    return False


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def _type_counts(components: list[Component]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for comp in components:
        counts[comp.ctype] = counts.get(comp.ctype, 0) + 1
    return counts


def _pair_key(comp: Component) -> tuple:
    """A deterministic sort key so same-type elements pair consistently.

    Ports sort by ``Num``; other elements sort by their parameter set.
    """

    if comp.ctype == "PORT":
        try:
            return (0, float(_norm_value(comp.params.get("Num", "0"))[1]))
        except (IndexError, TypeError, ValueError):
            return (0, 0.0)
    return (1, tuple(sorted((k, str(v)) for k, v in comp.params.items())))


def _pair_elements(
    elems_a: list[Component], elems_b: list[Component]
) -> list[tuple[Component, Component]]:
    """Pair elements of equal type (counts already checked equal)."""

    pairs: list[tuple[Component, Component]] = []
    by_type_b: dict[str, list[Component]] = {}
    for comp in elems_b:
        by_type_b.setdefault(comp.ctype, []).append(comp)
    by_type_a: dict[str, list[Component]] = {}
    for comp in elems_a:
        by_type_a.setdefault(comp.ctype, []).append(comp)
    for ctype, group_a in by_type_a.items():
        group_b = by_type_b.get(ctype, [])
        for ca, cb in zip(sorted(group_a, key=_pair_key),
                          sorted(group_b, key=_pair_key)):
            pairs.append((ca, cb))
    return pairs


def _best_terminal_order(
    net_map: dict[str, str], terms_a: tuple[str, ...], terms_b: tuple[str, ...]
) -> tuple[str, ...]:
    """Pick the ordering of ``terms_b`` most consistent with ``net_map``.

    A two-terminal element may have its nets listed in either order; the
    ordering that conflicts least with the net mapping built so far is
    chosen. Elements with other arities keep the listed order.
    """

    candidates = [terms_b]
    if len(terms_b) == 2:
        candidates.append((terms_b[1], terms_b[0]))

    def _conflicts(order: tuple[str, ...]) -> int:
        count = 0
        for a_net, b_net in zip(terms_a, order):
            if net_map.get(a_net, b_net) != b_net:
                count += 1
        return count

    return min(candidates, key=_conflicts)


def _net_isomorphism(
    pairs: list[tuple[Component, Component]]
) -> tuple[dict[str, str], list[str]]:
    """Build an A->B net-name mapping and report topology conflicts.

    Ports anchor the mapping (port ``Num`` k's signal net in A maps to
    port ``Num`` k's signal net in B); ground ``0`` maps to itself.
    Every other element's terminals must then be consistent with the
    mapping -- inconsistency is a topology difference.
    """

    net_map: dict[str, str] = {"0": "0"}
    conflicts: list[str] = []

    def _bind(a_net: str, b_net: str, where: str) -> None:
        existing = net_map.get(a_net)
        if existing is None:
            net_map[a_net] = b_net
        elif existing != b_net:
            conflicts.append(
                f"net topology conflict at {where}: "
                f"{a_net!r} maps to both {existing!r} and {b_net!r}"
            )

    for comp_a, comp_b in pairs:
        if comp_a.ctype == "PORT":
            for a_net, b_net in zip(comp_a.terminals, comp_b.terminals):
                _bind(a_net, b_net, f"port {comp_a.name}")

    # Non-port elements are resolved iteratively. A two-terminal element
    # whose nets are BOTH still unmapped has an ambiguous terminal order
    # (either order is conflict-free against the partial map), so it is
    # deferred until another element maps one of its nets. Without this
    # an element reached before its anchoring neighbour -- e.g. an open
    # stub processed before its MTEE -- could be bound reversed and only
    # surface a conflict later. Elements of other arity are positional.
    remaining: list[tuple[Component, Component]] = []
    for comp_a, comp_b in pairs:
        if comp_a.ctype == "PORT":
            continue
        if len(comp_a.terminals) != len(comp_b.terminals):
            conflicts.append(
                f"{comp_a.ctype} {comp_a.name}: terminal count "
                f"{len(comp_a.terminals)} vs {len(comp_b.terminals)}"
            )
            continue
        remaining.append((comp_a, comp_b))

    def _resolve(comp_a: Component, comp_b: Component) -> None:
        order = _best_terminal_order(net_map, comp_a.terminals, comp_b.terminals)
        for a_net, b_net in zip(comp_a.terminals, order):
            _bind(a_net, b_net, f"{comp_a.ctype} {comp_a.name}")

    progress = True
    while remaining and progress:
        progress = False
        deferred: list[tuple[Component, Component]] = []
        for comp_a, comp_b in remaining:
            ambiguous = (
                len(comp_a.terminals) == 2
                and not any(net in net_map for net in comp_a.terminals)
            )
            if ambiguous:
                deferred.append((comp_a, comp_b))
            else:
                _resolve(comp_a, comp_b)
                progress = True
        remaining = deferred
    # Anything still unanchored is a disconnected sub-circuit -- bind it
    # in its listed order; there is no information to do better.
    for comp_a, comp_b in remaining:
        _resolve(comp_a, comp_b)
    return net_map, conflicts


def _compare_params(
    comp_a: Component, comp_b: Component, notes: list[str]
) -> list[str]:
    """Compare two paired components' parameters. Shared params must match."""

    diffs: list[str] = []
    label = f"{comp_a.ctype} {comp_a.name}/{comp_b.name}"
    keys_a, keys_b = set(comp_a.params), set(comp_b.params)
    for key in sorted(keys_a & keys_b):
        va, vb = _norm_value(comp_a.params[key]), _norm_value(comp_b.params[key])
        if not _values_equal(va, vb):
            diffs.append(
                f"{label}: parameter {key} differs -- "
                f"{comp_a.params[key]!r} vs {comp_b.params[key]!r}"
            )
    for key in sorted(keys_a - keys_b):
        notes.append(f"{label}: parameter {key}={comp_a.params[key]!r} only in A")
    for key in sorted(keys_b - keys_a):
        notes.append(f"{label}: parameter {key}={comp_b.params[key]!r} only in B")
    return diffs


def _sweep_grid(plan: Component) -> tuple[float, float, int] | None:
    """Normalize a SweepPlan to ``(start_Hz, stop_Hz, n_points)``.

    ``Lin=N`` gives the point count directly; ``Step=`` gives the
    increment, from which the count is derived -- the two are different
    spellings of one frequency grid.
    """

    def _freq_hz(raw: str | None) -> float | None:
        if raw is None:
            return None
        value = _norm_value(raw)
        if value[0] == "q":
            return value[1] * _FREQ_SCALE.get(value[2], 1.0)
        if value[0] == "n":
            return value[1]
        return None

    start = _freq_hz(plan.params.get("Start"))
    stop = _freq_hz(plan.params.get("Stop"))
    if start is None or stop is None:
        return None
    if "Lin" in plan.params:
        n_points = int(round(float(_norm_value(plan.params["Lin"])[1])))
    elif "Step" in plan.params:
        step = _freq_hz(plan.params["Step"])
        if not step:
            return None
        n_points = int(round((stop - start) / step)) + 1
    else:
        return None
    return (start, stop, n_points)


def _compare_controllers(
    ctrls_a: list[Component], ctrls_b: list[Component], notes: list[str]
) -> list[str]:
    """Compare analysis controllers (S_Param + its SweepPlan)."""

    diffs: list[str] = []
    by_type_a = {c.ctype: c for c in ctrls_a}
    by_type_b = {c.ctype: c for c in ctrls_b}
    for ctype in ("S_PARAM", "SWEEPPLAN"):
        in_a, in_b = ctype in by_type_a, ctype in by_type_b
        if in_a != in_b:
            diffs.append(f"controller {ctype} present in only one netlist")
            continue
        if not in_a:
            continue
        if ctype == "SWEEPPLAN":
            grid_a = _sweep_grid(by_type_a[ctype])
            grid_b = _sweep_grid(by_type_b[ctype])
            if grid_a is None or grid_b is None:
                diffs.append("SweepPlan: could not normalize the frequency grid")
            elif (not math.isclose(grid_a[0], grid_b[0], rel_tol=1e-6)
                  or not math.isclose(grid_a[1], grid_b[1], rel_tol=1e-6)
                  or grid_a[2] != grid_b[2]):
                diffs.append(
                    f"SweepPlan frequency grid differs: "
                    f"{grid_a} vs {grid_b} (start_Hz, stop_Hz, n_points)"
                )
        else:
            diffs += _compare_params(by_type_a[ctype], by_type_b[ctype], notes)
    return diffs


def compare_circuits(
    circuit_a: NormalizedCircuit, circuit_b: NormalizedCircuit
) -> ComparisonResult:
    """Compare two normalized circuits for electrical equivalence.

    ``circuit_a`` / ``circuit_b`` are interchangeable. The result's
    ``match`` is True only if element types/counts agree, the
    port-anchored net topology is isomorphic, every shared parameter
    matches, and the analysis controllers agree. Representation-only
    differences (type aliases, terminal order, net names, ``Lin`` vs
    ``Step``, parameters present on only one side) are recorded in
    ``notes`` and do not break the match.
    """

    diffs: list[str] = []
    notes: list[str] = []

    counts_a = _type_counts(circuit_a.elements)
    counts_b = _type_counts(circuit_b.elements)
    if counts_a != counts_b:
        diffs.append(
            f"element type counts differ: {counts_a} vs {counts_b}"
        )
        return ComparisonResult(match=False, differences=diffs, notes=notes)

    pairs = _pair_elements(circuit_a.elements, circuit_b.elements)
    _, topo_conflicts = _net_isomorphism(pairs)
    diffs += topo_conflicts
    for comp_a, comp_b in pairs:
        diffs += _compare_params(comp_a, comp_b, notes)
    diffs += _compare_controllers(circuit_a.controllers, circuit_b.controllers, notes)

    if circuit_a.ignored or circuit_b.ignored:
        notes.append(
            f"ignored non-circuit directives: "
            f"{len(circuit_a.ignored)} in A, {len(circuit_b.ignored)} in B"
        )
    return ComparisonResult(match=not diffs, differences=diffs, notes=notes)


def format_report(result: ComparisonResult) -> str:
    """Render a human-readable comparison report."""

    lines: list[str] = []
    verdict = "MATCH" if result.match else "MISMATCH"
    lines.append(f"Netlist cross-check: {verdict}")
    lines.append("=" * 40)
    if result.differences:
        lines.append(f"\n{len(result.differences)} difference(s):")
        lines += [f"  - {d}" for d in result.differences]
    else:
        lines.append("\nNo circuit-level differences.")
    if result.notes:
        lines.append(f"\n{len(result.notes)} note(s) (representation only):")
        lines += [f"  - {n}" for n in result.notes]
    return "\n".join(lines) + "\n"


