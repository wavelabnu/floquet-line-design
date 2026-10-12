"""Schematic layout engine for a v2 ``DeviceSpec`` -- the shared source
of truth for component placement.

A circuit's *layout* is where each component sits and which way its pins
face. Two consumers need it and must agree:

* the topology SVG renderer (``tvtl_ads_builder_v2.topology``), which
  scales the layout to pixels and draws boxes + wires;
* the OA-Python emitter (``tvtl_ads_builder_v2.oa.oa_emitter``), which
  scales the layout to ADS schematic coordinates so every component is
  placed at an explicit, non-overlapping position.

Coordinates are in *layout units*, deliberately dimensionless:

* ``x`` increases left-to-right; one unit is one through-path column;
* ``y`` increases downward; one unit is one shunt-branch row. The
  through-path sits on ``y = 0``; a shunt branch hangs at ``y = 1, 2,
  ...`` below its tap point.

A consumer multiplies these by its own column/row pitch -- SVG pixels,
or ADS coordinate units -- so this module holds no pixel or ADS
constant.

Layout model
------------
* Through-path components -- the shortest electrical path between the
  two terminal ports -- are laid out along ``y = 0``, one column apart.
* Shunt branches hang vertically below their tap point (an MTEE branch
  pin, or a plain through-path net).
* A shunt-branch element with more than two pins (e.g. an ``SDD2P``
  2-port whose port 2 is an internal self-tie) is treated as
  *logically two-terminal* via the :data:`_PRIMARY_TERMINALS` metadata
  table: only its declared primary terminal pins take part in the
  branch chain.
* Ground / open-termination symbols are recorded at the relevant nets.
* ``P_1Tone`` power sources are placed next to the net each drives --
  left of the input net, right of the output net, below an interior
  net -- so a source's wire to its target pin stays short and
  axis-aligned instead of a long diagonal across the topology.
* A topology that does not fit the linear / one-shunt model falls back
  to a plain grid (with a logged warning) rather than failing.

The through-path / shunt model needs two through-path endpoint nets.
Two configurations supply them (see :func:`_terminal_nets`):

* a **two-port** circuit -- both endpoints are ``TermG`` ports
  numbered 1 and 2;
* a **source-driven-input** circuit -- one or more ``P_1Tone``
  sources drive a common input net and a single ``TermG`` is the
  output port. The sources replace the input port (v1's two-tone HB
  pattern), so the input endpoint is that source-driven net and the
  through-path row simply has no input-port box.

Any other port configuration (e.g. a lone 1-port circuit with no
sources) does not fit the model and falls back to grid placement
(with a logged warning) rather than failing.

:func:`compute_layout` is the simple public entry point (component id ->
coordinate). :func:`build_layout` returns the full :class:`CircuitLayout`
the renderer needs (roles, pin sides, termination symbols, and the
placed ``P_1Tone`` sources).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from .device_spec import DeviceSpec, Element

_LOG = logging.getLogger(__name__)

# The element type that marks a terminal port.
_PORT_TYPE = "TermG"

# Element types whose layout-relevant terminals are a declared subset
# of their pins. An ``SDD2P`` is a 4-pin 2-port, but for branch layout
# only port 1 matters: pin 1 is the tap-facing terminal, pin 2 the
# terminus-facing one. Port 2 (pins 3/4) is an internal self-tie, not
# part of any shunt chain. The shunt tracer reads this table to treat
# such an element as "logically two-terminal" despite its pin count;
# it is the generic hook for any future multi-terminal element.
_PRIMARY_TERMINALS: dict[str, tuple[int, ...]] = {
    "SDD2P": (1, 2),
}

# Pin-side tokens. A consumer maps each to an offset from the component
# centre: "left"/"right" to the horizontal box edges, "top"/"bottom" to
# the vertical ones.
_SIDE_LEFT = "left"
_SIDE_RIGHT = "right"
_SIDE_TOP = "top"
_SIDE_BOTTOM = "bottom"

# Source placement. A ``P_1Tone`` source is placed _SOURCE_OFFSET_UNITS
# from the net it drives; each further source on the same net is
# staggered by another _SOURCE_STAGGER_UNITS so the sources never
# overlap and each gets its own wire. Both are one layout column --
# the same scale as the through-path row's inter-element spacing.
_SOURCE_OFFSET_UNITS = 1.0
_SOURCE_STAGGER_UNITS = 1.0


class _UnclassifiedTopology(Exception):
    """Raised internally when a spec does not fit the linear/shunt model.

    Never escapes the module: :func:`build_layout` catches it and falls
    back to a grid layout.
    """


@dataclass(frozen=True)
class PlacedComponent:
    """One component placed on the layout grid.

    ``x`` / ``y`` are the component-centre coordinate in layout units.
    ``pin_sides`` maps each pin number to the box side it leaves from
    (one of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``).
    """

    element: Element
    role: str                              # "through" | "shunt" | "port" | "grid"
    x: float
    y: float
    pin_sides: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TerminationSymbol:
    """A ground or open-circuit marker anchored to a net.

    ``x`` / ``y`` are the layout-unit coordinate of the component the
    symbol hangs from. ``attach`` selects how the consumer finds the
    symbol's connection point:

    * ``"below_box"`` -- the symbol hangs off the bottom edge of the box
      at ``(x, y)`` (a shunt-branch terminus);
    * ``"on_row"`` -- the symbol connects directly at ``(x, y)`` (a rare
      ground/open straight on a through-path net).
    """

    kind: str                              # "ground" | "open"
    net: str
    x: float
    y: float
    attach: str                            # "below_box" | "on_row"


@dataclass(frozen=True)
class PlacedSource:
    """A ``P_1Tone`` power source placed on the layout grid.

    ``name`` is the source's instance name; ``x`` / ``y`` are its
    centre in layout units; ``target_net`` is the net it drives. A
    source is positioned by :func:`_place_sources` relative to that
    net, so its wire to the target pin is short and axis-aligned --
    never a long diagonal across the topology.
    """

    name: str
    x: float
    y: float
    target_net: str


@dataclass(frozen=True)
class CircuitLayout:
    """The full layout of one circuit.

    ``placements`` is ordered the way a renderer should draw it:
    through-path row first (port, components..., port), then each shunt
    branch top-to-bottom. ``is_grid_fallback`` is true when the linear
    model did not fit and a plain grid was used instead. ``sources``
    holds the placed ``P_1Tone`` power sources (empty for a spec with
    none); each is positioned next to the net it drives.
    """

    placements: tuple[PlacedComponent, ...]
    symbols: tuple[TerminationSymbol, ...]
    is_grid_fallback: bool
    sources: tuple[PlacedSource, ...] = ()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compute_layout(spec: DeviceSpec) -> dict[str, tuple[float, float]]:
    """Return ``{element_id: (x, y)}`` for every component in ``spec``.

    Coordinates are in layout units (see the module docstring). This is
    the placement contract shared by the topology renderer and the OA
    emitter; both scale it to their own coordinate system.

    Never raises for a layable spec: an unclassifiable topology falls
    back to a grid layout (with a logged warning).
    """

    return {p.element.id: (p.x, p.y) for p in build_layout(spec).placements}


def build_layout(spec: DeviceSpec) -> CircuitLayout:
    """Return the full :class:`CircuitLayout` for ``spec``.

    Tries the linear-through-path-with-shunt-branches model first; on
    any failure logs a warning and falls back to a plain grid. Never
    raises for a layable spec.
    """

    try:
        return _auto_layout(spec)
    except _UnclassifiedTopology as exc:
        _LOG.warning(
            "layout: %s -- grid fallback for circuit %r", exc, spec.label
        )
        return _grid_layout(spec)
    except Exception as exc:  # noqa: BLE001 -- layout must never fail a build
        _LOG.warning(
            "layout: unexpected error (%s) -- grid fallback for circuit %r",
            exc,
            spec.label,
        )
        return _grid_layout(spec)


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------


def _element_nets(element: Element) -> list[str]:
    """Nets of an element ordered by pin number (1, 2, 3, ...)."""

    by_pin = {pin.pin: pin.node for pin in element.pins}
    return [by_pin[k] for k in sorted(by_pin)]


def _layout_terminal_nets(element: Element) -> list[str]:
    """Nets of an element's layout-relevant terminals, ordered by pin.

    For most elements this is every pin net (:func:`_element_nets`).
    For an element type registered in :data:`_PRIMARY_TERMINALS` only
    the declared primary pins count -- an ``SDD2P`` is a 4-pin 2-port
    but for shunt-branch layout only its port-1 pins (1 = tap-facing,
    2 = terminus-facing) participate; port 2 is an internal self-tie.
    This is what lets the shunt tracer treat a multi-terminal element
    as logically two-terminal.
    """

    primary = _PRIMARY_TERMINALS.get(element.type)
    if primary is None:
        return _element_nets(element)
    by_pin = {pin.pin: pin.node for pin in element.pins}
    return [by_pin[k] for k in primary if k in by_pin]


def _net_to_pins(elements: list[Element]) -> dict[str, list[tuple[Element, int]]]:
    """Map ``net -> [(element, pin_number), ...]``."""

    mapping: dict[str, list[tuple[Element, int]]] = {}
    for element in elements:
        for pin in element.pins:
            mapping.setdefault(pin.node, []).append((element, pin.pin))
    return mapping


def _terminal_nets(spec: DeviceSpec) -> tuple[str, str, bool]:
    """Return ``(net_in, net_out, net_in_is_source_driven)``, or raise.

    The through-path model needs two endpoint nets. Two configurations
    are supported:

    * **two-port** -- ports numbered exactly ``{1, 2}``; ``net_in`` is
      port 1's net, ``net_out`` port 2's, and
      ``net_in_is_source_driven`` is ``False``.
    * **source-driven input** -- a single output ``TermG`` port plus
      one or more ``P_1Tone`` sources that all drive one common net.
      The sources replace the input port (v1's two-tone HB pattern),
      so ``net_in`` is that source-driven net, ``net_out`` the lone
      port's net, and ``net_in_is_source_driven`` is ``True``.

    Any other configuration raises :class:`_UnclassifiedTopology` so
    :func:`build_layout` grid-falls-back as before.
    """

    by_num = {port.num: port.node for port in spec.ports}
    if set(by_num) == {1, 2}:
        return by_num[1], by_num[2], False
    if spec.sources and len(by_num) == 1:
        source_nets = {source.port_node for source in spec.sources}
        if len(source_nets) != 1:
            raise _UnclassifiedTopology(
                f"P_1Tone sources drive {len(source_nets)} distinct nets; "
                f"the through-path model needs a single source-driven "
                f"input net"
            )
        (net_in,) = source_nets
        (net_out,) = by_num.values()
        if net_in == net_out:
            raise _UnclassifiedTopology(
                "the source-driven input net coincides with the output "
                "port net"
            )
        return net_in, net_out, True
    raise _UnclassifiedTopology(
        f"expected ports numbered {{1, 2}}, or a single output port plus "
        f"P_1Tone sources on one input net (got ports {sorted(by_num)} and "
        f"{len(spec.sources)} source(s))"
    )


def _through_path(
    elements: list[Element], net_in: str, net_out: str
) -> tuple[list[Element], list[str]]:
    """Shortest element path from ``net_in`` to ``net_out``.

    Returns ``(path_elements, path_nets)``. Each non-port element is an
    edge between its pin nets; an ``MTEE`` (three pins) is traversable
    between any two of its three nets.
    """

    adjacency: dict[str, list[tuple[Element, str]]] = {}
    for element in elements:
        if element.type == _PORT_TYPE:
            continue
        nets = _element_nets(element)
        if len(nets) < 2:
            continue
        for i, a in enumerate(nets):
            for j, b in enumerate(nets):
                if i != j:
                    adjacency.setdefault(a, []).append((element, b))

    prev: dict[str, tuple[str, Element] | None] = {net_in: None}
    queue: deque[str] = deque([net_in])
    while queue:
        current = queue.popleft()
        if current == net_out:
            break
        for element, nxt in adjacency.get(current, []):
            if nxt not in prev:
                prev[nxt] = (current, element)
                queue.append(nxt)

    if net_out not in prev:
        raise _UnclassifiedTopology("no through-path between the two ports")

    path_nets = [net_out]
    path_elements: list[Element] = []
    cursor = net_out
    while prev[cursor] is not None:
        prev_net, element = prev[cursor]  # type: ignore[misc]
        path_elements.append(element)
        path_nets.append(prev_net)
        cursor = prev_net
    path_elements.reverse()
    path_nets.reverse()
    return path_elements, path_nets


def _trace_shunt(
    tap_net: str,
    off_elements: dict[str, Element],
    n2p: dict[str, list[tuple[Element, int]]],
) -> tuple[list[Element], str, Element | None]:
    """Walk a simple shunt chain outward from ``tap_net``.

    ``off_elements`` is ``{id: element}`` of the not-yet-placed
    off-path elements. Returns ``(ordered_chain, terminus_net,
    terminus_element)``. ``terminus_element`` is a **1-port** element
    sitting on the terminus net (the physical via ``VIAGND``, which
    grounds internally, or the open-end-effect stub ``MLEF`` / ``MLOC``
    appended at the branch end), or ``None`` when the branch ends on a
    bare ground / open node (the lumped-``L`` via / ideal-open path).
    Raises :class:`_UnclassifiedTopology` if the branch forks.
    """

    chain: list[Element] = []
    used: set[str] = set()
    current = tap_net
    while True:
        candidates = [
            (element, _layout_terminal_nets(element))
            for element, _pin in n2p.get(current, [])
            if element.id in off_elements and element.id not in used
        ]
        if not candidates:
            return chain, current, None
        if len(candidates) > 1:
            raise _UnclassifiedTopology(
                f"shunt branch at net {current!r} forks into "
                f"{len(candidates)} elements"
            )
        element, nets = candidates[0]
        # A 1-port shunt-terminus element -- the physical via (``VIAGND``,
        # grounds internally) or the open-end-effect stub (``MLEF`` / ``MLOC``)
        # appended at the branch end. It sits on ``current`` with a single pin
        # and terminates the chain there (its ground/open is internal to the
        # element), so it is the branch's bottom box rather than a two-terminal
        # link. The default lumped path has no such element (``LVIA`` is a
        # 2-port ``L`` to a ground net; an ideal open dangles with no element),
        # so this branch is inert for the byte-identical recovery path.
        if len(element.pins) == 1:
            return chain, current, element
        # ``nets`` is the element's layout-relevant terminal nets -- the
        # primary terminals for a multi-terminal element (e.g. SDD2P),
        # all pins for an ordinary two-terminal element. It is layable
        # in a shunt chain only when those terminals span exactly two
        # distinct nets.
        if len(set(nets)) != 2:
            raise _UnclassifiedTopology(
                f"shunt element {element.id!r} is not two-terminal "
                f"(layout terminals span {len(set(nets))} net(s))"
            )
        chain.append(element)
        used.add(element.id)
        current = nets[1] if nets[0] == current else nets[0]


def _pin_of(element: Element, net: str) -> int:
    """Return the (first) pin number of ``element`` attached to ``net``."""

    for pin in element.pins:
        if pin.node == net:
            return pin.pin
    return element.pins[0].pin


# ---------------------------------------------------------------------------
# Automatic layout (linear through-path + vertical shunt branches)
# ---------------------------------------------------------------------------


def _auto_layout(spec: DeviceSpec) -> CircuitLayout:
    """Lay out a linear-through-path-with-shunt-branches topology.

    Handles a two-port circuit and a source-driven-input circuit
    (``P_1Tone`` sources at the input net, one output port) alike --
    see :func:`_terminal_nets`. The only structural difference is that
    a source-driven input has no input-port box: the through-path row
    begins at the first path element, whose input net the sources
    (not layout components) drive directly.
    """

    # ``NAME`` elements are a spec-level annotation declaring
    # "expose this net's voltage in the .ds dataset"; they have no
    # electrical role and no layout footprint. The filter is load-
    # bearing: a NAME on a branch tap net (e.g. ``n_c{k}_b``) counts
    # as a second element at that net during the shunt-trace walk
    # below, the tracer flags ``shunt branch ... forks into 2
    # elements`` and falls back to the grid layout, and the grid
    # places shunt-branch components at coordinates ADS's
    # connectivity engine then merges with the global ground net --
    # the empirically-observed failure mode that motivates
    # this filter. DO NOT re-add NAME to the tracer thinking the
    # filter is a bug: it pre-empts that exact regression.
    elements = [e for e in spec.elements if e.type != "NAME"]
    net_in, net_out, net_in_is_source_driven = _terminal_nets(spec)
    path_elements, path_nets = _through_path(elements, net_in, net_out)
    path_ids = {e.id for e in path_elements}

    ports = {e.pins[0].node: e for e in elements if e.type == _PORT_TYPE}
    port_in = ports.get(net_in)
    port_out = ports.get(net_out)
    if port_out is None:
        raise _UnclassifiedTopology("the output port net carries no TermG")
    if port_in is None and not net_in_is_source_driven:
        raise _UnclassifiedTopology("the input port net carries no TermG")

    n2p = _net_to_pins(elements)
    off_elements = {
        e.id: e
        for e in elements
        if e.type != _PORT_TYPE and e.id not in path_ids
    }

    placements: list[PlacedComponent] = []
    symbols: list[TerminationSymbol] = []

    # --- through-path row: [port_in?], elements..., port_out -----------
    # One column apart, all on y = 0. A source-driven input has no
    # port_in box -- the row starts at the first path element.
    row = [
        *([port_in] if port_in is not None else []),
        *path_elements,
        port_out,
    ]
    centres_x: dict[str, float] = {}
    for index, element in enumerate(row):
        x = float(index)
        role = "port" if element.type == _PORT_TYPE else "through"
        pin_sides = _through_pin_sides(
            element, path_elements, path_nets, net_in, net_out
        )
        placements.append(PlacedComponent(element, role, x, 0.0, pin_sides))
        centres_x[element.id] = x

    # x of each through-path *net* (a shunt may tap one). A net between
    # two path elements sits midway between them; an endpoint net sits
    # midway to its port, or -- for a source-driven input with no
    # port_in box -- half a column beyond the first path element.
    net_x: dict[str, float] = {}
    path_x = [centres_x[e.id] for e in path_elements]
    for i, net in enumerate(path_nets):
        if i == 0:
            left = (
                centres_x[port_in.id]
                if port_in is not None
                else path_x[0] - 1.0
            )
            net_x[net] = (left + path_x[0]) / 2.0
        elif i == len(path_nets) - 1:
            net_x[net] = (path_x[-1] + centres_x[port_out.id]) / 2.0
        else:
            net_x[net] = (path_x[i - 1] + path_x[i]) / 2.0

    # --- shunt branches -----------------------------------------------
    ground = set(spec.ground_nodes)
    opens = set(spec.open_terminations)
    placed_off: set[str] = set()

    # A tap net touches an on-path element and at least one off-path one.
    tap_nets: list[str] = []
    for net, pins in n2p.items():
        on = any(e.id in path_ids for e, _ in pins)
        off = any(e.id in off_elements for e, _ in pins)
        if on and off:
            tap_nets.append(net)

    for tap_net in sorted(tap_nets):
        chain, terminus, terminus_element = _trace_shunt(tap_net, off_elements, n2p)
        if not chain and terminus_element is None:
            continue
        tap_x = _tap_x(tap_net, path_elements, path_nets, net_x, centres_x)
        prev_net = tap_net
        last_y = 0.0
        for depth, element in enumerate(chain, start=1):
            y = float(depth)
            # The element's layout-relevant terminals -- the primary
            # terminals for a multi-terminal element (SDD2P), all pins
            # for an ordinary two-terminal element.
            nets = _layout_terminal_nets(element)
            # the pin on the net the chain arrived from faces up (toward
            # the tap); the other faces down (toward the terminus).
            if nets[0] == prev_net:
                up_net, down_net = nets[0], nets[1]
            else:
                up_net, down_net = nets[1], nets[0]
            pin_sides = {
                _pin_of(element, up_net): _SIDE_TOP,
                _pin_of(element, down_net): _SIDE_BOTTOM,
            }
            placements.append(
                PlacedComponent(element, "shunt", tap_x, y, pin_sides)
            )
            placed_off.add(element.id)
            prev_net = down_net
            last_y = y
        if terminus_element is not None:
            # A 1-port terminus element (physical via ``VIAGND`` / open-end
            # ``MLEF`` / ``MLOC``) is the branch's bottom box, its single pin
            # facing up toward the chain. It REPLACES the ground/open terminus
            # symbol the lumped path draws: ``VIAGND`` grounds internally, and
            # ``MLEF`` / ``MLOC`` are themselves the open end.
            y = last_y + 1.0
            pin = terminus_element.pins[0].pin
            placements.append(
                PlacedComponent(
                    terminus_element, "shunt", tap_x, y, {pin: _SIDE_TOP}
                )
            )
            placed_off.add(terminus_element.id)
        else:
            # terminus symbol, hanging below the last shunt box
            if terminus in ground:
                symbols.append(
                    TerminationSymbol("ground", terminus, tap_x, last_y, "below_box")
                )
            elif terminus in opens:
                symbols.append(
                    TerminationSymbol("open", terminus, tap_x, last_y, "below_box")
                )

    if len(placed_off) != len(off_elements):
        raise _UnclassifiedTopology(
            "some elements are neither on the through-path nor on a simple "
            "shunt branch"
        )

    # Ground / open symbols on through-path nets (rare but supported).
    for net in sorted(ground | opens):
        if any(s.net == net for s in symbols):
            continue
        if net in net_x:
            kind = "ground" if net in ground else "open"
            symbols.append(
                TerminationSymbol(kind, net, net_x[net], 0.0, "on_row")
            )

    return CircuitLayout(
        placements=tuple(placements),
        symbols=tuple(symbols),
        is_grid_fallback=False,
        sources=_place_sources(spec, net_in, net_out, net_x, placements),
    )


def _through_pin_sides(
    element: Element,
    path_elements: list[Element],
    path_nets: list[str],
    net_in: str,
    net_out: str,
) -> dict[int, str]:
    """Pin-side map for a placed through-path / port component."""

    if element.type == _PORT_TYPE:
        net = element.pins[0].node
        side = _SIDE_RIGHT if net == net_in else _SIDE_LEFT
        return {element.pins[0].pin: side}

    # The path enters this element from the net just before it and
    # leaves on the net just after it in path_nets.
    idx = path_elements.index(element)
    entry_net = path_nets[idx]
    exit_net = path_nets[idx + 1]
    sides: dict[int, str] = {}
    for pin in element.pins:
        if pin.node == entry_net and pin.pin not in sides:
            sides[pin.pin] = _SIDE_LEFT
        elif pin.node == exit_net and pin.pin not in sides:
            sides[pin.pin] = _SIDE_RIGHT
        else:
            sides[pin.pin] = _SIDE_BOTTOM
    return sides


def _tap_x(
    tap_net: str,
    path_elements: list[Element],
    path_nets: list[str],
    net_x: dict[str, float],
    centres_x: dict[str, float],
) -> float:
    """Layout-unit x a shunt branch should hang from."""

    if tap_net in net_x:
        return net_x[tap_net]
    # Otherwise it is a third pin of an on-path element (e.g. an MTEE
    # branch pin): hang the branch straight below that element.
    for element in path_elements:
        if tap_net in _element_nets(element):
            return centres_x[element.id]
    raise _UnclassifiedTopology(f"cannot locate tap net {tap_net!r}")


# ---------------------------------------------------------------------------
# Grid fallback
# ---------------------------------------------------------------------------


def _grid_layout(spec: DeviceSpec) -> CircuitLayout:
    """Place every element on a plain grid; pins by sorted pin order."""

    # NAME annotations are zero-electrical-role markers;
    # excluded from the grid for the same reason _classified_layout
    # excludes them -- they would otherwise be drawn as empty boxes
    # and the SVG/inspection renderer would warn about them.
    elements = [e for e in spec.elements if e.type != "NAME"]
    columns = max(1, int(len(elements) ** 0.5 + 0.999))
    placements: list[PlacedComponent] = []
    for index, element in enumerate(elements):
        col = index % columns
        rowi = index // columns
        role = "port" if element.type == _PORT_TYPE else "grid"
        sides: dict[int, str] = {}
        for i, pin in enumerate(sorted(element.pins, key=lambda p: p.pin)):
            if i == 0:
                sides[pin.pin] = _SIDE_LEFT
            elif i == 1:
                sides[pin.pin] = _SIDE_RIGHT
            else:
                sides[pin.pin] = _SIDE_BOTTOM
        placements.append(
            PlacedComponent(element, role, float(col), float(rowi), sides)
        )
    return CircuitLayout(
        placements=tuple(placements),
        symbols=(),
        is_grid_fallback=True,
        sources=_grid_sources(spec, placements),
    )


# ---------------------------------------------------------------------------
# Source placement
# ---------------------------------------------------------------------------


def _source_net_anchor(
    net: str,
    net_in: str,
    net_out: str,
    net_x: dict[str, float],
    placements: list[PlacedComponent],
) -> tuple[float, float, str, float] | None:
    """Locate a source's target net -> ``(x, y, axis, sign)`` or ``None``.

    ``axis`` is ``"x"`` (place the source left/right of the net, on the
    through-path row) or ``"y"`` (place it below the row); ``sign`` is
    the placement direction along that axis. ``None`` means no topology
    pin sits on the net at all -- the caller parks the source at a safe
    default rather than crashing.
    """

    if net == net_in:
        return net_x[net], 0.0, "x", -1.0       # input net: drive from left
    if net == net_out:
        return net_x[net], 0.0, "x", +1.0       # output net: drive from right
    if net in net_x:
        return net_x[net], 0.0, "y", +1.0       # interior net: drive from below
    # Not a through-path net: anchor below the first component (lowest
    # x, then y -- deterministic) carrying a pin on the net.
    carriers = sorted(
        (p for p in placements
         if any(pin.node == net for pin in p.element.pins)),
        key=lambda p: (p.x, p.y),
    )
    if carriers:
        return carriers[0].x, carriers[0].y, "y", +1.0
    return None


def _place_sources(
    spec: DeviceSpec,
    net_in: str,
    net_out: str,
    net_x: dict[str, float],
    placements: list[PlacedComponent],
) -> tuple[PlacedSource, ...]:
    """Place every ``P_1Tone`` source next to the net it drives.

    Each source is anchored to its target net (:func:`_source_net_anchor`)
    and offset ``_SOURCE_OFFSET_UNITS`` away; sources sharing one net
    are staggered a further ``_SOURCE_STAGGER_UNITS`` each so they never
    overlap and each gets its own short, axis-aligned wire:

    * a source on the input net is placed to the LEFT of the
      through-path row;
    * a source on the output net, to the RIGHT;
    * a source on any interior net, BELOW the row (a vertical wire).

    Multi-cell readiness -- this net-derived placement already covers
    multi-cell-cascade configurations: a source
    at the first cell's input falls on the input net (placed left), at
    the last cell's output on the output net (placed right), and at an
    interior cell on an interior through-path net (placed below the
    row). No coordinate is hardcoded, so none of those cases re-break
    the placement. Caveat for the interior case: a below-the-row source
    shares its column with any shunt branch tapping the same net, so
    a future extension should give an interior source its own column
    offset (and the OA emitter an L-shaped wire) when such a canonical
    first appears.
    """

    if not spec.sources:
        return ()
    fallback_x = min((p.x for p in placements), default=0.0) - 1.0
    by_net: dict[str, list] = {}
    for source in spec.sources:
        by_net.setdefault(source.port_node, []).append(source)
    placed: list[PlacedSource] = []
    for net, sources in by_net.items():
        anchor = _source_net_anchor(net, net_in, net_out, net_x, placements)
        for index, source in enumerate(sources):
            step = _SOURCE_OFFSET_UNITS + index * _SOURCE_STAGGER_UNITS
            if anchor is None:
                # Edge case -- no topology pin on this net. Park the
                # source(s) left of the layout, staggered downward, so
                # the build never crashes on a malformed spec.
                placed.append(
                    PlacedSource(source.name, fallback_x, float(index), net)
                )
                continue
            anchor_x, anchor_y, axis, sign = anchor
            if axis == "x":
                placed.append(
                    PlacedSource(
                        source.name, anchor_x + sign * step, anchor_y, net
                    )
                )
            else:  # axis == "y": below the row, staggered downward
                placed.append(
                    PlacedSource(
                        source.name, anchor_x, anchor_y + sign * step, net
                    )
                )
    return tuple(placed)


def _grid_sources(
    spec: DeviceSpec, placements: list[PlacedComponent]
) -> tuple[PlacedSource, ...]:
    """Place sources for the grid fallback: left of the grid, stacked.

    The grid fallback is for topologies the linear model cannot
    classify; a source-bearing spec should reach :func:`_auto_layout`
    instead, but if the fallback is ever invoked the sources are still
    placed sensibly -- one column left of the leftmost grid cell,
    staggered vertically so they do not overlap.
    """

    if not spec.sources:
        return ()
    left = min((p.x for p in placements), default=0.0) - 1.0
    return tuple(
        PlacedSource(source.name, left, float(index), source.port_node)
        for index, source in enumerate(spec.sources)
    )


