"""Auto-layout and wiring for the ADS inspection schematic.

Given a :class:`DeviceSpec`, this assigns (x, y) coordinates to every
element plus the spec-level items (substrate, analysis controllers,
circuit grounds), and computes the wire paths between connected pins.
The ADS schematic deck (:mod:`.ael_templates`) consumes this plan.

The layout is deliberately simple -- the canonical circuits are a
series chain of ``MLIN``/``MTEE`` between two ``TermG`` ports, with at
most one ``MTEE`` carrying a shunt branch. The schematic only has to be
readable and topology-correct, not pretty. Coordinate units match the
recorded ADS macros (small numbers around 1-6).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..device_spec import DeviceSpec, Element

# Main-path / branch element spacing (schematic units). The MLIN body
# length is 1.0 (pin 1 at the placement origin, pin 2 at +1.0); a step
# of 2.0 therefore always leaves a gap between pin 2 of one element and
# the next element's pin -- never a coincidence. That gap is essential:
# coincident pins produce a zero-length wire, which ADS rejects ("Could
# not Add Wire: Too few points specified"), so every connection must be
# spanned by a real, non-degenerate wire.
_DX = 2.0
_DY = 2.0

# Empirically-derived body length of an MLIN: pin 1 sits at the
# placement origin, pin 2 one unit along +x. See _MLIN_BODY discussion.
_MLIN_BODY = 1.0

# Pin offsets relative to an element's placement origin, derived from
# the recorded ADS macros (recorded.dem / recorded2.dem in the
# reference workspace). Evidence -- ADS places an element at a pin, and
# a wire vertex drawn to a pin equals (placement origin + pin offset):
#
#   recorded.dem L8  : de_place_item(MLIN,   2.75, 0.5)
#   recorded.dem L15 : de_add_wire(2.75, 0.5)   -> MLIN pin 1 at (0, 0)
#   recorded.dem L12 : de_place_item(TermG,  1, -0.5)
#   recorded.dem L16 : de_add_wire(1, -0.5)     -> TermG pin at (0, 0)
#   recorded2.dem L12: de_place_item(MLIN,  -0.75, 0.625)
#   recorded2.dem L26: de_place_item(MTEE,   1, 0.625)
#   recorded2.dem L93: de_add_wire(0.25, 0.625) -> MLIN pin 2 at
#                      (-0.75 + 1.0, 0)  => MLIN body length 1.0
#   recorded2.dem L94: de_add_wire(1, 0.625)    -> MTEE pin 1 at (0, 0)
#   recorded2.dem L97: de_add_wire(2, 0.625)    -> MTEE pin 2 at (1.0, 0)
#   recorded2.dem L105: de_add_wire(1.5, 0.125) -> MTEE pin 3 at
#                      (1.5 - 1, 0.125 - 0.625) = (0.5, -0.5)
PIN_OFFSETS_AT_DEFAULT_ROTATION: dict[str, dict[int, tuple[float, float]]] = {
    # Macro-confirmed (Levels 1-3 use these three).
    "MLIN": {1: (0.0, 0.0), 2: (_MLIN_BODY, 0.0)},
    "MTEE": {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.5, -0.5)},
    "TermG": {1: (0.0, 0.0)},
    # Single-pin: macro-confirmed origin = pin (recorded2.dem L139,
    # de_add_wire at the GROUND placement origin). VIAGND / MLOC follow
    # the same single-pin origin-is-pin convention.
    "GROUND": {1: (0.0, 0.0)},
    "VIAGND": {1: (0.0, 0.0)},
    "MLOC": {1: (0.0, 0.0)},
    # Two-pin body-axis elements for later levels. recorded2.dem has a
    # cascade of manual moves that desync placement from wiring for
    # these, so pin 2 is not independently confirmed; by assumption
    # (similar to MLIN) they reuse the MLIN body length. Refine when a
    # Level-4+ circuit first places one.
    "R": {1: (0.0, 0.0), 2: (_MLIN_BODY, 0.0)},
    "L": {1: (0.0, 0.0), 2: (_MLIN_BODY, 0.0)},
    "SDD2P": {1: (0.0, 0.0), 2: (_MLIN_BODY, 0.0)},
    # Controllers carry no schematic pins -- they are placed but not wired.
    "MSUB": {},
    "S_Param": {},
    "SweepPlan": {},
}

# Placement rotation each element type needs before its de_place_item
# call; None means the ADS default orientation is already correct.
#
# Evidence: an ADS 2026 session captured live via Tools -> Command Line
# while drawing TermG -> MLIN -> MLIN -> TermG with a rotated stub MLIN.
# The Command Line logger records every call ADS itself emits with no
# macro-recorder shortcuts, so it is the authoritative source -- more
# reliable than the recorded .dem macros for placement orientation. The
# log showed ADS places each TermG only after de_rotate_image("DOWN"),
# and the S_Param / SweepPlan analysis controllers after
# de_rotate_image("RIGHT"); MLIN, MTEE, MLOC, VIAGND, GROUND and MSUB
# were placed in the default orientation.
#
# These rotations change only the on-screen orientation: TermG's single
# pin and the (empty) controller pin sets all sit at offset (0, 0),
# which rotates to itself, so _compute_wires produces identical wire
# coordinates with or without them.
#
# Rotation is scoped per element block: a fresh de_init_item starts in
# the default orientation regardless of the previous element's
# rotation, so render_ael_script emits no rotation reset between blocks
# (the CLI log went straight from de_free_item to the next de_init_item
# with no reset and every element came out correctly oriented).
#
# R / L / SDD2P were not exercised in the log; left None (ADS default)
# until a circuit places one and a fresh CLI log confirms it.
ROTATION_FOR_ELEMENT: dict[str, str | None] = {
    "MLIN": None,
    "MTEE": None,
    "MLOC": None,
    "VIAGND": None,
    "GROUND": None,
    "TermG": "DOWN",
    "MSUB": None,
    "S_Param": "RIGHT",
    "SweepPlan": "RIGHT",
    "R": None,
    "L": None,
    "SDD2P": None,
}

# Spec-level controller symbols, placed below the circuit (unwired).
# Their on-screen labels are wide, so they get generous fixed spacing
# well clear of the circuit row at y = 0.
_CONTROLLER_COORDS: dict[str, tuple[float, float]] = {
    "MSUB": (-4.0, -3.0),
    "S_Param": (-1.0, -3.0),
    "SweepPlan": (3.0, -3.0),
}


@dataclass(frozen=True)
class PlacedItem:
    """One schematic item with its placement coordinate and rotation."""

    item_id: str          # device-spec element id, or a synthetic id
    element_type: str     # TVTL element type / controller / GROUND
    x: float
    y: float
    rotation: str | None  # None, or "UP"/"DOWN"/"LEFT"/"RIGHT"


@dataclass(frozen=True)
class Wire:
    """One wire connecting two pin positions that share a spec node."""

    node: str
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True)
class SchematicPlan:
    """The full placement + wiring plan for one circuit."""

    items: tuple[PlacedItem, ...]
    wires: tuple[Wire, ...]


def auto_layout(spec: DeviceSpec) -> dict[str, tuple[float, float]]:
    """Return a mapping ``item_id -> (x, y)`` placement coordinate.

    Covers every device-spec element plus the synthetic spec-level
    items (``MSUB``, ``S_Param``, ``SweepPlan``, and a ``GROUND_<node>``
    per ground node). See :func:`build_schematic_plan` for the richer
    plan (rotation + wires).
    """

    return {
        item.item_id: (item.x, item.y)
        for item in build_schematic_plan(spec).items
    }


def build_schematic_plan(spec: DeviceSpec) -> SchematicPlan:
    """Compute the full placement + wiring plan for ``spec``.

    Layout strategy:

    1. Walk the main path input port -> output port via series
       ``MLIN``/``MTEE`` elements; place it left-to-right at ``y = 0``.
    2. Each ``MTEE``'s shunt branch (the chain on pin 3) is placed
       below it, decreasing ``y``.
    3. A ``GROUND`` symbol is added under each spec ground node.
    4. The ``MSUB``/``S_Param``/``SweepPlan`` controllers are placed in
       a row below the circuit (unwired).
    """

    adjacency = _adjacency(spec)
    in_port = _port_element(spec, 1)
    out_port = _port_element(spec, 2)
    series = _series_path(spec, adjacency, in_port, out_port)

    items: list[PlacedItem] = []
    coords: dict[str, tuple[float, float]] = {}

    main = [in_port, *series, out_port]
    for index, element in enumerate(main):
        x = float(index) * _DX
        rotation = ROTATION_FOR_ELEMENT.get(element.type)
        items.append(PlacedItem(element.id, element.type, x, 0.0, rotation))
        coords[element.id] = (x, 0.0)

    # Shunt branches drop straight down from the MTEE's pin-3 column so
    # the branch wire is a clean vertical segment from the pin.
    for element in series:
        if element.type != "MTEE":
            continue
        mtee_x, mtee_y = coords[element.id]
        pin3_x = mtee_x + PIN_OFFSETS_AT_DEFAULT_ROTATION["MTEE"][3][0]
        for depth, branch_el in enumerate(_branch(spec, element, adjacency)):
            y = mtee_y - float(depth + 1) * _DY
            rotation = ROTATION_FOR_ELEMENT.get(branch_el.type)
            items.append(
                PlacedItem(branch_el.id, branch_el.type, pin3_x, y, rotation)
            )
            coords[branch_el.id] = (pin3_x, y)

    ground_items: dict[str, PlacedItem] = {}
    for ground_node in spec.ground_nodes:
        touching = adjacency.get(ground_node, [])
        placed = [coords[el.id] for el, _ in touching if el.id in coords]
        gx = placed[0][0] if placed else 0.0
        gy = (min(p[1] for p in placed) if placed else 0.0) - _DY
        item_id = f"GROUND_{ground_node}"
        item = PlacedItem(
            item_id, "GROUND", gx, gy, ROTATION_FOR_ELEMENT.get("GROUND")
        )
        items.append(item)
        ground_items[ground_node] = item

    for controller, (cx, cy) in _CONTROLLER_COORDS.items():
        rotation = ROTATION_FOR_ELEMENT.get(controller)
        items.append(PlacedItem(controller, controller, cx, cy, rotation))

    wires = _compute_wires(spec, items, ground_items)
    return SchematicPlan(tuple(items), tuple(wires))


def _adjacency(spec: DeviceSpec) -> dict[str, list[tuple[Element, int]]]:
    """Map each node id to the ``(element, pin_number)`` pairs on it."""

    by_node: dict[str, list[tuple[Element, int]]] = {}
    for element in spec.elements:
        for pin in element.pins:
            by_node.setdefault(pin.node, []).append((element, pin.pin))
    return by_node


def _port_element(spec: DeviceSpec, num: int) -> Element:
    """Return the TermG element acting as port ``num``."""

    for element in spec.elements:
        if element.type == "TermG" and int(element.parameters.get("num", -1)) == num:
            return element
    raise ValueError(f"no TermG element with Num={num} in spec '{spec.label}'.")


def _series_path(
    spec: DeviceSpec,
    adjacency: dict[str, list[tuple[Element, int]]],
    in_port: Element,
    out_port: Element,
) -> list[Element]:
    """Return the ordered series MLIN/MTEE elements between the ports.

    The walk starts at the input port's node and follows series
    elements (MLIN through both pins, MTEE through pins 1 and 2) until
    it reaches the output port's node.
    """

    in_node = in_port.pins[0].node
    out_node = out_port.pins[0].node
    series: list[Element] = []
    visited: set[str] = set()
    node = in_node
    for _ in range(len(spec.elements) + 1):
        if node == out_node:
            break
        step = None
        for element, pin in adjacency.get(node, []):
            if element.type == "TermG" or element.id in visited:
                continue
            if element.type == "MLIN":
                other = next(p for p in element.pins if p.pin != pin)
                step = (element, other.node)
                break
            if element.type == "MTEE" and pin in (1, 2):
                through_pin = 2 if pin == 1 else 1
                other_node = next(
                    p.node for p in element.pins if p.pin == through_pin
                )
                step = (element, other_node)
                break
        if step is None:
            break
        element, node = step
        series.append(element)
        visited.add(element.id)
    return series


def _branch(
    spec: DeviceSpec,
    mtee: Element,
    adjacency: dict[str, list[tuple[Element, int]]],
) -> list[Element]:
    """Return the ordered MLIN chain on an MTEE's pin-3 shunt branch."""

    node = next(p.node for p in mtee.pins if p.pin == 3)
    branch: list[Element] = []
    visited: set[str] = {mtee.id}
    for _ in range(len(spec.elements) + 1):
        step = None
        for element, pin in adjacency.get(node, []):
            if element.id in visited or element.type != "MLIN":
                continue
            other = next(p for p in element.pins if p.pin != pin)
            step = (element, other.node)
            break
        if step is None:
            break
        element, node = step
        branch.append(element)
        visited.add(element.id)
    return branch


def _rotate_offset(
    offset: tuple[float, float], rotation: str | None
) -> tuple[float, float]:
    """Rotate a pin offset by an element's placement rotation."""

    ox, oy = offset
    if rotation == "DOWN":
        return (-ox, -oy)
    if rotation == "LEFT":
        return (-oy, ox)
    if rotation == "RIGHT":
        return (oy, -ox)
    return (ox, oy)  # None / "UP"


def _pin_abs(item: PlacedItem, pin_number: int) -> tuple[float, float]:
    """Absolute (x, y) of one pin of a placed item."""

    offsets = PIN_OFFSETS_AT_DEFAULT_ROTATION.get(item.element_type, {})
    ox, oy = _rotate_offset(offsets.get(pin_number, (0.0, 0.0)), item.rotation)
    return (item.x + ox, item.y + oy)


def _compute_wires(
    spec: DeviceSpec,
    items: list[PlacedItem],
    ground_items: dict[str, PlacedItem],
) -> list[Wire]:
    """Compute one wire per connected pin pair sharing a spec node.

    A node connecting two or more pins is wired as a chain of straight
    segments between consecutive pins. A node with a single pin (an
    open end) gets no wire.

    A degenerate segment whose two pins land on the *same* coordinate
    is skipped: a zero-length wire is rejected by ADS ("Could not Add
    Wire: Too few points specified"). This is only a guard -- the 2.0
    element spacing keeps every connected pin pair a real distance
    apart, so in practice no segment is skipped and every connection
    gets a visible wire.
    """

    item_by_id = {item.item_id: item for item in items}
    # node -> list of (item, pin_number)
    pins_by_node: dict[str, list[tuple[PlacedItem, int]]] = {}
    for element in spec.elements:
        for pin in element.pins:
            item = item_by_id.get(element.id)
            if item is not None:
                pins_by_node.setdefault(pin.node, []).append((item, pin.pin))
    for node, ground in ground_items.items():
        pins_by_node.setdefault(node, []).append((ground, 1))

    wires: list[Wire] = []
    for node, pin_list in pins_by_node.items():
        if len(pin_list) < 2:
            continue
        positions = [_pin_abs(item, pin) for item, pin in pin_list]
        for start, end in zip(positions, positions[1:]):
            if start == end:
                continue  # abutting pins -- ADS connects them implicitly
            wires.append(Wire(node=node, start=start, end=end))
    return wires


