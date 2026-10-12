"""Hierarchical (unit-cell-symbol) OA emission.

A NEW, ADDITIVE emission path that renders a uniform self-pumped cascade as a clean
HIERARCHICAL ADS design:

* one **unit-cell subcircuit** (``<lib>:<uc>:schematic``) holding a single cell's
  elements, with two ordered terminals (IN=1, OUT=2) and an internal GROUND for the
  per-cell ground net (which maps to the global net inside the subcircuit, so the cell
  symbol needs no ground pin);
* a clean **unit-cell symbol** (``<lib>:<uc>:symbol``) -- a labelled box whose only
  pins are IN / OUT;
* a **top-level schematic** (``<lib>:<top>:schematic``) of N unit-cell symbol
  instances cascaded OUT->IN through the inter-cell I_Probes, with the headline
  instrumentation at the top-level ports only (input/output sources or Term, HB
  controller, S_Param, matching networks). Per-cell m_eff *plot* annotations (NAME)
  are NOT surfaced at the top; the per-cell varactor I_Probe stays INSIDE the unit cell
  so the ADS-flattened netlist is element-and-connectivity identical to the flat deck.

The FLAT path (:func:`oa_emitter.emit_workspace_from_spec`) is untouched: this module
only *imports* its element/param/routing helpers. Hierarchy is realized with the OA API
established by live OA probes: ``db.create_symbol``,
``design.add_numbered_term(net, name, number, db.TermType.INPUT_OUTPUT)``,
``design.add_pin_fig_for_term_type``, ``design.add_pin``, and wire-backed connectivity
for the top-level symbol instances.

Scope: the uniform fixed-geometry cascade (every cell identical, the proof case --
VarA amplifier N=10). The flat spec is partitioned by element id (``*_C{k}``) into the
per-cell template (cell 1) and the cascade scaffolding; ``generate_netlist`` then emits
``define UC (IN OUT) ... end UC`` + ``UC:Xk`` instances (hpeesofsim expands the subckt).
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from ..device_spec import DeviceSpec, Element, resolve_spec
from ..netlist.elements.hb_controller import HB_CONTROLLER_NAME
from .oa_emitter import (
    _GROUND_CELL,
    _HB_CELL,
    _ID_RESOLVED_PIN_TYPES,
    _MSUB_CELL,
    _OA_CELL,
    _P1TONE_CELL,
    _PIN_INDEX,
    _SPARAM_CELL,
    _add_polyline,
    _clear_track,
    _component_positions,
    _effective_elements,
    _element_params,
    _ensure_keysight_on_path,
    _msub_params,
    _p1tone_params,
    _pin_by_logical_id,
    _pin_rows,
    _pin_xy,
    _ROUTE_EPS,
    _route_wire,
    _set_hb_controller,
    _set_hb_two_tone_controller,
    _set_sdd2p_body,
    _source_terminal_pins,
    _sweep_params,
)

_LOG = logging.getLogger(__name__)

# A cell-owned element id ends with ``_C{k}`` or ``_C{k}_{seg}`` (e.g. LINE_LEFT_C3,
# STUB_C3_0, IPR_VAR_C3). The inter-cell probe ``IPR_C3C4`` does NOT match (``_C3``
# is followed by ``C4``), so it is correctly classified as scaffolding.
_CELL_ID_RE = re.compile(r"_C(\d+)(?:_\d+)?$")

# An inter-cell I_Probe (scaffolding) is ``IPR_C{k}C{k+1}`` -- two adjacent cell tags.
_INTER_CELL_RE = re.compile(r"_C(\d+)C(\d+)$")
_STUB_RE = re.compile(r"^STUB_C\d+_(\d+)$")
_VIAGND_RE = re.compile(r"^VIAGND(\d*)_C\d+$")

# --- Layout-refinement constants --------------------------------------
# Terminal clearance: the UC IN/OUT terminals sit this many ADS units beyond the cell's left/right
# extent, on the through-line row, each joined to its through-line end pin by a short
# off-row wire so it reads as connected to the through-path -- not parked at a bottom
# corner. Placement only; the terminal pin and the end pin already share the cell net,
# so the wire is cosmetic and electrical parity is unchanged.
_TERM_STUB = 3.0
# Symbol pitch: the unit-cell symbol's half-width and the top-level centre-to-centre
# pitch, so the N symbol instances read as a tight left->right row with short
# inter-cell connecting wires (instead of spread at their wide flat-layout x).
_UC_SYM_HALF_W = 3.0
_UC_SYM_HALF_H = 2.0
# Cosmetic (placement-only) widening: widen the inter-cell pitch from the
# original 10.0 so the unit_cell:Xk cascade -- and the inter-cell IPR labels that
# drop into each gap -- read with breathing room instead of cramped.
_TOP_CELL_STRIDE = 16.0
# Left-of-input instrumentation placement. The controller column
# (MSUB / S_Param / HB) and the signal/pump source stack park left of IPR_IN
# (which sits at x = -_TOP_CELL_STRIDE/2 on the rail), spread far enough apart --
# vertically within each column and horizontally between the two columns -- that
# the instance boxes and their name labels stop overlapping (the SMV2020 left-edge
# cluster was the worst case). Connectivity is by net assignment, independent of
# these coordinates: the controllers/MSUB carry no routed pins, and each source
# return keeps its own coincident GROUND, so the netlist is unchanged (parity gate).
_INSTR_DX = 18.0      # controller column: this far LEFT of IPR_IN
_SRC_DX = 7.0         # source column: this far LEFT of IPR_IN
_MSUB_Y = 12.0        # MSUB parked well above the rail
_SPARAM_Y = -7.0      # S_Param controller, below the rail
_HB_GAP = 8.0         # HB controller sits this far BELOW S_Param (clear of its label)
_SRC_Y0 = -3.0        # first source y (the stack descends from here)
_SRC_PITCH = 4.5      # vertical gap between stacked sources
_TOP_INPUT_TRACK = -1.5
# Matching-network rail slots (drawing only). A
# low-edge-impedance design carries single-stub (or quarter-wave) matching as
# scaffolding elements ``MATCH_{IN,OUT}_{SERIES,QW,STUB,STUB_VIA}``; the series /
# quarter-wave line sits on the through row beside the port probe and the shunt
# stub (and its ground via) drop onto rows this far below it.
_MATCH_BRANCH_DY = 4.0

_DIRECT_ROUTE = object()

_UC_THROUGH_Y = 0.0
_UC_BRANCH_X = 5.5
_UC_BRANCH_START_Y = -4.0
_UC_BRANCH_STEP = 4.0
_UC_FORK_DX = 5.0
_UC_OPEN_DX = 4.0
_UC_TERMINAL_VIA_DX = 8.0
_UC_TERMINAL_MLEF_DX = 12.0


def _cell_index(element_id: str) -> int | None:
    """The 1-based cell index an element belongs to, or None for scaffolding."""

    m = _CELL_ID_RE.search(element_id)
    return int(m.group(1)) if m else None


def _uc_pins_by_term(inst) -> dict[str, object]:
    """Map a unit-cell instance's inst_pins to their terminal name (IN / OUT)."""

    out: dict[str, object] = {}
    for pin in inst.inst_pins:
        try:
            name = pin.master_pin.term.name
        except Exception:  # noqa: BLE001
            name = None
        if name:
            out[name] = pin
    return out


def _same_row(coords: list[tuple[float, float]]) -> bool:
    """True when every coordinate lies on one horizontal drawing row."""

    return all(abs(coords[0][1] - other[1]) < _ROUTE_EPS for other in coords[1:])


def _coord_in(
    coord: tuple[float, float], coords: list[tuple[float, float]]
) -> bool:
    return any(
        abs(coord[0] - other[0]) < _ROUTE_EPS
        and abs(coord[1] - other[1]) < _ROUTE_EPS
        for other in coords
    )


def _same_row_is_clear(
    coords: list[tuple[float, float]],
    all_pin_xy: list[tuple[float, float]],
) -> bool:
    """True when direct horizontal wires would not pass through another pin."""

    if not _same_row(coords):
        return False
    y = coords[0][1]
    for end in coords[1:]:
        x_lo, x_hi = sorted((coords[0][0], end[0]))
        for pin_xy in all_pin_xy:
            if _coord_in(pin_xy, coords):
                continue
            if abs(pin_xy[1] - y) >= _ROUTE_EPS:
                continue
            if x_lo + _ROUTE_EPS < pin_xy[0] < x_hi - _ROUTE_EPS:
                return False
    return True


def _route_track_for_node(
    node_id: str,
    coords: list[tuple[float, float]],
    all_pin_xy: list[tuple[float, float]],
    pin_rows: list[float],
    used_tracks: set[float],
    track_for_node=None,
) -> float | None:
    """Return an off-row track only when a direct same-row wire is not wanted."""

    routed_x = len(coords) > 1 and any(
        abs(coords[0][0] - other[0]) > _ROUTE_EPS for other in coords[1:]
    )
    if not routed_x:
        return None
    override = track_for_node(node_id, coords) if track_for_node else None
    if override is _DIRECT_ROUTE:
        return None
    if override is not None:
        return float(override)
    if _same_row_is_clear(coords, all_pin_xy):
        return None
    ys = [coord[1] for coord in coords]
    return _clear_track(min(ys), max(ys), pin_rows, used_tracks)


def _route_node_wires(
    design,
    node_id: str,
    coords: list[tuple[float, float]],
    all_pin_xy: list[tuple[float, float]],
    pin_rows: list[float],
    used_tracks: set[float],
    track_for_node=None,
) -> None:
    track = _route_track_for_node(
        node_id, coords, all_pin_xy, pin_rows, used_tracks, track_for_node
    )
    for other in coords[1:]:
        _add_polyline(design, _route_wire(coords[0], other, track))


def _connect_and_route(
    design,
    node_pins: dict[str, list],
    *,
    ground_nets: dict | None = None,
    track_for_node=None,
) -> None:
    """Assign pins to nets and route with straight same-row wires by default.

    Off-row tracks are still used when pins span different rows, or when a
    caller supplies an explicit route override for a known ADS shorting case.
    """

    ground_nets = ground_nets or {}
    all_xy = [_pin_xy(p) for pins in node_pins.values() for p in pins]
    pin_rows = _pin_rows(all_xy)
    used_tracks: set[float] = set()
    for node_id, pin_list in node_pins.items():
        net = ground_nets.get(node_id) or design.find_or_add_net(node_id)
        for pin in pin_list:
            pin.net = net
        coords = [_pin_xy(p) for p in pin_list]
        _route_node_wires(
            design, node_id, coords, all_xy, pin_rows, used_tracks, track_for_node
        )


def _top_track_for_node(node_id: str, coords: list[tuple[float, float]]):
    """Presentation-only top-level exceptions to straight-first routing."""

    if node_id == "n_input":
        return _TOP_INPUT_TRACK
    return None


def _uc_track_for_node(node_id: str, coords: list[tuple[float, float]]):
    """Unit-cell drawing exceptions to conservative off-row routing."""

    through_nodes = {"n_pin", "n_c1_l", "n_c1_r", "n_c1_out"}
    if node_id in through_nodes and _same_row(coords):
        return _DIRECT_ROUTE
    return None


def _place_elements(
    design,
    elements: list[Element],
    positions: dict,
    substrate_name: str,
    *,
    angles: dict[str, float] | None = None,
) -> dict[str, object]:
    """Place a list of effective elements; return id -> OA instance. Reused by both levels."""

    def setp(inst, name, value):
        if name in inst.parameters:
            inst.parameters[name].value = value

    oa_by_id: dict[str, object] = {}
    for el in elements:
        cell = _OA_CELL.get(el.type)
        if cell is None:
            raise ValueError(f"hier emit: element {el.id!r} unsupported type {el.type!r}")
        inst = design.add_instance(cell, positions[el.id], name=el.id, angle=(angles or {}).get(el.id, 0))
        for key, value in _element_params(el, substrate_name).items():
            setp(inst, key, value)
        if el.type == "SDD2P":
            _set_sdd2p_body(inst, el)
        oa_by_id[el.id] = inst
    return oa_by_id


def _unit_cell_drawing(elements: list[Element], fallback: dict[str, tuple[float, float]]):
    """Return unit-cell-only placement/orientation for a clean shunt branch."""

    positions: dict[str, tuple[float, float]] = {}
    angles: dict[str, float] = {}

    def put(eid: str, xy: tuple[float, float], angle: float = 0.0) -> None:
        if any(el.id == eid for el in elements):
            positions[eid] = xy
            if angle:
                angles[eid] = angle

    put("LINE_LEFT_C1", (1.0, _UC_THROUGH_Y), 180.0)
    put("TEE_C1", (5.0, _UC_THROUGH_Y))
    put("LINE_RIGHT_C1", (11.0, _UC_THROUGH_Y), 180.0)

    branch_angles = {
        "LPKG_C1": 90.0,
        "RPKG_C1": -90.0,
        "VARACTOR_C1": 0.0,
        "IPR_VAR_C1": -90.0,
    }
    y = _UC_BRANCH_START_Y
    for eid, angle in branch_angles.items():
        if any(el.id == eid for el in elements):
            put(eid, (_UC_BRANCH_X, y), angle)
            y -= _UC_BRANCH_STEP

    stubs = sorted(
        (el.id for el in elements if _STUB_RE.match(el.id)),
        key=lambda eid: int(_STUB_RE.match(eid).group(1)),
    )
    stub_node_y: dict[str, float] = {}
    for eid in stubs:
        el = next(e for e in elements if e.id == eid)
        put(eid, (_UC_BRANCH_X, y - 1.0), 90.0)
        if len(el.pins) >= 2:
            stub_node_y[el.pins[1].node] = y - 1.0
        y -= _UC_BRANCH_STEP

    term_y = y + _UC_BRANCH_STEP - 1.0 if stubs else y
    ostub = next((el for el in elements if el.id == "OSTUB_C1"), None)
    ostub_input_node = ostub.pins[0].node if ostub is not None and ostub.pins else None
    if ostub is not None:
        put("OSTUB_C1", (_UC_BRANCH_X + _UC_OPEN_DX + 1.0, term_y), 180.0)
    if any(el.id == "MLEF_C1" for el in elements):
        mlef_y = term_y
        if ostub is not None and len(ostub.pins) >= 2:
            mlef_y = term_y
        elif stubs:
            mlef_y = stub_node_y.get(stubs[-1] and next(e for e in elements if e.id == stubs[-1]).pins[1].node, term_y)
        mlef_dx = _UC_TERMINAL_MLEF_DX if ostub is not None else 2.0 * _UC_OPEN_DX
        put("MLEF_C1", (_UC_BRANCH_X + mlef_dx, mlef_y), 0.0)

    via_index = 0
    for el in sorted((e for e in elements if _VIAGND_RE.match(e.id)), key=lambda e: e.id):
        node = el.pins[0].node if el.pins else ""
        via_y = stub_node_y.get(node, term_y - via_index * 2.0)
        via_dx = _UC_TERMINAL_VIA_DX if node == ostub_input_node else _UC_FORK_DX
        put(el.id, (_UC_BRANCH_X + via_dx, via_y), 0.0)
        via_index += 1

    for el in elements:
        positions.setdefault(el.id, fallback[el.id])
    return positions, angles


def _node_pins_for(elements: list[Element], oa_by_id: dict[str, object]) -> dict[str, list]:
    """Collect node -> [OA pin] for a list of placed effective elements."""

    node_pins: dict[str, list] = {}
    for el in elements:
        inst = oa_by_id[el.id]
        pins = list(inst.inst_pins)
        pin_index = _PIN_INDEX.get(el.type)
        id_resolved = el.type in _ID_RESOLVED_PIN_TYPES
        for ep in el.pins:
            pin_obj = _pin_by_logical_id(inst, ep.pin) if id_resolved else None
            if pin_obj is None:
                idx = pin_index[ep.pin] if pin_index is not None else ep.pin - 1
                if not 0 <= idx < len(pins):
                    raise ValueError(
                        f"hier emit: {el.id!r} pin {ep.pin} out of range ({len(pins)})"
                    )
                pin_obj = pins[idx]
            node_pins.setdefault(ep.node, []).append(pin_obj)
    return node_pins


def _has_match_in(scaffolding: list[Element]) -> bool:
    """True when an input matching network is present (MATCH_IN_* scaffolding)."""

    return any(el.id.startswith("MATCH_IN_") for el in scaffolding)


def _rail_overrides(
    scaffolding: list[Element], n_cells: int
) -> dict[str, tuple[float, float]]:
    """Top-level rail coordinates for the scaffolding that rides the cascade.

    Returns ``id -> (x, y)`` for the inter-cell probes, the input/output current
    probes, the output port, and the input/output
    matching networks. The input network parks one slot LEFT of ``IPR_IN`` and
    the output network one slot RIGHT of ``IPR_OUT`` (the load port stepping a
    further half-slot right so it stays the rightmost rail element); the series /
    quarter-wave line sits on the through row and the shunt stub / ground via on
    rows below, the same way ``IPR_IN`` / ``IPR_OUT`` / ``P_OUT`` are special-
    cased. Every other scaffolding element keeps its flat-layout coordinate.

    Placement only: the flattened ``.ckt`` carries no schematic geometry, so the
    coordinates returned here never affect connectivity or parity -- they only
    stop ``MATCH_IN_*`` landing on top of ``X1`` and ``MATCH_OUT_*`` stranding
    past the rail end at their uncompressed flat coordinates.
    """

    half_stride = _TOP_CELL_STRIDE / 2.0

    def _xk(k: int) -> float:
        return (k - 1) * _TOP_CELL_STRIDE

    has_match_out = any(el.id.startswith("MATCH_OUT_") for el in scaffolding)
    mx_in = _xk(1) - _TOP_CELL_STRIDE              # one slot left of IPR_IN
    mx_out = _xk(n_cells) + _TOP_CELL_STRIDE       # one slot right of IPR_OUT
    p_out_x = (
        mx_out + half_stride if has_match_out
        else _xk(n_cells) + _TOP_CELL_STRIDE
    )

    def _match_xy(eid: str) -> tuple[float, float]:
        base_x = mx_in if eid.startswith("MATCH_IN_") else mx_out
        if eid.endswith("_STUB_VIA"):           # shunt ground via, lowest row
            return (base_x, -2.0 * _MATCH_BRANCH_DY)
        if eid.endswith("_STUB"):               # shunt stub, one row down
            return (base_x, -_MATCH_BRANCH_DY)
        return (base_x, 0.0)                     # series / quarter-wave, on rail

    overrides: dict[str, tuple[float, float]] = {}
    for el in scaffolding:
        m = _INTER_CELL_RE.search(el.id)
        if m:  # inter-cell probe IPR_C{k}C{k+1} -> midpoint of the Xk..Xk+1 gap
            overrides[el.id] = (_xk(int(m.group(1))) + half_stride, 0.0)
        elif el.id == "IPR_IN":
            overrides[el.id] = (_xk(1) - half_stride, 0.0)
        elif el.id == "IPR_OUT":
            overrides[el.id] = (_xk(n_cells) + half_stride, 0.0)
        elif el.id == "P_OUT":
            overrides[el.id] = (p_out_x, 0.0)
        elif el.id.startswith("MATCH_IN_") or el.id.startswith("MATCH_OUT_"):
            overrides[el.id] = _match_xy(el.id)
    return overrides


def emit_hierarchical_workspace_from_spec(
    spec: DeviceSpec,
    workspace_dir: str | Path,
    *,
    library_name: str = "oa_hier_lib",
    top_cell_name: str | None = None,
    uc_cell_name: str = "UC",
    auto_netlist: bool = True,
) -> dict:
    """Emit a hierarchical (unit-cell-symbol) ADS workspace for a uniform cascade.

    Returns ``{"workspace_path", "ads_netlist_path", "n_cells", "uc_cell", "top_cell"}``.
    Requires ADS (``keysight.ads.de`` imported here). Does NOT touch the flat path.
    """

    spec = resolve_spec(spec)
    top_cell_name = top_cell_name or f"{spec.label}_top"
    workspace_dir = Path(workspace_dir)
    circuit_dir = workspace_dir.parent
    circuit_dir.mkdir(parents=True, exist_ok=True)
    substrate_name = spec.substrate.name

    # --- partition the flat spec: per-cell template vs cascade scaffolding ---
    effective = _effective_elements(spec)
    positions = _component_positions(spec)  # ADS coords for every effective element
    by_cell: dict[int, list[Element]] = {}
    scaffolding: list[Element] = []
    for el in effective:
        k = _cell_index(el.id)
        if k is None:
            scaffolding.append(el)
        else:
            by_cell.setdefault(k, []).append(el)
    if not by_cell:
        raise ValueError("hier emit: no per-cell elements found (expected ids like *_C1)")
    n_cells = max(by_cell)
    uc_template = by_cell[1]

    # Sanity: the cascade must be uniform (every cell the same element-type multiset).
    ref_types = sorted(e.type for e in by_cell[1])
    for k in range(2, n_cells + 1):
        if sorted(e.type for e in by_cell.get(k, [])) != ref_types:
            raise ValueError(
                f"hier emit: cell {k} differs from cell 1 -- non-uniform cascade "
                "is out of scope for the A3 proof"
            )

    _ensure_keysight_on_path()
    import keysight.ads.de as de
    from keysight.ads.de import db_uu as db

    IO = db.TermType.INPUT_OUTPUT

    if de.workspace_is_open():
        de.close_workspace()
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace = de.create_workspace(str(workspace_dir))
    workspace.open()
    lib_path = workspace_dir / library_name
    de.create_new_library(library_name, str(lib_path))
    workspace.add_library(library_name, str(lib_path), de.LibraryMode.NON_SHARED)
    library = workspace.open_library(library_name, str(lib_path), de.LibraryMode.NON_SHARED)
    library.setup_schematic_tech()

    # ============================ UNIT CELL ================================= #
    uc = db.create_schematic(f"{library_name}:{uc_cell_name}:schematic")
    msub = uc.add_instance(_MSUB_CELL, (-8.0, 6.0), name=spec.substrate.name, angle=0)
    for key, value in _msub_params(spec.substrate).items():
        if key in msub.parameters:
            msub.parameters[key].value = value

    # cell-1 element positions translated to a local origin
    cell_xy = {e.id: positions[e.id] for e in uc_template}
    min_x = min(x for x, _y in cell_xy.values())
    min_y = min(y for _x, y in cell_xy.values())
    uc_fallback_pos = {eid: (x - min_x, y - min_y) for eid, (x, y) in cell_xy.items()}
    uc_pos, uc_angles = _unit_cell_drawing(uc_template, uc_fallback_pos)
    uc_by_id = _place_elements(uc, uc_template, uc_pos, substrate_name, angles=uc_angles)

    # connectivity inside the cell (by node)
    uc_node_pins = _node_pins_for(uc_template, uc_by_id)

    # per-cell ground -> internal GROUND symbol(s) on the cell ground net(s)
    cell_ground_nodes = {f"n_c1_g"}
    uc_ground_nets: dict[str, object] = {}
    for gi, gnode in enumerate(sorted(n for n in cell_ground_nodes if n in uc_node_pins)):
        # place the GROUND at the cell's lowest pin on that node
        gpinxy = min((_pin_xy(p) for p in uc_node_pins[gnode]), key=lambda c: c[1])
        gnd = uc.add_instance(_GROUND_CELL, (gpinxy[0], gpinxy[1] - 3.0),
                              name=f"GND{gi + 1}", angle=0)
        gpin = list(gnd.inst_pins)[0]
        gnet = gpin.net or uc.find_or_add_net("0")
        gpin.net = gnet
        uc_ground_nets[gnode] = gnet
        uc_node_pins[gnode].append(gpin)

    _connect_and_route(
        uc, uc_node_pins, ground_nets=uc_ground_nets, track_for_node=_uc_track_for_node
    )

    # ordered terminals IN(1)=cell input net, OUT(2)=cell output net, placed at the
    # THROUGH-LINE ends and on the through-path row. Their cosmetic wires use the
    # same straight-first routing helper as every other node, so collinear terminals
    # connect directly while off-row cases still get clear tracks. Placement only:
    # the terminal and boundary pin already share the cell net (set by
    # add_numbered_term), so the connecting wire is cosmetic and parity is unchanged.
    in_net_name = "n_pin"          # cell-1 input net (assembler: cell_in for k==1)
    out_net_name = "n_c1_out"      # cell.output_terminal_node(1)

    def _boundary_xy(net_name):
        """Coordinate of the cell's boundary (through-line end) pin on ``net_name``."""
        pins = uc_node_pins.get(net_name)
        return _pin_xy(pins[0]) if pins else None

    in_b = _boundary_xy(in_net_name)
    out_b = _boundary_xy(out_net_name)
    all_xy = [_pin_xy(p) for pins in uc_node_pins.values() for p in pins] or [(0.0, 0.0)]
    cell_min_x = min(x for x, _y in all_xy)
    cell_max_x = max(x for x, _y in all_xy)
    through_y = (in_b or out_b or (0.0, 0.0))[1]
    in_xy = (cell_min_x - _TERM_STUB, (in_b[1] if in_b is not None else through_y))
    out_xy = (cell_max_x + _TERM_STUB, (out_b[1] if out_b is not None else through_y))
    term_all_xy = all_xy + [in_xy, out_xy]
    term_pin_rows = _pin_rows(term_all_xy)
    term_tracks: set[float] = set()
    t_in = uc.add_numbered_term(uc.find_or_add_net(in_net_name), "IN", 1, IO)
    uc.add_pin(t_in, uc.add_pin_fig_for_term_type(IO, in_xy))
    t_out = uc.add_numbered_term(uc.find_or_add_net(out_net_name), "OUT", 2, IO)
    uc.add_pin(t_out, uc.add_pin_fig_for_term_type(IO, out_xy))
    if in_b is not None:
        _route_node_wires(
            uc, in_net_name, [in_xy, in_b], term_all_xy, term_pin_rows, term_tracks,
            _uc_track_for_node,
        )
    if out_b is not None:
        _route_node_wires(
            uc, out_net_name, [out_xy, out_b], term_all_xy, term_pin_rows, term_tracks,
            _uc_track_for_node,
        )
    uc.save_design()

    # ============================ UNIT-CELL SYMBOL ========================== #
    sym = db.create_symbol(f"{library_name}:{uc_cell_name}:symbol")
    slay = sym.get_layer_for_pin()
    half_w, half_h = _UC_SYM_HALF_W, _UC_SYM_HALF_H
    sym.add_rectangle(slay, (-half_w, -half_h), (half_w, half_h))
    s_in = sym.add_numbered_term(sym.find_or_add_net("IN"), "IN", 1, IO)
    sym.add_pin(s_in, sym.add_pin_fig_for_term_type(IO, (-half_w, 0.0)))
    s_out = sym.add_numbered_term(sym.find_or_add_net("OUT"), "OUT", 2, IO)
    sym.add_pin(s_out, sym.add_pin_fig_for_term_type(IO, (half_w, 0.0)))
    sym.save_design()

    # ============================ TOP LEVEL ================================= #
    # Off-rail margins: the controller column (MSUB / S_Param / HB) parks far left
    # of IPR_IN (at x = -_TOP_CELL_STRIDE/2) and the source stack sits between them,
    # so neither the controllers nor the sources->IPR_IN feed crowd the input section
    # and their labels no longer overlap (placement only). When an
    # input matching network is present it claims the slot immediately left of
    # IPR_IN, so both columns step a further stride left to keep clear of it.
    match_in_span = _TOP_CELL_STRIDE if _has_match_in(scaffolding) else 0.0
    instr_x = -_TOP_CELL_STRIDE / 2.0 - match_in_span - _INSTR_DX
    src_x = -_TOP_CELL_STRIDE / 2.0 - match_in_span - _SRC_DX
    top = db.create_schematic(f"{library_name}:{top_cell_name}:schematic")
    tmsub = top.add_instance(_MSUB_CELL, (instr_x, _MSUB_Y), name=spec.substrate.name, angle=0)
    for key, value in _msub_params(spec.substrate).items():
        if key in tmsub.parameters:
            tmsub.parameters[key].value = value

    # ---- tight top-level layout: a single left->right rail at y=0 -------------------
    # The unit-cell symbols sit on a fixed pitch (``_TOP_CELL_STRIDE``) so the cascade
    # reads compactly; the inter-cell I_Probes drop into the gap between consecutive
    # symbols, IPR_IN / IPR_OUT / the output port sit just beyond the end symbols, and
    # the input/output matching networks (when present) park in the slot beside their
    # port probe -- so every connecting wire is short (see :func:`_rail_overrides`).
    def _xk(k: int) -> float:
        return (k - 1) * _TOP_CELL_STRIDE

    top_positions: dict[str, tuple[float, float]] = dict(positions)
    top_positions.update(_rail_overrides(scaffolding, n_cells))
    top_angles = {
        el.id: 180.0
        for el in scaffolding
        if el.id.startswith("MATCH_IN_")
        and not (el.id.endswith("_STUB") or el.id.endswith("_STUB_VIA"))
    }
    scaff_by_id = _place_elements(
        top, scaffolding, top_positions, substrate_name, angles=top_angles
    )

    # N unit-cell symbol instances on the tight pitch (clean left->right row)
    uc_insts: dict[int, object] = {}
    for k in range(1, n_cells + 1):
        uc_insts[k] = top.add_instance(
            (library_name, uc_cell_name, "symbol"), (_xk(k), 0.0), name=f"X{k}", angle=0
        )

    # S_Param controller
    sp = top.add_instance(_SPARAM_CELL, (instr_x, _SPARAM_Y), name="SP1", angle=0)
    for key, value in _sweep_params(spec.analysis).items():
        if key in sp.parameters:
            sp.parameters[key].value = value
    # HB controller(s)
    if spec.hb_single_tone is not None:
        hb = top.add_instance(_HB_CELL, (instr_x, _SPARAM_Y - _HB_GAP), name=HB_CONTROLLER_NAME, angle=0)
        _set_hb_controller(hb, spec.hb_single_tone)
    if spec.hb_two_tone is not None:
        hb = top.add_instance(_HB_CELL, (instr_x, _SPARAM_Y - _HB_GAP), name=HB_CONTROLLER_NAME, angle=0)
        _set_hb_two_tone_controller(hb, spec.hb_two_tone)

    # P_1Tone sources, stacked in their own column left of IPR_IN (clean input
    # section), on a wider pitch so the stacked source boxes/labels do not overlap.
    src_pos_pins: dict[str, list] = {}
    src_neg_pins: list = []
    for si, source in enumerate(spec.sources):
        s_inst = top.add_instance(_P1TONE_CELL, (src_x, _SRC_Y0 - _SRC_PITCH * si),
                                  name=source.name, angle=0)
        for key, value in _p1tone_params(source).items():
            if key in s_inst.parameters:
                s_inst.parameters[key].value = value
        sig, gnd = _source_terminal_pins(s_inst)
        src_pos_pins.setdefault(source.port_node, []).append(sig)
        src_neg_pins.append(gnd)

    # connectivity by node: scaffolding element pins + UC terminal pins + sources
    node_pins = _node_pins_for(scaffolding, scaff_by_id)
    for k in range(1, n_cells + 1):
        tp = _uc_pins_by_term(uc_insts[k])
        in_net = "n_pin" if k == 1 else f"n_pc{k - 1}c{k}"
        out_net = f"n_c{k}_out"
        if "IN" in tp:
            node_pins.setdefault(in_net, []).append(tp["IN"])
        if "OUT" in tp:
            node_pins.setdefault(out_net, []).append(tp["OUT"])
    for port_node, pins in src_pos_pins.items():
        node_pins.setdefault(port_node, []).extend(pins)

    # GROUND symbols for touched top-level ground nodes (matching-network vias, etc.)
    top_ground = set(spec.ground_nodes)
    touched = {p.node for el in scaffolding for p in el.pins}
    ground_nets: dict[str, object] = {}
    for gi, gnode in enumerate(sorted(n for n in top_ground if n in touched)):
        if gnode not in node_pins:
            continue
        gpinxy = min((_pin_xy(p) for p in node_pins[gnode]), key=lambda c: c[1])
        gnd = top.add_instance(_GROUND_CELL, (gpinxy[0], gpinxy[1] - 3.0),
                               name=f"GND{gi + 1}", angle=0)
        gpin = list(gnd.inst_pins)[0]
        gnet = gpin.net or top.find_or_add_net("0")
        gpin.net = gnet
        ground_nets[gnode] = gnet
        node_pins[gnode].append(gpin)

    _connect_and_route(top, node_pins, ground_nets=ground_nets, track_for_node=_top_track_for_node)

    # source returns -> coincident GROUND (the flat emitter's connectivity-safe pattern)
    for si, pin in enumerate(src_neg_pins):
        gsrc = top.add_instance(_GROUND_CELL, _pin_xy(pin), name=f"GND_SRC{si + 1}", angle=0)
        gpin = list(gsrc.inst_pins)[0]
        gnet = gpin.net or top.find_or_add_net("0")
        gpin.net = gnet
        pin.net = gnet

    top.save_design()

    ads_netlist_path = circuit_dir / "ads_hier_netlist.txt"
    if auto_netlist:
        try:
            ads_netlist_path.write_text(str(top.generate_netlist()), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("hier netlist generation failed: %s", exc)

    if de.workspace_is_open():
        de.close_workspace()

    return {
        "workspace_path": str(workspace_dir),
        "ads_netlist_path": str(ads_netlist_path),
        "n_cells": n_cells,
        "uc_cell": uc_cell_name,
        "top_cell": top_cell_name,
        "library_name": library_name,
    }


