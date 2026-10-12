"""Tests for the schematic layout engine (``tvtl_ads_builder_v2.layout``).

``compute_layout`` is the shared placement contract the topology
renderer and the OA emitter both consume; ``build_layout`` is the full
:class:`CircuitLayout` the renderer needs. All tests are pure Python --
no ADS, no third-party dependency.
"""

from __future__ import annotations

from tvtl_ads_builder_v2.device_spec import (
    DeviceSpec,
    Element,
    ElementPin,
    Node,
    Port,
)
from tvtl_ads_builder_v2.device_spec.schema import DEVICE_SPEC_VERSION
from tvtl_ads_builder_v2.layout import build_layout, compute_layout
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    LEVEL_3_CANONICAL_CIRCUITS,
    LEVEL_4_CANONICAL_CIRCUITS,
)

_REFERENCE = LEVEL_1_CANONICAL_CIRCUITS[0]


def _by_label(circuits, label):
    return next(c for c in circuits if c.label == label)


def _spec(nodes, elements, ports, *, ground_nodes=(), label="layout_test"):
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=_REFERENCE.analysis,
        substrate=_REFERENCE.substrate,
        nodes=tuple(Node(id=n) for n in nodes),
        elements=tuple(elements),
        ports=tuple(ports),
        label=label,
        ground_nodes=tuple(ground_nodes),
    )


def _termg(elem_id, net, num):
    return Element(
        id=elem_id, type="TermG",
        pins=(ElementPin(pin=1, node=net),),
        parameters={"z_ohm": 50.0, "num": num},
    )


def _two_terminal(elem_id, elem_type, net_a, net_b, parameters):
    return Element(
        id=elem_id, type=elem_type,
        pins=(ElementPin(pin=1, node=net_a), ElementPin(pin=2, node=net_b)),
        parameters=dict(parameters),
    )


def test_level1_two_port_lays_out_left_to_right():
    """A Level-1 two-port places P1 -> LINE -> P2 along the y = 0 row."""

    spec = LEVEL_1_CANONICAL_CIRCUITS[0]
    coords = compute_layout(spec)

    assert set(coords) == {e.id for e in spec.elements}
    assert all(y == 0.0 for _x, y in coords.values()), coords
    assert coords["P1"][0] < coords["LINE"][0] < coords["P2"][0]
    # one through-path column apart, starting at the origin column.
    assert {x for x, _y in coords.values()} == {0.0, 1.0, 2.0}


def test_level3_stub_hangs_below_the_through_path():
    """A Level-3 shorted tee stub is a shunt placed at y > 0."""

    spec = _by_label(LEVEL_3_CANONICAL_CIRCUITS, "tee_short_zero_stub")
    layout = build_layout(spec)
    assert not layout.is_grid_fallback

    by_id = {p.element.id: p for p in layout.placements}
    # the main line and the tee sit on the through row.
    for through_id in ("LINE_LEFT", "TEE1", "LINE_RIGHT"):
        assert by_id[through_id].y == 0.0
        assert by_id[through_id].role == "through"
    # the stub hangs below it.
    stub = by_id["STUB"]
    assert stub.role == "shunt"
    assert stub.y > 0.0
    # its grounded far end is recorded as a termination symbol.
    grounds = [s for s in layout.symbols if s.kind == "ground"]
    assert [s.net for s in grounds] == ["n_branch_end"]


def test_level3_open_stub_records_an_open_symbol():
    """The open-stub tee circuit records an open (not ground) symbol."""

    spec = _by_label(LEVEL_3_CANONICAL_CIRCUITS, "tee_open_zero_stub")
    layout = build_layout(spec)

    kinds = sorted(s.kind for s in layout.symbols)
    assert kinds == ["open"]
    assert layout.symbols[0].net == "n_branch_end"


def test_level4_mtee_branch_pin_faces_down():
    """The Level-4 MTEE keeps two through pins and one downward branch pin."""

    spec = _by_label(LEVEL_4_CANONICAL_CIRCUITS, "mtee_with_rlc_stub")
    layout = build_layout(spec)
    by_id = {p.element.id: p for p in layout.placements}

    tee = by_id["TEE1"]
    assert tee.role == "through"
    # exactly one branch pin faces down, the other two face left/right.
    assert sorted(tee.pin_sides.values()) == ["bottom", "left", "right"]

    # the lumped RLC stub is the shunt branch, hung below the row.
    stub = by_id["STUB"]
    assert stub.role == "shunt"
    assert stub.y > 0.0


def _sdd2p(elem_id, n_p1, n_m1, n_p2, n_m2, q_poly):
    """Build a 4-pin SDD2P element (port1 +/-, port2 +/-)."""

    return Element(
        id=elem_id,
        type="SDD2P",
        pins=(
            ElementPin(pin=1, node=n_p1),
            ElementPin(pin=2, node=n_m1),
            ElementPin(pin=3, node=n_p2),
            ElementPin(pin=4, node=n_m2),
        ),
        parameters={"q_poly_coefficients": q_poly},
    )


def _two_port_shunt_spec(shunt_element, extra_nodes, *, label):
    """A 2-port ``P1 - LINE_A - n_mid - LINE_B - P2`` with a shunt at n_mid.

    ``shunt_element`` taps ``n_mid`` and terminates on ``n_gnd``.
    """

    mlin_params = {
        "w_m": 2.95e-3, "l_m": 3.0e-3,
        "substrate_ref": _REFERENCE.substrate.name,
    }
    return _spec(
        nodes=("n_in", "n_mid", "n_out", "n_gnd", *extra_nodes),
        elements=(
            _termg("P1", "n_in", 1),
            _two_terminal("LINE_A", "MLIN", "n_in", "n_mid", mlin_params),
            shunt_element,
            _two_terminal("LINE_B", "MLIN", "n_mid", "n_out", mlin_params),
            _termg("P2", "n_out", 2),
        ),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        ground_nodes=("n_gnd",),
        label=label,
    )


def test_two_terminal_shunt_element_classifies_as_shunt():
    """An ordinary two-terminal R shunt classifies as a shunt branch."""

    resistor = _two_terminal("RSH", "R", "n_mid", "n_gnd", {"r_ohm": 25.0})
    spec = _two_port_shunt_spec(resistor, (), label="r_shunt")
    layout = build_layout(spec)
    assert not layout.is_grid_fallback
    by_id = {p.element.id: p for p in layout.placements}
    assert by_id["RSH"].role == "shunt"
    assert by_id["RSH"].y > 0.0


def test_sdd2p_shunt_branch_classifies_as_shunt():
    """A 4-pin SDD2P shunt classifies as a shunt branch -- no grid fallback.

    The SDD2P is a 2-port (4 pins) whose port 2 is a self-tie on
    ``n_return``; the layout's _PRIMARY_TERMINALS metadata makes the
    tracer treat it as logically two-terminal (port-1 pins 1 and 2).
    """

    q_poly = (0.0, 7.0e-12, -1.0e-12, 0.0, 0.0, 0.0, 0.0)
    varactor = _sdd2p(
        "VARACTOR", "n_mid", "n_gnd", "n_return", "n_return", q_poly
    )
    spec = _two_port_shunt_spec(
        varactor, ("n_return",), label="sdd2p_shunt"
    )
    layout = build_layout(spec)
    assert not layout.is_grid_fallback, "SDD2P shunt must not grid-fall-back"
    by_id = {p.element.id: p for p in layout.placements}
    # the through path stays on y = 0 ...
    for through_id in ("LINE_A", "LINE_B"):
        assert by_id[through_id].y == 0.0
        assert by_id[through_id].role == "through"
    # ... and the SDD2P hangs below it as a shunt.
    assert by_id["VARACTOR"].role == "shunt"
    assert by_id["VARACTOR"].y > 0.0
    # its grounded terminus is recorded as a ground symbol.
    assert [s.net for s in layout.symbols if s.kind == "ground"] == ["n_gnd"]


def _one_port(elem_id, elem_type, net, parameters):
    return Element(
        id=elem_id, type=elem_type,
        pins=(ElementPin(pin=1, node=net),),
        parameters=dict(parameters),
    )


def _two_port_chain_then_terminus_spec(terminus_element, *, label):
    """``P1 - LINE_A - n_mid - LINE_B - P2`` with a shunt STUB(n_mid->n_s) then a
    1-port ``terminus_element`` on ``n_s`` (the via/open-end emitter-accuracy case)."""

    mlin_params = {
        "w_m": 2.95e-3, "l_m": 3.0e-3,
        "substrate_ref": _REFERENCE.substrate.name,
    }
    return _spec(
        nodes=("n_in", "n_mid", "n_out", "n_s", "n_gnd"),
        elements=(
            _termg("P1", "n_in", 1),
            _two_terminal("LINE_A", "MLIN", "n_in", "n_mid", mlin_params),
            _two_terminal("STUB", "MLIN", "n_mid", "n_s", mlin_params),
            terminus_element,
            _two_terminal("LINE_B", "MLIN", "n_mid", "n_out", mlin_params),
            _termg("P2", "n_out", 2),
        ),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        ground_nodes=("n_gnd",),
        label=label,
    )


def test_one_port_viagnd_terminus_classifies_as_shunt():
    """A shunt branch terminated by a 1-port VIAGND (grounds internally) lays out
    as a shunt branch -- the 1-port is the bottom box, NOT a grid fallback.
    Regression for the _trace_shunt 1-port hook."""

    via = _one_port("VIA", "VIAGND", "n_s",
                    {"d_m": 1.5e-3, "t_m": 17.5e-6, "substrate_ref": _REFERENCE.substrate.name})
    spec = _two_port_chain_then_terminus_spec(via, label="viagnd_terminus")
    layout = build_layout(spec)
    assert not layout.is_grid_fallback, "1-port VIAGND terminus must not grid-fall-back"
    by_id = {p.element.id: p for p in layout.placements}
    assert by_id["STUB"].role == "shunt"
    assert by_id["VIA"].role == "shunt"
    # the 1-port terminus hangs BELOW the stub it terminates.
    assert by_id["VIA"].y > by_id["STUB"].y
    # VIAGND grounds internally -> no ground symbol is drawn for the branch.
    assert not [s for s in layout.symbols if s.kind == "ground"]


def test_one_port_mlef_open_end_classifies_as_shunt():
    """A shunt branch terminated by a 1-port MLEF (open-end fringe) lays out as a
    shunt branch with the MLEF as the bottom box -- no grid fallback, no symbol."""

    mlef = _one_port("OPENEND", "MLEF", "n_s",
                     {"w_m": 1.0e-3, "l_m": 0.0, "substrate_ref": _REFERENCE.substrate.name})
    spec = _two_port_chain_then_terminus_spec(mlef, label="mlef_terminus")
    layout = build_layout(spec)
    assert not layout.is_grid_fallback, "1-port MLEF terminus must not grid-fall-back"
    by_id = {p.element.id: p for p in layout.placements}
    assert by_id["OPENEND"].role == "shunt"
    assert by_id["OPENEND"].y > by_id["STUB"].y


def test_metadata_declared_multi_terminal_element_classifies(monkeypatch):
    """A hypothetical 3-terminal element with primary-terminal metadata
    classifies as a shunt -- the multi-terminal hook is generic."""

    import tvtl_ads_builder_v2.layout as layout_module

    # Register a hypothetical future 3-terminal element whose primary
    # (layout-relevant) terminals are pins 1 and 2.
    monkeypatch.setattr(
        layout_module,
        "_PRIMARY_TERMINALS",
        {**layout_module._PRIMARY_TERMINALS, "FAKE3": (1, 2)},
    )
    fake = Element(
        id="FAKE3X",
        type="FAKE3",
        pins=(
            ElementPin(pin=1, node="n_mid"),
            ElementPin(pin=2, node="n_gnd"),
            ElementPin(pin=3, node="n_aux"),
        ),
        parameters={},
    )
    spec = _two_port_shunt_spec(fake, ("n_aux",), label="fake3_shunt")
    layout = build_layout(spec)
    assert not layout.is_grid_fallback
    by_id = {p.element.id: p for p in layout.placements}
    assert by_id["FAKE3X"].role == "shunt"
    assert by_id["FAKE3X"].y > 0.0


def test_unclassifiable_topology_falls_back_to_grid():
    """A forking shunt does not fit the linear model -> grid fallback."""

    spec = _spec(
        nodes=("n_in", "n_mid", "n_out", "n_g1", "n_g2"),
        elements=(
            _termg("P1", "n_in", 1),
            _two_terminal("LA", "MLIN", "n_in", "n_mid",
                          {"w_m": 2.95e-3, "l_m": 3.0e-3,
                           "substrate_ref": _REFERENCE.substrate.name}),
            _two_terminal("LB", "MLIN", "n_mid", "n_out",
                          {"w_m": 2.95e-3, "l_m": 3.0e-3,
                           "substrate_ref": _REFERENCE.substrate.name}),
            _two_terminal("RS1", "R", "n_mid", "n_g1", {"r_ohm": 25.0}),
            _two_terminal("RS2", "R", "n_mid", "n_g2", {"r_ohm": 25.0}),
            _termg("P2", "n_out", 2),
        ),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        ground_nodes=("n_g1", "n_g2"),
        label="forking_shunt",
    )
    layout = build_layout(spec)
    assert layout.is_grid_fallback
    # the fallback still places -- and names -- every element.
    coords = compute_layout(spec)
    assert set(coords) == {e.id for e in spec.elements}


