"""Tests for the topology SVG renderer (``tvtl_ads_builder_v2.topology``).

Each test renders a DeviceSpec to an SVG file and parses it back with
the standard-library XML parser. All tests are pure Python: no ADS, no
browser, no third-party dependency.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from tvtl_ads_builder_v2.device_spec import (
    DeviceSpec,
    Element,
    ElementPin,
    Node,
    Port,
)
from tvtl_ads_builder_v2.device_spec.schema import DEVICE_SPEC_VERSION
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    LEVEL_3_CANONICAL_CIRCUITS,
)
from tvtl_ads_builder_v2.topology import render_topology_svg

# A known-good substrate / analysis, reused so the fixtures stay small.
_REFERENCE = LEVEL_1_CANONICAL_CIRCUITS[0]


def _spec(nodes, elements, ports, *, ground_nodes=(), open_terminations=(),
          label="topology_test"):
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=_REFERENCE.analysis,
        substrate=_REFERENCE.substrate,
        nodes=tuple(Node(id=n) for n in nodes),
        elements=tuple(elements),
        ports=tuple(ports),
        label=label,
        ground_nodes=tuple(ground_nodes),
        open_terminations=tuple(open_terminations),
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


def _by_class(root, css):
    """Every parsed element carrying ``class="<css>"`` (namespace-agnostic)."""

    return [el for el in root.iter() if el.get("class") == css]


def test_minimal_two_port_renders_one_box_and_a_wire(tmp_path: Path):
    """TermG -> R -> TermG renders one component box and at least one wire."""

    spec = _spec(
        nodes=("n_in", "n_out"),
        elements=(
            _termg("P1", "n_in", 1),
            _two_terminal("R1", "R", "n_in", "n_out", {"r_ohm": 50.0}),
            _termg("P2", "n_out", 2),
        ),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        label="minimal_two_port",
    )
    out = tmp_path / "minimal.svg"
    render_topology_svg(spec, out)

    root = ET.parse(out).getroot()  # raises on invalid XML
    components = _by_class(root, "component")
    wires = _by_class(root, "wire")
    assert [c.get("data-name") for c in components] == ["R1"]
    assert len(wires) >= 1


def test_shunt_branch_drawn_below_the_through_path(tmp_path: Path):
    """A shunt component box sits below the through-path component boxes."""

    spec = _spec(
        nodes=("n_in", "n_mid", "n_out", "n_gnd"),
        elements=(
            _termg("P1", "n_in", 1),
            _two_terminal("LINE_A", "MLIN", "n_in", "n_mid",
                          {"w_m": 2.95e-3, "l_m": 3.0e-3,
                           "substrate_ref": _REFERENCE.substrate.name}),
            _two_terminal("LINE_B", "MLIN", "n_mid", "n_out",
                          {"w_m": 2.95e-3, "l_m": 3.0e-3,
                           "substrate_ref": _REFERENCE.substrate.name}),
            _two_terminal("R_SHUNT", "R", "n_mid", "n_gnd", {"r_ohm": 25.0}),
            _termg("P2", "n_out", 2),
        ),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        ground_nodes=("n_gnd",),
        label="shunt_stub",
    )
    out = tmp_path / "shunt.svg"
    render_topology_svg(spec, out)

    root = ET.parse(out).getroot()
    by_role: dict[str, list] = {}
    for comp in _by_class(root, "component"):
        by_role.setdefault(comp.get("data-role"), []).append(comp)
    assert "through" in by_role and "shunt" in by_role, sorted(by_role)

    through_y = max(float(c.get("data-y")) for c in by_role["through"])
    shunt_y = min(float(c.get("data-y")) for c in by_role["shunt"])
    assert shunt_y > through_y, (shunt_y, through_y)
    assert {c.get("data-name") for c in by_role["shunt"]} == {"R_SHUNT"}


def test_open_termination_symbol_at_declared_net(tmp_path: Path):
    """The open-stub canonical circuit draws an open symbol at its open net."""

    spec = next(
        s for s in LEVEL_3_CANONICAL_CIRCUITS if s.label == "tee_open_zero_stub"
    )
    assert spec.open_terminations == ("n_branch_end",)
    out = tmp_path / "open.svg"
    render_topology_svg(spec, out)

    root = ET.parse(out).getroot()
    opens = _by_class(root, "open-termination")
    assert len(opens) == 1
    assert opens[0].get("data-net") == "n_branch_end"
    # the stub end is open, not grounded -- no ground symbol
    assert _by_class(root, "ground") == []


def test_unknown_entity_type_does_not_crash(tmp_path: Path, caplog):
    """An unrecognised element type renders without crashing and warns."""

    spec = _spec(
        nodes=("n_in", "n_out"),
        elements=(
            _termg("P1", "n_in", 1),
            _two_terminal("X1", "FUTURE_THING", "n_in", "n_out", {}),
            _termg("P2", "n_out", 2),
        ),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        label="unknown_entity",
    )
    out = tmp_path / "unknown.svg"
    with caplog.at_level(logging.WARNING, logger="tvtl_ads_builder_v2.topology"):
        render_topology_svg(spec, out)  # must not raise

    assert out.is_file()
    ET.parse(out)  # still produced valid XML
    assert any(
        "unrecognised element type" in record.getMessage()
        for record in caplog.records
    ), [r.getMessage() for r in caplog.records]


