"""Tests for the DeviceSpec ``open_terminations`` field and the
adjacency audit's handling of declared open-circuit terminations.

All tests are pure Python and run without an ADS installation.
"""

from __future__ import annotations

from tvtl_ads_builder_v2.audit import audit_circuit
from tvtl_ads_builder_v2.device_spec import (
    DeviceSpec,
    Element,
    ElementPin,
    Node,
    Port,
)
from tvtl_ads_builder_v2.device_spec.schema import DEVICE_SPEC_VERSION
from tvtl_ads_builder_v2.netlist.emit import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    LEVEL_3_CANONICAL_CIRCUITS,
)


def _minimal_spec(*, open_terminations=()):
    """A minimal valid DeviceSpec whose net ``n_open`` carries one pin.

    Topology: one ``TermG`` port element at ``n_a`` and one ``MLIN``
    from ``n_a`` to ``n_open``. ``n_open`` therefore has exactly one
    element pin -- a one-pin net the adjacency audit can flag (or skip,
    when it is declared in ``open_terminations``). Substrate and
    analysis are borrowed from a canonical circuit to keep the fixture
    small; it is a deliberately minimal audit fixture, not a runnable
    two-port.
    """

    reference = LEVEL_1_CANONICAL_CIRCUITS[0]
    port = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_a"),),
        parameters={"z_ohm": 50.0, "num": 1},
    )
    line = Element(
        id="LINE",
        type="MLIN",
        pins=(ElementPin(pin=1, node="n_a"), ElementPin(pin=2, node="n_open")),
        parameters={
            "w_m": 2.95e-3,
            "l_m": 1.0e-3,
            "substrate_ref": reference.substrate.name,
        },
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=reference.analysis,
        substrate=reference.substrate,
        nodes=(Node(id="n_a"), Node(id="n_open")),
        elements=(port, line),
        ports=(Port(num=1, node="n_a"),),
        label="open_terminations_unit_test",
        open_terminations=tuple(open_terminations),
    )


def _audit_flags(spec):
    """Run the adjacency audit over one DeviceSpec; return its flag list.

    The per-circuit audit logic now lives in
    ``tvtl_ads_builder_v2.audit.audit_circuit``.
    """

    netlist_text = emit_hpeesofsim_netlist(spec)
    return list(audit_circuit(spec, netlist_text).flags)


def test_open_terminations_round_trips():
    """A declared open termination survives a to_dict/from_dict round trip."""

    spec = _minimal_spec(open_terminations=("n_open",))
    assert spec.open_terminations == ("n_open",)

    payload = spec.to_dict()
    assert payload["open_terminations"] == ["n_open"]

    restored = DeviceSpec.from_dict(payload)
    assert restored.open_terminations == ("n_open",)


def test_open_terminations_defaults_to_empty():
    """Without the field a spec defaults to an empty open_terminations."""

    spec = _minimal_spec()
    assert spec.open_terminations == ()
    # The field is serialized only when non-empty, matching ground_nodes.
    assert "open_terminations" not in spec.to_dict()
    assert DeviceSpec.from_dict(spec.to_dict()).open_terminations == ()


def test_audit_flags_undeclared_one_pin_net():
    """An undeclared one-pin net is flagged as dangling by the audit."""

    flags = _audit_flags(_minimal_spec(open_terminations=()))
    assert any("DANGLING NET" in f and "n_open" in f for f in flags), flags


def test_audit_skips_declared_open_termination():
    """A one-pin net declared as an open termination is not flagged."""

    flags = _audit_flags(_minimal_spec(open_terminations=("n_open",)))
    assert flags == [], flags


def _sdd2p_port2_selftie_spec():
    """A minimal valid spec whose SDD2P ties port 2 to one shared net.

    ``P1`` at ``n_a``; an ``SDD2P`` whose port 1 spans ``n_a`` / ``n_gnd``
    and whose port-2 terminals (pins 3 and 4) are BOTH tied to a single
    shared ``n_return`` net -- the shared-return wiring fix. ``n_return``
    therefore carries two pins, both of one element instance: an
    intentional self-tie the adjacency audit must not flag as an
    internal short.
    """

    reference = LEVEL_1_CANONICAL_CIRCUITS[0]
    port = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_a"),),
        parameters={"z_ohm": 50.0, "num": 1},
    )
    varactor = Element(
        id="VARACTOR",
        type="SDD2P",
        pins=(
            ElementPin(pin=1, node="n_a"),
            ElementPin(pin=2, node="n_gnd"),
            ElementPin(pin=3, node="n_return"),
            ElementPin(pin=4, node="n_return"),
        ),
        parameters={
            "q_poly_coefficients": (0.0, 7.0e-12, -1.0e-12, 0.0, 0.0, 0.0, 0.0),
        },
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=reference.analysis,
        substrate=reference.substrate,
        nodes=(Node(id="n_a"), Node(id="n_gnd"), Node(id="n_return")),
        elements=(port, varactor),
        ports=(Port(num=1, node="n_a"),),
        label="sdd2p_port2_selftie_audit",
        ground_nodes=("n_gnd",),
    )


def test_audit_accepts_sdd2p_port2_self_tie():
    """An SDD2P port-2 self-tie audits CLEAN.

    ``n_return`` carries two pins, both belonging to the one SDD2P
    instance -- a net all of whose pins are one element's is an
    intentional self-tie, not an internal short, so the audit must
    raise no flag.
    """

    flags = _audit_flags(_sdd2p_port2_selftie_spec())
    assert flags == [], flags


def test_audit_still_flags_a_short_shared_with_another_component():
    """The self-tie exemption is scoped: a two-pin-on-one-net short on a
    net that also carries another component's pin is still flagged."""

    # Hand-crafted netlist: RBUG bridges both its pins onto n_x, which
    # also carries ROTHER's pin -- not a pure self-tie.
    netlist = (
        "; guard fixture\n"
        "R:RBUG n_x n_x R=10 Ohm\n"
        "R:ROTHER n_x n_y R=20 Ohm\n"
    )
    report = audit_circuit(
        _minimal_spec(open_terminations=("n_y",)),
        netlist,
        circuit_name="internal_short_guard",
    )
    assert any(
        "INTERNAL SHORT" in f and "RBUG" in f for f in report.flags
    ), report.flags


def test_open_stub_canonical_circuit_declares_its_open_end():
    """The open-stub canonical circuit declares its open end; the three
    grounded tee circuits declare none."""

    by_label = {s.label: s for s in LEVEL_3_CANONICAL_CIRCUITS}
    assert by_label["tee_open_zero_stub"].open_terminations == ("n_branch_end",)
    for label in (
        "tee_short_zero_stub",
        "tee_short_paper_dims",
        "tee_qw_short_3p1GHz",
    ):
        assert by_label[label].open_terminations == ()


