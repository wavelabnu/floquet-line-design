"""Tests for the I_Probe platform support.

ADS-free. The I_Probe element type is exercised through the device-spec
validator, the hpeesofsim netlist emitter, the netlist cross-check's
``Short``/``Mode`` classification, and the OA emitter against the stub
``keysight.ads.de``.

Empirical basis (the ADS behaviour these tests encode): I_Probe netlists
as ``Short ... Mode=0 SaveCurrent=yes``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test_oa_canonical import _install_fake_keysight

from tvtl_ads_builder_v2.device_spec import (
    Analysis,
    DeviceSpec,
    DeviceSpecError,
    Element,
    ElementPin,
    FrequencyGrid,
    Node,
    Port,
    Substrate,
    validate_device_spec,
)
from tvtl_ads_builder_v2.device_spec.schema import DEVICE_SPEC_VERSION
from tvtl_ads_builder_v2.netlist.elements.i_probe import emit_i_probe
from tvtl_ads_builder_v2.netlist.emit import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa import oa_emitter
from tvtl_ads_builder_v2.oa.netlist_compare import (
    compare_circuits,
    parse_netlist_text,
)

_SUBSTRATE = Substrate(
    name="msub1",
    model="hammerstad_kirschning",
    parameters={
        "eps_r": 3.4, "h_m": 1.5e-3, "tan_delta": 0.003,
        "sigma_S_per_m": 58.0e6, "t_m": 17.5e-6,
    },
)


def _i_probe(element_id: str, node1: str, node2: str) -> Element:
    """A two-pin I_Probe element (the spec-level element is parameterless)."""

    return Element(
        id=element_id,
        type="I_Probe",
        pins=(ElementPin(pin=1, node=node1), ElementPin(pin=2, node=node2)),
        parameters={},
    )


def _spec_with_probe(*, probe: Element, ground: tuple[str, ...] = ()) -> DeviceSpec:
    """A minimal 2-port: P1 -- I_Probe -- MLIN -- P2."""

    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=Analysis(
            type="s_parameters",
            frequency_grid_Hz=FrequencyGrid(start=1.0e9, stop=10.0e9, points=11),
            port_impedance_ohm=50.0,
        ),
        substrate=_SUBSTRATE,
        nodes=(Node(id="n_in"), Node(id="n_mid"), Node(id="n_out")),
        elements=(
            Element(
                id="P1", type="TermG",
                pins=(ElementPin(pin=1, node="n_in"),),
                parameters={"z_ohm": 50.0, "num": 1},
            ),
            probe,
            Element(
                id="LINE", type="MLIN",
                pins=(ElementPin(pin=1, node="n_mid"),
                      ElementPin(pin=2, node="n_out")),
                parameters={"w_m": 4.0e-3, "l_m": 3.0e-3, "substrate_ref": "msub1"},
            ),
            Element(
                id="P2", type="TermG",
                pins=(ElementPin(pin=1, node="n_out"),),
                parameters={"z_ohm": 50.0, "num": 2},
            ),
        ),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        label="i_probe_test",
        ground_nodes=ground,
    )


# --------------------------------------------------------------------------
# hpeesofsim emitter
# --------------------------------------------------------------------------


def test_emit_i_probe_line() -> None:
    """emit_i_probe produces the Short token with Mode=0 SaveCurrent=yes."""

    line = emit_i_probe(_i_probe("IPR1", "n_a", "n_b"))
    assert line == "Short:IPR1 n_a n_b Mode=0 SaveCurrent=yes"


def test_emit_i_probe_resolves_ground_node() -> None:
    """A pin on a ground node is emitted as the literal 0."""

    line = emit_i_probe(
        _i_probe("IPR1", "n_a", "n_gnd"),
        resolve_node=lambda n: "0" if n == "n_gnd" else n,
    )
    assert line == "Short:IPR1 n_a 0 Mode=0 SaveCurrent=yes"


def test_emit_i_probe_rejects_wrong_type() -> None:
    bad = Element(
        id="X", type="R",
        pins=(ElementPin(pin=1, node="a"), ElementPin(pin=2, node="b")),
        parameters={"r_ohm": 1.0},
    )
    with pytest.raises(ValueError, match="emit_i_probe"):
        emit_i_probe(bad)


def test_emit_i_probe_rejects_wrong_pin_count() -> None:
    bad = Element(
        id="X", type="I_Probe",
        pins=(ElementPin(pin=1, node="a"),),
        parameters={},
    )
    with pytest.raises(ValueError, match="exactly two pins"):
        emit_i_probe(bad)


def test_hpeesofsim_deck_carries_the_probe() -> None:
    """A spec with an I_Probe emits the Short line into the full deck."""

    deck = emit_hpeesofsim_netlist(_spec_with_probe(probe=_i_probe(
        "IPR1", "n_in", "n_mid"
    )))
    assert "Short:IPR1 n_in n_mid Mode=0 SaveCurrent=yes" in deck


# --------------------------------------------------------------------------
# validator
# --------------------------------------------------------------------------


def test_validator_accepts_two_pin_i_probe() -> None:
    validate_device_spec(_spec_with_probe(probe=_i_probe("IPR1", "n_in", "n_mid")))


def test_validator_rejects_three_pin_i_probe() -> None:
    bad = Element(
        id="IPR1", type="I_Probe",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_mid"),
            ElementPin(pin=3, node="n_out"),
        ),
        parameters={},
    )
    with pytest.raises(DeviceSpecError, match="exactly two pins"):
        validate_device_spec(_spec_with_probe(probe=bad))


# --------------------------------------------------------------------------
# netlist cross-check -- Short / Mode classification
# --------------------------------------------------------------------------


def test_short_mode0_classifies_as_i_probe_element() -> None:
    """A Short Mode=0 line is a counted I_PROBE element in the cross-check."""

    deck = emit_hpeesofsim_netlist(_spec_with_probe(probe=_i_probe(
        "IPR1", "n_in", "n_mid"
    )))
    circuit = parse_netlist_text(deck)
    types = [c.ctype for c in circuit.elements]
    assert "I_PROBE" in types
    assert sum(1 for t in types if t == "I_PROBE") == 1


def test_short_mode1_is_not_an_i_probe() -> None:
    """A Short Mode=1 (DC block) is NOT classified as an I_Probe element."""

    deck = emit_hpeesofsim_netlist(_spec_with_probe(probe=_i_probe(
        "IPR1", "n_in", "n_mid"
    )))
    dc_block = deck.replace(
        "Short:IPR1 n_in n_mid Mode=0 SaveCurrent=yes",
        "Short:DCB1 n_in n_mid Mode=1 C=100 pF",
    )
    circuit = parse_netlist_text(dc_block)
    assert "I_PROBE" not in [c.ctype for c in circuit.elements]
    assert any("Short:DCB1" in line for line in circuit.ignored)


def test_probed_deck_self_compare_matches() -> None:
    """A probed deck cross-checks MATCH against itself (I_Probe paired)."""

    deck = emit_hpeesofsim_netlist(_spec_with_probe(probe=_i_probe(
        "IPR1", "n_in", "n_mid"
    )))
    result = compare_circuits(parse_netlist_text(deck), parse_netlist_text(deck))
    assert result.match, result.differences


def test_probe_count_difference_is_a_mismatch() -> None:
    """A deck with an I_Probe vs one without is an element-count mismatch."""

    probed = emit_hpeesofsim_netlist(_spec_with_probe(probe=_i_probe(
        "IPR1", "n_in", "n_mid"
    )))
    # Drop the Short line -> one fewer I_PROBE element.
    unprobed = "\n".join(
        line for line in probed.splitlines() if not line.startswith("Short:")
    )
    result = compare_circuits(
        parse_netlist_text(probed), parse_netlist_text(unprobed)
    )
    assert not result.match
    assert any("count" in d for d in result.differences)


# --------------------------------------------------------------------------
# OA emitter (stub keysight)
# --------------------------------------------------------------------------


def test_oa_emitter_places_i_probe(tmp_path: Path, monkeypatch) -> None:
    """The OA emitter builds a probed spec's workspace without raising.

    Exercised against the stub keysight.ads.de; the emitter must place
    the I_Probe instance and wire its two pins. The hpeesofsim deck
    written alongside carries the Short line.
    """

    _install_fake_keysight(monkeypatch)
    spec = _spec_with_probe(probe=_i_probe("IPR1", "n_in", "n_mid"))
    info = oa_emitter.emit_workspace_from_spec(
        spec, tmp_path / spec.label / "oa_workspace"
    )
    hpeesofsim = Path(info["hpeesofsim_netlist_path"]).read_text(encoding="utf-8")
    assert "Short:IPR1 n_in n_mid Mode=0 SaveCurrent=yes" in hpeesofsim


