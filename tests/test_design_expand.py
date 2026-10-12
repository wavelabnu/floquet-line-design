"""Tests for the design expander :func:`to_device_spec`.

ADS-free: expansion is pure. These tests verify the expander is
deterministic, names per-cell elements and nets consistently, wires the
cascade correctly cell-to-cell, expands the VAR block, and produces a
device spec that the existing device-spec validator and hpeesofsim
emitter accept.
"""

from __future__ import annotations

import pytest

from tvtl_ads_builder_v2.design import (
    AnalysisSpec,
    BranchSpec,
    CellSpec,
    DesignError,
    ElementSpec,
    HBTwoToneSpec,
    MSUBSpec,
    MicrostripSpec,
    SParamSpec,
    SourceSpec,
    TVTLDesign,
    to_device_spec,
)
from tvtl_ads_builder_v2.device_spec import resolve_spec, validate_device_spec
from tvtl_ads_builder_v2.device_spec.io import dump_device_spec, load_device_spec
from tvtl_ads_builder_v2.netlist.emit import emit_hpeesofsim_netlist


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _rlc_sdd_branch() -> BranchSpec:
    return BranchSpec(
        elements=(
            ElementSpec(type="R", params={"r_ohm": "R"}),
            ElementSpec(type="L", params={"l_h": "L"}),
            ElementSpec(type="C", params={"c_f": "C"}),
            ElementSpec(
                type="SDD2P_varactor",
                params={"q_poly_reference": "smv1245_default_polynomial"},
            ),
        ),
        termination="ground",
    )


def _design(n_cells: int, *, branches=None, sources=None, hb=True) -> TVTLDesign:
    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(_rlc_sdd_branch(),) if branches is None else branches,
    )
    analysis = AnalysisSpec(
        s_param=SParamSpec(
            f_start_Hz=1.0e9, f_stop_Hz=10.0e9, n_points=91,
            port_impedance_ohm=50.0,
        ),
        hb_two_tone=(
            HBTwoToneSpec(
                fundamental_freq_1="f_signal", fundamental_order_1=5,
                fundamental_freq_2="f_pump", fundamental_order_2=5,
                max_order=6,
            )
            if hb
            else None
        ),
    )
    return TVTLDesign(
        n_cells=n_cells,
        cell=cell,
        parameters={
            "f_signal": 3.0e9, "f_pump": 6.0e9,
            "signal_power_dBm": -30.0, "pump_power_dBm": 5.0,
            "R": 2.0, "L": 2.0e-9, "C": 4.0e-12,
        },
        sources=(
            (
                SourceSpec(
                    name="SIGNAL_SRC", freq="f_signal",
                    power_dBm="signal_power_dBm",
                ),
                SourceSpec(
                    name="PUMP_SRC", freq="f_pump",
                    power_dBm="pump_power_dBm",
                ),
            )
            if sources is None
            else sources
        ),
        analysis=analysis,
        substrate=MSUBSpec(
            eps_r=3.4, h_m=1.5e-3, tan_delta=0.003, sigma_S_per_m=58.0e6
        ),
    )


def _by_id(spec, element_id):
    return next(e for e in spec.elements if e.id == element_id)


def _net_of(element, pin_number):
    return next(p.node for p in element.pins if p.pin == pin_number)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_expansion_is_deterministic() -> None:
    """The same design always expands to an equal device spec."""

    assert to_device_spec(_design(3)) == to_device_spec(_design(3))


def test_expansion_deterministic_across_cell_counts() -> None:
    """A 3-cell and a 5-cell design differ only by the extra cells."""

    assert to_device_spec(_design(3)) != to_device_spec(_design(5))


def test_expanded_spec_survives_json_round_trip(tmp_path) -> None:
    """The expanded device spec dumps and reloads unchanged."""

    spec = to_device_spec(_design(3))
    path = dump_device_spec(spec, tmp_path / "level8.device_spec.json")
    assert load_device_spec(path) == spec


# --------------------------------------------------------------------------
# Element count and per-cell naming
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_cells", [1, 2, 3, 5, 8])
def test_element_count_scales_with_cell_count(n_cells: int) -> None:
    """Each cell contributes MLIN + MTEE + 4 branch elements; +1 output port.

    The R - L - C - SDD2P branch is four elements, so a cell is six
    elements; the cascade adds one TermG output port.
    """

    spec = to_device_spec(_design(n_cells))
    assert len(spec.elements) == 6 * n_cells + 1


def test_per_cell_element_ids_follow_the_naming_convention() -> None:
    """Cell 0's elements carry the documented _c{k} / _b{b}_{i} affixes."""

    spec = to_device_spec(_design(3))
    ids = {e.id for e in spec.elements}
    for k in range(3):
        assert f"MLIN_c{k}" in ids
        assert f"MTEE_c{k}_0" in ids
        assert f"R_c{k}_b0_0" in ids
        assert f"L_c{k}_b0_1" in ids
        assert f"C_c{k}_b0_2" in ids
        assert f"SDD_c{k}_b0_3" in ids
    assert "P_OUT" in ids


def test_element_ids_are_unique() -> None:
    """Every generated element id is unique across all cells."""

    spec = to_device_spec(_design(5))
    ids = [e.id for e in spec.elements]
    assert len(ids) == len(set(ids))


def test_element_types_map_to_device_spec_types() -> None:
    """SDD2P_varactor expands to the device-spec SDD2P element type."""

    spec = to_device_spec(_design(2))
    sdd = _by_id(spec, "SDD_c0_b0_3")
    assert sdd.type == "SDD2P"
    assert len(sdd.pins) == 4


# --------------------------------------------------------------------------
# Multi-cell wiring
# --------------------------------------------------------------------------


def test_cascade_through_path_is_wired_cell_to_cell() -> None:
    """Cell k's output net is cell k+1's input net; ends are input/output."""

    spec = to_device_spec(_design(4))
    # Cell 0's through-line starts at the design input net.
    assert _net_of(_by_id(spec, "MLIN_c0"), 1) == "n_input"
    for k in range(4):
        mtee_out = _net_of(_by_id(spec, f"MTEE_c{k}_0"), 2)
        if k < 3:
            # MTEE pin 2 (cell output) feeds the next cell's MLIN pin 1.
            assert mtee_out == _net_of(_by_id(spec, f"MLIN_c{k + 1}"), 1)
        else:
            # The last cell's output is the design output net.
            assert mtee_out == "n_output"
    assert spec.ports == ((spec.ports[0]),)
    assert spec.ports[0].node == "n_output"


def test_branch_taps_the_cell_mtee() -> None:
    """Each cell's branch chain starts at its MTEE branch pin (pin 3)."""

    spec = to_device_spec(_design(3))
    for k in range(3):
        tap = _net_of(_by_id(spec, f"MTEE_c{k}_0"), 3)
        assert tap == f"n_cell{k}_br0_top"
        # The branch's first element (R) sits on that tap net.
        assert _net_of(_by_id(spec, f"R_c{k}_b0_0"), 1) == tap


def test_branch_chain_is_series_connected() -> None:
    """R -> L -> C -> SDD2P chain through fresh per-step internal nodes."""

    spec = to_device_spec(_design(2))
    r = _by_id(spec, "R_c0_b0_0")
    l = _by_id(spec, "L_c0_b0_1")
    c = _by_id(spec, "C_c0_b0_2")
    sdd = _by_id(spec, "SDD_c0_b0_3")
    assert _net_of(r, 2) == _net_of(l, 1)
    assert _net_of(l, 2) == _net_of(c, 1)
    assert _net_of(c, 2) == _net_of(sdd, 1)


def test_sdd2p_port2_self_tie_and_ground() -> None:
    """The SDD2P's pins 3/4 share one return net; pin 2 is the branch ground."""

    spec = to_device_spec(_design(3))
    for k in range(3):
        sdd = _by_id(spec, f"SDD_c{k}_b0_3")
        assert _net_of(sdd, 3) == _net_of(sdd, 4) == f"n_cell{k}_br0_ret"
        assert _net_of(sdd, 2) == f"n_cell{k}_br0_gnd"


def test_each_branch_grounds_to_its_own_net() -> None:
    """Every grounded branch has a distinct ground net (all emit as 0)."""

    spec = to_device_spec(_design(5))
    assert spec.ground_nodes == tuple(
        sorted(f"n_cell{k}_br0_gnd" for k in range(5))
    )


# --------------------------------------------------------------------------
# Spec-level singletons (sources, VAR block, HB, substrate, ports)
# --------------------------------------------------------------------------


def test_var_block_carries_every_design_parameter() -> None:
    """The expanded VAR block declares exactly the design's parameters."""

    design = _design(3)
    spec = to_device_spec(design)
    assert spec.var_block is not None
    declared = {v.name: v.value for v in spec.var_block.variables}
    assert declared == dict(design.parameters)


def test_sources_drive_the_design_input_net() -> None:
    """Both P_1Tone sources drive n_input; named refs are carried verbatim."""

    spec = to_device_spec(_design(3))
    assert {s.name for s in spec.sources} == {"SIGNAL_SRC", "PUMP_SRC"}
    for source in spec.sources:
        assert source.port_node == "n_input"
    signal = next(s for s in spec.sources if s.name == "SIGNAL_SRC")
    assert signal.freq == "f_signal"  # unresolved named reference


def test_hb_controller_passes_through_to_the_spec() -> None:
    """The two-tone HB controller reaches the device spec unchanged."""

    spec = to_device_spec(_design(3))
    assert spec.hb_two_tone is not None
    assert spec.hb_two_tone.fundamental_freq_1 == "f_signal"
    assert spec.hb_single_tone is None


def test_substrate_maps_to_device_spec_substrate() -> None:
    """The MSUBSpec becomes a device-spec Substrate with the required keys."""

    spec = to_device_spec(_design(2))
    assert spec.substrate.name == "msub1"
    assert spec.substrate.model == "hammerstad_kirschning"
    for key in ("eps_r", "h_m", "tan_delta", "sigma_S_per_m", "t_m"):
        assert key in spec.substrate.parameters
    assert spec.substrate.parameters["eps_r"] == 3.4


# --------------------------------------------------------------------------
# Topology variants
# --------------------------------------------------------------------------


def test_pure_transmission_line_cell_has_no_branch_elements() -> None:
    """A cell with no branches expands to a single through-line MLIN."""

    spec = to_device_spec(_design(3, branches=()))
    assert [e.type for e in spec.elements] == ["MLIN", "MLIN", "MLIN", "TermG"]
    # The MLINs chain straight through n_input -> ... -> n_output.
    assert _net_of(_by_id(spec, "MLIN_c0"), 1) == "n_input"
    assert _net_of(_by_id(spec, "MLIN_c2"), 2) == "n_output"


def test_floating_branch_declares_an_open_termination() -> None:
    """A 'floating' branch terminus is recorded in open_terminations."""

    branch = BranchSpec(
        elements=(ElementSpec(type="C", params={"c_f": "C"}),),
        termination="floating",
    )
    spec = to_device_spec(_design(2, branches=(branch,)))
    assert spec.open_terminations == ("n_cell0_br0_open", "n_cell1_br0_open")
    assert spec.ground_nodes == ()


# --------------------------------------------------------------------------
# Downstream-pipeline compatibility
# --------------------------------------------------------------------------


def test_expanded_spec_passes_the_device_spec_validator() -> None:
    """The expanded (resolved) spec is structurally valid."""

    for n in (1, 3, 5):
        validate_device_spec(resolve_spec(to_device_spec(_design(n))))


def test_expanded_spec_emits_a_hpeesofsim_deck() -> None:
    """The expanded spec produces a hpeesofsim flat netlist for every cell."""

    deck = emit_hpeesofsim_netlist(to_device_spec(_design(3)))
    for k in range(3):
        assert f"MLIN:MLIN_c{k} " in deck
        assert f"MTEE_ADS:MTEE_c{k}_0 " in deck
        assert f"SDD:SDD_c{k}_b0_3 " in deck
    # Named references resolved to literals; the SMV1245 polynomial.
    assert "R=2 Ohm" in deck
    assert "I[1,1]=(7e-12)*_v1+(-1e-12)*_v1**2" in deck
    assert "P_1Tone:SIGNAL_SRC n_input 0" in deck


def test_unknown_polynomial_reference_is_rejected() -> None:
    """An SDD2P_varactor naming an unknown charge polynomial fails expansion."""

    branch = BranchSpec(
        elements=(
            ElementSpec(
                type="SDD2P_varactor",
                params={"q_poly_reference": "no_such_polynomial"},
            ),
        ),
    )
    with pytest.raises(DesignError, match="no_such_polynomial"):
        to_device_spec(_design(1, branches=(branch,)))


