"""Lock the cell-agnostic CellSpec interface.

The cascade assembler (`optimized_self_pumped_spec`) now walks a `CellSpec`.
These tests pin the single-shunt-varactor cell's contract so a future enriched
PT-paper cell can't silently break the interface, and so the assembler keeps
producing the byte-identical A.3.1 anchor (the recovery gate).
"""

from __future__ import annotations

from tvtl.optimization.topology_generator import Termination, TopologyFamily, ViaConfig
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.cells import CellSpec, SingleShuntVaractorCell

VAR_A = RealisticVaractor(CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0,
                          L_pkg_h=0.3e-9, R_series_ohm=0.5)
# A.3.1 topology: n2_single_short.
TOPO = TopologyFamily(2, ViaConfig.SINGLE, Termination.SHORT)
PARAMS = (0.5111436888575555e-3, 2.2083339346572757e-3,
          0.9451326757669449e-3, 6.609422719571739e-3)


def _cell(varactor=VAR_A, topo=TOPO):
    return SingleShuntVaractorCell(
        topology=topo, params=PARAMS, varactor=varactor,
        substrate_name="ROGERS_4003C", main_width_m=4.575430198648918e-3,
        main_period_m=6.11e-3, q_poly=varactor.charge_polynomial().charge_polynomial_coefficients,
        q_poly_reference="test", l_via_H=1.0e-9,
    )


def test_single_shunt_cell_is_a_cellspec():
    assert isinstance(_cell(), CellSpec)


def test_output_terminal_and_extraction_nodes():
    cell = _cell()
    assert cell.output_terminal_node(3) == "n_c3_out"
    # SHORT termination -> end node is the via top; SINGLE short -> via grounded.
    # Extraction pair: (post-divider SDD in-line pin, SDD stub-side pin).
    # VAR_A has R_series>0 -> top node is the post-R node n_c{k}_rp.
    assert cell.varactor_terminal_nodes(2) == ("n_c2_rp", "n_c2_v")
    assert cell.ground_nodes(2) == ["n_c2_g"]
    assert cell.name_nodes(2) == ["n_c2_b", "n_c2_v"]


def test_extraction_top_node_tracks_package_parasitics():
    # No R_series, with L_pkg -> top node is the post-L node n_c{k}_lp.
    v_noR = RealisticVaractor(CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, L_pkg_h=0.3e-9, R_series_ohm=0.0)
    assert _cell(varactor=v_noR).varactor_terminal_nodes(1) == ("n_c1_lp", "n_c1_v")
    # No package at all -> top node is the branch-top n_c{k}_b.
    v_bare = RealisticVaractor(CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, L_pkg_h=0.0, R_series_ohm=0.0)
    assert _cell(varactor=v_bare).varactor_terminal_nodes(1) == ("n_c1_b", "n_c1_v")


def test_cell_elements_have_expected_ids_and_charge_sdd():
    cell = _cell()
    els = cell.cell_elements(1, input_net="n_pin", output_net="n_c1_out")
    ids = {e.id for e in els}
    # MLIN halves + MTEE + package L/R + charge SDD + I_Probe + 2 stub segs + via.
    for expect in ("LINE_LEFT_C1", "TEE_C1", "LINE_RIGHT_C1", "LPKG_C1", "RPKG_C1",
                   "VARACTOR_C1", "IPR_VAR_C1", "STUB_C1_0", "STUB_C1_1", "LVIA_C1"):
        assert expect in ids, f"missing {expect}"
    sdd = next(e for e in els if e.id == "VARACTOR_C1")
    assert sdd.type == "SDD2P"
    assert "q_poly_coefficients" in sdd.parameters
    # pins 3/4 self-tied to the per-cell return net.
    assert sdd.pins[2].node == sdd.pins[3].node == "n_c1_ret"


def test_open_termination_has_no_via():
    open_cell = _cell(topo=TopologyFamily(1, ViaConfig.NONE, Termination.OPEN))
    ids = {e.id for e in open_cell.cell_elements(1, input_net="n_pin", output_net="n_c1_out")}
    assert "LVIA_C1" not in ids


