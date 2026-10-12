"""Optimizer integration test (base Python, no ADS HB).

Confirms the end-to-end *spec-generation* path is clean: one Stage-1 stub ->
Stage 2 (Bloch match) -> Stage 3 (matching) -> device-spec generation ->
validator + flat-HB netlist emission, across matching combinations. The full
ADS HB run is Stage 4 (not exercised here).
"""

from __future__ import annotations

import pytest

from tvtl.optimization.stage1_per_cell import coarse_screen, refine_topk
from tvtl.optimization.stage2_main_tl import match_z_tl
from tvtl.optimization.stage3_matching import edge_impedance, select_matching
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import linear_families
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.device_spec.validator import validate_device_spec
from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
from tvtl_ads_builder_v2.parity.optimized_design import optimized_control_port_spec

_VAR = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
)


@pytest.fixture(scope="module")
def staged():
    res = coarse_screen(
        3.1e9, _VAR, substrate=ROGERS_4003C, families=linear_families(),
        n_samples=96, seed=0, top_k=5,
    )
    top = refine_topk(res.linear_topk, 3.1e9, _VAR, substrate=ROGERS_4003C, top_n=3, maxiter=60)
    cand = top[0]
    mtl = match_z_tl(cand, _VAR, substrate=ROGERS_4003C)
    z_in, z_out = edge_impedance(mtl, _VAR, 15, substrate=ROGERS_4003C)
    sm_in = select_matching(z_in, "input", 3.1e9, substrate=ROGERS_4003C)
    sm_out = select_matching(z_out, "output", 3.1e9, substrate=ROGERS_4003C)
    return cand, mtl, sm_in, sm_out


def test_stage2_to_spec_no_matching(staged):
    cand, mtl, _sm_in, _sm_out = staged
    spec = optimized_control_port_spec(
        topology=cand.topology, params=cand.params_array, varactor=_VAR,
        substrate=ROGERS_4003C, n_cells=15, main_width_m=mtl.w_main_m,
    )
    validate_device_spec(spec)
    netlist = emit_flat_hb_netlist(spec)
    assert "hb" in netlist.lower() or len(netlist.splitlines()) > 50


@pytest.mark.parametrize("which", ["qw", "single_stub"])
def test_full_stack_with_matching_validator_clean(staged, which):
    cand, mtl, sm_in, sm_out = staged
    im = sm_in.quarter_wave if which == "qw" else sm_in.single_stub
    om = sm_out.quarter_wave if which == "qw" else sm_out.single_stub
    spec = optimized_control_port_spec(
        topology=cand.topology, params=cand.params_array, varactor=_VAR,
        substrate=ROGERS_4003C, n_cells=15, main_width_m=mtl.w_main_m,
        input_match=im, output_match=om,
    )
    validate_device_spec(spec)  # raises if not clean
    netlist = emit_flat_hb_netlist(spec)
    assert len(netlist) > 0
    n_match = sum(1 for e in spec.elements if e.id.startswith("MATCH_"))
    assert n_match >= 1  # matching elements emitted


def test_mixed_matching_combination(staged):
    cand, mtl, sm_in, sm_out = staged
    spec = optimized_control_port_spec(
        topology=cand.topology, params=cand.params_array, varactor=_VAR,
        substrate=ROGERS_4003C, n_cells=8, main_width_m=mtl.w_main_m,
        input_match=sm_in.quarter_wave, output_match=sm_out.single_stub,
    )
    validate_device_spec(spec)
    assert emit_flat_hb_netlist(spec)


def test_main_width_threaded_into_spec(staged):
    """Stage-2 main width reaches the emitted main-line MLINs."""
    cand, mtl, _sm_in, _sm_out = staged
    spec = optimized_control_port_spec(
        topology=cand.topology, params=cand.params_array, varactor=_VAR,
        substrate=ROGERS_4003C, n_cells=4, main_width_m=mtl.w_main_m,
    )
    line_left = next(e for e in spec.elements if e.id == "LINE_LEFT_C1")
    assert float(line_left.parameters["w_m"]) == pytest.approx(mtl.w_main_m)


