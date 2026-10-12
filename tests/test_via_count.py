"""Via count in the generated deck (v1.0.1): a DOUBLE family emits two VIAGND per cell; a single
family is unchanged (the R1 SMV2019 gate-D record c3 decks reproduce their deposited SHA-256)."""
import hashlib
import math

import pytest

from tvtl.optimization.stage1_per_cell import StubCandidate
from tvtl.optimization.stage2_main_tl import match_z_tl
from tvtl.optimization.stage3_matching import edge_impedance, select_matching
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import VIA_RADIUS_M, Termination, TopologyFamily, ViaConfig
from tvtl.varactor_palette import get_entry
from tvtl_ads_builder_v2.device_spec import with_var_overrides
from tvtl_ads_builder_v2.models import MlefOpen, ViagndVia
from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
from tvtl_ads_builder_v2.parity.self_pumped_design import main_period_anchor_at_3p1GHz, optimized_self_pumped_spec

VIA = ViagndVia(via_radius_m=VIA_RADIUS_M, metal_thickness_m=17.5e-6, rho_relative=1.0)


def spec_for(topology, params, *, part='SMV2019-079LF', bias=20.0, fs=3.1e9, n_cells=15, pump=35.0, match=True):
    dev = get_entry(part).to_realistic_varactor(v_bias_v=bias)
    mtl = match_z_tl(StubCandidate(topology=topology, params=params, a1_score=0.0, f0_Hz=fs), dev,
                     substrate=ROGERS_4003C, f0_Hz=fs)
    im = om = None
    if match:
        z_in, z_out = edge_impedance(mtl, dev, n_cells, substrate=ROGERS_4003C)
        pick = lambda s: None if not s.needed else (s.single_stub if s.recommended == 'single_stub' else s.quarter_wave)
        im = pick(select_matching(z_in, 'input', fs, substrate=ROGERS_4003C))
        om = pick(select_matching(z_out, 'output', fs, substrate=ROGERS_4003C))
    return optimized_self_pumped_spec(
        topology=topology, params=params, varactor=dev, substrate=ROGERS_4003C, n_cells=n_cells,
        main_width_m=float(mtl.w_main_m), main_period_m=main_period_anchor_at_3p1GHz(fs), f_signal_Hz=fs,
        f_pump_Hz=2 * fs, signal_power_dBm=-20.0, pump_power_dBm=pump, pump_phase_rad=0.0,
        input_match=im, output_match=om, via_model=VIA, open_end_model=MlefOpen())


def test_double_short_cell_emits_two_vias_per_cell():
    topology = TopologyFamily(n_segments=2, via=ViaConfig.DOUBLE, termination=Termination.SHORT)
    spec = spec_for(topology, (0.930776e-3, 2.549375e-3, 1.287347e-3, 9.783757e-3), pump=23.0)
    vias = [e for e in spec.elements if e.type == 'VIAGND']
    assert len(vias) == 2 * 15
    for k in range(1, 16):
        cell = [e for e in vias if e.id in (f'VIAGND_C{k}', f'VIAGND_C{k}_2')]
        assert len(cell) == 2 and len({e.pins[0].node for e in cell}) == 1
        assert cell[0].parameters == cell[1].parameters
    assert emit_flat_hb_netlist(spec).count('VIAGND:') == 30


def test_viagnd_emit_and_analytical_termination_scale_with_the_via_count():
    one = VIA.emit(3, end_node='n', ground_node='g', l_via_H=0.0, width_m=1e-3)
    two = VIA.emit(3, end_node='n', ground_node='g', l_via_H=0.0, width_m=1e-3, n_parallel=2)
    assert [e.id for e in one] == ['VIAGND_C3'] and [e.id for e in two] == ['VIAGND_C3', 'VIAGND_C3_2']
    assert two[0] == one[0]
    with pytest.raises(ValueError):
        VIA.emit(3, end_node='n', ground_node='g', l_via_H=0.0, n_parallel=0)
    w = 2 * math.pi * 6.2e9
    z1, z2 = VIA.analytical_termination()(w), VIA.analytical_termination(n_parallel=2)(w)
    assert z2 == pytest.approx(z1 / 2, rel=1e-15)


def test_r1_smv2019_gate_d_c3_decks_are_byte_identical():
    """R1 rule point with gate D, record c3 (n1_single_short, 20 V, 3.1 GHz, N 15, +35 dBm): the on (210 deg)
    and off decks hash to the SHA-256 in the data deposit's runs/r1_redesign_under_law/decks/INDEX.json
    (files written on Windows, CRLF)."""
    topology = TopologyFamily(n_segments=1, via=ViaConfig.SINGLE, termination=Termination.SHORT)
    spec = spec_for(topology, (4.824516e-3, 8.914551e-3))
    on = emit_flat_hb_netlist(with_var_overrides(spec, {"pump_phase_rad": math.radians(210.0)}))
    off = emit_flat_hb_netlist(with_var_overrides(spec, {"pump_power_dBm": -100.0}))
    sha = lambda t: hashlib.sha256(t.replace('\n', '\r\n').encode('utf-8')).hexdigest()
    assert sha(on) == '26d5395d22c9fd0fcb8b7d1a6d9fadc3ba5a68a5d94dc99c19d9a7e78b02905b'
    assert sha(off) == 'b7a57cd52a4905b385783826149266bac8ab0e34d8b1d34d3b5987c09a7507b3'
    assert on.count('VIAGND:') == 15
