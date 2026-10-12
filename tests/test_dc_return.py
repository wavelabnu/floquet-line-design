"""DC return of the varactor (v1.0.1): reason no_dc_return_for_varactor in the deck generator and the gate.
An n4_none_open cell (open stub, no via) is refused without a bias-return element and accepted with a
100 kOhm return."""
import sys
from pathlib import Path

import pytest

from tvtl.design.feasibility import candidate_cell_factory, feasibility
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import Termination, TopologyFamily, ViaConfig
from tvtl.varactor_palette import get_entry
from tvtl_ads_builder_v2.models import MlefOpen, ViagndVia
from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
from tvtl_ads_builder_v2.parity.self_pumped_design import optimized_self_pumped_spec
from tvtl_ads_builder_v2.validation.dc_return import REASON, NoDCReturn, floating_varactor_terminals

ROOT = Path(__file__).resolve().parents[1]
OPEN4 = TopologyFamily(n_segments=4, via=ViaConfig.NONE, termination=Termination.OPEN)
SHORT1 = TopologyFamily(n_segments=1, via=ViaConfig.SINGLE, termination=Termination.SHORT)
P_OPEN4 = tuple(x * 1e-3 for x in (3.5571, 8.253, 2.2175, 6.1194, 2.2998, 7.5783, 1.6872, 14.6611))
VAR = get_entry('SMV2019-079LF').to_realistic_varactor(v_bias_v=12.0)
FS = 1.3e9


def spec(topology, params, **kw):
    return optimized_self_pumped_spec(topology=topology, params=params, varactor=VAR, substrate=ROGERS_4003C,
                                      n_cells=3, f_signal_Hz=FS, f_pump_Hz=2 * FS, via_model=ViagndVia(),
                                      open_end_model=MlefOpen(), **kw)


def test_deck_generator_refuses_an_open_stub_without_bias_return():
    s = spec(OPEN4, P_OPEN4)
    assert [n for _, _, n in floating_varactor_terminals(s)] == ['n_c1_v', 'n_c2_v', 'n_c3_v']
    with pytest.raises(NoDCReturn, match=f'^{REASON}'):
        emit_flat_hb_netlist(s)


def test_deck_with_a_100_kohm_return():
    s = spec(OPEN4, P_OPEN4, bias_return_ohm=1e5)
    assert floating_varactor_terminals(s) == []
    deck = emit_flat_hb_netlist(s)
    assert deck.count('R:RDCRET_C') == 3 and 'MLEF:MLEF_C1' in deck
    assert all(e.parameters['r_ohm'] == 1e5 for e in s.elements if e.id.startswith('RDCRET_C'))


def test_via_terminated_and_mid_via_cells_have_a_dc_return():
    assert floating_varactor_terminals(spec(SHORT1, (1e-3, 8e-3))) == []
    sys.path.insert(0, str(ROOT / 'examples' / 'real_parts'))
    from real_part_designs import spec_smv1245                       # OPEN stub with two mid-stub vias
    assert floating_varactor_terminals(spec_smv1245()) == []


def test_gate_refuses_without_and_admits_with_a_bias_return():
    with pytest.raises(NoDCReturn):
        candidate_cell_factory(varactor=VAR, substrate=ROGERS_4003C, f_s=FS, N=20)(OPEN4, P_OPEN4)
    cell = candidate_cell_factory(varactor=VAR, substrate=ROGERS_4003C, f_s=FS, N=20,
                                  bias_return_ohm=1e5)(OPEN4, P_OPEN4)
    out = feasibility(cell, FS, N=20)
    assert out['feasible'] and out['reasons'] == []


def test_screen_records_the_reason():
    from tvtl.optimization.stage1_per_cell import coarse_screen

    class Unit:
        def score(self, *a, **k):
            return 1.0
    factory = candidate_cell_factory(varactor=VAR, substrate=ROGERS_4003C, f_s=FS, N=20)
    res = coarse_screen(FS, VAR, families=[OPEN4], n_samples=4, gate='stage0', cell_factory=factory,
                        fom=Unit(), gate_options={'N': 20})
    assert res.feasibility_records and all(r['reasons'] == [REASON] for r in res.feasibility_records)
    assert not res.scored
