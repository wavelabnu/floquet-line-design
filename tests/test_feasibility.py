import numpy as np
import pytest
from tvtl.design.feasibility import feasibility
from tvtl_ads_builder_v2.analytical.delivered_gain import BranchCell

def cell(f):
    t=.2*f/1e9
    l=np.array([[np.cos(t),50j*np.sin(t)],[1j*np.sin(t)/50,np.cos(t)]])
    return BranchCell(l,l,1+2j*np.pi*f*.1e-9,.1e-12,1,3)

def test_delivery_and_quantitative_attenuation():
    out=feasibility(cell,1e9,N=7,board_slot_mm2=10,stub_area_mm2=5)
    assert out['T_2fs']>.1438
    assert out['pump_survival_e_minus_N_alpha']==pytest.approx(np.exp(-7*out['alpha_2fs_Np_per_cell']))
    assert out['constructible']
    assert not feasibility(cell,1e9,N=7,board_slot_mm2=1,stub_area_mm2=5)['feasible']

def test_filter_precedes_scoring(monkeypatch):
    import tvtl.optimization.stage1_per_cell as module
    from tvtl.optimization.topology_generator import linear_families
    class Score:
        def score(self,*args,**kwargs):raise AssertionError('Infeasible candidate reached scorer')
    result=module.coarse_screen(1e9,None,families=linear_families()[:1],n_samples=2,
        fom=Score(),gate='stage0',cell_factory=lambda t,p:cell,gate_options={'T_req':1e9})
    assert not result.scored and result.feasibility_records

@pytest.mark.parametrize('args',[{'f_s':0},{'N':0},{'alpha_max':-1}])
def test_invalid(args):
    options={'f_s':1e9};options.update(args)
    with pytest.raises(ValueError):feasibility(cell,**options)


def test_fabricated_nominal_two_physical_exclusion_reasons():
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'examples/real_parts'))
    sys.path.insert(0,str(root/'examples/real_parts/boundary'))
    from _stub_table import load_s1p,make_y_override,STUB_80GHz
    from real_part_designs import spec_smv1245
    from tvtl.varactor_palette import get_entry
    from tvtl_ads_builder_v2.analytical.delivered_gain import cell_functions_from_spec
    var=get_entry('SMV1245-079LF').to_realistic_varactor(v_bias_v=0)
    f=cell_functions_from_spec(spec=spec_smv1245(n_cells=1),varactor=var,
                              stub_y_fn=make_y_override(*load_s1p(STUB_80GHz)))[0]
    a=feasibility(f,3.1e9,N=20);b=feasibility(f,5.3e9,N=20)
    assert a['T_2fs']==pytest.approx(.00260578469767373,rel=1e-8)
    # v1.0.1 gate D: alpha(f_s) = 0.0377 Np/cell > 0.02 (was ['pump_delivery_below_threshold'] in 1.0.0)
    assert a['alpha_fs']==pytest.approx(.03765331117754617,rel=1e-8)
    assert a['reasons']==['pump_delivery_below_threshold','signal_transport_above_budget']
    assert b['alpha_2fs_Np_per_cell']==pytest.approx(1.800520124012738,rel=1e-8)
    assert b['reasons']==['pump_attenuation_above_threshold']

def test_stage2_varactor_a_cells_feasible():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/real_parts'))
    from proxy_designs import PROXY_PICKS,PROXY_VARACTOR,SUBSTRATE
    from tvtl.design.feasibility import candidate_cell_factory
    factory=candidate_cell_factory(varactor=PROXY_VARACTOR,substrate=SUBSTRATE,f_s=2e9,N=15)
    assert all(feasibility(factory(p.topology,p.params),2e9,N=15)['feasible'] for p in PROXY_PICKS.values())


def test_k7_slot_is_not_board_area_divided_by_n():
    opts=dict(period_mm=6.11,board_width_mm=30,main_width_mm=4.27,
              board_area_per_cell_mm2=225,N=20)
    nominal=feasibility(cell,1e9,stub_area_mm2=46.53,**opts)
    optimum=feasibility(cell,1e9,stub_area_mm2=46.53*4.15,**opts)
    assert nominal['routing_slot_mm2']==pytest.approx(78.60515)
    assert nominal['constructible']
    assert optimum['board_area_ok'] and not optimum['constructible']
    assert 'stub_exceeds_cell_slot' in optimum['reasons']
    with pytest.raises(ValueError):
        feasibility(cell,1e9,board_slot_mm2=225,**opts)
    constrained=feasibility(cell,1e9,board_slot_mm2=100,
                            board_area_per_cell_mm2=20,stub_area_mm2=40)
    assert not constrained['constructible'] and constrained['routing_slot_ok']
