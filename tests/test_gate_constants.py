"""Adopted stage-0 gate constants (v1.0.1): T_req = 0.17378 by default, 0.1438 kept as a named constant."""
import sys
from pathlib import Path

import numpy as np
import pytest

from tvtl.design import feasibility as F

ROOT = Path(__file__).resolve().parents[1]


def test_adopted_constants_and_default():
    assert F.T_REQ == 0.17378
    assert F.T_REQ_SCREEN_2026_09 == 0.1438
    assert F.ALPHA_MAX == 0.02
    assert F.feasibility.__defaults__[:2] == (F.T_REQ, F.ALPHA_MAX)


def test_t_req_is_the_cascade_anchor():
    # m*_D / (k V_bt^max); k is (M/V_J)(C_J0/C_0) of the SMV1245 junction at 0 V
    from tvtl.varactor_palette import get_entry
    e = get_entry('SMV1245-079LF')
    k = e.m / e.vj_v * e.cj0_pf / (e.cj0_pf + e.cp_pf)
    assert round(k, 4) == 0.4547
    assert round(0.235625 / (0.4547 * 2.98197), 5) == F.T_REQ


def _smv1245_tabulated_cell():
    sys.path[:0] = [str(ROOT / 'examples/real_parts'), str(ROOT / 'examples/real_parts/boundary')]
    from _stub_table import load_s1p, make_y_override, STUB_80GHz
    from real_part_designs import spec_smv1245
    from tvtl.varactor_palette import get_entry
    from tvtl_ads_builder_v2.analytical.delivered_gain import cell_functions_from_spec
    var = get_entry('SMV1245-079LF').to_realistic_varactor(v_bias_v=0)
    return cell_functions_from_spec(spec=spec_smv1245(n_cells=1), varactor=var,
                                    stub_y_fn=make_y_override(*load_s1p(STUB_80GHz)))[0]


def test_smv1245_example_refused_for_delivery_at_both_thresholds():
    cell = _smv1245_tabulated_cell()
    for t_req in (F.T_REQ, F.T_REQ_SCREEN_2026_09):
        out = F.feasibility(cell, 3.1e9, T_req=t_req, N=20)
        assert 'pump_delivery_below_threshold' in out['reasons'] and out['T_req'] == t_req
    assert F.feasibility(cell, 3.1e9, N=20)['T_req'] == 0.17378


def test_smv1245_zero_bias_screen_has_no_delivery_and_transport_pass_at_both_thresholds():
    """The 2,034 evaluable geometries of the 0 V SMV1245 screen (f_s 3.1 GHz, N 15, 300 Sobol samples per
    linear family): none passes delivery and pump transport together, at either threshold.
    The open-stub families carry a declared 100 kOhm bias return (v1.0.1 refuses them without one)."""
    from tvtl.varactor_palette import get_entry
    from tvtl.optimization.substrate import ROGERS_4003C
    from tvtl.optimization.topology_generator import linear_families
    from tvtl.optimization.stage1_per_cell import coarse_screen
    var = get_entry('SMV1245-079LF').to_realistic_varactor(v_bias_v=0.0)
    factory = F.candidate_cell_factory(varactor=var, substrate=ROGERS_4003C, f_s=3.1e9, N=15, bias_return_ohm=1e5)

    class Unit:
        def score(self, *args, **kwargs):
            return 1.0
    res = coarse_screen(3.1e9, var, families=linear_families(), n_samples=300, seed=0, gate='stage0',
                        cell_factory=factory, fom=Unit(),
                        gate_options={'N': 15, 'T_req': 1e-12, 'alpha_max': 1e9})
    rows = [r for r in res.feasibility_records if 'T_2fs' in r]
    assert len(res.feasibility_records) == 3245 and len(rows) == 2034
    T = np.array([r['T_2fs'] for r in rows])
    a = np.array([r['alpha_2fs_Np_per_cell'] for r in rows])
    for t_req in (F.T_REQ, F.T_REQ_SCREEN_2026_09):
        assert int(np.sum((T >= t_req) & (a <= F.ALPHA_MAX))) == 0
    m = (T > 0) & (a > 0)
    assert np.corrcoef(np.log(T[m]), np.log(a[m]))[0, 1] == pytest.approx(0.88, abs=0.005)
