"""Delivered-depth ceiling before harmonic balance (v1.0.1): reason delivered_depth_above_ceiling."""
import numpy as np
import pytest

from tvtl.design.depth_ceiling import REASON, depth_ceiling_check, part_v_max_V, reverse_swing_limit_V
from tvtl.design.feasibility import feasibility
from tvtl_ads_builder_v2.analytical.delivered_gain import BranchCell

C_J = 1e-12
F_S = 1e9


def cell(resonant):
    """Line halves around a branch; resonant: series resonance at 1.2 x the pump (2.4 GHz), so the branch
    multiplies the pump voltage at the junction (T(2 f_s) > 3) while the pump still propagates."""
    l_s = 1 / ((2 * np.pi * 1.2 * 2 * F_S) ** 2 * C_J) if resonant else .1e-9

    def sample(f):
        t = .2 * f / 1e9
        line = np.array([[np.cos(t), 50j * np.sin(t)], [1j * np.sin(t) / 50, np.cos(t)]])
        return BranchCell(line, line, .5 + 2j * np.pi * f * l_s, C_J, 1, 3)
    return sample


def test_swing_limit_and_part_ceiling():
    assert reverse_swing_limit_V(0.0, 12.0) == 0.0             # 0 V: any pump swing forward-biases
    assert reverse_swing_limit_V(16.0, 20.0) == 4.0
    assert reverse_swing_limit_V(2.0, None) == 2.0
    from tvtl.varactor_palette import get_entry
    assert part_v_max_V(get_entry('SMV1245-079LF'))[0] == 12.0
    e = get_entry('SMV2020-079LF')
    assert part_v_max_V(e)[0] == max(e.cv_table)


def test_constructed_resonance_is_refused_before_hb():
    opts = dict(bias_V=2.0, v_max_V=12.0)
    out = feasibility(cell(True), F_S, N=10, P_pump_dBm=10.0, depth_ceiling=opts)
    d = out['depth_ceiling']
    assert out['T_2fs'] > 3 and out['alpha_2fs_Np_per_cell'] <= 0.02
    assert REASON in out['reasons'] and not out['feasible']
    assert d['max_junction_pump_V'] > d['swing_limit_V'] == 2.0 and d['m_del_raw'] > d['m_ceiling']
    assert d['m_ceiling'] == pytest.approx(1 / 3 * 2.0)            # (M/V_j) A_max
    calm = feasibility(cell(False), F_S, N=10, P_pump_dBm=10.0, depth_ceiling=opts)
    assert REASON not in calm['reasons'] and calm['depth_ceiling']['ok']
    assert feasibility(cell(True), F_S, N=10, P_pump_dBm=10.0)['depth_ceiling'] is None   # off by default


def test_check_matches_the_delivered_pump_solve():
    from tvtl_ads_builder_v2.analytical.delivered_gain import delivered_pump_profile
    d = depth_ceiling_check(cell(True), F_S, N=10, P_pump_dBm=-10.0, bias_V=3.0, v_max_V=None)
    vj = np.abs(delivered_pump_profile(cell_fn=cell(True), N=10, f_p=2 * F_S, P_inc_dBm=-10.0))
    assert d['max_junction_pump_V'] == pytest.approx(vj.max()) and d['swing_limit_V'] == 3.0


def test_workflow_refuses_before_any_hb_job(monkeypatch):
    import tvtl.design.depth_ceiling as DC
    import tvtl.optimization.stage4_validation as S4
    from tvtl.optimization.topology_generator import linear_families
    from tvtl.optimization.workflow import WorkflowConfig, decoupled_design_workflow
    from tvtl.varactor_palette import get_entry
    calls = []
    monkeypatch.setattr(DC, 'depth_ceiling_check', lambda *a, **k: calls.append(k) or
                        {'ok': False, 'reason': REASON, 'max_junction_pump_V': 9.0, 'swing_limit_V': 2.0})
    monkeypatch.setattr(S4, '_ads_gains', lambda *a, **k: pytest.fail('an HB job was emitted'))
    monkeypatch.setattr(S4, '_ads_available', lambda: True)
    var = get_entry('SMV2019-079LF').to_realistic_varactor(v_bias_v=20.0)
    cfg = WorkflowConfig(n_samples=4, top_k=2, top_m=1, refine_maxiter=1, families=tuple(linear_families()[:2]),
                         run_ads=True, ads_tmp_dir='unused', pumps_dBm=(0.0,), phases_deg=(0.0,), min_coherent_dB=-1e9)
    res = decoupled_design_workflow(2e9, var, cfg)
    assert res.reports and calls
    assert all(r.refusal and r.refusal['reason'] == REASON and r.passed is False and not r.ads for r in res.reports)
    assert all(k['bias_V'] == 20.0 and k['P_pump_dBm'] == 0.0 for k in calls)
