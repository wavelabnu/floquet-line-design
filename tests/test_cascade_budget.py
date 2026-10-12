"""Cascade layer (v1.0.1): phase mismatch, kappa/m, m_EP, m* and the phase-margin gate.

Reference values: the as-built SMV1245 cell (stub scale 1, 0 V, N = 20), threshold run of the paper
(Results 6, Fig. 6): Delta_theta 55.55 deg, kappa/m 0.0541, m_EP 8.96 (model dispersion) and 3.99
(measured dispersion), m*_D 0.235625 at 3.3 GHz, V_bt^max 2.98197 V."""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tvtl.design.feasibility import T_REQ, feasibility
from tvtl_ads_builder_v2.analytical import cascade_budget as cb

ROOT = Path(__file__).resolve().parents[1]
MEAS = ROOT / 'tests' / 'fixtures' / 'measured_dispersion' / 'dispersion_from_fft_ridge_smooth_b.csv'


@pytest.fixture(scope='module')
def asbuilt():
    for p in (ROOT / 'scripts' / 'dense_map', ROOT / 'scripts' / 'two_filter_sweep'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import dense_map
    return dense_map.asbuilt_cell()


def measured():
    rows = list(csv.DictReader(MEAS.open(encoding='utf-8')))
    return [float(r['Frequency_GHz']) for r in rows], [float(r['beta_norm']) for r in rows]


def test_budget_at_3p1_ghz_model_and_measured_dispersion(asbuilt):
    b = cb.cascade_budget(asbuilt, 3.1e9, N=20, measured=measured())
    assert b['delta_theta_deg'] == pytest.approx(55.55, rel=1e-2)
    assert b['kappa_per_m'] == pytest.approx(0.0541, rel=1e-2)
    assert b['kappa_fit_residual'] < 0.01
    assert b['m_EP'] == pytest.approx(8.96, rel=1e-2)
    assert b['delta_theta_meas_deg'] == pytest.approx(24.75, rel=1e-2)
    assert b['m_EP_meas'] == pytest.approx(3.99, rel=1e-2)
    assert b['m_EP'] == pytest.approx(abs(b['delta_theta_rad']) / (2 * b['kappa_per_m']))
    assert b['phase_margin'] == pytest.approx(0.5 * b['m_del'] - b['m_EP'])
    assert b['g_m_del'] == pytest.approx(-b['alpha_s_Np'])        # kappa < |Delta_theta|/2: no growth


def test_t_req_inputs_reproduce(asbuilt):
    m = cb.m_star(asbuilt, 3.3e9, N=20)
    assert m['m_star'] == pytest.approx(0.235625, abs=1e-6)
    vbt = np.abs(cb.branch_top_profile([asbuilt] * 20, 6.2e9, 19.5))
    assert vbt.max() == pytest.approx(2.98197, abs=1e-5) and int(vbt.argmax()) + 1 == 4
    k = cb.delivered_depth(asbuilt, 3.1e9)['k_per_V']
    assert round(k, 4) == 0.4547
    assert round(m['m_star'] / (0.4547 * 2.98197), 5) == T_REQ


def test_phase_margin_gate_is_off_by_default_and_refuses_the_as_built_cell(asbuilt):
    off = feasibility(asbuilt, 3.1e9, N=20)
    assert 'phase_margin_negative' not in off['reasons'] and off['phase_margin'] is None
    on = feasibility(asbuilt, 3.1e9, N=20, phase_margin=True)
    assert 'phase_margin_negative' in on['reasons']
    assert on['phase_margin'] == pytest.approx(0.5 * on['cascade']['m_del'] - 8.96, abs=0.1)
    with pytest.raises(ValueError):
        feasibility(asbuilt, 3.1e9, phase_margin=True)               # needs N


def test_phase_margin_positive_on_a_phase_matched_combined_cell():
    """SMV2020 n4_single_combined geometry (14.2 V, f_s 2.593 GHz, N 34) built by the release factory."""
    from tvtl.design.feasibility import candidate_cell_factory
    from tvtl.optimization.substrate import ROGERS_4003C
    from tvtl.optimization.topology_generator import Termination, TopologyFamily, ViaConfig
    from tvtl.varactor_palette import get_entry
    params = (0.0035170897531587825, 0.005485121465946426, 0.0015067719882817842, 0.011771474895690174,
              0.0035598381547818332, 0.01154448306502339, 0.0006017877067045959, 0.0077472441476443485,
              0.003991563990135333, 0.003235846390466071)
    fs = 2.5925677967512786e9
    var = get_entry('SMV2020-079LF').to_realistic_varactor(v_bias_v=14.2069662713099)
    cell = candidate_cell_factory(varactor=var, substrate=ROGERS_4003C, f_s=fs, N=34,
                                  main_period_m=0.006798199233908647)(
        TopologyFamily(n_segments=4, via=ViaConfig.SINGLE, termination=Termination.COMBINED), params)
    out = feasibility(cell, fs, N=34, phase_margin=True, P_pump_dBm=20.0)
    assert 'phase_margin_negative' not in out['reasons']
    assert out['phase_margin'] == pytest.approx(0.1478, abs=1e-3)


def test_cascade_gate_refuses_when_the_third_harmonic_is_above_the_tee_edge(asbuilt):
    out = feasibility(asbuilt, 4e9, N=20, phase_margin=True)         # 3 f_s = 12 GHz
    assert 'tee_model_out_of_range' in out['reasons'] and out['phase_margin'] is None
    assert out['tee_model_refusal'].startswith('tee_model_out_of_range')
    assert out['T_2fs'] is not None                                  # f_s and 2 f_s are in range


def test_optimiser_cascade_gate_turns_the_phase_margin_on(monkeypatch):
    """... and the delivered-depth ceiling at the varactor's bias (F5)."""
    import types
    import tvtl.design.feasibility as F
    import tvtl.optimization.stage1_per_cell as module
    from tvtl.optimization.topology_generator import linear_families
    seen = []

    def fake(cell_fn, f_s, **kw):
        seen.append(kw)
        return dict(feasible=False, reasons=['phase_margin_negative'])
    monkeypatch.setattr(F, 'feasibility', fake)
    module.coarse_screen(1e9, types.SimpleNamespace(V_bias_v=7.8), families=linear_families()[:1], n_samples=2,
                         gate='cascade', cell_factory=lambda t, p: None, gate_options={'N': 20, 'v_max_V': 22.0})
    assert seen and all(kw.get('phase_margin') is True and kw.get('N') == 20 for kw in seen)
    assert all(kw['depth_ceiling'] == {'bias_V': 7.8, 'v_max_V': 22.0} for kw in seen)
    with pytest.raises(ValueError):
        module.coarse_screen(1e9, None, families=linear_families()[:1], n_samples=2, gate='cascade',
                             cell_factory=lambda t, p: None, gate_options={'N': 20})
