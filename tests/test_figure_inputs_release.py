"""Fig. 2 and Fig. 7 inputs from the release alone (v1.0.1): mid-stub shunt stub functions,
the two-filter sweep (scripts/two_filter_sweep) and the dense map (scripts/dense_map).

Reference values are copied from the published runs: the R6c dense tables behind Fig. 7 and the
two-filter task-2 sweep behind Fig. 2(c)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tvtl_ads_builder_v2.analytical.floquet_stub import (
    StubSegment, Y_stub_at_freq, Y_stub_at_freq_with_shunts)

ROOT = Path(__file__).resolve().parents[1]
SUB = {"eps_r": 3.38, "h_m": 1.524e-3, "t_m": 17.5e-6}
SEGS = (StubSegment(1.93e-3, 6.83e-3), StubSegment(1.41e-3, 5.25e-3), StubSegment(1.98e-3, 1.04e-3))


def _scripts():
    for p in (ROOT / 'scripts' / 'dense_map', ROOT / 'scripts' / 'two_filter_sweep'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import dense_map
    import sweep
    import _common
    return dense_map, sweep, _common


@pytest.mark.parametrize('f', [3.1e9, 6.2e9, 9.3e9])
def test_shunted_stub_reduces_to_series_stub_without_shunts(f):
    w = 2 * math.pi * f
    for term in ('open', 'short'):
        assert Y_stub_at_freq_with_shunts(w, SEGS, SUB, termination=term) == Y_stub_at_freq(w, SEGS, SUB, termination=term)
        assert Y_stub_at_freq_with_shunts(w, SEGS, SUB, termination=term, mid_shunt_loads={}) == \
            Y_stub_at_freq(w, SEGS, SUB, termination=term)


def test_short_at_a_section_junction_isolates_the_outer_sections():
    w = 2 * math.pi * 6.2e9
    y = Y_stub_at_freq_with_shunts(w, SEGS, SUB, termination='open', mid_shunt_loads={1: 'short'})
    assert y == pytest.approx(Y_stub_at_freq(w, SEGS[:1], SUB, termination='short'), rel=1e-12)
    with pytest.raises(ValueError):
        Y_stub_at_freq_with_shunts(w, SEGS, SUB, mid_shunt_loads={3: 'short'})


# R6c dense tables (Fig. 7 inputs), exact text of the published rows
STUB_ROWS = [
    "stub,0.25,0.5,0.0,1.0,,,7.369999999999999,1.3321399767462694,0.01178837822275915,0.002135805424887829,True,46.53226666666667,True,True,True,True,True,True,19.335819507497995,1.4265229351422806,3.1033678815884866,1.4265229351422806,B",
    "stub,3.1,6.2,0.0,1.0,,,7.369999999999999,0.0016031449172384177,0.0016977421460566587,0.032238176192241175,True,46.53226666666667,False,True,True,True,False,False,-39.055722072203864,3.1794682343870373,-2.126015968116137,-39.055722072203864,A",
    "stub,5.3,10.6,0.0,1.0,,,7.369999999999999,0.24463543074124958,0.775118803425757,0.0020269607049853895,True,46.53226666666667,True,False,True,True,False,False,4.615209408381042,-131.1792385311225,3.122276386329938,-131.1792385311225,B",
    "stub,6.0,12.0,0.0,1.0,,,7.369999999999999,0.03971124914551613,0.02622859669857418,0.0016926323464638857,True,46.53226666666667,False,False,True,True,False,False,-11.176906754541339,-1.0820318184763067,3.180355908772294,-11.176906754541339,A",
]
ROW_305 = "stub,3.05,6.1,3.0,1.6,,,2.878862772043601,0.174182037270106,0.018103386873269168,0.003149934919699052,True,74.45162666666668,True,True,True,True,True,True,1.664889595973377,0.32947963237568023,2.9271933057498805,0.32947963237568023,B"
PLAIN_ROW = "plain,3.1,6.2,0.0,,0.3,0.0,7.369999999999999,0.09803421092267202,0.11063327149378102,1.3107559824011643,False,,False,False,False,,False,False,-3.327624569901568,-15.744811923899638,-224.23012926273026,-224.23012926273026,D"
PROXY_ROWS = [
    "stub,0.2,0.0,0.4,,,10,0.8402775211245991,-0.8802626379121657,0.15195765399833894,0.3290742760332972,False,13.364948011982975,0.1089361701128376",
    "stub,0.25,0.0,0.4,,,10,1.8499534751992264,-0.03263915264394947,-0.012524785298797886,0.25266368441308024,False,13.271702981127117,0.13956265358527897",
]


def test_dense_map_rows_reproduce_the_fig7_tables():
    dm, _, _ = _scripts()
    line = lambda r: ",".join(dm.fmt(r[c]) for c in dm.GATE_COLS)
    cell = dm.make_cell('stub', 0.0, 1.0)
    for ref in STUB_ROWS:
        fs = float(ref.split(",")[1])
        assert line(dm.gate_row(cell, 'stub', fs, 0.0, 1.0, None, None, dm.c0_pf(0.0))) == ref
    assert line(dm.gate_row(dm.make_cell('stub', 3.0, 1.6), 'stub', 3.05, 3.0, 1.6, None, None, dm.c0_pf(3.0))) == ROW_305
    plain = dm.make_cell('plain', 0.0, 1, 0.3, 0.0)
    assert line(dm.gate_row(plain, 'plain', 3.1, 0.0, None, 0.3, 0.0, dm.c0_pf(0.0))) == PLAIN_ROW


def test_dense_map_proxy_rows(tmp_path):
    dm, _, _ = _scripts()
    group = [('stub', 0.2, 0.0, 0.4, None, None), ('stub', 0.25, 0.0, 0.4, None, None)]
    dm.proxy_chunk((group, str(tmp_path)))
    assert (tmp_path / 'stub_V0.00_s0.4.csv').read_text(encoding='utf-8').splitlines() == PROXY_ROWS


def test_dense_map_uses_the_reduced_tee_explicitly_and_the_release_refuses_it():
    dm, _, _ = _scripts()
    from tvtl_ads_builder_v2.analytical.delivered_gain import TeeModelOutOfRange
    dm.make_cell('stub', 0.0, 1.0)(12e9)                          # the tables' reduced tee model
    with pytest.raises(TeeModelOutOfRange):
        dm.asbuilt_cell()(12e9)                                   # release default
    first = dm.tee_table()["first_fs_GHz_with_reduced_tee"]
    assert first == {"1fs": None, "2fs": 5.85, "3fs": 3.9, "5fs": 2.35, "7fs": 1.7}


def test_fig7_counts_rule(tmp_path):
    """Admissible = law margin >= 0 at the adopted constants (delivery, pump and signal transport) and gate C."""
    dm, _, _ = _scripts()
    p = tmp_path / 'dense_stub.csv'
    p.write_text(",".join(dm.GATE_COLS) + "\n" + "\n".join(STUB_ROWS + [ROW_305]) + "\n", encoding='utf-8')
    c = dm.counts(p)
    assert (c['points'], c['law'], c['admissible']) == (5, 2, 2)   # 0.25 GHz at 0 V and 3.05 GHz at 3 V, s 1.6
    assert c['T_req'] == 0.17378 and c['reduced_tee_pump_from_fs_GHz'] == 5.85


def test_two_filter_sweep_reproduces_the_fig2c_task2_grid_point():
    """Task-2 sweep at C0 = 7.37 pF, s = 1, f_p = 6.2 and 12 GHz (published npz values)."""
    _, sweep, common = _scripts()
    assert common.T_REQ == 0.1438
    ev = common.CellEvaluator(None)                             # c0 = C0_F_NOMINAL
    y = sweep.stub_y_fn_for_s(1.0, 0.1949e-9)
    ev.stub_y_fn = y
    for fp, T, half_p, alpha_p in ((6.2, 0.0016031449172384182, 0.2684112337729077, 0.001697742146056659),
                                   (12.0, 0.03971124914551613, 0.31171282634965397, 0.026228596698574178)):
        _, a, h = ev.dispersion(fp * 1e9)
        assert abs(h) == pytest.approx(half_p, rel=1e-12)
        assert a == pytest.approx(alpha_p, rel=1e-12)
        assert common.branch_T(fp * 1e9, y, c0_f=common.C0_F_NOMINAL) == pytest.approx(T, rel=1e-12)
    assert abs(ev.dispersion(3.1e9)[2]) == pytest.approx(0.9869816836722362, rel=1e-12)
    for c0, frac in ((0.3e-12, 0.9881784507984476), (7.37e-12, 0.7275086385990869)):
        assert sweep.difficulty_law_fraction(c0, 3.1e9, sweep.main_line_z0(3.1e9)) == pytest.approx(frac, rel=1e-12)
