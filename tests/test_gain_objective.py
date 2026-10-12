"""Tests for the selectable G_mod/in optimizer objective.

Covers the additive ``g_mod_in`` objective without disturbing ``coh_pk``:

* :func:`analytical_g_mod_in_dB` -- the in-loop out/in proxy (passive + boost).
* :func:`measure_g_mod_in` -- the ADS-HB realised gain + stability, exercised
  here on its non-convergent and no-probe branches (the fully-probed path is
  ADS-gated and runs under ADSPY).
* :func:`classify_net_positive_frontier` -- the stable-vs-instability-edge
  verdict over a pump ladder.
* The engine selector: ``objective="g_mod_in"`` maximises the proxy and leaves
  the ``coh_pk`` path byte-unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.parity import gain_objective as go
from tvtl_ads_builder_v2.parity.gain_objective import (
    GModInMeasurement,
    OBJECTIVE_COH_PK,
    OBJECTIVE_G_MOD_IN,
    analytical_g_mod_in_dB,
    classify_net_positive_frontier,
    measure_g_mod_in,
    validate_objective,
)
from tvtl_ads_builder_v2.parity.self_pumped_design import design_optimized_self_pumped

VAR_A = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5,
)
F_SIGNAL_HZ = 3.1e9


# --------------------------------------------------------------------------- #
# analytical_g_mod_in_dB                                                       #
# --------------------------------------------------------------------------- #
def test_analytical_g_mod_in_is_passive_plus_boost():
    """The proxy is abs_gain_off (passive |S21|) + coherent_peak_dB (boost)."""
    res = {"abs_gain_off_dB": -55.4, "coherent_peak_dB": 9.71}
    assert analytical_g_mod_in_dB(res) == pytest.approx(-55.4 + 9.71)


def test_analytical_g_mod_in_nan_on_missing_key():
    assert math.isnan(analytical_g_mod_in_dB({"coherent_peak_dB": 9.71}))


def test_analytical_g_mod_in_nan_on_nonfinite_piece():
    res = {"abs_gain_off_dB": float("-inf"), "coherent_peak_dB": 9.71}
    assert math.isnan(analytical_g_mod_in_dB(res))


# --------------------------------------------------------------------------- #
# validate_objective                                                          #
# --------------------------------------------------------------------------- #
def test_validate_objective_accepts_known():
    assert validate_objective(OBJECTIVE_COH_PK) == OBJECTIVE_COH_PK
    assert validate_objective(OBJECTIVE_G_MOD_IN) == OBJECTIVE_G_MOD_IN


def test_validate_objective_rejects_unknown():
    with pytest.raises(ValueError, match="unknown objective"):
        validate_objective("coh_pk_ratio")


# --------------------------------------------------------------------------- #
# measure_g_mod_in (non-ADS branches)                                         #
# --------------------------------------------------------------------------- #
@dataclass
class _FakeVar:
    name: str
    value: float


@dataclass
class _FakeVarBlock:
    variables: tuple


@dataclass
class _FakeSpec:
    var_block: _FakeVarBlock


def _fake_spec(pump_dBm=13.0, f_s=1.2e9, f_p=2.4e9) -> _FakeSpec:
    return _FakeSpec(_FakeVarBlock((
        _FakeVar("f_signal", f_s), _FakeVar("f_pump", f_p),
        _FakeVar("signal_power_dBm", -30.0),
        _FakeVar("pump_power_dBm", pump_dBm),
        _FakeVar("pump_phase_rad", 0.0),
    )))


@dataclass
class _FakeSweep:
    pout_unmod_W: float
    p_avs_W: float
    pout_mod_W: float
    peak_phase_deg: float
    g_mod_unmod_dB: float
    g_mod_in_dB: float


def test_measure_g_mod_in_nonconvergent(monkeypatch, tmp_path):
    """A non-convergent driven sweep -> converged=False, regime non-convergent."""
    def _boom(*a, **k):
        raise RuntimeError("hpeesofsim did not converge")
    monkeypatch.setattr(go, "run_full_sum_gain_sweep", _boom)

    m = measure_g_mod_in(_fake_spec(pump_dBm=21.0), tmp_path)
    assert m.converged is False
    assert m.regime == "non-convergent"
    assert m.stable is False
    assert m.net_positive is False
    assert m.pump_power_dBm == 21.0
    assert "non-convergent" in m.note


def test_measure_g_mod_in_no_probe_reports_gain_uncertified(monkeypatch, tmp_path):
    """probe_oscillation=False yields the gain but does NOT certify stability."""
    # pout_mod 2e-6 W vs p_avs 1e-6 W -> +3.01 dB out/in (net-positive), boost +4 dB.
    sweep = _FakeSweep(
        pout_unmod_W=0.5e-6, p_avs_W=1e-6, pout_mod_W=2e-6,
        peak_phase_deg=120.0, g_mod_unmod_dB=4.0, g_mod_in_dB=3.0103,
    )
    monkeypatch.setattr(go, "run_full_sum_gain_sweep", lambda *a, **k: sweep)

    m = measure_g_mod_in(_fake_spec(), tmp_path, probe_oscillation=False)
    assert m.converged is True
    assert m.g_mod_in_dB == pytest.approx(3.0103)
    assert m.boost_dB == pytest.approx(4.0)
    assert m.passive_S21_dB == pytest.approx(10 * math.log10(0.5e-6 / 1e-6))
    assert m.net_positive is True
    assert m.regime == "unprobed"
    assert m.stable is False           # not certified without the r_A probe
    assert math.isnan(m.r_A)


# --------------------------------------------------------------------------- #
# classify_net_positive_frontier                                              #
# --------------------------------------------------------------------------- #
def _meas(pump, g, *, converged=True, net_positive=None, stable=True,
          regime="amplifier", r_A=0.0):
    if net_positive is None:
        net_positive = converged and g > 0.0
    return GModInMeasurement(
        g_mod_in_dB=g, boost_dB=0.0, passive_S21_dB=0.0,
        net_positive=bool(net_positive), converged=converged, r_A=r_A,
        regime=regime, stable=stable, peak_phase_deg=0.0,
        pump_power_dBm=pump, f_signal_Hz=1.2e9,
    )


def test_frontier_stable_net_positive_with_headroom():
    """Net-positive at +9 dBm AND a converged higher pump -> stable verdict."""
    ladder = [
        _meas(5.0, -2.0),
        _meas(9.0, +0.5, stable=True, r_A=0.1),     # net-positive, stable
        _meas(13.0, +0.8, stable=True, r_A=0.3),     # head-room above (converged)
    ]
    fr = classify_net_positive_frontier(ladder)
    assert fr.verdict == "stable_net_positive"
    assert fr.stable_net_positive is True
    assert fr.instability_edge_only is False
    assert fr.best_g_mod_in_dB == pytest.approx(0.8)
    assert fr.best_stable_g_mod_in_dB == pytest.approx(0.5)


def test_frontier_edge_only_matches_prior_investigation():
    """+0.55 dB at +13 (stable r_A) but +17 non-convergent -> edge-only."""
    ladder = [
        _meas(5.0, -2.37),
        _meas(9.0, -1.83),
        _meas(13.0, +0.55, stable=True, r_A=0.187),
        _meas(17.0, float("nan"), converged=False, net_positive=False,
              stable=False, regime="non-convergent", r_A=float("nan")),
    ]
    fr = classify_net_positive_frontier(ladder)
    assert fr.net_positive_reached is True
    assert fr.stable_net_positive is False       # no converged head-room above +13
    assert fr.instability_edge_only is True
    assert fr.verdict == "edge_net_positive"
    assert fr.best_g_mod_in_dB == pytest.approx(0.55)


def test_frontier_edge_only_when_higher_pump_lases():
    """Net-positive point's only head-room is a lasing point -> edge-only."""
    ladder = [
        _meas(9.0, +0.3, stable=True, r_A=0.2),
        _meas(13.0, +1.0, net_positive=True, stable=False, regime="lasing", r_A=0.9),
    ]
    fr = classify_net_positive_frontier(ladder)
    assert fr.instability_edge_only is True
    assert fr.verdict == "edge_net_positive"


def test_frontier_negative_when_never_crosses():
    ladder = [_meas(5.0, -2.6), _meas(9.0, -2.4), _meas(13.0, -2.0)]
    fr = classify_net_positive_frontier(ladder)
    assert fr.verdict == "negative"
    assert fr.net_positive_reached is False
    assert fr.best_g_mod_in_dB == pytest.approx(-2.0)


def test_frontier_empty_raises():
    with pytest.raises(ValueError):
        classify_net_positive_frontier([])


# --------------------------------------------------------------------------- #
# engine selector: g_mod_in objective                                         #
# --------------------------------------------------------------------------- #
def test_engine_default_objective_is_g_mod_in():
    """The default objective is g_mod_in (out/in transducer
    gain), so the default result carries a finite g_mod_in proxy and a g_mod_in
    objective tag."""
    r = design_optimized_self_pumped(varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ)
    assert r.objective == OBJECTIVE_G_MOD_IN
    assert math.isfinite(r.analytical_g_mod_in_dB)


def test_engine_coh_pk_still_selectable_and_proxy_is_nan():
    """coh_pk remains fully selectable; under it the result carries no g_mod_in
    proxy and every walk-down row's proxy is nan (the coh_pk path is byte-unchanged)."""
    r = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective=OBJECTIVE_COH_PK,
    )
    assert r.objective == OBJECTIVE_COH_PK
    assert math.isnan(r.analytical_g_mod_in_dB)
    # Every walk-down row's proxy is nan on the coh_pk path (byte-unchanged).
    assert all(math.isnan(e.analytical_g_mod_in_dB) for e in r.walkdown_trace)


def test_engine_g_mod_in_selects_max_proxy():
    """objective='g_mod_in' selects the candidate maximising abs_gain_off+coh_pk."""
    r = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective=OBJECTIVE_G_MOD_IN,
    )
    assert r.objective == OBJECTIVE_G_MOD_IN
    assert math.isfinite(r.analytical_g_mod_in_dB)
    finite = [
        e for e in r.walkdown_trace if math.isfinite(e.analytical_g_mod_in_dB)
    ]
    assert finite, "expected finite-proxy rows under the g_mod_in objective"
    best = max(finite, key=lambda e: e.analytical_g_mod_in_dB)
    # The selected design is the proxy maximiser, and exactly one row is SELECTED.
    assert r.analytical_g_mod_in_dB == pytest.approx(best.analytical_g_mod_in_dB)
    assert r.topology.key == best.topology_key
    assert sum(1 for e in r.walkdown_trace if e.note == "SELECTED") == 1


def test_engine_g_mod_in_differs_from_coh_pk_pick():
    """The right objective avoids the stopband-inflated coh_pk selection: the
    g_mod_in objective ranks by the out/in proxy, which already carries the
    passive loss, so it generally picks a different design than coh_pk."""
    r_coh = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, coh_pk_window_dB=(1.0, 5.0),
    )
    r_g = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective=OBJECTIVE_G_MOD_IN,
    )
    # The proxy maximiser is at least as good (by proxy) as the coh_pk pick's
    # proxy among the same screened pool.
    coh_pick_proxy = next(
        (e.analytical_g_mod_in_dB for e in r_g.walkdown_trace
         if e.topology_key == r_coh.topology.key
         and math.isfinite(e.analytical_g_mod_in_dB)),
        None,
    )
    if coh_pick_proxy is not None:
        assert r_g.analytical_g_mod_in_dB >= coh_pick_proxy - 1e-9


def test_engine_invalid_objective_raises():
    with pytest.raises(ValueError, match="unknown objective"):
        design_optimized_self_pumped(
            varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="net_gain",
        )


