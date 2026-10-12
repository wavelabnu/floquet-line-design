"""Tests for the cheap AMPLIFICATION-vs-OSCILLATION check.

Two layers, all ADS-free:

* pure helpers — :func:`log_amplitude_slope` (the input-dependence fit),
  :func:`classify_amplification` (category -> binary verdict), and
  :func:`pumped_linearized_stability_available` (the honest K/mu status).
* :func:`run_amplification_check` exercised end-to-end with the HB layer mocked
  (the reused full-sum sweep + the per-point runner are monkeypatched), covering
  the amplifier, oscillator, non-convergent, and signal-off-non-convergent
  branches. The fully-wired ADS path runs under ADSPY in the STEP-2 driver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

import tvtl_ads_builder_v2.analysis.amplification_check as ac
from tvtl_ads_builder_v2.analysis.amplification_check import (
    AMPLIFICATION,
    INDETERMINATE,
    NON_CONVERGENT,
    OSCILLATION,
    classify_amplification,
    log_amplitude_slope,
    pumped_linearized_stability_available,
    run_amplification_check,
)


# --------------------------------------------------------------------------- #
# log_amplitude_slope (the input-dependence fit)                              #
# --------------------------------------------------------------------------- #
def test_slope_is_one_when_output_tracks_input():
    """|V_out| proportional to |V_in| (constant transducer gain) -> slope 1."""
    dbm = [-60.0, -50.0, -40.0, -30.0, -20.0]
    # out = 1e-3 * 10**((P - (-30))/20): linear voltage tracking, gain pinned.
    out = [1e-3 * 10 ** ((p + 30.0) / 20.0) for p in dbm]
    assert log_amplitude_slope(dbm, out) == pytest.approx(1.0, abs=1e-9)


def test_slope_is_zero_when_output_pinned():
    """Output independent of input (oscillator) -> slope 0."""
    dbm = [-60.0, -50.0, -40.0, -30.0, -20.0]
    out = [1.0e-3] * 5
    assert log_amplitude_slope(dbm, out) == pytest.approx(0.0, abs=1e-9)


def test_slope_drops_floored_and_missing_points():
    """At/below-floor and None points carry no slope info and are dropped."""
    dbm = [-60.0, -50.0, -40.0, -30.0]
    # first two unusable; remaining two are |V|=10**(P/20) -> voltage tracking, slope 1.
    out = [1e-20, None, 10 ** (-40.0 / 20.0), 10 ** (-30.0 / 20.0)]
    s = log_amplitude_slope(dbm, out, floor_V=1e-15)
    assert s == pytest.approx(1.0, abs=1e-9)


def test_slope_nan_with_too_few_usable_points():
    assert math.isnan(log_amplitude_slope([-30.0], [1e-3]))
    assert math.isnan(log_amplitude_slope([-30.0, -20.0], [1e-20, None], floor_V=1e-15))


# --------------------------------------------------------------------------- #
# classify_amplification (category -> binary verdict)                         #
# --------------------------------------------------------------------------- #
def test_classify_lasing_is_oscillation():
    assert classify_amplification("lasing", True) == OSCILLATION


def test_classify_amplifier_and_passive_are_amplification():
    assert classify_amplification("amplifier", True) == AMPLIFICATION
    assert classify_amplification("passive", True) == AMPLIFICATION


def test_classify_non_convergent():
    assert classify_amplification("non-convergent", True) == NON_CONVERGENT


def test_classify_disagreement_is_indeterminate():
    # Even a lasing category is not asserted when the two tests disagree.
    assert classify_amplification("lasing", False) == INDETERMINATE
    assert classify_amplification("amplifier", False) == INDETERMINATE


def test_classify_unprobed_is_indeterminate():
    assert classify_amplification("amplifier", True, probed=False) == INDETERMINATE


# --------------------------------------------------------------------------- #
# pumped_linearized_stability_available (the K/mu status)                      #
# --------------------------------------------------------------------------- #
def test_k_mu_not_available_on_this_framework():
    available, reason = pumped_linearized_stability_available()
    assert available is False
    assert "conversion S-parameters" in reason or "conversion" in reason.lower()


# --------------------------------------------------------------------------- #
# run_amplification_check (HB layer mocked)                                    #
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
    peak_phase_deg: float
    pout_mod_W: float
    pout_unmod_W: float
    p_avs_W: float
    g_mod_unmod_dB: float
    g_mod_in_dB: float
    rows: tuple
    family_mixes: tuple


def _fake_sweep(*, peak_phase=120.0, pout_mod_W=1e-8, g_mod_unmod_dB=9.7,
                g_mod_in_dB=-1.0, pout_unmod_W=1e-9, p_avs_W=1e-6, idler=0.35):
    # out_on_ref = sqrt(2*50*pout_mod_W); pout_mod_W=1e-8 -> 1e-3 V.
    rows = ({"phase_deg": peak_phase, "family_re": [1.0, idler], "family_im": [0.0, 0.0]},)
    return _FakeSweep(
        peak_phase_deg=peak_phase, pout_mod_W=pout_mod_W, pout_unmod_W=pout_unmod_W,
        p_avs_W=p_avs_W, g_mod_unmod_dB=g_mod_unmod_dB, g_mod_in_dB=g_mod_in_dB,
        rows=rows, family_mixes=((1, 0), (-1, 1)),
    )


@pytest.fixture
def _identity_overrides(monkeypatch):
    """with_var_overrides is a real DeviceSpec rebuild; the fakes ignore the spec
    contents, so neutralise it to identity for the mocked HB path."""
    monkeypatch.setattr(
        "tvtl_ads_builder_v2.device_spec.with_var_overrides",
        lambda spec, overrides: spec,
    )


def test_run_check_amplifier(monkeypatch, tmp_path, _identity_overrides):
    """Output tracks input (slope~1), signal-off at floor (r_A~0) -> amplification."""
    monkeypatch.setattr(ac, "run_full_sum_gain_sweep", lambda *a, **k: _fake_sweep())

    def _amp_point(spec, run_dir, tag, *, signal_dBm, phase_deg, f_s, f_p, timeout_s):
        if signal_dBm <= -150.0:          # signal-off -> numerical floor (no osc)
            return 1e-16
        return 1e-3 * 10 ** ((signal_dBm + 30.0) / 20.0)   # linear voltage tracking
    monkeypatch.setattr(ac, "_run_point", _amp_point)

    r = run_amplification_check(_fake_spec(), tmp_path)
    assert r.converged is True
    assert r.classification == AMPLIFICATION
    assert r.regime == "amplifier"
    assert r.input_dependence_slope == pytest.approx(1.0, abs=1e-6)
    assert r.r_A < 0.01
    assert r.signal_off_above_floor is False
    assert r.tests_agree is True
    assert r.is_laser_input_independence is False
    assert r.is_laser_gain_linearity is False
    assert r.k_mu_available is False
    assert r.idler_over_sig == pytest.approx(0.35)


def test_run_check_oscillator(monkeypatch, tmp_path, _identity_overrides):
    """Output pinned (slope~0), persists signal-off (r_A>1) -> oscillation."""
    monkeypatch.setattr(ac, "run_full_sum_gain_sweep", lambda *a, **k: _fake_sweep())

    def _osc_point(spec, run_dir, tag, *, signal_dBm, phase_deg, f_s, f_p, timeout_s):
        if signal_dBm <= -150.0:          # self-sustained output with no input
            return 1.31e-3
        return 1.0e-3                      # pinned regardless of input
    monkeypatch.setattr(ac, "_run_point", _osc_point)

    r = run_amplification_check(_fake_spec(pump_dBm=18.0), tmp_path)
    assert r.converged is True
    assert r.classification == OSCILLATION
    assert r.regime == "lasing"
    assert r.input_dependence_slope == pytest.approx(0.0, abs=1e-6)
    assert r.r_A == pytest.approx(1.31, abs=1e-3)
    assert r.signal_off_above_floor is True
    assert r.tests_agree is True
    assert r.is_laser_input_independence is True
    assert r.is_laser_gain_linearity is True


def test_run_check_nonconvergent_driven_sweep(monkeypatch, tmp_path, _identity_overrides):
    """A non-convergent driven sweep -> non-convergent verdict (instability proxy)."""
    def _boom(*a, **k):
        raise RuntimeError("hpeesofsim did not converge")
    monkeypatch.setattr(ac, "run_full_sum_gain_sweep", _boom)

    r = run_amplification_check(_fake_spec(pump_dBm=26.0), tmp_path)
    assert r.converged is False
    assert r.classification == NON_CONVERGENT
    assert r.regime == "non-convergent"
    assert math.isnan(r.r_A)
    assert r.pump_power_dBm == 26.0


def test_run_check_signal_off_nonconvergent_is_indeterminate(
    monkeypatch, tmp_path, _identity_overrides
):
    """Driven sweep converges but the autonomous signal-off run does not: r_A is
    unmeasurable, so the cheap pair cannot certify a verdict -> indeterminate."""
    monkeypatch.setattr(ac, "run_full_sum_gain_sweep", lambda *a, **k: _fake_sweep())

    def _point(spec, run_dir, tag, *, signal_dBm, phase_deg, f_s, f_p, timeout_s):
        if signal_dBm <= -150.0:
            return None                    # autonomous problem fails to converge
        return 1e-3 * 10 ** ((signal_dBm + 30.0) / 20.0)
    monkeypatch.setattr(ac, "_run_point", _point)

    r = run_amplification_check(_fake_spec(), tmp_path)
    assert r.converged is True
    assert r.signal_off_converged is False
    assert r.classification == INDETERMINATE
    assert r.regime == "unprobed"
    assert math.isnan(r.r_A)
    # the slope is still measured (and reads stable here)
    assert r.input_dependence_slope == pytest.approx(1.0, abs=1e-6)
    assert r.is_laser_gain_linearity is False


def test_run_check_to_dict_is_json_safe(monkeypatch, tmp_path, _identity_overrides):
    monkeypatch.setattr(ac, "run_full_sum_gain_sweep", lambda *a, **k: _fake_sweep())
    monkeypatch.setattr(
        ac, "_run_point",
        lambda *a, **k: 1e-16 if k["signal_dBm"] <= -150.0
        else 1e-3 * 10 ** ((k["signal_dBm"] + 30.0) / 20.0),
    )
    d = run_amplification_check(_fake_spec(), tmp_path).to_dict()
    import json
    json.dumps(d)   # must not raise
    for key in ("classification", "regime", "input_dependence_slope", "r_A",
                "k_mu_available", "k_mu_note", "tests_agree"):
        assert key in d


