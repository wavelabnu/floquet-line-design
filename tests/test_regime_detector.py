"""Unit tests for the loss-invariant regime detector (src/.../analysis/regime_detector.py).

Pure-logic known-answer cases -- no HB. Locks the physical-definition classification and
the loss-invariance property (r_A is a same-config ratio, so a common loss factor cancels).
"""
from __future__ import annotations

import pytest

from tvtl_ads_builder_v2.analysis.regime_detector import (
    classify_regime_loss_invariant as classify,
)


def test_amplifier_input_tracks_output():
    # signal off -> output collapses (tracks input); slope ~ 1; net enhancement + idler.
    v = classify(out_signal_off=3.0e-8, out_signal_on=1.0e-4, input_scaling_slope=1.0,
                 enhancement_dB=9.7, idler_over_sig=0.35)
    assert v.category == "amplifier"
    assert v.is_laser_input_independence is False
    assert v.is_laser_gain_linearity is False
    assert v.tests_agree


def test_laser_input_independent_output():
    # signal off -> output persists (~ same as on); slope ~ 0.
    v = classify(out_signal_off=9.0e-4, out_signal_on=1.0e-3, input_scaling_slope=0.02,
                 enhancement_dB=83.0, idler_over_sig=0.9)
    assert v.category == "lasing"
    assert v.is_laser_input_independence is True
    assert v.is_laser_gain_linearity is True
    assert v.tests_agree


def test_passive_no_enhancement():
    # stable (output tracks input) but no net parametric enhancement -> passive.
    v = classify(out_signal_off=2.0e-9, out_signal_on=1.0e-5, input_scaling_slope=1.0,
                 enhancement_dB=0.2, idler_over_sig=0.001)
    assert v.category == "passive"
    assert v.is_laser_input_independence is False


def test_trivial_floor_is_passive():
    # zero-modulation / pump-off floor: output proportional to input, no enhancement.
    v = classify(out_signal_off=1.0e-12, out_signal_on=1.0e-5, input_scaling_slope=1.0,
                 enhancement_dB=0.0, idler_over_sig=0.0)
    assert v.category == "passive"


def test_non_convergent_handled_consistently():
    v = classify(out_signal_off=None, out_signal_on=None, converged=False)
    assert v.category == "non-convergent"
    assert v.tests_agree  # not silently mis-handled as a stable point


def test_tests_disagree_is_flagged():
    # input-independence says laser (r_A high) but linearity says stable (slope ~ 1).
    v = classify(out_signal_off=8.0e-4, out_signal_on=1.0e-3, input_scaling_slope=0.95,
                 enhancement_dB=30.0, idler_over_sig=0.5)
    assert v.tests_agree is False     # the gate must reject a disagreement


def test_loss_invariance_of_r_A():
    # The SAME regime at three loss levels: scale BOTH on and off outputs by a common
    # loss factor (what higher R does). r_A is unchanged -> classification is invariant.
    base_off, base_on = 9.0e-4, 1.0e-3      # a self-sustained (laser) point, r_A=0.9
    cats = []
    for loss in (1.0, 10.0 ** (-24.9 / 20), 10.0 ** (-44.7 / 20)):  # R=0.5/2/5 ~ -0/-25/-45 dB
        v = classify(out_signal_off=base_off * loss, out_signal_on=base_on * loss,
                     input_scaling_slope=0.02, enhancement_dB=31.0, idler_over_sig=0.7)
        cats.append(v.category)
    assert cats == ["lasing", "lasing", "lasing"]   # invariant across 45 dB of loss


def test_high_gain_amplifier_not_lasing():
    # A high steady gain (large enhancement_dB) that STILL tracks the input is an
    # amplifier, NOT a laser -- the exact PSA failure (magnitude -> lasing) is avoided.
    v = classify(out_signal_off=3.0e-8, out_signal_on=1.0e-4, input_scaling_slope=1.0,
                 enhancement_dB=31.0, idler_over_sig=0.7)
    assert v.category == "amplifier"   # +31 dB enhancement but stable -> amplifier


