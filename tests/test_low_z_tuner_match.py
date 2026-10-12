"""Opt-in low-Z tuner-match Stage-2 policy.

Locks two properties of the ``match_low_z_with_tuner`` flag on
``design_optimized_self_pumped``:

1. Default OFF preserves the historical Stage-2 gate: at a low-Z point (5 GHz)
   every candidate is dropped as not-realizable (no candidate passes Stage-2),
   so the anchor / every existing caller is byte-identical.
2. Flag ON keeps low-Z candidates whose cascade edge is tuner-matchable -- they
   now pass Stage-2 (stage2_realizable entries appear in the walk-down trace).

The A-3.1 anchor reproduction itself is guarded by
``test_self_pumped_design.py``; this file only exercises the new flag.
"""

from __future__ import annotations

from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.parity.self_pumped_design import (
    NoSelfPumpedCandidateError,
    design_optimized_self_pumped,
)

VAR_A = RealisticVaractor(CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0,
                          L_pkg_h=0.3e-9, R_series_ohm=0.5)
# Small, fast screen; the 5 GHz point is uniformly low-Z (all candidates fail
# the Bloch-to-50 main-line gate).
_KW = dict(
    varactor=VAR_A, f_signal_Hz=5.0e9, coh_pk_window_dB=(-1.0e6, 1.0e6),
    sobol_n_samples=64, walk_down_top_k=6, matching_loss_threshold_dB=3.5,
)


def _trace_of(**kw):
    try:
        r = design_optimized_self_pumped(**_KW, **kw)
        return r.walkdown_trace
    except NoSelfPumpedCandidateError as exc:
        return exc.walkdown_trace


def _stage2_count(trace):
    return sum(1 for e in trace if e.stage2_realizable)


def test_default_equals_explicit_flag_off():
    """The flag defaults to OFF -> default call == match_low_z_with_tuner=False."""
    d = _trace_of()
    off = _trace_of(match_low_z_with_tuner=False)
    assert len(d) == len(off)
    assert _stage2_count(d) == _stage2_count(off)


def test_flag_on_admits_strictly_more_low_z_past_stage2():
    """At the low-Z 5 GHz point the tuner policy keeps candidates the Bloch-to-50
    gate drops -> strictly more pass Stage-2 with the flag on."""
    off = _stage2_count(_trace_of(match_low_z_with_tuner=False))
    on = _stage2_count(_trace_of(match_low_z_with_tuner=True))
    assert on > off
    assert on > 0


def test_flag_on_keeps_low_z_candidates():
    assert _stage2_count(_trace_of(match_low_z_with_tuner=True)) > 0


