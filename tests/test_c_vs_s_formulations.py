"""Gating tests for the C-linear / S-linear / reciprocal modulated-element
formulations exposed on :func:`level10_realistic_param_gain`.

Three contracts (E1):

* **default zero-drift** -- omitting ``modulation_formulation`` is byte-identical
  to ``"capacitance"`` (the historical path); the rest of the suite locks the
  A.3.1 anchor and the deck byte-identity, unchanged by this additive kwarg.
* **first-order agreement** -- the capacitance-linear and elastance-linear
  models are equal to first order in the modulation depth (the residual is
  second-order, vanishing as ``m -> 0``); the "reciprocal" route (exact
  ``S(t) = 1/C(t)`` of the same device) reproduces the capacitance path to
  numerical precision, bounded only by the finite harmonic truncation.
* **divergence pin** -- at deep modulation (``m = 0.5``) the C and S
  formulations diverge by several dB; that margin is pinned here as the
  executable sub-contribution.

The formulation comparison is done on the ideal-varactor canonical (unity
package divider, single-tone C-V), which isolates the solver-formulation
difference from the package/curvature physics of a catalog part.
"""

from __future__ import annotations

import pytest

from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.analytical import level10_realistic_param_gain
from tvtl_ads_builder_v2.parity import make_level10_ideal_c_family


def _spec():
    return make_level10_ideal_c_family(
        "k0p4", n_cells=15, f_signal_Hz=3.1e9, f_pump_Hz=6.2e9
    )


def _ideal():
    return RealisticVaractor.ideal(0.3e-12)


def _peak(spec, varactor, *, N=6, **kw):
    return level10_realistic_param_gain(
        spec, varactor, truncation_N=N, **kw
    )["coherent_peak_dB"]


# --------------------------------------------------------------------------- #
# (a) default zero-drift: the additive kwarg default is byte-identical.        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m", [0.01, 0.1, 0.3])
def test_default_formulation_is_byte_identical(m):
    spec, dev = _spec(), _ideal()
    a = level10_realistic_param_gain(spec, dev, m_branch_pump=m)
    b = level10_realistic_param_gain(
        spec, dev, m_branch_pump=m, modulation_formulation="capacitance"
    )
    for key in ("coherent_peak_dB", "coherent_min_dB", "param_gain_dB",
                "abs_gain_on_dB", "abs_gain_off_dB", "depth_effective"):
        assert a[key] == b[key], f"default drift on {key}: {a[key]!r} != {b[key]!r}"


# --------------------------------------------------------------------------- #
# (b) first-order agreement + reciprocal solver identity.                      #
# --------------------------------------------------------------------------- #
def test_capacitance_elastance_agree_to_first_order():
    spec, dev = _spec(), _ideal()
    d01 = abs(_peak(spec, dev, m_branch_pump=0.01)
              - _peak(spec, dev, m_s_branch=0.01, modulation_formulation="elastance"))
    d05 = abs(_peak(spec, dev, m_branch_pump=0.05)
              - _peak(spec, dev, m_s_branch=0.05, modulation_formulation="elastance"))
    # First order: the O(m) idler terms cancel; residual is small and shrinks
    # toward zero faster than linearly (second-order) as m -> 0.
    assert d01 < 1.0e-4          # within 1e-4 dB at m = 0.01
    assert d05 < 1.0e-2
    assert d05 > 8.0 * d01       # super-linear shrink -> first-order agreement


@pytest.mark.parametrize("m", [0.01, 0.05, 0.1])
def test_reciprocal_matches_capacitance_machine_precision_N6(m):
    # Same physical device (C(t)) solved via the elastance S(t)=1/C(t) route:
    # a solver identity, machine-precision at moderate depth for N = 6.
    spec, dev = _spec(), _ideal()
    c = _peak(spec, dev, m_branch_pump=m)
    r = _peak(spec, dev, m_branch_pump=m, modulation_formulation="reciprocal")
    assert abs(c - r) < 1.0e-9


@pytest.mark.parametrize("m", [0.01, 0.1, 0.3, 0.5])
def test_reciprocal_matches_capacitance_all_depths_deep_truncation(m):
    # Across ALL depths the reciprocal route reproduces the capacitance path to
    # <= 1e-9 dB once the harmonic basis is deep enough (the residual is finite-
    # truncation error, tightening with N; N = 10 clears the full depth grid).
    spec, dev = _spec(), _ideal()
    c = _peak(spec, dev, N=10, m_branch_pump=m)
    r = _peak(spec, dev, N=10, m_branch_pump=m, modulation_formulation="reciprocal")
    assert abs(c - r) < 1.0e-9


# --------------------------------------------------------------------------- #
# (c) divergence pin (the paper's executable sub-contribution).                #
# --------------------------------------------------------------------------- #
def test_c_vs_s_divergence_pinned_at_deep_modulation():
    spec, dev = _spec(), _ideal()
    c = _peak(spec, dev, m_branch_pump=0.5)
    s = _peak(spec, dev, m_s_branch=0.5, modulation_formulation="elastance")
    margin = c - s
    # At m = 0.5 the capacitance-linear and elastance-linear models are NOT the
    # same device; they diverge by several dB. Pinned so a regression is caught.
    assert margin == pytest.approx(-7.7813, abs=0.05)
    assert abs(margin) > 3.0


# --------------------------------------------------------------------------- #
# depth-semantics validation: exactly one drive per formulation.               #
# --------------------------------------------------------------------------- #
def test_depth_semantics_errors():
    spec, dev = _spec(), _ideal()
    with pytest.raises(ValueError):  # m_s without elastance
        level10_realistic_param_gain(spec, dev, m_s_branch=0.1)
    with pytest.raises(ValueError):  # both m_c and m_s
        level10_realistic_param_gain(spec, dev, m_branch_pump=0.1, m_s_branch=0.1)
    with pytest.raises(ValueError):  # m_c with elastance
        level10_realistic_param_gain(
            spec, dev, m_branch_pump=0.1, modulation_formulation="elastance")
    with pytest.raises(ValueError):  # elastance needs m_s
        level10_realistic_param_gain(spec, dev, modulation_formulation="elastance")
    with pytest.raises(ValueError):  # unknown formulation
        level10_realistic_param_gain(
            spec, dev, m_branch_pump=0.1, modulation_formulation="bogus")
