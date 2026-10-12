"""Unit tests for the realistic varactor device model.

Covers the C-V law and its bias re-expansion, the package-divider pump
shielding (the L_pkg finding), the impedance / effective-modulation
helpers, the CP-folding charge polynomial (bit-for-bit with the paper-fidelity
canonical mechanism), and the ideal-mode reduction.
"""

from __future__ import annotations

import cmath
import math

import pytest

from tvtl.design.nonlinear_capacitor import smv1245_skyworks_spice_polynomial
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    Level10VaractorSpec,
    _q_poly_from_varactor,
)


# --------------------------------------------------------------------------- #
# C-V law and bias re-expansion                                               #
# --------------------------------------------------------------------------- #
def test_cv_law_matches_spice_form():
    v = RealisticVaractor.smv1245()
    for volt in (-2.0, -1.0, 0.0, 1.0, 3.0, 5.0):
        expected = 6.9e-12 / (1.0 + volt / 3.5) ** 1.7
        assert v.C(volt) == pytest.approx(expected, rel=1e-12)


def test_cv_zero_bias_is_cj0():
    v = RealisticVaractor.smv1245()
    assert v.C(0.0) == pytest.approx(6.9e-12, rel=1e-12)


def test_reverse_bias_lowers_capacitance():
    v = RealisticVaractor.smv1245()
    # reverse bias is v > 0 in this convention; C must decrease monotonically
    cs = [v.C(volt) for volt in (-1.0, 0.0, 1.0, 3.0, 5.0)]
    assert all(a > b for a, b in zip(cs, cs[1:]))


def test_bias_sweep_reproduces_round2e_probe_C0():
    # DC-bias probe table (C0 incl. nothing extra; junction only).
    expected = {-2.0: 29.13, -1.0: 12.23, 0.0: 6.90, 1.0: 4.50, 2.0: 3.20, 5.0: 1.53}
    for vb, pf in expected.items():
        v = RealisticVaractor.smv1245(V_bias_v=vb, CP_f=0.0, L_pkg_h=0.0, R_series_ohm=0.0)
        # junction static C at the bias point (CP=0 here)
        assert v.C(vb) * 1e12 == pytest.approx(pf, abs=0.02)


def test_dC_dV_matches_finite_difference():
    v = RealisticVaractor.smv1245(V_bias_v=1.0)
    h = 1e-6
    fd = (v.C(1.0 + h) - v.C(1.0 - h)) / (2.0 * h)
    assert v.dC_dV_f_per_v == pytest.approx(fd, rel=1e-5)
    assert v.dC_dV_f_per_v < 0.0  # reverse-bias varactor


def test_C_undefined_past_barrier():
    v = RealisticVaractor.smv1245()
    with pytest.raises(ValueError):
        v.C(-3.5)  # v = -VJ
    with pytest.raises(ValueError):
        RealisticVaractor.smv1245(V_bias_v=-3.5)  # bias at the barrier


# --------------------------------------------------------------------------- #
# C0 (incl. CP) and the package divider / m_effective                          #
# --------------------------------------------------------------------------- #
def test_C0_includes_CP():
    v = RealisticVaractor.smv1245()  # CJ0=6.9, CP=0.47
    assert v.C0_f == pytest.approx(7.37e-12, rel=1e-12)


def test_pump_divider_reproduces_lpkg_shielding():
    # L_pkg probe at f_pump=6.2 GHz, L_pkg=1.7 nH: the junction sees
    # ~6-8% of the branch-top pump. Bracket generously (C0/divider modelling).
    v = RealisticVaractor.smv1245()
    mag = abs(v.pump_divider(6.2e9))
    assert 0.03 < mag < 0.15, mag


def test_pump_divider_rises_as_lpkg_falls():
    f = 6.2e9
    mags = [
        abs(RealisticVaractor.smv1245(L_pkg_h=L).pump_divider(f))
        for L in (1.7e-9, 1.0e-9, 0.3e-9, 0.1e-9)
    ]
    assert all(a < b for a, b in zip(mags, mags[1:]))  # monotonically toward 1


def test_m_effective_is_shielded_then_recovers():
    v = RealisticVaractor.smv1245()
    assert v.m_effective(6.2e9, 0.3) < 0.3  # shielded below the branch depth
    low = RealisticVaractor.smv1245(L_pkg_h=1e-15, R_series_ohm=0.0)
    assert low.m_effective(6.2e9, 0.3) == pytest.approx(0.3, abs=2e-3)


def test_Z_branch_form():
    v = RealisticVaractor.smv1245()
    f = 3.1e9
    omega = 2.0 * math.pi * f
    z = v.Z_branch(f)
    expected = 1.5 + 1j * omega * 1.7e-9 + 1.0 / (1j * omega * v.C0_f)
    assert z == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# Charge polynomial + CP fold (bit-for-bit with the paper-fidelity mechanism)  #
# --------------------------------------------------------------------------- #
def test_charge_polynomial_cp_fold_matches_paper_canonical_bitforbit():
    # The paper-fidelity canonical uses analytical_c0_f=7.0 pF (override) and
    # RS=2.0; the realistic varactor reproduces its q_poly bit-for-bit when
    # configured identically (the c0_static_override knob).
    paper = Level10VaractorSpec()  # cj0=6.9, vj=3.5, m=1.7, cp=0.47, analytical_c0=7.0
    ref = _q_poly_from_varactor(paper)
    v = RealisticVaractor(
        CJ0_f=6.9e-12, VJ_v=3.5, M=1.7, CP_f=0.47e-12,
        L_pkg_h=1.7e-9, R_series_ohm=2.0, V_bias_v=0.0,
        c0_static_override_f=7.0e-12,
    )
    got = v.charge_polynomial().charge_polynomial_coefficients
    assert tuple(got) == tuple(ref)


def test_charge_polynomial_cp_fold_is_q1_plus_cp():
    # Without override, q[1] = junction CJ0 + CP (true datasheet device).
    v = RealisticVaractor.smv1245()
    q = v.charge_polynomial().charge_polynomial_coefficients
    assert q[0] == 0.0
    assert q[1] == pytest.approx(6.9e-12 + 0.47e-12, rel=1e-12)


def test_charge_polynomial_bias_reexpansion():
    v = RealisticVaractor.smv1245(V_bias_v=2.0)
    q = v.charge_polynomial().charge_polynomial_coefficients
    # q[1] = junction C(+2) + CP
    assert q[1] == pytest.approx(v.C(2.0) + 0.47e-12, rel=1e-12)


# --------------------------------------------------------------------------- #
# Ideal-mode reduction                                                         #
# --------------------------------------------------------------------------- #
def test_ideal_constant_capacitance():
    v = RealisticVaractor.ideal(0.3e-12)
    assert v.C(0.0) == v.C(5.0) == v.C(-1.0) == 0.3e-12
    assert v.dC_dV_f_per_v == 0.0
    assert v.C0_f == 0.3e-12


def test_ideal_divider_unity_exact():
    v = RealisticVaractor.ideal(0.3e-12)
    assert v.pump_divider(6.2e9) == complex(1.0, 0.0)  # EXACT
    assert v.m_effective(6.2e9, 0.3) == 0.3  # EXACT


def test_ideal_charge_polynomial_linear():
    v = RealisticVaractor.ideal(0.3e-12)
    q = v.charge_polynomial().charge_polynomial_coefficients
    assert q[0] == 0.0
    assert q[1] == 0.3e-12
    assert all(qi == 0.0 for qi in q[2:])


def test_ideal_effective_coeffs_pure_cosine():
    v = RealisticVaractor.ideal(0.3e-12)
    cc = v.effective_capacitance_coeffs(6.2e9, 0.3, phi=0.0)
    assert len(cc) == 2  # only DC + fundamental
    assert cc[0] == 0.3e-12
    assert cc[1] == 0.5 * 0.3 * 0.3e-12 + 0j  # EXACT cosine amplitude


# --------------------------------------------------------------------------- #
# Effective C-V spectrum (curvature)                                           #
# --------------------------------------------------------------------------- #
def test_effective_coeffs_fundamental_pinned():
    v = RealisticVaractor.smv1245()
    cc = v.effective_capacitance_coeffs(6.2e9, 0.3, phi=0.0)
    m_eff = v.m_effective(6.2e9, 0.3)
    # the fundamental equals 0.5*m_eff*C0 up to the small k>=3 curvature add
    assert cc[1].real == pytest.approx(0.5 * m_eff * v.C0_f, rel=1e-2)
    # curvature produces non-zero higher bands
    assert abs(cc[2]) > 0.0


def test_effective_coeffs_phase_propagates_per_band():
    v = RealisticVaractor.smv1245()
    phi = 0.4
    cc = v.effective_capacitance_coeffs(6.2e9, 0.3, phi=phi)
    # band d carries e^{i d phi}
    assert cmath.phase(cc[1]) == pytest.approx(phi, abs=1e-9)
    assert cmath.phase(cc[2]) == pytest.approx(2.0 * phi, abs=1e-9)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def test_rejects_nonpositive_cj0():
    with pytest.raises(ValueError):
        RealisticVaractor(CJ0_f=0.0, VJ_v=3.5, M=1.7)


def test_rejects_negative_parasitics():
    with pytest.raises(ValueError):
        RealisticVaractor(CJ0_f=1e-12, VJ_v=3.5, M=1.7, L_pkg_h=-1e-9)
    with pytest.raises(ValueError):
        RealisticVaractor(CJ0_f=1e-12, VJ_v=3.5, M=1.7, R_series_ohm=-1.0)


def test_control_port_capacitance_coeffs():
    v = RealisticVaractor.smv1245()
    a = v.control_port_capacitance_coeffs()
    assert a[0] == pytest.approx(v.C0_f, rel=1e-12)       # static incl CP
    assert a[1] == pytest.approx(v.dC_dV_f_per_v, rel=1e-12)  # dC/dV
    assert len(a) == 6  # a0..a5


def test_control_port_capacitance_coeffs_rejects_ideal():
    with pytest.raises(ValueError):
        RealisticVaractor.ideal(0.3e-12).control_port_capacitance_coeffs()


def test_coeffs_from_junction_vpump_dc_and_fundamental():
    v = RealisticVaractor.smv1245()
    cc = v.capacitance_coeffs_from_junction_vpump(0.3, phi=0.0)
    # DC band carries the curvature shift but stays near C0
    assert cc[0].real == pytest.approx(v.C0_f, rel=0.05)
    # fundamental = a1*Vp/2 to leading order
    assert cc[1].real == pytest.approx(v.dC_dV_f_per_v * 0.3 / 2.0, rel=0.05)
    assert cc[0].imag == 0.0


def test_coeffs_from_junction_vpump_zero_pump_is_static():
    v = RealisticVaractor.smv1245()
    cc = v.capacitance_coeffs_from_junction_vpump(0.0)
    assert cc[0].real == pytest.approx(v.C0_f, rel=1e-12)
    assert all(abs(c) == 0.0 for c in cc[1:])


def test_coeffs_from_junction_vpump_rejects_ideal():
    with pytest.raises(ValueError):
        RealisticVaractor.ideal(0.3e-12).capacitance_coeffs_from_junction_vpump(0.3)


def test_bias_expansion_bitforbit_at_zero():
    # The smv1245 helper used by charge_polynomial is bit-for-bit at v=0.
    a = smv1245_skyworks_spice_polynomial(cj0_f=6.9e-12, vj_v=3.5, m_grade=1.7, order=5)
    b = smv1245_skyworks_spice_polynomial(
        cj0_f=6.9e-12, vj_v=3.5, m_grade=1.7, order=5, v_bias_v=0.0
    )
    assert a.charge_polynomial_coefficients == b.charge_polynomial_coefficients


