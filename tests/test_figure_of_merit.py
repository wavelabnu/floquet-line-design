"""Unit tests for the Stage-1 figure of merit (A1 + A2), base Python."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.optimization.figure_of_merit import (
    ClosedFormA1,
    ClosedFormA1SelfPumped,
    LorentzianA1,
    SingleCellFloquetA2,
    a1_factors,
    m_branch_from_source_pump_dBm,
    standardized_drive_voltage,
)
from tvtl.optimization.stage1_per_cell import coarse_screen
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    ViaConfig,
    linear_families,
    sample_family,
)
from tvtl.varactors import RealisticVaractor

_VAR = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
)
_IDEAL = RealisticVaractor.ideal(0.3e-12)


def test_standardized_drive_voltage():
    # Voc = 2 sqrt(2 * 50 * 10^((P-30)/10)); -3 dBm -> ~0.4477 V
    assert standardized_drive_voltage(-3.0) == pytest.approx(0.4477, abs=1e-3)
    assert standardized_drive_voltage(10.0) == pytest.approx(2.0, abs=1e-3)


def test_a1_factors_ranges():
    fam = TopologyFamily(3, ViaConfig.DOUBLE, Termination.SHORT)
    p = next(sample_family(fam, 64, substrate=ROGERS_4003C, seed=3))
    m_eff, kappa, x = a1_factors(fam, p, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
    assert m_eff > 0.0
    assert 0.0 <= kappa <= 1.0
    assert x >= 0.0


def test_a1_intrinsic_cap():
    """A1 = m_eff * kappa * rho with kappa,rho in [0,1] => A1 <= m_eff (no divergence)."""
    a1 = ClosedFormA1()
    for fam in (
        TopologyFamily(2, ViaConfig.NONE, Termination.OPEN),
        TopologyFamily(3, ViaConfig.DOUBLE, Termination.SHORT),
        TopologyFamily(2, ViaConfig.SINGLE, Termination.COMBINED),
    ):
        for p in list(sample_family(fam, 48, substrate=ROGERS_4003C, seed=4))[:10]:
            m_eff, _kappa, _x = a1_factors(
                fam, p, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C
            )
            s = a1.score(fam, p, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
            assert 0.0 <= s <= m_eff + 1e-12
            assert math.isfinite(s)


def test_sigma_widens_guard_for_offresonance():
    """Larger sigma -> higher rho for an off-resonance candidate (guard direction)."""
    fam = TopologyFamily(2, ViaConfig.NONE, Termination.OPEN)
    # pick a candidate with x far from delta* (off resonance)
    cand = None
    for p in sample_family(fam, 128, substrate=ROGERS_4003C, seed=5):
        _m, _k, x = a1_factors(fam, p, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
        if x > 2.0:
            cand = p
            break
    assert cand is not None
    narrow = ClosedFormA1(sigma=0.35).score(fam, cand, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
    wide = ClosedFormA1(sigma=0.65).score(fam, cand, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
    assert wide > narrow  # broader guard rewards off-resonance more


def test_lorentzian_heavy_tail_no_underflow():
    """Form B (Lorentzian) keeps off-resonance A1 polynomially small (no denormal
    underflow) and within the intrinsic [0, m_eff] cap."""
    lor = LorentzianA1(x0=0.25, gamma=0.20)
    fam = TopologyFamily(2, ViaConfig.NONE, Termination.OPEN)
    # find a far-off-resonance candidate (large x)
    cand = None
    for p in sample_family(fam, 128, substrate=ROGERS_4003C, seed=5):
        _m, _k, x = a1_factors(fam, p, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
        if x > 5.0:
            cand = p
            break
    assert cand is not None
    m_eff, _k, _x = a1_factors(fam, cand, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
    s_lor = lor.score(fam, cand, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
    s_gauss = ClosedFormA1().score(fam, cand, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
    # heavy tail: Lorentzian keeps a far-off candidate well above denormal,
    # and far above the Gaussian (which underflows there)
    assert 0.0 < s_lor <= m_eff + 1e-12
    assert s_lor > s_gauss
    assert s_lor > 1e-12  # not underflowed


def test_lorentzian_shares_interface():
    fam = TopologyFamily(2, ViaConfig.SINGLE, Termination.SHORT)
    p = next(sample_family(fam, 32, substrate=ROGERS_4003C, seed=8))
    kwargs = dict(f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C, drive_dBm=-3.0)
    s = LorentzianA1(x0=0.25, gamma=0.20).score(fam, p, **kwargs)
    assert math.isfinite(s) and s >= 0.0


def test_a1_requires_curved_varactor():
    fam = TopologyFamily(1, ViaConfig.SINGLE, Termination.SHORT)
    p = next(sample_family(fam, 32, substrate=ROGERS_4003C, seed=6))
    with pytest.raises(ValueError):
        a1_factors(fam, p, f0_Hz=3.1e9, varactor=_IDEAL, substrate=ROGERS_4003C)


def test_a2_linear_finite_and_combined_raises():
    a2 = SingleCellFloquetA2()
    lin = TopologyFamily(3, ViaConfig.DOUBLE, Termination.SHORT)
    p = next(sample_family(lin, 32, substrate=ROGERS_4003C, seed=7))
    s = a2.score(lin, p, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
    assert math.isfinite(s)
    comb = TopologyFamily(2, ViaConfig.SINGLE, Termination.COMBINED)
    pc = next(sample_family(comb, 32, substrate=ROGERS_4003C, seed=7))
    with pytest.raises(NotImplementedError):
        a2.score(comb, pc, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)


def test_a1_a2_share_interface():
    """Both scorers accept the same call signature (interchangeability)."""
    fam = TopologyFamily(2, ViaConfig.SINGLE, Termination.SHORT)
    p = next(sample_family(fam, 32, substrate=ROGERS_4003C, seed=8))
    kwargs = dict(f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C, drive_dBm=-3.0)
    s1 = ClosedFormA1().score(fam, p, **kwargs)
    s2 = SingleCellFloquetA2().score(fam, p, **kwargs)
    assert math.isfinite(s1) and math.isfinite(s2)


# --------------------------------------------------------------------------- #
# Self-pumped A1                                                               #
# --------------------------------------------------------------------------- #
def test_m_branch_from_source_pump_dBm_monotone():
    """The provisional m_branch convention is monotone in pump power."""
    a = m_branch_from_source_pump_dBm(-10.0, _VAR)
    b = m_branch_from_source_pump_dBm(0.0, _VAR)
    c = m_branch_from_source_pump_dBm(10.0, _VAR)
    assert 0.0 < a < b < c
    # V scales as sqrt(P), so 10 dB more pump = ~sqrt(10)x more depth.
    assert (b / a) == pytest.approx(math.sqrt(10.0), rel=1e-9)


def test_closed_form_a1_self_pumped_satisfies_protocol():
    """ClosedFormA1SelfPumped is a drop-in for LorentzianA1 (Stage-1 protocol)."""
    fam = TopologyFamily(2, ViaConfig.SINGLE, Termination.SHORT)
    p = next(sample_family(fam, 32, substrate=ROGERS_4003C, seed=8))
    fom = ClosedFormA1SelfPumped(f_pump_Hz=6.2e9, pump_power_dBm=16.0)
    s = fom.score(fam, p, f0_Hz=3.1e9, varactor=_VAR, substrate=ROGERS_4003C)
    assert math.isfinite(s) and s > 0.0


def _spearman(a, b):
    """Spearman rank correlation, broken ties averaged."""
    n = len(a)
    assert n == len(b) and n >= 2

    def _ranks(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = _ranks(a), _ranks(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den_a = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / (den_a * den_b)


def test_self_pumped_rank_equivalence_to_lorentzian():
    """LorentzianA1 (control-port) and ClosedFormA1SelfPumped produce identical
    Sobol rankings at fixed varactor.

    The control-port and self-pumped A1 scorers share kappa_branch and
    rho_lorentzian (signal-path-only, regime-agnostic) and differ only in
    the m_eff factor, which is a scalar invariant over any geometry sweep
    at fixed varactor. Therefore screens must produce identically-ordered
    candidate rankings; the absolute A1 magnitudes differ only by a
    constant multiplicative scale factor. This test locks that property.

    Empirical reference (A.3.1, 2157 candidates): Spearman = 1.000000,
    ratio constant at 9.32x to floating-point precision.
    """
    short = tuple(f for f in linear_families() if f.termination == Termination.SHORT)
    cp = coarse_screen(
        3.1e9, _VAR, fom=LorentzianA1(),
        families=list(short), n_samples=24, seed=0, top_k=5, drive_dBm=-3.0,
    )
    sp = coarse_screen(
        3.1e9, _VAR,
        fom=ClosedFormA1SelfPumped(f_pump_Hz=6.2e9, pump_power_dBm=16.0),
        families=list(short), n_samples=24, seed=0, top_k=5, drive_dBm=-3.0,
    )

    # Same Sobol seed + same families => same (width, length) tuples scored
    # under both scorers.
    assert len(cp.scored) == len(sp.scored) > 0

    def _key(c):
        return (c.topology.key, float(c.params[0]), float(c.params[1]))

    cp_by_key = {_key(c): c.a1_score for c in cp.scored}
    sp_by_key = {_key(c): c.a1_score for c in sp.scored}
    common = sorted(set(cp_by_key.keys()) & set(sp_by_key.keys()))
    assert len(common) == len(cp.scored)  # same candidates evaluated

    cp_scores = [cp_by_key[k] for k in common]
    sp_scores = [sp_by_key[k] for k in common]

    # 1) Rank invariance: Spearman correlation effectively unity.
    rho = _spearman(cp_scores, sp_scores)
    assert rho >= 0.9999, f"Spearman correlation {rho:.6f} below 0.9999"

    # 2) Constant multiplicative scale factor across all candidates.
    ratios = [s / c for s, c in zip(sp_scores, cp_scores) if c > 0]
    assert ratios
    rmin, rmax = min(ratios), max(ratios)
    # Floating-point tightness: ratio spread should be <<1 dB.
    assert (rmax / rmin) - 1.0 < 1.0e-9, (
        f"sp/cp ratio spread {(rmax/rmin) - 1.0:.3e} exceeds 1e-9; "
        f"the regime-dependent factor is supposed to be invariant"
    )

    # 3) Top-K identity (set equality; floating-point may shuffle ties).
    cp_top_keys = {_key(c) for c in cp.linear_topk}
    sp_top_keys = {_key(c) for c in sp.linear_topk}
    assert cp_top_keys == sp_top_keys, (
        f"Top-K sets differ: cp_only={cp_top_keys - sp_top_keys}, "
        f"sp_only={sp_top_keys - cp_top_keys}"
    )


