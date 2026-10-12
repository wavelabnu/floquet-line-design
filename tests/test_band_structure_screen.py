"""Tests for the band-structure screen + geometry refinement.

Two opt-in, backward-compatible engine controls that fix the stopband
mis-selection (the analytical ``g_mod_in`` proxy ranking Bloch-stopband cells to
the top, giving poor realized transducer gain):

* ``reject_signal_stopband`` -- build the walk-down pool from the A1-ranked screen
  keeping only Bloch-*passband* candidates (the A1 ranking is stopband-biased), and
  never SELECT a stopband candidate.
* ``refine`` -- Nelder-Mead-refine the top candidates' geometry against the
  passband-penalised out/in proxy (additive).

The crux being locked: at the VarA operating frequency (1.2 GHz) the *default*
``g_mod_in`` path selects a deep Bloch stopband (alpha_p ~ 1.1 Np/cell), while the
band screen selects a passband (alpha_p < 0.02) with a far better proxy. Both
controls default OFF -> the default path (and the A.3.1 anchor) are byte-identical.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.optimization.stage1_per_cell import StubCandidate
from tvtl.optimization.stage2_main_tl import (
    MAIN_PERIOD_M,
    bloch_attenuation_per_cell,
    static_cell_abcd,
)
from tvtl.optimization.topology_generator import linear_families
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.parity.self_pumped_design import (
    STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL,
    design_optimized_self_pumped,
    main_period_anchor_at_3p1GHz,
)

VAR_A = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5,
)
F_SIGNAL_HZ = 3.1e9
# The VarA operating-frequency case exhibiting the stopband mis-selection.
F_OP_HZ = 1.2e9
F_OP_PUMP_HZ = 2.4e9


def _alpha_p(result, f_signal_Hz: float) -> float:
    """Per-cell Bloch attenuation of a design result at its emitted period."""
    cand = StubCandidate(
        topology=result.topology, params=tuple(result.stub_params),
        a1_score=0.0, f0_Hz=f_signal_Hz,
    )
    return bloch_attenuation_per_cell(
        cand, VAR_A, result.z_tl_ohm, f0_Hz=f_signal_Hz,
        main_period_m=main_period_anchor_at_3p1GHz(f_signal_Hz),
    )


# --------------------------------------------------------------------------- #
# Stage-2 primitives: static_cell_abcd period param + bloch_attenuation_per_cell #
# --------------------------------------------------------------------------- #
def test_static_cell_abcd_default_period_is_byte_identical():
    """main_period_m=None reproduces the fixed-period cell exactly (every existing
    Stage-2/Stage-3 caller is unchanged)."""
    fam = next(f for f in linear_families() if f.n_params == 4)
    cand = StubCandidate(
        topology=fam, params=(1.5e-3, 5.0e-3, 1.2e-3, 4.0e-3),
        a1_score=0.0, f0_Hz=F_SIGNAL_HZ,
    )
    c_default = static_cell_abcd(cand, VAR_A, 50.0, f0_Hz=F_SIGNAL_HZ)
    c_explicit = static_cell_abcd(
        cand, VAR_A, 50.0, f0_Hz=F_SIGNAL_HZ, main_period_m=MAIN_PERIOD_M
    )
    assert np.array_equal(c_default, c_explicit)


def test_static_cell_abcd_period_changes_cell():
    """A different main-line period changes the cell ABCD (line lengths differ)."""
    fam = next(f for f in linear_families() if f.n_params == 2)
    cand = StubCandidate(
        topology=fam, params=(1.5e-3, 5.0e-3), a1_score=0.0, f0_Hz=F_OP_HZ,
    )
    c6 = static_cell_abcd(cand, VAR_A, 50.0, f0_Hz=F_OP_HZ)
    c16 = static_cell_abcd(
        cand, VAR_A, 50.0, f0_Hz=F_OP_HZ,
        main_period_m=main_period_anchor_at_3p1GHz(F_OP_HZ),
    )
    assert not np.allclose(c6, c16)


def test_bloch_attenuation_non_negative_and_finite():
    fam = next(f for f in linear_families() if f.n_params == 2)
    cand = StubCandidate(
        topology=fam, params=(1.5e-3, 5.0e-3), a1_score=0.0, f0_Hz=F_OP_HZ,
    )
    a = bloch_attenuation_per_cell(cand, VAR_A, 50.0, f0_Hz=F_OP_HZ)
    assert math.isfinite(a) and a >= 0.0


def test_threshold_constant_matches_net_gain_convention():
    """The propagating threshold is the net-gain propagating-threshold convention's 0.02 Np/cell value."""
    assert STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# Byte-identity: both controls default OFF -> default path unchanged           #
# --------------------------------------------------------------------------- #
def test_default_path_anchor_byte_identical():
    """coh_pk anchor recovery is unchanged by the new (default-off) controls."""
    a = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
        coh_pk_window_dB=(1.0, 5.0),
    )
    assert a.topology.key == "n2_single_short"
    assert a.z_tl_ohm == pytest.approx(42.22222222222222)
    assert a.analytical_coh_pk_dB == pytest.approx(4.879618686498491)
    # Every walk-down row's alpha_p is nan on the default (screen-off) path.
    assert all(math.isnan(e.bloch_alpha_p_np) for e in a.walkdown_trace)


def test_default_g_mod_in_selection_unchanged():
    """g_mod_in default selection at 3.1 GHz is unchanged (controls default off)."""
    g = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="g_mod_in",
    )
    assert g.topology.key == "n1_double_short"
    assert g.analytical_g_mod_in_dB == pytest.approx(2.0519, abs=1e-3)
    assert all(math.isnan(e.bloch_alpha_p_np) for e in g.walkdown_trace)


# --------------------------------------------------------------------------- #
# STEP 1 -- the band-structure screen fixes the stopband mis-selection         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def op_default():
    return design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_OP_HZ, f_pump_Hz=F_OP_PUMP_HZ,
        objective="g_mod_in",
    )


@pytest.fixture(scope="module")
def op_screened():
    return design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_OP_HZ, f_pump_Hz=F_OP_PUMP_HZ,
        objective="g_mod_in", reject_signal_stopband=True,
    )


def test_default_selects_stopband_at_operating_freq(op_default):
    """Reproduces the crux: the default g_mod_in path selects a deep Bloch
    stopband cell at the VarA operating frequency."""
    assert _alpha_p(op_default, F_OP_HZ) > STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL


def test_screen_selects_passband_with_better_proxy(op_default, op_screened):
    """The band screen selects a PASSBAND design with a far better proxy."""
    assert _alpha_p(op_screened, F_OP_HZ) <= STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL
    assert op_screened.analytical_g_mod_in_dB > op_default.analytical_g_mod_in_dB


def test_screen_records_alpha_p_and_selected_is_passband(op_screened):
    """alpha_p is recorded per realizable row, and the SELECTED row is passband."""
    realiz = [e for e in op_screened.walkdown_trace if e.stage2_realizable]
    assert realiz, "expected realizable rows under the band screen"
    assert all(math.isfinite(e.bloch_alpha_p_np) for e in realiz)
    sel = [e for e in op_screened.walkdown_trace if e.note.startswith("SELECTED")]
    assert len(sel) == 1
    assert sel[0].bloch_alpha_p_np <= STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL


def test_screen_off_is_byte_identical_to_default(op_default):
    """reject_signal_stopband=False reproduces the default path exactly."""
    same = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_OP_HZ, f_pump_Hz=F_OP_PUMP_HZ,
        objective="g_mod_in", reject_signal_stopband=False,
    )
    assert same.topology.key == op_default.topology.key
    assert same.stub_params == op_default.stub_params
    assert same.analytical_g_mod_in_dB == op_default.analytical_g_mod_in_dB


def test_selected_respects_custom_threshold():
    """The threshold is honoured: with a stricter cutoff the SELECTED design's
    per-cell attenuation stays at or below it (the gate guarantees passband)."""
    strict = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_OP_HZ, f_pump_Hz=F_OP_PUMP_HZ,
        objective="g_mod_in", reject_signal_stopband=True,
        stopband_alpha_threshold=0.01,
    )
    assert _alpha_p(strict, F_OP_HZ) <= 0.01
    sel = [e for e in strict.walkdown_trace if e.note.startswith("SELECTED")]
    assert sel and sel[0].bloch_alpha_p_np <= 0.01


# --------------------------------------------------------------------------- #
# STEP 2 -- Nelder-Mead refinement is additive and stays in the passband       #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def op_refined():
    return design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_OP_HZ, f_pump_Hz=F_OP_PUMP_HZ,
        objective="g_mod_in", reject_signal_stopband=True,
        refine=True, refine_top_n=2, refine_maxiter=25,
    )


def test_refine_is_additive(op_screened, op_refined):
    """Refinement never lowers the proxy vs the coarse band-screen selection."""
    assert op_refined.analytical_g_mod_in_dB >= op_screened.analytical_g_mod_in_dB - 1e-9


def test_refine_stays_in_passband(op_refined):
    assert _alpha_p(op_refined, F_OP_HZ) <= STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL


def test_refine_off_is_byte_identical(op_screened):
    """refine=False reproduces the coarse band-screen selection exactly."""
    same = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_OP_HZ, f_pump_Hz=F_OP_PUMP_HZ,
        objective="g_mod_in", reject_signal_stopband=True, refine=False,
    )
    assert same.stub_params == op_screened.stub_params
    assert same.analytical_g_mod_in_dB == op_screened.analytical_g_mod_in_dB


def test_refine_default_off_does_not_change_default(op_default):
    """refine defaults off, so the default g_mod_in path is unaffected by it."""
    same = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_OP_HZ, f_pump_Hz=F_OP_PUMP_HZ,
        objective="g_mod_in",
    )
    assert same.stub_params == op_default.stub_params


def test_refine_noop_under_coh_pk():
    """refine applies under g_mod_in only; under coh_pk it is a no-op."""
    base = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
        coh_pk_window_dB=(1.0, 5.0),
    )
    with_refine = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
        coh_pk_window_dB=(1.0, 5.0), refine=True, refine_maxiter=10,
    )
    assert base.stub_params == with_refine.stub_params
    assert base.analytical_coh_pk_dB == with_refine.analytical_coh_pk_dB


