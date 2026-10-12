"""Tests for the expanded search (length + pump + stability).

Covers the pure-Python search structure and the in-loop stability constraint
without ADS (the regime check is injected as a stub). The ADS-realised
validation lives in ``tests/parity/test_expanded_search_adspy.py``.

The crux being locked here: an expanded search ranked by the analytical
``g_mod_in`` proxy climbs toward the instability edge, so the stability
constraint must *reject* oscillator-regime finalists and never return one as the
optimum. The stub regime check emulates the measured physics
(low pump -> amplifier, band-edge -> boundary, high pump -> non-convergent /
lasing) so the constraint logic is exercised deterministically.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.analysis.amplification_check import (
    AMPLIFICATION,
    INDETERMINATE,
    NON_CONVERGENT,
    OSCILLATION,
)
from tvtl_ads_builder_v2.parity.expanded_search import (
    REGIME_AMPLIFIER,
    REGIME_BOUNDARY,
    REGIME_INDETERMINATE,
    REGIME_NON_CONVERGENT,
    REGIME_OSCILLATOR,
    STABLE_OR_BOUNDARY,
    default_pump_ladder_dBm,
    design_optimized_self_pumped_search,
    main_line_guided_wavelength_m,
    n_cells_grid_for_length,
    regime_label_from_check,
)
from tvtl_ads_builder_v2.parity.self_pumped_design import (
    design_optimized_self_pumped,
    main_period_anchor_at_3p1GHz,
)

VAR_A = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5,
)
F_SIGNAL_HZ = 3.1e9
F_PUMP_HZ = 6.2e9


# --------------------------------------------------------------------------- #
# STEP 1 -- length (N) grid bounded by the distributed-gain physics            #
# --------------------------------------------------------------------------- #
def test_guided_wavelength_decreases_with_frequency():
    """lambda_g ~ 1/f for a fixed-Z line."""
    lam_low = main_line_guided_wavelength_m(1.2e9)
    lam_high = main_line_guided_wavelength_m(8.0e9)
    assert lam_low > lam_high > 0.0
    # ~ inverse-f scaling (eps_eff drifts mildly with frequency)
    assert lam_low / lam_high == pytest.approx(8.0 / 1.2, rel=0.1)


def test_n_cells_grid_spans_target_wavelengths():
    """Default grid spans ~1..4 guided wavelengths with the anchor N=15 inside."""
    grid = n_cells_grid_for_length(F_SIGNAL_HZ)
    assert len(grid) == 5
    assert grid == tuple(sorted(grid))           # ascending, de-duped
    period = main_period_anchor_at_3p1GHz(F_SIGNAL_HZ)
    lam_g = main_line_guided_wavelength_m(F_SIGNAL_HZ)
    lo_lambda = grid[0] * period / lam_g
    hi_lambda = grid[-1] * period / lam_g
    assert lo_lambda == pytest.approx(1.0, abs=0.2)   # >= ~1 wavelength
    assert hi_lambda == pytest.approx(4.0, abs=0.2)   # <= ~4 wavelengths
    # Anchor cell count (15 ~ 1.55 lambda) lies inside the span.
    assert grid[0] <= 15 <= grid[-1]


def test_n_cells_grid_frequency_invariant_under_anchored_period():
    """With the anchored period the per-cell electrical length is constant, so
    the wavelength-targeted N grid is (essentially) frequency-independent."""
    g31 = n_cells_grid_for_length(3.1e9)
    g12 = n_cells_grid_for_length(1.2e9)
    assert g31 == g12


def test_n_cells_grid_custom_period_adapts():
    """A custom (non-anchored) period shifts the grid: a longer per-cell period
    needs fewer cells per wavelength."""
    lam_g = main_line_guided_wavelength_m(F_SIGNAL_HZ)
    long_period = lam_g / 4.0  # 4 cells per wavelength
    grid = n_cells_grid_for_length(
        F_SIGNAL_HZ, main_period_m=long_period, target_lambda=(1.0, 4.0), n_points=4
    )
    assert grid[0] == pytest.approx(4, abs=1)
    assert grid[-1] == pytest.approx(16, abs=1)


def test_n_cells_grid_absolute_clamp():
    """n_abs_bounds is a hard backstop on the derived grid."""
    grid = n_cells_grid_for_length(
        F_SIGNAL_HZ, target_lambda=(0.1, 100.0), n_points=5, n_abs_bounds=(6, 30)
    )
    assert min(grid) >= 6
    assert max(grid) <= 30


def test_n_cells_grid_rejects_bad_target():
    with pytest.raises(ValueError):
        n_cells_grid_for_length(F_SIGNAL_HZ, target_lambda=(4.0, 1.0))


# --------------------------------------------------------------------------- #
# STEP 2 -- pump ladder + regime mapping                                       #
# --------------------------------------------------------------------------- #
def test_default_pump_ladder_folds_calibration():
    base = default_pump_ladder_dBm()
    assert base == (5.0, 9.0, 13.0, 17.0)
    withcal = default_pump_ladder_dBm(1.6697)
    assert withcal[0] == pytest.approx(1.6697)
    assert withcal[1:] == (5.0, 9.0, 13.0, 17.0)
    # idempotent / sorted-unique
    assert default_pump_ladder_dBm(9.0) == (5.0, 9.0, 13.0, 17.0)


def _check(classification, *, probed=True, agree=True, g=0.0):
    return SimpleNamespace(
        classification=classification,
        signal_off_converged=probed,
        tests_agree=agree,
        g_mod_in_dB=g,
    )


def test_regime_label_mapping():
    assert regime_label_from_check(_check(AMPLIFICATION)) == REGIME_AMPLIFIER
    assert regime_label_from_check(_check(OSCILLATION)) == REGIME_OSCILLATOR
    assert regime_label_from_check(_check(NON_CONVERGENT)) == REGIME_NON_CONVERGENT
    # INDETERMINATE while probed + tests disagree => band-edge BOUNDARY
    assert (
        regime_label_from_check(_check(INDETERMINATE, probed=True, agree=False))
        == REGIME_BOUNDARY
    )
    # INDETERMINATE unprobed => cannot certify => indeterminate
    assert (
        regime_label_from_check(_check(INDETERMINATE, probed=False, agree=False))
        == REGIME_INDETERMINATE
    )


def test_admissibility_sets():
    assert REGIME_AMPLIFIER in STABLE_OR_BOUNDARY
    assert REGIME_BOUNDARY in STABLE_OR_BOUNDARY
    for r in (REGIME_OSCILLATOR, REGIME_NON_CONVERGENT, REGIME_INDETERMINATE):
        assert r not in STABLE_OR_BOUNDARY


# --------------------------------------------------------------------------- #
# Backward compatibility -- pump_dBm_override default path is byte-identical    #
# --------------------------------------------------------------------------- #
def test_pump_override_none_is_byte_identical():
    """pump_dBm_override=None reproduces the calibrated path exactly."""
    a = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
        coh_pk_window_dB=(1.0, 5.0),
    )
    b = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
        coh_pk_window_dB=(1.0, 5.0), pump_dBm_override=None,
    )
    assert a.topology.key == b.topology.key
    assert a.calibrated_pump_dBm == b.calibrated_pump_dBm
    assert a.analytical_coh_pk_dB == b.analytical_coh_pk_dB
    assert a.stub_params == b.stub_params


def test_pump_override_sets_operating_point():
    """An explicit override bypasses calibration and changes the depth."""
    calibrated = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ,
    )
    overridden = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, pump_dBm_override=13.0,
    )
    assert calibrated.calibrated_pump_dBm == pytest.approx(1.6697, abs=0.02)
    assert overridden.calibrated_pump_dBm == pytest.approx(13.0)
    # Higher pump -> larger junction modulation depth.
    assert overridden.depth_effective > calibrated.depth_effective


# --------------------------------------------------------------------------- #
# The expanded search + in-loop stability constraint (stub regime check)        #
# --------------------------------------------------------------------------- #
# A small grid keeps the analytical engine cost low; the stub emulates the
# measured physics so the constraint is exercised deterministically.
_N_GRID = (12, 20)
_PUMP_GRID = (5.0, 13.0, 17.0)


def _physics_stub(spec, run_dir, **kw):
    """Low pump -> stable amplifier; band-edge (13 dBm) -> boundary; high pump
    (>=17) -> non-convergent. Realised gain rises with pump."""
    pump = float(next(v.value for v in spec.var_block.variables if v.name == "pump_power_dBm"))
    g = -3.0 + 0.30 * (pump - 5.0)
    if pump >= 16.5:
        return _check(NON_CONVERGENT, probed=False, agree=True, g=float("nan"))
    if pump >= 12.5:
        return _check(INDETERMINATE, probed=True, agree=False, g=g)
    return _check(AMPLIFICATION, probed=True, agree=True, g=g)


def _all_oscillator_stub(spec, run_dir, **kw):
    pump = float(next(v.value for v in spec.var_block.variables if v.name == "pump_power_dBm"))
    return _check(OSCILLATION, probed=True, agree=True, g=5.0 + pump)


@pytest.fixture(scope="module")
def mixed_result():
    return design_optimized_self_pumped_search(
        varactor=VAR_A, f_signal_Hz=1.2e9, f_pump_Hz=2.4e9,
        search_n_cells=True, search_pump=True,
        n_cells_grid=_N_GRID, pump_grid_dBm=_PUMP_GRID,
        regime_check=_physics_stub, n_finalists=6,
    )


def test_search_grid_built(mixed_result):
    assert mixed_result.n_cells_grid == _N_GRID
    assert mixed_result.pump_grid_dBm == _PUMP_GRID
    # full Cartesian grid evaluated (every (N, pump) found a candidate at 1.2 GHz)
    assert len(mixed_result.all_grid_points) == len(_N_GRID) * len(_PUMP_GRID)
    # ranked descending by the analytical proxy
    anas = [c.analytical_g_mod_in_dB for c in mixed_result.all_grid_points]
    assert anas == sorted(anas, reverse=True)


def test_constraint_never_returns_oscillator_or_nonconvergent(mixed_result):
    assert mixed_result.selected is not None
    assert mixed_result.selected.regime in STABLE_OR_BOUNDARY
    # every rejected finalist is genuinely non-admissible
    for c in mixed_result.finalists:
        if not c.admissible:
            assert c.regime in (
                REGIME_OSCILLATOR, REGIME_NON_CONVERGENT, REGIME_INDETERMINATE,
            )


def test_constraint_selects_best_realized_admissible(mixed_result):
    """Selected is the max realised G_mod/in over admissible finalists."""
    admissible = [c for c in mixed_result.finalists if c.admissible]
    assert admissible
    best = max(admissible, key=lambda c: c.realized_g_mod_in_dB)
    assert mixed_result.selected.realized_g_mod_in_dB == pytest.approx(
        best.realized_g_mod_in_dB
    )
    # best_stable is amplifier-only (the comfortably-stable optimum)
    if mixed_result.best_stable is not None:
        assert mixed_result.best_stable.regime == REGIME_AMPLIFIER


def test_high_pump_top_analytical_is_rejected(mixed_result):
    """The analytically top-ranked finalist is a +17 dBm point that the
    constraint rejects (non-convergent) -- proving the analytical proxy alone
    would have returned an unstable design."""
    top = mixed_result.finalists[0]
    assert top.pump_dBm == pytest.approx(17.0)
    assert not top.admissible


def test_all_oscillator_device_yields_no_stable_optimum():
    """A device that lases across the whole ladder -> selected=None (an honest
    'no stable optimum' answer, not an error, and crucially not the oscillator)."""
    res = design_optimized_self_pumped_search(
        varactor=VAR_A, f_signal_Hz=1.2e9, f_pump_Hz=2.4e9,
        search_n_cells=True, search_pump=True,
        n_cells_grid=_N_GRID, pump_grid_dBm=_PUMP_GRID,
        regime_check=_all_oscillator_stub, n_finalists=6,
    )
    assert res.selected is None
    assert res.best_stable is None
    assert "no stable" in res.note.lower()
    # all finalists classified oscillator, none admissible
    assert all(c.regime == REGIME_OSCILLATOR for c in res.finalists)


# --------------------------------------------------------------------------- #
# The constraint vs the KNOWN recorded ADS regime verdicts                      #
# --------------------------------------------------------------------------- #
# Ground-truth AmplificationCheck verdicts recorded under ADS for regression.
# These are exactly
# the known answers the stability constraint must reproduce: the SMV2019
# +9.54 dB lasing point MUST be rejected (not returned as an optimum), the VarA
# anchor is a clean amplifier, and the VarA band-edge +0.55 dB net-positive point
# is a boundary (tests disagree). Locks the regime mapping + admissibility
# against the measured ADS reality, reproducibly and without re-running ADS.
_RECORDED_ADS_POINTS = {
    "anchor": {
        "classification": AMPLIFICATION,
        "signal_off_converged": True,
        "tests_agree": True,
        "g_mod_in_dB": -45.68418326336156,
        "expected_label": REGIME_AMPLIFIER,
        "expected_admissible": True,
    },
    "vara_band_edge": {
        "classification": INDETERMINATE,
        "signal_off_converged": True,
        "tests_agree": False,           # slope says laser, r_A says stable -> disagree
        "g_mod_in_dB": 0.5468413001260559,
        "expected_label": REGIME_BOUNDARY,
        "expected_admissible": True,
    },
    "smv2019_lasing": {
        "classification": OSCILLATION,
        "signal_off_converged": True,
        "tests_agree": True,            # slope ~0 and r_A 1.31 BOTH say laser
        "g_mod_in_dB": 9.544864149019112,
        "expected_label": REGIME_OSCILLATOR,
        "expected_admissible": False,   # the +9.54 dB lasing point is REJECTED
    },
}


@pytest.mark.parametrize("name", sorted(_RECORDED_ADS_POINTS))
def test_constraint_matches_recorded_ads_verdicts(name):
    rec = _RECORDED_ADS_POINTS[name]
    chk = _check(
        rec["classification"],
        probed=rec["signal_off_converged"],
        agree=rec["tests_agree"],
        g=rec["g_mod_in_dB"],
    )
    label = regime_label_from_check(chk)
    assert label == rec["expected_label"]
    assert (label in STABLE_OR_BOUNDARY) == rec["expected_admissible"]


def test_smv2019_lasing_point_is_rejected():
    """The headline STEP-3 lock: the SMV2019 +9.54 dB net-positive point is
    self-oscillation, and the constraint refuses it as an optimum."""
    rec = _RECORDED_ADS_POINTS["smv2019_lasing"]
    chk = _check(rec["classification"], probed=True, agree=True, g=rec["g_mod_in_dB"])
    assert regime_label_from_check(chk) == REGIME_OSCILLATOR
    assert regime_label_from_check(chk) not in STABLE_OR_BOUNDARY


# --------------------------------------------------------------------------- #
# Guards                                                                        #
# --------------------------------------------------------------------------- #
def test_expansion_requires_regime_check():
    with pytest.raises(ValueError, match="regime_check"):
        design_optimized_self_pumped_search(
            varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ,
            search_pump=True, pump_grid_dBm=(5.0,),
        )


def test_pump_override_kwarg_forbidden():
    with pytest.raises(ValueError, match="pump_dBm_override"):
        design_optimized_self_pumped_search(
            varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ,
            regime_check=_physics_stub, pump_dBm_override=5.0,
        )


def test_coh_pk_objective_rejected():
    with pytest.raises(ValueError, match="g_mod_in"):
        design_optimized_self_pumped_search(
            varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ,
            regime_check=_physics_stub, objective="coh_pk",
        )


def test_unconstrained_opt_in_runs_without_regime_check():
    """require_regime_check=False allows an explicit analytical-only ranking;
    it does NOT certify stability, so selected stays None."""
    res = design_optimized_self_pumped_search(
        varactor=VAR_A, f_signal_Hz=1.2e9, f_pump_Hz=2.4e9,
        search_pump=True, pump_grid_dBm=(5.0, 9.0),
        n_cells_grid=(12,), search_n_cells=True,
        require_regime_check=False, n_finalists=3,
    )
    assert res.selected is None
    assert res.best_stable is None
    assert res.all_grid_points  # analytical screen still ran
    assert "analytical-only" in res.note


