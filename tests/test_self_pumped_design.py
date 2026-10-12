"""Regression tests for the productionized self-pumped design API.

Locks the A.3.1 anchor outcomes against the
productionized :mod:`tvtl_ads_builder_v2.parity.self_pumped_design` API so
future framework changes can't silently break the self-pumped workflow.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from tvtl.optimization.figure_of_merit import (
    CalibrationError,
    calibrate_self_pumped_pump_dBm,
)
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.parity.self_pumped_design import (
    NoSelfPumpedCandidateError,
    SelfPumpedDesignResult,
    design_optimized_self_pumped,
    main_period_anchor_at_3p1GHz,
    optimized_self_pumped_spec,
)


# --------------------------------------------------------------------------- #
# A.3.1 fixtures (the locked operating point)                                  #
# --------------------------------------------------------------------------- #
VAR_A = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5,
)
F_SIGNAL_HZ = 3.1e9
F_PUMP_HZ = 6.2e9

# Locked A.3.1 outcomes (from the original selection artifacts).
A3P1_PUMP_DBM = 1.6696957672611337
A3P1_TOPOLOGY_KEY = "n2_single_short"
A3P1_STUB_PARAMS_M = (
    0.5111436888575555e-3, 2.2083339346572757e-3,
    0.9451326757669449e-3, 6.609422719571739e-3,
)
A3P1_Z_TL_OHM = 42.22222222222222
A3P1_W_MAIN_M = 4.575430198648918e-3
A3P1_COH_PK_DB = 4.879618686498491
A3P1_MLOSS_IN_DB = 2.9844855522322913
A3P1_MLOSS_OUT_DB = 2.9769433600048356


# --------------------------------------------------------------------------- #
# Calibrator                                                                   #
# --------------------------------------------------------------------------- #
def test_calibrate_self_pumped_pump_dBm_a3p1():
    """Calibrator returns +1.6697 dBm for Varactor A @ f_pump=6.2 GHz,
    m_eff_target=0.22 (the locked A.3.1 calibration).
    """
    pump_dBm = calibrate_self_pumped_pump_dBm(
        varactor=VAR_A, f_pump_Hz=F_PUMP_HZ, m_eff_target=0.22,
    )
    # Spec: within +/- 0.01 dB of the locked value.
    assert pump_dBm == pytest.approx(A3P1_PUMP_DBM, abs=0.01)


def test_calibrate_self_pumped_pump_dBm_target_out_of_bracket():
    """m_eff_target outside the bracket raises CalibrationError; the error
    message includes observed m_eff at the bracket endpoints AND a
    bracket-widening suggestion (a diagnostic improvement).
    """
    with pytest.raises(CalibrationError) as exc_info:
        calibrate_self_pumped_pump_dBm(
            varactor=VAR_A, f_pump_Hz=F_PUMP_HZ, m_eff_target=0.22,
            bracket_dBm=(-30.0, -20.0),  # m_eff at -20 dBm is well below 0.22
        )
    msg = str(exc_info.value)
    assert "not bracketed" in msg
    assert "m_eff(-30" in msg  # observed value at lo endpoint
    assert "m_eff(-20" in msg  # observed value at hi endpoint
    assert "bracket_dBm" in msg  # references the parameter to widen
    assert "widening" in msg.lower() or "wider" in msg.lower()


# --------------------------------------------------------------------------- #
# A.3.1 high-level reproduction                                                #
# --------------------------------------------------------------------------- #
def test_optimized_self_pumped_spec_a3p1_reproduction():
    """High-level ``design_optimized_self_pumped`` reproduces the A.3.1 anchor
    bit-for-bit when called with the original parameters: same topology,
    stub params, Z_TL, main width, calibrated pump, analytical coh_pk,
    and matching losses.

    Note: A.3.1's published selection (n2_single_short rank
    9, analytical coh_pk = +4.88 dB) was made under the original window
    ``coh_pk_window_dB = (1.0, 5.0)``. The *default* window was later
    widened to ``(1.0, 10.0)`` for general-design-space use;
    under that wider window the walk-down's first-in-window candidate
    is rank 1 (n1_double_short, coh_pk ~ 8 dB) rather than rank 9.
    This test explicitly passes the original window to lock the
    published A.3.1 selection. The methodology adaptation is additive
    (new capabilities) -- callers can still reproduce the original
    behavior by passing the original parameters.

    Note: the default objective is ``g_mod_in``
    (out/in transducer gain); ``objective="coh_pk"`` is passed
    explicitly to lock the historical coherent-peak A.3.1 selection.
    """
    result = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
        coh_pk_window_dB=(1.0, 5.0),  # the original window; locks A.3.1 selection
    )
    assert isinstance(result, SelfPumpedDesignResult)
    # Topology + Sobol rank
    assert result.topology.key == A3P1_TOPOLOGY_KEY
    assert result.sobol_rank == 9
    # Geometry (bit-for-bit; the Sobol seed + family scope guarantee
    # identical stub params)
    assert len(result.stub_params) == len(A3P1_STUB_PARAMS_M)
    for got, exp in zip(result.stub_params, A3P1_STUB_PARAMS_M):
        assert got == pytest.approx(exp, rel=1e-12)
    assert result.z_tl_ohm == pytest.approx(A3P1_Z_TL_OHM, rel=1e-12)
    assert result.main_width_m == pytest.approx(A3P1_W_MAIN_M, rel=1e-12)
    # Pump + analytical numbers (spec tolerance: +/- 0.01 dB).
    assert result.calibrated_pump_dBm == pytest.approx(A3P1_PUMP_DBM, abs=0.01)
    assert result.analytical_coh_pk_dB == pytest.approx(A3P1_COH_PK_DB, abs=0.01)
    assert result.matching_loss_input_dB == pytest.approx(A3P1_MLOSS_IN_DB, abs=0.01)
    assert result.matching_loss_output_dB == pytest.approx(A3P1_MLOSS_OUT_DB, abs=0.01)
    # depth_effective at the calibrated pump should match m_eff_target.
    assert result.depth_effective == pytest.approx(0.22, abs=0.01)
    # A.3.1 is found via the primary (SHORT) families attempt, no fallback.
    assert result.families_used == 1
    # Anchor-period scaling at 3.1 GHz is exactly 6.11 mm.
    assert result.main_period_m == pytest.approx(6.11e-3, abs=1e-15)


def test_design_optimized_self_pumped_returns_spec_with_calibrated_pump():
    """The DeviceSpec in the result carries the calibrated pump in its VAR block."""
    result = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ,
    )
    pump_var = next(
        v for v in result.spec.var_block.variables if v.name == "pump_power_dBm"
    )
    assert pump_var.value == pytest.approx(result.calibrated_pump_dBm)
    # The spec also names per-cell branch-top nodes (used downstream by
    # the standing-wave diagnostic).
    names = {
        e.pins[0].node for e in result.spec.elements if e.type == "NAME"
    }
    for k in range(1, 16):
        assert f"n_c{k}_b" in names, f"missing NAME element for n_c{k}_b"


def test_design_optimized_self_pumped_no_candidate_error():
    """If the matching threshold is impossibly tight,
    NoSelfPumpedCandidateError is raised with a populated walkdown_trace
    after exhausting the default families_fallback (SHORT primary AND
    SHORT+OPEN fallback both fail under this
    impossible threshold).

    Uses ``objective="coh_pk"`` -- the matching-loss-threshold gate that
    raises ``NoSelfPumpedCandidateError`` is part of the coh_pk selection
    path (the g_mod_in objective ranks by the out/in proxy and has no
    separate matching-loss window gate).
    """
    with pytest.raises(NoSelfPumpedCandidateError) as exc_info:
        design_optimized_self_pumped(
            varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
            matching_loss_threshold_dB=0.001,  # impossibly tight
        )
    assert exc_info.value.walkdown_trace
    # The trace should show real walk-down failures, not just empty rows.
    realized = [
        e for e in exc_info.value.walkdown_trace if e.stage2_realizable
    ]
    assert realized, "expected at least one Stage-2-realizable row in trace"
    # Both primary (attempt 1) and fallback (attempt 2) should
    # be exhausted -- the trace must contain entries from BOTH attempts.
    attempts = {e.families_attempt for e in exc_info.value.walkdown_trace}
    assert 1 in attempts, "primary (SHORT) attempt should produce trace entries"
    assert 2 in attempts, (
        "fallback (SHORT+OPEN) attempt should also produce trace entries; "
        "the matching threshold is impossible across both family pools."
    )
    # The error message references the multi-attempt failure.
    assert "families_fallback" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Methodology adaptation for generalized design space                          #
# --------------------------------------------------------------------------- #
def test_main_period_anchor_at_3p1_GHz():
    """At f_signal = 3.1 GHz the anchor period is exactly 6.11 mm
    (preserves A.3.1 bit-for-bit)."""
    assert main_period_anchor_at_3p1GHz(3.1e9) == 6.11e-3


def test_main_period_scales_inversely_with_f_signal():
    """The period scales inversely with f_signal."""
    # At 6.2 GHz: exactly half of 6.11 mm.
    assert main_period_anchor_at_3p1GHz(6.2e9) == 6.11e-3 / 2.0
    # At 5.0 GHz: 6.11 mm * (3.1 / 5.0) (within float precision).
    assert main_period_anchor_at_3p1GHz(5.0e9) == pytest.approx(
        6.11e-3 * (3.1 / 5.0), rel=1e-12,
    )
    # Reciprocal scaling: period * f_signal is invariant.
    p31 = main_period_anchor_at_3p1GHz(3.1e9) * 3.1e9
    p50 = main_period_anchor_at_3p1GHz(5.0e9) * 5.0e9
    p70 = main_period_anchor_at_3p1GHz(7.0e9) * 7.0e9
    assert p31 == pytest.approx(p50, rel=1e-12)
    assert p31 == pytest.approx(p70, rel=1e-12)


def test_design_optimized_self_pumped_default_window_is_wider():
    """Default coh_pk window widened from (1, 5) to (1, 10)
    dB for the self-pumped workflow (analytical-leg conservatism per
    the standing-wave finding)."""
    import inspect
    sig = inspect.signature(design_optimized_self_pumped)
    default = sig.parameters["coh_pk_window_dB"].default
    assert default == (1.0, 10.0)


def test_design_optimized_self_pumped_topology_fallback_used():
    """When the primary families pool produces no candidate, the fallback
    families pool is tried. Construct a primary attempt
    with no families (immediate no-candidate) and a fallback with the
    standard SHORT pool; assert families_used == 2 and the result still
    matches A.3.1."""
    from tvtl.optimization.topology_generator import Termination, linear_families
    short_only = tuple(
        f for f in linear_families() if f.termination == Termination.SHORT
    )
    # Primary: empty families (zero candidates -> no-candidate failure).
    # Fallback: standard SHORT pool.
    result = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
        families_fallback=((), short_only),
        coh_pk_window_dB=(1.0, 5.0),  # narrow window to lock A.3.1 selection
    )
    assert result.families_used == 2
    # The result should still match A.3.1 (the fallback uses the same
    # SHORT pool the default primary uses; with the original window
    # passed explicitly, n2_single_short rank 9 is selected).
    assert result.topology.key == A3P1_TOPOLOGY_KEY
    assert result.sobol_rank == 9
    # Aggregate trace should contain only the fallback's entries (primary
    # produced no Sobol candidates because families was empty).
    attempts = {e.families_attempt for e in result.walkdown_trace}
    assert attempts == {2}, (
        f"primary attempt produced no entries (empty families); "
        f"walkdown_trace should contain only attempt-2 entries, got {attempts}"
    )


def test_design_optimized_self_pumped_a3p1_uses_primary():
    """A.3.1 finds its candidate in the primary (SHORT) families pool
    and never invokes the SHORT+OPEN fallback (a
    backward-compatibility check; ``objective="coh_pk"`` for the A.3.1
    selection)."""
    result = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
    )
    assert result.families_used == 1
    # Walkdown trace from the SUCCESSFUL primary attempt should
    # contain only attempt-1 entries.
    attempts = {e.families_attempt for e in result.walkdown_trace}
    assert attempts == {1}


def test_optimized_self_pumped_spec_main_period_default_matches_anchor():
    """Spec builder default uses main_period_anchor_at_3p1GHz when no
    main_period_m is passed."""
    result = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ,
    )
    # Cell main-line halves should be 6.11 mm / 2 = 3.055 mm at 3.1 GHz.
    line_lefts = [
        e for e in result.spec.elements if e.type == "MLIN" and e.id.startswith("LINE_LEFT_")
    ]
    assert line_lefts
    for el in line_lefts:
        assert el.parameters["l_m"] == pytest.approx(6.11e-3 / 2.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# Low-level spec builder                                                       #
# --------------------------------------------------------------------------- #
def test_optimized_self_pumped_spec_n_input_carries_both_sources():
    """Self-pumped spec has both signal (Num=1) and pump (Num=2) on n_input
    (no separate control-port net).
    """
    spec = optimized_self_pumped_spec(
        topology=design_optimized_self_pumped(
            varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ,
        ).topology,
        params=A3P1_STUB_PARAMS_M,
        varactor=VAR_A,
        n_cells=15,
        f_signal_Hz=F_SIGNAL_HZ, f_pump_Hz=F_PUMP_HZ,
        pump_power_dBm=A3P1_PUMP_DBM,
        main_width_m=A3P1_W_MAIN_M,
    )
    signal_src = next(s for s in spec.sources if s.num == 1)
    pump_src = next(s for s in spec.sources if s.num == 2)
    assert signal_src.port_node == "n_input"
    assert pump_src.port_node == "n_input"
    # No control-port net.
    assert all(n.id != "n_pump_ctrl" for n in spec.nodes)


def test_optimized_self_pumped_spec_uses_charge_polynomial():
    """SDD2P bodies use q_poly_coefficients (charge), not tvc_poly_coefficients."""
    result = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ,
    )
    sdds = [e for e in result.spec.elements if e.type == "SDD2P"]
    assert sdds, "spec has no SDD2P elements"
    for sdd in sdds:
        assert "q_poly_coefficients" in sdd.parameters
        assert "tvc_poly_coefficients" not in sdd.parameters


def test_optimized_self_pumped_spec_rejects_ideal_varactor():
    """Ideal (constant-C) varactors have no pump response -> raise ValueError."""
    with pytest.raises(ValueError, match="ideal"):
        optimized_self_pumped_spec(
            topology=design_optimized_self_pumped(
                varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ,
            ).topology,
            params=A3P1_STUB_PARAMS_M,
            varactor=RealisticVaractor.ideal(0.3e-12),
            n_cells=15,
        )


# --------------------------------------------------------------------------- #
# Optional ADSPY cross-val (skipped if hpeesofsim is not on PATH)              #
# --------------------------------------------------------------------------- #
def _hpeesofsim_available() -> bool:
    if os.environ.get("HPEESOF_DIR"):
        return True
    return any(shutil.which(n) for n in (
        "adssim_win.exe", "adssim", "hpeesofsim.exe", "hpeesofsim",
    ))


@pytest.mark.skipif(
    not _hpeesofsim_available(),
    reason="hpeesofsim not on PATH; ADSPY cross-val skipped",
)
def test_optimized_self_pumped_spec_a3p1_cross_val_adspy(ads_tmp_path):
    """ADS HB cross-val on A.3.1 passes the eps_peak <= 5 dB gate.

    Single-shot reproduction of the original ADS HB result.
    """
    from tvtl_ads_builder_v2.analysis.hb_dataset import read_hb_dataset
    from tvtl_ads_builder_v2.device_spec import with_var_overrides
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.simulate.runner import run_hpeesofsim

    result = design_optimized_self_pumped(
        varactor=VAR_A, f_signal_Hz=F_SIGNAL_HZ, objective="coh_pk",
        coh_pk_window_dB=(1.0, 5.0),  # the original window; locks A.3.1 selection
    )
    # The original run found the ADS PSA peak at phase = 120 deg with coh_pk = 9.71 dB.
    # Reproduce by running HB ON + OFF at that phase only (a single-pair run
    # rather than the full 6-phase PSA sweep).
    fav_phase_rad = math.radians(120.0)
    spec_on = with_var_overrides(result.spec, {"pump_phase_rad": fav_phase_rad})
    spec_off = with_var_overrides(spec_on, {"pump_power_dBm": -100.0})

    def _run(spec, tag):
        ws = ads_tmp_path / tag
        ws.mkdir(parents=True, exist_ok=True)
        netlist = ws / f"{tag}.ckt"
        netlist.write_text(emit_flat_hb_netlist(spec), encoding="utf-8")
        res = run_hpeesofsim(netlist, timeout_s=900)
        assert res["status"] == "completed", res.get("error_message")
        return read_hb_dataset(res["dataset_path"], (F_SIGNAL_HZ, F_PUMP_HZ))

    ds_on = _run(spec_on, "a3p1_fav_on")
    ds_off = _run(spec_off, "a3p1_fav_off")
    v10_on = complex(ds_on.node_voltage("n_output").at_mix((1, 0)))
    vm11_on = complex(ds_on.node_voltage("n_output").at_mix((-1, 1)))
    v10_off = complex(ds_off.node_voltage("n_output").at_mix((1, 0)))
    coh_pk_dB = 20.0 * math.log10(abs(v10_on + vm11_on) / abs(v10_off))
    eps_peak = abs(result.analytical_coh_pk_dB - coh_pk_dB)
    # The original run found ADS PSA peak +9.71 dB at this phase; eps_peak ~4.83 dB.
    # Gate ceiling: 5 dB. Allow +/- 0.3 dB headroom around the locked value.
    assert coh_pk_dB == pytest.approx(9.71, abs=0.3)
    assert eps_peak <= 5.0


