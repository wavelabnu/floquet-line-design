"""Public command-line interface for TVTL optimizer workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


DEFAULT_SAMPLES_PER_FAMILY = 300
DEFAULT_TOP_K = 10
DEFAULT_TOP_M = 3
DEFAULT_REFINE_MAXITER = 200

QUICK_SAMPLES_PER_FAMILY = 64
QUICK_TOP_K = 8
QUICK_TOP_M = 1
QUICK_REFINE_MAXITER = 60

# --------------------------------------------------------------------------- #
# Predicted-leg labeling. Every analytical figure/number this CLI emits is a
# PREDICTION (pure Python; run_ads=False); none is ADS-verified. An analytical
# ranking is not a gain claim: the coherent-gain proxy can over-state the ADS
# harmonic-balance record by a family-dependent margin (see docs/LIMITATIONS.md
# 'Characterized Performance Limits'), and a boost ratio over a near-zero pump-off
# floor is never gain. Only a harmonic-balance record certified as in
# docs/GAIN_DEFINITIONS.md is a gain claim. These strings are reused across the
# JSON, CSV, report.md and the embedded figures so no predicted result can be
# mistaken for a realized one.
# --------------------------------------------------------------------------- #
PREDICTION_LABEL = "analytical_prediction_not_ads_verified"
PREDICTED_GAIN_QUALIFIER = (
    "ANALYTICAL PREDICTION (optimistic; not ADS-verified). This is an analytical "
    "ranking, not a gain claim: the coherent-gain proxy can over-state the ADS "
    "harmonic-balance record by a family-dependent margin, and a boost ratio over "
    "a near-zero pump-off floor is never gain. Only a harmonic-balance record "
    "certified as in docs/GAIN_DEFINITIONS.md is a gain claim. See "
    "docs/LIMITATIONS.md 'Characterized Performance Limits'."
)
# Predicted S-parameter sweep grid (relative to the signal frequency f_s).
S_SWEEP_LO_FACTOR = 0.5
S_SWEEP_HI_FACTOR = 2.5
S_SWEEP_N_POINTS = 161
S_SWEEP_Z_REF_OHM = 50.0

# --------------------------------------------------------------------------- #
# Coherence-length context. The N_coh reported per candidate below is a COARSE
# heuristic computed on the simplified analytical Stage-2 cell (static_cell_abcd,
# the same cell the Bloch match uses): a mid-band cell, per-cell phase ~95 deg.
# It is NOT the measured robust-optimum length. That optimum -- ~15 cells at
# 2.0 GHz and ~18 at 1.7 GHz -- is established directly by ADS pump/length sweeps
# on the coherence-length design family (accurate VIAGND/MLEF terminations); see
# docs/LIMITATIONS.md "Optimal length and its mechanism" and
# docs/methodology/OPTIMIZER_METHODOLOGY.md "Length selection". The measured
# optimum is consistent, at the dispersion level, with degenerate phase-matching
# (N_coh = pi/|theta(2 f_s) - 2 theta(f_s)| on the slow-wave BAND-EDGE realized
# cell, per-cell phase ~133-156 deg); the exact analytical N_coh depends on the
# cell model and is under derivation for a follow-up. Coherence-length values
# reported by prior band-edge analysis -- ~15.3 cells at 2.0 GHz, ~18.5 at 1.7 --
# are NOT reproduced by this simplified mid-band cell. A gain-max pick is
# generally NOT the coherence-length design and is generally NOT phase-matched
# (N_coh << chosen N, stability margin < 0); its heuristic N_coh on the simplified
# cell therefore does not, and is not meant to, match the measured optimum.
# --------------------------------------------------------------------------- #
VALIDATED_COHERENCE_ANCHORS_GHZ: dict[float, tuple[float, int]] = {
    2.0: (15.3, 15),
    1.7: (18.5, 18),
}
VALIDATED_COHERENCE_REFERENCE = (
    "Measured robust optimum length from direct ADS pump/length sweeps: ~15 "
    "cells at 2.0 GHz (n3_single_short, shipped) and ~18 cells at 1.7 GHz. This "
    "is consistent, at the dispersion level, with degenerate phase-matching "
    "coherence-length design (slow-wave band-edge cell, accurate VIAGND/MLEF "
    "terminations); the exact analytical N_coh depends on the cell model and is "
    "under derivation for a follow-up. Coherence-length values reported by prior "
    "band-edge analysis -- ~15.3 cells at 2.0 GHz, ~18.5 at 1.7 GHz -- are not "
    "reproduced by this public optimizer's simplified mid-band cell. See "
    "docs/LIMITATIONS.md 'Optimal length and its mechanism' and "
    "docs/methodology/OPTIMIZER_METHODOLOGY.md 'Length selection'."
)


def _validated_coherence_anchor(f_s_GHz: float) -> tuple[float, int] | None:
    """Documented (predicted N_coh, measured optimum N) at f_s, if anchored."""

    for f_anchor, pair in VALIDATED_COHERENCE_ANCHORS_GHZ.items():
        if abs(float(f_s_GHz) - f_anchor) < 1e-6:
            return pair
    return None


def _positive_float(value: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return out


def _positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return out


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _complex_dict(value: complex) -> dict[str, float | None]:
    z = complex(value)
    return {
        "real": _finite_or_none(z.real),
        "imag": _finite_or_none(z.imag),
        "magnitude": _finite_or_none(abs(z)),
    }


def _default_varactor():
    from tvtl.varactors import RealisticVaractor

    return RealisticVaractor(
        CJ0_f=0.3e-12,
        VJ_v=3.5,
        M=1.7,
        CP_f=0.0,
        L_pkg_h=0.3e-9,
        R_series_ohm=0.5,
    )


def _varactor_summary(varactor: Any) -> dict[str, Any]:
    # Operating point (G7): the static shunt capacitance at the DC bias point.
    # ``C0_f`` is the junction C(V_bias) plus any package CP (the capacitance the
    # cell actually presents); ``C(V_bias)`` is the bare-junction value. Both are
    # analytical design values (predicted), not measured.
    v_bias = getattr(varactor, "V_bias_v", 0.0)
    c_at_bias_f = _finite_or_none(getattr(varactor, "C0_f", None))
    try:
        c_junction_f = _finite_or_none(varactor.C(float(v_bias)))
    except Exception:  # noqa: BLE001  (ideal/edge models)
        c_junction_f = None
    return {
        "model": "low-parasitic analytical design proxy",
        "CJ0_f": _finite_or_none(varactor.CJ0_f),
        "VJ_v": _finite_or_none(varactor.VJ_v),
        "M": _finite_or_none(varactor.M),
        "CP_f": _finite_or_none(varactor.CP_f),
        "L_pkg_h": _finite_or_none(varactor.L_pkg_h),
        "R_series_ohm": _finite_or_none(varactor.R_series_ohm),
        # Operating capacitance (G7, predicted).
        "V_bias_v": _finite_or_none(v_bias),
        "C_at_bias_f": c_at_bias_f,
        "C_at_bias_pF": None if c_at_bias_f is None else c_at_bias_f * 1e12,
        "C_junction_at_bias_f": c_junction_f,
        "operating_point_note": (
            "Operating capacitance C(V_bias) = static shunt capacitance at the DC "
            "bias point (junction C plus package CP). Analytical design value "
            "(prediction), not an ADS/measured operating point."
        ),
        "note": (
            "Default used by the public optimizer CLI. ADS/HB/OpenAccess "
            "validation, if desired, is a separate optional workflow."
        ),
    }


def _electrical_length_deg(
    width_m: Any, length_m: Any, f0_Hz: float, substrate: Any
) -> float | None:
    """Predicted microstrip electrical length [deg] at ``f0`` (G2).

    ``E = beta * length * 180/pi`` with ``beta = 2*pi*f0*sqrt(eps_eff)/c`` and
    ``eps_eff`` the Hammerstad-Kirschning effective permittivity for the section
    width. Returns ``None`` for missing/non-physical geometry.
    """

    if width_m is None or length_m is None:
        return None
    try:
        w = float(width_m)
        ell = float(length_m)
    except (TypeError, ValueError):
        return None
    if not (w > 0.0) or not math.isfinite(ell) or float(f0_Hz) <= 0.0:
        return None
    from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
    from tvtl.microstrip.models import hammerstad_kirschning

    m = substrate.stub_substrate_mapping()
    try:
        _z0, eps_eff_arr = hammerstad_kirschning(
            w, m["t_m"], m["h_m"], m["eps_r"], [float(f0_Hz)]
        )
    except Exception:  # noqa: BLE001
        return None
    eps_eff = float(eps_eff_arr[0])
    beta = 2.0 * math.pi * float(f0_Hz) * math.sqrt(eps_eff) / SPEED_OF_LIGHT_M_PER_S
    return float(beta * ell * 180.0 / math.pi)


def _matching_dict(
    section: Any | None, f0_Hz: float | None = None, substrate: Any | None = None
) -> dict[str, Any] | None:
    if section is None:
        return None
    return {
        "topology": section.topology,
        "elements": [
            {
                "kind": kind,
                "width_m": _finite_or_none(width_m),
                "width_mm": _finite_or_none(width_m * 1e3),
                "length_m": _finite_or_none(length_m),
                "length_mm": _finite_or_none(length_m * 1e3),
                # Electrical length in degrees at f_s (G2, predicted).
                "length_deg": (
                    None
                    if f0_Hz is None or substrate is None
                    else _electrical_length_deg(width_m, length_m, f0_Hz, substrate)
                ),
                "z0_ohm": _finite_or_none(z0_ohm),
            }
            for kind, width_m, length_m, z0_ohm in section.elements
        ],
        "s11_before_dB": _finite_or_none(section.s11_before_dB),
        "s11_after_dB": _finite_or_none(section.s11_after_dB),
        "frac_bandwidth_pct": _finite_or_none(section.frac_bandwidth_pct),
        "realizable": bool(section.realizable),
        "note": section.note,
    }


def _stub_parameters(
    stub: Any, f0_Hz: float | None = None, substrate: Any | None = None
) -> list[dict[str, float | int | None]]:
    params = list(stub.params)
    out: list[dict[str, float | int | None]] = []
    for idx in range(0, len(params), 2):
        segment = idx // 2 + 1
        width_m = params[idx]
        length_m = params[idx + 1]
        out.append(
            {
                "segment": segment,
                "width_m": _finite_or_none(width_m),
                "width_mm": _finite_or_none(width_m * 1e3),
                "length_m": _finite_or_none(length_m),
                "length_mm": _finite_or_none(length_m * 1e3),
                # Electrical length in degrees at f_s (G2, predicted).
                "length_deg": (
                    None
                    if f0_Hz is None or substrate is None
                    else _electrical_length_deg(width_m, length_m, f0_Hz, substrate)
                ),
            }
        )
    return out


def _analytical_metrics(report: Any) -> dict[str, dict[str, float | None]]:
    return {
        f"{float(pump_dBm):g}": {
            key: _finite_or_none(value)
            for key, value in sorted(metrics.items())
        }
        for pump_dBm, metrics in sorted(report.analytical.items())
    }


def _predicted_s_sweep(design: Any) -> dict[str, Any] | None:
    """Predicted S11/S21 sweep of the N-cell Bloch-matched cascade (G1).

    Passive small-signal (unpumped) ABCD of one matched static unit cell
    (:func:`...stage2_main_tl.static_cell_abcd`, the exact cell Stage-2 matched
    to 50 ohm) raised to ``n_cells`` and converted to S-parameters at the port
    reference. This is the analytical *prediction* the numerical (ADS) leg
    overlays the realized S-parameters onto; the Stage-3 input/output matching
    networks are reported separately and are NOT folded into this passive curve.
    """

    import numpy as np

    from tvtl.optimization.stage2_main_tl import static_cell_abcd
    from tvtl.theory.sparams import abcd_to_sparameters

    try:
        z_tl = float(design.main_tl.z_tl_ohm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(z_tl):
        return None
    f_s = float(design.f0_Hz)
    n_cells = int(design.n_cells)
    z_ref = float(S_SWEEP_Z_REF_OHM)
    lo = S_SWEEP_LO_FACTOR * f_s
    hi = S_SWEEP_HI_FACTOR * f_s
    freqs = np.linspace(lo, hi, int(S_SWEEP_N_POINTS))

    def _cell_s_dB(f_hz: float) -> tuple[float, float]:
        cell = static_cell_abcd(
            design.stub, design.varactor, z_tl,
            substrate=design.substrate, f0_Hz=float(f_hz), use_mtee=True,
        )
        casc = np.linalg.matrix_power(np.asarray(cell, dtype=np.complex128), n_cells)
        sp = abcd_to_sparameters(casc, reference_impedance_ohm=z_ref)
        s11 = abs(complex(sp.s11))
        s21 = abs(complex(sp.s21))
        s11_dB = 20.0 * math.log10(s11) if s11 > 1e-12 else -240.0
        s21_dB = 20.0 * math.log10(s21) if s21 > 1e-12 else -240.0
        return s11_dB, s21_dB

    s11_db: list[float] = []
    s21_db: list[float] = []
    for f in freqs:
        try:
            a, b = _cell_s_dB(float(f))
        except Exception:  # noqa: BLE001  (band-edge non-realizable widths -> nan)
            a, b = float("nan"), float("nan")
        s11_db.append(a)
        s21_db.append(b)

    idx_fs = int(np.argmin(np.abs(freqs - f_s)))
    idx_2fs = int(np.argmin(np.abs(freqs - 2.0 * f_s)))
    return {
        "frequency_GHz": [float(f) / 1e9 for f in freqs],
        "s11_dB": s11_db,
        "s21_dB": s21_db,
        "reference_impedance_ohm": z_ref,
        "n_cells": n_cells,
        "band_GHz": [lo / 1e9, hi / 1e9],
        "n_points": int(S_SWEEP_N_POINTS),
        "f_s_GHz": f_s / 1e9,
        "f_pump_GHz": 2.0 * f_s / 1e9,
        "s11_at_fs_dB": _finite_or_none(s11_db[idx_fs]),
        "s21_at_fs_dB": _finite_or_none(s21_db[idx_fs]),
        "s11_at_2fs_dB": _finite_or_none(s11_db[idx_2fs]),
        "s21_at_2fs_dB": _finite_or_none(s21_db[idx_2fs]),
        "method": (
            "passive small-signal ABCD of the N-cell Bloch-matched cascade "
            "(static_cell_abcd^N -> S, Z_ref=50 ohm); unpumped structure, "
            "Stage-3 input/output matching networks NOT folded in"
        ),
        "label": PREDICTION_LABEL,
    }


def _coherence_summary(design: Any) -> dict[str, Any] | None:
    """Per-candidate coherence-length heuristic N_coh (G5).

    ``N_coh = pi / |theta(2 f_s) - 2 theta(f_s)|`` where ``theta(f)`` is the
    per-cell Bloch phase of the matched static unit cell, extracted via the
    trace-arccosh dispersion (:func:`tvtl.theory.dispersion.bloch_dispersion`:
    ``theta = Im(gamma) * period``).

    Scope (read this before comparing to the measured optimum). This is a COARSE
    heuristic computed on the simplified analytical Stage-2 cell
    (:func:`...stage2_main_tl.static_cell_abcd`) -- a mid-band cell
    (per-cell phase ~95 deg at the A1 ceiling). It is NOT the measured robust
    optimum length (~15 cells at 2.0 GHz from ADS pump/length sweeps), and it is
    not the band-edge coherence-length analysis (a reported ~15.3 cells on the
    slow-wave BAND-EDGE realized cell, per-cell phase ~133-156 deg, accurate
    VIAGND/MLEF terminations, which this simplified cell does not reproduce; see
    :data:`VALIDATED_COHERENCE_REFERENCE`). A gain-max pick is generally not
    phase-matched (``N_coh << chosen N``, stability margin < 0), so its heuristic
    value here is small and does not, and is not meant to, match that optimum.
    The robust optimum length comes from the numerical (ADS) leg.
    """

    import numpy as np

    from tvtl.optimization.stage2_main_tl import MAIN_PERIOD_M, static_cell_abcd
    from tvtl.theory.dispersion import bloch_dispersion

    try:
        z_tl = float(design.main_tl.z_tl_ohm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(z_tl):
        return None
    f_s = float(design.f0_Hz)
    period = float(MAIN_PERIOD_M)

    def _theta(f_hz: float) -> float:
        cell = static_cell_abcd(
            design.stub, design.varactor, z_tl,
            substrate=design.substrate, f0_Hz=float(f_hz), use_mtee=True,
        )
        a, b = cell[0, 0], cell[0, 1]
        c, d = cell[1, 0], cell[1, 1]
        gamma = complex(np.asarray(bloch_dispersion((a, b, c, d), period)).reshape(-1)[0])
        return float(gamma.imag * period)  # per-cell Bloch phase [rad]

    try:
        th_fs = _theta(f_s)
        th_2fs = _theta(2.0 * f_s)
        # Band-edge confidence: the principal-branch theta is the true continuous
        # per-cell phase only if no Bloch band edge (theta -> 180 deg) is crossed
        # between f_s and 2 f_s. Probe a few interior points and flag if the phase
        # approaches the band edge (where trace-arccosh folds and N_coh would be
        # unreliable).
        probe = [_theta(f_s + (2.0 * f_s - f_s) * k / 10.0) for k in range(11)]
        theta_max = max(probe)
    except Exception:  # noqa: BLE001
        return None
    raw_delta = th_2fs - 2.0 * th_fs
    # Wrap into (-pi, pi]: the coherent-buildup kernel is 2*pi-periodic in the
    # per-cell mismatch, so the wrapped value is the representative (longest)
    # coherence length.
    delta = (raw_delta + math.pi) % (2.0 * math.pi) - math.pi
    n_coh = math.pi / abs(delta) if abs(delta) > 1e-12 else float("inf")
    band_edge_near = theta_max > math.pi - math.radians(5.0)
    chosen_n = int(design.n_cells)
    # Heuristic phase-matched flag: a coherence-length design has N_coh comparable
    # to (or above) the chosen cell count; a gain-max pick has N_coh << N.
    phase_matched = bool(math.isfinite(n_coh) and n_coh >= 0.5 * chosen_n)
    f_s_GHz = f_s / 1e9
    anchor = _validated_coherence_anchor(f_s_GHz)
    return {
        "N_coh_cells": _finite_or_none(n_coh),
        "theta_fs_deg": math.degrees(th_fs),
        "theta_2fs_deg": math.degrees(th_2fs),
        "delta_theta_deg": math.degrees(delta),
        "delta_theta_raw_deg": math.degrees(raw_delta),
        "theta_max_in_band_deg": math.degrees(theta_max),
        "band_edge_crossing_warning": bool(band_edge_near),
        "chosen_N": chosen_n,
        "is_phase_matched": phase_matched,
        "f_s_GHz": f_s_GHz,
        "f_pump_GHz": 2.0 * f_s / 1e9,
        "cell_model": (
            "analytical Stage-2 static cell (static_cell_abcd); simplified "
            "mid-band cell, NOT the accurate VIAGND/MLEF band-edge realized cell "
            "used in the reported 15.3-cell coherence-length analysis"
        ),
        "scope": (
            "coarse per-candidate heuristic on the simplified analytical cell, "
            "NOT the measured coherence-length optimum (see validated_law)"
        ),
        "validated_law": {
            "f_s_anchored": anchor is not None,
            "predicted_N_coh_cells": None if anchor is None else float(anchor[0]),
            "measured_optimum_N": None if anchor is None else int(anchor[1]),
            "reference": VALIDATED_COHERENCE_REFERENCE,
        },
        "formula": (
            "N_coh = pi / |theta(2 f_s) - 2 theta(f_s)| (mismatch wrapped to "
            "(-pi, pi]); theta = principal-branch per-cell Bloch phase "
            "(trace-arccosh dispersion of the simplified analytical Stage-2 cell). "
            "Coarse per-candidate heuristic -- confirm by the numerical N-sweep; "
            "the measured robust optimum and reported coherence-length values are "
            "in validated_law."
        ),
        "label": PREDICTION_LABEL,
    }


def _candidate_dict(report: Any, rank: int, drive_dBm: float) -> dict[str, Any]:
    design = report.design
    stub = design.stub
    main_tl = design.main_tl
    f0_Hz = float(design.f0_Hz)
    substrate = design.substrate
    drive_metrics = report.analytical.get(float(drive_dBm), {})
    from tvtl.design.feasibility import candidate_cell_factory, feasibility
    from tvtl.optimization.figure_of_merit import a1_factors, LorentzianA1
    factory=candidate_cell_factory(varactor=design.varactor, substrate=substrate, f_s=f0_Hz, N=design.n_cells)
    try:
        detail=feasibility(factory(stub.topology,stub.params),f0_Hz,N=design.n_cells)
        _,kappa,x=a1_factors(stub.topology,stub.params,f0_Hz=f0_Hz,varactor=design.varactor,substrate=substrate,drive_dBm=drive_dBm)
        guard=LorentzianA1();rho=guard.gamma**2/((x-guard.x0)**2+guard.gamma**2)
        detail.update(kappa_branch=kappa,rho_para=rho,A1_prime=None if detail['T_2fs'] is None else
                      detail['T_2fs']*detail['pump_survival_e_minus_N_alpha']*rho)
    except (ValueError, np.linalg.LinAlgError) as exc:
        detail={'error':str(exc)}
    return {
        "feasibility": detail,
        "rank": rank,
        "topology": stub.topology.key,
        "n_segments": int(stub.topology.n_segments),
        "via": stub.topology.via.name.lower(),
        "termination": stub.topology.termination.value,
        "a1_score": _finite_or_none(stub.a1_score),
        "stub_parameters": _stub_parameters(stub, f0_Hz, substrate),
        "main_tl": {
            "z_tl_ohm": _finite_or_none(main_tl.z_tl_ohm),
            "width_m": _finite_or_none(main_tl.w_main_m),
            "width_mm": _finite_or_none(main_tl.w_main_m * 1e3),
            "bloch_z_ohm": _complex_dict(main_tl.bloch_z_ohm),
            "match_residual_ohm": _finite_or_none(main_tl.match_residual_ohm),
            "realizable": bool(main_tl.realizable),
            "matched": bool(main_tl.matched),
            "note": main_tl.note,
        },
        "input_match": _matching_dict(design.input_match, f0_Hz, substrate),
        "output_match": _matching_dict(design.output_match, f0_Hz, substrate),
        "metrics_at_drive": {
            key: _finite_or_none(value)
            for key, value in sorted(drive_metrics.items())
        },
        "analytical": _analytical_metrics(report),
        # Predicted (analytical-only) additions. Every field is a prediction
        # (run_ads=False); none is ADS-verified. See PREDICTION_LABEL.
        "predicted_gain_note": PREDICTED_GAIN_QUALIFIER,
        "predicted_s_sweep": _predicted_s_sweep(design),
        "coherence": _coherence_summary(design),
        "analytical_stability": dict(getattr(report, "stability", {}) or {}),
        "ads": report.ads,
        "passed": report.passed,
        "note": report.note,
    }


def _candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cand in candidates:
        metrics = cand["metrics_at_drive"]
        stub_mm = "; ".join(
            f"s{p['segment']}:w={p['width_mm']:.4g}mm,l={p['length_mm']:.4g}mm"
            for p in cand["stub_parameters"]
            if p["width_mm"] is not None and p["length_mm"] is not None
        )
        coherence = cand.get("coherence") or {}
        stability = cand.get("analytical_stability") or {}
        rows.append(
            {
                "rank": cand["rank"],
                "topology": cand["topology"],
                "a1_score": cand["a1_score"],
                "coherent_peak_dB": metrics.get("coherent_peak"),
                "absolute_outin_dB": metrics.get("absolute_outin_dB"),
                "coherent_at0_dB": metrics.get("coherent_at0"),
                "mix10_dB": metrics.get("mix10"),
                "psa_swing_dB": metrics.get("psa_swing"),
                "main_z_tl_ohm": cand["main_tl"]["z_tl_ohm"],
                "main_width_mm": cand["main_tl"]["width_mm"],
                "input_match": None
                if cand["input_match"] is None
                else cand["input_match"]["topology"],
                "output_match": None
                if cand["output_match"] is None
                else cand["output_match"]["topology"],
                "stub_parameters_mm": stub_mm,
                # Appended predicted columns (G5 / G4); existing columns above
                # are unchanged for backward compatibility.
                "predicted_N_coh_cells": coherence.get("N_coh_cells"),
                "predicted_stability_margin": stability.get("margin"),
                **{key:(cand.get('feasibility') or {}).get(key) for key in
                   ('T_2fs','T_fs','alpha_2fs_Np_per_cell','alpha_fs','kappa_branch','rho_para','A1_prime','feasible')},
            }
        )
    return rows


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "topology",
        "a1_score",
        "coherent_peak_dB",
        "absolute_outin_dB",
        "coherent_at0_dB",
        "mix10_dB",
        "psa_swing_dB",
        "main_z_tl_ohm",
        "main_width_mm",
        "input_match",
        "output_match",
        "stub_parameters_mm",
        # Appended predicted columns (additive; legacy columns above unchanged).
        "predicted_N_coh_cells",
        "predicted_stability_margin",
        "T_2fs", "T_fs", "alpha_2fs_Np_per_cell", "alpha_fs", "kappa_branch", "rho_para", "A1_prime", "feasible",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Predicted figures (matplotlib PNGs). All carry a visible PREDICTION /        #
# not-ADS-verified mark in the title and caption, and F2/F3 are laid out so    #
# the numerical (ADS) leg can overlay the realized curves on the same axes.     #
# --------------------------------------------------------------------------- #
def _new_axes(figsize=(6.4, 4.0), dpi=110):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return plt, fig, ax


def _save_sparams_figure(path: Path, candidate: dict[str, Any]) -> bool:
    """F2: predicted S11/S21 magnitude (dB) vs frequency."""

    sweep = candidate.get("predicted_s_sweep")
    if not sweep:
        return False
    import numpy as np

    plt, fig, ax = _new_axes()
    try:
        f = np.asarray(sweep["frequency_GHz"], dtype=float)
        s11 = np.asarray(sweep["s11_dB"], dtype=float)
        s21 = np.asarray(sweep["s21_dB"], dtype=float)
        f_s = float(sweep["f_s_GHz"])
        f_2s = float(sweep["f_pump_GHz"])
        ax.plot(f, s21, color="#1f77b4", lw=1.9, label="S21 predicted")
        ax.plot(f, s11, color="#d62728", lw=1.4, label="S11 predicted")
        ax.axvline(f_s, color="0.45", ls="--", lw=1.0)
        ax.axvline(f_2s, color="0.45", ls=":", lw=1.0)
        finite = np.concatenate([s11[np.isfinite(s11)], s21[np.isfinite(s21)]])
        y_lo = max(-60.0, float(np.min(finite)) - 5.0) if finite.size else -60.0
        ax.set_ylim(y_lo, 5.0)  # headroom above 0 dB for the realized overlay
        ymax = ax.get_ylim()[1]
        ax.text(f_s, ymax, " f_s", va="top", ha="left", fontsize=8, color="0.3")
        ax.text(f_2s, ymax, " 2 f_s", va="top", ha="left", fontsize=8, color="0.3")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("|S| (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower center", fontsize=8, ncol=2)
        ax.set_title(
            "Predicted S-parameters — analytical, NOT ADS-verified",
            fontsize=10,
        )
        fig.text(
            0.5, 0.005,
            "Theoretical prediction (passive small-signal, N-cell matched "
            "cascade). Realized ADS overlay: pending (numerical leg).",
            ha="center", fontsize=7, color="0.35",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(path)
        return True
    finally:
        plt.close(fig)


def _save_nsweep_figure(path: Path, candidate: dict[str, Any]) -> bool:
    """F3: predicted coherent-buildup vs N with N_coh + chosen-N markers."""

    coh = candidate.get("coherence")
    if not coh:
        return False
    import numpy as np

    plt, fig, ax = _new_axes()
    try:
        delta = math.radians(float(coh["delta_theta_deg"]))
        chosen_n = int(coh["chosen_N"])
        n_coh = coh.get("N_coh_cells")
        n_coh_finite = n_coh is not None and math.isfinite(float(n_coh))
        law = coh.get("validated_law") or {}
        validated_opt = law.get("measured_optimum_N")
        # Keep both the candidate's N_coh, the chosen N, and the validated optimum
        # in view so the reader sees all three at once.
        n_anchors = [chosen_n]
        if n_coh_finite:
            n_anchors.append(2 * math.ceil(float(n_coh)))
        if validated_opt:
            n_anchors.append(int(validated_opt) + 5)
        n_max = int(min(200, max(*n_anchors, 30)))
        n = np.arange(1, n_max + 1, dtype=float)
        if abs(delta) < 1e-9:
            buildup = n.copy()
        else:
            buildup = np.abs(np.sin(n * delta / 2.0) / math.sin(delta / 2.0))
        boost_db = 20.0 * np.log10(np.maximum(buildup, 1e-9))
        ax.plot(n, boost_db, color="#1f77b4", lw=1.9,
                label="this candidate: coherent-buildup proxy (analytical cell)")
        if n_coh_finite:
            ax.axvline(float(n_coh), color="#2ca02c", ls="--", lw=1.3,
                       label=f"this candidate N_coh ≈ {float(n_coh):.1f} (heuristic)")
        ax.axvline(chosen_n, color="#d62728", ls=":", lw=1.3,
                   label=f"chosen N = {chosen_n}")
        if validated_opt:
            ax.axvline(int(validated_opt), color="#9467bd", ls="-.", lw=1.6,
                       label=f"measured robust optimum ≈ {int(validated_opt)} "
                             f"(coherence-length design)")
        ax.set_xlabel("Number of cells N")
        ax.set_ylabel("Relative coherent buildup (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
        if not coh.get("is_phase_matched", False):
            ax.text(0.02, 0.03,
                    "gain-max pick: NOT phase-matched (N_coh ≪ N).\n"
                    "Measured robust optimum ≈ 15 cells @ 2 GHz is on the\n"
                    "coherence-length design (ADS sweeps; see LIMITATIONS.md).",
                    transform=ax.transAxes, fontsize=6.8, color="0.25", va="bottom")
        ax.set_title(
            "Coherence-length N-sweep — THIS candidate (analytical), NOT "
            "ADS-verified",
            fontsize=9.5,
        )
        fig.text(
            0.5, 0.005,
            "Per-candidate heuristic on the simplified analytical cell; NOT the "
            "measured optimum. Measured robust optimum ≈ 15 cells @ 2 GHz "
            "(coherence-length design). Realized boost-vs-N: numerical (ADS) leg, "
            "pending.",
            ha="center", fontsize=6.8, color="0.35",
        )
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(path)
        return True
    finally:
        plt.close(fig)


def _save_topology_figure(
    path: Path, candidate: dict[str, Any], summary: dict[str, Any]
) -> bool:
    """F1: clean ladder schematic of the found design (pure-Python, from params)."""

    from matplotlib.patches import Rectangle

    plt, fig, ax = _new_axes(figsize=(7.6, 4.2), dpi=110)
    try:
        n_cells = int(summary["config"]["n_cells"])
        f_s = float(summary["target_frequency_GHz"])
        topology = str(candidate["topology"])
        termination = str(candidate["termination"])
        via = str(candidate["via"])
        z_tl = candidate["main_tl"]["z_tl_ohm"]
        w_main_mm = candidate["main_tl"]["width_mm"]
        in_match = candidate["input_match"]
        out_match = candidate["output_match"]
        n_drawn = 4  # representative cells drawn explicitly; total annotated as x N

        y_main = 0.0
        x0, x1 = 0.0, 10.0
        ax.plot([x0, x1], [y_main, y_main], color="black", lw=2.2, zorder=2)

        # Ports + matching boxes.
        def _match_box(x, label):
            ax.add_patch(Rectangle((x - 0.55, y_main - 0.3), 1.1, 0.6,
                                    fill=False, ec="#555", lw=1.3, zorder=3))
            ax.text(x, y_main + 0.55, label, ha="center", va="bottom", fontsize=7.5)

        ax.text(x0 - 0.15, y_main, "50 Ω", ha="right", va="center", fontsize=8)
        ax.text(x1 + 0.15, y_main, "50 Ω", ha="left", va="center", fontsize=8)
        _match_box(1.0, "IN match\n" + (in_match["topology"] if in_match else "none"))
        _match_box(9.0, "OUT match\n" + (out_match["topology"] if out_match else "none"))

        # Unit-cell taps with shunt stub + varactor + termination.
        taps_x = [3.0, 4.6, 6.2, 7.8]

        def _draw_cell(xc, last):
            ax.plot([xc, xc], [y_main, y_main - 0.7], color="black", lw=1.6, zorder=2)
            ax.add_patch(Rectangle((xc - 0.16, y_main - 1.5), 0.32, 0.8,
                                    fill=True, fc="#cfe2f3", ec="#1f4e79", lw=1.1, zorder=3))
            ax.text(xc + 0.22, y_main - 1.1, "stub", fontsize=6.5, va="center")
            # varactor (capacitor symbol) below the stub
            yv = y_main - 1.5
            ax.plot([xc, xc], [yv, yv - 0.18], color="black", lw=1.4)
            ax.plot([xc - 0.16, xc + 0.16], [yv - 0.18, yv - 0.18], color="black", lw=1.8)
            ax.plot([xc - 0.16, xc + 0.16], [yv - 0.34, yv - 0.34], color="black", lw=1.8)
            ax.annotate("", xy=(xc + 0.2, yv - 0.13), xytext=(xc - 0.2, yv - 0.39),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color="#777"))
            ax.text(xc + 0.24, yv - 0.26, "var", fontsize=6.5, va="center")
            yend = yv - 0.34
            ax.plot([xc, xc], [yend, yend - 0.18], color="black", lw=1.4)
            if termination == "open":
                ax.plot([xc - 0.13, xc + 0.13], [yend - 0.18, yend - 0.18],
                        color="black", lw=1.6)
                if last:
                    ax.text(xc + 0.2, yend - 0.18, "open", fontsize=6.5, va="center")
            else:  # short / via to ground
                gy = yend - 0.18
                for i, hw in enumerate((0.18, 0.12, 0.06)):
                    ax.plot([xc - hw, xc + hw], [gy - i * 0.06, gy - i * 0.06],
                            color="black", lw=1.4)
                if last:
                    lab = "via gnd" if via != "none" else "short"
                    ax.text(xc + 0.22, gy, lab, fontsize=6.5, va="center")

        for i, xc in enumerate(taps_x[:n_drawn]):
            _draw_cell(xc, last=(i == n_drawn - 1))
        ax.text((taps_x[0] + taps_x[-1]) / 2.0, y_main + 0.45,
                f"unit cell  × {n_cells}", ha="center", va="bottom",
                fontsize=8.5, style="italic")

        z_tl_s = f"{z_tl:.1f}" if isinstance(z_tl, (int, float)) else "n/a"
        w_s = f"{w_main_mm:.3g}" if isinstance(w_main_mm, (int, float)) else "n/a"
        nseg = candidate.get("n_segments")
        subtitle = (
            f"family {topology} | {nseg}-seg {termination} stub"
            f"{' + via' if via != 'none' else ''} | "
            f"main Z_TL={z_tl_s} Ω, w={w_s} mm | N={n_cells} cells | "
            f"f_s={f_s:.3g} GHz, pump 2 f_s={2 * f_s:.3g} GHz"
        )
        ax.set_title("Predicted topology schematic — analytical (from design "
                     "parameters)", fontsize=10)
        ax.text(5.0, -3.05, subtitle, ha="center", va="top", fontsize=7.5, color="0.2")
        ax.set_xlim(-0.8, 10.8)
        ax.set_ylim(-3.4, 1.3)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path)
        return True
    finally:
        plt.close(fig)


def _generate_figures(out_dir: Path, candidate: dict[str, Any], summary: dict[str, Any]) -> dict[str, str | None]:
    """Write F1/F2/F3 PNGs into ``out_dir``; return {key: filename or None}."""

    names = {
        "topology": "fig_topology.png",
        "sparams": "fig_predicted_sparams.png",
        "nsweep": "fig_predicted_nsweep.png",
    }
    figures: dict[str, str | None] = {"topology": None, "sparams": None, "nsweep": None}
    try:
        if _save_topology_figure(out_dir / names["topology"], candidate, summary):
            figures["topology"] = names["topology"]
        if _save_sparams_figure(out_dir / names["sparams"], candidate):
            figures["sparams"] = names["sparams"]
        if _save_nsweep_figure(out_dir / names["nsweep"], candidate):
            figures["nsweep"] = names["nsweep"]
    except Exception as exc:  # noqa: BLE001  (figures are best-effort; never fail the run)
        print(f"warning: figure generation incomplete ({exc})", file=sys.stderr)
    return figures


def _fmt(value: Any, spec: str = ".3g", suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):{spec}}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _geometry_lines(candidate: dict[str, Any]) -> list[str]:
    """Report lines for stub/matching geometry with electrical length (G2)."""

    lines: list[str] = []
    stub_segs = candidate.get("stub_parameters") or []
    if stub_segs:
        lines.append("- Shunt stub segments (mm | electrical degrees @ f_s):")
        for p in stub_segs:
            lines.append(
                f"  - s{p['segment']}: w={_fmt(p['width_mm'], '.4g')} mm, "
                f"l={_fmt(p['length_mm'], '.4g')} mm "
                f"({_fmt(p['length_deg'], '.4g')}°)"
            )
    for side, key in (("Input", "input_match"), ("Output", "output_match")):
        section = candidate.get(key)
        if not section:
            lines.append(f"- {side} match: none")
            continue
        lines.append(f"- {side} match ({section['topology']}):")
        for el in section["elements"]:
            lines.append(
                f"  - {el['kind']}: w={_fmt(el['width_mm'], '.4g')} mm, "
                f"l={_fmt(el['length_mm'], '.4g')} mm "
                f"({_fmt(el['length_deg'], '.4g')}°), "
                f"Z0={_fmt(el['z0_ohm'], '.4g')} Ω"
            )
    return lines


def _write_report(
    path: Path,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    figures: dict[str, str | None] | None = None,
) -> None:
    figures = figures or {}
    f_s = float(summary["target_frequency_GHz"])
    lines = [
        "# Optimizer Run — Theoretical (Predicted) Leg",
        "",
        f"Target signal frequency (f_s): {f_s:.6g} GHz",
        f"Degenerate pump frequency (2 f_s): {2 * f_s:.6g} GHz",
        f"Workflow: {summary['workflow']}",
        f"Cells (N): {summary['config']['n_cells']}",
        f"Screened candidates: {summary['counts']['screened']}",
        f"Refined top-k: {summary['config']['top_k']}",
        f"Selected candidates: {summary['counts']['diverse']}",
        "",
        "> **PREDICTION ONLY — analytical, NOT ADS-verified.** Every number and "
        "every figure in this report is an analytical prediction from the "
        "pure-Python optimizer (`run_ads=False`).",
        "> " + PREDICTED_GAIN_QUALIFIER,
        "> ADS validation is optional and separate; no `.env` file is required for "
        "this pure-Python optimizer run.",
        "",
    ]
    if not candidates:
        lines.extend(["No positive-gain candidates were selected.", ""])
        _append_pending_numerical(lines, figures)
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    best = candidates[0]
    metrics = best["metrics_at_drive"]
    varac = summary.get("varactor", {})
    coherence = best.get("coherence") or {}
    stability = best.get("analytical_stability") or {}
    sweep = best.get("predicted_s_sweep") or {}

    lines.extend(
        [
            "## Best Candidate (predicted)",
            "",
            f"- Topology: {best['topology']}",
            f"- Predicted coherent peak (analytical proxy, optimistic; see note "
            f"above): {_fmt(metrics.get('coherent_peak'))} dB — PREDICTION, not "
            "ADS-verified",
            f"- Main line Z_TL: {_fmt(best['main_tl']['z_tl_ohm'])} Ω",
            f"- Main line width: {_fmt(best['main_tl']['width_mm'])} mm",
            f"- Operating varactor capacitance (G7, predicted): "
            f"V_bias = {_fmt(varac.get('V_bias_v'), '.3g', ' V')}, "
            f"C(V_bias) = {_fmt(varac.get('C_at_bias_pF'), '.4g', ' pF')}",
            f"- Analytical stability margin 1−max|λ_k| (G4, proxy): "
            f"{_fmt(stability.get('margin'), '.4g')} "
            f"(>0 sub-threshold/stable, <0 above-threshold/oscillator; "
            f"max|λ|={_fmt(stability.get('max_abs_lambda'), '.4g')} at "
            f"{_fmt(stability.get('pump_dBm'), '.3g', ' dBm')})",
            f"- Coherence-length heuristic for THIS candidate (G5; coarse, on "
            f"the simplified analytical Stage-2 cell — NOT the measured "
            f"coherence-length optimum): "
            f"N_coh = π/|θ(2f_s)−2θ(f_s)| = "
            f"{_fmt(coherence.get('N_coh_cells'), '.4g')} cells "
            f"(chosen N = {coherence.get('chosen_N', summary['config']['n_cells'])}; "
            f"θ(f_s)={_fmt(coherence.get('theta_fs_deg'), '.3g', '°')}, "
            f"θ(2f_s)={_fmt(coherence.get('theta_2fs_deg'), '.3g', '°')}). "
            + (
                "This gain-max pick is NOT phase-matched (N_coh ≪ N, margin < 0); "
                "see the coherence-length note below."
                if not coherence.get("is_phase_matched", False)
                else "See the coherence-length note below."
            ),
            "",
        ]
    )

    # Coherence-length reconciliation (the candidate heuristic vs the law).
    _append_coherence_note(lines, coherence, f_s)

    # Predicted S-parameters (G1 / F2).
    lines.extend(
        [
            "## Predicted S-parameters (analytical, not ADS-verified) — G1/F2",
            "",
            "Passive small-signal S-parameters of the N-cell Bloch-matched "
            "cascade (unpumped structure; Stage-3 input/output matching not "
            "folded in). Grid: "
            f"{_fmt(S_SWEEP_LO_FACTOR * f_s)}–{_fmt(S_SWEEP_HI_FACTOR * f_s)} "
            f"GHz, {S_SWEEP_N_POINTS} points, "
            f"Z_ref = {S_SWEEP_Z_REF_OHM:.0f} Ω.",
            "",
        ]
    )
    if sweep:
        lines.extend(
            [
                "| Frequency | S11 (dB) | S21 (dB) |",
                "| --- | --- | --- |",
                f"| f_s = {_fmt(sweep.get('f_s_GHz'))} GHz | "
                f"{_fmt(sweep.get('s11_at_fs_dB'), '.3g')} | "
                f"{_fmt(sweep.get('s21_at_fs_dB'), '.3g')} |",
                f"| 2 f_s = {_fmt(sweep.get('f_pump_GHz'))} GHz | "
                f"{_fmt(sweep.get('s11_at_2fs_dB'), '.3g')} | "
                f"{_fmt(sweep.get('s21_at_2fs_dB'), '.3g')} |",
                "",
            ]
        )
    if figures.get("sparams"):
        lines.extend(
            [
                f"![Predicted S-parameters (analytical, not ADS-verified); "
                f"realized overlay pending]({figures['sparams']})",
                "",
            ]
        )

    # Topology schematic (F1).
    if figures.get("topology"):
        lines.extend(
            [
                "## Predicted topology schematic — F1",
                "",
                f"![Predicted topology schematic of the found design "
                f"(analytical, from design parameters)]({figures['topology']})",
                "",
            ]
        )

    # Predicted N-sweep (F3).
    if figures.get("nsweep"):
        lines.extend(
            [
                "## Predicted coherence-length N-sweep — F3",
                "",
                f"![Predicted coherent-buildup vs N for this candidate "
                f"(analytical coherence-length proxy; this candidate's N_coh and "
                f"chosen N marked, with the measured robust optimum overlaid for "
                f"reference; not ADS-verified)]"
                f"({figures['nsweep']})",
                "",
            ]
        )

    # Geometry + electrical lengths (G2).
    geom = _geometry_lines(best)
    if geom:
        lines.extend(
            [
                "## Geometry — electrical lengths (G2, predicted)",
                "",
                "Physical lengths in mm with electrical length in degrees at f_s.",
                "",
                *geom,
                "",
            ]
        )

    _append_pending_numerical(lines, figures)
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_coherence_note(lines: list[str], coherence: dict[str, Any], f_s: float) -> None:
    """Reconcile the candidate coherence heuristic with the measured optimum (G5)."""

    if not coherence:
        return
    law = coherence.get("validated_law") or {}
    n_coh = coherence.get("N_coh_cells")
    chosen_n = coherence.get("chosen_N")
    if law.get("f_s_anchored"):
        measured = (
            f"At {f_s:.6g} GHz the **measured robust optimum length is "
            f"~{law.get('measured_optimum_N')} cells**, established by direct ADS "
            f"pump/length sweeps on the coherence-length design (n3_single_short, "
            f"accurate VIAGND/MLEF terminations); see "
            f"[docs/LIMITATIONS.md](../../docs/LIMITATIONS.md) "
            f"(\"Optimal length and its mechanism\")."
        )
        consistency = (
            f"This is consistent, at the dispersion level, with degenerate "
            f"phase-matching; the exact analytical coherence length N_coh depends "
            f"on the cell model and is under derivation for a follow-up. The value "
            f"reported by prior band-edge analysis at this frequency, ≈"
            f"{_fmt(law.get('predicted_N_coh_cells'), '.3g')} cells, is not "
            f"reproduced by this public optimizer's simplified mid-band cell."
        )
    else:
        measured = (
            "The measured robust optimum length is ~15 cells at the 2.0 GHz "
            "design and ~18 at 1.7 GHz, established by direct ADS pump/length "
            "sweeps on the coherence-length design (accurate VIAGND/MLEF "
            "terminations); see "
            "[docs/LIMITATIONS.md](../../docs/LIMITATIONS.md) "
            "(\"Optimal length and its mechanism\"). It tracks the design "
            "frequency, and this run's f_s is not one of the documented anchors."
        )
        consistency = (
            "This is consistent, at the dispersion level, with degenerate "
            "phase-matching; the exact analytical coherence length depends on the "
            "cell model and is under derivation for a follow-up. Coherence-length "
            "values reported by prior band-edge analysis (≈15.3 cells at 2.0 GHz, "
            "≈18.5 at 1.7 GHz) are not reproduced by this public optimizer's "
            "simplified mid-band cell."
        )
    lines.extend(
        [
            "## Coherence length — this candidate vs the measured optimum (G5)",
            "",
            f"The {_fmt(n_coh, '.3g')}-cell figure above is **this candidate's "
            f"coarse heuristic**, computed on the simplified analytical Stage-2 "
            f"cell (a mid-band cell, per-cell phase ~95°). It is NOT the method's "
            f"headline prediction and NOT the measured coherence-length optimum: "
            f"a gain-max pick is generally not the coherence-length design and is "
            f"generally not phase-matched (here N_coh ≈ {_fmt(n_coh, '.2g')} ≪ N "
            f"= {chosen_n}, and the analytical stability margin is < 0).",
            "",
            f"{measured} {consistency} The reported band-edge value lives on the "
            "slow-wave BAND-EDGE realized cell (per-cell phase ~133–156°), a "
            "different and more rigorous cell than the simplified analytical cell "
            "used for the per-candidate heuristic here; the robust optimum itself "
            "is corroborated by direct measurement. See also "
            "[docs/methodology/OPTIMIZER_METHODOLOGY.md]"
            "(../../docs/methodology/OPTIMIZER_METHODOLOGY.md) "
            "(\"Length selection\").",
            "",
            f"In short: **~{law.get('measured_optimum_N') or 15} cells is the "
            "measured robust optimum (coherence-length design); the small N_coh "
            "here is this gain-max candidate's off-phase-matching value on a "
            "coarse cell** — the two are different designs measured on different "
            "cells, not a contradiction.",
            "",
        ]
    )


def _append_pending_numerical(lines: list[str], figures: dict[str, str | None]) -> None:
    lines.extend(
        [
            "## Numerical verification (ADS): pending",
            "",
            "This is the **theoretical (predicted) leg**. The numerical column "
            "is filled by the numerical (ADS) leg and will add, alongside each "
            "prediction above:",
            "",
            "- ADS harmonic-balance realized gain (the realized counterpart to "
            "the optimistic coherent-gain proxy);",
            "- realized S-parameters **overlaid on the predicted S-parameter "
            "plot above** (F2 is laid out with headroom for the overlay);",
            "- the pump-power sweep and the realized boost-vs-N curve "
            "**overlaid on the predicted N-sweep** (F3);",
            "- the realized stability class for the analytical margin above.",
            "",
            "Until then, treat every figure in this report as an analytical "
            "prediction (optimistic; not ADS-verified).",
            "",
        ]
    )


def _run_optimize_amplifier(args: argparse.Namespace) -> int:
    from tvtl.optimization.topology_generator import linear_families
    from tvtl.optimization.workflow import WorkflowConfig, decoupled_design_workflow

    families = tuple(linear_families())
    samples_per_family = args.samples_per_family
    if samples_per_family is None and args.max_candidates is not None:
        samples_per_family = max(1, math.ceil(args.max_candidates / len(families)))
    if samples_per_family is None:
        samples_per_family = QUICK_SAMPLES_PER_FAMILY if args.quick else DEFAULT_SAMPLES_PER_FAMILY

    top_k = args.top_k if args.top_k is not None else (QUICK_TOP_K if args.quick else DEFAULT_TOP_K)
    top_m = args.top_m if args.top_m is not None else (QUICK_TOP_M if args.quick else DEFAULT_TOP_M)
    refine_maxiter = (
        args.refine_maxiter
        if args.refine_maxiter is not None
        else (QUICK_REFINE_MAXITER if args.quick else DEFAULT_REFINE_MAXITER)
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    varactor = _default_varactor()
    cfg = WorkflowConfig(
        n_samples=samples_per_family,
        seed=args.seed,
        top_k=top_k,
        top_m=top_m,
        n_cells=args.n_cells,
        gate=getattr(args,"gate","none"),
        gate_options=({"bias_return_ohm": args.bias_return_ohm}
                      if getattr(args, "bias_return_ohm", None) is not None else None),
        order_by=getattr(args,"order_by","absolute_outin_dB"),
        drive_dBm=args.drive_dbm,
        refine_maxiter=refine_maxiter,
        min_coherent_dB=args.min_coherent_db,
        pumps_dBm=(args.drive_dbm,),
        phases_deg=(0.0,) if args.quick else (0.0, 72.0, 144.0, 216.0, 288.0),
        run_ads=False,
        families=families,
    )
    result = decoupled_design_workflow(args.freq_ghz * 1e9, varactor, cfg)
    candidates = [
        _candidate_dict(report, rank, args.drive_dbm)
        for rank, report in enumerate(result.reports, start=1)
    ]
    rows = _candidate_rows(candidates)

    summary_path = out_dir / "summary.json"
    candidates_json_path = out_dir / "candidates.json"
    candidates_csv_path = out_dir / "candidates.csv"
    report_path = out_dir / "report.md"

    summary = {
        "workflow": "optimize-amplifier",
        "objective": (
            "Amplifier-like analytical design search using the existing "
            "decoupled TVTL workflow; selected candidates are ranked by "
            "the selected order_by gain metric at the requested signal frequency."
        ),
        "target_frequency_GHz": float(args.freq_ghz),
        "target_frequency_Hz": float(args.freq_ghz * 1e9),
        "varactor": _varactor_summary(varactor),
        "config": {
            "samples_per_family": int(samples_per_family),
            "linear_families": len(families),
            "top_k": int(top_k),
            "top_m": int(top_m),
            "n_cells": int(args.n_cells),
            "drive_dBm": float(args.drive_dbm),
            "refine_maxiter": int(refine_maxiter),
            "min_coherent_dB": float(args.min_coherent_db),
            "seed": int(args.seed),
            "quick": bool(args.quick),
        },
        "counts": {
            "screened": int(result.n_screened),
            "realizable": int(result.n_realizable),
            "positive_gain": int(result.n_positive_gain),
            "diverse": int(result.n_diverse),
            "reports": len(result.reports),
        },
        "best_candidate": candidates[0] if candidates else None,
        "note": result.note,
        "results_are_predictions": True,
        "prediction_label": PREDICTION_LABEL,
        "ads_validation": {
            "required": False,
            "run_ads": False,
            "note": "ADS/HB/OpenAccess validation is optional and separate.",
        },
        "numerical_verification": {
            "status": "pending",
            "note": (
                "Theoretical (predicted) leg only. Realized gain, realized "
                "S-parameter overlay, pump-power sweep, realized boost-vs-N, and "
                "the realized stability class are filled by the numerical (ADS) "
                "leg."
            ),
        },
        "env_required": False,
        "files": {
            "summary_json": str(summary_path),
            "candidates_json": str(candidates_json_path),
            "candidates_csv": str(candidates_csv_path),
            "report_md": str(report_path),
        },
    }

    figures = (
        _generate_figures(out_dir, candidates[0], summary) if candidates else {}
    )
    summary["figures"] = {
        "topology_png": figures.get("topology"),
        "predicted_sparams_png": figures.get("sparams"),
        "predicted_nsweep_png": figures.get("nsweep"),
        "label": PREDICTION_LABEL,
    }

    _write_json(summary_path, summary)
    _write_json(candidates_json_path, candidates)
    _write_csv(candidates_csv_path, rows)
    _write_report(report_path, summary, candidates, figures)

    print(f"Target frequency: {args.freq_ghz:.6g} GHz")
    print(
        "Candidates: "
        f"{result.n_screened} screened, top-{top_k} refined, "
        f"{result.n_realizable} realizable, {result.n_diverse} selected"
    )
    if candidates:
        best = candidates[0]
        metrics = best["metrics_at_drive"]
        print(
            "Best candidate: "
            f"{best['topology']} "
            f"(coherent_peak={metrics.get('coherent_peak'):.3g} dB, "
            f"Z_TL={best['main_tl']['z_tl_ohm']:.3g} ohm, "
            f"main_width={best['main_tl']['width_mm']:.3g} mm)"
        )
        print(
            "Predicted-leg figures: "
            + ", ".join(
                str(out_dir / name)
                for name in (
                    figures.get("topology"),
                    figures.get("sparams"),
                    figures.get("nsweep"),
                )
                if name
            )
        )
    else:
        print("Best candidate: none selected; inspect summary.json for run metadata.")
    print(f"Output path: {out_dir}")
    print(
        "Predictions only (analytical, not ADS-verified); ADS validation is "
        "optional and separate, and no .env file is required for this pure "
        "Python optimizer."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    from tvtl_ads_builder_v2.cli import add_export_gui_project_parser

    parser = argparse.ArgumentParser(
        prog="floquet-line-design",
        description="Public command-line tools for Floquet line design workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    opt = subparsers.add_parser(
        "optimize-amplifier",
        help="Run a pure Python amplifier-like optimizer search.",
        description=(
            "Run the existing decoupled TVTL design workflow around a requested "
            "signal frequency and write summary/candidate files. ADS validation "
            "is optional and is not required by this command."
        ),
    )
    opt.add_argument("--freq-ghz", required=True, type=_positive_float, help="Target signal frequency in GHz.")
    opt.add_argument("--out", required=True, type=Path, help="Output directory for summary and candidate files.")
    opt.add_argument("--quick", action="store_true", help="Use a small smoke-run budget.")
    budget = opt.add_mutually_exclusive_group()
    budget.add_argument(
        "--samples-per-family",
        type=_positive_int,
        default=None,
        help="Sobol sample budget per linear topology family.",
    )
    budget.add_argument(
        "--max-candidates",
        type=_positive_int,
        default=None,
        help="Approximate total Sobol sample budget across linear topology families.",
    )
    opt.add_argument("--top-k", type=_positive_int, default=None, help="Number of candidates refined after screening.")
    opt.add_argument("--top-m", type=_positive_int, default=None, help="Number of diverse designs to report.")
    opt.add_argument("--refine-maxiter", type=_positive_int, default=None, help="Maximum Nelder-Mead iterations per candidate.")
    opt.add_argument("--seed", type=int, default=0, help="Random seed for Sobol scrambling.")
    opt.add_argument("--n-cells", type=_positive_int, default=15, help="Number of TVTL cells in the analytical cascade.")
    opt.add_argument("--order-by", choices=("absolute_outin_dB","coherent_peak"), default="absolute_outin_dB", help="Gain ranking metric; coherent_peak replays legacy boost ordering.")
    opt.add_argument("--gate", choices=("none","stage0","cascade"), default="none",
                     help="Passive feasibility prefilter before scoring; cascade adds the phase margin 0.5 m_del - m_EP >= 0.")
    opt.add_argument("--bias-return-ohm", type=float, default=None,
                     help="Bias-return resistor (ohm) for OPEN-stub families; without it the gate refuses them "
                          "(no_dc_return_for_varactor).")
    opt.add_argument("--drive-dbm", type=float, default=-3.0, help="Control-port drive used for scoring/reporting.")
    opt.add_argument("--min-coherent-db", type=float, default=0.0, help="Minimum coherent peak gain for selection.")
    opt.set_defaults(func=_run_optimize_amplifier)

    add_export_gui_project_parser(
        subparsers,
        name="ads-export",
        help_text="Export optimizer candidates for ADS GUI inspection.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except Exception as exc:
        if exc.__class__.__name__ == "ExportError":
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
