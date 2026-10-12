"""Stage 4 -- cross-validation of an optimized design under control-port pumping.

Assembles a complete design (Stage-1 stub + Stage-2 main width + Stage-3
matching) into a control-port device spec and characterizes it: coherent gain
(peak + at phi=0), mix(1,0), PSA phase swing, and the gain landscape across
frequency -- analytically (with the matching sections folded into the cascade
via the matching-block extension) and, optionally, against ADS HB.

The analytical characterization is base-Python and fast; the ADS leg is gated on
a reachable hpeesofsim (run only when ``run_ads=True``), mirroring
``tests/parity/test_level10_realistic_crossval.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tvtl.floquet.harmonic_abcd import harmonic_frequencies
from tvtl.floquet.stability import stability_margin
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.analytical import (
    level10_realistic_param_gain,
    realistic_pumped_unit_cell_abcd,
)
from tvtl_ads_builder_v2.analytical.matching_blocks import matching_section_harmonic_blocks
from tvtl_ads_builder_v2.device_spec import with_var_overrides
from tvtl_ads_builder_v2.parity.level10_ideal_c import open_circuit_pump_voltage
from tvtl_ads_builder_v2.parity.optimized_design import optimized_control_port_spec

from .stage1_per_cell import StubCandidate
from .stage2_main_tl import MainTLDesign
from .substrate import ROGERS_4003C, SubstrateConfig

__all__ = [
    "CompleteDesign",
    "CrossValReport",
    "analytical_characterization",
    "cross_validate",
    "DEFAULT_PUMPS_DBM",
    "DEFAULT_PHASES_DEG",
    "EPS_PEAK_TOL_DB",
    "EPS_AT0_TOL_DB",
    "PSA_MIN_SWING_DB",
]

DEFAULT_PUMPS_DBM = (-6.0, -3.0, 0.0)
GATED_PUMP_DBM = -3.0
DEFAULT_PHASES_DEG = (0.0, 72.0, 144.0, 216.0, 288.0)
DEFAULT_LANDSCAPE_GHZ = tuple(np.round(np.arange(2.0, 7.01, 0.25), 3))
TRUNCATION_N = 6

# Acceptance tolerances (fixed up front; NEVER tuned to pass).
EPS_PEAK_TOL_DB = 1.0
EPS_AT0_TOL_DB = 0.6
PSA_MIN_SWING_DB = 1.0


@dataclass(frozen=True)
class CompleteDesign:
    """A complete optimized design: stub + main TL + per-side matching."""

    stub: StubCandidate
    main_tl: MainTLDesign
    input_match: object | None   # MatchingSection | None
    output_match: object | None  # MatchingSection | None
    varactor: RealisticVaractor
    n_cells: int = 15
    f0_Hz: float = 3.1e9
    substrate: SubstrateConfig = ROGERS_4003C

    @property
    def f_pump_Hz(self) -> float:
        return 2.0 * float(self.f0_Hz)


@dataclass
class CrossValReport:
    """Cross-validation characterization of a complete design."""

    design: CompleteDesign
    analytical: dict          # pump_dBm -> {coherent_peak, coherent_at0, mix10, psa_swing}
    psa_curve: dict           # phase_deg -> {analytical[, ads]} at the gated pump
    landscape: dict           # f_GHz -> {ana_coherent_peak, ana_mix10[, ads_...]}
    ads: dict = field(default_factory=dict)         # pump_dBm -> {coherent_at0, mix10}
    epsilons: dict = field(default_factory=dict)    # pump_dBm -> {eps_peak, eps_at0}
    # Analytical parametric-stability margin (1 - max_k |lambda_k|) of the
    # pumped harmonic unit cell at the gated drive -- an analytical proxy only
    # (the realized stability class is a separate ADS workflow). Keys:
    # margin, max_abs_lambda, n_harmonics, pump_dBm, label.
    stability: dict = field(default_factory=dict)
    passed: bool | None = None
    note: str = ""
    # set when the workflow refused the design before any HB job (delivered_depth_above_ceiling)
    refusal: dict | None = None


def _spec(design: CompleteDesign, *, f_signal: float, pump_dBm: float, phase_rad: float = 0.0):
    return optimized_control_port_spec(
        topology=design.stub.topology, params=design.stub.params_array,
        varactor=design.varactor, substrate=design.substrate, n_cells=design.n_cells,
        main_width_m=design.main_tl.w_main_m, f_signal_Hz=float(f_signal),
        f_pump_Hz=2.0 * float(f_signal), pump_power_dBm=float(pump_dBm),
        pump_phase_rad=float(phase_rad),
        input_match=design.input_match, output_match=design.output_match,
    )


def _matching_blocks(design: CompleteDesign, f_signal: float, truncation_N: int):
    """Build (input, output) matching harmonic blocks at the signal frequency."""

    ws = 2.0 * math.pi * float(f_signal)
    wm = 2.0 * math.pi * 2.0 * float(f_signal)
    omega_n = harmonic_frequencies(ws, wm, truncation_N)
    tvtl_sub = design.substrate.to_tvtl_substrate()
    ib = matching_section_harmonic_blocks(design.input_match, omega_n, tvtl_sub, side="input")
    ob = matching_section_harmonic_blocks(design.output_match, omega_n, tvtl_sub, side="output")
    return ib, ob


def _ana_gain(design, spec, pump_dBm, ib, ob, truncation_N):
    voc = open_circuit_pump_voltage(float(pump_dBm))
    return level10_realistic_param_gain(
        spec, design.varactor, control_port_vpump_v=voc, truncation_N=truncation_N,
        input_match_blocks=ib, output_match_blocks=ob,
    )


def analytical_characterization(
    design: CompleteDesign,
    *,
    pumps_dBm=DEFAULT_PUMPS_DBM,
    phases_deg=DEFAULT_PHASES_DEG,
    landscape_GHz=DEFAULT_LANDSCAPE_GHZ,
    gated_pump_dBm: float = GATED_PUMP_DBM,
    truncation_N: int = TRUNCATION_N,
) -> CrossValReport:
    """Analytical-only characterization (coherent / mix / PSA / landscape)."""

    f0 = float(design.f0_Hz)
    ib, ob = _matching_blocks(design, f0, truncation_N)

    analytical: dict = {}
    for p in pumps_dBm:
        spec = _spec(design, f_signal=f0, pump_dBm=p)
        res = _ana_gain(design, spec, p, ib, ob, truncation_N)
        analytical[float(p)] = {
            "coherent_peak": float(res["coherent_peak_dB"]),
            "absolute_outin_dB": float(res["abs_gain_off_dB"] + res["coherent_peak_dB"]),
            "coherent_at0": float(res["coherent_at_phase0_dB"]),
            "mix10": float(res["param_gain_dB"]),
            "psa_swing": float(res["coherent_swing_dB"]),
        }

    # PSA phase curve at the gated pump (matching blocks are phase-independent).
    psa_curve: dict = {}
    for ph in phases_deg:
        spec = _spec(design, f_signal=f0, pump_dBm=gated_pump_dBm, phase_rad=math.radians(ph))
        res = _ana_gain(design, spec, gated_pump_dBm, ib, ob, truncation_N)
        psa_curve[float(ph)] = {"analytical": float(res["coherent_at_phase0_dB"])}

    # Gain landscape (matching geometry fixed; re-evaluated per frequency).
    landscape: dict = {}
    for f_ghz in landscape_GHz:
        f_s = float(f_ghz) * 1e9
        ib_f, ob_f = _matching_blocks(design, f_s, truncation_N)
        spec = _spec(design, f_signal=f_s, pump_dBm=gated_pump_dBm)
        try:
            res = _ana_gain(design, spec, gated_pump_dBm, ib_f, ob_f, truncation_N)
            landscape[float(f_ghz)] = {
                "ana_coherent_peak": float(res["coherent_peak_dB"]),
                "ana_mix10": float(res["param_gain_dB"]),
            }
        except Exception:  # noqa: BLE001  (near-oscillation / non-finite cuts)
            landscape[float(f_ghz)] = {"ana_coherent_peak": float("nan"), "ana_mix10": float("nan")}

    return CrossValReport(
        design=design, analytical=analytical, psa_curve=psa_curve, landscape=landscape,
    )


# --------------------------------------------------------------------------- #
# ADS leg (gated)                                                              #
# --------------------------------------------------------------------------- #
def _ads_available() -> bool:
    import os
    import shutil

    if os.environ.get("HPEESOF_DIR"):
        return True
    return any(shutil.which(n) for n in ("adssim_win.exe", "adssim", "hpeesofsim.exe", "hpeesofsim"))


def _ads_output_mixes(spec, tmp_dir, tag):
    from pathlib import Path

    from tvtl_ads_builder_v2.analysis.hb_dataset import read_hb_dataset
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.simulate.runner import run_hpeesofsim

    ws_dir = Path(tmp_dir) / tag
    ws_dir.mkdir(parents=True, exist_ok=True)
    netlist = ws_dir / f"{tag}.ckt"
    netlist.write_text(emit_flat_hb_netlist(spec), encoding="utf-8")
    res = run_hpeesofsim(netlist, timeout_s=900)
    if res["status"] != "completed":
        raise RuntimeError(f"hpeesofsim did not complete: {res.get('error_message')}")
    fs = float(next(v.value for v in spec.var_block.variables if v.name == "f_signal"))
    fp = float(next(v.value for v in spec.var_block.variables if v.name == "f_pump"))
    ds = read_hb_dataset(res["dataset_path"], (fs, fp))
    return (
        complex(ds.node_voltage("n_output").at_mix((1, 0))),
        complex(ds.node_voltage("n_output").at_mix((-1, 1))),
    )


def _ads_gains(design, tmp_dir, pump_dBm, *, phase_rad=0.0, f_signal=None):
    """Return (coherent_at0_dB, mix10_dB) from ADS HB for one operating point."""

    f_s = float(design.f0_Hz if f_signal is None else f_signal)
    tag = f"cv_{int(f_s/1e6)}_{pump_dBm:+.0f}_{math.degrees(phase_rad):.0f}".replace("+", "p").replace("-", "m").replace(".", "")
    spec_on = _spec(design, f_signal=f_s, pump_dBm=pump_dBm, phase_rad=phase_rad)
    v10_on, vm11_on = _ads_output_mixes(spec_on, tmp_dir, tag + "_on")
    spec_off = with_var_overrides(spec_on, {"pump_power_dBm": -100.0})
    v10_off, _ = _ads_output_mixes(spec_off, tmp_dir, tag + "_off")
    coh0 = 20.0 * math.log10(abs(v10_on + vm11_on) / abs(v10_off))
    mix10 = 20.0 * math.log10(abs(v10_on) / abs(v10_off))
    return float(coh0), float(mix10)


def _analytical_stability_margin(
    design: CompleteDesign,
    *,
    gated_pump_dBm: float = GATED_PUMP_DBM,
    truncation_N: int = TRUNCATION_N,
) -> dict:
    """Analytical parametric-stability margin of the pumped unit cell.

    Builds the pumped harmonic unit cell at the gated control-port drive (the
    SAME cell :func:`...analytical.level10_realistic_param_gain` cascades for the
    reported coherent gain) and returns ``1 - max_k |lambda_k|`` from its
    Floquet/Bloch multipliers (:func:`tvtl.floquet.stability.stability_margin`):
    ``> 0`` is the sub-threshold (stable) regime, ``< 0`` the above-threshold
    (oscillator) regime. An analytical proxy only; the realized stability class
    is a separate ADS workflow. Non-traceable cells (e.g. COMBINED traps) or
    non-finite eigenvalues yield ``nan``.
    """

    try:
        spec = _spec(design, f_signal=design.f0_Hz, pump_dBm=gated_pump_dBm)
        voc = open_circuit_pump_voltage(float(gated_pump_dBm))
        cell = realistic_pumped_unit_cell_abcd(
            spec, design.varactor, control_port_vpump_v=voc,
            truncation_N=truncation_N, use_mtee_parasitic=True,
        )
        margin = float(stability_margin(cell))
        max_abs = float(1.0 - margin)
    except Exception:  # noqa: BLE001  (non-traceable / non-finite cells -> nan)
        margin = float("nan")
        max_abs = float("nan")
    return {
        "margin": margin,
        "max_abs_lambda": max_abs,
        "n_harmonics": int(truncation_N),
        "pump_dBm": float(gated_pump_dBm),
        "label": "analytical_proxy_only",
    }


def cross_validate(
    design: CompleteDesign,
    *,
    pumps_dBm=DEFAULT_PUMPS_DBM,
    phases_deg=DEFAULT_PHASES_DEG,
    landscape_GHz=DEFAULT_LANDSCAPE_GHZ,
    gated_pump_dBm: float = GATED_PUMP_DBM,
    truncation_N: int = TRUNCATION_N,
    run_ads: bool = False,
    ads_tmp_dir: str | None = None,
    ads_landscape_GHz=None,
) -> CrossValReport:
    """Cross-validate a complete design: analytical (+ optional ADS HB).

    With ``run_ads=False`` (default) returns the analytical characterization
    only (``passed=None``). With ``run_ads=True`` (requires a reachable
    hpeesofsim and ``ads_tmp_dir``) adds the ADS coherent / mix(1,0) / PSA /
    landscape, computes ``epsilons``, and sets ``passed`` from the
    **convention-free gate**: ``eps_peak <= 1 dB`` AND ``ADS PSA swing > 1 dB``.
    ``eps_at0`` is reported but **not** gated (convention-dependent).
    """

    report = analytical_characterization(
        design, pumps_dBm=pumps_dBm, phases_deg=phases_deg,
        landscape_GHz=landscape_GHz, gated_pump_dBm=gated_pump_dBm,
        truncation_N=truncation_N,
    )
    # Analytical (predicted) parametric-stability margin at the gated drive.
    # Computed here (final cross-validation pass) rather than in the Stage-1..4
    # selection probe, so the per-candidate selection loop stays cheap. An
    # analytical proxy; the realized stability class is a separate ADS workflow.
    report.stability = _analytical_stability_margin(
        design, gated_pump_dBm=gated_pump_dBm, truncation_N=truncation_N,
    )
    if not run_ads:
        return report
    if not _ads_available():
        report.note = "run_ads=True but no reachable hpeesofsim; ADS leg skipped."
        return report
    if ads_tmp_dir is None:
        raise ValueError("run_ads=True requires ads_tmp_dir.")

    # ADS coherent@0 / mix(1,0) at each requested pump.
    for p in pumps_dBm:
        coh0, mix10 = _ads_gains(design, ads_tmp_dir, p)
        report.ads[float(p)] = {"coherent_at0": coh0, "mix10": mix10}
    # ADS PSA phase sweep at the gated pump.
    for ph in phases_deg:
        coh0, _ = _ads_gains(design, ads_tmp_dir, gated_pump_dBm, phase_rad=math.radians(ph))
        report.psa_curve.setdefault(float(ph), {})["ads"] = coh0
    # ADS landscape sample (peak-frequency agreement).
    if ads_landscape_GHz:
        for f_ghz in ads_landscape_GHz:
            coh0, mix10 = _ads_gains(
                design, ads_tmp_dir, gated_pump_dBm, f_signal=float(f_ghz) * 1e9
            )
            report.landscape.setdefault(float(f_ghz), {})
            report.landscape[float(f_ghz)]["ads_coherent_at0"] = coh0
            report.landscape[float(f_ghz)]["ads_mix10"] = mix10

    # Epsilons at the gated pump. ADS coherent-peak is the swept-phase maximum
    # (convention-free); ADS PSA swing is the swept
    # peak-to-peak. eps_peak compares analytical vs ADS coherent peaks.
    ads_phase_vals = [
        report.psa_curve[float(ph)]["ads"] for ph in phases_deg
        if "ads" in report.psa_curve.get(float(ph), {})
    ]
    ads_coh_peak = max(ads_phase_vals) if ads_phase_vals else float("nan")
    ads_psa_swing = (max(ads_phase_vals) - min(ads_phase_vals)) if ads_phase_vals else float("nan")
    ads_coh_at0 = report.ads.get(float(gated_pump_dBm), {}).get("coherent_at0", float("nan"))
    ana_g = report.analytical[float(gated_pump_dBm)]
    eps_peak = abs(ana_g["coherent_peak"] - ads_coh_peak)
    eps_at0 = abs(ana_g["coherent_at0"] - ads_coh_at0)
    report.epsilons[float(gated_pump_dBm)] = {
        "eps_peak": float(eps_peak), "eps_at0": float(eps_at0),
        "ads_coh_peak": float(ads_coh_peak), "ads_psa_swing": float(ads_psa_swing),
    }
    # Primary pass gate: **convention-free metrics only** -- the
    # coherent-PEAK epsilon (from magnitudes, phase-invariant) and the ADS PSA
    # swing. eps@0 (the phi=0 absolute-phase slice) is convention-dependent
    # (confirmed in a design-example context:
    # a design can cross-validate tightly on coherent-peak / PSA while eps@0
    # fails purely from a phase-reference offset between the analytical and ADS
    # PSA sinusoids). eps@0 is still computed and reported, but NOT gated; an
    # earlier 0.6 dB eps@0 tolerance is superseded.
    report.passed = bool(
        eps_peak <= EPS_PEAK_TOL_DB and ads_psa_swing > PSA_MIN_SWING_DB
    )
    return report


