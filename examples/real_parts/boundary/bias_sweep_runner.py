"""R3 replay -- regenerates ``results/smv1245_bias_sweep.json``.

In-silico SMV1245 effective-reverse-bias sweep on the fabricated board: does a
modest reverse bias recover a double-digit pumped boost? Geometry frozen at the
15-cell solution; only ``v_bias_v`` (and pump power) varies.

Conditions
----------
* f_signal = 3.1 GHz, f_pump = 6.2 GHz (= 2 f_s), 15 cells, signal -20 dBm,
  P_avs = 1e-5 W; 24 pump phases (0..345 deg, 15 deg step).
* biases_V = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]; pumps = [29, 20] dBm;
  pump-off reference at -100 dBm.
* SMV1245-079LF: CJ0 6.9 pF, VJ 3.5 V, M 1.7, CP 0.47 pF, RS 2.0 ohm,
  L_package 0.7 nH.

Replayability: requires the ADS python (``run_full_sum_gain_sweep`` runs ADS
harmonic balance per bias x pump). Reads the committed
``results/headline_smv1245.json`` for the baseline operating point. Writes HB
scratch under ``examples/real_parts/outputs/bias_sweep`` (gitignored).
"""
import json, math, sys, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples" / "real_parts"))

from real_part_designs import spec_smv1245  # noqa: E402
from tvtl_ads_builder_v2.oa.psa_gain_artifact import run_full_sum_gain_sweep  # noqa: E402

PHASES_24 = tuple(float(d) for d in range(0, 360, 15))
OUTPUTS = REPO / "examples" / "real_parts" / "outputs" / "bias_sweep"   # gitignored scratch
RESULTS = REPO / "examples" / "real_parts" / "results"
OUT_JSON = RESULTS / "smv1245_bias_sweep.json"

BIASES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
PUMPS = [29.0, 20.0]

# SMV1245-079LF palette constants (src/tvtl/varactor_palette.py)
CJ0_pF, VJ_V, M, CP_pF, RS_ohm, LPKG_nH = 6.9, 3.5, 1.7, 0.47, 2.0, 0.7


def _fin(x):
    """JSON-safe: non-finite float -> None."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def cj_srf(v):
    cj_pF = CJ0_pF / (1.0 + v / VJ_V) ** M           # bare junction C(v)  [pF]
    ctot_pF = cj_pF + CP_pF                            # junction + package CP
    L = LPKG_nH * 1e-9

    def srf(c_pF):
        return 1.0 / (2.0 * math.pi * math.sqrt(L * c_pF * 1e-12)) / 1e9  # GHz
    return {"v_bias_V": v, "Cj_pF": cj_pF, "Cj_plus_Cp_pF": ctot_pF,
            "SRF_Cj_GHz": srf(cj_pF), "SRF_Cj_plus_Cp_GHz": srf(ctot_pF)}


def ads_version():
    exe = Path(sys.executable)
    tag = next((p for p in exe.parts if p.upper().startswith("ADS")), "unknown")
    return {"install_tag": tag, "interpreter": "<ADS_install>/tools/python/python.exe",
            "python": sys.version.split()[0]}


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    # Step 1: baseline echo from committed operating-point record.
    base = json.loads((RESULTS / "headline_smv1245.json").read_text(encoding="utf-8"))
    baseline = {
        "source": "results/headline_smv1245.json",
        "boost_dB_pump_on_over_off": base["boost_dB_pump_on_over_off"],
        "net_out_in_G_mod_in_dB": base["net_out_in_G_mod_in_dB"],
        "peak_phase_deg": base["peak_phase_deg"],
        "f_signal_GHz": base["f_signal_GHz"], "f_pump_GHz": base["f_pump_GHz"],
        "pump_power_dBm": base["pump_power_dBm"], "signal_power_dBm": base["signal_power_dBm"],
        "v_bias_V": 0.0,
    }
    print("BASELINE (headline_smv1245.json): boost=%+.5f dB  net=%+.3f dB  peak_phase=%.0f deg"
          % (baseline["boost_dB_pump_on_over_off"], baseline["net_out_in_G_mod_in_dB"],
             baseline["peak_phase_deg"]), flush=True)

    payload = {
        "headline": "SMV1245 fabricated-board in-silico effective-reverse-bias sweep "
                    "(geometry frozen; does modest bias recover double-digit pumped boost?)",
        "board": "fabricated SMV1245-079LF, frozen 15-cell geometry, f_pump = 2 f_signal",
        "definition_boost": "boost_dB = coh_full_dB = 20*log10(|Sum V_(m+2n=1)|_on / |V(1,0)_off|) "
                            "(full-family pump-on/off, phase-peaked)",
        "definition_net": "net_G_mod_in_dB = 10*log10(0.5|Sum V|^2/50 / P_avs) at the boost-peak phase",
        "conditions": {
            "f_signal_GHz": 3.1, "f_pump_GHz": 6.2, "signal_power_dBm": -20.0,
            "P_avs_W": 1e-05, "n_cells": 15, "phases_deg": list(PHASES_24),
            "off_pump_power_dBm": -100.0, "pump_powers_dBm": PUMPS, "biases_V": BIASES,
            "varactor": "SMV1245-079LF (Skyworks Si hyperabrupt); CJ0=6.9pF VJ=3.5V M=1.7 "
                        "CP=0.47pF RS=2.0ohm L_package=0.7nH",
            "geometry_frozen": True,
        },
        "ads": ads_version(),
        "generator": "boundary/bias_sweep_runner.py (additive; spec_smv1245(v_bias_v=..., "
                     "pump_dBm=...) x run_full_sum_gain_sweep, PHASES_24).",
        "baseline_operating_point": baseline,
        "cj_srf_table": [cj_srf(v) for v in BIASES],
        "rows": [],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for bias in BIASES:
        for pump in PUMPS:
            tag = "b%s_p%s" % (str(bias).replace(".", "p"), str(pump).replace(".", "p"))
            row = {"v_bias_V": bias, "pump_power_dBm": pump}
            print("RUN bias=%.2f V pump=%.1f dBm ..." % (bias, pump), flush=True)
            try:
                sw = run_full_sum_gain_sweep(
                    spec_smv1245(v_bias_v=bias, pump_dBm=pump), OUTPUTS / tag,
                    phases_deg=PHASES_24, off_pump_power_dBm=-100.0, timeout_s=1200)
                curve = [{"phase_deg": float(r["phase_deg"]),
                          "boost_dB": _fin(r["coh_full_dB"])} for r in sw.rows]
                row.update({
                    "all_phases_converged": True,
                    "peak_boost_dB": _fin(sw.peak_coh_full_dB),
                    "peak_boost_phase_deg": float(sw.peak_phase_deg),
                    "net_G_mod_in_dB_at_peak": _fin(sw.g_mod_in_dB),
                    "off_V": _fin(abs(sw.v_sig_off)),
                    "coh_2term_peak_dB": _fin(sw.peak_coh_2term_dB),
                    "correction_dB": _fin(sw.correction_dB),
                    "gamma_in_sq": _fin(sw.gamma_in_sq),
                    "boost_curve": curve,
                })
                print("   -> peak_boost=%+.3f dB @ %.0f deg  net=%+.3f dB  off_V=%.3e"
                      % (row["peak_boost_dB"], row["peak_boost_phase_deg"],
                         row["net_G_mod_in_dB_at_peak"], row["off_V"]), flush=True)
            except Exception as e:  # noqa: BLE001 -- a non-convergent/failed sweep is a data row
                row.update({"all_phases_converged": False,
                            "error": "%s: %s" % (type(e).__name__, e),
                            "traceback": traceback.format_exc()[-800:]})
                print("   -> FAILED: %s: %s" % (type(e).__name__, e), flush=True)
            payload["rows"].append(row)
            OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")  # checkpoint

    print("DONE. wrote %s" % OUT_JSON, flush=True)
    print("=== SUMMARY (bias, pump -> peak_boost, net, converged) ===", flush=True)
    for r in payload["rows"]:
        if r.get("all_phases_converged"):
            print("  %.2f V  %.0f dBm  ->  boost %+.3f dB @ %3.0f deg   net %+.3f dB"
                  % (r["v_bias_V"], r["pump_power_dBm"], r["peak_boost_dB"],
                     r["peak_boost_phase_deg"], r["net_G_mod_in_dB_at_peak"]), flush=True)
        else:
            print("  %.2f V  %.0f dBm  ->  NOT CONVERGED (%s)"
                  % (r["v_bias_V"], r["pump_power_dBm"], r.get("error", "")), flush=True)


if __name__ == "__main__":
    main()
