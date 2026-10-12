"""STEP-2 analysis for the three optimizer-picked PROXY example designs.

For each design (``proxy_designs.EXAMPLES_PROXY``) this reports, kept strictly
separate, the analytical PREDICTION and the ADS-realized RECORD, plus a stability
verdict -- the same method that characterised the n2_double_short generalization
test:

  1. PUMP-POWER SCAN @ phase 0 (flat-deck HB on/off): find the CONVERGENT window
     and the pump-on/off boost; the operating point is the convergent pump that
     maximises the phase-0 boost (the boost-optimal convergent pump). The lowest
     NON-convergent pump is the oscillation-edge onset (HB cannot find a steady
     state) -- the example must NOT sit there.
  2. PHASE SWEEP @ the operating pump (run_full_sum_gain_sweep): a BOUNDED single
     lobe (finite peak + anti-phase deamplification trough below 0 dB) is the
     stable degenerate-PSA signature -- not a runaway.
  3. PREDICTED vs REALIZED: the optimizer's analytical coherent_peak (a ranking
     proxy, optimistic) next to the ADS-realized boost / absolute out/in. The
     proxy varactor is idealized, so the realized number is a faithfulness RECORD,
     never a real-world device-gain claim.
  4. FLAT==HIER parity @ the operating pump (the emitter is faithful).

The proxy varactor's idealization is why these run net-modest, not net-huge: the
prediction-vs-record discipline is the point.

Run under the ADS Python interpreter (PYTHONPATH=src):

    & $env:ADS_PYTHON analyze_proxy.py all                  # all three
    & $env:ADS_PYTHON analyze_proxy.py proxy_n3_single_short # one

Writes results/proxy_analysis_<key>.json + results/proxy_analysis_summary.json.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxy_designs import (  # noqa: E402
    EXAMPLES_PROXY,
    PROXY_PICKS,
    build_proxy_spec,
)

_HERE = Path(__file__).resolve().parent
OUTPUTS = _HERE / "outputs" / "proxy_analysis"   # git-ignored HB scratch
RESULTS = _HERE / "results"                      # committed analysis JSON

PUMP_SCAN = (-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0)
PHASES = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
_TIMEOUT_S = 600


def _funds(spec):
    """(f_signal_Hz, f_pump_Hz) read from the UNRESOLVED spec's VAR block."""
    fs = float(next(v.value for v in spec.var_block.variables if v.name == "f_signal"))
    fp = float(next(v.value for v in spec.var_block.variables if v.name == "f_pump"))
    return fs, fp


def _run_flat(spec, rundir, tag):
    """Emit + HB a flat deck; return an hb_dataset or None (None = HB did not converge)."""
    from tvtl_ads_builder_v2.analysis.hb_dataset import read_hb_dataset
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.simulate.runner import run_hpeesofsim

    ws = rundir / tag
    ws.mkdir(parents=True, exist_ok=True)
    deck = ws / f"{tag}.ckt"
    deck.write_text(emit_flat_hb_netlist(spec), encoding="utf-8")
    res = run_hpeesofsim(deck, timeout_s=_TIMEOUT_S)
    if res["status"] != "completed":
        return None
    return read_hb_dataset(res["dataset_path"], _funds(spec))


def _phase0_metrics(ds_on, v_sig_off, p_avs_W):
    """(boost_dB, abs_gain_dB, mix10_dB, vout_sig_on) for a converged on-dataset."""
    from tvtl_ads_builder_v2.oa.psa_gain_artifact import (
        degenerate_folding_mixes,
        full_family_phasor_sum,
        full_sum_gain_dB,
    )

    out_on = ds_on.node_voltage("n_output")
    fam = degenerate_folding_mixes(out_on)
    full_on = full_family_phasor_sum(out_on, fam)
    v10_on = complex(out_on.at_mix((1, 0)))
    pout_mod = 0.5 * abs(full_on) ** 2 / 50.0
    return {
        "boost_dB": full_sum_gain_dB(full_on, v_sig_off),
        "abs_gain_dB": 10.0 * math.log10(pout_mod / p_avs_W) if pout_mod > 0 and p_avs_W > 0 else float("-inf"),
        "mix10_dB": 20.0 * math.log10(abs(v10_on) / abs(v_sig_off)) if abs(v_sig_off) > 0 else float("nan"),
        "vout_sig_on_V": abs(v10_on),
        "n_family": len(fam),
    }


def _pump_scan(key, rundir):
    """Scan pump power @ phase 0; return (scan rows, v_sig_off, p_avs_W)."""
    from tvtl_ads_builder_v2.device_spec import with_var_overrides

    # OFF dataset is pump-independent -> run once (pump driven to -100 dBm).
    spec0 = build_proxy_spec(key, pump_dBm=PUMP_SCAN[0])
    sig_dBm = float(next(v.value for v in spec0.var_block.variables if v.name == "signal_power_dBm"))
    p_avs_W = 10.0 ** ((sig_dBm - 30.0) / 10.0)
    ds_off = _run_flat(with_var_overrides(spec0, {"pump_power_dBm": -100.0}), rundir, "off")
    if ds_off is None:
        return [], None, p_avs_W
    v_sig_off = complex(ds_off.node_voltage("n_output").at_mix((1, 0)))

    rows = []
    for pump in PUMP_SCAN:
        spec = build_proxy_spec(key, pump_dBm=pump)
        ds_on = _run_flat(with_var_overrides(spec, {"pump_phase_rad": 0.0}), rundir, f"on_p{int(round(pump))}")
        if ds_on is None:
            rows.append({"pump_dBm": pump, "converged": False})
            continue
        m = _phase0_metrics(ds_on, v_sig_off, p_avs_W)
        rows.append({"pump_dBm": pump, "converged": True, **m})
    return rows, v_sig_off, p_avs_W


def _phase_sweep(key, pump_dBm, rundir):
    from tvtl_ads_builder_v2.oa.psa_gain_artifact import run_full_sum_gain_sweep

    spec = build_proxy_spec(key, pump_dBm=pump_dBm)
    return run_full_sum_gain_sweep(spec, rundir / "phase_sweep", phases_deg=PHASES, timeout_s=_TIMEOUT_S)


def _parity(key, pump_dBm, rundir):
    from build_examples import _rename_boundary_nets, _run_and_read
    from tvtl_ads_builder_v2.analysis.hb_dataset import SIGNAL_MIX_2TONE, hb_dataset_from_raw
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.oa.oa_hier_emitter import emit_hierarchical_workspace_from_spec

    spec = build_proxy_spec(key, pump_dBm=pump_dBm)
    funds = _funds(spec)
    ws = rundir / "parity_ws"
    if ws.parent.exists():
        shutil.rmtree(ws / "..", ignore_errors=True)
    res = emit_hierarchical_workspace_from_spec(
        spec, ws, library_name=f"{key}_par", top_cell_name=f"{key}_par_top", uc_cell_name="unit_cell")
    hier_nl = Path(res["ads_netlist_path"]).read_text(encoding="utf-8")
    sim = rundir / "parity_sim"
    if sim.exists():
        shutil.rmtree(sim, ignore_errors=True)
    flat_info, flat_raw = _run_and_read(emit_flat_hb_netlist(spec), sim / "flat", "flat_run")
    hier_info, hier_raw = _run_and_read(_rename_boundary_nets(hier_nl), sim / "hier", "hier_run")
    out = {"flat_run": flat_info["status"], "hier_run": hier_info["status"]}
    if flat_raw is None or hier_raw is None:
        out["verdict"] = "RUN_FAILED"
        return out
    fv = hb_dataset_from_raw(flat_raw, funds).node_voltage("n_output")
    hv = hb_dataset_from_raw(hier_raw, funds).node_voltage("n_output")
    fmix = {h.mix: h.amplitude for h in fv.harmonics}
    hmix = {h.mix: h.amplitude for h in hv.harmonics}
    common = sorted(set(fmix) & set(hmix))
    max_d = max((abs(fmix[m] - hmix[m]) for m in common), default=float("nan"))
    vsf = abs(fv.at_mix(SIGNAL_MIX_2TONE))
    vsh = abs(hv.at_mix(SIGNAL_MIX_2TONE))
    out.update({
        "Vout_signal_flat": f"{vsf:.6e}", "Vout_signal_hier": f"{vsh:.6e}",
        "max_abs_dV": max_d, "bit_identical": f"{vsf:.6e}" == f"{vsh:.6e}",
        "verdict": "IDENTICAL" if max_d < 1e-9 else ("CLOSE" if max_d < 1e-6 else "DIFFER"),
    })
    return out


def analyze_one(key, headline):
    rundir = OUTPUTS / key
    if rundir.exists():
        shutil.rmtree(rundir, ignore_errors=True)
    rundir.mkdir(parents=True, exist_ok=True)
    pick = PROXY_PICKS[key]
    print(f"\n=== {key}  ({pick.topology.key}) ===", flush=True)
    print(f"  PREDICTION (analytical coherent_peak, optimistic ranking proxy): "
          f"{pick.analytical_coherent_peak_dB:+.2f} dB", flush=True)

    print("  -- pump-power scan @ phase 0 (convergent window + boost) --", flush=True)
    scan, v_sig_off, p_avs_W = _pump_scan(key, rundir)
    for r in scan:
        if r["converged"]:
            print(f"     pump={r['pump_dBm']:+6.1f} dBm  CONVERGED  boost@0={r['boost_dB']:+.2f} dB  "
                  f"abs@0={r['abs_gain_dB']:+.2f} dB  mix10@0={r['mix10_dB']:+.2f} dB", flush=True)
        else:
            print(f"     pump={r['pump_dBm']:+6.1f} dBm  NON-CONVERGENT  <- oscillation-edge onset", flush=True)

    converged = [r for r in scan if r["converged"]]
    nonconv = [r for r in scan if not r["converged"]]
    onset = min((r["pump_dBm"] for r in nonconv), default=None)
    # operating point: convergent pump that maximises the phase-0 pump-on/off boost.
    op = max(converged, key=lambda r: r["boost_dB"]) if converged else None
    result = {
        "key": key, "topology": pick.topology.key, "headline": headline,
        "f_signal_GHz": pick.f_s_Hz / 1e9, "f_pump_GHz": 2.0 * pick.f_s_Hz / 1e9,
        "n_cells": pick.n_cells, "P_avs_W": p_avs_W,
        "Vout_passive_pumpoff_V": abs(v_sig_off) if v_sig_off is not None else None,
        "prediction": {
            "analytical_coherent_peak_dB": pick.analytical_coherent_peak_dB,
            "note": "optimizer analytical ranking proxy (Floquet, optimistic; omits loss/mismatch/large-signal)",
        },
        "pump_scan": scan,
        "nonconvergence_onset_dBm": onset,
        "operating_pump_dBm": (op["pump_dBm"] if op else None),
    }

    sw = None
    if op is not None:
        op_pump = op["pump_dBm"]
        print(f"  -- phase sweep @ pump={op_pump:+.1f} dBm (bounded lobe?) --", flush=True)
        try:
            sw = _phase_sweep(key, op_pump, rundir)
        except Exception as exc:  # noqa: BLE001
            print(f"     phase sweep FAILED at pump={op_pump}: {type(exc).__name__}: {exc}", flush=True)
            result["phase_sweep_error"] = f"{type(exc).__name__}: {exc}"

    if sw is not None:
        rows = [(r["phase_deg"], r["coh_full_dB"], r["coh_2term_dB"], r["mix10_dB"]) for r in sw.rows]
        for ph, full, two, m10 in rows:
            print(f"     phase={ph:5.0f}  boost(full)={full:+7.2f} dB  coh2={two:+7.2f}  mix10={m10:+7.2f}", flush=True)
        peak = max(r[1] for r in rows)
        trough = min(r[1] for r in rows)
        bounded = bool(math.isfinite(peak) and peak < 40.0 and trough < 0.0)
        finite_onset = onset is not None
        print(f"     >>> peak(boost)={peak:+.2f} dB @ {sw.peak_phase_deg:.0f} deg ; "
              f"trough={trough:+.2f} dB ; swing={peak - trough:.2f} dB", flush=True)
        print(f"     >>> abs out/in (full-sum peak phase) = {sw.g_mod_in_dB:+.2f} dB", flush=True)
        print(f"     >>> BOUNDED PSA LOBE: {bounded}   (non-convergence onset @ {onset} dBm)", flush=True)

        result["realized"] = {
            "operating_pump_dBm": op_pump,
            "boost_peak_dB": peak, "boost_peak_phase_deg": sw.peak_phase_deg,
            "boost_trough_dB": trough, "boost_swing_dB": peak - trough,
            "abs_out_in_dB": sw.g_mod_in_dB,
            "gamma_in_sq": sw.gamma_in_sq,
            "note": "ADS harmonic-balance realized record (idealized proxy varactor; not a real-world device gain)",
        }
        result["stability"] = {
            "bounded_lobe": bounded,
            "peak_dB": peak, "trough_dB": trough, "swing_dB": peak - trough,
            "nonconvergence_onset_dBm": onset,
            "verdict": ("STABLE: bounded phase-sensitive PSA lobe, not an oscillation-edge runaway"
                        if bounded and finite_onset else
                        ("STABLE: bounded lobe (no non-convergence within scan)" if bounded else "REVIEW")),
        }
        result["phase_sweep"] = sw.to_dict()

    if op is not None:
        print(f"  -- flat==hier parity @ pump={op['pump_dBm']:+.1f} dBm --", flush=True)
        par = _parity(key, op["pump_dBm"], rundir)
        print(f"     verdict={par.get('verdict')}  flat={par.get('Vout_signal_flat')}  "
              f"hier={par.get('Vout_signal_hier')}  bit_identical={par.get('bit_identical')}", flush=True)
        result["parity"] = par
        result["reference_vout_signal"] = par.get("Vout_signal_flat")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"proxy_analysis_{key}.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {RESULTS / f'proxy_analysis_{key}.json'}", flush=True)
    return result


def main(argv):
    keys = [k for k, *_ in EXAMPLES_PROXY] if (not argv or argv[0] == "all") else [argv[0]]
    by_key = {k: hl for (k, _b, _l, _t, hl) in EXAMPLES_PROXY}
    summary = {}
    for k in keys:
        r = analyze_one(k, by_key[k])
        summary[k] = {
            "topology": r["topology"],
            "operating_pump_dBm": r.get("operating_pump_dBm"),
            "nonconvergence_onset_dBm": r.get("nonconvergence_onset_dBm"),
            "prediction_dB": r["prediction"]["analytical_coherent_peak_dB"],
            "realized_boost_peak_dB": r.get("realized", {}).get("boost_peak_dB"),
            "realized_abs_out_in_dB": r.get("realized", {}).get("abs_out_in_dB"),
            "bounded_lobe": r.get("stability", {}).get("bounded_lobe"),
            "reference_vout_signal": r.get("reference_vout_signal"),
            "parity_verdict": r.get("parity", {}).get("verdict"),
        }
    (RESULTS / "proxy_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== PROXY ANALYSIS SUMMARY ===", flush=True)
    for k, s in summary.items():
        print(f"  {k:22s} op={s['operating_pump_dBm']} dBm  onset={s['nonconvergence_onset_dBm']}  "
              f"pred={s['prediction_dB']:+.2f}  realized_boost={s['realized_boost_peak_dB']}  "
              f"abs={s['realized_abs_out_in_dB']}  bounded={s['bounded_lobe']}  "
              f"parity={s['parity_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
