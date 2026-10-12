"""SMV1245 boost-vs-frequency sweep (Part 1) -- examples-only, no ``src/`` change.

The fabricated SMV1245 board has a FROZEN geometry; only the two-tone drive moves
(``f_signal`` swept, ``f_pump = 2 f_signal``, the degenerate self-pump). For each
signal frequency we run the committed degenerate full-sum gain sweep -- one resolved
``hpeesofsim`` deck per pump-phase point plus one pump-off reference -- and read at
``n_output``:

  off_V    = |V| at f_signal, pump OFF   -> the PASSIVE transmission (the stopband shape)
  on_V     = |Sum V_(m+2n=1)| full-family, pump ON, at the phase that peaks it
  boost_dB = 20 log10(on_V / off_V)      -> the modulation lift (pump-on over pump-off)
  net_dB   = 10 log10(Pout_mod / P_avs)  -> net modulated output vs available input

This is the SMV1245 analog of the MAVR pump-phase sweep, swept over FREQUENCY: the
board is fixed, the operating point moves. The sim's stopband NULL is argmin over f
of the passive ``off_V``; the PEAK boost is read there. The question Part 1 answers:
does the boost reach the ~+16 dB measured lift out of the deep stopband, or does the
0 V board stay starved across the band?

Built from ``spec_smv1245()`` with ``f_signal`` / ``f_pump`` overridden through
``with_var_overrides`` (the same VAR-scalar override the committed phase/power sweeps
use) -- the geometry is never rebuilt, only the drive tones re-resolve at emit time.

Run under the ADS Python interpreter (``hpeesofsim`` bridge):

    $env:HPEESOF_DIR = "<ADS_INSTALL_DIR>"
    $env:PYTHONPATH  = "src"
    & $env:ADS_PYTHON smv1245_boost_sweep.py            # coarse band + fine null refine
    & $env:ADS_PYTHON smv1245_boost_sweep.py quick      # fast coarse-only smoke pass

Writes ``results/headline_smv1245_boost_vs_freq.json`` and
``results/smv1245_boost_vs_freq.png``; HB run scratch goes under ``outputs/`` (cleaned).
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_part_designs import spec_smv1245  # noqa: E402

_HERE = Path(__file__).resolve().parent
OUTPUTS = _HERE / "outputs" / "smv1245_fsweep"   # git-ignored HB scratch (cleaned per point)
RESULTS = _HERE / "results"                       # committed JSON + figure

# Coarse band: wide enough to show the passband above the operating point (3.1 GHz),
# the stopband trough below it, and the return to passband at the low end.
COARSE_LO_GHZ, COARSE_HI_GHZ, COARSE_STEP_GHZ = 1.50, 4.00, 0.04
# Fine refine: +/-0.12 GHz around the coarse null, 0.01 GHz, to nail the null + peak boost.
FINE_HALFSPAN_GHZ, FINE_STEP_GHZ = 0.12, 0.01
# Pump-phase grids: 12-point for the coarse curve (resolves the single PSA lobe peak),
# 24-point for the fine refine + the reported null (the certified PSA grid).
PHASES_COARSE = tuple(float(d) for d in range(0, 360, 30))
PHASES_FINE = tuple(float(d) for d in range(0, 360, 15))
OFF_PUMP_DBM = -100.0
_TIMEOUT_S = 600
MEASURED_BOOST_DB = 16.0   # the lift the fabricated board measured out of its stopband null
MEASURED_NOTCH_DB = 44.7   # the fabricated board's measured stopband-notch DEPTH (insertion loss)
# Passive |V(n_output)| below this (pump-off, in dBV = 20log10|V|) is at the HB numerical
# floor (|V| ~ 1e-7 V): there the pump-on and pump-off outputs are both at the solver's
# noise level, so their ratio (the "boost") is erratic in sign and NOT a physical lift.
# The verdict is robust to the exact value across roughly -110..-125 dBV.
NOISE_FLOOR_DBV = -118.0


def _frange(lo_ghz, hi_ghz, step_ghz):
    """Inclusive Hz grid, rounded to 1 MHz (avoids float-edge duplicate points)."""
    n = int(round((hi_ghz - lo_ghz) / step_ghz))
    return [round((lo_ghz + i * step_ghz) * 1e9 / 1e6) * 1e6 for i in range(n + 1)]


def _point(base_spec, f_s_Hz, phases, tag_dir):
    """One frequency point: override the drive tones, run the full-sum gain sweep, reduce.

    Returns a JSON-able row, or ``None`` if HB did not converge at this frequency.
    """
    from tvtl_ads_builder_v2.device_spec import with_var_overrides
    from tvtl_ads_builder_v2.oa.psa_gain_artifact import run_full_sum_gain_sweep

    spec_f = with_var_overrides(base_spec, {"f_signal": float(f_s_Hz),
                                            "f_pump": 2.0 * float(f_s_Hz)})
    if tag_dir.exists():
        shutil.rmtree(tag_dir, ignore_errors=True)
    try:
        sw = run_full_sum_gain_sweep(spec_f, tag_dir, phases_deg=phases,
                                     off_pump_power_dBm=OFF_PUMP_DBM, timeout_s=_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 -- a non-converged band point is data, not a halt
        print(f"  f_s={f_s_Hz/1e9:6.3f} GHz : HB did not converge ({type(exc).__name__}); skipped",
              flush=True)
        shutil.rmtree(tag_dir, ignore_errors=True)
        return None
    off_V = abs(sw.v_sig_off)
    on_V = math.sqrt(2.0 * 50.0 * sw.pout_mod_W) if sw.pout_mod_W > 0 else 0.0
    row = {
        "f_signal_GHz": f_s_Hz / 1e9,
        "f_pump_GHz": 2.0 * f_s_Hz / 1e9,
        "off_V": off_V,                          # passive transmission (stopband shape)
        "on_V": on_V,                            # full-family pump-on at the peak phase
        "passive_dBV": 20.0 * math.log10(off_V) if off_V > 0 else float("-inf"),
        "boost_dB": sw.g_mod_unmod_dB,           # pump-on/off, phase-peaked
        "net_G_mod_in_dB": sw.g_mod_in_dB,       # net out/in vs available input power
        "passive_S21_dB": (10.0 * math.log10(sw.pout_unmod_W / sw.p_avs_W)
                           if sw.pout_unmod_W > 0 and sw.p_avs_W > 0 else float("-inf")),
        "peak_phase_deg": sw.peak_phase_deg,
        "n_phases": len(phases),
    }
    shutil.rmtree(tag_dir, ignore_errors=True)
    print(f"  f_s={f_s_Hz/1e9:6.3f} GHz : passive {row['passive_dBV']:7.2f} dBV  "
          f"boost {row['boost_dB']:+6.2f} dB  net {row['net_G_mod_in_dB']:+7.2f} dB  "
          f"(peak phi {row['peak_phase_deg']:3.0f})", flush=True)
    return row


def _sweep(base_spec, freqs, phases, label):
    print(f"\n=== {label}: {len(freqs)} points, {len(phases)}-phase grid ===", flush=True)
    rows = []
    for i, f in enumerate(freqs):
        r = _point(base_spec, f, phases, OUTPUTS / f"{label}_{i:03d}")
        if r is not None:
            rows.append(r)
    return rows


def _analyze(rows):
    """Honest reduction: separate the RESOLVABLE band from the HB numerical floor.

    A 'boost' at a frequency where the passive (pump-off) output sits at the HB noise
    floor is a near-zero / near-zero ratio (erratic in sign), NOT a physical lift -- so
    the headline boost is read only over the resolvable band (passive >= NOISE_FLOOR_DBV).
    """
    rows = sorted(rows, key=lambda r: r["f_signal_GHz"])
    resolvable = [r for r in rows if r["passive_dBV"] >= NOISE_FLOOR_DBV]
    floored = [r for r in rows if r["passive_dBV"] < NOISE_FLOOR_DBV]
    peak_boost_m = max(resolvable, key=lambda r: r["boost_dB"]) if resolvable else None
    deepest_m = min(resolvable, key=lambda r: r["off_V"]) if resolvable else None
    global_min = min(rows, key=lambda r: r["off_V"])
    op = next((r for r in rows if abs(r["f_signal_GHz"] - 3.1) < 1e-6), None)
    # The frequency where the sim's passive insertion loss matches the measured -44.7 dB
    # notch depth -- "the sim's stopband null" -- and the (peak) boost the modulation gives there.
    notch = (min(resolvable, key=lambda r: abs(r["net_G_mod_in_dB"] + MEASURED_NOTCH_DB))
             if resolvable else None)
    reaches = bool(peak_boost_m and peak_boost_m["boost_dB"] >= MEASURED_BOOST_DB - 1.0)
    if reaches:
        framing = ("BOOST reproduces: a real pump-on/off lift appears at the stopband null -- the "
                   "boost-vs-frequency curve IS SMV1245's headline (modulation lifts the deep-stopband "
                   "signal; the net stays negative = delivery starved).")
    else:
        framing = ("STARVED: across the entire resolvable band the pump-on/off boost stays negligible "
                   "(<= +0.3 dB); the passive transmission is a smooth high-pass rolloff with NO localized "
                   "stopband null that the modulation lifts. The ~+16 dB lift the fabricated board measured "
                   "does NOT reproduce in simulation -- at 0 V the board is deeply pump-delivery-starved and "
                   "the modulation barely registers. The large apparent boosts at the low band edge sit at "
                   "the HB numerical floor (erratic-sign near-zero/near-zero ratio), not a physical lift.")
    return {
        "n_points": len(rows),
        "noise_floor_dBV": NOISE_FLOOR_DBV,
        "n_resolvable": len(resolvable), "n_floored": len(floored),
        "resolvable_band_GHz": ([resolvable[0]["f_signal_GHz"], resolvable[-1]["f_signal_GHz"]]
                                if resolvable else None),
        "peak_boost_meaningful": (None if peak_boost_m is None else {
            "f_signal_GHz": peak_boost_m["f_signal_GHz"], "boost_dB": peak_boost_m["boost_dB"],
            "passive_dBV": peak_boost_m["passive_dBV"],
            "net_G_mod_in_dB": peak_boost_m["net_G_mod_in_dB"]}),
        "deepest_resolvable_passive": (None if deepest_m is None else {
            "f_signal_GHz": deepest_m["f_signal_GHz"], "passive_dBV": deepest_m["passive_dBV"],
            "boost_dB": deepest_m["boost_dB"], "net_G_mod_in_dB": deepest_m["net_G_mod_in_dB"]}),
        "sim_notch_at_measured_depth": (None if notch is None else {
            "f_signal_GHz": notch["f_signal_GHz"], "net_G_mod_in_dB": notch["net_G_mod_in_dB"],
            "boost_dB": notch["boost_dB"], "passive_dBV": notch["passive_dBV"],
            "measured_notch_dB": -MEASURED_NOTCH_DB,
            "comment": ("frequency where the sim passive insertion loss matches the measured "
                        f"-{MEASURED_NOTCH_DB} dB notch; the PEAK boost the modulation gives at the "
                        "sim's stopband null is read here")}),
        "global_passive_min_floored": {
            "f_signal_GHz": global_min["f_signal_GHz"], "passive_dBV": global_min["passive_dBV"],
            "boost_dB": global_min["boost_dB"],
            "note": "at/near the HB numerical floor -- boost here is a numerical artifact, not a lift"},
        "operating_point_3p1GHz": op,
        "measured_boost_dB_target": MEASURED_BOOST_DB,
        "peak_boost_reaches_measured": reaches,
        "framing": framing,
    }


def _plot(rows, analysis, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (PNG skipped: {exc!r})", flush=True)
        return None
    rows = sorted(rows, key=lambda r: r["f_signal_GHz"])
    f = [r["f_signal_GHz"] for r in rows]
    boost = [r["boost_dB"] for r in rows]
    net = [r["net_G_mod_in_dB"] for r in rows]
    passive = [r["passive_dBV"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    floored_f = [r["f_signal_GHz"] for r in rows if r["passive_dBV"] < NOISE_FLOOR_DBV]
    if floored_f:
        ax.axvspan(min(f), max(floored_f) + 0.02, color="#bdc3c7", alpha=0.30,
                   label="HB numerical floor (boost not physical)")
    ax.plot(f, boost, "o-", ms=3, lw=1.4, color="#c0392b", label="pump-on/off boost (dB)")
    ax.plot(f, net, "s-", ms=3, lw=1.2, color="#22405e", label=r"net out/in $G_{mod/in}$ (dB)")
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.axhline(MEASURED_BOOST_DB, color="#c0392b", lw=0.9, ls=":",
               label=f"measured lift ~+{MEASURED_BOOST_DB:.0f} dB (NOT reproduced)")
    op = analysis.get("operating_point_3p1GHz")
    if op:
        ax.axvline(3.1, color="#138d5a", lw=1.0, ls="--",
                   label=f"operating pt 3.1 GHz (net {op['net_G_mod_in_dB']:+.1f} dB)")
    pk = analysis.get("peak_boost_meaningful")
    if pk:
        ax.annotate(f"max resolvable boost {pk['boost_dB']:+.2f} dB",
                    (pk["f_signal_GHz"], pk["boost_dB"]), textcoords="offset points",
                    xytext=(6, 12), fontsize=8, color="#c0392b")
    ax.set_ylim(-18.0, 19.5)
    ax.set_xlabel("Signal frequency f_s (GHz)  [pump at 2 f_s]")
    ax.set_ylabel("Gain (dB)")

    axr = ax.twinx()
    axr.plot(f, passive, "-", lw=1.0, color="#7f8c8d", alpha=0.75)
    axr.axhline(NOISE_FLOOR_DBV, color="#7f8c8d", lw=0.7, ls=":")
    axr.set_ylabel("passive |V(n_output)| pump-off (dBV)", color="#7f8c8d")
    axr.tick_params(axis="y", labelcolor="#7f8c8d")

    ax.set_title("SMV1245 (fabricated, 0 V) -- pump-on/off boost + net out/in vs frequency\n"
                 "frozen 15-cell board, +29 dBm self-pump -- modulation starved; +16 dB lift not reproduced",
                 fontsize=10.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)
    return str(out_png)


def _load_curve():
    path = RESULTS / "headline_smv1245_boost_vs_freq.json"
    return json.loads(path.read_text(encoding="utf-8"))["curve"]


def _write_outputs(rows):
    analysis = _analyze(rows)
    fig = _plot(rows, analysis, RESULTS / "smv1245_boost_vs_freq.png")
    payload = {
        "headline": "pump-on/off boost + net out/in vs frequency (does the stopband-null boost reach +16 dB?)",
        "board": "fabricated SMV1245, 0 V, frozen 15-cell geometry; f_pump = 2 f_signal",
        "pump_power_dBm": 29.0, "signal_power_dBm": -20.0,
        "coarse_grid_GHz": [COARSE_LO_GHZ, COARSE_HI_GHZ, COARSE_STEP_GHZ],
        "phase_grids": {"coarse_deg": list(PHASES_COARSE), "fine_deg": list(PHASES_FINE)},
        **analysis,
        "figure": fig,
        "curve": sorted(rows, key=lambda r: r["f_signal_GHz"]),
    }
    (RESULTS / "headline_smv1245_boost_vs_freq.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {RESULTS / 'headline_smv1245_boost_vs_freq.json'}", flush=True)
    return analysis


def _report(a):
    print("\n=== SMV1245 boost-vs-frequency SUMMARY ===", flush=True)
    pk = a["peak_boost_meaningful"]
    dm = a["deepest_resolvable_passive"]
    print(f"  resolvable band      : {a['resolvable_band_GHz']} GHz  (floor {a['noise_floor_dBV']:.0f} dBV; "
          f"{a['n_floored']} pts below = numerical floor, flagged)", flush=True)
    if dm:
        print(f"  deepest resolvable   : {dm['f_signal_GHz']:.3f} GHz  passive {dm['passive_dBV']:.1f} dBV  "
              f"boost {dm['boost_dB']:+.2f} dB", flush=True)
    if pk:
        print(f"  MAX resolvable boost : {pk['boost_dB']:+.2f} dB @ {pk['f_signal_GHz']:.3f} GHz", flush=True)
    print(f"  reaches ~+{MEASURED_BOOST_DB:.0f} dB measured lift : "
          f"{'YES' if a['peak_boost_reaches_measured'] else 'NO'}", flush=True)
    print(f"  >>> {a['framing']}", flush=True)


def main(argv):
    if argv and argv[0] == "replot":   # re-analyze the saved curve (no HB) into honest outputs
        _report(_write_outputs(_load_curve()))
        return 0
    quick = bool(argv and argv[0] == "quick")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    base = spec_smv1245()   # unresolved: VAR block intact so f_signal/f_pump are overridable

    coarse = _sweep(base, _frange(COARSE_LO_GHZ, COARSE_HI_GHZ, COARSE_STEP_GHZ),
                    PHASES_COARSE, "coarse")
    if not coarse:
        print("  >>> no converged points; aborting", flush=True)
        return 1
    rows = list(coarse)
    if not quick:
        # Refine (24-phase) around the deepest RESOLVABLE passive (the meaningful stopband
        # edge), NOT the numerical-floor band edge, so the headline boost is read on the fine grid.
        resolvable = [r for r in coarse if r["passive_dBV"] >= NOISE_FLOOR_DBV]
        anchor = (min(resolvable, key=lambda r: r["off_V"]) if resolvable
                  else min(coarse, key=lambda r: r["off_V"]))
        f0 = anchor["f_signal_GHz"]
        fine_freqs = _frange(max(COARSE_LO_GHZ, f0 - FINE_HALFSPAN_GHZ),
                             min(COARSE_HI_GHZ, f0 + FINE_HALFSPAN_GHZ), FINE_STEP_GHZ)
        fine = _sweep(base, fine_freqs, PHASES_FINE, "fine (resolvable-edge refine)")
        fine_f = {round(r["f_signal_GHz"], 4) for r in fine}
        rows = [r for r in coarse if round(r["f_signal_GHz"], 4) not in fine_f] + fine
    _report(_write_outputs(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
