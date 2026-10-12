"""R9 replay -- regenerates ``results/smv1245_lobe2_prediction.json``.

Second-lobe (4.8-5.7 GHz) boost prediction for the fixed-C SMV1245 board: the
pump 2*f_s lands near the R7 branch-delivery peak (~10.6 GHz), so this checks
whether the model predicts a usable boost lobe there.

Conditions
----------
* Second-lobe grid f_s = 4.80..5.70 GHz (0.05 step); prescribed-depth leg and
  junction readout centered at f_s = 5.3 GHz (pump 10.6 GHz).
* Stub table re-tabulated to 0.05-80 GHz (7996 pts), relocated to
  ``results/stub_tables/smv1245_stub_80GHz.s1p`` -- 80 GHz so the truncation_N=6
  z_stub reach (13*f = 74 GHz) stays in-table. Exact ADS recipe in the .s1p
  header.
* HB legs at pump 29 and 20 dBm; junction m_j via route B (V(n_rp)-V(n_v) at the
  (0,1) mix).

Replayability: the stub load + prescribed-depth leg + analytical S21 are PURE
analytical (no ADS). The HB boost legs and junction readout require the ADS
python (``run_full_sum_gain_sweep`` + HB dataset reads); the committed JSON holds
those recorded values. This script only writes when run as ``__main__`` under an
ADS-capable environment.
"""
import sys, math, json, glob
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples" / "real_parts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_part_designs import spec_smv1245  # noqa: E402
from tvtl.varactor_palette import get_entry  # noqa: E402
from tvtl_ads_builder_v2.analytical.realistic_c_gain import level10_realistic_param_gain  # noqa: E402
from tvtl_ads_builder_v2.device_spec import with_var_overrides  # noqa: E402
from tvtl_ads_builder_v2.oa.psa_gain_artifact import run_full_sum_gain_sweep  # noqa: E402
from tvtl_ads_builder_v2.analysis.hb_dataset import read_hb_dataset  # noqa: E402
from tvtl_ads_builder_v2.analysis.floquet_comparison import (  # noqa: E402
    _spec_varactor_q_poly, _capacitance_coefficients_from_q_poly)
from _stub_table import STUB_80GHz, load_s1p, make_y_override  # noqa: E402

# Exact ADS S_Param recipe that produced smv1245_stub_80GHz.s1p (provenance only):
ADS_STUB_DECK = (
    'model msub1 MSUB H=1.54 mm Er=3.38 Mur=1 Cond=58000000 T=0.0175 mm TanD=0.0027 Rough=0 mm\n'
    'Port:P1 n_s0 0 Num=1 Z=50 Ohm Noise=no\n'
    'MLIN:STUB_0 n_s0 n_s1 Subst="msub1" W=1.93 mm L=6.83 mm\n'
    'VIAGND:VG1 n_s1 Subst="msub1" D=1.6 mm W=1.93 mm T=0.0175 mm Rho=0.706613906162\n'
    'MLIN:STUB_1 n_s1 n_s2 Subst="msub1" W=1.41 mm L=5.25 mm\n'
    'VIAGND:VG2 n_s2 Subst="msub1" D=1.6 mm W=1.41 mm T=0.0175 mm Rho=0.706613906162\n'
    'MLIN:STUB_2 n_s2 n_open Subst="msub1" W=1.98 mm L=1.04 mm\n'
    'MLEF:MLEFEND n_open Subst="msub1" W=1.98 mm L=0 mm\n'
    'S_Param:SP1 CalcS=yes CalcY=no CalcZ=no SweepVar="freq" SweepPlan="SP1_stim"\n'
    'SweepPlan:SP1_stim Start=0.05 GHz Stop=80 GHz Lin=7996\n')

OUT = REPO / "examples" / "real_parts" / "results" / "smv1245_lobe2_prediction.json"
OUTPUTS = REPO / "examples" / "real_parts" / "outputs" / "lobe2_hb"   # gitignored scratch
DEV = get_entry("SMV1245-079LF").to_realistic_varactor(v_bias_v=0.0)
C0F, C1F = 7.37e-12, -3.3514285714285714e-12
PH12 = tuple(float(d) for d in range(0, 360, 30))
PH24 = tuple(float(d) for d in range(0, 360, 15))
FGRID = [round(4.80 + 0.05 * i, 2) for i in range(int((5.70 - 4.80) / 0.05) + 1)]

# stub admittance from the relocated 80 GHz .s1p (no ADS)
F_TAB, Y_TAB = load_s1p(STUB_80GHz)
FLO, FHI = F_TAB[0], F_TAB[-1]
y_override = make_y_override(F_TAB, Y_TAB)


def gpre(m, fs, fm):
    return level10_realistic_param_gain(with_var_overrides(spec_smv1245(), {"f_signal": fs, "f_pump": fm}),
                                        DEV, m_branch_pump=float(m), truncation_N=6, stub_y_fn_override=y_override)


def main():
    payload = {"headline": "SMV1245 fixed-C board: second-lobe (4.8-5.7 GHz) boost prediction; "
                           "pump 2*f_s lands near the R7 delivery peak (~10.6 GHz)"}

    def save():
        OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    payload["stub_table"] = {"range_GHz": [FLO / 1e9, FHI / 1e9], "npts": len(F_TAB),
                             "note": "re-tabulated to 80 GHz so truncation_N=6 z_stub reach (13*f=74 GHz) stays in-table; "
                                     "relocated to results/stub_tables/smv1245_stub_80GHz.s1p"}

    # ---------- prescribed-depth leg at f_s=5.3 (corrected stub, N=6) ----------
    print("prescribed leg @5.3...", flush=True)
    prescribed = []
    for m in [0.01, 0.03, 0.1, 0.2, 0.3, 0.5]:
        r = gpre(m, 5.3e9, 10.6e9)
        prescribed.append({"m_branch": m, "depth_eff_junction": r["depth_effective"],
                           "coherent_peak_dB": r["coherent_peak_dB"], "coherent_min_dB": r["coherent_min_dB"],
                           "net_out_avail_dB": r["coherent_peak_dB"] + r["abs_gain_off_dB"],
                           "pumpoff_ref_dB": r["abs_gain_off_dB"]})
    payload["prescribed_leg_5p3GHz"] = prescribed
    # analytical linear S21_off 4.8-5.7
    ana = [{"f_GHz": f, "S21_off_dB": gpre(1e-9, f * 1e9, 2 * f * 1e9)["abs_gain_off_dB"]} for f in FGRID]
    payload["analytical_S21_off_4p8_5p7"] = ana
    save()

    # ---------- HB leg (requires ADS python) ----------
    def hb_point(pump, f):
        outd = OUTPUTS / f"p{pump}" / f"f{f:.2f}"
        spec = with_var_overrides(spec_smv1245(pump_dBm=pump), {"f_signal": f * 1e9, "f_pump": 2 * f * 1e9})

        def run(phases):
            return run_full_sum_gain_sweep(spec, outd / f"n{len(phases)}", phases_deg=phases,
                                           off_pump_power_dBm=-100.0, timeout_s=1200)
        sw = run(PH12); nph = 12
        if sw.peak_coh_full_dB > 1.0:
            sw = run(PH24); nph = 24
        return {"f_GHz": f, "pump_dBm": pump, "peak_boost_dB": sw.peak_coh_full_dB,
                "net_G_mod_in_dB": sw.g_mod_in_dB, "off_V": abs(sw.v_sig_off),
                "peak_phase_deg": sw.peak_phase_deg, "n_phases": nph, "converged": True}, (spec, outd)

    payload["hb_leg"] = {"29": [], "20": []}
    spec53 = {}
    for pump in [29.0, 20.0]:
        for f in FGRID:
            print(f"HB pump={pump} f={f} ...", flush=True)
            try:
                r, (spec, outd) = hb_point(pump, f)
                if abs(f - 5.3) < 1e-9:
                    spec53[pump] = (spec, outd)
            except Exception as e:  # noqa: BLE001
                r = {"f_GHz": f, "pump_dBm": pump, "converged": False, "error": f"{type(e).__name__}: {e}"}
                print("  FAIL", e, flush=True)
            payload["hb_leg"][str(int(pump))].append(r)
            save()

    # ---------- junction V_j / m_j at the 10.6 GHz pump (route B), from f_s=5.3 HB ----------
    def junction(pump):
        if pump not in spec53:
            return {"status": "no_5p3_dataset"}
        spec, outd = spec53[pump]
        dss = sorted(glob.glob(str(outd / "**" / "on_*" / "*.ds"), recursive=True))
        if not dss:
            return {"status": "no_on_dataset"}
        ds = read_hb_dataset(dss[0], (5.3e9, 10.6e9))
        rows = []
        for k in range(1, 16):
            q = _spec_varactor_q_poly(spec, k - 1); c0, c1 = _capacitance_coefficients_from_q_poly(q)
            try:
                v_rp = complex(ds.node_voltage(f"n_c{k}_rp").at_mix((0, 1)))
                v_v = complex(ds.node_voltage(f"n_c{k}_v").at_mix((0, 1)))
                vj = abs(v_rp - v_v); mj = abs(c1) * vj / c0
                rows.append({"cell": k, "Vj_direct_V": vj, "m_j": mj})
            except Exception as e:  # noqa: BLE001
                rows.append({"cell": k, "error": str(e)})
        mjs = [r["m_j"] for r in rows if "m_j" in r]
        return {"dataset": "lobe2_hb/.../on_0/on_0.ckt.ds (temporary run directory; not kept)",
                "pump_freq_GHz": 10.6, "route": "B (V(n_rp)-V(n_v) at mix (0,1))",
                "per_cell": rows, "m_j_mean": float(np.mean(mjs)) if mjs else None,
                "m_j_range": [float(min(mjs)), float(max(mjs))] if mjs else None}

    payload["junction_at_10p6_pump"] = {"29dBm": junction(29.0), "20dBm": junction(20.0)}
    save()
    print("DONE wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
