"""R6 replay -- regenerates ``results/smv1245_mstar_arc_v2.json``.

Corrected-stub m* arc (v2): the element-accurate ADS-tabulated stub is injected
into the analytical engine via ``stub_y_fn_override``, and the modulation depth
is referenced to the varactor JUNCTION (below the SDD2P + package divider). It
asks how deep a junction modulation the +16 dB coherent target needs (D16) and
whether the fabricated board delivers it -- the "model boundary" the SM figure
documents.

Conditions
----------
* f_signal = 3.1 GHz, f_pump = 6.2 GHz, N = 6 truncation, SMV1245-079LF at 0 V.
* Stub table: the R6 element-accurate 1-port ``S_Param`` sweep, relocated to
  ``results/stub_tables/smv1245_stub_55GHz.s1p`` (0.05-55 GHz, 5496 pts). The
  exact ADS recipe that produced it is recorded in ``ADS_STUB_DECK`` below and in
  the ``.s1p`` header. Extended to 55 GHz because truncation_N=6 evaluates
  z_stub up to 13*f_max = 52 GHz; a 15 GHz table would hard-error at n>=3.
* m-grid = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95]; D16 target = 16 dB.

Replayability: the stub load + linear parity gate + m-sweep + D16 bisection are
PURE analytical (no ADS). Only ``step4`` (delivered junction depth) reads the R3
ADS HB dataset under ``examples/real_parts/outputs/bias_sweep/...`` and degrades
to a ``datasets_absent`` status when that gitignored scratch is not present.
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
from tvtl_ads_builder_v2.analytical import build_stub_y_fn_from_spec  # noqa: E402
from tvtl_ads_builder_v2.device_spec import with_var_overrides  # noqa: E402
from tvtl_ads_builder_v2.analysis.hb_dataset import read_hb_dataset  # noqa: E402
from tvtl_ads_builder_v2.analysis.floquet_comparison import (  # noqa: E402
    _spec_varactor_q_poly, _capacitance_coefficients_from_q_poly)
from _stub_table import STUB_55GHz, load_s1p, make_y_override  # noqa: E402

# Exact ADS S_Param recipe that produced smv1245_stub_55GHz.s1p (provenance only;
# reading the relocated .s1p needs no ADS):
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
    'SweepPlan:SP1_stim Start=0.05 GHz Stop=55 GHz Lin=5496\n')

OUT = REPO / "examples" / "real_parts" / "results" / "smv1245_mstar_arc_v2.json"
FS0, FM0 = 3.1e9, 6.2e9
DEV = get_entry("SMV1245-079LF").to_realistic_varactor(v_bias_v=0.0)
SPEC0 = spec_smv1245()
C0F, C1F, RPK, LPK = 7.37e-12, -3.3514285714285714e-12, 2.0, 0.7e-9

# ---------- STEP 0b: element-accurate stub, from the relocated .s1p ----------
F_TAB, Y_TAB = load_s1p(STUB_55GHz)
FLO, FHI = F_TAB[0], F_TAB[-1]
y_stub_override = make_y_override(F_TAB, Y_TAB)


def gain(m, fs=FS0, fm=FM0):
    return level10_realistic_param_gain(
        with_var_overrides(SPEC0, {"f_signal": fs, "f_pump": fm}), DEV,
        m_branch_pump=float(m), truncation_N=6, stub_y_fn_override=y_stub_override)


# sanity: tabulated vs R5 single-chain tracer at 3.1 GHz
y_tab_31 = y_stub_override(2 * math.pi * FS0)
y_tr_31 = build_stub_y_fn_from_spec(SPEC0, 0)(2 * math.pi * FS0)


# ---------- STEP 1: linear parity gate ----------
def pumpoff_s21(fs):
    return gain(1e-9, fs=fs, fm=2.0 * fs)["abs_gain_off_dB"]


fs_list = [round(1.5 + 0.02 * i, 3) for i in range(int((4.0 - 1.5) / 0.02) + 1)]
curve = [{"f_GHz": f, "S21_off_dB": pumpoff_s21(f * 1e9)} for f in fs_list]
notch = min(curve, key=lambda r: r["S21_off_dB"])
s21_31 = next(r["S21_off_dB"] for r in curve if abs(r["f_GHz"] - 3.1) < 1e-9)
res_s21 = abs(s21_31 - (-6.457)); res_notch = abs(notch["f_GHz"] - 2.46)
status = ("PASS" if (res_s21 <= 1.0 and res_notch <= 0.08)
          else "FLAGGED" if (res_s21 <= 2.5 and res_notch <= 0.15) else "FAIL")
NOTCH_FS = notch["f_GHz"] * 1e9
step1 = {"S21_off_at_3p1_dB": s21_31, "hb_ref_dB": -6.457, "residual_S21_dB": res_s21,
         "deepest_notch": notch, "notch_target_GHz": 2.46, "residual_notch_GHz": res_notch,
         "gate_status": status, "notch_floor_below_-60dB_artifact": notch["S21_off_dB"] < -60,
         "curve": curve,
         "note": "analytical-vs-MEASURED lineage gap: model notch ~2.46 vs measured board 3.1 GHz (documented, not tuned)",
         "sanity_y_stub_3p1GHz": {"tabulated_S": [y_tab_31.real, y_tab_31.imag],
                                  "r5_tracer_S": [y_tr_31.real, y_tr_31.imag]}}


def net_at_peak(r):
    return r["coherent_peak_dB"] + r["abs_gain_off_dB"]  # 20log10(|Hs|+|Hi|) = out/available


def row(m, fs, fm):
    r = gain(m, fs=fs, fm=fm)
    return {"m_branch": m, "depth_eff_junction": r["depth_effective"],
            "coherent_peak_dB": r["coherent_peak_dB"], "coherent_min_dB": r["coherent_min_dB"],
            "net_out_avail_at_peak_dB": net_at_peak(r), "pumpoff_ref_dB": r["abs_gain_off_dB"]}


M_GRID = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95]


# ---------- D16 bisection on junction depth_eff ----------
def d16(fs, fm, target=16.0, mmax=0.95):
    hi = gain(mmax, fs=fs, fm=fm)
    if hi["coherent_peak_dB"] < target:
        return {"bracketed": False, "m_max": mmax, "coherent_peak_at_mmax_dB": hi["coherent_peak_dB"],
                "depth_eff_at_mmax": hi["depth_effective"], "net_at_mmax_dB": net_at_peak(hi)}
    lo_m, hi_m = 0.0, mmax
    for _ in range(40):
        mid = 0.5 * (lo_m + hi_m); g = gain(mid, fs=fs, fm=fm)
        if g["coherent_peak_dB"] >= target: hi_m = mid
        else: lo_m = mid
        if abs(gain(hi_m, fs=fs, fm=fm)["depth_effective"] - gain(lo_m, fs=fs, fm=fm)["depth_effective"]) < 0.002:
            break
    g = gain(hi_m, fs=fs, fm=fm)
    return {"bracketed": True, "m_star": hi_m, "D16_depth_eff_junction": g["depth_effective"],
            "coherent_peak_dB": g["coherent_peak_dB"], "net_out_avail_dB": net_at_peak(g)}


print("stub loaded. running step1...", flush=True)
step2 = [row(m, FS0, FM0) for m in M_GRID]
print("step1 status=%s S21(3.1)=%.2f notch=%.2f@%.1fdB" %
      (status, s21_31, notch["f_GHz"], notch["S21_off_dB"]), flush=True)
step3 = {"notch_f_GHz": notch["f_GHz"], "floor_at_notch_dB": notch["S21_off_dB"],
         "rows": [row(m, NOTCH_FS, 2 * NOTCH_FS) for m in M_GRID]}
D16_31 = d16(FS0, FM0); D16_notch = d16(NOTCH_FS, 2 * NOTCH_FS)


# ---------- STEP 4: delivered junction depth at 0 V (reads the R3 ADS HB dataset) ----------
def step4():
    dp = sorted(glob.glob(str(REPO / "examples/real_parts/outputs/bias_sweep/b0p0_p29p0/on_255/*.ds")))
    if not dp:
        return {"status": "datasets_absent",
                "note": "R3 ADS HB scratch (outputs/bias_sweep/, gitignored) not present; "
                        "re-run bias_sweep_runner.py under the ADS python to regenerate it."}
    ds = read_hb_dataset(dp[0], (FS0, FM0))
    w = 2 * math.pi * FM0
    Zc = 1 / (1j * w * C0F); Zpk = RPK + 1j * w * LPK
    divider = abs(Zc / (Zpk + Zc))
    scale20 = 10 ** ((20.0 - 29.0) / 20.0)
    rows = []
    for k in range(1, 16):
        q = _spec_varactor_q_poly(SPEC0, k - 1); c0, c1 = _capacitance_coefficients_from_q_poly(q)
        vbt = complex(ds.node_voltage(f"n_c{k}_b").at_mix((0, 1)))
        vj_div = abs(vbt) * divider
        try:
            v_rp = complex(ds.node_voltage(f"n_c{k}_rp").at_mix((0, 1)))
            v_v = complex(ds.node_voltage(f"n_c{k}_v").at_mix((0, 1)))
            vj_direct = abs(v_rp - v_v); vj_rp = abs(v_rp)
        except Exception:
            vj_direct = vj_rp = None
        vj = vj_direct if vj_direct is not None else vj_div
        mj = abs(c1) * vj / c0
        rows.append({"cell": k, "Vbt_V": abs(vbt), "Vj_divider_V": vj_div,
                     "Vj_direct_V": vj_direct, "Vj_rp_V": vj_rp,
                     "m_j_29dBm": mj, "m_j_20dBm_scaled": mj * scale20})
    mj29 = [r["m_j_29dBm"] for r in rows]; mj20 = [r["m_j_20dBm_scaled"] for r in rows]
    return {"divider_formula": "Vj = Vbt*|Zc/(Rpk+jwLpk+Zc)|, Zc=1/(jwC0), w=2pi*6.2GHz, C0=7.37pF,R=2,L=0.7nH",
            "divider_magnitude": divider, "junction_side_net": "n_c{k}_rp (post R_pkg) minus n_c{k}_v (SDD far side)",
            "note_20dBm": "sqrt(P) voltage-scaled from 29dBm (flagged, not re-solved)",
            "per_cell": rows, "m_j_29_mean": float(np.mean(mj29)), "m_j_29_range": [float(min(mj29)), float(max(mj29))],
            "m_j_20_mean": float(np.mean(mj20)), "m_j_20_range": [float(min(mj20)), float(max(mj20))]}


s4 = step4()


# ---------- STEP 5: verdict ----------
def verdict():
    if not D16_notch.get("bracketed"):
        return {"line": "not-bracketed: +16 dB coherent_peak not reached at the null even at m=0.95",
                "D16_required": None, "delivered_m_j_29_mean": s4.get("m_j_29_mean"),
                "delivered_m_j_20_mean": s4.get("m_j_20_mean")}
    d16v = D16_notch["D16_depth_eff_junction"]
    if "m_j_29_range" not in s4:
        return {"line": "delivered depth unavailable (step4 datasets absent)",
                "D16_required_junction": d16v}
    lo29, hi29 = s4["m_j_29_range"]; lo20, hi20 = s4["m_j_20_range"]
    cons29 = lo29 <= d16v <= hi29; cons20 = lo20 <= d16v <= hi20
    line = ("consistent" if (cons29 or cons20) else "inconsistent") + \
           f": D16(required,junction)={d16v:.4f} vs delivered m_j 29dBm={s4['m_j_29_mean']:.4f} 20dBm={s4['m_j_20_mean']:.4f}"
    return {"line": line, "D16_required_junction": d16v,
            "delivered_m_j_29_mean": s4["m_j_29_mean"], "delivered_m_j_20_mean": s4["m_j_20_mean"]}


v = verdict()
note6 = ("D16>0.2: an S-linear twin would need the 3-6 line src edit in _branch_admittance "
         "(a separate change, not made here)."
         if (D16_notch.get("D16_depth_eff_junction", 0) or 0) > 0.2 or
            (D16_31.get("D16_depth_eff_junction", 0) or 0) > 0.2 else "D16<=0.2 at bracketed points; no note.")

payload = {
    "headline": "SMV1245 corrected-stub (ADS-tabulated element-accurate) junction-referenced m* arc v2",
    "step0_reference_plane": {
        "what_override_replaces": "the PASSIVE microstrip stub admittance y_stub looking into the stub from the "
            "varactor's bottom terminal (below SDD2P + package), plane n_c{k}_s0",
        "cites": ["realistic_c_gain.py:169-172 (y_stub=build_stub_y_fn_from_spec; z_stub_fn=1/y_stub)",
                  "realistic_c_gain.py:243 (z_stub_fn passed to tv_capacitor_harmonic_admittance_general)",
                  "realistic_c_gain.py:291 (stub_y_fn_override kwarg)"],
        "call_signature": "level10_realistic_param_gain(spec, dev, m_branch_pump=m, truncation_N=6, "
            "stub_y_fn_override=y_stub_override)  [y_stub_override(omega)->Y_stub S; omega rad/s; conj for omega<0]"},
    "step0b_stub_table": {
        "topology": "1-port: n_s0 -MLIN(1.93x6.83)- n_s1 [VIAGND d1.6 w1.93] -MLIN(1.41x5.25)- n_s2 "
                    "[VIAGND d1.6 w1.41] -MLIN(1.98x1.04)- n_open [MLEF w1.98]; ROGERS_4003C",
        "params_source": "real_part_designs.py:128 (_SMV1245_STUB_PARAMS_M), :129 (_SMV1245_SEC_WIDTHS_M), "
                         ":143-148 (_viagnd), :82 (_OPEN=MlefOpen)",
        "sweep": "ADS S_Param 0.05-55 GHz, 5496 pts (~0.01 step), relocated to "
                 "results/stub_tables/smv1245_stub_55GHz.s1p. Extended above the originally planned 15 GHz because "
                 "truncation_N=6 evaluates z_stub up to 13*f_max=52 GHz; a 15 GHz table would hard-error at n>=3.",
        "y_stub_3p1GHz_tabulated_S": [y_tab_31.real, y_tab_31.imag],
        "y_stub_3p1GHz_r5_tracer_S": [y_tr_31.real, y_tr_31.imag],
        "differ": "YES (element-accurate vs single-chain tracer that drops the 2 mid-stub VIAGND)"},
    "step1_linear_parity_gate": step1,
    "step2_msweep_3p1GHz": step2,
    "step3_null_referenced": step3,
    "D16": {"at_3p1GHz": D16_31, "at_notch": D16_notch, "target_dB": 16.0},
    "step4_delivered_junction_depth": s4,
    "step5_verdict": v,
    "step6_note": note6,
    "provenance": {"dev": "SMV1245-079LF CJ0=6.9 VJ=3.5 M=1.7 CP=0.47 R=2.0 L_pkg=0.7nH; C0=7.37pF C1=-3.351pF/V",
                   "runner": "boundary/mstar_arc_v2_runner.py (additive; stub from relocated .s1p + R3 HB reads)"},
}
if __name__ == "__main__":
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("DONE wrote", OUT, flush=True)
    print("VERDICT:", v["line"], flush=True)
    print("D16@3.1:", D16_31.get("D16_depth_eff_junction", D16_31), flush=True)
    print("D16@notch:", D16_notch.get("D16_depth_eff_junction", D16_notch), flush=True)
