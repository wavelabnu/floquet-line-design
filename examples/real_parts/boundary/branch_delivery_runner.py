"""R7 replay -- regenerates ``results/branch_delivery_vs_freq.json``.

Branch junction-delivery transfer ``T(f) = |V_j / V_bt|`` for the SMV1245
fabricated board: the fraction of the branch-top drive that reaches the varactor
junction across 2.0-14.0 GHz (step 0.01), through the package R-L, the junction
C, and the element-accurate corrected stub.

Conditions
----------
* Divider ``T = |Z_C / (Z_pkg + Z_C + Z_stub)|`` with
  ``Z_pkg = 2.0 + j w * 0.7 nH``, ``Z_C = 1/(j w C0)``, ``C0 = 7.37 pF``
  (junction CJ0 = 6.9 pF + package CP = 0.47 pF), and ``Z_stub = 1/Y_stub(f)``
  from the relocated R6 55 GHz stub ``.s1p``.
* Marked spot frequencies: 3.1 / 5.3 / 6.2 / 8.4 / 10.6 GHz.

Replayability: PURE post-processing of the committed 55 GHz stub table -- no ADS,
no HB datasets. Running this reproduces the committed JSON bit-for-bit in its
numeric arrays. Optional argv[1] overrides the output path (used for
verification without clobbering the committed artifact).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stub_table import STUB_55GHz, load_s1p  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "examples" / "real_parts" / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "branch_delivery_vs_freq.json"

# ---- 1) load stub table (read-only), admittance -> impedance ----
f_tab, Y_stub_tab = load_s1p(STUB_55GHz)
Z_stub_tab = 1.0 / Y_stub_tab

# ---- 2) T(f) over 2.0-14.0 GHz step 0.01 ----
RPK, LPK = 2.0, 0.7e-9
C0, CJ = 7.37e-12, 6.9e-12   # total(junction+CP) vs junction-only
grid = np.round(np.arange(2.0, 14.0 + 1e-9, 0.01), 2)


def T_of(fghz, Ccap):
    w = 2 * math.pi * fghz * 1e9
    Zc = 1.0 / (1j * w * Ccap)
    Zpk = RPK + 1j * w * LPK
    zr = np.interp(fghz * 1e9, f_tab, Z_stub_tab.real)
    zi = np.interp(fghz * 1e9, f_tab, Z_stub_tab.imag)
    Zst = complex(zr, zi)
    return abs(Zc / (Zpk + Zc + Zst)), abs(Zst)


T = np.array([T_of(f, C0)[0] for f in grid])
Zst_abs = np.array([T_of(f, C0)[1] for f in grid])
T_cj = np.array([T_of(f, CJ)[0] for f in grid])
maxpct = float(np.max(np.abs(T_cj - T) / np.maximum(T, 1e-30)) * 100.0)


def at(fq):
    t, z = T_of(fq, C0)
    return {"f_GHz": fq, "T": t, "Z_stub_abs_ohm": z, "T_dB": 20 * math.log10(max(t, 1e-30))}


spots = {str(fq): at(fq) for fq in (3.1, 5.3, 6.2, 8.4, 10.6)}
imax = int(np.argmax(T)); imin = int(np.argmin(T))
Tmax = {"f_GHz": float(grid[imax]), "T": float(T[imax])}
Tmin = {"f_GHz": float(grid[imin]), "T": float(T[imin])}
t62 = spots["6.2"]["T"]; t106 = spots["10.6"]["T"]; t84 = spots["8.4"]["T"]
ratios = {"T(6.2)/T(10.6)": t62 / t106, "T(6.2)/T(8.4)": t62 / t84}
sanity = {"T(6.2)*5.8V_predicted_Vj": t62 * 5.8, "route_B_direct_Vj_V": 0.0151,
          "ratio_pred_over_directB": (t62 * 5.8) / 0.0151}


# ---- ASCII sketch (log T) ----
def sketch(grid, T):
    ncol = 60
    idx = np.linspace(0, len(grid) - 1, ncol).astype(int)
    fs = grid[idx]; ts = np.log10(np.maximum(T[idx], 1e-12))
    tmin, tmax = ts.min(), ts.max()
    rows = 10
    lines = []
    for r in range(rows):
        hi = tmax - (tmax - tmin) * r / (rows - 1)
        lo = tmax - (tmax - tmin) * (r + 0.999) / (rows - 1)
        line = "".join("#" if lo <= v <= hi + (tmax - tmin) / rows else " " for v in ts)
        lines.append("%6.1f |%s" % (hi, line))
    lines.append("       +" + "-" * ncol)
    lines.append("        %-6.1f%s%6.1f GHz  (y=log10 T)" % (fs[0], " " * (ncol - 12), fs[-1]))
    return "\n".join(lines)


sk = sketch(grid, T)

payload = {
    "headline": "SMV1245 branch junction-delivery transfer T(f)=|V_j/V_bt| (package R-L + junction C + element-accurate stub)",
    "provenance": {
        "stub_table_source": "R6 element-accurate stub S-param, relocated to "
                             "results/stub_tables/smv1245_stub_55GHz.s1p (0.05-55 GHz, 5496 pts); read read-only.",
        "formula": "T = |Z_C/(Z_pkg+Z_C+Z_stub)|; Z_pkg=2.0+jw*0.7nH; Z_C=1/(jwC0); Z_stub=1/Y_stub(f) from table",
        "C0_convention": "C0=7.37pF (junction CJ0=6.9pF + CP=0.47pF, R6 convention). CAVEAT: CP strictly PARALLELS "
                         "the junction, so the total shunt cap at the node IS 7.37pF and is the correct divider C; "
                         "junction-only CJ0=6.9pF gives Z_C ~6.8%% smaller.",
        "C_convention_max_pct_diff_T_2to14GHz": maxpct,
        "post_processing_only": True,
    },
    "grid_GHz": [float(x) for x in grid],
    "T": [float(x) for x in T],
    "Z_stub_abs_ohm": [float(x) for x in Zst_abs],
    "spot_table": spots,
    "T_global_max": Tmax, "T_global_min": Tmin,
    "ratios": ratios, "sanity_vs_R6_routeB": sanity,
    "ascii_sketch": sk,
}
if __name__ == "__main__":
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("DONE wrote", OUT)
    print("C-convention max %diff (2-14GHz):", round(maxpct, 3), "%")
    print("SPOT TABLE (f, T, T_dB, |Z_stub|):")
    for k, v in spots.items():
        print("  %-5s  T=%.4e  T_dB=%+.2f  |Z_stub|=%.1f ohm" % (k, v["T"], v["T_dB"], v["Z_stub_abs_ohm"]))
    print("T_max=%.4e @ %.2f GHz ; T_min=%.4e @ %.2f GHz" % (Tmax["T"], Tmax["f_GHz"], Tmin["T"], Tmin["f_GHz"]))
    print("T(6.2)/T(10.6)=%.3f  T(6.2)/T(8.4)=%.3f" % (ratios["T(6.2)/T(10.6)"], ratios["T(6.2)/T(8.4)"]))
