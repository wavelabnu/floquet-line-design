"""TASK 1 -- feasibility sweep over stub geometry.

Two passes, differing only in the ADS-``VIAGND`` effective inductance used by the
mid-stub shunts:

  nominal    L_eff = 0.1949 nH -- the repo default, untouched. Characterized at
             connecting width W = 0.945 mm.
  W-matched  L_eff = 0.154 nH  -- the W = 1.6 mm end of the same documented
             characterization, the nearest point to this board's W = 1.93 /
             1.41 mm mid-stub vias.

The nominal pass is the primary result (zero tuning). The W-matched pass is the
sensitivity replicate; GATE 0 shows it is the more faithful of the two against
the ADS table, so agreement between the two passes is the robustness statement.

Writes task1_<label>.npz, task1_windows_<label>.json, windows.csv and task1_summary.json.

    python scripts/two_filter_sweep/task1.py [--out runs/two_filter_sweep]   (about 15 min)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from _common import C0_F_NOMINAL, REPO, T_REQ
from sweep import (
    FP_HI_GHZ, FP_LO_GHZ, FP_STEP_GHZ_DEFAULT, S_HI, S_LO, S_STEP,
    main_line_z0, run_sweep, save, windows_from,
)

PASSES = (("nominal_L0p1949nH", 0.1949e-9),
          ("Wmatched_L0p154nH", 0.154e-9))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs" / "two_filter_sweep"))
    HERE = Path(ap.parse_args(argv).out)
    HERE.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    summary = {}
    all_windows = {}
    grids = {}

    for label, l_eff in PASSES:
        print(f"\n=== TASK 1 pass: {label} (L_eff = {l_eff * 1e9:.4f} nH) ===",
              flush=True)
        res = run_sweep(l_eff_h=l_eff, c0_f=C0_F_NOMINAL, label=label)
        save(res, HERE / f"task1_{label}.npz")
        wins = windows_from(res)
        all_windows[label] = wins
        grids[label] = res
        n_pts = res["usable"].size
        summary[label] = {
            "l_eff_nH": l_eff * 1e9,
            "runtime_s": res["runtime_s"],
            "grid": {"n_s": int(res["s"].size), "n_fp": int(res["fp_GHz"].size),
                     "n_points": int(n_pts)},
            "n_feasible": int(res["feasible"].sum()),
            "n_usable": int(res["usable"].sum()),
            "frac_feasible": float(res["feasible"].mean()),
            "frac_usable": float(res["usable"].mean()),
            "n_windows": len(wins),
            "n_s_with_a_window": len({w["s"] for w in wins}),
            "widest_window": (max(wins, key=lambda w: w["rel_bw_pct"])
                              if wins else None),
        }
        print(f"  -> {res['runtime_s']:.1f} s; feasible {res['feasible'].sum()}"
              f"/{n_pts}, usable {res['usable'].sum()}/{n_pts}, "
              f"{len(wins)} windows", flush=True)
        (HERE / f"task1_windows_{label}.json").write_text(
            json.dumps(wins, indent=2), encoding="utf-8")

    # ---- windows.csv: the primary (nominal) pass, sorted by relative BW ---- #
    prim_label = PASSES[0][0]
    prim = sorted(all_windows[prim_label], key=lambda w: -w["rel_bw_pct"])
    with open(HERE / "windows.csv", "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["s", "f_lo_GHz", "f_hi_GHz", "rel_bw_pct",
                      "alpha_at_center", "T_at_center", "f_s_GHz",
                      "abs_bw_GHz", "n_points", "pass"])
        for lab in (prim_label, PASSES[1][0]):
            for w in sorted(all_windows[lab], key=lambda x: -x["rel_bw_pct"]):
                wtr.writerow([f"{w['s']:.2f}", f"{w['f_lo_GHz']:.3f}",
                              f"{w['f_hi_GHz']:.3f}", f"{w['rel_bw_pct']:.4f}",
                              f"{w['alpha_at_center']:.6e}",
                              f"{w['T_at_center']:.6f}", f"{w['f_s_GHz']:.4f}",
                              f"{w['abs_bw_GHz']:.3f}", w["n_points"], lab])

    # ---- cross-pass agreement -------------------------------------------- #
    a, b = grids[PASSES[0][0]], grids[PASSES[1][0]]
    agree = {
        "usable_boolean_agreement_pct": float(100 * np.mean(a["usable"] == b["usable"])),
        "feasible_boolean_agreement_pct": float(100 * np.mean(a["feasible"] == b["feasible"])),
        "usable_in_both": int((a["usable"] & b["usable"]).sum()),
        "usable_in_nominal_only": int((a["usable"] & ~b["usable"]).sum()),
        "usable_in_Wmatched_only": int((~a["usable"] & b["usable"]).sum()),
    }

    payload = {
        "task": "TASK 1 -- two-filter feasibility sweep over stub geometry",
        "T_req": T_REQ,
        "grid_spec": {"s": [S_LO, S_HI, S_STEP],
                      "f_p_GHz": [FP_LO_GHZ, FP_HI_GHZ, FP_STEP_GHZ_DEFAULT],
                      "s_scales": "all three stub section LENGTHS; widths fixed"},
        "held_fixed": {
            "main_width_mm": 4.27, "period_mm": 6.11, "substrate": "ROGERS_4003C",
            "R_series_ohm": 2.0, "L_pkg_nH": 0.7, "C0_pF": 7.37,
            "baseline_s1_stub_mm": [[1.93, 6.83], [1.41, 5.25], [1.98, 1.04]],
        },
        "main_line_Z0_ohm": {"3.1GHz": main_line_z0(3.1e9),
                             "6.2GHz": main_line_z0(6.2e9),
                             "9.0GHz": main_line_z0(9.0e9)},
        "passes": summary,
        "cross_pass_agreement": agree,
        "total_runtime_s": time.perf_counter() - t_all,
        "window_edge_convention": ("first/last usable grid sample; a 1-sample "
                                   "window therefore has abs_bw = 0"),
    }
    (HERE / "task1_summary.json").write_text(json.dumps(payload, indent=2),
                                             encoding="utf-8")

    print("\n=== TASK 1 summary ===")
    for label, s in summary.items():
        ww = s["widest_window"]
        print(f"  {label}: usable {s['n_usable']}/{s['grid']['n_points']} "
              f"({100 * s['frac_usable']:.3f}%), {s['n_windows']} windows over "
              f"{s['n_s_with_a_window']} s-values")
        if ww:
            print(f"     widest: s={ww['s']:.2f}  {ww['f_lo_GHz']:.2f}-"
                  f"{ww['f_hi_GHz']:.2f} GHz  rel_bw={ww['rel_bw_pct']:.2f}%  "
                  f"f_s={ww['f_s_GHz']:.3f} GHz")
    print(f"  cross-pass usable agreement: "
          f"{agree['usable_boolean_agreement_pct']:.3f}%")
    print(f"  total {payload['total_runtime_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
