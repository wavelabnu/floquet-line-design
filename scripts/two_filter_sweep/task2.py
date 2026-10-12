"""TASK 2 -- the difficulty law: usable fraction vs C0.

Repeats TASK 1's grid for C0 in {0.3, 1.0, 3.0, 7.37} pF with the geometry, the
package R-L, and the stub otherwise unchanged. This is a MODEL PROBE, not a bias
sweep: C0 is set directly as the total static shunt capacitance the branch
presents (package CP folded to zero and the whole target on the static override),
which at m_branch_pump = 0 makes the shunt element exactly j*w*C0. No real bias
point produces C0 = 0.3 pF on an SMV1245.

Closed form under test:

    u_min = Z0 * T_req * w * C0 / 2
    frac  = (pi - 2*arccos(1/sqrt(1 + u_min^2))) / pi

``frac`` is the admissible fraction of electrical length theta at one frequency
and one C0, under conditions (A)+(B) only -- i.e. it is the closed-form
counterpart of the *feasible* boolean, not the *usable* one (which adds the
signal-side passband test). Both are reported.

Runs at the nominal (repo-default) VIAGND effective L. Writes task2_summary.json.
``frac_usable_measured`` per C0 is the "realized" fraction of Fig. 2(c):
0.6593 / 0.3396 / 0.0904 / 0.0139 at C0 = 0.3 / 1 / 3 / 7.37 pF.

    python scripts/two_filter_sweep/task2.py [--out runs/two_filter_sweep]   (about 30 min)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import REPO, T_REQ
from sweep import (
    difficulty_law_fraction, fp_grid, main_line_z0, run_sweep, save,
    windows_from,
)

C0_LIST_PF = (0.3, 1.0, 3.0, 7.37)
L_EFF_H = 0.1949e-9              # repo default, unchanged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs" / "two_filter_sweep"))
    ap.add_argument("--c0-pf", type=float, nargs="*", default=list(C0_LIST_PF),
                    help="subset of C0 values (pF); the summary lists only these")
    args = ap.parse_args(argv)
    HERE = Path(args.out)
    HERE.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    fps = fp_grid()
    z0_band = np.array([main_line_z0(f * 1e9) for f in fps])

    rows = []
    for c0_pf in args.c0_pf:
        label = f"C0_{c0_pf:g}pF".replace(".", "p")
        print(f"\n=== TASK 2: C0 = {c0_pf} pF ===", flush=True)
        res = run_sweep(l_eff_h=L_EFF_H, c0_f=c0_pf * 1e-12, label=label)
        save(res, HERE / f"task2_{label}.npz")
        wins = windows_from(res)

        frac_band = difficulty_law_fraction(c0_pf * 1e-12, fps * 1e9, 0.0)
        # per-frequency Z0 (Hammerstad, dispersive) rather than one scalar
        frac_band = np.array([
            difficulty_law_fraction(c0_pf * 1e-12, f * 1e9, float(z0))
            for f, z0 in zip(fps, z0_band)]).ravel()

        rec = {
            "C0_pF": c0_pf,
            "runtime_s": res["runtime_s"],
            "n_points": int(res["usable"].size),
            "n_feasible": int(res["feasible"].sum()),
            "n_usable": int(res["usable"].sum()),
            "frac_feasible_measured": float(res["feasible"].mean()),
            "frac_usable_measured": float(res["usable"].mean()),
            "closed_form_frac_band_mean": float(frac_band.mean()),
            "closed_form_frac_at_3p1GHz": float(difficulty_law_fraction(
                c0_pf * 1e-12, 3.1e9, main_line_z0(3.1e9))),
            "closed_form_frac_at_6p2GHz": float(difficulty_law_fraction(
                c0_pf * 1e-12, 6.2e9, main_line_z0(6.2e9))),
            "closed_form_frac_at_9GHz": float(difficulty_law_fraction(
                c0_pf * 1e-12, 9.0e9, main_line_z0(9.0e9))),
            "n_windows": len(wins),
            "widest_window": (max(wins, key=lambda w: w["rel_bw_pct"])
                              if wins else None),
        }
        rows.append(rec)
        print("  -> feasible %.5f  usable %.5f  closed-form(band mean) %.5f  "
              "%d windows"
              % (rec["frac_feasible_measured"], rec["frac_usable_measured"],
                 rec["closed_form_frac_band_mean"], len(wins)), flush=True)
        (HERE / f"task2_windows_{label}.json").write_text(
            json.dumps(wins, indent=2), encoding="utf-8")

    order_meas_feas = [r["frac_feasible_measured"] for r in rows]
    order_meas_use = [r["frac_usable_measured"] for r in rows]
    order_cf = [r["closed_form_frac_band_mean"] for r in rows]

    def monotone_decreasing_in_C0(vals):
        return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    payload = {
        "task": "TASK 2 -- difficulty law, usable fraction vs C0",
        "probe_note": ("MODEL PROBE, not a bias sweep: C0 is set directly as the "
                       "total static shunt capacitance (CP folded to 0, static "
                       "override carries the target). Geometry, package R-L, "
                       "stub and grid unchanged."),
        "T_req": T_REQ,
        "L_eff_nH": L_EFF_H * 1e9,
        "C0_list_pF": list(args.c0_pf),
        "rows": rows,
        "monotonicity": {
            "expected": "usable/feasible fraction increases as C0 decreases",
            "measured_feasible_monotone": monotone_decreasing_in_C0(order_meas_feas),
            "measured_usable_monotone": monotone_decreasing_in_C0(order_meas_use),
            "closed_form_monotone": monotone_decreasing_in_C0(order_cf),
        },
        "total_runtime_s": time.perf_counter() - t_all,
    }
    (HERE / "task2_summary.json").write_text(json.dumps(payload, indent=2),
                                             encoding="utf-8")

    print("\n=== TASK 2 summary ===")
    print("%8s %14s %14s %18s %10s" % ("C0 [pF]", "frac_feasible", "frac_usable",
                                       "closed-form frac", "windows"))
    for r in rows:
        print("%8.2f %14.6f %14.6f %18.6f %10d"
              % (r["C0_pF"], r["frac_feasible_measured"],
                 r["frac_usable_measured"], r["closed_form_frac_band_mean"],
                 r["n_windows"]))
    m = payload["monotonicity"]
    print("  monotone (increases as C0 decreases): feasible=%s usable=%s "
          "closed-form=%s" % (m["measured_feasible_monotone"],
                              m["measured_usable_monotone"],
                              m["closed_form_monotone"]))
    print("  total %.1f s" % payload["total_runtime_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
