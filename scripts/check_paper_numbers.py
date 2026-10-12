"""Recompute, from this release, every analytical number of the paper that the code produces.

Each line prints the recomputed value, the value the paper quotes, and PASS when the value rounds to the
quoted value at the quoted precision (counts: exact). A FAIL is reported, never adjusted.

    python scripts/check_paper_numbers.py [--workers 15] [--out runs/paper_numbers.json] [--skip-slow]

Runtime: about 4 min (the 0 V SMV1245 screen about 2.5 min, the Fig. 7 dense map under 1 min with
15 workers). ``--skip-slow`` leaves out those two. Tier of every number: ANALYTICAL.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (HERE / "dense_map", HERE / "two_filter_sweep", REPO / "examples" / "real_parts" / "boundary",
           REPO / "examples" / "real_parts", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                                                # noqa: E402

import dense_map                                                                  # noqa: E402
from tvtl.design.feasibility import (                                            # noqa: E402
    ALPHA_MAX, T_REQ, T_REQ_SCREEN_2026_09, candidate_cell_factory, feasibility)
from tvtl.varactor_palette import get_entry                                       # noqa: E402
from tvtl_ads_builder_v2.analytical import cascade_budget as cb                   # noqa: E402

MEAS = REPO / "tests" / "fixtures" / "measured_dispersion" / "dispersion_from_fft_ridge_smooth_b.csv"
ROWS: list[dict] = []


def check(key, what, value, expected, decimals=None, source=""):
    """decimals None: exact equality; else round(value, decimals) == expected."""
    ok = (value == expected) if decimals is None else (value is not None and round(value, decimals) == expected)
    ROWS.append({"key": key, "quantity": what, "value": value, "expected": expected,
                 "rule": "exact" if decimals is None else f"rounds to {decimals} decimals",
                 "result": "PASS" if ok else "FAIL", "source": source})


def t_req_section():
    cell = dense_map.asbuilt_cell()
    e = get_entry("SMV1245-079LF")
    k = e.m / e.vj_v * e.cj0_pf / (e.cj0_pf + e.cp_pf)
    vbt = np.abs(cb.branch_top_profile([cell] * 20, 6.2e9, 19.5))
    gate_d, refused, mstar = [], [], {}
    for fs in [round(2.0 + 0.1 * j, 1) for j in range(41)]:
        out = feasibility(cell, fs * 1e9, N=20)
        if "tee_model_out_of_range" in out["reasons"]:
            refused.append(fs)
            continue
        if out["alpha_fs"] > ALPHA_MAX:
            continue
        gate_d.append(fs)
        try:
            r = cb.m_star(cell, fs * 1e9, N=20)
        except ValueError as exc:
            if getattr(exc, "reason", None) != "tee_model_out_of_range":
                raise
            refused.append(fs)
            continue
        if r["reached"]:
            mstar[fs] = r["m_star"]
    fs_d = min(mstar, key=lambda f: (mstar[f], f))
    m_d = mstar[fs_d]
    src = "cascade-anchored threshold run (data deposit runs/r10_mstar_anchor)"
    check("k", "k = |C1|/C0 per volt, SMV1245 at 0 V", k, 0.4547, 4, "palette SPICE (M/V_J)(C_J0/C_0)")
    check("Vbt", "V_bt^max, +19.5 dBm, f_p 6.2 GHz, N 20 (V)", float(vbt.max()), 2.98197, 5, src)
    check("Vbt_cell", "cell of V_bt^max", int(vbt.argmax()) + 1, 4, None, src)
    check("fs_gateD", "first f_s (2.0-6.0 GHz, 0.1 GHz) passing gate D (GHz)", gate_d[0], 3.3, 1, src)
    check("fs_mstar_D", "f_s of m*_D (GHz)", fs_d, 3.3, 1, src)
    check("mstar_D", "m*_D = min m* over gate-D f_s", m_d, 0.236, 3, "paper m* = 0.236")
    check("mstar_D_exact", "m*_D (bisection upper bracket)", m_d, 0.235625, 6, src)
    t = m_d / (0.4547 * float(vbt.max()))
    check("T_req", "T_req = m*_D / (k V_bt^max)", t, 0.17378, 5, src)
    check("T_req_const", "release default T_REQ", T_REQ, 0.17378, None, "tvtl.design.feasibility")
    check("T_req_screen", "named 2026-09 screening threshold", T_REQ_SCREEN_2026_09, 0.1438, None, "Fig. 2b,c")
    return {"gate_D_fs_GHz": gate_d, "tee_refused_fs_GHz": refused, "m_star_by_fs": mstar}


def cascade_section():
    cell = dense_map.asbuilt_cell()
    rows = list(csv.DictReader(MEAS.open(encoding="utf-8")))
    meas = ([float(r["Frequency_GHz"]) for r in rows], [float(r["beta_norm"]) for r in rows])
    b = cb.cascade_budget(cell, 3.1e9, N=20, measured=meas)
    src = "per-cell budget of the threshold run (data deposit runs/r10_mstar_anchor), Fig. 6"
    check("dtheta", "Delta_theta at 3.1 GHz (deg)", b["delta_theta_deg"], 55.55, 2, src)
    check("Ncoh", "N_coh at 3.1 GHz (cells)", b["N_coh"], 3.24, 2, src)
    check("kappa_m", "kappa/m at 3.1 GHz", b["kappa_per_m"], 0.0541, 4, src)
    check("mEP_model", "m_EP at 3.1 GHz, model dispersion", b["m_EP"], 8.96, 2, src)
    check("dtheta_meas", "Delta_theta at 3.1 GHz, measured dispersion (deg)", b["delta_theta_meas_deg"], 24.75, 2, src)
    check("mEP_meas", "m_EP at 3.1 GHz, measured dispersion", b["m_EP_meas"], 3.99, 2, src)
    m31 = cb.m_star(cell, 3.1e9, N=20)["m_star"]
    check("mstar_3p1", "m*(3.1 GHz)", m31, 0.108, 3, "Fig. 6 (a)")
    return b


def law_section():
    from _stub_table import load_s1p, make_y_override, STUB_80GHz
    from real_part_designs import spec_smv1245
    from tvtl_ads_builder_v2.analytical.delivered_gain import BranchCell, cell_functions_from_spec
    var = get_entry("SMV1245-079LF").to_realistic_varactor(v_bias_v=0)
    tab = cell_functions_from_spec(spec=spec_smv1245(n_cells=1), varactor=var,
                                   stub_y_fn=make_y_override(*load_s1p(STUB_80GHz)))[0]
    # Results 2: the nominal (as-built, tabulated-stub) cell at the 3.1 GHz and 5.3 GHz signals, and the
    # via-fitted cell, with the values the paper quotes
    a, b = feasibility(tab, 3.1e9, N=20), feasibility(tab, 5.3e9, N=20)
    src = "Results 2"
    check("T62_nominal", "nominal cell: T(6.2 GHz)", a["T_2fs"], 0.0026, 4, src)
    check("a62_nominal", "nominal cell: alpha(6.2 GHz) (Np/cell)", a["alpha_2fs_Np_per_cell"], 0.0024, 4, src)
    refs = json.loads((REPO / "tests/fixtures/r6_delivery_core/r5_reference_cells.json").read_text())
    row = next(r for r in refs if r["name"] == "via_fitted")
    vals = {c["f_Hz"]: c for c in row["cells"]}
    dec = lambda v: np.asarray(v["re"]) + 1j * np.asarray(v["im"])
    fit = lambda f: BranchCell(dec(vals[f]["left"]), dec(vals[f]["right"]), complex(dec(vals[f]["series"])),
                               row["C0"], row["M"], row["Vj"], complex(dec(vals[f]["shunt"])))
    check("T62_fitted", "via-fitted cell: T(6.2 GHz)", feasibility(fit, 3.1e9, N=20)["T_2fs"], 0.134, 3,
          "Results 2 (R5 reference cells, tests/fixtures/r6_delivery_core)")
    check("T106_nominal", "nominal cell, 5.3 GHz signal: T(10.6 GHz)", b["T_2fs"], 0.47, 2, src)
    check("a106_nominal", "nominal cell, 5.3 GHz signal: alpha(10.6 GHz) (Np/cell)", b["alpha_2fs_Np_per_cell"], 1.8, 1, src)


def screen_section():
    from tvtl.optimization.substrate import ROGERS_4003C
    from tvtl.optimization.topology_generator import linear_families
    from tvtl.optimization.stage1_per_cell import coarse_screen
    var = get_entry("SMV1245-079LF").to_realistic_varactor(v_bias_v=0.0)
    factory = candidate_cell_factory(varactor=var, substrate=ROGERS_4003C, f_s=3.1e9, N=15, bias_return_ohm=1e5)

    class Unit:
        def score(self, *a, **k):
            return 1.0
    res = coarse_screen(3.1e9, var, families=linear_families(), n_samples=300, seed=0, gate="stage0",
                        cell_factory=factory, fom=Unit(), gate_options={"N": 15, "T_req": 1e-12, "alpha_max": 1e9})
    rows = [r for r in res.feasibility_records if r.get("T_2fs") is not None]
    T = np.array([r["T_2fs"] for r in rows])
    a = np.array([r["alpha_2fs_Np_per_cell"] for r in rows])
    src = "R1 leg 1 (data deposit runs/r1_redesign_under_law/REPORT.md section 3.1)"
    check("n_sampled", "0 V SMV1245 screen: geometries sampled", len(res.feasibility_records), 3245, None, src)
    check("n_eval", "0 V SMV1245 screen: geometries evaluable", len(rows), 2034, None, src)
    check("AB_adopted", "pass delivery and pump transport at T_req 0.17378", int(np.sum((T >= T_REQ) & (a <= ALPHA_MAX))), 0, None, src)
    check("AB_screen", "pass delivery and pump transport at T_req 0.1438", int(np.sum((T >= T_REQ_SCREEN_2026_09) & (a <= ALPHA_MAX))), 0, None, src)
    m = (T > 0) & (a > 0)
    check("corr", "corr(log T, log alpha)", float(np.corrcoef(np.log(T[m]), np.log(a[m]))[0, 1]), 0.88, 2, src)


def fig7_section(workers):
    with tempfile.TemporaryDirectory() as tmp:
        dense_map.main(["--mode", "gates", "--family", "stub", "--workers", str(workers), "--out-dir", tmp])
        c = dense_map.counts(Path(tmp) / "dense_stub.csv")
    src = "Fig. 7 caption; re-gate at the adopted constants"
    check("fig7_points", "Fig. 7 points (117 f_s x 21 V x 57 s)", c["points"], 140049, None, src)
    check("fig7_admissible", "Fig. 7 admissible at T_req 0.17378, alpha_max 0.02, gate C", c["admissible"], 1776, None, src)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=max(1, min(15, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--out", default=str(REPO / "runs" / "paper_numbers.json"))
    ap.add_argument("--skip-slow", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    info = {"t_req": t_req_section(), "cascade": None, "fig7": None}
    info["cascade"] = cascade_section()
    law_section()
    if not a.skip_slow:
        screen_section()
        info["fig7"] = fig7_section(a.workers)
    w = max(len(r["quantity"]) for r in ROWS)
    print(f"{'quantity':<{w}}  {'value':>22}  {'expected':>12}  result")
    for r in ROWS:
        v = r["value"]
        vs = f"{v:.6g}" if isinstance(v, float) else str(v)
        print(f"{r['quantity']:<{w}}  {vs:>22}  {str(r['expected']):>12}  {r['result']}")
    n_fail = sum(r["result"] == "FAIL" for r in ROWS)
    print(f"\n{len(ROWS)} checks, {n_fail} FAIL, {time.perf_counter() - t0:.0f} s")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checks": ROWS, "details": info, "tier": "ANALYTICAL"}, indent=2, default=str) + "\n",
                   encoding="utf-8")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
