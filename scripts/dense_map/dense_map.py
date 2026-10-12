"""Dense analytical feasibility map of the SMV1245 line (the inputs of Fig. 7 of the paper).

Tier ANALYTICAL. No simulator, no harmonic balance, no matching. The model is the R6 Stage-4 one:
``make_cell`` is the deposited ``runs/r6_delivery_core/stage4_map.py`` (lines 22-36) with one change,
the explicit ``tee_out_of_range="reduced"`` (see below). Its stub is the two-filter-sweep analytical
stub with the board's two mid-stub vias (``scripts/two_filter_sweep``,
``Y_stub_at_freq_with_shunts``).

Outputs (in ``--out-dir``, default ``runs/dense_map``):

  dense_stub.csv    140,049 points: 117 f_s (0.20-6.00 GHz) x 21 V (0-5 V) x 57 stub scales (0.4-6.0)
  dense_plain.csv   39,312 points of the plain-shunt family (lumped via 0.3-1.0 nH, 0 or 0.3 ohm)
  proxy.csv         the delivered-modulation proxy at +10 dBm on the points with ``admissible`` True
  tee_model.json    which harmonic samples of the stub cell lie above its tee-model edge (11.64 GHz)

The three CSVs reproduce byte-for-byte the R6c tables behind Fig. 7 (SHA-256 in ``REFERENCE_SHA256``;
``--mode check``). Two columns keep the conventions of those tables:

* ``feasible_tool`` / ``admissible`` are the 2026-09 tool verdict: delivery at ``T_REQ_SCREEN_2026_09``
  (0.1438), pump transport and the signal passband flag; gate D (signal transport) is not part of
  them, and ``M_A_dB`` is relative to 0.1438. Fig. 7 does not read these two columns: it re-gates
  ``T_2fs``, ``alpha_2fs``, ``alpha_fs`` and ``footprint_mm2`` at the adopted constants
  (``--mode counts``: 140,049 points, 1,776 admissible at T_req 0.17378, alpha_max 0.02, gate C).
* Samples above the tee-model edge use the reduced tee model (no tee correction), as the R6c tables
  did: f_s >= 5.85 GHz for the pump columns and, in the proxy, the 3/5/7 f_s harmonics. The release
  default refuses such samples (``tee_model_out_of_range``); this script requests the reduced model
  explicitly and lists the affected frequencies in ``tee_model.json``.

    python scripts/dense_map/dense_map.py --mode gates --family stub --workers 15
    python scripts/dense_map/dense_map.py --mode gates --family plain --workers 15
    python scripts/dense_map/dense_map.py --mode proxy --workers 15
    python scripts/dense_map/dense_map.py --mode tee
    python scripts/dense_map/dense_map.py --mode counts
    python scripts/dense_map/dense_map.py --mode check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (HERE.parent / "two_filter_sweep", REPO / "examples" / "real_parts" / "boundary",
           REPO / "examples" / "real_parts", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                                                # noqa: E402

import _common as legacy                                                          # noqa: E402
import sweep                                                                      # noqa: E402
from tvtl.varactor_palette import get_entry                                       # noqa: E402
from tvtl.materials import Substrate                                              # noqa: E402
from tvtl_ads_builder_v2.analytical.delivered_gain import (                       # noqa: E402
    BranchCell, cell_functions_from_spec, delivered_gain)
from tvtl_ads_builder_v2.analytical.level10_sparams import _line_abcd_lossy       # noqa: E402
from tvtl.design.feasibility import ALPHA_MAX, T_REQ, T_REQ_SCREEN_2026_09, feasibility  # noqa: E402
from tvtl.optimization.stage2_main_tl import microstrip_width_for_z0              # noqa: E402
from tvtl.optimization.substrate import ROGERS_4003C                              # noqa: E402

# --- frozen inputs ---------------------------------------------------------------------------------
T_REQ_TABLE, N_CELLS = T_REQ_SCREEN_2026_09, 20     # the tables' tool verdict and M_A_dB use 0.1438
DB_NP = 8.686                         # dB per neper, as in the published margins
SLOT_MM2 = 6.11 * (30 - 4.27) / 2     # K7 routing slot (period x half-clearance), 78.605 mm^2
PUMP_PROXY_DBM = 10
HS_LIMIT_DB = 1.0
V_MAX = 5.0                           # min(5 V, 12 V table range of the SMV1245 card)
FS_GRID = [round(0.20 + 0.05 * k, 2) for k in range(117)]
V_GRID = [round(0.25 * k, 2) for k in range(int(round(V_MAX / 0.25)) + 1)]
S_GRID = [round(0.4 + 0.1 * k, 1) for k in range(57)]
L_GRID = [round(0.3 + 0.1 * k, 1) for k in range(8)]
R_GRID = [0.0, 0.3]
REFERENCE_SHA256 = {                  # the R6c tables read by the Fig. 7 build (LF line endings)
    "dense_stub.csv": "1bea73b89094b273e075d09d4ba8097595150734083d4b34a77ba8f4e8bf2c5f",
    "dense_plain.csv": "c4757bafab33e8552e490377312d1e91800e0c2cac32813e68bb37e070c3329b",
    "proxy.csv": "f1856d1914c61add1465342bbef0f02f64c19e73eda436e1838f9f5aca4fb6fb",
}
OUT = REPO / "runs" / "dense_map"


# --- R6 model: runs/r6_delivery_core/stage4_map.py:22-36, plus tee_out_of_range="reduced" -----------
def make_cell(family,bias,scale=1,via_nH=.3,via_R=0):
    var=get_entry('SMV1245-079LF').to_realistic_varactor(v_bias_v=bias)
    if family=='stub':
        y=sweep.stub_y_fn_for_s(scale,.1949e-9)
        spec=legacy.spec_smv1245(n_cells=1,v_bias_v=bias)
        return cell_functions_from_spec(spec=spec,varactor=var,stub_y_fn=y,tee_out_of_range="reduced")[0]
    sub=Substrate(3.38,1.54e-3,17.5e-6,.0027,5.8e7)
    width=microstrip_width_for_z0(50,ROGERS_4003C,3.1e9)
    @lru_cache(maxsize=512)
    def cell(f):
        line=_line_abcd_lossy(width,6.11e-3/2,f,sub)
        c=var.C0_f;cj=var.CJ0_f/(1+bias/var.VJ_v)**var.M
        return BranchCell(line,line,var.R_series_ohm+via_R+2j*np.pi*f*(var.L_pkg_h+via_nH*1e-9),c,var.M*cj/c,var.VJ_v+bias)
    return cell


def asbuilt_cell(tee_out_of_range="raise"):
    """The as-built SMV1245 cell (stub family, 0 V, scale 1) with the release's tee policy (default refuse)."""
    var = get_entry('SMV1245-079LF').to_realistic_varactor(v_bias_v=0.0)
    return cell_functions_from_spec(spec=legacy.spec_smv1245(n_cells=1, v_bias_v=0.0), varactor=var,
                                    stub_y_fn=sweep.stub_y_fn_for_s(1.0, .1949e-9),
                                    tee_out_of_range=tee_out_of_range)[0]


def footprint_mm2(s: float) -> float:
    """R6 stage4_map.py:45 / R6b update_map.py:11."""
    return 2*((1.93+1.41+1.98)/3)*(6.83+5.25+1.04)*s


def loaded_z(fn, fs_hz: float) -> complex:
    """R6 stage4_map.py:47."""
    a = fn(fs_hz).abcd(frequency_hz=fs_hz)
    return complex(np.sqrt(complex(a[0, 1] / a[1, 0])))


def c0_pf(bias: float) -> float:
    return float(get_entry('SMV1245-079LF').to_realistic_varactor(v_bias_v=bias).C0_f) * 1e12


# --- one point -------------------------------------------------------------------------------------
GATE_COLS = ["family", "fs_GHz", "fp_GHz", "V", "s", "L_via_nH", "R_via_ohm", "C0_pF", "T_2fs",
             "alpha_2fs", "alpha_fs", "passband_fs", "footprint_mm2", "gate_A", "gate_B",
             "gate_D_tool", "gate_C", "feasible_tool", "admissible", "M_A_dB", "M_B_dB", "M_D_dB",
             "margin_dB", "binding"]


def gate_row(fn, family: str, fs: float, V: float, s, L, R, c0: float) -> dict:
    g = feasibility(fn, fs * 1e9, T_req=T_REQ_TABLE, N=N_CELLS)
    T, a2, a1 = float(g["T_2fs"]), float(g["alpha_2fs_Np_per_cell"]), float(g["alpha_fs"])
    mA = 20 * math.log10(T / T_REQ_TABLE) if T > 0 else -math.inf
    mB = DB_NP * N_CELLS * (ALPHA_MAX - a2)
    mD = DB_NP * N_CELLS * (ALPHA_MAX - a1)
    margins = {"A": mA, "B": mB, "D": mD}
    binding = min(margins, key=lambda k: (margins[k], "ABD".index(k)))
    area = footprint_mm2(s) if family == "stub" else None
    gate_c = (area <= SLOT_MM2) if family == "stub" else None
    # 2026-09 tool verdict: the gate's reasons without gate D
    feas = not [r for r in g["reasons"] if r != "signal_transport_above_budget"]
    return {"family": family, "fs_GHz": fs, "fp_GHz": round(2 * fs, 2), "V": V, "s": s,
            "L_via_nH": L, "R_via_ohm": R, "C0_pF": c0, "T_2fs": T, "alpha_2fs": a2,
            "alpha_fs": a1, "passband_fs": bool(g["in_band_s"]), "footprint_mm2": area,
            "gate_A": T >= T_REQ_TABLE, "gate_B": a2 <= ALPHA_MAX, "gate_D_tool": bool(g["in_band_s"]),
            "gate_C": gate_c, "feasible_tool": feas,
            "admissible": feas and (gate_c if family == "stub" else True),
            "M_A_dB": mA, "M_B_dB": mB, "M_D_dB": mD, "margin_dB": margins[binding],
            "binding": binding}


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return repr(v)
    return str(v)


# --- chunks ----------------------------------------------------------------------------------------
def chunks(family: str) -> list:
    if family == "stub":
        return [("stub", V, s, None, None) for V in V_GRID for s in S_GRID]
    return [("plain", V, None, L, R) for V in V_GRID for L in L_GRID for R in R_GRID]


def chunk_name(c) -> str:
    fam, V, s, L, R = c
    return (f"{fam}_V{V:.2f}_s{s:.1f}" if fam == "stub" else f"{fam}_V{V:.2f}_L{L:.1f}_R{R:.1f}")


def run_chunk(args):
    c, parts = args
    out = Path(parts) / (chunk_name(c) + ".csv")
    if out.is_file():
        return out.name, 0, 0.0
    fam, V, s, L, R = c
    t0 = time.perf_counter()
    fn = make_cell(fam, V, s, L, R) if fam == "stub" else make_cell(fam, V, 1, L, R)
    c0 = c0_pf(V)
    rows = [gate_row(fn, fam, fs, V, s, L, R, c0) for fs in FS_GRID]
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        for r in rows:
            w.writerow([fmt(r[k]) for k in GATE_COLS])
    os.replace(tmp, out)
    return out.name, len(rows), time.perf_counter() - t0


def assemble(family: str, parts: Path, out: Path) -> int:
    lines = []
    for c in chunks(family):
        lines.extend((parts / (chunk_name(c) + ".csv")).read_text(encoding="utf-8").splitlines())
    key = (lambda ln: (float(ln.split(",")[3]), float(ln.split(",")[4]), float(ln.split(",")[1]))) \
        if family == "stub" else \
        (lambda ln: (float(ln.split(",")[3]), float(ln.split(",")[5]), float(ln.split(",")[6]),
                     float(ln.split(",")[1])))
    lines.sort(key=key)
    out.write_text(",".join(GATE_COLS) + "\n" + "\n".join(lines) + "\n", encoding="utf-8",
                   newline="\n")
    return len(lines)


def _pool(workers):
    from multiprocessing import get_context
    return get_context("spawn").Pool(workers)


def mode_gates(a) -> None:
    out = Path(a.out_dir) / f"dense_{a.family}.csv"
    parts = Path(a.out_dir) / "_work" / f"parts_{out.stem}"       # one chunk store per output file
    parts.mkdir(parents=True, exist_ok=True)
    todo = chunks(a.family)
    t0 = time.perf_counter()
    jobs = [(c, str(parts)) for c in todo]
    if a.workers <= 1:
        for j in jobs:
            run_chunk(j)
    else:
        with _pool(a.workers) as pool:
            for done, _ in enumerate(pool.imap_unordered(run_chunk, jobs), 1):
                if done % 100 == 0:
                    print(f"{done}/{len(todo)} chunks, {time.perf_counter() - t0:.0f} s", flush=True)
    n = assemble(a.family, parts, out)
    print(json.dumps({"family": a.family, "points": n, "chunks": len(todo), "workers": a.workers,
                      "wall_s": round(time.perf_counter() - t0, 1), "out": str(out)}))


# --- proxy ----------------------------------------------------------------------------------------
PROXY_COLS = ["family", "fs_GHz", "V", "s", "L_via_nH", "R_via_ohm", "pump_dBm", "onoff_dB",
              "absolute_outin_dB", "harmonic_sensitivity_dB", "max_m", "largem",
              "loaded_Z_real_ohm", "loaded_Z_imag_ohm"]


def admissible_keys(out_dir) -> list:
    keys = []
    for fam in ("stub", "plain"):
        p = Path(out_dir) / f"dense_{fam}.csv"
        for r in csv.DictReader(p.open(encoding="utf-8")):
            if r["admissible"] == "True":
                keys.append((fam, float(r["fs_GHz"]), float(r["V"]),
                             float(r["s"]) if r["s"] else None,
                             float(r["L_via_nH"]) if r["L_via_nH"] else None,
                             float(r["R_via_ohm"]) if r["R_via_ohm"] else None))
    return keys


def proxy_chunk(args):
    group, parts = args
    fam, V, s, L, R = group[0][0], group[0][2], group[0][3], group[0][4], group[0][5]
    out = Path(parts) / (chunk_name((fam, V, s, L, R)) + ".csv")
    if out.is_file():
        return out.name, 0
    fn = make_cell(fam, V, s, L, R) if fam == "stub" else make_cell(fam, V, 1, L, R)
    lines = []
    for (_, fs, _, _, _, _) in group:
        dg = delivered_gain(cell_fn=fn, N=N_CELLS, f_s=fs * 1e9, P_pump_dBm=PUMP_PROXY_DBM, P_sig_dBm=-56)
        z = loaded_z(fn, fs * 1e9)
        hs = float(dg["harmonic_sensitivity_dB"])
        r = {"family": fam, "fs_GHz": fs, "V": V, "s": s, "L_via_nH": L, "R_via_ohm": R,
             "pump_dBm": PUMP_PROXY_DBM, "onoff_dB": float(dg["onoff_dB"]),
             "absolute_outin_dB": float(dg["absolute_outin_dB"]), "harmonic_sensitivity_dB": hs,
             "max_m": float(dg["max_m"]), "largem": abs(hs) > HS_LIMIT_DB,
             "loaded_Z_real_ohm": z.real, "loaded_Z_imag_ohm": z.imag}
        lines.append(",".join(fmt(r[c]) for c in PROXY_COLS))
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, out)
    return out.name, len(lines)


def grouped(keys):
    g = {}
    for k in keys:
        g.setdefault((k[0], k[2], k[3], k[4], k[5]), []).append(k)
    return [g[k] for k in sorted(g, key=lambda t: (t[0], t[1], t[2] or 0, t[3] or 0, t[4] or 0))]


def mode_proxy(a) -> None:
    keys = admissible_keys(a.out_dir)
    out = Path(a.out_dir) / "proxy.csv"
    parts = Path(a.out_dir) / "_work" / "parts_proxy"
    parts.mkdir(parents=True, exist_ok=True)
    groups = grouped(keys)
    t0 = time.perf_counter()
    jobs = [(g, str(parts)) for g in groups]
    if a.workers <= 1:
        for j in jobs:
            proxy_chunk(j)
    else:
        with _pool(a.workers) as pool:
            for _ in pool.imap_unordered(proxy_chunk, jobs):
                pass
    lines = []
    for g in groups:
        k = g[0]
        lines.extend((parts / (chunk_name((k[0], k[2], k[3], k[4], k[5])) + ".csv"))
                     .read_text(encoding="utf-8").splitlines())
    order = {(k[0], k[1], k[2], k[3], k[4], k[5]): i for i, k in enumerate(keys)}

    def key(ln):
        v = ln.split(",")
        return order[(v[0], float(v[1]), float(v[2]), float(v[3]) if v[3] else None,
                      float(v[4]) if v[4] else None, float(v[5]) if v[5] else None)]
    lines.sort(key=key)
    out.write_text(",".join(PROXY_COLS) + "\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"points": len(lines), "groups": len(groups), "workers": a.workers,
                      "wall_s": round(time.perf_counter() - t0, 1), "out": str(out)}))


# --- tee-model edge, counts at the adopted constants, check -----------------------------------------
def tee_table() -> dict:
    """Per f_s of the grid: is the stub cell's tee model valid at k f_s (k = 1, 2, 3, 5, 7)?"""
    cell = asbuilt_cell("raise")
    rows = {}
    for fs in FS_GRID:
        flags = {}
        for k in (1, 2, 3, 5, 7):
            try:
                cell(k * fs * 1e9)
                flags[f"{k}fs"] = True
            except ValueError as exc:
                if getattr(exc, "reason", None) != "tee_model_out_of_range":
                    raise
                flags[f"{k}fs"] = False
        rows[repr(fs)] = flags
    first = {k: next((fs for fs in FS_GRID if not rows[repr(fs)][k]), None) for k in ("1fs", "2fs", "3fs", "5fs", "7fs")}
    return {"note": "the stub cell's tee geometry does not depend on V or s; reduced tee model wherever False",
            "first_fs_GHz_with_reduced_tee": first, "by_fs_GHz": rows}


def mode_tee(a) -> None:
    t = tee_table()
    out = Path(a.out_dir) / "tee_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(t, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(t["first_fs_GHz_with_reduced_tee"]))


def counts(stub_csv, t_req=T_REQ, alpha_max=ALPHA_MAX) -> dict:
    """Fig. 7 counts: law margin >= 0 (delivery at t_req, pump and signal transport) and gate C."""
    edge = tee_table()["first_fs_GHz_with_reduced_tee"]["2fs"]
    n = law = adm = adm_reduced_tee = 0
    for r in csv.DictReader(Path(stub_csv).open(encoding="utf-8")):
        n += 1
        T, a2, a1 = float(r["T_2fs"]), float(r["alpha_2fs"]), float(r["alpha_fs"])
        mA = 20 * math.log10(T / t_req) if T > 0 else -math.inf
        if min(mA, DB_NP * N_CELLS * (alpha_max - a2), DB_NP * N_CELLS * (alpha_max - a1)) < 0:
            continue
        law += 1
        if float(r["footprint_mm2"]) <= SLOT_MM2:
            adm += 1
            adm_reduced_tee += edge is not None and float(r["fs_GHz"]) >= edge
    return {"points": n, "law": law, "admissible": adm, "admissible_with_reduced_tee_pump": adm_reduced_tee,
            "reduced_tee_pump_from_fs_GHz": edge, "T_req": t_req, "alpha_max": alpha_max,
            "slot_mm2": SLOT_MM2}


def mode_counts(a) -> None:
    print(json.dumps(counts(Path(a.out_dir) / "dense_stub.csv"), indent=1))


def mode_check(a) -> None:
    res = {}
    for name, ref in REFERENCE_SHA256.items():
        p = Path(a.out_dir) / name
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        res[name] = {"sha256": got, "reference": ref, "identical": got == ref}
    print(json.dumps(res, indent=1))
    if not all(v["identical"] for v in res.values()):
        raise SystemExit(1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="gates", choices=["gates", "proxy", "tee", "counts", "check"])
    ap.add_argument("--family", choices=["stub", "plain"], default="stub")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    a = ap.parse_args(argv)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    {"gates": mode_gates, "proxy": mode_proxy, "tee": mode_tee, "counts": mode_counts,
     "check": mode_check}[a.mode](a)


if __name__ == "__main__":
    main()
