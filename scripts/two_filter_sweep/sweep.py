"""Two-filter feasibility sweep core (TASK 1 / TASK 2).

At every ``(s, f_p)`` on the grid:

    T(f_p)          branch junction-delivery transfer  (condition A)
    |(A+D)/2|(f_p)  pump-side Bloch half-trace         (condition B)
    |(A+D)/2|(f_s)  signal-side Bloch half-trace, f_s = f_p / 2

    feasible = (T >= T_req) and (|(A+D)/2|(f_p) <= 1)
    usable   = feasible and (|(A+D)/2|(f_s) <= 1)

``s`` scales all three stub section LENGTHS; widths, main line, period,
substrate, package and junction stay at the fabricated-board values.

The stub admittance is the additive mid-shunt analytical model
(``build_stub_y_fn_with_mid_shunts``) -- the ADS Touchstone table only exists at
s = 1.0, so a sweep over geometry has no choice but to use the analytical stub.
The stub-fidelity gates of the deposited sweep record (GATE 0 / GATE 0b) quantify
what that costs.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np

from _common import (
    C0_F_NOMINAL, OPEN_END, PERIOD_M, SEC_WIDTHS_M, STUB_PARAMS_M, SUBSTRATE_CFG,
    T_REQ, VIA_BOARD, CellEvaluator, _tvtl_substrate, branch_T,
)
from tvtl.microstrip.models import hammerstad_kirschning
from tvtl_ads_builder_v2.analytical.floquet_stub import (
    StubSegment, Y_stub_at_freq_with_shunts,
)
from tvtl.microstrip.discontinuities import open_end_capacitance_f

HERE = Path(__file__).resolve().parent

# ---- grids (task-specified) ----------------------------------------------- #
S_LO, S_HI, S_STEP = 0.4, 3.0, 0.02                 # 131 points
FP_LO_GHZ, FP_HI_GHZ = 2.0, 16.0
FP_STEP_GHZ_DEFAULT = 0.01                          # 10 MHz

BASE_SEGMENTS_MM = ((1.93, 6.83), (1.41, 5.25), (1.98, 1.04))
MAIN_W_M = 4.27e-3

_SUB = _tvtl_substrate()
_SUB_DICT = {"eps_r": SUBSTRATE_CFG.eps_r, "h_m": SUBSTRATE_CFG.h_m,
             "t_m": SUBSTRATE_CFG.t_m}


def s_grid() -> np.ndarray:
    return np.round(S_LO + S_STEP * np.arange(
        int(round((S_HI - S_LO) / S_STEP)) + 1), 6)


def fp_grid(step_ghz: float = FP_STEP_GHZ_DEFAULT) -> np.ndarray:
    n = int(round((FP_HI_GHZ - FP_LO_GHZ) / step_ghz))
    return np.round(FP_LO_GHZ + step_ghz * np.arange(n + 1), 6)


# ---- the analytical stub at scale ``s`` ----------------------------------- #
def _mlef_termination(width_m: float):
    """Frequency-tracking Kirschning-Franz open-end fringe (MlefOpen physics).

    ``MlefOpen.analytical_termination`` freezes ``C_end`` at one reference
    frequency; over a 2-16 GHz sweep the fringe is re-evaluated at each point
    instead (same formula, same width/substrate). At 3.1 GHz the two agree by
    construction.
    """

    def _z(omega: float) -> complex:
        w = float(omega)
        if w == 0.0:
            return complex("inf")
        f = abs(w) / (2.0 * math.pi)
        z0_a, ee_a = hammerstad_kirschning(
            float(width_m), SUBSTRATE_CFG.t_m, SUBSTRATE_CFG.h_m,
            SUBSTRATE_CFG.eps_r, [f])
        c_end = float(open_end_capacitance_f(
            float(width_m), float(z0_a[0]), float(ee_a[0]), _SUB))
        if c_end <= 0.0:
            return complex("inf")
        return 1.0 / complex(0.0, w * c_end)

    return _z


_TERM_MLEF = _mlef_termination(BASE_SEGMENTS_MM[2][0] * 1e-3)


def stub_y_fn_for_s(s: float, l_eff_h: float):
    """``omega -> Y_stub`` for the board stub with all three lengths scaled by s."""
    segs = tuple(StubSegment(width_m=w * 1e-3, length_m=s * l * 1e-3)
                 for (w, l) in BASE_SEGMENTS_MM)
    via = dataclasses.replace(VIA_BOARD, effective_L_h=float(l_eff_h))
    z_via = via.analytical_termination(substrate=_SUB)
    shunts = {1: z_via, 2: z_via}

    def _y(omega: float) -> complex:
        return Y_stub_at_freq_with_shunts(
            omega, segs, _SUB_DICT, termination=_TERM_MLEF,
            mid_shunt_loads=shunts)

    return _y


# ---- the sweep ------------------------------------------------------------ #
def run_sweep(*, l_eff_h: float, c0_f: float, label: str,
              fp_step_ghz: float = FP_STEP_GHZ_DEFAULT,
              progress_every: int = 10) -> dict:
    """Full (s, f_p) sweep; returns the grids + boolean/scalar fields."""
    ss = s_grid()
    fps = fp_grid(fp_step_ghz)
    fss = np.round(fps / 2.0, 6)
    all_f = np.unique(np.concatenate([fps, fss]))
    idx_fp = np.searchsorted(all_f, fps)
    idx_fs = np.searchsorted(all_f, fss)
    assert np.all(all_f[idx_fp] == fps) and np.all(all_f[idx_fs] == fss)
    all_f_hz = all_f * 1e9

    ev = CellEvaluator(None, c0_override_f=None if c0_f == C0_F_NOMINAL else c0_f)
    n_s, n_f = ss.size, fps.size

    T = np.empty((n_s, n_f))
    half_p = np.empty((n_s, n_f))
    half_s = np.empty((n_s, n_f))
    alpha_p = np.empty((n_s, n_f))

    t0 = time.perf_counter()
    for i, s in enumerate(ss):
        y_fn = stub_y_fn_for_s(float(s), l_eff_h)
        ev.stub_y_fn = y_fn
        half_all = np.empty(all_f.size, dtype=complex)
        alpha_all = np.empty(all_f.size)
        for k, fhz in enumerate(all_f_hz):
            _b, a_np, h = ev.dispersion(fhz)
            half_all[k] = h
            alpha_all[k] = a_np
        half_p[i] = np.abs(half_all[idx_fp])
        half_s[i] = np.abs(half_all[idx_fs])
        alpha_p[i] = alpha_all[idx_fp]
        T[i] = [branch_T(f * 1e9, y_fn, c0_f=c0_f) for f in fps]
        if progress_every and (i % progress_every == 0 or i == n_s - 1):
            el = time.perf_counter() - t0
            print("  [%s] s=%.2f  %3d/%3d  %6.1f s elapsed  (eta %6.1f s)"
                  % (label, s, i + 1, n_s, el, el / (i + 1) * (n_s - i - 1)),
                  flush=True)

    feasible = (T >= T_REQ) & (half_p <= 1.0)
    usable = feasible & (half_s <= 1.0)
    return {"label": label, "l_eff_h": l_eff_h, "c0_f": c0_f,
            "s": ss, "fp_GHz": fps, "T": T, "half_p": half_p, "half_s": half_s,
            "alpha_p": alpha_p, "feasible": feasible, "usable": usable,
            "fp_step_GHz": fp_step_ghz,
            "runtime_s": time.perf_counter() - t0}


# ---- window extraction ---------------------------------------------------- #
def windows_from(res: dict) -> list[dict]:
    """Contiguous runs of ``usable`` along f_p, one record per (s, window).

    Window edges are the first/last usable grid point, so a single-sample
    window has zero absolute bandwidth (``n_points`` is recorded alongside).
    """
    ss, fps = res["s"], res["fp_GHz"]
    out: list[dict] = []
    for i, s in enumerate(ss):
        u = res["usable"][i]
        if not u.any():
            continue
        d = np.diff(u.astype(np.int8))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0])
        if u[0]:
            starts = [0] + starts
        if u[-1]:
            ends = ends + [u.size - 1]
        for a, b in zip(starts, ends):
            f_lo, f_hi = float(fps[a]), float(fps[b])
            f_ctr = 0.5 * (f_lo + f_hi)
            j = int(round((a + b) / 2.0))
            out.append({
                "s": float(s), "f_lo_GHz": f_lo, "f_hi_GHz": f_hi,
                "abs_bw_GHz": f_hi - f_lo,
                "rel_bw_pct": 100.0 * (f_hi - f_lo) / f_ctr,
                "f_center_GHz": f_ctr, "n_points": int(b - a + 1),
                "alpha_at_center": float(res["alpha_p"][i, j]),
                "T_at_center": float(res["T"][i, j]),
                "half_p_at_center": float(res["half_p"][i, j]),
                "half_s_at_center": float(res["half_s"][i, j]),
                "f_s_GHz": f_ctr / 2.0,
            })
    return out


# ---- TASK-2 closed form --------------------------------------------------- #
def difficulty_law_fraction(c0_f: float, f_hz, z0_ohm: float,
                            t_req: float = T_REQ):
    """``frac(theta)`` admissible fraction from the closed form in ``task2.py``."""
    w = 2.0 * math.pi * np.asarray(f_hz, dtype=float)
    u_min = z0_ohm * t_req * w * float(c0_f) / 2.0
    return (math.pi - 2.0 * np.arccos(1.0 / np.sqrt(1.0 + u_min ** 2))) / math.pi


def main_line_z0(f_hz: float) -> float:
    z0, _ = hammerstad_kirschning(MAIN_W_M, SUBSTRATE_CFG.t_m, SUBSTRATE_CFG.h_m,
                                  SUBSTRATE_CFG.eps_r, [float(f_hz)])
    return float(z0[0])


def save(res: dict, path: Path) -> None:
    np.savez_compressed(
        path, s=res["s"], fp_GHz=res["fp_GHz"], T=res["T"],
        half_p=res["half_p"], half_s=res["half_s"], alpha_p=res["alpha_p"],
        feasible=res["feasible"], usable=res["usable"])
