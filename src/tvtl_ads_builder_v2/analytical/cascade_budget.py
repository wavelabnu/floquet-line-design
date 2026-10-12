"""Cascade layer: Bloch phase matching, harmonic coupling and the exceptional point (ANALYTICAL only).

Per-cell budget of a uniform line of ``N`` BranchCells (Results 6 and Fig. 6 of the paper):

``gamma p``      ``arccosh((A + D)/2)`` of the cell, the tool's ``bloch_dispersion`` (numpy principal
                 branch, Re >= 0); ``alpha = Re(gamma p)`` Np/cell, ``theta = Im(gamma p)`` rad/cell.
``Delta_theta``  ``theta(2 f_s) - 2 theta(f_s)`` wrapped to (-pi, pi]; ``N_coh = pi / |Delta_theta|``.
``kappa/m``      small-depth coupling per unit junction depth: the boost of the harmonic cascade
                 (basis (1, 3)) with a uniform prescribed depth m, ideal phase matching
                 ``phi_k = 2 k theta_fwd(f_s)`` and the best of 24 global pump phases, fitted through the
                 origin as ``boost_dB = 8.686 N (kappa/m) m`` on m = 1e-3, 2e-3, 5e-3, 1e-2.
``g``            per-cell growth ``Re sqrt(kappa^2 - (Delta_theta/2)^2) - alpha(f_s)``, kappa = (kappa/m) m.
``m_EP``         ``|Delta_theta| / (2 kappa/m)``: the depth at which the coupling first overcomes the
                 phase mismatch (the exceptional point).
``m_del``        delivered depth: the largest ``(M/V_j) |V_j|`` over the cells, linear pump solve at the
                 board-plane pump power (``delivered_pump_profile``).
``phase_margin`` ``0.5 m_del - m_EP``; negative means the delivered pump cannot reach half the
                 exceptional-point depth.

``theta_fwd`` is the phase of the forward Bloch eigenwave, ``angle(V_{k+1}/V_k)`` (negative in a
passband in the tool's exp(+j omega t) convention); the cascade uses it for the prescribed pump phases,
as the threshold run behind ``T_req`` did. The per-cell budget uses the principal-branch ``theta``.

A sample above a cell's tee-model edge raises ``TeeModelOutOfRange`` (cells built with the default
``tee_out_of_range="raise"``); the gate turns it into the reason ``tee_model_out_of_range``.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .delivered_gain import (
    _db, _passive, _sample, delivered_pump_profile, harmonic_cell, loaded_harmonic_response)

__all__ = [
    "DB_PER_NP", "M_FIT", "PHASE_GRID_DEG",
    "wrap_phase", "bloch_phase", "forward_bloch", "phase_mismatch", "measured_phase_mismatch",
    "prescribed_ab", "passive_db", "amplitude_db", "boost_at", "kappa_per_m", "growth_rate", "m_ep",
    "branch_top_profile", "delivered_depth", "m_star", "cascade_budget",
]

DB_PER_NP = 8.686                       # the published margins and fit use this rounding
M_FIT = (1e-3, 2e-3, 5e-3, 1e-2)
PHASE_GRID_DEG = tuple(range(0, 360, 15))
FIT_TOL = 0.05


def wrap_phase(x: float) -> float:
    """Wrap to (-pi, pi]."""
    return -((-x + math.pi) % (2 * math.pi) - math.pi)


def bloch_phase(cell_fn, f_hz: float):
    """(theta, alpha, half-trace) of one cell, principal branch of arccosh((A + D)/2)."""
    from tvtl.theory.dispersion import bloch_dispersion
    a = cell_fn(f_hz).abcd(frequency_hz=f_hz)
    gp = complex(bloch_dispersion(tuple(a.ravel()), 1.))
    return gp.imag, gp.real, complex((a[0, 0] + a[1, 1]) / 2)


def forward_bloch(cell_fn, f_hz: float):
    """(theta_fwd, alpha) of the forward Bloch eigenwave, V_{k+1}/V_k = exp(-alpha + j theta_fwd)."""
    mu = np.linalg.eigvals(cell_fn(f_hz).abcd(frequency_hz=f_hz))
    r = 1 / mu[int(np.argmax(np.abs(mu)))]
    return float(np.angle(r)), float(-np.log(abs(r)))


def phase_mismatch(cell_fn, f_s: float) -> float:
    """Delta_theta = theta(2 f_s) - 2 theta(f_s), wrapped to (-pi, pi], rad/cell."""
    return wrap_phase(bloch_phase(cell_fn, 2 * f_s)[0] - 2 * bloch_phase(cell_fn, f_s)[0])


def measured_phase_mismatch(f_ghz: Sequence[float], beta_norm: Sequence[float], f_s: float) -> float:
    """Delta_theta from a measured dispersion table, beta_norm = beta p / pi, linear interpolation."""
    f = np.asarray(f_ghz, float)
    b = np.asarray(beta_norm, float)
    fs = f_s / 1e9
    if not (f.min() <= fs and 2 * fs <= f.max()):
        raise ValueError("f_s and 2 f_s must lie inside the measured table")
    return wrap_phase(math.pi * float(np.interp(2 * fs, f, b)) - 2 * math.pi * float(np.interp(fs, f, b)))


def prescribed_ab(fns, f_s: float, m_k, phi_k, *, basis=(1, 3), Z0=50.):
    """(a, b): output at +f_s per unit incident at +f_s and at -f_s, prescribed per-cell (m_k, phi_k).

    The calls of ``delivered_gain`` with the linear pump solve replaced by the prescribed profile.
    """
    samples = [_sample(f, f_s) for f in fns]
    indices = np.r_[-np.array(basis)[::-1], basis]
    blocks = [harmonic_cell(cell_fn=fn, f_s=f_s, indices=indices,
                            coefficients=[s.junction_capacitance_f, .5*s.junction_capacitance_f*mk*np.exp(1j*pk)])
              for fn, s, mk, pk in zip(fns, samples, m_k, phi_k)]
    trans, _ = loaded_harmonic_response(cells=blocks, Z0=Z0)
    pos, neg = indices.tolist().index(1), indices.tolist().index(-1)
    return trans[pos, pos], trans[pos, neg]


def passive_db(fns, f_s: float, Z0=50.) -> float:
    """Pump-off |S21| in dB."""
    samples = [_sample(f, f_s) for f in fns]
    _, passive, _ = _passive(samples, f_s, 1.0, Z0)
    return _db(passive[-1])


def amplitude_db(a, b, phase) -> float:
    """Absolute out/in in dB at a global pump phase (radians) or 'best'."""
    return _db(abs(a) + abs(b)) if phase == "best" else _db(abs(a + b * np.exp(1j * phase)))


def boost_at(cell_fn, f_s: float, m: float, dphi: float, pdb: float, *, N=20, basis=(1, 3), Z0=50.,
             phases_deg=PHASE_GRID_DEG) -> dict:
    """On/off boost for a uniform depth m and phi_k = k dphi: best over the phase grid and exact best."""
    fns = [cell_fn] * N
    a, b = prescribed_ab(fns, f_s, [m] * N, [k * dphi for k in range(N)], basis=basis, Z0=Z0)
    amps = [amplitude_db(a, b, float(np.deg2rad(d))) for d in phases_deg]
    j = int(np.argmax(amps))
    best = amplitude_db(a, b, "best")
    return {"boost": amps[j] - pdb, "outin": amps[j], "phi_deg": phases_deg[j],
            "boost_exact": best - pdb, "outin_exact": best}


def kappa_per_m(cell_fn, f_s: float, *, N=20, m_fit=M_FIT, basis=(1, 3), Z0=50.) -> dict:
    """Small-depth kappa/m (per cell, per unit depth) from the ideal-phase-matched cascade."""
    dphi = 2 * forward_bloch(cell_fn, f_s)[0]
    pdb = passive_db([cell_fn] * N, f_s, Z0)
    b = [boost_at(cell_fn, f_s, m, dphi, pdb, N=N, basis=basis, Z0=Z0)["boost"] for m in m_fit]
    ms = np.asarray(m_fit, float)
    slope = float(np.dot(b, ms) / np.dot(ms, ms))
    resid = max(abs(bi - slope * mi) / abs(bi) for bi, mi in zip(b, m_fit))
    flag = resid > FIT_TOL
    k = (b[0] / m_fit[0] if flag else slope) / (DB_PER_NP * N)
    return {"kappa_per_m": k, "boost_dB": b, "slope_dB_per_m": slope, "fit_residual_max_rel": resid,
            "fit_flag": flag, "source": "m=1e-3 slope" if flag else "fit"}


def growth_rate(kappa: float, delta_theta: float, alpha_s: float) -> float:
    """g = Re sqrt(kappa^2 - (Delta_theta/2)^2) - alpha(f_s), Np/cell."""
    return float(np.sqrt(complex(kappa ** 2 - (delta_theta / 2) ** 2)).real) - alpha_s


def m_ep(delta_theta: float, kappa_per_unit_m: float) -> float:
    """Exceptional-point depth |Delta_theta| / (2 kappa/m)."""
    return abs(delta_theta) / (2 * kappa_per_unit_m)


def branch_top_profile(fns, f_p: float, P_dBm: float, Z0=50.) -> np.ndarray:
    """Branch-top pump phasors of the linear pump solve: junction phasors over the branch divider."""
    vj = delivered_pump_profile(cell_fn=fns, N=len(fns), f_p=f_p, P_inc_dBm=P_dBm, Z0=Z0)
    out = []
    for fn, v in zip(fns, vj):
        s = fn(f_p)
        zc = 1 / (2j * np.pi * f_p * s.junction_capacitance_f)
        out.append(v / (zc / (s.series_impedance_ohm + zc)))
    return np.asarray(out)


def delivered_depth(cell_fn, f_s: float, *, N=20, P_pump_dBm=19.5, Z0=50.) -> dict:
    """m_del: largest (M/V_j)|V_j| over the N cells, linear pump solve at the board plane."""
    fns = [cell_fn] * N
    vj = delivered_pump_profile(cell_fn=fns, N=N, f_p=2 * f_s, P_inc_dBm=P_pump_dBm, Z0=Z0)
    s = cell_fn(2 * f_s)
    m = s.M / s.Vj * np.abs(vj)
    return {"m_del": float(m.max()), "m_del_cell": int(m.argmax()) + 1, "k_per_V": s.M / s.Vj}


def m_star(cell_fn, f_s: float, *, N=20, target_dB=1.0, mode="ideal", ladder=None, tol=1e-4,
           basis=(1, 3), Z0=50.) -> dict:
    """Smallest uniform depth with an on/off boost >= target_dB (best of the 15-degree phase grid).

    ``mode`` "ideal": phi_k = 2 k theta_fwd(f_s); "phys": phi_k = k theta_fwd(2 f_s). Ladder 0-0.60 in
    steps of 0.02, then bisection to ``tol``; returns the upper bracket, None when unreached.
    """
    ladder = tuple(round(0.02 * j, 2) for j in range(31)) if ladder is None else tuple(ladder)
    th_s = forward_bloch(cell_fn, f_s)[0]
    dphi = 2 * th_s if mode == "ideal" else forward_bloch(cell_fn, 2 * f_s)[0]
    pdb = passive_db([cell_fn] * N, f_s, Z0)
    vals = [boost_at(cell_fn, f_s, m, dphi, pdb, N=N, basis=basis, Z0=Z0)["boost"] for m in ladder]
    cross = next((j for j, v in enumerate(vals) if v >= target_dB), None)
    if cross is None:
        return {"m_star": None, "reached": False, "passive_S21_dB": pdb}
    if cross == 0:
        raise ValueError("boost already at target with zero depth")
    lo, hi = ladder[cross - 1], ladder[cross]
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if boost_at(cell_fn, f_s, mid, dphi, pdb, N=N, basis=basis, Z0=Z0)["boost"] >= target_dB:
            hi = mid
        else:
            lo = mid
    return {"m_star": hi, "bracket_lo": lo, "reached": True, "passive_S21_dB": pdb}


def cascade_budget(cell_fn, f_s: float, *, N=20, P_pump_dBm=19.5, Z0=50., measured=None) -> dict:
    """Per-cell budget at f_s: phases, Delta_theta, kappa/m, m_del, g, m_EP and the phase margin.

    ``measured`` = (f_GHz, beta_norm) of a measured dispersion table adds ``delta_theta_meas`` and
    ``m_EP_meas`` (model kappa/m with the measured phase mismatch).
    """
    th_s, a_s, _ = bloch_phase(cell_fn, f_s)
    th_p, a_p, _ = bloch_phase(cell_fn, 2 * f_s)
    dth = wrap_phase(th_p - 2 * th_s)
    kf = kappa_per_m(cell_fn, f_s, N=N, Z0=Z0)
    md = delivered_depth(cell_fn, f_s, N=N, P_pump_dBm=P_pump_dBm, Z0=Z0)
    kpm = kf["kappa_per_m"]
    out = {"f_s_Hz": float(f_s), "N": int(N), "P_pump_dBm": float(P_pump_dBm),
           "theta_s_rad": th_s, "theta_p_rad": th_p, "alpha_s_Np": a_s, "alpha_p_Np": a_p,
           "delta_theta_rad": dth, "delta_theta_deg": math.degrees(dth),
           "N_coh": math.pi / abs(dth) if dth else math.inf,
           "kappa_per_m": kpm, "kappa_fit_residual": kf["fit_residual_max_rel"], "kappa_fit_flag": kf["fit_flag"],
           "m_del": md["m_del"], "m_del_cell": md["m_del_cell"],
           "g_m_del": growth_rate(kpm * md["m_del"], dth, a_s),
           "m_EP": m_ep(dth, kpm), "tier": "ANALYTICAL"}
    out["phase_margin"] = 0.5 * out["m_del"] - out["m_EP"]
    if measured is not None:
        dm = measured_phase_mismatch(measured[0], measured[1], f_s)
        out.update(delta_theta_meas_rad=dm, delta_theta_meas_deg=math.degrees(dm), m_EP_meas=m_ep(dm, kpm))
    return out
