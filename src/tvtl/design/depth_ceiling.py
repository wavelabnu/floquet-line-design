"""Delivered-depth ceiling of a varactor at its bias point (ANALYTICAL; checked before any HB job).

The pump may swing the junction only between 0 V (no forward-biased interval) and the part's maximum
reverse voltage ``V_max``. At DC reverse bias ``V_bias`` the largest allowed junction pump amplitude is

    A_max = min(V_bias, V_max - V_bias)

and, in the tool's linear depth convention m = (M/V_j)|V_j| (``delivered_gain.modulation_profile``), the
C(V) depth ceiling at the bias point is

    m_ceiling = (M/V_j) A_max.

A candidate whose delivered depth exceeds it, i.e. whose largest junction pump amplitude from the linear
pump solve exceeds A_max, is refused with the reason ``delivered_depth_above_ceiling``: its gain proxy
rests on a swing the device cannot make, and harmonic balance is not spent on it.

``V_max`` defaults, for a palette part, to its datasheet-backed reverse-bias ceiling: the largest voltage
of its C(V) table, else its polynomial validity limit, else 12 V (flagged).
"""
from __future__ import annotations

import numpy as np

REASON = "delivered_depth_above_ceiling"


def part_v_max_V(entry) -> tuple[float, str]:
    """(V_max, source) for a ``tvtl.varactor_palette.PaletteEntry``."""
    if entry.cv_table:
        return float(max(entry.cv_table)), "largest VR of the datasheet C(V) table"
    if entry.cv_poly_valid_max_v is not None:
        return float(entry.cv_poly_valid_max_v), "polynomial validity limit"
    return 12.0, "no published ceiling: 12 V (flagged)"


def reverse_swing_limit_V(bias_V: float, v_max_V: float | None) -> float:
    """A_max = min(V_bias, V_max - V_bias), never negative; without V_max only the forward side limits."""
    if not np.isfinite(bias_V) or bias_V < 0:
        raise ValueError("bias_V must be a finite reverse bias >= 0")
    a = float(bias_V) if v_max_V is None else min(float(bias_V), float(v_max_V) - float(bias_V))
    return max(0.0, a)


def depth_ceiling_check(cell_fn, f_s: float, *, N: int, P_pump_dBm: float, bias_V: float,
                        v_max_V: float | None, Z0: float = 50.) -> dict:
    """Delivered depth (linear pump solve at the board-plane pump) against the C(V) ceiling."""
    from tvtl_ads_builder_v2.analytical.delivered_gain import _sample, delivered_pump_profile
    vj = np.abs(delivered_pump_profile(cell_fn=cell_fn, N=N, f_p=2 * f_s, P_inc_dBm=P_pump_dBm, Z0=Z0))
    cell = _sample(cell_fn if callable(cell_fn) else cell_fn[0], 2 * f_s)
    k = cell.M / cell.Vj
    a_max = reverse_swing_limit_V(bias_V, v_max_V)
    ok = bool(vj.max() <= a_max)
    return {"ok": ok, "reason": None if ok else REASON, "m_del_raw": float(k * vj.max()),
            "m_ceiling": float(k * a_max), "max_junction_pump_V": float(vj.max()),
            "swing_limit_V": a_max, "bias_V": float(bias_V), "v_max_V": v_max_V,
            "P_pump_dBm": float(P_pump_dBm), "tier": "ANALYTICAL"}
