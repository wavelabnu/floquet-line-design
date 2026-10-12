"""Analytical parametric gain of a Level10IdealC stub-loaded cascade.

The analytical (TVTL) leg of the ideal-C cross-validation. Unlike the pump-informed
:mod:`.parametric_gain` proxy (which reads ``V_pump`` from a live HB dataset),
the ideal time-varying capacitor's modulation depth is **known by design** --
``depth = K*Voc/C0`` where ``Voc`` is the control-port pump's open-circuit
voltage -- so this leg is computed entirely from the spec, with no dataset.

It composes the capacitance-linear branch
(:func:`tvtl.floquet.harmonic_abcd.tv_capacitor_harmonic_abcd`, the model the
control-port ideal-C SDD realises) in series with the cell's stub-MLIN + via
network (per-harmonic impedance from the lifted
:func:`...build_stub_y_fn_from_spec`), between the validated Figure-4
through-line halves, and returns the signal-proxy parametric gain
``20*log10|S21_on[0,0]| - 20*log10|S21_off[0,0]|`` (pumped minus unpumped),
matched 50 ohm at all harmonics.

The result carries ``label="analytical_proxy_only"`` -- it is the analytical
proxy for the parametric gain mechanism, the cross-validation counterpart to
the ADS-HB device gain, not a device-gain claim.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from tvtl.floquet.harmonic_abcd import (
    cascade_harmonic_abcd,
    device_coherent_gain,
    device_proxy_gain,
    harmonic_frequencies,
    harmonic_s_parameters,
    tv_capacitor_harmonic_abcd,
)

from ..device_spec import DeviceSpec
from .floquet_stub import build_stub_y_fn_from_spec
from .mtee_blocks import tee_harmonic_blocks
from .parametric_gain import _lossy_tl_harmonic_block, _tvtl_substrate_from_spec

__all__ = [
    "ideal_c_realised_depth",
    "level10_ideal_c_param_gain",
]


def _open_circuit_pump_voltage(power_dBm: float, z_ohm: float) -> float:
    """Open-circuit peak voltage of an ADS ``Port`` source into a zero-current
    control port: ``Voc = 2*sqrt(2*z*P_watts)`` (mirrors
    ``parity.level10_ideal_c.open_circuit_pump_voltage`` -- inlined to keep the
    analytical package free of a parity-package import)."""

    p_watts = 10.0 ** ((float(power_dBm) - 30.0) / 10.0)
    return 2.0 * math.sqrt(2.0 * float(z_ohm) * p_watts)


def _var(spec: DeviceSpec, name: str) -> float:
    if spec.var_block is None:
        raise ValueError("spec has no VAR block.")
    for entry in spec.var_block.variables:
        if entry.name == name:
            return float(entry.value)
    raise KeyError(f"VAR {name!r} not declared (have "
                   f"{[v.name for v in spec.var_block.variables]}).")


def _ideal_c_sdds(spec: DeviceSpec) -> list:
    sdds = [
        e for e in spec.elements
        if e.type == "SDD2P" and "tvc_c0_f" in e.parameters
    ]
    if not sdds:
        raise ValueError(
            "no ideal-C (time-varying-capacitor) SDD2P elements in spec; this "
            "is for a Level10IdealC cascade."
        )
    return sdds


def ideal_c_realised_depth(spec: DeviceSpec, *, z_ohm: float = 50.0) -> float:
    """Return the by-design modulation depth ``m = K*Voc/C0`` of the spec.

    ``C0`` and ``K`` come from the first ideal-C SDD; ``Voc`` from the VAR
    ``pump_power_dBm`` (the control-port open-circuit voltage). All cells share
    the control net, so the depth is uniform.
    """

    sdd = _ideal_c_sdds(spec)[0]
    c0 = float(sdd.parameters["tvc_c0_f"])
    k = float(sdd.parameters["tvc_k_f_per_v"])
    voc = _open_circuit_pump_voltage(_var(spec, "pump_power_dBm"), z_ohm)
    return k * voc / c0


def level10_ideal_c_param_gain(
    spec: DeviceSpec, *, truncation_N: int = 6, use_mtee_parasitic: bool = True
) -> Mapping[str, object]:
    """Return the analytical parametric-gain dict for an ideal-C cascade.

    Reads C0, the by-design depth, the operating frequencies, and the stub
    geometry from ``spec``; composes the capacitance-linear harmonic-ABCD
    cascade; and returns the signal-proxy gains.

    Parameters
    ----------
    spec : DeviceSpec
        A Level10IdealC stub-loaded cascade spec.
    truncation_N : int, optional
        Floquet / harmonic truncation order; the block size is ``2*N + 1``.
        Default 6. Must be >= 1 (the idler at ``n = -1`` is needed).
    use_mtee_parasitic : bool, optional
        When ``True`` (the **default for the Level10IdealC family**) the
        reference-plane :func:`tvtl.microstrip.discontinuities.tee_junction`
        model is inserted per harmonic via :func:`.mtee_blocks.tee_harmonic_blocks`
        (the ``ta`` / ``tb`` main-line scaling + ``l_a`` / ``l_b`` length
        corrections + the ``j*bt`` junction shunt), with a per-harmonic
        fallback to the ideal junction above the model's high-frequency
        validity edge (``|abcd_left[0,0]| < 0.7`` or a non-finite return).
        When ``False`` the MTEE T-junction is the **ideal junction** (a pure
        shunt at the tap; the historical behaviour). The MTEE-model upgrade made
        ``True`` the family default; pass ``False`` to recover the historical
        ideal-junction prediction.

    Returns
    -------
    Mapping
        ``param_gain_dB`` (pumped minus unpumped),
        ``abs_gain_on_dB`` / ``abs_gain_off_dB`` (``20*log10|S21[0,0]|``),
        ``idler_over_signal`` (at the output, pumped),
        ``depth`` (by-design), ``n_cells``, ``label="analytical_proxy_only"``.
    """

    if int(truncation_N) < 1:
        raise ValueError("truncation_N must be >= 1 (idler harmonic needed).")
    z_ref = float(spec.analysis.port_impedance_ohm)
    sdd = _ideal_c_sdds(spec)[0]
    c0 = float(sdd.parameters["tvc_c0_f"])
    depth = ideal_c_realised_depth(spec, z_ohm=z_ref)
    n_cells = len(_ideal_c_sdds(spec))

    f_signal = _var(spec, "f_signal")
    f_pump = _var(spec, "f_pump")
    ws = 2.0 * math.pi * f_signal
    wm = 2.0 * math.pi * f_pump
    omega_n = harmonic_frequencies(ws, wm, int(truncation_N))

    substrate = _tvtl_substrate_from_spec(spec)
    left = next(e for e in spec.elements if e.id == "LINE_LEFT_C1")
    w_m = float(left.parameters["w_m"])
    l_m = float(left.parameters["l_m"])

    y_stub = build_stub_y_fn_from_spec(spec, 0)
    z_stub_fn = lambda omega: 1.0 / y_stub(omega)

    if use_mtee_parasitic:
        tee = next(e for e in spec.elements if e.type == "MTEE" and e.id == "TEE_C1")
        w3_m = float(tee.parameters["w3_m"])
        block_left, block_right, j_bt_vec, l_a, l_b = tee_harmonic_blocks(
            w_m, w3_m, omega_n, substrate,
        )
        tl_left = _lossy_tl_harmonic_block(w_m, l_m - l_a, omega_n, substrate)
        tl_right = _lossy_tl_harmonic_block(w_m, l_m - l_b, omega_n, substrate)
    else:
        tl_left = tl_right = _lossy_tl_harmonic_block(w_m, l_m, omega_n, substrate)
        block_left = block_right = None
        j_bt_vec = None

    def _gain(d: float) -> tuple[float, float, np.ndarray]:
        a, b, y, dd = tv_capacitor_harmonic_abcd(
            c0, d, 0.0, ws, wm, int(truncation_N), z_stub_fn=z_stub_fn
        )
        if use_mtee_parasitic:
            branch = (a, b, y + np.diag(j_bt_vec), dd)
            cell = cascade_harmonic_abcd(
                [tl_left, block_left, branch, block_right, tl_right]
            )
        else:
            branch = (a, b, y, dd)
            cell = cascade_harmonic_abcd([tl_left, branch, tl_right])
        cur = None
        for _ in range(n_cells):
            cur = cell if cur is None else cascade_harmonic_abcd([cur, cell])
        _s11, s21 = harmonic_s_parameters(*cur, z_ref, z_ref)
        g = device_proxy_gain(s21, int(truncation_N))
        sig = abs(s21[truncation_N, truncation_N])
        idl = abs(s21[truncation_N - 1, truncation_N])
        return float(g["gain_signal_proxy_dB"]), (idl / sig if sig > 0 else float("nan")), s21

    g_on, idler_over_signal, s21_on = _gain(depth)
    g_off, _, s21_off = _gain(0.0)
    coh = device_coherent_gain(s21_on, s21_off, int(truncation_N))
    return {
        "param_gain_dB": float(g_on - g_off),
        "abs_gain_on_dB": float(g_on),
        "abs_gain_off_dB": float(g_off),
        "idler_over_signal": float(idler_over_signal),
        "depth": float(depth),
        "n_cells": int(n_cells),
        "coherent_peak_dB": coh["coherent_peak_dB"],
        "coherent_min_dB": coh["coherent_min_dB"],
        "coherent_swing_dB": coh["coherent_swing_dB"],
        "coherent_at_phase0_dB": coh["coherent_at_block_phase_dB"],
        "label": "analytical_proxy_only",
    }


