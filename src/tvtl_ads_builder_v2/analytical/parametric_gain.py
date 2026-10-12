"""Analytical parametric-gain proxy for a Level-10 cascade (harmonic ABCD + lift).

This is the analytical leg of the Figure-6 reproduction (Approach A,
"pump-informed"). For each operating point it predicts the cascade's
signal-harmonic gain by composing the *harmonic ABCD lift* into a
block-ABCD cascade:

* the through-line halves (``LINE_LEFT_C{k}`` / ``LINE_RIGHT_C{k}``) are
  block-diagonal in the harmonic basis -- each harmonic ``n`` propagates
  with its own lossy ``gamma(omega_n)`` from the same
  Hammerstad-Kirschning + ``calc_gamma_legacy`` machinery the Figure-4
  path (:mod:`.level10_sparams`) uses;
* the shunt branch is the time-varying varactor: its Floquet harmonic
  admittance matrix ``Y_branch[n,m] = j*omega_n * G[n,m]`` (with
  ``G = inv(tridiagonal)``) couples harmonics. The tridiagonal is built
  with the lifted ``y_stub_fn`` (the stub-MLIN-with-via per-harmonic
  admittance), so the branch is varactor (time-varying C, with the
  package R/L as the local series RLC) in series with the full stub
  cascade -- identical physics to the validated Figure-4 branch in the
  ``dS = 0`` limit.

The per-cell modulation depth ``dS = |C1*V_pump| / C0^2`` is read from the
live HB dataset's pump-frequency voltage at each varactor branch top
(:func:`extract_varactor_parameters`, mix ``(0, 1)``) -- the
"pump-informed" design choice. The cascade is then converted
to block S-parameters and the signal proxy gain
``20*log10|S21[n=0, n=0]|`` is returned via
:func:`tvtl.floquet.harmonic_abcd.device_proxy_gain`.

Every result carries ``label="analytical_proxy_only"`` (inherited from
``device_proxy_gain``): this is an analytical proxy for the parametric
gain mechanism, NOT a substitute for the ADS-HB device gain.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from tvtl.floquet.harmonic_abcd import (
    cascade_harmonic_abcd,
    device_proxy_gain,
    harmonic_frequencies,
    harmonic_s_parameters,
)
from tvtl.floquet.harmonics import build_stub_tridiagonal
from tvtl.materials import Substrate as TvtlSubstrate
from tvtl.microstrip.models import calc_gamma_legacy, hammerstad_kirschning

from ..analysis.floquet_comparison import extract_varactor_parameters
from ..analysis.hb_dataset import HBDataset
from ..analysis.per_cell import CellMap, cell_count
from ..device_spec import DeviceSpec
from .floquet_stub import build_stub_y_fn_from_spec
from .mtee_blocks import tee_harmonic_blocks

__all__ = [
    "level10_parametric_gain_proxy",
    "level10_signal_gain_proxy_dB",
]


def _tvtl_substrate_from_spec(spec: DeviceSpec) -> TvtlSubstrate:
    """Build a :class:`tvtl.materials.Substrate` from the spec's substrate.

    Mirrors :func:`.level10_sparams._tvtl_substrate_from_spec` so the
    analytical gain leg uses the identical line model the Figure-4 path
    validated against ADS.
    """

    params = spec.substrate.parameters
    return TvtlSubstrate(
        epsilon_r=float(params["eps_r"]),
        thickness_m=float(params["h_m"]),
        copper_thickness_m=float(params.get("t_m", 17.5e-6)),
        loss_tangent=float(params.get("tan_delta", 0.0)),
        conductivity_s_per_m=float(params.get("sigma_S_per_m", 5.8e7)),
    )


def _lossy_tl_harmonic_block(
    width_m: float,
    length_m: float,
    omega_n: np.ndarray,
    substrate: TvtlSubstrate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Block-diagonal lossy-TL ABCD over the harmonic axis.

    Each harmonic ``n`` propagates independently with its own
    ``gamma(omega_n)``. ``Z_0`` and ``gamma`` come from
    Hammerstad-Kirschning + ``calc_gamma_legacy`` evaluated at
    ``|omega_n| / (2*pi)`` (the microstrip model is defined for positive
    frequency); for a negative harmonic frequency the per-harmonic 2x2
    ABCD is the complex conjugate of the positive-frequency block
    (``ABCD(-omega) = conj(ABCD(omega))`` for a real network), which the
    sign flip below applies via ``theta -> conj(theta)``.
    """

    size = omega_n.shape[0]
    abs_freqs = np.abs(omega_n) / (2.0 * math.pi)
    # Guard a (non-physical for ws>0) exact-zero harmonic frequency.
    safe_freqs = np.where(abs_freqs > 0.0, abs_freqs, 1.0)
    z0_arr, eps_eff_arr = hammerstad_kirschning(
        float(width_m),
        substrate.copper_thickness_m,
        substrate.thickness_m,
        substrate.epsilon_r,
        safe_freqs,
    )
    gamma_arr = calc_gamma_legacy(safe_freqs, z0_arr, eps_eff_arr, substrate)

    a = np.zeros(size, dtype=np.complex128)
    b = np.zeros(size, dtype=np.complex128)
    c = np.zeros(size, dtype=np.complex128)
    d = np.zeros(size, dtype=np.complex128)
    for k in range(size):
        z0 = complex(z0_arr[k])
        theta = complex(gamma_arr[k]) * float(length_m)
        if omega_n[k] < 0.0:
            theta = theta.conjugate()
        cosh = np.cosh(theta)
        sinh = np.sinh(theta)
        a[k] = cosh
        b[k] = z0 * sinh
        c[k] = sinh / z0
        d[k] = cosh
    return np.diag(a), np.diag(b), np.diag(c), np.diag(d)


def _branch_harmonic_abcd(
    spec: DeviceSpec,
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    ws: float,
    wm: float,
    omega_n: np.ndarray,
    truncation_N: int,
    *,
    pump_mix: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shunt-branch harmonic ABCD ``(I, 0, Y_branch, I)`` for one cell.

    ``Y_branch[n,m] = j*omega_n * G[n,m]`` with ``G = inv(tridiagonal)``;
    the tridiagonal is the lifted Floquet system (local series
    RLC = package R/L + time-varying varactor C, plus the per-harmonic
    stub-MLIN-with-via admittance). The per-cell modulation depth comes
    from the live-HB pump voltage at this cell's varactor branch top.
    """

    k1 = cell_index + 1  # element ids are 1-indexed
    lpkg = next(e for e in spec.elements if e.id == f"LPKG_C{k1}")
    rpkg = next(e for e in spec.elements if e.id == f"RPKG_C{k1}")
    l_pkg = float(lpkg.parameters["l_h"])
    r_pkg = float(rpkg.parameters["r_ohm"])

    params = extract_varactor_parameters(
        spec, dataset, cell_map, cell_index,
        R=r_pkg, L=l_pkg, pump_mix=pump_mix,
    )
    y_stub_fn = build_stub_y_fn_from_spec(spec, cell_index)
    matrix, _rhs = build_stub_tridiagonal(
        params.S0, params.dS, params.phi, params.R, params.L,
        ws, wm, int(truncation_N), y_stub_fn=y_stub_fn,
    )
    size = matrix.shape[0]
    green = np.linalg.solve(matrix, np.eye(size, dtype=np.complex128))
    if not np.all(np.isfinite(green)):
        raise FloatingPointError(
            f"cell {cell_index}: Floquet Green's function non-finite."
        )
    y_branch = 1j * omega_n[:, np.newaxis] * green
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    return identity, zero, y_branch, identity


def level10_parametric_gain_proxy(
    spec: DeviceSpec,
    dataset: HBDataset,
    cell_map: CellMap,
    *,
    truncation_N: int = 6,
    pump_mix: tuple[int, ...] = (0, 1),
    use_mtee_parasitic: bool = False,
) -> Mapping[str, object]:
    """Return the analytical parametric-gain proxy dict for one operating point.

    Composes the harmonic-ABCD-with-lift cascade
    ``(tl_left . branch . tl_right)`` per cell, cascades all cells (each
    with its own live-HB pump depth), converts to block S-parameters at
    the spec's port impedance, and returns
    :func:`tvtl.floquet.harmonic_abcd.device_proxy_gain` of the block
    ``S21``.

    Parameters
    ----------
    spec : DeviceSpec
        A Level-10-shaped spec (SDD2P per cell, ``LINE_LEFT_C{k}`` /
        ``LINE_RIGHT_C{k}`` through-line halves, ``LPKG_C{k}`` /
        ``RPKG_C{k}`` package R/L, a stub-to-via shunt chain).
    dataset : HBDataset
        The live HB result at this operating point. ``V_pump`` is read
        per cell from it (pump-informed Approach A).
    cell_map : CellMap
        The cascade resolution (:func:`build_cell_map`).
    truncation_N : int, optional
        Floquet / harmonic truncation order; the block size is
        ``2*N + 1``. Default 6 matches the Level-9/10 Floquet-comparison
        harness. Must be >= 1 (the idler at ``n = -1`` is needed).
    pump_mix : tuple of int, optional
        ADS mix tuple at which the pump voltage is read; default
        ``(0, 1)`` (the pump fundamental).
    use_mtee_parasitic : bool, optional
        When ``False`` (default) the MTEE is the ideal junction (a pure shunt
        at the tap). When ``True`` the reference-plane
        :func:`tvtl.microstrip.discontinuities.tee_junction` is inserted per
        harmonic via :func:`.mtee_blocks.tee_harmonic_blocks` (with the
        high-frequency ideal-junction fallback). The Level-10 paper-fidelity
        canonical leaves this ``False`` to preserve its validated baseline;
        the flag is provided for parity with the ideal-C cross-validation leg.

    Returns
    -------
    Mapping
        ``device_proxy_gain`` output: ``gain_signal_proxy_dB``,
        ``gain_idler_proxy_dB``, ``gain_total_proxy_dB`` (floats) and
        ``label = "analytical_proxy_only"``.

    Raises
    ------
    ValueError
        If the cascade has no cells, ``truncation_N < 1``, or the
        small-signal elastance linearisation is exceeded at this pump
        amplitude (propagated from :func:`extract_varactor_parameters`).
    """

    if int(truncation_N) < 1:
        raise ValueError("truncation_N must be >= 1 (idler harmonic needed).")
    n_cells = cell_count(cell_map)
    if n_cells == 0:
        raise ValueError("cascade has no cells.")

    f_signal = float(dataset.fundamental_freqs_Hz[0])
    f_pump = float(dataset.fundamental_freqs_Hz[1])
    ws = 2.0 * math.pi * f_signal
    wm = 2.0 * math.pi * f_pump
    omega_n = harmonic_frequencies(ws, wm, int(truncation_N))

    substrate = _tvtl_substrate_from_spec(spec)
    z_ref = float(spec.analysis.port_impedance_ohm)

    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for k in range(n_cells):
        k1 = k + 1
        left = next(e for e in spec.elements if e.id == f"LINE_LEFT_C{k1}")
        right = next(e for e in spec.elements if e.id == f"LINE_RIGHT_C{k1}")
        branch = _branch_harmonic_abcd(
            spec, dataset, cell_map, k, ws, wm, omega_n,
            int(truncation_N), pump_mix=pump_mix,
        )
        if use_mtee_parasitic:
            tee = next(
                e for e in spec.elements if e.type == "MTEE" and e.id == f"TEE_C{k1}"
            )
            block_left, block_right, j_bt_vec, l_a, l_b = tee_harmonic_blocks(
                float(left.parameters["w_m"]),
                float(tee.parameters["w3_m"]),
                omega_n, substrate,
            )
            tl_left = _lossy_tl_harmonic_block(
                float(left.parameters["w_m"]),
                float(left.parameters["l_m"]) - l_a,
                omega_n, substrate,
            )
            tl_right = _lossy_tl_harmonic_block(
                float(right.parameters["w_m"]),
                float(right.parameters["l_m"]) - l_b,
                omega_n, substrate,
            )
            ba, bb, by, bd = branch
            branch = (ba, bb, by + np.diag(j_bt_vec), bd)
            blocks.append(cascade_harmonic_abcd(
                [tl_left, block_left, branch, block_right, tl_right]
            ))
        else:
            tl_left = _lossy_tl_harmonic_block(
                float(left.parameters["w_m"]),
                float(left.parameters["l_m"]),
                omega_n, substrate,
            )
            tl_right = _lossy_tl_harmonic_block(
                float(right.parameters["w_m"]),
                float(right.parameters["l_m"]),
                omega_n, substrate,
            )
            blocks.append(cascade_harmonic_abcd([tl_left, branch, tl_right]))

    total = cascade_harmonic_abcd(blocks)
    _s11, s21 = harmonic_s_parameters(*total, z_ref, z_ref)
    return device_proxy_gain(s21, int(truncation_N))


def level10_signal_gain_proxy_dB(
    spec: DeviceSpec,
    dataset: HBDataset,
    cell_map: CellMap,
    *,
    truncation_N: int = 6,
    pump_mix: tuple[int, ...] = (0, 1),
) -> float:
    """Return just the signal proxy gain ``20*log10|S21[0,0]|`` in dB.

    Thin wrapper over :func:`level10_parametric_gain_proxy` returning the
    ``"gain_signal_proxy_dB"`` entry -- the locked Figure-6 analytical
    metric (D3), matched to the ADS leg's signal-mix ``(1, 0)`` gain.
    """

    return float(
        level10_parametric_gain_proxy(
            spec, dataset, cell_map,
            truncation_N=truncation_N, pump_mix=pump_mix,
        )["gain_signal_proxy_dB"]
    )


