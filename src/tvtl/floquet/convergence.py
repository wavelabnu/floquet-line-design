"""Convergence checks for local Floquet proxy solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harmonics import solve_harmonic_coefficients
from .uc_matrix import FloquetBranchParameters, solve_local_floquet_admittance


@dataclass(frozen=True, slots=True)
class FloquetConvergenceCheck:
    relative_error: float
    current_order: int
    next_order: int
    converged_by_proxy: bool


def compare_truncation_orders(
    parameters: FloquetBranchParameters,
    signal_angular_frequency_rad_s: float,
    modulation_angular_frequency_rad_s: float,
    truncation_order: int,
    *,
    tolerance: float = 1.0e-3,
) -> FloquetConvergenceCheck:
    current = solve_local_floquet_admittance(parameters, signal_angular_frequency_rad_s, modulation_angular_frequency_rad_s, truncation_order)
    next_result = solve_local_floquet_admittance(parameters, signal_angular_frequency_rad_s, modulation_angular_frequency_rad_s, truncation_order + 1)
    current_center = int(np.where(current.harmonic_indices == 0)[0][0])
    next_center = int(np.where(next_result.harmonic_indices == 0)[0][0])
    current_value = current.admittance_matrix_s[current_center, current_center]
    next_value = next_result.admittance_matrix_s[next_center, next_center]
    denominator = max(abs(next_value), np.finfo(float).tiny)
    error = float(abs(next_value - current_value) / denominator)
    return FloquetConvergenceCheck(
        relative_error=error,
        current_order=int(truncation_order),
        next_order=int(truncation_order) + 1,
        converged_by_proxy=error <= float(tolerance),
    )


def check_floquet_truncation_convergence(
    S0: float,
    dS: float,
    phi: float,
    R: float,
    L: float,
    ws: float,
    wm: float,
    N_min: int = 3,
    N_max: int = 20,
    tol: float = 1.0e-4,
) -> dict:
    """Sweep the Floquet truncation order and locate convergence.

    For each truncation ``N`` in ``[N_min, N_max]`` the shunt-stub tridiagonal
    system is solved and the magnitudes of the signal charge ``|Q_0|`` and the
    parametric idler charge ``|Q_{-1}|`` are recorded. The smallest ``N`` at
    which both magnitudes change by less than ``tol`` (relative to their
    value at ``N + 1``) is reported as ``N_converged``.

    Parameters
    ----------
    S0, dS, phi, R, L, ws, wm
        Branch parameters and operating frequencies. See
        :func:`tvtl.floquet.harmonics.build_stub_tridiagonal` for definitions.
    N_min, N_max : int
        Inclusive sweep bounds for the truncation order. Must satisfy
        ``0 <= N_min < N_max``.
    tol : float
        Relative-change threshold applied to both ``|Q_0|`` and ``|Q_{-1}|``.

    Returns
    -------
    dict
        Keys: ``"N_converged"`` (int), ``"Q0_vs_N"`` and ``"Qm1_vs_N"`` (lists
        of ``(N, magnitude)`` pairs), and ``"converged"`` (bool). When no ``N``
        in the sweep meets the tolerance, ``converged`` is ``False`` and
        ``N_converged`` falls back to ``N_max``.
    """

    if int(N_min) < 0:
        raise ValueError("N_min must be non-negative.")
    if int(N_max) <= int(N_min):
        raise ValueError("N_max must be greater than N_min.")
    if float(tol) <= 0.0:
        raise ValueError("tol must be positive.")

    magnitudes_q0: dict[int, float] = {}
    magnitudes_qm1: dict[int, float] = {}
    for order in range(int(N_min), int(N_max) + 1):
        coefficients = solve_harmonic_coefficients(S0, dS, phi, R, L, ws, wm, order)
        center = order  # index of n=0 in the (2*order+1)-length vector
        magnitudes_q0[order] = float(abs(coefficients[center]))
        if order >= 1:
            magnitudes_qm1[order] = float(abs(coefficients[center - 1]))

    q0_vs_n = [(order, magnitudes_q0[order]) for order in sorted(magnitudes_q0)]
    qm1_vs_n = [(order, magnitudes_qm1[order]) for order in sorted(magnitudes_qm1)]

    n_converged: int | None = None
    converged = False
    for order in range(int(N_min), int(N_max)):
        next_order = order + 1
        q0_now = magnitudes_q0[order]
        q0_next = magnitudes_q0[next_order]
        qm1_now = magnitudes_qm1[order]
        qm1_next = magnitudes_qm1[next_order]
        q0_denom = max(q0_next, np.finfo(float).tiny)
        qm1_denom = max(qm1_next, np.finfo(float).tiny)
        q0_rel = abs(q0_next - q0_now) / q0_denom
        qm1_rel = abs(qm1_next - qm1_now) / qm1_denom
        if q0_rel <= float(tol) and qm1_rel <= float(tol):
            n_converged = order
            converged = True
            break
    if n_converged is None:
        n_converged = int(N_max)
    return {
        "N_converged": int(n_converged),
        "Q0_vs_N": q0_vs_n,
        "Qm1_vs_N": qm1_vs_n,
        "converged": bool(converged),
    }


