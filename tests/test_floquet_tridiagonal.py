"""Tests for the analytical Floquet tridiagonal solver for the shunt-stub RLC.

The solver targets the Valagiannopoulos derivation (eq. 3-6): a series RLC
with a time-varying elastance ``S(t) = S0 + dS*cos(wm*t + phi)`` driven by
``v(t) = V0*exp(+j*ws*t)``. All numbers below are analytical / proxy only and
make no ADS-HB gain claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.convergence import check_floquet_truncation_convergence
from tvtl.floquet.coupling_metrics import analytical_stub_gain_proxy
from tvtl.floquet.harmonics import (
    build_stub_tridiagonal,
    current_spectrum,
    solve_harmonic_coefficients,
)


# SMV1245-representative branch parameters used across the parametric tests.
C0_REF = 7.0e-12
S0_REF = 1.0 / C0_REF
R_REF = 1.5
L_REF = 1.7e-9
FS_REF = 3.1e9
WS_REF = 2.0 * math.pi * FS_REF
WM_REF = 2.0 * WS_REF


def test_static_limit_recovers_single_mode_rlc() -> None:
    """When dS=0 the system decouples; only Q_0 survives."""

    N = 6
    Q = solve_harmonic_coefficients(
        S0=S0_REF, dS=0.0, phi=0.0, R=R_REF, L=L_REF, ws=WS_REF, wm=WM_REF, N=N
    )
    expected_q0 = 1.0 / (S0_REF + 1j * R_REF * WS_REF - L_REF * WS_REF**2)
    assert Q[N] == pytest.approx(expected_q0, rel=1e-12, abs=1e-18)
    for index, value in enumerate(Q):
        if index == N:
            continue
        assert abs(value) < 1.0e-10, f"Q[{index - N}] should be zero in the static limit."


def test_phi_zero_lossless_matrix_is_hermitian() -> None:
    """For phi=0 and R=0 the tridiagonal matrix is Hermitian.

    The conjugate-symmetry property ``A[i, j] = conj(A[j, i])`` requires real
    diagonal entries, which only holds in the lossless limit. Off-diagonal
    conjugate symmetry holds for any phi but is included implicitly here.
    """

    for N in (1, 3, 7):
        A, _ = build_stub_tridiagonal(
            S0=S0_REF, dS=0.4 * S0_REF, phi=0.0, R=0.0, L=L_REF, ws=WS_REF, wm=WM_REF, N=N
        )
        assert np.allclose(A, A.conj().T, atol=1.0e-20)


def test_convergence_sweep_reaches_smv1245_tolerance() -> None:
    result = check_floquet_truncation_convergence(
        S0=S0_REF,
        dS=0.3 * S0_REF,
        phi=0.0,
        R=R_REF,
        L=L_REF,
        ws=WS_REF,
        wm=WM_REF,
        N_min=3,
        N_max=20,
        tol=1.0e-4,
    )
    assert result["converged"], result
    assert result["N_converged"] <= 10, result
    assert len(result["Q0_vs_N"]) >= 1
    assert len(result["Qm1_vs_N"]) >= 1


def test_modulation_increases_idler_current_at_parametric_resonance() -> None:
    """At wm = 2*ws the idler current grows with the modulation depth."""

    N = 12
    Q_static = solve_harmonic_coefficients(
        S0=S0_REF, dS=0.0, phi=0.0, R=R_REF, L=L_REF, ws=WS_REF, wm=WM_REF, N=N
    )
    Q_modulated = solve_harmonic_coefficients(
        S0=S0_REF, dS=0.3 * S0_REF, phi=0.0, R=R_REF, L=L_REF, ws=WS_REF, wm=WM_REF, N=N
    )
    _, currents_static = current_spectrum(Q_static, WS_REF, WM_REF, N)
    _, currents_modulated = current_spectrum(Q_modulated, WS_REF, WM_REF, N)
    idler_static = abs(currents_static[N - 1])
    idler_modulated = abs(currents_modulated[N - 1])
    # Static idler should be numerically zero; modulated case must produce a
    # finite parametric idler.
    assert idler_static < 1.0e-10
    assert idler_modulated > idler_static
    assert idler_modulated > 1.0e-12


def test_gain_proxy_returns_finite_dict_with_disclaimer_label() -> None:
    N = 10
    Q = solve_harmonic_coefficients(
        S0=S0_REF, dS=0.3 * S0_REF, phi=0.0, R=R_REF, L=L_REF, ws=WS_REF, wm=WM_REF, N=N
    )
    report = analytical_stub_gain_proxy(Q, WS_REF, WM_REF, N)
    assert set(report.keys()) == {"gain_proxy_dB", "I_signal_amp", "I_idler_amp", "label"}
    assert report["label"] == "analytical_proxy_only"
    assert math.isfinite(float(report["gain_proxy_dB"]))
    assert isinstance(report["I_signal_amp"], complex)
    assert isinstance(report["I_idler_amp"], complex)


