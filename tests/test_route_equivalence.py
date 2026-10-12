"""Public route-equivalence artifact for the two-formulation admittance claim.

This test is the PUBLIC ARTIFACT backing the two-formulation equivalence claim
(the internal "route-equivalence note bundle", reported at ~1e-15). It builds
the per-cell harmonic admittance of the series R-L-S(t) branch two independent
ways and asserts they agree to machine precision on the full harmonic block:

Route (i) -- the repository's existing elastance/tridiagonal solver
    (:func:`tvtl.floquet.harmonics.build_stub_tridiagonal`). Its matrix ``A`` is
    the charge-basis Floquet system ``A @ Q = V`` (applied-voltage drive); with
    harmonic currents ``I_n = j*omega_n*Q_n`` the branch admittance is
    ``Y_i = (j*Omega) @ inv(A)``.

Route (ii) -- an in-test impedance-matrix assembly, independent of the repo
    construction:
        ``Z_c = S_toep @ inv(j*Omega) + R*I + j*Omega*L`` , ``Y_ii = inv(Z_c)``
    with ``Omega = diag(w_s + n*w_m)`` and ``S_toep`` the elastance Toeplitz
    matrix (``S0`` on the diagonal, ``dS/2 * exp(+-j*phi)`` on the first
    off-diagonals).

The two routes are algebraically identical -- ``Z_c == A @ inv(j*Omega)`` -- so
the residual is pure floating-point round-off (~1e-15). The achieved deviation
is printed for the reproduction report.

Config (fixed): C0 = 7 pF, R = 1.5 ohm, L = 1.7 nH, f_s = 3.1 GHz,
w_m = 2*w_s, dS/S0 in {0.1, 0.3}, N_h in {6, 10}. Pure analytical: no ADS.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.harmonics import build_stub_tridiagonal

# Fixed branch configuration.
C0 = 7.0e-12
S0 = 1.0 / C0
R = 1.5
L = 1.7e-9
FS = 3.1e9
WS = 2.0 * math.pi * FS
WM = 2.0 * WS


def _omega(N: int) -> np.ndarray:
    """Harmonic angular frequencies ``w_s + n*w_m`` for ``n = -N..N``."""
    n = np.arange(-N, N + 1, dtype=float)
    return WS + n * WM


def _admittance_route_tridiagonal(dS: float, phi: float, N: int) -> np.ndarray:
    """Route (i): the repo elastance/tridiagonal matrix -> Y = (j*Omega) @ inv(A)."""
    A, _rhs = build_stub_tridiagonal(
        S0=S0, dS=dS, phi=phi, R=R, L=L, ws=WS, wm=WM, N=N
    )
    j_omega = np.diag(1j * _omega(N))
    return j_omega @ np.linalg.inv(A)


def _admittance_route_impedance_matrix(dS: float, phi: float, N: int) -> np.ndarray:
    """Route (ii): independent impedance-matrix assembly -> Y = inv(Z_c)."""
    size = 2 * N + 1
    j_omega = np.diag(1j * _omega(N))
    inv_j_omega = np.diag(1.0 / (1j * _omega(N)))

    # Elastance Toeplitz S_toep: S0 on the diagonal, dS/2 on the first
    # off-diagonals, matching build_stub_tridiagonal's coupling convention
    # (matrix[row, row-1] = dS/2 * exp(+j*phi); matrix[row, row+1] = dS/2 * exp(-j*phi)).
    s_toep = S0 * np.eye(size, dtype=complex)
    sub = 0.5 * dS * np.exp(1j * phi)
    sup = 0.5 * dS * np.exp(-1j * phi)
    for row in range(size):
        if row >= 1:
            s_toep[row, row - 1] = sub
        if row + 1 < size:
            s_toep[row, row + 1] = sup

    z_c = s_toep @ inv_j_omega + R * np.eye(size, dtype=complex) + j_omega * L
    return np.linalg.inv(z_c)


def _max_relative_deviation(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Max absolute entrywise difference normalised by the block's peak magnitude."""
    scale = float(np.max(np.abs(y_b)))
    return float(np.max(np.abs(y_a - y_b)) / scale)


@pytest.mark.parametrize("n_harmonics", [6, 10])
@pytest.mark.parametrize("dS_fraction", [0.1, 0.3])
def test_route_equivalence_full_block(dS_fraction, n_harmonics, capsys) -> None:
    """The tridiagonal-solver and impedance-matrix admittances agree < 1e-12."""

    dS = dS_fraction * S0
    phi = 0.0
    y_tridiag = _admittance_route_tridiagonal(dS, phi, n_harmonics)
    y_impedance = _admittance_route_impedance_matrix(dS, phi, n_harmonics)

    deviation = _max_relative_deviation(y_tridiag, y_impedance)
    with capsys.disabled():
        print(
            f"[route-equivalence] N={n_harmonics} dS/S0={dS_fraction}: "
            f"max relative deviation = {deviation:.3e}"
        )

    assert deviation < 1.0e-12, (
        f"route admittances diverged: N={n_harmonics}, dS/S0={dS_fraction}, "
        f"max relative deviation = {deviation:.3e}"
    )
