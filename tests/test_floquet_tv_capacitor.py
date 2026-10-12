"""Tests for the capacitance-linear ideal time-varying-capacitor branch.

:func:`tvtl.floquet.harmonic_abcd.tv_capacitor_harmonic_admittance` models an
ideal *linear* time-varying capacitor ``C(t) = C0*(1 + depth*cos(wm*t + phi))``
with ``Q = C(t)*v`` -- the model an externally-pumped ideal-C device (a
control-port SDD) realises, and the one needed to cross-validate such a device
(distinct from the elastance-linear :func:`stub_harmonic_admittance`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.harmonic_abcd import (
    harmonic_frequencies,
    stub_harmonic_admittance,
    tv_capacitor_harmonic_abcd,
    tv_capacitor_harmonic_admittance,
)

C0 = 0.5e-12
FS = 3.1e9
WS = 2.0 * math.pi * FS
WM = 2.0 * WS  # degenerate pump
N = 4


def test_bare_depth_zero_is_diagonal_jw_c0():
    """depth=0 -> a plain (time-invariant) capacitor: Y = diag(j*omega_n*C0)."""
    omega = harmonic_frequencies(WS, WM, N)
    Y = tv_capacitor_harmonic_admittance(C0, 0.0, 0.0, WS, WM, N)
    assert np.allclose(np.diag(Y), 1j * omega * C0)
    off = Y - np.diag(np.diag(Y))
    assert np.allclose(off, 0.0)


def test_bare_offdiagonal_closed_form():
    """Off-diagonals are the closed form Y[n, n+/-1] = j*omega_n*0.5*depth*C0."""
    omega = harmonic_frequencies(WS, WM, N)
    depth, phi = 0.3, 0.0
    Y = tv_capacitor_harmonic_admittance(C0, depth, phi, WS, WM, N)
    size = 2 * N + 1
    for n in range(size):
        if n + 1 < size:
            assert np.isclose(Y[n, n + 1], 1j * omega[n] * 0.5 * depth * C0)
        if n - 1 >= 0:
            assert np.isclose(Y[n, n - 1], 1j * omega[n] * 0.5 * depth * C0)


def test_phase_enters_offdiagonals():
    """phi rotates the upper/lower off-diagonals by exp(-/+ j*phi)."""
    omega = harmonic_frequencies(WS, WM, N)
    depth, phi = 0.4, 0.7
    Y = tv_capacitor_harmonic_admittance(C0, depth, phi, WS, WM, N)
    n = N  # the n = 0 row
    assert np.isclose(Y[n, n + 1], 1j * omega[n] * 0.5 * depth * C0 * np.exp(-1j * phi))
    assert np.isclose(Y[n, n - 1], 1j * omega[n] * 0.5 * depth * C0 * np.exp(1j * phi))


def test_series_stub_zero_reduces_to_bare():
    """A zero series-stub impedance reproduces the bare-capacitor admittance."""
    Y_bare = tv_capacitor_harmonic_admittance(C0, 0.3, 0.2, WS, WM, N)
    Y_stub = tv_capacitor_harmonic_admittance(
        C0, 0.3, 0.2, WS, WM, N, z_stub_fn=lambda w: 0.0
    )
    assert np.allclose(Y_bare, Y_stub)


def test_series_stub_depth_zero_is_series_combination():
    """At depth=0 the n=0 admittance is exactly 1/(Z_C + Z_stub) (series law)."""
    L = 1.0e-9
    z_stub = lambda w: 1j * w * L
    Y = tv_capacitor_harmonic_admittance(
        C0, 0.0, 0.0, WS, WM, N, z_stub_fn=z_stub
    )
    z_c = 1.0 / (1j * WS * C0)
    expected = 1.0 / (z_c + z_stub(WS))
    assert np.isclose(Y[N, N], expected)


def test_small_depth_matches_elastance_to_first_order():
    """At small depth, |coupling| matches the elastance solver to O(depth).

    cap-linear C_1 = +0.5*depth*C0 and elastance S_1 = -0.5*depth/C0 carry the
    same magnitude (opposite sign, since S = 1/C), so the off-diagonal coupling
    magnitudes agree to first order; the relative gap is O(depth).
    """
    depth = 0.02
    Yc = tv_capacitor_harmonic_admittance(C0, depth, 0.0, WS, WM, N)
    Ye = stub_harmonic_admittance(1.0 / C0, depth / C0, 0.0, 0.0, 0.0, WS, WM, N)
    # diagonals (the dominant terms) agree closely; off-diagonal magnitudes too.
    assert np.allclose(np.diag(Yc), np.diag(Ye), rtol=2 * depth)
    assert np.isclose(abs(Yc[N, N + 1]), abs(Ye[N, N + 1]), rtol=3 * depth)


def test_abcd_is_shunt_block():
    """The ABCD wrapper is the shunt block (I, 0, Y, I)."""
    a, b, c, d = tv_capacitor_harmonic_abcd(C0, 0.3, 0.0, WS, WM, N)
    size = 2 * N + 1
    assert np.allclose(a, np.eye(size))
    assert np.allclose(b, 0.0)
    assert np.allclose(d, np.eye(size))
    assert np.allclose(c, tv_capacitor_harmonic_admittance(C0, 0.3, 0.0, WS, WM, N))


@pytest.mark.parametrize("bad_depth", [-0.1, 1.5])
def test_rejects_out_of_range_depth(bad_depth):
    with pytest.raises(ValueError):
        tv_capacitor_harmonic_admittance(C0, bad_depth, 0.0, WS, WM, N)


def test_rejects_nonpositive_c0():
    with pytest.raises(ValueError):
        tv_capacitor_harmonic_admittance(0.0, 0.3, 0.0, WS, WM, N)


