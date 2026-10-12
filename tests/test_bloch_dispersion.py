"""Tests for ``tvtl.theory.dispersion.bloch_dispersion``.

The cleanest cross-check: take a passive transmission line of length
``p`` and use *that single line* as the unit cell. The Bloch
propagation constant of the resulting "cascade" of identical line
sections is the line's own per-unit-length ``gamma_line`` by
construction -- if the line is uniform, repeating it does not change
its dispersion. This is independent of the line's electrical length
or impedance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.config import BlochExtractionStrategy
from tvtl.theory import bloch_dispersion, line_abcd


def test_uniform_line_unit_cell_recovers_line_gamma():
    """A single line section as the unit cell: Bloch gamma == line gamma.

    The frequency range is bounded so that ``beta * period`` stays in
    the principal-branch range ``[0, pi]``. Above the Brillouin zone
    edge ``beta * period > pi`` the arccosh wraps and a branch-tracking
    step would be needed; that is out of scope for this helper.
    """

    z0 = 50.0
    period = 10.0e-3
    eps_eff = 2.3
    c0 = 2.998e8
    # The Brillouin-zone edge is at beta * period = pi. Keep the highest
    # tested frequency well inside that range.
    f_brillouin_Hz = c0 / (2.0 * period * math.sqrt(eps_eff))
    omega = 2.0 * np.pi * np.linspace(1.0e9, 0.9 * f_brillouin_Hz, 91)
    beta_per_m = omega * math.sqrt(eps_eff) / c0
    gamma_per_m = 1j * beta_per_m

    # Build the unit-cell ABCD per frequency.
    A = np.empty_like(gamma_per_m)
    B = np.empty_like(gamma_per_m)
    C = np.empty_like(gamma_per_m)
    D = np.empty_like(gamma_per_m)
    for i, gp in enumerate(gamma_per_m):
        mat = line_abcd(z0, gp * period)
        A[i], B[i] = mat[0, 0], mat[0, 1]
        C[i], D[i] = mat[1, 0], mat[1, 1]

    # The Bloch gamma per unit length should match gamma_per_m.
    gamma_bloch = bloch_dispersion((A, B, C, D), period)

    # Im(gamma) = beta. The two should agree to machine precision; the
    # principal-branch arccosh introduces no error for a lossless line
    # whose electrical length is in the principal range.
    assert np.allclose(np.imag(gamma_bloch), beta_per_m, rtol=1e-10, atol=1e-10)
    assert np.allclose(np.real(gamma_bloch), 0.0, atol=1e-10)


def test_lossy_line_attenuation_is_recovered():
    """A lossy line: Re(gamma_bloch) == alpha (per-unit-length loss)."""

    z0 = 50.0
    period = 5.0e-3
    omega = 2.0 * np.pi * 3.0e9
    eps_eff = 2.3
    c0 = 2.998e8
    beta_per_m = omega * math.sqrt(eps_eff) / c0
    alpha_per_m = 0.5  # Np/m
    gamma_per_m = alpha_per_m + 1j * beta_per_m

    mat = line_abcd(z0, gamma_per_m * period)
    gamma_bloch = bloch_dispersion(
        (mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]),
        period,
    )

    assert np.isclose(np.real(gamma_bloch), alpha_per_m, rtol=1e-10, atol=1e-10)
    assert np.isclose(np.imag(gamma_bloch), beta_per_m, rtol=1e-10, atol=1e-10)


def test_period_must_be_positive():
    """Zero or negative period is rejected."""

    A = D = np.array([1.0 + 0j])
    B = C = np.array([0.0 + 0j])
    with pytest.raises(ValueError):
        bloch_dispersion((A, B, C, D), 0.0)
    with pytest.raises(ValueError):
        bloch_dispersion((A, B, C, D), -1.0e-3)


def test_eigenvalue_log_strategy_is_not_implemented_here():
    """The harmonic-basis strategy is at floquet_exponents, not here."""

    A = D = np.array([1.0 + 0j])
    B = C = np.array([0.0 + 0j])
    with pytest.raises(NotImplementedError):
        bloch_dispersion(
            (A, B, C, D),
            1.0e-3,
            strategy=BlochExtractionStrategy.EIGENVALUE_LOG,
        )


def test_default_strategy_is_trace_arccosh():
    """The default keyword matches the config enum's TRACE_ARCCOSH value."""

    z0 = 50.0
    period = 5.0e-3
    gamma_per_m = 1j * 0.5

    mat = line_abcd(z0, gamma_per_m * period)
    abcd = (mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1])

    default = bloch_dispersion(abcd, period)
    explicit = bloch_dispersion(
        abcd, period, strategy=BlochExtractionStrategy.TRACE_ARCCOSH
    )
    assert np.array_equal(default, explicit)


