"""Unit tests for the harmonic ABCD lift.

The lift extends :func:`tvtl.floquet.harmonics.build_stub_tridiagonal`
with an optional ``y_stub_fn`` callable that, when provided, augments
the diagonal terms with the per-harmonic admittance of a passive stub
network in series with the local-RLC varactor model. The off-diagonal
modulation coupling is unchanged.

This file covers:

* the lift's backward-compatibility — ``y_stub_fn=None`` reproduces the
  historical local-RLC matrix and solution bit-for-bit;
* the structural shape of the lifted matrix (tridiagonal preserved,
  off-diagonals unchanged);
* the lifted solver in the testable limit ``Y_stub -> inf`` (a perfect
  short to ground at every harmonic), which collapses to the local-RLC
  case;
* a sanity check of :func:`Y_stub_at_freq` against a hand-evaluated
  short-terminated transmission-line input impedance at a single
  frequency;
* the spec-driven extraction wrapper
  :func:`build_stub_y_fn_from_spec` against the Level-9 canonical's
  expected stub geometry (1.93 mm × 5 mm short-terminated MLIN on
  Rogers-4003C-like substrate).

The companion integration test in
``tests/parity/test_level9_floquet_comparison.py`` runs the lift
end-to-end against a live HB ``.ds``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.harmonics import (
    build_stub_tridiagonal,
    current_spectrum,
    solve_harmonic_coefficients,
)
from tvtl_ads_builder_v2.analytical import (
    StubSegment,
    Y_stub_at_freq,
    build_stub_y_fn_from_spec,
    extract_level9_stub_segments,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_9_CANONICAL_CIRCUITS,
)


# Operating-point fixtures matching the local-RLC comparison harness.
C0_REF = 7.0e-12
S0_REF = 1.0 / C0_REF
R_REF = 0.0
L_REF = 0.0
FS_REF = 3.1e9
FM_REF = 6.2e9
WS_REF = 2.0 * math.pi * FS_REF
WM_REF = 2.0 * math.pi * FM_REF


# --------------------------------------------------------------------------
# Backward compatibility (y_stub_fn=None preserves local-RLC behaviour)
# --------------------------------------------------------------------------


def test_y_stub_fn_none_preserves_local_RLC_matrix() -> None:
    """``y_stub_fn=None`` must produce the same matrix as the local-RLC solver."""

    dS = 0.05 * S0_REF
    A_baseline, b_baseline = build_stub_tridiagonal(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, 5,
    )
    A_default, b_default = build_stub_tridiagonal(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, 5,
        y_stub_fn=None,
    )
    np.testing.assert_array_equal(A_baseline, A_default)
    np.testing.assert_array_equal(b_baseline, b_default)


def test_y_stub_fn_none_preserves_local_RLC_solution() -> None:
    dS = 0.05 * S0_REF
    Qn_baseline = solve_harmonic_coefficients(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, 4,
    )
    Qn_default = solve_harmonic_coefficients(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, 4,
        y_stub_fn=None,
    )
    np.testing.assert_array_equal(Qn_baseline, Qn_default)


# --------------------------------------------------------------------------
# Lifted matrix structure
# --------------------------------------------------------------------------


def test_lifted_diagonal_differs_from_local_off_diagonals_unchanged() -> None:
    """The lift touches only the diagonal; off-diagonals are unchanged."""

    dS = 0.05 * S0_REF
    N = 4

    def y_fn(omega: float) -> complex:
        # A simple constant admittance — Z_stub = 50 Ohm at every harmonic.
        return 1.0 / 50.0 + 0.0j

    A_local, _ = build_stub_tridiagonal(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, N,
    )
    A_lift, _ = build_stub_tridiagonal(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, N,
        y_stub_fn=y_fn,
    )
    # Off-diagonal entries must be identical.
    size = 2 * N + 1
    for row in range(size):
        for col in range(size):
            if row == col:
                continue
            assert A_local[row, col] == A_lift[row, col], (
                f"off-diagonal ({row}, {col}) changed by the lift"
            )
    # Diagonal entries must DIFFER by j*omega_n*Z_stub at every harmonic.
    indices = np.arange(-N, N + 1, dtype=float)
    omega_n = WS_REF + indices * WM_REF
    z_stub = 50.0 + 0.0j
    for k in range(size):
        expected_delta = 1j * float(omega_n[k]) * z_stub
        observed_delta = A_lift[k, k] - A_local[k, k]
        assert observed_delta == pytest.approx(expected_delta, rel=1e-12), (
            f"diagonal[{k}] augmentation mismatch: "
            f"expected {expected_delta!r}, got {observed_delta!r}"
        )


def test_lifted_matrix_remains_tridiagonal() -> None:
    """The lift preserves the tridiagonal sparsity pattern."""

    dS = 0.05 * S0_REF
    N = 3

    def y_fn(omega: float) -> complex:
        return 0.01 + 0.01j

    A_lift, _ = build_stub_tridiagonal(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, N,
        y_stub_fn=y_fn,
    )
    size = 2 * N + 1
    for row in range(size):
        for col in range(size):
            if abs(row - col) > 1:
                assert A_lift[row, col] == 0.0, (
                    f"non-tridiagonal entry at ({row}, {col})"
                )


def test_lifted_solver_short_stub_limit_matches_local_RLC() -> None:
    """In the limit ``Y_stub -> inf`` (perfect short), the lift reduces to local-RLC.

    Equivalent statement: ``Z_stub = 0 -> j*omega*Z_stub = 0``, so the
    diagonal augmentation vanishes and the lifted solution should match
    the local-RLC solution.
    """

    dS = 0.05 * S0_REF
    N = 4

    def y_short(omega: float) -> complex:
        # Very large admittance ~ near short.
        return complex(1.0e30, 0.0)

    Qn_local = solve_harmonic_coefficients(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, N,
    )
    Qn_lift = solve_harmonic_coefficients(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, N,
        y_stub_fn=y_short,
    )
    # Within FP noise, the lifted Q_n should match the local Q_n.
    np.testing.assert_allclose(Qn_lift, Qn_local, rtol=1e-10, atol=1e-12)


def test_lifted_solver_nonfinite_admittance_treated_as_short() -> None:
    """``y_stub_fn`` returning ``complex(inf, 0)`` -> diagonal contribution zero.

    A non-finite admittance encodes a perfect short to ground -> Z_stub = 0
    and the diagonal augmentation is zero. The lifted matrix at the
    affected harmonics matches the local-RLC matrix exactly there.
    """

    dS = 0.05 * S0_REF
    N = 3

    def y_inf(omega: float) -> complex:
        return complex(float("inf"), 0.0)

    A_local, _ = build_stub_tridiagonal(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, N,
    )
    A_lift, _ = build_stub_tridiagonal(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, N,
        y_stub_fn=y_inf,
    )
    np.testing.assert_array_equal(A_local, A_lift)


# --------------------------------------------------------------------------
# Y_stub_at_freq — sanity against hand-evaluated single-segment short stub
# --------------------------------------------------------------------------


def test_Y_stub_at_freq_single_segment_short_stub_matches_z0_tan() -> None:
    """``Z_in_short = j*Z_0*tan(beta*l)`` for one segment (Eq 6 limit)."""

    seg = StubSegment(width_m=1.93e-3, length_m=5.0e-3)
    substrate = {"eps_r": 3.4, "h_m": 1.5e-3, "t_m": 17.5e-6}
    omega = 2.0 * math.pi * 3.1e9

    # Reach inside the implementation only to evaluate Z_0(f), v_phase(f)
    # at the test frequency the same way; this is a value-vs-formula
    # equivalence check.
    from tvtl_ads_builder_v2.analytical.floquet_stub import _z0_and_v_phase

    z0, v_phase = _z0_and_v_phase(seg.width_m, substrate, omega / (2.0 * math.pi))
    beta = omega / v_phase
    expected_Z = 1j * z0 * math.tan(beta * seg.length_m)

    y_observed = Y_stub_at_freq(omega, [seg], substrate, termination="short")
    z_observed = 1.0 / y_observed
    assert z_observed.real == pytest.approx(expected_Z.real, abs=1e-9)
    assert z_observed.imag == pytest.approx(expected_Z.imag, rel=1e-9)


def test_Y_stub_at_freq_rejects_missing_segments() -> None:
    substrate = {"eps_r": 3.4, "h_m": 1.5e-3}
    with pytest.raises(ValueError, match="stub_segments"):
        Y_stub_at_freq(2.0 * math.pi * 3.0e9, [], substrate)


def test_Y_stub_at_freq_dc_short_returns_infinite_admittance() -> None:
    """DC limit of a short-terminated stub: Z_in = 0 -> Y = inf."""

    seg = StubSegment(width_m=1.93e-3, length_m=5.0e-3)
    substrate = {"eps_r": 3.4, "h_m": 1.5e-3}
    y = Y_stub_at_freq(0.0, [seg], substrate, termination="short")
    assert not math.isfinite(abs(y))


def test_Y_stub_at_freq_handles_negative_omega() -> None:
    """Floquet harmonic n*wm > ws produces a negative omega; Y must be defined."""

    seg = StubSegment(width_m=1.93e-3, length_m=5.0e-3)
    substrate = {"eps_r": 3.4, "h_m": 1.5e-3}
    omega = 2.0 * math.pi * 3.1e9
    y_pos = Y_stub_at_freq(+omega, [seg], substrate)
    y_neg = Y_stub_at_freq(-omega, [seg], substrate)
    # tan(beta*l) is odd in beta; Z_in = j*Z0*tan is odd in omega; Y is
    # also odd in omega.
    assert y_neg.imag == pytest.approx(-y_pos.imag, rel=1e-9)


def test_Y_stub_at_freq_rejects_unknown_termination() -> None:
    seg = StubSegment(width_m=1.93e-3, length_m=5.0e-3)
    substrate = {"eps_r": 3.4, "h_m": 1.5e-3}
    with pytest.raises(ValueError, match="termination"):
        Y_stub_at_freq(2.0 * math.pi * 3.0e9, [seg], substrate, termination="weird")


# --------------------------------------------------------------------------
# Spec-driven extraction on the Level-9 canonical
# --------------------------------------------------------------------------


_SPEC_L9_2CELL = LEVEL_9_CANONICAL_CIRCUITS[0]


def test_extract_level9_stub_segments_returns_single_5mm_mlin() -> None:
    """Level-9 has ONE 1.93 mm × 5 mm short-terminated stub per cell."""

    segments, substrate, termination = extract_level9_stub_segments(
        _SPEC_L9_2CELL, 0,
    )
    assert termination == "short"
    assert len(segments) == 1
    assert segments[0].width_m == pytest.approx(1.93e-3)
    assert segments[0].length_m == pytest.approx(5.0e-3)
    assert substrate["eps_r"] == pytest.approx(3.4)
    assert substrate["h_m"] == pytest.approx(1.5e-3)


def test_extract_level9_stub_segments_rejects_out_of_range_cell() -> None:
    with pytest.raises(IndexError, match="cell_index"):
        extract_level9_stub_segments(_SPEC_L9_2CELL, 99)


def test_build_stub_y_fn_from_spec_returns_callable_at_omega() -> None:
    """Spec-built callable should match hand-built Y_stub for the same geometry."""

    yfn = build_stub_y_fn_from_spec(_SPEC_L9_2CELL, 0)
    omega = 2.0 * math.pi * 3.1e9
    y_from_spec = yfn(omega)

    seg = StubSegment(width_m=1.93e-3, length_m=5.0e-3)
    substrate = {"eps_r": 3.4, "h_m": 1.5e-3, "t_m": 17.5e-6,
                 "tan_delta": 0.003, "sigma_S_per_m": 58.0e6,
                 "rough_m": 0.0, "mur": 1.0}
    y_direct = Y_stub_at_freq(omega, [seg], substrate, termination="short")

    assert y_from_spec.real == pytest.approx(y_direct.real, abs=1e-12)
    assert y_from_spec.imag == pytest.approx(y_direct.imag, abs=1e-12)


# --------------------------------------------------------------------------
# End-to-end: solver with spec-built lift narrows |I[n=0]| toward the divider
# --------------------------------------------------------------------------


def test_lifted_solver_reduces_n0_current_relative_to_local() -> None:
    """The lift should reduce |I[n=0]| in line with Z_var/(Z_var + Z_stub).

    At 3.1 GHz on the Level-9 geometry the static analysis predicts
    ``20*log10(|Z_var|/(|Z_var| + |Z_stub|)) ≈ -16 dB``; the lifted
    Floquet solver should reduce |I_n_local| by a similar order at
    n = 0 (the divider is approximate because Z_var and Z_stub are
    complex and at different reactance signs; the lift is the exact
    statement).
    """

    yfn = build_stub_y_fn_from_spec(_SPEC_L9_2CELL, 0)
    dS = 0.05 * S0_REF
    N = 5
    Qn_local = solve_harmonic_coefficients(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, N,
    )
    Qn_lift = solve_harmonic_coefficients(
        S0_REF, dS, 0.0, R_REF, L_REF, WS_REF, WM_REF, N,
        y_stub_fn=yfn,
    )
    _, I_local = current_spectrum(Qn_local, WS_REF, WM_REF, N)
    _, I_lift = current_spectrum(Qn_lift, WS_REF, WM_REF, N)
    # n = 0 is at index N.
    i_local_n0 = abs(I_local[N])
    i_lift_n0 = abs(I_lift[N])
    # The lift must REDUCE the predicted current at n = 0 (a series
    # impedance reduces current at the resonant element). The
    # magnitude reduction should land roughly between 6 dB and 25 dB
    # — large enough to be unambiguous, narrow enough to catch
    # accidental sign flips or magnitude bugs in the lift.
    reduction_dB = 20.0 * math.log10(i_lift_n0 / i_local_n0)
    assert -25.0 < reduction_dB < -6.0, (
        f"lift reduction at n=0 = {reduction_dB:.3f} dB "
        f"(expected -25..-6 dB on Level-9)"
    )


