"""Generic Floquet harmonic helpers for local proxy analysis.

This module also exposes the analytical tridiagonal Floquet solver for the
shunt stub model of Figure 3 in the paper (series RLC with a time-varying
elastance ``S(t) = S0 + dS * cos(wm*t + phi)`` driven by a harmonic source
``v(t) = V0 * exp(+j*ws*t)``). The solver is a generic numerical helper; it
contains no example-specific branches and makes no gain claims.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def harmonic_indices(truncation_order: int) -> np.ndarray:
    if isinstance(truncation_order, bool) or int(truncation_order) < 0:
        raise ValueError("truncation_order must be a non-negative integer.")
    return np.arange(-int(truncation_order), int(truncation_order) + 1, dtype=int)


def harmonic_frequencies(
    signal_angular_frequency_rad_s: float,
    modulation_angular_frequency_rad_s: float,
    truncation_order: int,
) -> np.ndarray:
    omega_signal = float(signal_angular_frequency_rad_s)
    omega_modulation = float(modulation_angular_frequency_rad_s)
    if omega_signal <= 0.0 or omega_modulation <= 0.0:
        raise ValueError("Floquet angular frequencies must be positive.")
    return np.asarray(omega_signal + harmonic_indices(truncation_order) * omega_modulation, dtype=float)


def sinusoidal_elastance_coefficients(
    average_elastance: float,
    modulation_elastance: float,
    phase_rad: float = 0.0,
) -> dict[int, complex]:
    s0 = float(average_elastance)
    delta = float(modulation_elastance)
    phase = float(phase_rad)
    if s0 <= 0.0:
        raise ValueError("average_elastance must be positive.")
    return {
        0: complex(s0),
        1: complex(0.5 * delta * np.exp(1j * phase)),
        -1: complex(0.5 * delta * np.exp(-1j * phase)),
    }


def elastance_coefficients_from_capacitance_waveform(
    capacitance_f: float,
    modulation_depth: float,
    max_harmonic: int,
    *,
    sample_count: int = 4096,
    phase_rad: float = 0.0,
) -> dict[int, complex]:
    c0 = float(capacitance_f)
    depth = float(modulation_depth)
    if c0 <= 0.0:
        raise ValueError("capacitance_f must be positive.")
    if abs(depth) >= 1.0:
        raise ValueError("modulation_depth must satisfy abs(depth) < 1.")
    if int(max_harmonic) < 0:
        raise ValueError("max_harmonic must be non-negative.")
    theta = np.linspace(0.0, 2.0 * np.pi, int(sample_count), endpoint=False)
    waveform = c0 * (1.0 + depth * np.cos(theta + float(phase_rad)))
    if np.any(waveform <= 0.0):
        raise ValueError("capacitance waveform must remain positive.")
    elastance = 1.0 / waveform
    fft = np.fft.fft(elastance) / int(sample_count)
    return {harmonic: complex(fft[harmonic % int(sample_count)]) for harmonic in range(-int(max_harmonic), int(max_harmonic) + 1)}


def build_stub_tridiagonal(
    S0: float,
    dS: float,
    phi: float,
    R: float,
    L: float,
    ws: float,
    wm: float,
    N: int,
    *,
    y_stub_fn: Callable[[float], complex] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the Floquet tridiagonal system for the shunt-stub RLC model.

    The governing ODE is ``L*Q''(t) + R*Q'(t) + S(t)*Q(t) = V0*exp(+j*ws*t)``
    with ``S(t) = S0 + dS*cos(wm*t + phi)``. Expanding the charge as
    ``Q(t) = sum_n Q_n * exp(j*(ws + n*wm)*t)`` yields, for each harmonic
    index ``n``,

        (dS * exp(+j*phi) / 2) * Q_{n-1}
        + [S0 + j*R*(ws + n*wm) - L*(ws + n*wm)**2] * Q_n
        + (dS * exp(-j*phi) / 2) * Q_{n+1}
        = V0 * delta_{n, 0}.

    Harmonic ABCD lift (optional ``y_stub_fn``)
    ------------------------------------------
    When ``y_stub_fn`` is provided, the solver represents a varactor in
    series with a passive stub network whose per-harmonic input
    admittance is ``y_stub_fn(omega_n)``. The diagonal is augmented as

        diag_n -> diag_n + j * omega_n * Z_stub(omega_n)
                where Z_stub(omega_n) = 1 / y_stub_fn(omega_n).

    The off-diagonal modulation coupling is unchanged because the
    time-varying capacitance is local to the varactor; the stub
    contributes only a per-harmonic series impedance that touches the
    diagonal. The default ``y_stub_fn=None`` preserves the existing
    local-RLC behaviour exactly (regression — covered by
    unit test).

    Parameters
    ----------
    S0 : float
        Static elastance ``1/C0`` [F^-1]. Must be positive.
    dS : float
        Modulation amplitude of the elastance [F^-1]. Must satisfy
        ``0 <= dS <= S0`` so that ``S(t)`` remains positive.
    phi : float
        Modulation phase [rad].
    R : float
        Series resistance [Ohm]. Must be non-negative.
    L : float
        Series inductance [H]. Must be non-negative.
    ws : float
        Signal angular frequency [rad/s]. Must be positive.
    wm : float
        Modulation angular frequency [rad/s]. Must be positive.
    N : int
        Truncation order. The system size is ``2*N + 1``.
    y_stub_fn : Callable[[float], complex], optional
        Per-harmonic stub admittance ``Y_stub(omega) = 1 / Z_stub(omega)``
        for the harmonic ABCD lift. When ``None`` (default)
        the system reduces exactly to the local-RLC form. The
        v2 helper
        :func:`tvtl_ads_builder_v2.analytical.build_stub_y_fn_from_spec`
        constructs an appropriate callable from a Level-9-style spec.

    Returns
    -------
    A : numpy.ndarray, shape ``(2*N+1, 2*N+1)``, complex
        Floquet coupling matrix.
    b : numpy.ndarray, shape ``(2*N+1,)``, complex
        Right-hand side vector with ``b[N] = V0 = 1`` (delta_{n,0}) and zeros
        elsewhere; downstream callers can rescale by their own ``V0``.
    """

    s0 = float(S0)
    delta_s = float(dS)
    phase = float(phi)
    resistance = float(R)
    inductance = float(L)
    omega_signal = float(ws)
    omega_modulation = float(wm)
    truncation = int(N)
    if s0 <= 0.0:
        raise ValueError("S0 must be positive.")
    if delta_s < 0.0 or delta_s > s0:
        raise ValueError("dS must satisfy 0 <= dS <= S0 to keep S(t) positive.")
    if resistance < 0.0 or inductance < 0.0:
        raise ValueError("R and L must be non-negative.")
    if omega_signal <= 0.0 or omega_modulation <= 0.0:
        raise ValueError("ws and wm must be positive.")
    if truncation < 0:
        raise ValueError("N must be non-negative.")

    size = 2 * truncation + 1
    indices = np.arange(-truncation, truncation + 1, dtype=float)
    omega_n = omega_signal + indices * omega_modulation
    diagonal = s0 + 1j * resistance * omega_n - inductance * omega_n * omega_n
    if y_stub_fn is not None:
        # Harmonic ABCD lift: add j*omega_n*Z_stub(omega_n) per harmonic.
        # The stub contributes only to the diagonal; the off-diagonal
        # modulation coupling is unchanged.
        stub_contrib = np.empty(size, dtype=np.complex128)
        for k in range(size):
            y_val = complex(y_stub_fn(float(omega_n[k])))
            y_real = y_val.real
            y_imag = y_val.imag
            # A non-finite admittance encodes a short to ground (|Y| =
            # inf -> Z_stub = 0 -> no contribution to the diagonal).
            if not (np.isfinite(y_real) and np.isfinite(y_imag)):
                stub_contrib[k] = 0.0 + 0.0j
                continue
            if abs(y_val) == 0.0:
                # Open stub at this harmonic -> infinite series Z; the
                # diagonal blows up and that harmonic's Q_n is driven to
                # zero, which is the physical answer.
                stub_contrib[k] = complex(float("inf"), 0.0)
            else:
                z_stub = 1.0 / y_val
                stub_contrib[k] = 1j * float(omega_n[k]) * z_stub
        diagonal = diagonal + stub_contrib
    sub = 0.5 * delta_s * np.exp(1j * phase)
    sup = 0.5 * delta_s * np.exp(-1j * phase)
    matrix = np.zeros((size, size), dtype=np.complex128)
    for row in range(size):
        matrix[row, row] = diagonal[row]
        if row >= 1:
            matrix[row, row - 1] = sub
        if row + 1 < size:
            matrix[row, row + 1] = sup
    rhs = np.zeros(size, dtype=np.complex128)
    rhs[truncation] = 1.0 + 0.0j
    return matrix, rhs


def solve_harmonic_coefficients(
    S0: float,
    dS: float,
    phi: float,
    R: float,
    L: float,
    ws: float,
    wm: float,
    N: int,
    *,
    y_stub_fn: Callable[[float], complex] | None = None,
) -> np.ndarray:
    """Solve the Floquet tridiagonal system and return the charge harmonics.

    Parameters are described in :func:`build_stub_tridiagonal`, including
    the optional ``y_stub_fn`` callable for the harmonic ABCD lift.

    Returns
    -------
    Qn : numpy.ndarray, shape ``(2*N+1,)``, complex
        Floquet charge coefficients ordered ``n = -N, ..., 0, ..., +N``.
        The ``n = 0`` entry is at index ``N``. The result is normalised to
        ``V0 = 1``; rescale by the caller's source amplitude to recover
        physical charges.
    """

    matrix, rhs = build_stub_tridiagonal(
        S0, dS, phi, R, L, ws, wm, N, y_stub_fn=y_stub_fn,
    )
    solution = np.linalg.solve(matrix, rhs)
    if not np.all(np.isfinite(solution)):
        raise FloatingPointError("Floquet tridiagonal solve produced non-finite values.")
    return solution


def current_spectrum(
    Qn: np.ndarray,
    ws: float,
    wm: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return harmonic angular frequencies and complex current amplitudes.

    Using ``i(t) = dQ/dt`` and the Floquet expansion, the spectrum is
    ``I_n = j * (ws + n*wm) * Q_n`` at the harmonic angular frequency
    ``ws + n*wm``.

    Parameters
    ----------
    Qn : numpy.ndarray
        Charge harmonic coefficients ordered ``n = -N, ..., +N``.
    ws : float
        Signal angular frequency [rad/s].
    wm : float
        Modulation angular frequency [rad/s].
    N : int
        Truncation order; the input array must have length ``2*N + 1``.

    Returns
    -------
    freqs : numpy.ndarray, shape ``(2*N+1,)``, float
        Harmonic angular frequencies ``ws + n*wm``.
    I_amps : numpy.ndarray, shape ``(2*N+1,)``, complex
        Complex current amplitudes ``j*(ws + n*wm)*Q_n``.
    """

    truncation = int(N)
    if truncation < 0:
        raise ValueError("N must be non-negative.")
    coefficients = np.asarray(Qn, dtype=np.complex128)
    expected = 2 * truncation + 1
    if coefficients.shape != (expected,):
        raise ValueError(f"Qn must have length 2*N+1 = {expected}; got {coefficients.shape}.")
    indices = np.arange(-truncation, truncation + 1, dtype=float)
    freqs = float(ws) + indices * float(wm)
    return freqs, 1j * freqs * coefficients


