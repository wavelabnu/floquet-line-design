"""Block-ABCD primitives in the Floquet harmonic basis.

Each harmonic ``n`` has angular frequency ``omega_n = ws + n * wm``. Linear
(time-invariant) network elements act independently on every harmonic, so
their ABCD parametrisation in the harmonic basis is block-diagonal: each of
``A, B, C, D`` is a ``(2N+1) x (2N+1)`` diagonal matrix. A time-varying
shunt branch couples harmonics through its Floquet Green's function and
contributes a full (generally non-diagonal) ``Y_stub`` block to the ``C``
block of a shunt ABCD.

This module is generic. It contains no example IDs and emits no device-gain
claims; every gain-bearing result is labelled ``"analytical_proxy_only"``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .harmonics import build_stub_tridiagonal


def harmonic_frequencies(
    ws: float,
    wm: float,
    N: int,
) -> np.ndarray:
    """Return the angular frequencies ``omega_n = ws + n*wm`` for ``n=-N..+N``.

    Parameters
    ----------
    ws : float
        Signal angular frequency [rad/s]. Must be positive.
    wm : float
        Modulation angular frequency [rad/s]. Must be positive.
    N : int
        Truncation order. Must be non-negative.

    Returns
    -------
    numpy.ndarray
        Real array of shape ``(2*N+1,)`` with ``omega_n`` for
        ``n = -N, ..., 0, ..., +N``.
    """

    omega_signal = float(ws)
    omega_modulation = float(wm)
    if omega_signal <= 0.0 or omega_modulation <= 0.0:
        raise ValueError("ws and wm must be positive.")
    if int(N) < 0:
        raise ValueError("N must be non-negative.")
    indices = np.arange(-int(N), int(N) + 1, dtype=float)
    return omega_signal + indices * omega_modulation


def _block_diag(values: np.ndarray) -> np.ndarray:
    """Promote a 1-D array to a diagonal complex matrix."""

    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError("values must be 1-D.")
    return np.diag(array)


def tl_section_harmonic_abcd(
    length: float,
    Z0: float,
    v_phase: float,
    ws: float,
    wm: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Block-diagonal ABCD of a lossless series transmission line section.

    Each harmonic ``n`` propagates independently with ``beta_n = omega_n / v_phase``.
    The per-harmonic 2-by-2 ABCD is the standard lossless TL form
    ``[[cosh(theta), Z0*sinh(theta)], [sinh(theta)/Z0, cosh(theta)]]`` evaluated
    at ``theta = j * beta_n * length``. The harmonic-basis blocks are therefore
    diagonal in the ``(2N+1)`` harmonic index.

    Parameters
    ----------
    length : float
        Section length [m]. Must be non-negative.
    Z0 : float
        Characteristic impedance [Ohm]. Must be positive.
    v_phase : float
        Phase velocity [m/s]. Must be positive.
    ws, wm : float
        Signal and modulation angular frequencies [rad/s].
    N : int
        Truncation order.

    Returns
    -------
    tuple of numpy.ndarray
        ``(A, B, C, D)``, each of shape ``(2*N+1, 2*N+1)`` and complex,
        all diagonal in the harmonic basis.
    """

    section_length = float(length)
    char_impedance = float(Z0)
    phase_velocity = float(v_phase)
    if section_length < 0.0:
        raise ValueError("length must be non-negative.")
    if char_impedance <= 0.0:
        raise ValueError("Z0 must be positive.")
    if phase_velocity <= 0.0:
        raise ValueError("v_phase must be positive.")
    omega = harmonic_frequencies(ws, wm, N)
    theta = 1j * (omega / phase_velocity) * section_length
    cosh = np.cosh(theta)
    sinh = np.sinh(theta)
    A = _block_diag(cosh)
    B = _block_diag(char_impedance * sinh)
    C = _block_diag(sinh / char_impedance)
    D = _block_diag(cosh)
    return A, B, C, D


def stub_green_function(
    S0: float,
    dS: float,
    phi: float,
    R: float,
    L: float,
    ws: float,
    wm: float,
    N: int,
) -> np.ndarray:
    """Floquet Green's function of the shunt-stub RLC tridiagonal system.

    ``G = inv(A_tridiag)``. The matrix ``A_tridiag`` is the one built by
    :func:`tvtl.floquet.harmonics.build_stub_tridiagonal` so that
    ``A_tridiag @ Q = V_drive``.

    Returns
    -------
    numpy.ndarray
        Complex matrix of shape ``(2*N+1, 2*N+1)``.
    """

    matrix, _ = build_stub_tridiagonal(S0, dS, phi, R, L, ws, wm, N)
    green = np.linalg.solve(matrix, np.eye(matrix.shape[0], dtype=np.complex128))
    if not np.all(np.isfinite(green)):
        raise FloatingPointError("Floquet Green's function contains non-finite values.")
    return green


def stub_harmonic_admittance(
    S0: float,
    dS: float,
    phi: float,
    R: float,
    L: float,
    ws: float,
    wm: float,
    N: int,
) -> np.ndarray:
    """Harmonic-basis admittance matrix of the time-varying RLC stub.

    ``Y_stub[n, m] = j*(ws + n*wm) * G[n, m]``. The diagonal entries reduce to
    the per-harmonic series-RLC admittance ``1/(R + j*omega_n*L - j/(omega_n*C0))``
    in the static limit ``dS = 0``.

    Returns
    -------
    numpy.ndarray
        Complex matrix of shape ``(2*N+1, 2*N+1)``.
    """

    green = stub_green_function(S0, dS, phi, R, L, ws, wm, N)
    omega = harmonic_frequencies(ws, wm, N)
    return 1j * omega[:, np.newaxis] * green


def stub_harmonic_abcd(
    S0: float,
    dS: float,
    phi: float,
    R: float,
    L: float,
    ws: float,
    wm: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic-basis ABCD of the time-varying shunt stub.

    The blocks are ``A = D = I_(2N+1)``, ``B = 0``, ``C = Y_stub``, which is
    the block generalisation of the single-frequency shunt-admittance ABCD.

    Returns
    -------
    tuple of numpy.ndarray
        ``(A, B, C, D)``, each of shape ``(2*N+1, 2*N+1)`` and complex.
    """

    size = 2 * int(N) + 1
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    Y = stub_harmonic_admittance(S0, dS, phi, R, L, ws, wm, N)
    return identity, zero, Y, identity


def _capacitance_toeplitz(
    C0: float, depth: float, phi: float, size: int
) -> np.ndarray:
    """Harmonic Toeplitz of ``C(t) = C0*(1 + depth*cos(wm*t + phi))``.

    The capacitance Fourier coefficients are ``C_0 = C0`` and
    ``C_{+/-1} = 0.5*depth*C0*exp(+/- j*phi)``; the matrix entry
    ``[n, m] = C_{n-m}`` is tridiagonal (nearest-neighbour coupling).
    """

    matrix = np.zeros((size, size), dtype=np.complex128)
    c_plus = 0.5 * depth * C0 * np.exp(1j * phi)   # C_{+1}
    c_minus = 0.5 * depth * C0 * np.exp(-1j * phi)  # C_{-1}
    for row in range(size):
        matrix[row, row] = C0
        if row + 1 < size:
            matrix[row, row + 1] = c_minus  # n - m = -1
        if row - 1 >= 0:
            matrix[row, row - 1] = c_plus   # n - m = +1
    return matrix


def tv_capacitor_harmonic_admittance(
    C0: float,
    depth: float,
    phi: float,
    ws: float,
    wm: float,
    N: int,
    *,
    z_stub_fn: Callable[[float], complex] | None = None,
) -> np.ndarray:
    """Harmonic-basis admittance of an ideal *linear* time-varying capacitor.

    The capacitor is **capacitance-linear**: ``C(t) = C0*(1 + depth*cos(wm*t +
    phi))`` with the charge constitutive law ``Q(t) = C(t)*v(t)`` (so the
    branch current is ``i = dQ/dt``). This is the model an externally-pumped
    ideal capacitor realises (e.g. a control-port-driven SDD whose port-1
    charge is ``C0*v1 + (dC/Vp)*v2*v1``). It is **distinct** from the
    *elastance*-linear :func:`stub_harmonic_admittance`
    (``S(t) = S0*(1 + depth*cos)``); the two agree only to first order in
    ``depth`` and diverge at deep modulation -- so cross-validating an ideal-C
    ADS device requires *this* function, not the elastance solver.

    Bare capacitor (``z_stub_fn=None``)::

        Y[n, m] = j*omega_n * C_{n-m}

    a closed form needing no matrix inverse (``C_{n-m}`` is the Toeplitz of
    the capacitance waveform, :func:`_capacitance_toeplitz`).

    Capacitor in series with a linear stub (``z_stub_fn`` given). With the
    per-harmonic stub impedance ``Z_stub(omega_n) = 1/y_stub_fn`` diagonal in
    the harmonic basis, the branch admittance seen at the tap is

        Y = j*W * Ctoep * ( I + diag(j*omega_n * Z_stub(omega_n)) * Ctoep )^{-1}

    with ``W = diag(omega_n)``. The ``z_stub_fn=None`` form is the
    ``Z_stub = 0`` limit of this expression. The stub touches only this
    linear post-multiply; the capacitor's harmonic coupling (the Toeplitz
    off-diagonals) is unchanged.

    Parameters
    ----------
    C0 : float
        Static capacitance [F]. Must be positive.
    depth : float
        Modulation depth ``dC/C0`` [dimensionless]. Must satisfy
        ``0 <= depth <= 1`` so ``C(t)`` stays positive.
    phi : float
        Modulation phase [rad].
    ws, wm : float
        Signal and modulation angular frequencies [rad/s]. Must be positive.
    N : int
        Truncation order; the matrix is ``(2*N+1)``-square.
    z_stub_fn : Callable[[float], complex], optional
        ``omega -> Z_stub(omega)`` giving the per-harmonic series impedance of
        a passive network in series with the capacitor (e.g. a stub-MLIN
        cascade + via to ground). When ``None`` (default) the capacitor is a
        bare shunt. Note this takes the stub *impedance*; to reuse the v2
        admittance helper
        :func:`tvtl_ads_builder_v2.analytical.build_stub_y_fn_from_spec`
        wrap it as ``lambda w: 1.0 / y_stub_fn(w)``.

    Returns
    -------
    numpy.ndarray
        Complex ``(2*N+1, 2*N+1)`` admittance matrix.
    """

    c0 = float(C0)
    m = float(depth)
    if c0 <= 0.0:
        raise ValueError("C0 must be positive.")
    if m < 0.0 or m > 1.0:
        raise ValueError("depth must satisfy 0 <= depth <= 1 (C(t) > 0).")
    omega = harmonic_frequencies(ws, wm, N)
    size = omega.shape[0]
    ctoep = _capacitance_toeplitz(c0, m, float(phi), size)
    jw = 1j * omega
    if z_stub_fn is None:
        return jw[:, np.newaxis] * ctoep
    z_stub = np.array(
        [complex(z_stub_fn(float(w))) for w in omega], dtype=np.complex128
    )
    if not np.all(np.isfinite(z_stub)):
        raise FloatingPointError("z_stub_fn returned a non-finite impedance.")
    coupling = np.diag(jw * z_stub) @ ctoep
    m_matrix = np.eye(size, dtype=np.complex128) + coupling
    green = np.linalg.solve(m_matrix, np.eye(size, dtype=np.complex128))
    admittance = (jw[:, np.newaxis] * ctoep) @ green
    if not np.all(np.isfinite(admittance)):
        raise FloatingPointError(
            "time-varying-capacitor branch admittance is non-finite."
        )
    return admittance


def tv_capacitor_harmonic_abcd(
    C0: float,
    depth: float,
    phi: float,
    ws: float,
    wm: float,
    N: int,
    *,
    z_stub_fn: Callable[[float], complex] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shunt-branch ABCD ``(I, 0, Y, I)`` for the ideal time-varying capacitor.

    ``Y`` is :func:`tv_capacitor_harmonic_admittance`. See it for the
    capacitance-linear model and the optional series stub.
    """

    size = 2 * int(N) + 1
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    Y = tv_capacitor_harmonic_admittance(
        C0, depth, phi, ws, wm, N, z_stub_fn=z_stub_fn
    )
    return identity, zero, Y, identity


def _capacitance_toeplitz_from_coeffs(
    c_coeffs: np.ndarray, size: int
) -> np.ndarray:
    """Harmonic Toeplitz of a general periodic ``C(t)`` from its one-sided
    Fourier coefficients.

    ``c_coeffs`` is the one-sided spectrum ``[C_0, C_1, ..., C_K]`` of a
    real, even-extended capacitance waveform ``C(t) = sum_d C_d e^{i*d*wm*t}``
    with the Hermitian closure ``C_{-d} = conj(C_d)``. The matrix entry
    ``[n, m] = C_{n-m}`` (bands ``|n - m| <= K``, zero beyond). This
    generalises :func:`_capacitance_toeplitz` (which hard-codes the
    nearest-neighbour cosine ``[C_0, C_1]``); the two produce a byte-identical
    matrix for ``c_coeffs = [C0, 0.5*depth*C0*exp(i*phi)]`` because the
    placement and the Hermitian closure match exactly.
    """

    coeffs = np.asarray(c_coeffs, dtype=np.complex128)
    if coeffs.ndim != 1 or coeffs.shape[0] < 1:
        raise ValueError("c_coeffs must be a non-empty 1-D sequence.")
    order = coeffs.shape[0] - 1
    matrix = np.zeros((size, size), dtype=np.complex128)
    for row in range(size):
        for col in range(size):
            d = row - col
            if d == 0:
                matrix[row, col] = coeffs[0]
            elif 0 < d <= order:
                matrix[row, col] = coeffs[d]
            elif -order <= d < 0:
                matrix[row, col] = np.conj(coeffs[-d])
    return matrix


def tv_capacitor_harmonic_admittance_general(
    c_coeffs,
    ws: float,
    wm: float,
    N: int,
    *,
    z_stub_fn: Callable[[float], complex] | None = None,
) -> np.ndarray:
    """Harmonic-basis admittance of a *general* periodic time-varying capacitor.

    The capacitance waveform is given by its one-sided Fourier spectrum
    ``c_coeffs = [C_0, C_1, ..., C_K]`` (``C(t) = sum_d C_d e^{i*d*wm*t}``,
    ``C_{-d} = conj(C_d)``) rather than the single-tone ``C0*(1 + depth*cos)``
    of :func:`tv_capacitor_harmonic_admittance`. This carries the higher-order
    C-V content a realistic varactor produces even from a sinusoidal pump
    (the ``C_{|d|>1}`` bands). The constitutive law is still
    capacitance-linear (``Q = C(t)*v``).

    The admittance formulae are *identical* to
    :func:`tv_capacitor_harmonic_admittance` -- bare
    ``Y = jW * Ctoep``; with a series stub
    ``Y = jW * Ctoep * (I + diag(jW*Z_stub) * Ctoep)^{-1}`` -- only the
    Toeplitz is the general one. Consequently, passing
    ``c_coeffs = [C0, 0.5*depth*C0*exp(i*phi)]`` reproduces
    :func:`tv_capacitor_harmonic_admittance` to machine precision (the band
    placement, Hermitian closure, ``jW`` axis and matrix ops all match).

    Parameters
    ----------
    c_coeffs : sequence of complex
        One-sided capacitance Fourier coefficients ``[C_0, C_1, ...]`` [F].
        ``C_0`` must be real-positive (the static capacitance); higher bands
        are arbitrary complex.
    ws, wm : float
        Signal and modulation angular frequencies [rad/s]. Must be positive.
    N : int
        Truncation order; the matrix is ``(2*N+1)``-square.
    z_stub_fn : Callable[[float], complex], optional
        ``omega -> Z_stub(omega)`` per-harmonic series stub impedance below
        the capacitor. ``None`` (default) is the bare-shunt ``Z_stub = 0``
        limit. (To reuse a v2 admittance helper, wrap it as
        ``lambda w: 1.0 / y_stub_fn(w)``.)

    Returns
    -------
    numpy.ndarray
        Complex ``(2*N+1, 2*N+1)`` admittance matrix.
    """

    coeffs = np.asarray(c_coeffs, dtype=np.complex128)
    if coeffs.ndim != 1 or coeffs.shape[0] < 1:
        raise ValueError("c_coeffs must be a non-empty 1-D sequence.")
    c0 = complex(coeffs[0])
    if not (c0.real > 0.0 and abs(c0.imag) <= 1e-30):
        raise ValueError(
            f"c_coeffs[0] (static C_0) must be real-positive (got {c0})."
        )
    omega = harmonic_frequencies(ws, wm, N)
    size = omega.shape[0]
    ctoep = _capacitance_toeplitz_from_coeffs(coeffs, size)
    jw = 1j * omega
    if z_stub_fn is None:
        return jw[:, np.newaxis] * ctoep
    z_stub = np.array(
        [complex(z_stub_fn(float(w))) for w in omega], dtype=np.complex128
    )
    if not np.all(np.isfinite(z_stub)):
        raise FloatingPointError("z_stub_fn returned a non-finite impedance.")
    coupling = np.diag(jw * z_stub) @ ctoep
    m_matrix = np.eye(size, dtype=np.complex128) + coupling
    green = np.linalg.solve(m_matrix, np.eye(size, dtype=np.complex128))
    admittance = (jw[:, np.newaxis] * ctoep) @ green
    if not np.all(np.isfinite(admittance)):
        raise FloatingPointError(
            "time-varying-capacitor branch admittance is non-finite."
        )
    return admittance


def _multiply_blocks(
    left: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multiply two harmonic block ABCDs in the conventional ABCD order."""

    a1, b1, c1, d1 = left
    a2, b2, c2, d2 = right
    return (
        a1 @ a2 + b1 @ c2,
        a1 @ b2 + b1 @ d2,
        c1 @ a2 + d1 @ c2,
        c1 @ b2 + d1 @ d2,
    )


def cascade_harmonic_abcd(
    abcd_list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cascade a sequence of harmonic block ABCDs by block matrix multiply.

    Parameters
    ----------
    abcd_list : iterable of 4-tuples
        Each element is ``(A, B, C, D)`` with matching square shape.

    Returns
    -------
    tuple of numpy.ndarray
        Total ``(A, B, C, D)``.
    """

    items = list(abcd_list)
    if not items:
        raise ValueError("abcd_list must contain at least one block ABCD.")
    size = items[0][0].shape[0]
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = (identity, zero, zero, identity)
    for block in items:
        result = _multiply_blocks(result, block)
    if not np.all([np.all(np.isfinite(component)) for component in result]):
        raise FloatingPointError("Harmonic ABCD cascade produced non-finite values.")
    return result


def unit_cell_harmonic_abcd(
    tl1: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    stub: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tl2: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cascade the three block ABCDs that compose one TVTL unit cell.

    The conventional order is series TL, shunt stub, series TL.
    """

    return cascade_harmonic_abcd([tl1, stub, tl2])


def _pack_full_matrix(
    block: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    a, b, c, d = block
    top = np.concatenate([a, b], axis=1)
    bottom = np.concatenate([c, d], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _unpack_full_matrix(
    matrix: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = matrix[:block_size, :block_size]
    b = matrix[:block_size, block_size:]
    c = matrix[block_size:, :block_size]
    d = matrix[block_size:, block_size:]
    return a, b, c, d


def n_cells_harmonic_abcd(
    unit_cell_abcd: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cascade ``n_cells`` identical unit cells via a block-matrix power.

    Parameters
    ----------
    unit_cell_abcd : tuple of numpy.ndarray
        ``(A, B, C, D)`` for one unit cell.
    n_cells : int
        Non-negative number of repeated cells. ``n_cells = 0`` returns the
        block identity.

    Returns
    -------
    tuple of numpy.ndarray
        Total ``(A, B, C, D)`` after the cascade.
    """

    count = int(n_cells)
    if count < 0:
        raise ValueError("n_cells must be non-negative.")
    size = unit_cell_abcd[0].shape[0]
    if count == 0:
        identity = np.eye(size, dtype=np.complex128)
        zero = np.zeros((size, size), dtype=np.complex128)
        return identity, zero, zero, identity
    full = _pack_full_matrix(unit_cell_abcd)
    powered = np.linalg.matrix_power(full, count)
    if not np.all(np.isfinite(powered)):
        raise FloatingPointError("Repeated unit-cell harmonic cascade produced non-finite values.")
    return _unpack_full_matrix(np.asarray(powered, dtype=np.complex128), size)


def harmonic_s_parameters(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    Z_source: float,
    Z_load: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Block ABCD -> ``(S11, S21)`` with scalar source/load impedances.

    The block generalisation of the standard ABCD -> S conversion is

        denom = A + B / Z_L + C * Z_S + D * (Z_S / Z_L)
        S11   = (A + B / Z_L - C * Z_S - D * (Z_S / Z_L)) @ inv(denom)
        S21   = 2 * sqrt(Z_S / Z_L) * inv(denom).

    With ``Z_S = Z_L = Z_0`` the source/load terms reduce to identity matrices
    scaled by ``Z_0`` or ``1/Z_0``.

    Parameters
    ----------
    A, B, C, D : numpy.ndarray
        Block matrices of identical square shape.
    Z_source, Z_load : float
        Reference impedances [Ohm]; must be positive.

    Returns
    -------
    tuple of numpy.ndarray
        ``(S11, S21)``, each of the same shape as the input blocks.
    """

    z_s = float(Z_source)
    z_l = float(Z_load)
    if z_s <= 0.0 or z_l <= 0.0:
        raise ValueError("Z_source and Z_load must be positive.")
    a = np.asarray(A, dtype=np.complex128)
    b = np.asarray(B, dtype=np.complex128)
    c = np.asarray(C, dtype=np.complex128)
    d = np.asarray(D, dtype=np.complex128)
    if not (a.shape == b.shape == c.shape == d.shape and a.shape[0] == a.shape[1]):
        raise ValueError("A, B, C, D must be square matrices of identical shape.")
    denom = a + b / z_l + c * z_s + d * (z_s / z_l)
    numerator_11 = a + b / z_l - c * z_s - d * (z_s / z_l)
    inverse_denom = np.linalg.solve(denom, np.eye(denom.shape[0], dtype=np.complex128))
    s11 = numerator_11 @ inverse_denom
    s21 = 2.0 * np.sqrt(z_s / z_l) * inverse_denom
    if not (np.all(np.isfinite(s11)) and np.all(np.isfinite(s21))):
        raise FloatingPointError("Harmonic S-parameter conversion produced non-finite values.")
    return s11, s21


def device_coherent_gain(
    S21_on: np.ndarray,
    S21_off: np.ndarray,
    N: int,
) -> dict:
    """Coherent-sum (physical / phase-sensitive) gain metrics for degenerate operation.

    For degenerate operation (``f_pump = 2*f_signal``) the signal sideband
    (mix ``(1,0)``, harmonic ``n=0``, block index ``N``) and the degenerate
    idler (mix ``(-1,+1)``, harmonic ``n=-1``, block index ``N-1``) land on the
    **same physical frequency** ``f_signal``. A spectrum analyzer there measures
    the **coherent sum**, whose gain relative to the pump-off signal is
    **phase-sensitive** (degenerate phase-sensitive amplification, PSA): as the
    signal-pump relative phase ``Phi`` is swept, the idler adds constructively or
    destructively, ``|H_s + H_i*e^{jPhi}| / |H_s,off|``.

    Definitions (``H_s = S21_on[N,N]``, ``H_i = S21_on[N-1,N]``,
    ``H_s,off = S21_off[N,N]``):

    * ``mix10_gain_dB``    = ``20*log10(|H_s| / |H_s,off|)`` -- the **phase-
      invariant** signal-sideband gain (what the historical framework metric
      reports; independent of pump phase).
    * ``coherent_peak_dB`` = ``20*log10((|H_s| + |H_i|) / |H_s,off|)`` -- the
      maximum-over-phase (best-quadrature) coherent gain.
    * ``coherent_min_dB``  = ``20*log10(||H_s| - |H_i|| / |H_s,off|)`` -- the
      minimum-over-phase (deamplification quadrature).
    * ``coherent_swing_dB`` = ``coherent_peak_dB - coherent_min_dB`` -- the PSA
      depth (peak-to-peak).
    * ``coherent_at_block_phase_dB`` = ``20*log10(|H_s + H_i| / |H_s,off|)`` --
      the coherent gain at the *block's* modulation phase (the phase baked into
      the supplied ``S21`` blocks; for the φ=0 default this is the zero-phase
      slice). **The absolute phase reference of a single slice is convention-
      dependent across simulators; the peak/min/swing are convention-free** (they
      use magnitudes only) and are the robust cross-validation metrics.
    * ``idler_over_signal`` = ``|H_i| / |H_s|``.

    All gain-bearing values carry the ``"analytical_proxy_only"`` label.

    Parameters
    ----------
    S21_on, S21_off : numpy.ndarray
        Pumped / unpumped block ``S21`` matrices, shape ``(2N+1, 2N+1)``.
    N : int
        Truncation order (>= 1; the idler at ``n=-1`` is index ``N-1``).
    """

    truncation = int(N)
    if truncation < 1:
        raise ValueError("N must be at least 1 to access the n=-1 idler harmonic.")
    on = np.asarray(S21_on, dtype=np.complex128)
    off = np.asarray(S21_off, dtype=np.complex128)
    expected = 2 * truncation + 1
    for name, m in (("S21_on", on), ("S21_off", off)):
        if m.shape != (expected, expected):
            raise ValueError(f"{name} must have shape ({expected}, {expected}); got {m.shape}.")
    h_s = on[truncation, truncation]
    h_i = on[truncation - 1, truncation]
    h_ref = off[truncation, truncation]
    tiny = np.finfo(float).tiny
    ms = abs(h_s)
    mi = abs(h_i)
    mref = max(abs(h_ref), tiny)
    return {
        "mix10_gain_dB": float(20.0 * np.log10(max(ms, tiny) / mref)),
        "coherent_peak_dB": float(20.0 * np.log10(max(ms + mi, tiny) / mref)),
        "coherent_min_dB": float(20.0 * np.log10(max(abs(ms - mi), tiny) / mref)),
        "coherent_swing_dB": float(
            20.0 * np.log10(max(ms + mi, tiny) / mref)
            - 20.0 * np.log10(max(abs(ms - mi), tiny) / mref)
        ),
        "coherent_at_block_phase_dB": float(20.0 * np.log10(max(abs(h_s + h_i), tiny) / mref)),
        "idler_over_signal": float(mi / ms) if ms > 0 else float("nan"),
        "label": "analytical_proxy_only",
    }


def device_proxy_gain(
    S21: np.ndarray,
    N: int,
) -> dict:
    """Compute the analytical proxy device-gain dict from the block ``S21``.

    Definitions, with index ``N`` representing harmonic ``n = 0`` in the
    ``(2N+1)``-length harmonic axis:

        gain_signal_proxy_dB = 20 * log10( |S21[N, N]| )
        gain_idler_proxy_dB  = 20 * log10( |S21[N-1, N]| )
        gain_total_proxy_dB  = 10 * log10( sum_k |S21[k, N]|^2 )

    All three values are analytical proxies only. The returned dict always
    carries ``"label": "analytical_proxy_only"`` so callers can propagate the
    disclaimer into downstream artifacts; this proxy is NOT equivalent to
    ADS-HB or device gain.

    Parameters
    ----------
    S21 : numpy.ndarray
        Block S21 matrix of shape ``(2N+1, 2N+1)``.
    N : int
        Truncation order; the array must have ``2*N+1`` rows and columns.

    Returns
    -------
    dict
        Keys ``"gain_signal_proxy_dB"``, ``"gain_idler_proxy_dB"``,
        ``"gain_total_proxy_dB"`` (all floats) and ``"label"``.
    """

    truncation = int(N)
    if truncation < 1:
        raise ValueError("N must be at least 1 to access the n = -1 idler harmonic.")
    matrix = np.asarray(S21, dtype=np.complex128)
    expected = 2 * truncation + 1
    if matrix.shape != (expected, expected):
        raise ValueError(f"S21 must have shape ({expected}, {expected}); got {matrix.shape}.")
    signal = abs(matrix[truncation, truncation])
    idler = abs(matrix[truncation - 1, truncation])
    total_power = float(np.sum(np.abs(matrix[:, truncation]) ** 2))
    tiny = np.finfo(float).tiny
    return {
        "gain_signal_proxy_dB": float(20.0 * np.log10(max(signal, tiny))),
        "gain_idler_proxy_dB": float(20.0 * np.log10(max(idler, tiny))),
        "gain_total_proxy_dB": float(10.0 * np.log10(max(total_power, tiny))),
        "label": "analytical_proxy_only",
    }


