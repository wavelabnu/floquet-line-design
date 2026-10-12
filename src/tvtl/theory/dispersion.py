"""Bloch dispersion extraction from a scalar (2x2) unit-cell ABCD.

For a passive, reciprocal, two-port unit cell of length ``period``, the
Bloch-Floquet theorem says the per-cell propagation constant ``gamma``
satisfies

    cosh(gamma * period) = (A + D) / 2

where ``(A, B, C, D)`` are the unit cell's ABCD entries (per ABCD
matrix convention: ``[[A, B], [C, D]]``). The propagation constant is
complex; its real part is the attenuation ``alpha`` per unit length
and its imaginary part is the Bloch wavenumber ``beta``.

This is the closed-form **trace-arccosh** extraction strategy declared
as :data:`tvtl.config.BlochExtractionStrategy.TRACE_ARCCOSH`. The
harmonic-basis equivalent for time-modulated unit cells lives at
:func:`tvtl.floquet.floquet_exponents` (the
:data:`~tvtl.config.BlochExtractionStrategy.EIGENVALUE_LOG` strategy):
that one operates on the (2N+1)x(2N+1) harmonic-extended ABCD and
returns one ``gamma`` per harmonic. This module's
:func:`bloch_dispersion` operates on the scalar passive cell and
returns a single ``gamma``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import BlochExtractionStrategy


def bloch_dispersion(
    abcd: tuple[Any, Any, Any, Any],
    period: float,
    *,
    strategy: BlochExtractionStrategy = BlochExtractionStrategy.TRACE_ARCCOSH,
) -> NDArray[np.complex128]:
    """Compute the Bloch propagation constant ``gamma`` of a unit cell.

    Given the ABCD entries ``(A, B, C, D)`` of a single unit cell and
    the cell length ``period`` in metres, returns the per-cell
    propagation constant ``gamma`` where
    ``cosh(gamma * period) = (A + D) / 2``. The real part of ``gamma``
    is the per-unit-length attenuation; the imaginary part is the
    Bloch wavenumber ``beta``.

    Parameters
    ----------
    abcd : tuple of (A, B, C, D)
        The unit cell's ABCD entries. Each may be a scalar or a
        frequency-indexed array. Broadcasting is NumPy-standard.
    period : float
        The unit cell length in metres. Must be positive.
    strategy : BlochExtractionStrategy, optional
        Extraction strategy. Only
        :data:`~tvtl.config.BlochExtractionStrategy.TRACE_ARCCOSH` is
        implemented for the scalar (2x2) case; the harmonic-basis
        :data:`~tvtl.config.BlochExtractionStrategy.EIGENVALUE_LOG` is
        handled by :func:`tvtl.floquet.floquet_exponents` for the
        (2N+1)x(2N+1) time-modulated unit cell and raises here.

    Returns
    -------
    gamma : ndarray of complex128
        The complex propagation constant ``gamma = alpha + 1j * beta``
        per unit length, broadcast against the shape of ``A + D``.

    Raises
    ------
    ValueError
        If ``period`` is not strictly positive.
    NotImplementedError
        If ``strategy`` is :data:`EIGENVALUE_LOG`; that path is the
        harmonic-basis :func:`tvtl.floquet.floquet_exponents`.

    See Also
    --------
    tvtl.floquet.floquet_exponents :
        The harmonic-basis Bloch extraction for a time-modulated unit
        cell.
    tvtl.theory.repeat_unit_cell :
        Build an N-cell cascade ABCD from a single unit-cell ABCD.
    """

    if not period > 0.0:
        raise ValueError(f"period must be strictly positive (got {period!r}).")
    if strategy == BlochExtractionStrategy.EIGENVALUE_LOG:
        raise NotImplementedError(
            "EIGENVALUE_LOG is the harmonic-basis strategy implemented by "
            "tvtl.floquet.floquet_exponents; for the scalar passive cell "
            "use TRACE_ARCCOSH (the default)."
        )
    if strategy != BlochExtractionStrategy.TRACE_ARCCOSH:
        raise ValueError(f"unknown Bloch extraction strategy: {strategy!r}")

    A, _, _, D = abcd
    half_trace = (np.asarray(A, dtype=np.complex128)
                  + np.asarray(D, dtype=np.complex128)) / 2.0
    return np.asarray(np.arccosh(half_trace) / period, dtype=np.complex128)


