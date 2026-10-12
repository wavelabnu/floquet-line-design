"""Proxy stability checks and Floquet eigenvalue analysis.

The dataclass-based ``assess_stability_proxy`` is the legacy proxy screen
used in candidate ranking. The functions below it implement Bloch-mode
eigenvalue analysis of the harmonic-basis unit-cell ABCD matrix
``M_uc = [[A, B], [C, D]]`` produced by
:mod:`tvtl.floquet.harmonic_abcd`. Their physical interpretation:

* Each eigenvalue ``lambda_k`` is the per-cell multiplier of a Bloch mode.
* ``max_k |lambda_k| = 1`` is the parametric instability threshold.
* ``max_k |lambda_k| > 1`` indicates parametric gain above threshold.
* Cascade amplification of mode ``k`` over ``n_cells`` cells is
  ``|lambda_k|^n_cells``.

All gain or stability-margin numbers returned by helpers in this module are
analytical proxies only and carry the label ``"analytical_proxy_only"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class StabilityAssessment:
    stable_by_proxy: bool
    full_device_gain_available: bool = False
    condition_number: float | None = None
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def assess_stability_proxy(
    *,
    condition_number: float | None = None,
    max_condition_number: float = 1.0e10,
    extra_reasons: tuple[str, ...] = (),
) -> StabilityAssessment:
    reasons: list[str] = list(extra_reasons)
    stable = True
    if condition_number is not None and float(condition_number) > float(max_condition_number):
        stable = False
        reasons.append("condition number exceeds proxy threshold")
    return StabilityAssessment(
        stable_by_proxy=stable,
        full_device_gain_available=False,
        condition_number=None if condition_number is None else float(condition_number),
        reasons=tuple(reasons),
    )


def assemble_full_unit_cell_matrix(
    unit_cell_abcd: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Stack a block ABCD tuple into the full unit-cell transfer matrix.

    Given ``(A, B, C, D)``, each of shape ``(2N+1, 2N+1)``, return

        M = [[A, B],
             [C, D]]

    of shape ``(2*(2N+1), 2*(2N+1))``. The block layout matches the
    convention used by :mod:`tvtl.floquet.harmonic_abcd` cascade helpers.

    Parameters
    ----------
    unit_cell_abcd : tuple of numpy.ndarray
        ``(A, B, C, D)`` of identical square complex shape.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(2*(2N+1), 2*(2N+1))``.
    """

    a, b, c, d = (np.asarray(block, dtype=np.complex128) for block in unit_cell_abcd)
    if not (a.shape == b.shape == c.shape == d.shape and a.shape[0] == a.shape[1]):
        raise ValueError("A, B, C, D must be square matrices of identical shape.")
    top = np.concatenate([a, b], axis=1)
    bottom = np.concatenate([c, d], axis=1)
    return np.concatenate([top, bottom], axis=0)


def unit_cell_eigenvalues(
    unit_cell_abcd: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Eigenvalues of the assembled harmonic-basis unit-cell ABCD matrix.

    Parameters
    ----------
    unit_cell_abcd : tuple of numpy.ndarray
        ``(A, B, C, D)`` for one unit cell.

    Returns
    -------
    numpy.ndarray
        Complex eigenvalues, shape ``(2*(2N+1),)``.
    """

    matrix = assemble_full_unit_cell_matrix(unit_cell_abcd)
    eigenvalues = np.linalg.eigvals(matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("Floquet unit-cell eigvals produced non-finite values.")
    return np.asarray(eigenvalues, dtype=np.complex128)


def floquet_exponents(
    unit_cell_abcd: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    p: float,
) -> np.ndarray:
    """Floquet propagation constants ``gamma_k = log(lambda_k) / p``.

    Returns the principal branch of the complex logarithm.

    Parameters
    ----------
    unit_cell_abcd : tuple of numpy.ndarray
        ``(A, B, C, D)`` for one unit cell.
    p : float
        Unit-cell period [m]. Must be positive.

    Returns
    -------
    numpy.ndarray
        Complex array of Floquet exponents with units ``1/m``.
    """

    period = float(p)
    if period <= 0.0:
        raise ValueError("p must be a positive unit-cell period.")
    eigenvalues = unit_cell_eigenvalues(unit_cell_abcd)
    return np.log(eigenvalues) / period


def dominant_growth_rate(
    unit_cell_abcd: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """Largest per-cell amplification ``max_k |lambda_k|`` over Bloch modes.

    Parameters
    ----------
    unit_cell_abcd : tuple of numpy.ndarray
        ``(A, B, C, D)`` for one unit cell.

    Returns
    -------
    float
        Maximum of the absolute eigenvalues.
    """

    eigenvalues = unit_cell_eigenvalues(unit_cell_abcd)
    return float(np.max(np.abs(eigenvalues)))


def stability_margin(
    unit_cell_abcd: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """Dimensionless stability margin ``1 - max_k |lambda_k|``.

    ``> 0`` -> stable (subthreshold parametric regime).
    ``= 0`` -> threshold.
    ``< 0`` -> unstable (above-threshold oscillator regime).

    Returns
    -------
    float
        Real-valued margin.
    """

    return 1.0 - dominant_growth_rate(unit_cell_abcd)


def cascade_gain_estimate(
    unit_cell_abcd: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_cells: int,
) -> dict:
    """Per-mode cascade amplification estimate from the dominant eigenvalue.

    The estimate ``20 * n_cells * log10(|lambda_max|)`` is the asymptotic
    per-cell rate continued to ``n_cells`` cells under the Bloch-mode picture.
    It does NOT include finite-N boundary effects or the source/load
    matching contained in the full S-parameter cascade, so it may differ
    from ``floquet_uc_cascade_response``'s ``gain_total_proxy_dB``.

    All numbers in the returned dict are analytical proxies only.

    Parameters
    ----------
    unit_cell_abcd : tuple of numpy.ndarray
        ``(A, B, C, D)`` for one unit cell.
    n_cells : int
        Number of cells in the cascade. Non-negative.

    Returns
    -------
    dict
        Keys ``"lambda_max_abs"`` (float), ``"lambda_max"`` (complex),
        ``"cascade_gain_estimate_dB"`` (float), ``"stability_margin"``
        (float), and ``"label"`` equal to ``"analytical_proxy_only"``.
    """

    count = int(n_cells)
    if count < 0:
        raise ValueError("n_cells must be non-negative.")
    eigenvalues = unit_cell_eigenvalues(unit_cell_abcd)
    magnitudes = np.abs(eigenvalues)
    index_max = int(np.argmax(magnitudes))
    lambda_max_abs = float(magnitudes[index_max])
    lambda_max = complex(eigenvalues[index_max])
    gain_db = float(20.0 * count * np.log10(max(lambda_max_abs, np.finfo(float).tiny)))
    return {
        "lambda_max_abs": lambda_max_abs,
        "lambda_max": lambda_max,
        "cascade_gain_estimate_dB": gain_db,
        "stability_margin": float(1.0 - lambda_max_abs),
        "label": "analytical_proxy_only",
    }


