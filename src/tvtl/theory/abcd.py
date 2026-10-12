"""ABCD matrix primitives for generic TVTL calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


ComplexMatrix = NDArray[np.complex128]


def identity_abcd() -> ComplexMatrix:
    return np.eye(2, dtype=np.complex128)


def as_abcd(matrix: Any) -> ComplexMatrix:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (2, 2):
        raise ValueError("ABCD matrix must have shape (2, 2).")
    if not np.all(np.isfinite(value)):
        raise ValueError("ABCD matrix must contain only finite values.")
    return value


def line_abcd(z0_ohm: float, propagation: complex) -> ComplexMatrix:
    z0 = float(z0_ohm)
    if z0 <= 0.0:
        raise ValueError("z0_ohm must be positive.")
    theta = complex(propagation)
    return np.asarray(
        [
            [np.cosh(theta), z0 * np.sinh(theta)],
            [np.sinh(theta) / z0, np.cosh(theta)],
        ],
        dtype=np.complex128,
    )


def series_impedance_abcd(impedance_ohm: complex) -> ComplexMatrix:
    return np.asarray([[1.0, complex(impedance_ohm)], [0.0, 1.0]], dtype=np.complex128)


def shunt_admittance_abcd(admittance_siemens: complex) -> ComplexMatrix:
    return np.asarray([[1.0, 0.0], [complex(admittance_siemens), 1.0]], dtype=np.complex128)


