"""Small ABCD matrix utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def line_abcd(z0_ohm: float, theta: complex) -> NDArray[np.complex128]:
    return np.asarray(
        [
            [np.cosh(theta), z0_ohm * np.sinh(theta)],
            [np.sinh(theta) / z0_ohm, np.cosh(theta)],
        ],
        dtype=np.complex128,
    )


def shunt_admittance_abcd(admittance_siemens: complex) -> NDArray[np.complex128]:
    return np.asarray([[1.0, 0.0], [admittance_siemens, 1.0]], dtype=np.complex128)


def series_impedance_abcd(impedance_ohm: complex) -> NDArray[np.complex128]:
    return np.asarray([[1.0, impedance_ohm], [0.0, 1.0]], dtype=np.complex128)


def cascade(*matrices: NDArray[np.complex128]) -> NDArray[np.complex128]:
    result = np.eye(2, dtype=np.complex128)
    for matrix in matrices:
        result = result @ np.asarray(matrix, dtype=np.complex128)
    return result


def matrix_power(matrix: NDArray[np.complex128], exponent: int) -> NDArray[np.complex128]:
    return np.linalg.matrix_power(np.asarray(matrix, dtype=np.complex128), exponent)


def s_parameters_from_abcd(
    abcd: NDArray[np.complex128],
    reference_impedance_ohm: float,
) -> tuple[complex, complex]:
    a_val, b_val = abcd[0, 0], abcd[0, 1]
    c_val, d_val = abcd[1, 0], abcd[1, 1]
    denom = a_val + (b_val / reference_impedance_ohm) + (c_val * reference_impedance_ohm) + d_val
    s11 = (a_val + (b_val / reference_impedance_ohm) - (c_val * reference_impedance_ohm) - d_val) / denom
    s21 = 2.0 / denom
    return complex(s11), complex(s21)


