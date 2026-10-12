"""Local Floquet matrix construction for generic unit-cell branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .harmonics import harmonic_frequencies, harmonic_indices


@dataclass(frozen=True, slots=True)
class FloquetBranchParameters:
    resistance_ohm: float
    inductance_h: float
    elastance_coefficients: Mapping[int, complex]

    def __post_init__(self) -> None:
        if float(self.resistance_ohm) < 0.0 or float(self.inductance_h) < 0.0:
            raise ValueError("Floquet branch resistance and inductance must be non-negative.")
        if 0 not in self.elastance_coefficients:
            raise ValueError("elastance_coefficients must include harmonic 0.")


@dataclass(frozen=True, slots=True)
class FloquetMatrixSolution:
    harmonic_indices: np.ndarray
    angular_frequencies_rad_s: np.ndarray
    matrix: np.ndarray
    admittance_matrix_s: np.ndarray
    condition_number: float


def build_local_floquet_matrix(
    parameters: FloquetBranchParameters,
    signal_angular_frequency_rad_s: float,
    modulation_angular_frequency_rad_s: float,
    truncation_order: int,
) -> np.ndarray:
    indices = harmonic_indices(truncation_order)
    frequencies = harmonic_frequencies(signal_angular_frequency_rad_s, modulation_angular_frequency_rad_s, truncation_order)
    matrix = np.zeros((indices.size, indices.size), dtype=np.complex128)
    for row, n_index in enumerate(indices):
        omega_n = frequencies[row]
        for col, ell_index in enumerate(indices):
            value = complex(parameters.elastance_coefficients.get(int(n_index - ell_index), 0.0))
            if row == col:
                value += 1j * float(parameters.resistance_ohm) * omega_n - float(parameters.inductance_h) * omega_n**2
            matrix[row, col] = value
    return matrix


def solve_local_floquet_admittance(
    parameters: FloquetBranchParameters,
    signal_angular_frequency_rad_s: float,
    modulation_angular_frequency_rad_s: float,
    truncation_order: int,
) -> FloquetMatrixSolution:
    indices = harmonic_indices(truncation_order)
    frequencies = harmonic_frequencies(signal_angular_frequency_rad_s, modulation_angular_frequency_rad_s, truncation_order)
    matrix = build_local_floquet_matrix(parameters, signal_angular_frequency_rad_s, modulation_angular_frequency_rad_s, truncation_order)
    charge_per_voltage = np.linalg.solve(matrix, np.eye(matrix.shape[0], dtype=np.complex128))
    admittance = (1j * frequencies[:, np.newaxis]) * charge_per_voltage
    condition = float(np.linalg.cond(matrix))
    if not np.all(np.isfinite(admittance)):
        raise FloatingPointError("Floquet admittance solve produced non-finite values.")
    return FloquetMatrixSolution(indices, frequencies, matrix, admittance, condition)


