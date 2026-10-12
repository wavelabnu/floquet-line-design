"""ABCD to S-parameter conversion with stable reciprocal defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .abcd import as_abcd


@dataclass(frozen=True, slots=True)
class SParameters:
    s11: complex
    s21: complex
    s12: complex
    s22: complex

    def as_matrix(self) -> np.ndarray:
        return np.asarray([[self.s11, self.s12], [self.s21, self.s22]], dtype=np.complex128)


def abcd_to_sparameters(
    matrix: Any,
    reference_impedance_ohm: float = 50.0,
    *,
    reciprocal_assumption: bool = True,
    determinant: complex | None = None,
) -> SParameters:
    """Convert one ABCD matrix to S-parameters.

    For reciprocal passive TVTL cascades, the default sets ``S12 = S21`` rather
    than recomputing a determinant from a long repeated cascade. This avoids the
    old numerical S12 determinant spike while preserving the intended reciprocal
    network semantics. Set ``reciprocal_assumption=False`` and pass a trusted
    determinant when a non-reciprocal matrix is genuinely needed.
    """

    z0 = float(reference_impedance_ohm)
    if z0 <= 0.0:
        raise ValueError("reference_impedance_ohm must be positive.")
    abcd = as_abcd(matrix)
    a_val, b_val = abcd[0, 0], abcd[0, 1]
    c_val, d_val = abcd[1, 0], abcd[1, 1]
    denominator = a_val + (b_val / z0) + (c_val * z0) + d_val
    if abs(denominator) <= np.finfo(float).tiny:
        raise FloatingPointError("ABCD to S-parameter conversion denominator is too small.")
    s11 = (a_val + (b_val / z0) - (c_val * z0) - d_val) / denominator
    s21 = 2.0 / denominator
    if reciprocal_assumption:
        s12 = s21
    else:
        det = complex(np.linalg.det(abcd) if determinant is None else determinant)
        s12 = 2.0 * det / denominator
    s22 = (-a_val + (b_val / z0) - (c_val * z0) + d_val) / denominator
    values = SParameters(complex(s11), complex(s21), complex(s12), complex(s22))
    if not np.all(np.isfinite(values.as_matrix())):
        raise FloatingPointError("ABCD to S-parameter conversion produced non-finite values.")
    return values


