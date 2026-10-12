"""Passive and reciprocal sanity checks for S-parameter matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .sparams import SParameters


@dataclass(frozen=True, slots=True)
class PassiveReciprocalCheck:
    finite: bool
    reciprocal: bool
    passive_by_singular_value: bool
    max_singular_value: float
    max_reciprocity_error: float

    @property
    def passing(self) -> bool:
        return self.finite and self.reciprocal and self.passive_by_singular_value


def check_passive_reciprocal(
    sparameters: SParameters | Any,
    *,
    tolerance: float = 1.0e-9,
) -> PassiveReciprocalCheck:
    matrix = sparameters.as_matrix() if isinstance(sparameters, SParameters) else np.asarray(sparameters, dtype=np.complex128)
    if matrix.shape != (2, 2):
        raise ValueError("S-parameter matrix must have shape (2, 2).")
    finite = bool(np.all(np.isfinite(matrix)))
    max_sv = float(np.linalg.svd(matrix, compute_uv=False)[0]) if finite else float("inf")
    reciprocity_error = float(abs(matrix[0, 1] - matrix[1, 0])) if finite else float("inf")
    return PassiveReciprocalCheck(
        finite=finite,
        reciprocal=reciprocity_error <= float(tolerance),
        passive_by_singular_value=max_sv <= 1.0 + float(tolerance),
        max_singular_value=max_sv,
        max_reciprocity_error=reciprocity_error,
    )


