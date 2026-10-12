"""Generic cascade helpers for unit-cell ABCD matrices."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .abcd import ComplexMatrix, as_abcd, identity_abcd


def cascade_abcd(matrices: Iterable[Any]) -> ComplexMatrix:
    result = identity_abcd()
    for matrix in matrices:
        result = result @ as_abcd(matrix)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("ABCD cascade produced a non-finite value.")
    return result


def repeat_unit_cell(unit_cell_abcd: Any, count: int) -> ComplexMatrix:
    if int(count) < 0:
        raise ValueError("count must be non-negative.")
    matrix = as_abcd(unit_cell_abcd)
    if int(count) == 0:
        return identity_abcd()
    result = np.linalg.matrix_power(matrix, int(count))
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Repeated unit-cell cascade produced a non-finite value.")
    return np.asarray(result, dtype=np.complex128)


def cascade_unit_cells(unit_cell_matrices: Iterable[Any]) -> ComplexMatrix:
    return cascade_abcd(unit_cell_matrices)


