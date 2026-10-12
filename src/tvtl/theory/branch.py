"""Generic composition helpers for static shunt branch unit cells."""

from __future__ import annotations

from typing import Any

from .abcd import ComplexMatrix, shunt_admittance_abcd
from .cascade import cascade_abcd
from .lumped import shunt_branch_admittance


def compose_shunt_loaded_unit_cell(
    left_abcd: Any,
    branch_impedance_ohm: complex,
    right_abcd: Any,
) -> ComplexMatrix:
    """Return left through section, shunt branch load, right through section."""

    return cascade_abcd(
        [
            left_abcd,
            shunt_admittance_abcd(shunt_branch_admittance(branch_impedance_ohm)),
            right_abcd,
        ]
    )


