"""Generic lumped passive impedances for static TVTL branch models."""

from __future__ import annotations

import numpy as np


def capacitor_impedance(frequency_hz: float, capacitance_f: float) -> complex:
    frequency = float(frequency_hz)
    capacitance = float(capacitance_f)
    if frequency <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    if capacitance <= 0.0:
        raise ValueError("capacitance_f must be positive.")
    return complex(1.0 / (1j * 2.0 * np.pi * frequency * capacitance))


def series_rc_branch_impedance(
    frequency_hz: float,
    capacitance_f: float,
    resistance_ohm: float,
    *,
    return_impedance_ohm: complex = 0.0,
) -> complex:
    resistance = float(resistance_ohm)
    if resistance < 0.0:
        raise ValueError("resistance_ohm must be non-negative.")
    return complex(resistance + capacitor_impedance(frequency_hz, capacitance_f) + complex(return_impedance_ohm))


def shunt_branch_admittance(branch_impedance_ohm: complex) -> complex:
    impedance = complex(branch_impedance_ohm)
    if abs(impedance) <= np.finfo(float).tiny:
        raise ValueError("branch impedance is too small.")
    return complex(1.0 / impedance)


