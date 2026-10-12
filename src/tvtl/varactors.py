"""Varactor models shared by the analytical and FDTD layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Re-exported for discoverability: the realistic varactor device
# model (SPICE C-V + package parasitics) lives under ``tvtl.design`` (it
# builds on the charge-polynomial machinery there) but is surfaced here so
# ``from tvtl.varactors import RealisticVaractor`` works alongside the
# lumped static models below.
from .design.realistic_varactor import RealisticVaractor


@dataclass(frozen=True, slots=True)
class VaractorModel:
    """Lumped series capacitor model."""

    capacitance_f: float
    series_resistance_ohm: float = 0.0

    def impedance_ohm(self, frequency_hz: ArrayLike) -> NDArray[np.complex128]:
        frequency = np.asarray(frequency_hz, dtype=float)
        if np.any(frequency <= 0.0):
            raise ValueError("frequency_hz must be strictly positive")
        omega = 2.0 * np.pi * frequency
        reactance = -1j / (omega * self.capacitance_f)
        return self.series_resistance_ohm + reactance


@dataclass(frozen=True, slots=True)
class PolynomialVaractorModel:
    """Legacy polynomial capacitance law used by the MATLAB FDTD code."""

    coefficients_f: tuple[float, ...]

    def capacitance_f(self, voltage_v: ArrayLike) -> NDArray[np.float64]:
        voltage = np.asarray(voltage_v, dtype=float)
        total = np.zeros_like(voltage, dtype=float)
        for order, coefficient in enumerate(self.coefficients_f):
            total += coefficient * np.power(voltage, order)
        return total

    def distributed_capacitance_f_per_m(
        self,
        voltage_v: ArrayLike,
        dz_m: float,
        varactor_spacing_m: float,
    ) -> NDArray[np.float64]:
        return self.capacitance_f(voltage_v) * dz_m / varactor_spacing_m


def legacy_fdtd_varactor_model() -> PolynomialVaractorModel:
    """Return the polynomial coefficients embedded in the principal MATLAB file."""

    return PolynomialVaractorModel(
        coefficients_f=(3.14e-11, 1.88e-11, 6.43e-12, 1.24e-12, 1.23e-13, 4.8e-15)
    )


