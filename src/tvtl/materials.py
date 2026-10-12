"""Material models and physical constants used throughout the package."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

VACUUM_PERMEABILITY_H_PER_M = 4.0 * pi * 1e-7
VACUUM_PERMITTIVITY_F_PER_M = 8.854e-12
SPEED_OF_LIGHT_M_PER_S = 1.0 / sqrt(VACUUM_PERMEABILITY_H_PER_M * VACUUM_PERMITTIVITY_F_PER_M)
FREE_SPACE_IMPEDANCE_OHM = sqrt(VACUUM_PERMEABILITY_H_PER_M / VACUUM_PERMITTIVITY_F_PER_M)
COPPER_RESISTIVITY_OHM_M = 1.85e-8


@dataclass(frozen=True, slots=True)
class Substrate:
    """Uniform microstrip substrate in SI units."""

    epsilon_r: float
    thickness_m: float
    copper_thickness_m: float
    loss_tangent: float = 0.0
    conductivity_s_per_m: float = 5.8e7


def legacy_circuit1_substrate() -> Substrate:
    """Notebook substrate used in the principal Mathematica analytical files."""

    return Substrate(
        epsilon_r=3.38,
        thickness_m=1.54e-3,
        copper_thickness_m=17.5e-6,
        loss_tangent=0.0027,
        conductivity_s_per_m=5.8e7,
    )


