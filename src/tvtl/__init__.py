"""Analytical TVTL stack -- the source-of-truth side of the paper supplement.

The :mod:`tvtl` package collects the closed-form physics the manuscript
"Physics-Guided Design of Varactor-Loaded Time-Varying Transmission
Lines" rests on: passive transmission-line ABCD primitives
(:mod:`tvtl.theory`), microstrip dispersion via Hammerstad-Kirschning
(:mod:`tvtl.microstrip`), the harmonic-basis Floquet machinery for the
time-modulated unit cell (:mod:`tvtl.floquet`), and the design-search
scaffolding (:mod:`tvtl.design`, :mod:`tvtl.optimization`).

It is the analytical truth against which the ADS-side builder
(:mod:`tvtl_ads_builder_v2`) is validated.

Public API
----------
Symbols listed in ``__all__`` below form the stable public surface;
see ``STABILITY.md`` for the public/internal split. Subpackages
themselves are importable and carry their own ``__all__`` declarations
for finer-grained imports (``from tvtl.floquet import ...``).
"""

from .floquet import (
    compute_pump_voltage_profile,
    floquet_exponents,
    modulation_depth_sweep,
    pump_coherence_check,
    pump_power_profile,
    unit_cell_eigenvalues,
    unit_cell_harmonic_abcd,
)
from .materials import Substrate, legacy_circuit1_substrate
from .microstrip import (
    hammerstad_kirschning,
    hammerstad_thin_metal,
    open_end_capacitance_f,
    tee_junction,
)
from .theory import (
    bloch_dispersion,
    cascade_abcd,
    line_abcd,
    repeat_unit_cell,
    series_impedance_abcd,
    shunt_admittance_abcd,
)
from .varactors import (
    PolynomialVaractorModel,
    VaractorModel,
    legacy_fdtd_varactor_model,
)


__all__ = [
    # ---- theory (passive ABCD primitives) ----
    "bloch_dispersion",
    "cascade_abcd",
    "line_abcd",
    "repeat_unit_cell",
    "series_impedance_abcd",
    "shunt_admittance_abcd",
    # ---- floquet (time-modulated unit-cell harmonics) ----
    "compute_pump_voltage_profile",
    "floquet_exponents",
    "modulation_depth_sweep",
    "pump_coherence_check",
    "pump_power_profile",
    "unit_cell_eigenvalues",
    "unit_cell_harmonic_abcd",
    # ---- microstrip ----
    "hammerstad_kirschning",
    "hammerstad_thin_metal",
    "open_end_capacitance_f",
    "tee_junction",
    # ---- materials ----
    "Substrate",
    "legacy_circuit1_substrate",
    # ---- varactors ----
    "PolynomialVaractorModel",
    "VaractorModel",
    "legacy_fdtd_varactor_model",
]


