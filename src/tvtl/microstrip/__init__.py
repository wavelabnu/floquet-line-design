"""Microstrip models and discontinuity helpers."""

from .discontinuities import TeeJunction, open_end_capacitance_f, open_end_input_impedance_ohm, tee_junction, via_impedance_ohm
from .models import calc_gamma_legacy, hammerstad_kirschning, hammerstad_thin_metal

__all__ = [
    "TeeJunction",
    "calc_gamma_legacy",
    "hammerstad_kirschning",
    "hammerstad_thin_metal",
    "open_end_capacitance_f",
    "open_end_input_impedance_ohm",
    "tee_junction",
    "via_impedance_ohm",
]


