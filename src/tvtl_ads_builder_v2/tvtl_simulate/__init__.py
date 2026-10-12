"""TVTL analytical S-parameter simulator for a device spec.

This is the "source of truth" side of the parity contract: the same
device spec that the ADS netlist emitter consumes is also evaluated
analytically here. The parity check compares the two.
"""

from .s_parameters import simulate_s_parameters

__all__ = ["simulate_s_parameters"]


