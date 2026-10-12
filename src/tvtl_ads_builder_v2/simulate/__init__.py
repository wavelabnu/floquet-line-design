"""Subprocess wrappers for hpeesofsim and the resulting ``.ds`` dataset.

The runner invokes the simulator on a flat ``.ckt``; the parser reads
the resulting ``.ds`` via the Keysight ADS Python API (when available)
or via an ADS-Python subprocess fallback.
"""

from .runner import run_hpeesofsim
from .parser import parse_s_parameter_results
from .sweep_runner import (
    SweepPointResult,
    SweepResult,
    SweepRunner,
    ads_runtime_available,
)

__all__ = [
    "SweepPointResult",
    "SweepResult",
    "SweepRunner",
    "ads_runtime_available",
    "parse_s_parameter_results",
    "run_hpeesofsim",
]


