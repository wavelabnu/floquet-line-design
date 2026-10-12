"""Analytical checks for the minimal self-pumped validation circuit.

This module intentionally evaluates only the linear passive RF path at
``fs``. It does not claim to predict nonlinear parametric gain; the first
validation target is topology/connectivity/netlist equivalence across the
analytical, direct hpeesofsim, and ADS OA schematic paths.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..circuit_spec import SelfPumpedCircuitSpec


def evaluate_analytical(spec: SelfPumpedCircuitSpec) -> dict[str, Any]:
    """Return the analytical RF-path expectations for ``spec``."""

    matched = (
        math.isclose(spec.source_z_parallel_ohm, spec.line_z_ohm, rel_tol=1e-12)
        and math.isclose(spec.line_z_ohm, spec.load_z_ohm, rel_tol=1e-12)
        and math.isclose(spec.line_freq_hz or spec.fs_hz, spec.fs_hz, rel_tol=1e-12)
    )
    return {
        "fs_hz": float(spec.fs_hz),
        "source_z_parallel_ohm": float(spec.source_z_parallel_ohm),
        "line_z_ohm": float(spec.line_z_ohm),
        "load_z_ohm": float(spec.load_z_ohm),
        "line_e_deg": float(spec.line_e_deg),
        "line_freq_hz": float(spec.effective_line_freq_hz),
        "expected_rf_main_net": spec.expected_rf_main_net,
        "expected_matched": bool(matched),
        "comparison_scope": (
            "linear passive RF path at fs; topology/connectivity equivalence"
        ),
        "not_compared": (
            "nonlinear parametric gain is not predicted by this minimal "
            "linear analytical check"
        ),
    }


def write_analytical_result(
    spec: SelfPumpedCircuitSpec,
    out_dir: str | Path,
) -> Path:
    """Write ``analytical.json`` under ``out_dir`` and return its path."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_path = out_path / "analytical.json"
    result_path.write_text(
        json.dumps(evaluate_analytical(spec), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result_path
