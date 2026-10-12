"""Direct hpeesofsim amplifier deck generation and optional simulation."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from ..amplifier_spec import (
    AmplifierCircuitSpec,
    PUMP_SOURCE_NAME,
    RF_MAIN_NET,
    SIGNAL_SOURCE_NAME,
    load_amplifier_circuit_spec,
)
from ..netlist import emit_flat_hb_netlist
from .direct_hpeesofsim import HpeesofsimRunResult, _mtimes, _new_or_updated

FORBIDDEN_LANE_NETS = {"PUMP", "CONTROL", "BIAS"}
DEPRECATED_MARKERS = ("L:LVIA", "LVIA", "MLOC:")


def direct_amplifier_netlist(spec: AmplifierCircuitSpec) -> str:
    """Return the flat two-tone HB deck for an amplifier candidate."""

    text = emit_flat_hb_netlist(spec.device_spec)
    assert_direct_amplifier_netlist_policy(text, spec)
    return text


def write_direct_amplifier_deck(
    input_summary: str | Path,
    output_dir: str | Path,
    *,
    candidate_index: int | None = None,
    candidate_name: str | None = None,
    signal_power_dbm: float | None = None,
    pump_power_dbm: float | None = None,
    pump_phase_rad: float = 0.0,
    filename: str = "amplifier_direct.ckt",
) -> dict[str, Any]:
    """Write the direct deck plus its JSON spec and return path metadata."""

    spec = load_amplifier_circuit_spec(
        input_summary,
        candidate_index=candidate_index,
        candidate_name=candidate_name,
        signal_power_dbm=signal_power_dbm,
        pump_power_dbm=pump_power_dbm,
        pump_phase_rad=pump_phase_rad,
    )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    netlist_path = out_path / filename
    spec_path = out_path / "amplifier_direct_spec.json"
    netlist = direct_amplifier_netlist(spec)
    netlist_path.write_text(netlist, encoding="utf-8")
    spec_path.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "written",
        "candidate": spec.candidate_name,
        "netlist_path": str(netlist_path),
        "spec_path": str(spec_path),
        "output_dir": str(out_path),
        "spec": spec,
    }


def run_direct_amplifier_hpeesofsim(
    spec: AmplifierCircuitSpec,
    output_dir: str | Path,
    *,
    filename: str = "amplifier_direct.ckt",
) -> HpeesofsimRunResult:
    """Run the direct amplifier deck through ``CircuitSimulator.run_netlist``."""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    netlist_path = out_path / filename
    netlist = direct_amplifier_netlist(spec)
    netlist_path.write_text(netlist, encoding="utf-8")
    before = _mtimes(out_path)
    started = time.time()
    try:
        from keysight.edatoolbox import ads, util

        util.safe_makedirs(str(out_path))
        ads.CircuitSimulator().run_netlist(netlist, output_dir=str(out_path))
    except Exception as exc:  # noqa: BLE001 - reported in validation JSON
        output_files = _new_or_updated(out_path, before, started)
        return HpeesofsimRunResult(
            status="error",
            netlist_path=str(netlist_path),
            output_dir=str(out_path),
            dataset_files=[],
            output_files=output_files,
            error_message=_simulator_error_message(str(exc), out_path, output_files),
        )

    output_files = _new_or_updated(out_path, before, started)
    dataset_files = sorted(name for name in output_files if name.endswith(".ds"))
    return HpeesofsimRunResult(
        status="passed" if dataset_files else "error",
        netlist_path=str(netlist_path),
        output_dir=str(out_path),
        dataset_files=dataset_files,
        output_files=output_files,
        error_message=None if dataset_files else "hpeesofsim wrote no .ds dataset",
    )


def _simulator_error_message(message: str, output_dir: Path, output_files: list[str]) -> str:
    tail = _latest_simulator_out_tail(output_dir, output_files)
    if not tail:
        return message
    return f"{message}\nSimulator .out tail:\n{tail}"


def _latest_simulator_out_tail(
    output_dir: Path,
    output_files: list[str],
    *,
    lines: int = 24,
) -> str | None:
    candidates = [
        output_dir / name
        for name in output_files
        if name.endswith(".out") and (output_dir / name).is_file()
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    text = newest.read_text(encoding="utf-8", errors="replace")
    tail_lines = text.splitlines()[-lines:]
    return "\n".join(tail_lines).strip() or None


def assert_direct_amplifier_netlist_policy(
    text: str,
    spec: AmplifierCircuitSpec | None = None,
) -> None:
    """Raise when the direct amplifier deck violates public topology policy."""

    for name in (SIGNAL_SOURCE_NAME, PUMP_SOURCE_NAME):
        prefix = f"Port:{name} {RF_MAIN_NET} 0 "
        if prefix not in text:
            raise ValueError(f"{name} is not connected as `{prefix.strip()}`")
    for marker in DEPRECATED_MARKERS:
        if marker in text:
            raise ValueError(f"direct amplifier deck contains deprecated marker {marker!r}")
    tokens = {token.strip(" ,;()") for token in text.replace("\n", " ").split()}
    forbidden = sorted(FORBIDDEN_LANE_NETS & tokens)
    if forbidden:
        raise ValueError(f"direct amplifier deck contains forbidden lane net(s): {forbidden}")
    if spec is not None:
        viagnd_expected = spec.termination_type in {"short", "combined"}
        mlef_expected = spec.termination_type in {"open", "combined"}
        if viagnd_expected and "VIAGND:" not in text:
            raise ValueError("direct amplifier deck is missing VIAGND termination(s)")
        if mlef_expected and "MLEF:" not in text:
            raise ValueError("direct amplifier deck is missing MLEF termination(s)")
