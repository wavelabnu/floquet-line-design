"""Per-circuit workspace generation for the AEL diagnostic ladder.

For each diagnostic D0-D4 this builds the device spec, emits the flat
hpeesofsim ``.ckt`` deck and the AEL schematic deck from that one spec,
runs hpeesofsim, validates Path 1 (TVTL analytical vs hpeesofsim), and
packages everything into a per-circuit directory under
``<output_root>/ael_diagnostics/``.

Path 1 is the prerequisite: the flat-netlist side must agree with TVTL
before the AEL-built schematic is worth inspecting. A diagnostic whose
Path 1 fails (or is blocked because no ADS runtime was found) is marked
accordingly in its report so the user does not waste time inspecting a
schematic with no valid reference.

This module never launches the ADS GUI. It runs the hpeesofsim
*simulator* subprocess (the same engine the main parity gate uses);
building and inspecting the AEL schematic in the ADS GUI is the user's
manual step via ``open_workspace.bat``.
"""

from __future__ import annotations

import datetime
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from ...device_spec import DeviceSpec
from ...inspection.ads_workspace import (
    REFERENCE_LIBRARY_NAME,
    REFERENCE_WORKSPACE_PATH,
)
from ...inspection.ael_templates import OPENER_BAT, render_ael_script
from ...inspection.schematic_layout import build_schematic_plan
from ..check import parity_check
from .expected import write_tvtl_expected_csv
from .specs import DIAGNOSTIC_ORDER, DIAGNOSTIC_SPECS

# Path-1 tolerance; the same 0.5 dB envelope as the main parity gate.
PATH1_TOLERANCE_DB = 0.5

# What each diagnostic isolates -- used in the per-circuit README.
_DIAGNOSTIC_INFO: dict[str, dict[str, str]] = {
    "d0_bare_ports": {
        "title": "D0: Bare ports (near-zero-length MLIN)",
        "tests": (
            "The minimal port-to-element-to-port wiring. A literal "
            "element-free port-to-port wire cannot be analysed (the TVTL "
            "simulator and hpeesofsim both require at least one element), "
            "so D0 uses a 0.1 mm MLIN -- electrically a direct through "
            "(|S21| ~ 0 dB)."
        ),
        "schematic": (
            "TermG (left) -> MLIN -> TermG (right), with the MSUB / "
            "S_Param / SweepPlan controllers placed clear of the circuit."
        ),
        "expect": "|S21| ~ 0 dB across 1-10 GHz (a near-perfect through).",
    },
    "d1_single_mlin": {
        "title": "D1: Single MLIN",
        "tests": (
            "A single 50-ohm microstrip line between two ports -- the "
            "basic port-to-MLIN and MLIN-to-port connection wiring."
        ),
        "schematic": "TermG -> MLIN -> TermG, plus the controllers.",
        "expect": "|S21| ~ 0 dB across 1-10 GHz.",
    },
    "d2_two_mlins": {
        "title": "D2: Two MLINs in series",
        "tests": (
            "Two MLINs cascaded through a shared internal node -- the "
            "element-to-element connection wiring."
        ),
        "schematic": "TermG -> MLIN -> MLIN -> TermG, plus the controllers.",
        "expect": "|S21| ~ 0 dB across 1-10 GHz.",
    },
    "d3_mtee_short_stub": {
        "title": "D3: MTEE with a grounded shunt stub",
        "tests": (
            "An MTEE wired directly between the two ports, with pin 3 "
            "feeding a grounded shunt MLIN -- the MTEE pin-1 / pin-2 / "
            "pin-3 connections and the stub-to-ground wiring."
        ),
        "schematic": (
            "TermG -> MTEE -> TermG on the main line; MTEE pin 3 -> MLIN "
            "(stub) -> GROUND below it."
        ),
        "expect": (
            "A shorted-stub response: |S21| dips at low frequency where "
            "the stub presents a near-short, rising toward 0 dB as the "
            "stub reactance grows."
        ),
    },
    "d4_full_level3": {
        "title": "D4: Full Level-3 topology",
        "tests": (
            "MLIN + MTEE + MLIN on the main line with a grounded shunt "
            "MLIN -- the combined element + MTEE + stub wiring pattern, "
            "the paper-relevant T-junction geometry."
        ),
        "schematic": (
            "TermG -> MLIN -> MTEE -> MLIN -> TermG on the main line; "
            "MTEE pin 3 -> MLIN (stub) -> GROUND below it."
        ),
        "expect": (
            "The paper-dimensioned shorted-stub T-junction response set "
            "by the stub input admittance transformed to the junction."
        ),
    },
}


def resolve_ads_environment() -> dict[str, str] | None:
    """Return an ``ads_environment`` for the hpeesofsim runner, or None.

    Honours ``$HPEESOF_DIR``. ``None`` means no ADS runtime was found -- Path 1 is then reported as blocked rather than crashing.
    """

    hpeesof_dir = os.environ.get("HPEESOF_DIR")
    if hpeesof_dir and Path(hpeesof_dir).is_dir():
        return {"hpeesof_dir": hpeesof_dir}
    return None


def generate_one(name: str, diagnostics_root: str | Path) -> dict[str, Any]:
    """Generate one diagnostic workspace and validate its Path 1.

    Parameters
    ----------
    name : str
        A diagnostic name (a key of ``DIAGNOSTIC_SPECS``).
    diagnostics_root : str or Path
        The ``ael_diagnostics`` directory; the workspace is created in
        ``<diagnostics_root>/<name>/``.

    Returns
    -------
    dict
        ``{"name", "circuit_dir", "path1_status", "path1_reason",
        "max_abs_delta_dB"}``. ``path1_status`` is ``PASS`` / ``FAIL`` /
        ``BLOCKED`` / ``ERROR``.
    """

    spec = DIAGNOSTIC_SPECS[name]()
    circuit_dir = Path(diagnostics_root) / name
    circuit_dir.mkdir(parents=True, exist_ok=True)
    ads_environment = resolve_ads_environment()

    # parity_check runs TVTL + hpeesofsim and writes device_spec.json,
    # hpeesofsim.ckt and tvtl_s_parameters.csv into the circuit dir.
    result = parity_check(
        spec,
        circuit_dir,
        ads_environment=ads_environment,
        tolerance_dB_S21=PATH1_TOLERANCE_DB,
        level_label=name,
    )

    write_tvtl_expected_csv(spec, circuit_dir / "tvtl_expected.csv")
    _write_hpeesofsim_result_csv(circuit_dir / "hpeesofsim_result.csv", result)
    _write_ael_deck(circuit_dir, spec, name)
    (circuit_dir / "open_workspace.bat").write_text(OPENER_BAT, encoding="utf-8")

    status = _path1_status(result)
    _write_path1_report(circuit_dir / "path1_parity_report.txt", name, result, status)
    (circuit_dir / "README.md").write_text(
        _render_readme(name, spec, result, status), encoding="utf-8"
    )
    return {
        "name": name,
        "circuit_dir": circuit_dir,
        "path1_status": status,
        "path1_reason": str(result.get("reason", "")),
        "max_abs_delta_dB": result.get("max_abs_delta_dB"),
    }


def generate_all(output_root: str | Path) -> list[dict[str, Any]]:
    """Generate every diagnostic workspace D0-D4 under ``output_root``.

    Returns the per-circuit summary dicts in ladder order.
    """

    diagnostics_root = Path(output_root) / "ael_diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    return [generate_one(name, diagnostics_root) for name in DIAGNOSTIC_ORDER]


def _path1_status(result: dict[str, Any]) -> str:
    """Map a parity-check reason to a Path-1 status string."""

    reason = str(result.get("reason", ""))
    if reason == "within_tolerance":
        return "PASS"
    if reason == "exceeds_tolerance":
        return "FAIL"
    if reason == "ads_blocked":
        return "BLOCKED"
    return "ERROR"


def _write_ael_deck(circuit_dir: Path, spec: DeviceSpec, cell: str) -> None:
    """Emit ``build_workspace.ael`` for one diagnostic circuit."""

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    plan = build_schematic_plan(spec)
    deck = render_ael_script(
        REFERENCE_LIBRARY_NAME, cell, plan, timestamp, REFERENCE_WORKSPACE_PATH
    )
    (circuit_dir / "build_workspace.ael").write_text(
        deck, encoding="utf-8", newline="\n"
    )


def _write_hpeesofsim_result_csv(path: Path, result: dict[str, Any]) -> None:
    """Write the hpeesofsim |S21| result onto the TVTL frequency grid."""

    freqs = np.asarray(result.get("frequencies_Hz", []), dtype=float)
    ads_dB = np.asarray(result.get("ads_S21_dB", []), dtype=float)
    lines = ["freq_Hz,S21_dB\n"]
    for index in range(freqs.size):
        value = ads_dB[index] if index < ads_dB.size else float("nan")
        cell = "NaN" if not math.isfinite(float(value)) else f"{float(value):.6f}"
        lines.append(f"{float(freqs[index]):.6e},{cell}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _delta_line(result: dict[str, Any]) -> str:
    """A one-line ``max |delta_S21|`` summary for the report / table."""

    max_abs = result.get("max_abs_delta_dB")
    try:
        value = float(max_abs)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(value) else f"{value:.3f} dB"


def _write_path1_report(
    path: Path, name: str, result: dict[str, Any], status: str
) -> None:
    """Write the Path-1 (TVTL vs hpeesofsim) parity report."""

    lines: list[str] = []
    lines.append(f"AEL diagnostic Path-1 parity report -- {name}\n")
    lines.append("=" * 60 + "\n\n")
    lines.append("Path 1: TVTL analytical  vs  hpeesofsim flat-netlist run.\n")
    lines.append(f"Tolerance        : {PATH1_TOLERANCE_DB:.3f} dB\n")
    lines.append(f"Status           : {status}\n")
    lines.append(f"reason           : {result.get('reason', '')}\n")
    lines.append(f"max |delta_S21|  : {_delta_line(result)}\n")
    ads_run = result.get("ads_run", {})
    lines.append(f"hpeesofsim status: {ads_run.get('status', '?')}\n")
    if ads_run.get("error_message"):
        lines.append(f"hpeesofsim error : {ads_run['error_message']}\n")
    lines.append("\n")
    if status == "PASS":
        lines.append(
            "Path 1 PASSES. The flat-netlist side agrees with TVTL, so the\n"
            "AEL-built schematic now has a valid numerical reference to\n"
            "compare against once it is built and simulated in ADS.\n"
        )
    elif status == "FAIL":
        lines.append(
            "Path 1 FAILS: the flat netlist disagrees with TVTL beyond\n"
            "tolerance. This is a flat-netlist-emitter or TVTL-analytical\n"
            "problem, NOT an AEL-builder problem -- inspecting the AEL\n"
            "schematic is premature until Path 1 passes.\n"
        )
    elif status == "BLOCKED":
        lines.append(
            "Path 1 is BLOCKED: no ADS runtime was found, so hpeesofsim\n"
            "could not run. The TVTL analytical reference is still valid\n"
            "(see tvtl_expected.csv); set HPEESOF_DIR and regenerate to\n"
            "complete Path 1.\n"
        )
    else:
        lines.append(
            "Path 1 hit an ERROR -- see 'reason' above. The diagnostic\n"
            "workspace is not usable until this is resolved.\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _render_readme(
    name: str, spec: DeviceSpec, result: dict[str, Any], status: str
) -> str:
    """Render the per-diagnostic ``README.md``."""

    info = _DIAGNOSTIC_INFO[name]
    n_elements = len(spec.elements)
    lines: list[str] = []
    lines.append(f"# Diagnostic {info['title']}\n\n")

    lines.append("## What this tests\n\n")
    lines.append(info["tests"] + "\n\n")

    lines.append("## Path 1 status (TVTL <-> hpeesofsim)\n\n")
    if status == "PASS":
        lines.append(
            f"**PASS** at {_delta_line(result)} max delta across 1-10 GHz.\n\n"
            "This is the prerequisite. The flat-netlist side agrees with "
            "TVTL, so the AEL inspection below has a valid reference.\n\n"
        )
    elif status == "BLOCKED":
        lines.append(
            "**BLOCKED** -- no ADS runtime was found, so hpeesofsim could "
            "not run. The TVTL reference (`tvtl_expected.csv`) is still "
            "valid; set `HPEESOF_DIR` and regenerate to complete Path 1.\n\n"
        )
    else:
        lines.append(
            f"**{status}** -- see `path1_parity_report.txt`. Path 1 is the "
            "prerequisite; resolve it before inspecting the AEL schematic.\n\n"
        )

    lines.append("## How to verify the AEL builder\n\n")
    lines.append(
        "1. Double-click `open_workspace.bat`. ADS launches, opens the "
        "reference workspace, builds the schematic and simulates it.\n"
        f"2. After ADS finishes, the schematic should show: {info['schematic']}\n"
        "   Look for visible wires connecting each adjacent pin pair -- no "
        "orphaned red pin markers.\n"
        "3. Open the result dataset (Data Display) and plot `|S21|`. "
        f"Expected: {info['expect']}\n"
        "   It should match `tvtl_expected.csv` (and `hpeesofsim_result.csv`, "
        "since Path 1 already agrees with TVTL).\n\n"
    )

    lines.append("## If the schematic looks wrong\n\n")
    lines.append(
        "Report the specific failure to the maintainer. Useful detail:\n\n"
        "- Which elements are present / missing / rotated wrong.\n"
        "- Which pin pairs lack a visible connecting wire.\n"
        "- The schematic's `|S21|`, if it differs from "
        "`tvtl_expected.csv`.\n\n"
    )

    lines.append("## What the AEL builder generated to produce this\n\n")
    lines.append(
        f"- Device spec: {n_elements} element(s) "
        f"({_element_summary(spec)}), 1 substrate, S-parameter analysis.\n"
        "- Flat netlist: `hpeesofsim.ckt` (Path 1 input).\n"
        "- AEL deck: `build_workspace.ael` (the schematic builder).\n"
        "- TVTL reference: `tvtl_expected.csv`.\n"
        "- hpeesofsim result: `hpeesofsim_result.csv`.\n"
        "- Path-1 report: `path1_parity_report.txt`.\n"
    )
    return "".join(lines)


def _element_summary(spec: DeviceSpec) -> str:
    """A compact element-type tally for the README."""

    counts: dict[str, int] = {}
    for element in spec.elements:
        counts[element.type] = counts.get(element.type, 0) + 1
    return ", ".join(f"{count}x {etype}" for etype, count in sorted(counts.items()))



