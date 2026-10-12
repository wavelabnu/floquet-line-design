"""Emit the ADS inspection artifacts for a parity level.

ADS builds schematics by *element placement* -- the function vocabulary
was recovered from two macros recorded in ADS 2026 (see
:mod:`.verified_ael_functions`). For each canonical circuit this module
emits, into the circuit's results directory:

* ``<circuit>.net`` -- the ADS-native netlist (reference / handoff
  form; see :func:`tvtl_ads_builder_v2.netlist.emit_ads_native_netlist`).
* ``build_workspace.ael`` -- an AEL deck that builds the circuit's
  schematic cellview by placing and wiring every element.
* ``open_workspace.bat`` -- a launcher that runs the deck in ADS.

It also writes a per-level ``ads_inspection_workspace/README.md``
describing the workflow. The generator never invokes ADS; it only
writes files, so it is fully usable and testable with no ADS installed.

Element *parameter values* are not set by the deck -- the schematic
shows the correct topology with ADS default values. Parameter
substitution is a separate follow-up.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from ..netlist import emit_ads_native_netlist
from .ael_templates import OPENER_BAT, README_TIMESTAMP_MARKER, render_ael_script
from .metadata import metadata_for
from .schematic_layout import build_schematic_plan

DEFAULT_WORKSPACE_NAME = "ads_inspection_workspace"
DEFAULT_OUTPUT_ROOT = Path.cwd() / "runs"
# The design library inside the reference workspace that the AEL deck
# creates its cellviews in.
REFERENCE_LIBRARY_NAME = "tvtl_ael_reference_lib"
# The reference workspace directory the AEL deck opens first so it
# self-bootstraps when ADS is launched with no workspace open.
REFERENCE_WORKSPACE_PATH = os.environ.get("TVTL_AEL_REFERENCE_WORKSPACE", "<ADS_REFERENCE_WORKSPACE>")
_LEVEL_DIR_RE = re.compile(r"ads_builder_v2_level_(\d+)")


def generate_ads_workspace(
    level_results_dir: str | Path,
    canonical_circuits: Sequence[Any],
    workspace_name: str = DEFAULT_WORKSPACE_NAME,
    workspace_path: str = REFERENCE_WORKSPACE_PATH,
) -> dict[str, Any]:
    """Emit the inspection artifacts for one parity level.

    Parameters
    ----------
    level_results_dir : str or Path
        The level's parity-root directory
        (``<output_root>/ads_builder_v2_level_<N>``).
    canonical_circuits : sequence of DeviceSpec
        The circuits of the level. Each must expose ``.label`` and be a
        valid device spec.
    workspace_name : str
        Subdirectory under ``level_results_dir`` for the per-level
        README.
    workspace_path : str
        The reference ADS workspace directory the AEL deck opens to
        self-bootstrap. Defaults to :data:`REFERENCE_WORKSPACE_PATH`.

    Returns
    -------
    dict
        ``{"workspace_dir": Path, "readme_path": Path,
        "net_paths": list[Path], "deck_paths": list[Path],
        "circuit_labels": list[str], "label": str}``.
    """

    level_results_dir = Path(level_results_dir)
    workspace_dir = level_results_dir / workspace_name
    workspace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    net_paths: list[Path] = []
    deck_paths: list[Path] = []
    circuit_labels: list[str] = []
    for circuit in canonical_circuits:
        label = str(circuit.label)
        circuit_labels.append(label)
        circuit_dir = level_results_dir / label
        circuit_dir.mkdir(parents=True, exist_ok=True)

        net_path = circuit_dir / f"{label}.net"
        net_path.write_text(
            emit_ads_native_netlist(circuit), encoding="utf-8", newline="\n"
        )
        net_paths.append(net_path)

        plan = build_schematic_plan(circuit)
        deck = render_ael_script(
            REFERENCE_LIBRARY_NAME, label, plan, timestamp, workspace_path
        )
        deck_path = circuit_dir / "build_workspace.ael"
        # Plain ASCII, no BOM, consistent LF endings.
        deck_path.write_text(deck, encoding="utf-8", newline="\n")
        deck_paths.append(deck_path)

        (circuit_dir / "open_workspace.bat").write_text(
            OPENER_BAT, encoding="utf-8"
        )

    level_number = _level_number(level_results_dir)
    readme_path = workspace_dir / "README.md"
    readme_path.write_text(
        _render_readme(
            level_number=level_number,
            circuit_labels=circuit_labels,
            level_results_dir=level_results_dir,
            timestamp=timestamp,
        ),
        encoding="utf-8",
    )

    return {
        "workspace_dir": workspace_dir,
        "readme_path": readme_path,
        "net_paths": net_paths,
        "deck_paths": deck_paths,
        "circuit_labels": circuit_labels,
        "label": workspace_name,
    }


def _level_number(level_results_dir: Path) -> str:
    """Extract the ladder level number from the results-dir name."""

    match = _LEVEL_DIR_RE.search(level_results_dir.name)
    return match.group(1) if match else "?"


def _parity_status_map(level_results_dir: Path) -> dict[str, str]:
    """Map each circuit label to a human-readable parity-status string."""

    status_path = level_results_dir / "inspection_status.json"
    if not status_path.is_file():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, str] = {}
    for circuit in data.get("circuits", []):
        label = str(circuit.get("label", ""))
        verdict = str(circuit.get("verdict", "")).upper() or "UNKNOWN"
        delta = circuit.get("max_abs_delta_dB")
        if isinstance(delta, (int, float)):
            out[label] = f"{verdict} ({float(delta):.3f} dB)"
        else:
            out[label] = verdict
    return out


def _render_readme(
    *,
    level_number: str,
    circuit_labels: Sequence[str],
    level_results_dir: Path,
    timestamp: str,
) -> str:
    """Render the per-level ``README.md``."""

    status_map = _parity_status_map(level_results_dir)

    lines: list[str] = []
    lines.append(f"# ADS Inspection Schematics -- Level {level_number}\n\n")
    lines.append(f"{README_TIMESTAMP_MARKER} {timestamp} -->\n\n")
    lines.append(
        f"Per-circuit ADS schematic builders for the Level {level_number} "
        "canonical circuits.\n\n"
    )

    lines.append("## How to view a circuit's schematic in ADS\n\n")
    lines.append(
        "Each circuit's results directory (`../<circuit>/`) contains a\n"
        "`build_workspace.ael` deck and an `open_workspace.bat` launcher.\n"
        "Double-click the circuit's `open_workspace.bat` (or run the deck\n"
        "via `Tools > Command Line` in ADS). The deck creates a schematic\n"
        "cellview in the reference workspace's library "
        f"(`{REFERENCE_LIBRARY_NAME}`)\nand places + wires every element of "
        "that circuit.\n\n"
    )
    lines.append(
        "**Element parameter values are not set** by the deck -- the\n"
        "schematic shows the correct *topology* with ADS default values\n"
        "(MLIN width/length, substrate, sweep range are defaults). The\n"
        "circuit's exact dimensions are in its `<circuit>.net` (ADS-native\n"
        "netlist) and `device_spec.json`. Parameter substitution is a\n"
        "planned follow-up.\n\n"
    )
    lines.append(
        "The deck requires the reference workspace at\n"
        "`<OUTPUT_DIR>\\ael_reference\\"
        "tvtl_ael_reference_wrk\\` to exist.\n\n"
    )

    lines.append("## Circuits\n\n")
    lines.append("| Circuit | Schematic deck | Parity Status | Topology |\n")
    lines.append("|---------|----------------|---------------|----------|\n")
    for label in circuit_labels:
        status = status_map.get(label, "n/a")
        topology = metadata_for(label).topology_description.replace("|", r"\|")
        lines.append(
            f"| {label} | `../{label}/build_workspace.ael` | {status} "
            f"| {topology} |\n"
        )
    lines.append("\n")

    lines.append("## Regeneration\n\n")
    lines.append("```\n")
    lines.append(
        "PS> scripts\\run_ads_inspection_workspace_for_level.ps1 "
        f"-Level {level_number}\n"
    )
    lines.append("```\n\n")
    lines.append(
        "The decks and `.net` files are emitted automatically by every "
        "parity-gate run.\n"
    )
    return "".join(lines)


def _canonical_circuits_for_level(level: int) -> Sequence[Any]:
    """Return the canonical circuits for a ladder level (lazy import)."""

    from ..parity import (
        LEVEL_1_CANONICAL_CIRCUITS,
        LEVEL_2_CANONICAL_CIRCUITS,
        LEVEL_3_CANONICAL_CIRCUITS,
    )

    table = {
        1: LEVEL_1_CANONICAL_CIRCUITS,
        2: LEVEL_2_CANONICAL_CIRCUITS,
        3: LEVEL_3_CANONICAL_CIRCUITS,
    }
    if level not in table:
        raise SystemExit(
            f"no canonical circuits registered for level {level} "
            f"(known levels: {sorted(table)})"
        )
    return table[level]


def _existing_levels(output_root: Path) -> list[int]:
    """Return the ladder levels that have a parity-root directory."""

    levels: list[int] = []
    if not output_root.is_dir():
        return levels
    for entry in sorted(output_root.iterdir()):
        match = _LEVEL_DIR_RE.fullmatch(entry.name) if entry.is_dir() else None
        if match:
            levels.append(int(match.group(1)))
    return sorted(set(levels))


def main(argv: list[str] | None = None) -> int:
    """CLI: emit ADS inspection artifacts for one level or all.

    Returns ``0`` on success, ``2`` if no matching level directory exists.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Emit ADS inspection artifacts (ADS-native .net files, "
            "element-placement AEL decks, launchers, README) for the ADS "
            "Builder v2 parity levels. Writes files only; never launches ADS."
        )
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--level",
        default="all",
        help="Ladder level number, or 'all' (default) for every level "
        "with a results directory under output_root.",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).resolve()
    if str(args.level).lower() == "all":
        levels = _existing_levels(output_root)
    else:
        levels = [int(args.level)]
    if not levels:
        print(f"No ads_builder_v2_level_* directory found under {output_root}")
        return 2

    for level in levels:
        level_dir = output_root / f"ads_builder_v2_level_{level}"
        if not level_dir.is_dir():
            print(f"Level {level}: results directory missing ({level_dir}); skipped.")
            continue
        circuits = _canonical_circuits_for_level(level)
        info = generate_ads_workspace(level_dir, list(circuits))
        print(
            f"Level {level}: {len(info['deck_paths'])} schematic decks "
            f"+ {len(info['net_paths'])} .net files"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




