"""Master CLI for the AEL diagnostic ladder D0-D4.

Usage::

    python -m tvtl_ads_builder_v2.parity.ael_diagnostics_cli generate

``generate`` builds all five diagnostic workspaces under
``<output_root>/ael_diagnostics/``, runs and validates Path 1
(TVTL analytical vs hpeesofsim) for each, and prints a summary table.

A failed or blocked Path 1 is surfaced clearly: it is a flat-netlist /
TVTL / ADS-runtime problem, not an AEL-builder problem, and the user
should not inspect the AEL schematic until Path 1 passes.

Exit code: ``0`` if every diagnostic's Path 1 passes, ``3`` if any
Path 1 exceeds tolerance, ``4`` if any Path 1 was blocked (no ADS
runtime) but none failed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .ael_diagnostics.generate import generate_all

DEFAULT_OUTPUT_ROOT = Path.cwd() / "runs"


def _status_cell(row: dict) -> str:
    """Render the ``Path1 status`` column for one diagnostic."""

    status = str(row["path1_status"])
    if status in ("PASS", "FAIL"):
        delta = row.get("max_abs_delta_dB")
        try:
            value = float(delta)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            return f"{status} ({value:.3f} dB)"
    return status


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AEL diagnostic ladder D0-D4: generate per-circuit workspaces "
            "and validate Path 1 (TVTL vs hpeesofsim)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser(
        "generate",
        help="Generate all five diagnostic workspaces and validate Path 1.",
    )
    generate.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the AEL diagnostics CLI.

    Parameters
    ----------
    argv : list of str or None
        Optional argv override (used by tests).

    Returns
    -------
    int
        ``0`` all Path 1 passed, ``3`` any Path 1 failed, ``4`` any
        Path 1 blocked (and none failed).
    """

    args = _build_arg_parser().parse_args(argv)
    if args.command != "generate":  # pragma: no cover - argparse enforces
        return 2

    output_root = Path(args.output_root).resolve()
    rows = generate_all(output_root)

    name_width = max(len(row["name"]) for row in rows)
    status_width = max(len(_status_cell(row)) for row in rows)
    status_width = max(status_width, len("Path1 status"))
    header = (
        f"  {'Circuit':<{name_width}} | {'Path1 status':<{status_width}} | "
        "Workspace"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        print(
            f"  {row['name']:<{name_width}} | "
            f"{_status_cell(row):<{status_width}} | {row['circuit_dir']}"
        )
    print()

    any_fail = any(row["path1_status"] == "FAIL" for row in rows)
    any_error = any(row["path1_status"] == "ERROR" for row in rows)
    any_blocked = any(row["path1_status"] == "BLOCKED" for row in rows)
    if any_fail or any_error:
        print(
            "Path 1 FAILED for at least one diagnostic. This is a "
            "flat-netlist-emitter or TVTL-analytical problem, NOT an "
            "AEL-builder problem -- see each path1_parity_report.txt."
        )
        return 3
    if any_blocked:
        print(
            "Path 1 was BLOCKED for at least one diagnostic (no ADS "
            "runtime found). Set HPEESOF_DIR and regenerate to complete "
            "Path 1 before inspecting the AEL schematics."
        )
        return 4
    print(
        "All Path 1 checks passed. Open each open_workspace.bat in ADS, "
        "inspect the schematic, and report any wiring failure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



