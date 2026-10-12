"""CLI: OA-Python <-> hpeesofsim cross-check for the Level-8 cascades.

::

    python -m tvtl_ads_builder_v2.design.canonical_cli

For each Level-8 canonical (the topology-driven TVTL cascades) this
builds the OA workspace, emits the hpeesofsim flat netlist,
cross-checks the two (Path A), runs the SDD2P / VAR / two-tone-HB /
P_1Tone content checks, confirms the hpeesofsim<->TVTL leg (Path B),
and prints a summary table -- exactly the Level-7 verdict shape.

Why a separate CLI rather than a ``level8`` entry on
``oa.oa_canonical_cli``: the Level-8 canonicals are built from the
design API, so registering them on ``oa_canonical_cli`` would make a
file outside ``design/`` depend on ``design/``, against the
load-bearing dependency rule. Instead this CLI -- which IS inside
``design/`` -- reuses ``oa_canonical_cli``'s per-circuit machinery
(``design/`` may import ``oa/``; the reverse is forbidden) and
registers Level 8 into its per-level table at import time.
``oa.oa_canonical_cli`` itself is left byte-for-byte unmodified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..oa import oa_canonical_cli
from ..parity.tolerance import PARITY_TOLERANCE_DB_S21_LEVEL_8
from .canonicals import LEVEL_8_CANONICAL_CIRCUITS, LEVEL_8_LABEL

_DEFAULT_OUTPUT_ROOT = Path.cwd() / "runs" / "canonical_workspaces"

# The level key this CLI processes.
_LEVEL_KEY = "level8"

# Register Level 8 into oa_canonical_cli's per-level table so the reused
# _process_circuit / _print_level helpers can resolve "level8". This is
# the allowed dependency direction -- design/ reaching into oa/ -- and
# only ADDS an entry; the existing Level 1-7 entries are untouched, so
# `python -m ...oa.oa_canonical_cli` is unaffected.
oa_canonical_cli._LEVELS[_LEVEL_KEY] = (
    LEVEL_8_CANONICAL_CIRCUITS,
    LEVEL_8_LABEL,
    PARITY_TOLERANCE_DB_S21_LEVEL_8,
)


def _run(output_root: str | Path) -> int:
    """Process every Level-8 canonical; print the table; return exit code."""

    rows = [
        oa_canonical_cli._process_circuit(_LEVEL_KEY, spec, output_root)
        for spec in LEVEL_8_CANONICAL_CIRCUITS
    ]
    oa_canonical_cli._print_level(_LEVEL_KEY, rows)
    n_pass = sum(1 for row in rows if row["passed"])
    total = len(rows)
    if n_pass == total:
        print(
            f"All {total} Level-8 canonical circuit(s) passed: every OA "
            f"workspace is netlist-equivalent to its hpeesofsim flat deck."
        )
    else:
        print(
            f"{n_pass}/{total} Level-8 canonical circuit(s) passed -- see "
            f"the rows above."
        )
    return 0 if n_pass == total else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""

    parser = argparse.ArgumentParser(
        prog="design_canonical_cli",
        description=(
            "OA-Python <-> hpeesofsim cross-check for the Level-8 "
            "topology-driven TVTL cascades."
        ),
    )
    parser.add_argument(
        "--output_root",
        default=str(_DEFAULT_OUTPUT_ROOT),
        help="root directory for oa_canonical/ outputs",
    )
    args = parser.parse_args(argv)
    return _run(args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())



