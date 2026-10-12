"""CLI: OA-Python <-> hpeesofsim netlist cross-check (D0-D4).

::

    python -m tvtl_ads_builder_v2.oa.oa_diagnostic_cli all
        For every diagnostic d0..d4: build the OA workspace, auto-generate
        its ADS netlist, emit the hpeesofsim flat netlist, cross-check the
        two, and confirm the hpeesofsim<->TVTL leg. Print a summary table.

    python -m tvtl_ads_builder_v2.oa.oa_diagnostic_cli d3
        Same, for a single diagnostic.

    python -m tvtl_ads_builder_v2.oa.oa_diagnostic_cli d3 --manual
        Build the OA workspace + hpeesofsim deck for one diagnostic and
        print manual ADS-inspection instructions (no auto cross-check).

The ADS netlist is generated automatically via the Python API
(``design.generate_netlist()``); the earlier manual ``Tools -> Generate
Netlist`` step is no longer needed. ``--compare`` is kept as a
deprecated no-op alias.

Exit code: 0 if every cross-check matched, 1 otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_DEFAULT_OUTPUT_ROOT = Path.cwd() / "runs"

# Short CLI name -> DIAGNOSTIC_SPECS key.
_SHORT_TO_FULL = {
    "d0": "d0_bare_ports",
    "d1": "d1_single_mlin",
    "d2": "d2_two_mlins",
    "d3": "d3_mtee_short_stub",
    "d4": "d4_full_level3",
}
_SHORT_ORDER = ("d0", "d1", "d2", "d3", "d4")


def _circuit_dir(output_root: str | Path, short: str) -> Path:
    return Path(output_root) / "oa_diagnostics" / short


def _cross_check(ads_path: Path, hpeesofsim_path: Path):
    """Parse both netlists and compare them (ADS-free)."""

    from .netlist_compare import (
        compare_circuits,
        parse_ads_netlist,
        parse_hpeesofsim_deck,
    )

    return compare_circuits(
        parse_ads_netlist(ads_path),
        parse_hpeesofsim_deck(hpeesofsim_path),
    )


def _leg_b(full_id: str, circuit_dir: Path) -> str:
    """hpeesofsim <-> TVTL via the existing v2 parity gate (ADS-required).

    Returns a short cell for the summary table's Path B column.
    """

    try:
        from ..parity.ael_diagnostics.generate import (
            PATH1_TOLERANCE_DB,
            resolve_ads_environment,
        )
        from ..parity.ael_diagnostics.specs import DIAGNOSTIC_SPECS
        from ..parity.check import parity_check

        result = parity_check(
            DIAGNOSTIC_SPECS[full_id](),
            circuit_dir,
            ads_environment=resolve_ads_environment(),
            tolerance_dB_S21=PATH1_TOLERANCE_DB,
            level_label=full_id,
        )
        reason = str(result.get("reason", ""))
        if reason == "within_tolerance":
            return f"{float(result.get('max_abs_delta_dB', float('nan'))):.3f} dB"
        if reason == "exceeds_tolerance":
            return f"FAIL ({float(result.get('max_abs_delta_dB', float('nan'))):.3f} dB)"
        if reason == "ads_blocked":
            return "BLOCKED (no ADS)"
        return reason or "error"
    except Exception as exc:  # noqa: BLE001 -- leg B is the ADS-required path
        return f"error: {type(exc).__name__}"


def _process_one(short: str, output_root: str | Path) -> dict:
    """Build, cross-check and validate one diagnostic; return a row dict."""

    full_id = _SHORT_TO_FULL[short]
    circuit_dir = _circuit_dir(output_root, short)
    circuit_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "circuit": full_id,
        "crosscheck": "ERROR",
        "path_a": "-",
        "path_b": "-",
        "differences": [],
    }

    try:
        from .oa_emitter import emit_diagnostic_workspace

        emit_diagnostic_workspace(full_id, circuit_dir / "oa_workspace")
    except Exception as exc:  # noqa: BLE001 -- a build failure is one ERROR row
        row["path_a"] = f"emit failed ({type(exc).__name__})"
        return row

    ads_path = circuit_dir / "ads_generated_netlist.txt"
    hpeesofsim_path = circuit_dir / "hpeesofsim.ckt"
    if not ads_path.is_file():
        row["path_a"] = "ADS netlist not generated"
        return row

    result = _cross_check(ads_path, hpeesofsim_path)
    row["crosscheck"] = "MATCH" if result.match else "MISMATCH"
    row["path_a"] = (
        "0 differences" if result.match
        else f"{len(result.differences)} difference(s)"
    )
    row["differences"] = result.differences
    row["path_b"] = _leg_b(full_id, circuit_dir)
    return row


def _print_table(rows: list[dict]) -> None:
    """Print the cross-check summary table and any mismatch detail."""

    head = (
        f"  {'Circuit':<20} | {'Cross-check':<9} | "
        f"{'Path A (OA==hpeesofsim)':<22} | Path B (hpeesofsim==TVTL)"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for row in rows:
        print(
            f"  {row['circuit']:<20} | {row['crosscheck']:<9} | "
            f"{row['path_a']:<22} | {row['path_b']}"
        )
    for row in rows:
        if row["crosscheck"] == "MISMATCH" and row["differences"]:
            print(f"\n  {row['circuit']} -- circuit-level differences:")
            for diff in row["differences"]:
                print(f"    - {diff}")


def _run_cross_check(circuit: str, output_root: str | Path) -> int:
    """Run the cross-check for one circuit or all of them."""

    shorts = _SHORT_ORDER if circuit == "all" else (circuit,)
    rows = [_process_one(short, output_root) for short in shorts]
    _print_table(rows)
    passed = all(row["crosscheck"] == "MATCH" for row in rows)
    print()
    if passed:
        print("All cross-checks passed: every OA workspace is netlist-"
              "equivalent to its hpeesofsim flat deck.")
    else:
        print("Some cross-checks did not pass -- see the rows above. Use "
              "`<circuit> --manual` to inspect a failing one visually.")
    return 0 if passed else 1


def _run_manual(short: str, output_root: str | Path) -> int:
    """Build one workspace for manual ADS inspection (no auto cross-check)."""

    full_id = _SHORT_TO_FULL[short]
    circuit_dir = _circuit_dir(output_root, short)
    circuit_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = circuit_dir / "oa_workspace"
    try:
        from .oa_emitter import emit_diagnostic_workspace

        info = emit_diagnostic_workspace(full_id, workspace_dir, auto_netlist=False)
        print(f"OA workspace            : {info['workspace_path']}")
        print(f"hpeesofsim flat netlist : {info['hpeesofsim_netlist_path']}")
    except Exception as exc:  # noqa: BLE001 -- ADS-free fallback
        print(f"OA workspace            : NOT built ({type(exc).__name__}: {exc})")
        print("  The OA workspace step needs an ADS installation.")

    print()
    print("Manual verification:")
    print(f"  1. Open the OA workspace in ADS: {workspace_dir}")
    print(f"  2. Inspect the {full_id}_oa_test schematic visually.")
    print(
        f"  3. Tools -> Generate Netlist; save it to "
        f"{circuit_dir / 'ads_generated_netlist.txt'}"
    )
    print(
        f"  4. Re-run the automated cross-check:  python -m "
        f"tvtl_ads_builder_v2.oa.oa_diagnostic_cli {short}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""

    parser = argparse.ArgumentParser(
        prog="oa_diagnostic_cli",
        description=(
            "OA-Python <-> hpeesofsim netlist cross-check (diagnostics D0-D4)."
        ),
    )
    parser.add_argument(
        "circuit",
        choices=("all",) + _SHORT_ORDER,
        help="diagnostic to process: d0, d1, d2, d3, d4, or 'all'",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="build the workspace for manual ADS inspection (single circuit)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="(deprecated) no-op -- the cross-check now runs automatically",
    )
    parser.add_argument(
        "--output_root",
        default=str(_DEFAULT_OUTPUT_ROOT),
        help="root directory for oa_diagnostics/ outputs",
    )
    args = parser.parse_args(argv)

    if args.compare:
        print(
            "note: --compare is deprecated and ignored -- the cross-check "
            "now runs automatically (the ADS netlist is generated via the "
            "Python API)."
        )

    if args.manual:
        if args.circuit == "all":
            parser.error("--manual operates on a single circuit, not 'all'")
        return _run_manual(args.circuit, args.output_root)

    return _run_cross_check(args.circuit, args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())



