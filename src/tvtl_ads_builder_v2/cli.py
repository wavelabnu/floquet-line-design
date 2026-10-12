"""Command-line interface for public ADS Builder v2 workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .amplifier_oa_export import export_amplifier_oa
from .circuit_spec import SelfPumpedCircuitSpec
from .gui_export import ExportError, export_gui_project
from .validation.gain_comparison import format_visible_summary, run_gain_comparison
from .validation.hb_convergence_ladder import (
    parse_cell_list,
    parse_hb_order_list,
    parse_pump_dbm_list,
    run_hb_convergence_ladder,
    simulation_runtime_available,
)
from .validation.pump_power_sweep import SUPPORTED_SWEEP_METRICS, run_pump_power_sweep
from .validation.compare_self_pumped import run_self_pumped_validation
from .validation.validate_amplifier_hb import hb_status_is_strict_failure, run_validate_amplifier_hb


def _positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return out


def _cell_list(value: str) -> list[int]:
    try:
        return parse_cell_list(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _pump_dbm_list(value: str) -> list[float]:
    try:
        return parse_pump_dbm_list(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _hb_order_list(value: str) -> list[int]:
    try:
        return parse_hb_order_list(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _run_export_gui_project(args: argparse.Namespace) -> int:
    manifest = export_gui_project(
        args.input,
        args.out,
        candidate_index=args.candidate_index,
        candidate_name=args.candidate_name,
        all_candidates=args.all,
        backend=args.backend,
    )
    candidates = manifest["candidates"]
    print(f"Exported {len(candidates)} candidate project(s).")
    for candidate in candidates:
        print(f"Project path: {candidate['project_dir']}")
        if candidate["oa_workspace"]:
            print(f"ADS workspace to open: {candidate['oa_workspace']}")
        else:
            print(
                "ADS workspace to open: run build_oa_workspace.bat with ADS "
                "Python, then open the generated oa_workspace directory."
            )
        print(f"Open instructions: {candidate['files']['open_in_ads']}")
    print(f"Manifest: {Path(args.out) / 'export_manifest.json'}")
    return 0


def _run_export_amplifier_oa(args: argparse.Namespace) -> int:
    manifest = export_amplifier_oa(
        args.input,
        args.out,
        candidate_index=args.candidate_index,
        candidate_name=args.candidate_name,
        simulate=args.simulate,
        ads_home=args.ads_home,
        gain_comparison_report=args.gain_comparison_report,
        gain_comparison_json=args.gain_comparison_json,
        pump_sweep_report=args.pump_sweep_report,
        pump_sweep_json=args.pump_sweep_json,
        pump_sweep_plot=args.pump_sweep_plot,
    )
    summary = manifest["candidate_summary"]
    print(f"Candidate: {summary['candidate_name']}")
    print(f"Target frequency: {summary['target_frequency_GHz']:g} GHz")
    print(f"Mode: {manifest['mode']}")
    print(f"Sim-ready amplifier: {manifest['sim_ready']}")
    if manifest.get("unit_cell_internal_status") == "implemented":
        print(
            "Generated connected ADS GUI scaffold with implemented unit-cell "
            "internals; full amplifier simulation validation is not claimed."
        )
    else:
        print(
            "Generated connected ADS GUI scaffold only; unit-cell internals "
            "are placeholders."
        )
    print(f"Project path: {manifest['project_dir']}")
    print(f"ADS workspace to open: {manifest['oa_workspace']}")
    print(f"Generated netlist: {manifest['generated_netlist_path']}")
    print(f"Plan: {manifest['plan_path']}")
    print(f"Simulation: {manifest['simulation_status']}")
    return 0


def _run_validate_self_pumped(args: argparse.Namespace) -> int:
    spec = SelfPumpedCircuitSpec(fs_hz=float(args.freq_ghz) * 1.0e9)
    report = run_self_pumped_validation(
        spec,
        args.out,
        simulate=args.simulate,
        draw_oa=args.draw_oa,
        oa_workspace_dir=args.oa_workspace,
        ads_home=args.ads_home,
    )

    analytical_status = "passed" if report["analytical"]["expected_matched"] else "error"
    direct_status = report["simulation"]["direct_hpeesofsim"]["status"]
    oa = report["simulation"]["ads_oa_hpeesofsim"]
    oa_status = oa.get("status", "not_run")
    oa_dataset_status = oa.get("dataset_status", "not_run")
    topology_status = report["topology_comparison"]["status"]
    numerical_status = report["numerical_comparison"]["status"]

    print(f"Analytical: {analytical_status}")
    print(f"Direct hpeesofsim: {direct_status}")
    print(f"ADS OA schematic generation: {oa_status}")
    print(f"ADS OA hpeesofsim: {oa_dataset_status}")
    print(f"Topology comparison: {topology_status}")
    print(f"Numerical comparison: {numerical_status}")
    print(f"Report: {Path(args.out) / 'comparison_report.json'}")

    required_statuses = [analytical_status, topology_status]
    if args.simulate:
        required_statuses.append(direct_status)
    if args.draw_oa:
        required_statuses.append(oa_status)
        if args.simulate:
            required_statuses.append(oa_dataset_status)
    return 0 if all(status == "passed" for status in required_statuses) else 2


def _run_compare_gain(args: argparse.Namespace) -> int:
    report = run_gain_comparison(
        args.input,
        args.out,
        simulate=args.simulate,
        draw_oa=args.draw_oa,
        ads_home=args.ads_home,
        oa_project_dir=args.oa_project_dir,
    )
    print(format_visible_summary(report))
    print("")
    audit = report["ads_element_audit"]
    direct_status = _cli_gain_status(report["numerical_direct"]["status"])
    oa_status = _cli_gain_status(report["numerical_oa"]["status"])
    mlef = audit.get("MLEF_cellview_validated")
    viagnd = audit.get("VIAGND_cellview_validated")
    lv = "absent" if not audit.get("LVIA_present") else "present"
    mloc = "absent" if not audit.get("MLOC_present") else "present"

    print(f"Analytical gain metrics: {report['analytical']['status']}")
    print(f"Direct hpeesofsim gain metrics: {direct_status}")
    print(f"ADS OA hpeesofsim gain metrics: {oa_status}")
    print(
        "ADS element audit: "
        f"LVIA {lv}, MLOC {mloc}, "
        f"MLEF validated={mlef}, VIAGND validated={viagnd}"
    )
    print(f"Comparison report: {Path(args.out) / 'gain_comparison_report.json'}")
    print(f"Simulation-ready amplifier: {report['numerical_oa'].get('simulation_ready')}")
    if not report["numerical_oa"].get("simulation_ready"):
        print(
            "Full amplifier gain comparison is preliminary or unavailable; "
            "unit-cell OA internals require validated MLEF/VIAGND mapping."
        )
    if args.strict_convergence and _gain_report_has_strict_hb_failure(report):
        print("Strict convergence: failed because one or more requested HB results did not converge.")
        return 2
    if args.simulate:
        print(
            "Exit status: success because non-convergence is allowed; "
            "use --strict-convergence to fail."
        )
    return 0


def _run_sweep_pump_gain(args: argparse.Namespace) -> int:
    report = run_pump_power_sweep(
        args.input,
        args.out,
        pump_start_dbm=args.pump_start_dbm,
        pump_stop_dbm=args.pump_stop_dbm,
        pump_step_dbm=args.pump_step_dbm,
        metrics=args.metric,
        simulate=args.simulate,
        draw_oa=args.draw_oa,
        analytical_only=args.analytical_only,
        plot_y_min=args.plot_y_min,
        plot_y_max=args.plot_y_max,
        clip_plot_db=args.clip_plot_db,
        invalid_gain_floor_db=args.invalid_gain_floor_db,
    )
    best = report.get("best_gain_point")
    flagged_count = len(report.get("plot", {}).get("flagged_points", []))
    print("Pump power sweep summary")
    print("------------------------")
    print(f"candidate: {report['candidate']}")
    print(f"fs: {float(report['fs_hz']) / 1.0e9:.3f} GHz")
    print(f"architecture: {report['architecture']}")
    print("metric: analytical coherent gain")
    if best is None:
        print("best point: unavailable")
    else:
        print(
            "best point: "
            f"{float(best['gain_db']):.2f} dB at Ppump = {float(best['pump_dbm']):.2f} dBm"
        )
    print(f"flagged points: {flagged_count} null/singular")
    print(f"plot: {report['paths']['png']}")
    print(f"numerical sweep: {report['numerical_sweep_status']}")
    print("reports:")
    print(f"  {report['paths']['csv']}")
    print(f"  {report['paths']['json']}")
    print(f"  {report['paths']['markdown']}")
    if args.strict_convergence and _pump_sweep_has_strict_hb_failure(report):
        print("Strict convergence: failed because one or more requested HB points did not converge.")
        return 2
    return 0


def _run_validate_amplifier_hb(args: argparse.Namespace) -> int:
    report = run_validate_amplifier_hb(
        args.input,
        args.out,
        simulate=args.simulate,
        ads_home=args.ads_home,
        candidate_index=args.candidate_index,
        candidate_name=args.candidate_name,
    )
    print(f"Candidate: {report['candidate']}")
    print(f"Direct deck: {report['direct_deck']['path']}")
    print(f"ADS OA netlist: {report['ads_oa']['netlist_path']}")
    print(f"Netlist comparison: {report['netlist_comparison']['status']}")
    availability = report["gain_comparison_report"].get("numerical_availability", {})
    direct_status = availability.get("direct_hb_status", report["simulation"]["direct_hpeesofsim"]["status"])
    oa_status = availability.get("ads_oa_hb_status", report["simulation"]["ads_oa_hpeesofsim"]["status"])
    print(f"Direct HB: {_hb_cli_phrase(str(direct_status))}")
    print(f"ADS-OA HB: {_hb_cli_phrase(str(oa_status))}")
    print(f"sim_ready: {report['sim_ready']}")
    print(f"Gain report: {report['paths']['gain_comparison_json']}")
    if not report["netlist_comparison"].get("match"):
        return 2
    strict_failure = hb_status_is_strict_failure(str(direct_status)) or hb_status_is_strict_failure(str(oa_status))
    if args.strict_convergence and strict_failure:
        print("Exit status: failure because --strict-convergence was requested.")
        return 2
    if args.simulate and strict_failure:
        print(
            "Exit status: success because non-convergence is allowed; "
            "use --strict-convergence to fail."
        )
    return 0


def _run_hb_convergence_ladder(args: argparse.Namespace) -> int:
    if args.simulate and args.dry_run:
        print("error: --simulate and --dry-run are mutually exclusive", file=sys.stderr)
        return 2
    if args.simulate and not simulation_runtime_available(args.ads_home):
        print(
            "error: --simulate requires an ADS/hpeesofsim runtime. "
            "Run with Keysight ADS Python, set HPEESOF_DIR, or pass --ads-home.",
            file=sys.stderr,
        )
        return 2
    report = run_hb_convergence_ladder(
        args.input,
        args.out,
        cells=args.cells,
        pump_dbm=args.pump_dbm,
        hb_orders=args.hb_orders,
        simulate=args.simulate,
        dry_run=args.dry_run,
        stop_on_first_fail=args.stop_on_first_fail,
        candidate_index=args.candidate_index,
        candidate_name=args.candidate_name,
        include_oa=args.include_oa,
        timeout_s=args.timeout_s,
        ads_home=args.ads_home,
        solver_options={
            "max_iterations": args.max_iterations,
            "oversample": args.oversample,
            "tolerance": args.tolerance,
            "use_previous_solution": args.use_previous_solution,
            "solver_mode": args.solver_mode,
        },
    )
    counts = report["summaries"]["status_counts"]
    print("HB convergence ladder summary")
    print("-----------------------------")
    print(f"candidate: {report['candidate']}")
    print(f"architecture: {report['architecture']}")
    print(f"points: {len(report['results'])}")
    print(
        "status: "
        f"converged={counts.get('converged', 0)}, "
        f"failed={counts.get('failed', 0)}, "
        f"error={counts.get('error', 0)}, "
        f"skipped={counts.get('skipped', 0)}"
    )
    print(f"first failure: {report['first_failure'] or 'none'}")
    print(f"largest converged: {report['largest_converged_case'] or 'none'}")
    print("reports:")
    print(f"  {report['paths']['json']}")
    print(f"  {report['paths']['csv']}")
    print(f"  {report['paths']['markdown']}")
    print(f"  {report['paths']['png']}")
    if args.strict_convergence and any(
        row.get("status") in {"failed", "error"} for row in report["results"]
    ):
        print("Strict convergence: failed because one or more ladder points failed/error.")
        return 2
    return 0


def _cli_gain_status(status: str) -> str:
    if status in {"passed", "skipped", "hb_converged", "hb_no_convergence", "hb_error", "hb_skipped"}:
        return status
    if status == "preliminary":
        return "unavailable"
    return "unavailable"


def _hb_cli_phrase(status: str) -> str:
    if status == "hb_converged":
        return "converged -> numerical metrics available"
    if status == "hb_no_convergence":
        return "no convergence -> numerical metrics NaN"
    if status == "hb_skipped":
        return "skipped -> numerical metrics NaN"
    return f"{status} -> numerical metrics NaN"


def _gain_report_has_strict_hb_failure(report: Mapping[str, Any]) -> bool:
    direct = str(report.get("numerical_direct", {}).get("status") or "")
    oa = str(report.get("numerical_oa", {}).get("status") or "")
    return hb_status_is_strict_failure(direct) or hb_status_is_strict_failure(oa)


def _pump_sweep_has_strict_hb_failure(report: Mapping[str, Any]) -> bool:
    for row in report.get("rows", []):
        direct = str(row.get("direct_hb_status") or "")
        oa = str(row.get("oa_hb_status") or "")
        if hb_status_is_strict_failure(direct) or hb_status_is_strict_failure(oa):
            return True
    return False


def add_export_gui_project_parser(
    subparsers: argparse._SubParsersAction,
    *,
    name: str = "export-gui-project",
    help_text: str = "Export optimizer candidates for ADS GUI inspection.",
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        name,
        help=help_text,
        description=(
            "Read optimizer summary.json output and write ADS GUI export "
            "artifacts. Normal Python writes a portable project scaffold. "
            "Run with ADS Python, or use --backend oa, to materialize a real "
            "OpenAccess workspace."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Optimizer summary.json path.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for exported project files.")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--candidate-index", type=_positive_int, help="1-based candidate index from candidates.json.")
    choice.add_argument("--candidate-name", help="Candidate topology/name to export.")
    choice.add_argument("--all", action="store_true", help="Export every selected optimizer candidate.")
    parser.add_argument(
        "--backend",
        choices=("auto", "scaffold", "oa"),
        default="auto",
        help=(
            "auto: create OA workspace when ADS Python is available, otherwise "
            "write scaffold; scaffold: never import ADS APIs; oa: require ADS "
            "OpenAccess APIs."
        ),
    )
    parser.set_defaults(func=_run_export_gui_project)
    return parser


def add_export_amplifier_oa_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "export-amplifier-oa",
        help="Export a self-pumped optimizer amplifier candidate as ADS OA.",
        description=(
            "Read optimizer summary.json output and create a connected ADS "
            "OpenAccess amplifier scaffold using the verified snap-point OA "
            "drawing method."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Optimizer summary.json path.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the ADS OA project.")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--candidate-index", type=_positive_int, help="1-based candidate index from candidates.json.")
    choice.add_argument("--candidate-name", help="Candidate topology/name to export.")
    parser.add_argument("--simulate", action="store_true", help="Request simulation when the export is sim-ready.")
    parser.add_argument("--ads-home", type=Path, help="ADS install root used to locate Keysight Python packages.")
    parser.add_argument("--gain-comparison-report", type=Path, help="Existing gain_comparison_report.md to copy into the ADS project.")
    parser.add_argument("--gain-comparison-json", type=Path, help="Existing gain_comparison_report.json to copy into the ADS project.")
    parser.add_argument("--pump-sweep-report", type=Path, help="Existing pump_power_sweep_report.md to copy into the ADS project.")
    parser.add_argument("--pump-sweep-json", type=Path, help="Existing pump_sweep.json to copy into the ADS project.")
    parser.add_argument("--pump-sweep-plot", type=Path, help="Existing gain_vs_pump_power.png to copy into the ADS project.")
    parser.set_defaults(func=_run_export_amplifier_oa)
    return parser


def add_validate_self_pumped_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate-self-pumped",
        help="Validate the minimal self-pumped ADS OA topology.",
        description=(
            "Run the shared SelfPumpedCircuitSpec through analytical, direct "
            "hpeesofsim, and optional ADS OA schematic paths."
        ),
    )
    parser.add_argument("--freq-ghz", type=float, default=2.0, help="Signal frequency in GHz.")
    parser.add_argument("--out", required=True, type=Path, help="Validation report/output directory.")
    parser.add_argument("--simulate", action="store_true", help="Run hpeesofsim via ADS CircuitSimulator.")
    parser.add_argument("--draw-oa", action="store_true", help="Generate the ADS OA schematic and netlist.")
    parser.add_argument("--oa-workspace", type=Path, help="ADS OA workspace directory.")
    parser.add_argument("--ads-home", type=Path, help="ADS install root used to locate Keysight Python packages.")
    parser.set_defaults(func=_run_validate_self_pumped)
    return parser


def add_compare_gain_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "compare-gain",
        help="Write analytical and numerical gain comparison reports.",
        description=(
            "Read optimizer summary.json output, write gain definitions, "
            "analytical candidate metrics, optional direct hpeesofsim metrics, "
            "and optional ADS OA amplifier netlist/audit reports."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Optimizer summary.json path.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for gain reports.")
    parser.add_argument("--simulate", action="store_true", help="Run hpeesofsim metrics where supported.")
    parser.add_argument("--draw-oa", action="store_true", help="Generate the ADS OA amplifier project and audit its netlist.")
    parser.add_argument("--ads-home", type=Path, help="ADS install root used to locate Keysight Python packages.")
    parser.add_argument("--oa-project-dir", type=Path, help="ADS OA amplifier project directory for --draw-oa.")
    _add_convergence_strictness_args(parser)
    parser.set_defaults(func=_run_compare_gain)
    return parser


def add_sweep_pump_gain_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "sweep-pump-gain",
        help="Sweep pump power and write gain reports/plot.",
        description=(
            "Read optimizer summary.json output and sweep the self-pumped "
            "pump source power. Normal Python writes analytical/report-only "
            "outputs; numerical HB metrics are attempted only when the "
            "amplifier model is marked sim-ready."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Optimizer summary.json path.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for pump sweep reports.")
    parser.add_argument("--pump-start-dbm", required=True, type=float, help="First pump power in dBm.")
    parser.add_argument("--pump-stop-dbm", required=True, type=float, help="Last pump power in dBm.")
    parser.add_argument("--pump-step-dbm", required=True, type=float, help="Pump power step in dB.")
    parser.add_argument(
        "--metric",
        action="append",
        choices=SUPPORTED_SWEEP_METRICS,
        help="Metric to include in the sweep focus. Repeat for multiple metrics.",
    )
    parser.add_argument("--simulate", action="store_true", help="Attempt numerical hpeesofsim metrics when sim-ready.")
    parser.add_argument("--draw-oa", action="store_true", help="Generate/audit ADS OA artifacts when supported.")
    parser.add_argument("--analytical-only", action="store_true", help="Force analytical/report-only sweep.")
    parser.add_argument("--plot-y-min", type=float, default=-20.0, help="Lower y-axis bound for the gain plot in dB.")
    parser.add_argument("--plot-y-max", type=float, default=50.0, help="Upper y-axis bound for the gain plot in dB.")
    parser.add_argument(
        "--clip-plot-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip invalid/singular analytical gain points to --invalid-gain-floor-db for plotting.",
    )
    parser.add_argument(
        "--invalid-gain-floor-db",
        type=float,
        default=-100.0,
        help="Raw coherent gain below this floor is flagged null_or_singular.",
    )
    _add_convergence_strictness_args(parser)
    parser.set_defaults(func=_run_sweep_pump_gain)
    return parser


def add_validate_amplifier_hb_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate-amplifier-hb",
        help="Validate the public ADS-OA amplifier against a direct HB deck.",
        description=(
            "Build the shared amplifier spec, write the direct hpeesofsim deck, "
            "generate the ADS-OA amplifier netlist, compare topology, and "
            "optionally run both HB simulations."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Optimizer summary.json path.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for reports.")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--candidate-index", type=_positive_int, help="1-based candidate index from candidates.json.")
    choice.add_argument("--candidate-name", help="Candidate topology/name to validate.")
    parser.add_argument("--simulate", action="store_true", help="Run direct and ADS-OA hpeesofsim.")
    parser.add_argument("--ads-home", type=Path, help="ADS install root used to locate Keysight Python packages.")
    _add_convergence_strictness_args(parser)
    parser.set_defaults(func=_run_validate_amplifier_hb)
    return parser


def add_hb_convergence_ladder_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "hb-convergence-ladder",
        help="Map self-pumped amplifier HB convergence over cells, pump power, and HB order.",
        description=(
            "Generate direct self-pumped amplifier HB decks for a convergence ladder. "
            "Without --simulate, writes a dry-run plan, decks, CSV, JSON, Markdown, and plot. "
            "With --simulate, runs hpeesofsim per point and keeps failures as row data."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Optimizer summary.json path.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for ladder artifacts.")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--candidate-index", type=_positive_int, help="1-based candidate index from candidates.json.")
    choice.add_argument("--candidate-name", help="Candidate topology/name to validate.")
    parser.add_argument("--cells", required=True, type=_cell_list, help="Comma-separated cell counts, e.g. 1,3,5,10,15.")
    parser.add_argument("--pump-dbm", required=True, type=_pump_dbm_list, help="Comma-separated pump powers in dBm.")
    parser.add_argument("--hb-orders", required=True, type=_hb_order_list, help="Comma-separated HB orders, e.g. 1,3,5.")
    parser.add_argument("--simulate", action="store_true", help="Run hpeesofsim for each direct ladder point.")
    parser.add_argument("--dry-run", action="store_true", help="Write decks/reports without running hpeesofsim.")
    parser.add_argument("--stop-on-first-fail", action="store_true", help="Stop after the first failed/error point.")
    parser.add_argument("--include-oa", action="store_true", help="Record ADS-OA ladder points as skipped until OA overrides are audited.")
    parser.add_argument("--timeout-s", type=_positive_int, default=300, help="Per-point hpeesofsim timeout in seconds.")
    parser.add_argument("--ads-home", type=Path, help="ADS install root used to locate adssim/hpeesofsim.")
    parser.add_argument("--max-iterations", type=_positive_int, help="Recorded only; ADS HB parameter name not audited yet.")
    parser.add_argument("--oversample", type=float, help="Recorded only; ADS HB parameter name not audited yet.")
    parser.add_argument("--tolerance", type=float, help="Recorded only; ADS HB parameter name not audited yet.")
    parser.add_argument(
        "--use-previous-solution",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recorded only; ADS HB continuation parameter names are not audited yet.",
    )
    parser.add_argument("--solver-mode", help="Recorded only; ADS HB parameter name not audited yet.")
    _add_convergence_strictness_args(parser)
    parser.set_defaults(func=_run_hb_convergence_ladder)
    return parser


def _add_convergence_strictness_args(parser: argparse.ArgumentParser) -> None:
    strictness = parser.add_mutually_exclusive_group()
    strictness.add_argument(
        "--strict-convergence",
        action="store_true",
        help="Return nonzero when a requested HB simulation has no convergence or errors.",
    )
    strictness.add_argument(
        "--allow-nonconvergence",
        action="store_false",
        dest="strict_convergence",
        help="Default: record HB no-convergence as report data and exit success if infrastructure succeeded.",
    )
    parser.set_defaults(strict_convergence=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tvtl_ads_builder_v2",
        description="Public ADS Builder v2 command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command")
    add_export_gui_project_parser(subparsers)
    add_export_amplifier_oa_parser(subparsers)
    add_validate_self_pumped_parser(subparsers)
    add_compare_gain_parser(subparsers)
    add_validate_amplifier_hb_parser(subparsers)
    add_hb_convergence_ladder_parser(subparsers)
    add_sweep_pump_gain_parser(subparsers)
    return parser


def _normalize_argv(argv: Sequence[str] | None) -> Sequence[str] | None:
    if argv is None:
        raw = sys.argv[1:]
    else:
        raw = list(argv)
    normalized: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if (
            token == "--pump-dbm"
            and index + 1 < len(raw)
            and raw[index + 1].startswith("-")
            and not raw[index + 1].startswith("--")
        ):
            normalized.append(f"--pump-dbm={raw[index + 1]}")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except (ExportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
