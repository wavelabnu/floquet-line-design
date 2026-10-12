"""HB convergence ladder for the public self-pumped amplifier."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

from ..amplifier_spec import (
    PUMP_SOURCE_NAME,
    RF_MAIN_NET,
    SIGNAL_SOURCE_NAME,
    load_amplifier_circuit_spec,
)
from ..gui_export import ExportError
from ..simulate.runner import SIMULATOR_BINARIES, run_hpeesofsim
from .direct_amplifier_hpeesofsim import direct_amplifier_netlist


ARCHITECTURE = "self_pumped_same_line"
CSV_COLUMNS = [
    "deck_kind",
    "n_cells",
    "pump_dbm",
    "hb_order",
    "status",
    "dataset_exists",
    "netlist_path",
    "dataset_path",
    "error_summary",
    "elapsed_seconds",
]
STATUS_ORDER = {"converged": 0, "failed": 1, "error": 2, "skipped": 3}
CONVERGED = "converged"
FAILED = "failed"
ERROR = "error"
SKIPPED = "skipped"


@dataclass(frozen=True)
class LadderPoint:
    """One planned convergence probe."""

    deck_kind: str
    n_cells: int
    pump_dbm: float
    hb_order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_cell_list(value: str) -> list[int]:
    """Parse a comma-separated positive cell-count list."""

    return _parse_positive_int_list(value, label="cell count")


def parse_hb_order_list(value: str) -> list[int]:
    """Parse a comma-separated positive HB-order list."""

    return _parse_positive_int_list(value, label="HB order")


def parse_pump_dbm_list(value: str) -> list[float]:
    """Parse a comma-separated finite pump-power list."""

    values = _split_csv_values(value, label="pump power")
    out: list[float] = []
    for raw in values:
        try:
            parsed = float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid pump power {raw!r}") from exc
        if not math.isfinite(parsed):
            raise ValueError("pump powers must be finite")
        out.append(parsed)
    return out


def build_ladder_plan(
    *,
    cells: Sequence[int],
    pump_dbm: Sequence[float],
    hb_orders: Sequence[int],
    deck_kinds: Sequence[str] = ("direct",),
) -> list[LadderPoint]:
    """Return the planned direct/OA convergence probes."""

    normalized_cells = _normalize_positive_ints(cells, label="cell count")
    normalized_pumps = _normalize_floats(pump_dbm, label="pump power")
    normalized_orders = _normalize_positive_ints(hb_orders, label="HB order")
    normalized_kinds = _normalize_deck_kinds(deck_kinds)
    return [
        LadderPoint(kind, n_cells, pump, hb_order)
        for kind in normalized_kinds
        for n_cells in normalized_cells
        for pump in normalized_pumps
        for hb_order in normalized_orders
    ]


def run_hb_convergence_ladder(
    input_summary: str | Path,
    out_dir: str | Path,
    *,
    cells: Sequence[int],
    pump_dbm: Sequence[float],
    hb_orders: Sequence[int],
    simulate: bool = False,
    dry_run: bool = False,
    stop_on_first_fail: bool = False,
    candidate_index: int | None = None,
    candidate_name: str | None = None,
    include_oa: bool = False,
    timeout_s: int = 300,
    ads_home: str | Path | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a convergence ladder and optionally run hpeesofsim per point."""

    if simulate and dry_run:
        raise ValueError("--simulate and --dry-run are mutually exclusive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    summary_path = Path(input_summary).resolve()
    out_path = Path(out_dir).resolve()
    decks_dir = out_path / "decks"
    datasets_dir = out_path / "datasets"
    logs_dir = out_path / "logs"
    for folder in (decks_dir, datasets_dir, logs_dir):
        folder.mkdir(parents=True, exist_ok=True)

    deck_kinds = ("direct", "oa") if include_oa else ("direct",)
    plan = build_ladder_plan(
        cells=cells,
        pump_dbm=pump_dbm,
        hb_orders=hb_orders,
        deck_kinds=deck_kinds,
    )
    notes = [
        "Direct decks reuse AmplifierSpec and direct_amplifier_netlist.",
        "The self-pumped architecture keeps PORT_SIG and PORT_PUMP on RF_MAIN.",
    ]
    if not include_oa:
        notes.append("ADS-OA ladder was not requested; direct deck only.")
    else:
        notes.append(
            "ADS-OA ladder points are recorded as skipped because the OA exporter "
            "does not yet expose audited per-point n_cells, pump power, and HB-order overrides."
        )
    if solver_options and any(value is not None for value in solver_options.values()):
        notes.append(
            "solver setting sweep not implemented because ADS HB parameter names need audit"
        )

    results: list[dict[str, Any]] = []
    spec_for_header = None
    ads_environment = _ads_environment(ads_home)
    for point in plan:
        started = time.monotonic()
        try:
            if point.deck_kind == "oa":
                row = _skipped_oa_row(point, decks_dir, logs_dir, started)
            else:
                spec = load_amplifier_circuit_spec(
                    summary_path,
                    candidate_index=candidate_index,
                    candidate_name=candidate_name,
                    pump_power_dbm=point.pump_dbm,
                    n_cells=point.n_cells,
                    hb_order=point.hb_order,
                )
                spec_for_header = spec_for_header or spec
                netlist = direct_amplifier_netlist(spec)
                netlist_path = decks_dir / _case_filename(point)
                netlist_path.write_text(netlist, encoding="utf-8")
                if simulate:
                    row = _simulate_direct_point(
                        point,
                        netlist,
                        netlist_path,
                        datasets_dir,
                        logs_dir,
                        ads_environment=ads_environment,
                        timeout_s=timeout_s,
                        started=started,
                    )
                else:
                    row = _dry_run_row(point, netlist_path, started)
        except Exception as exc:  # noqa: BLE001 - captured per ladder point
            row = _exception_row(point, logs_dir, started, exc)
        results.append(row)
        if stop_on_first_fail and row["status"] in {FAILED, ERROR}:
            notes.append("Stopped on first failed/error point because stop_on_first_fail=True.")
            break

    if spec_for_header is None:
        spec_for_header = load_amplifier_circuit_spec(
            summary_path,
            candidate_index=candidate_index,
            candidate_name=candidate_name,
        )

    payload = _payload(
        spec_for_header,
        cells=cells,
        pump_dbm=pump_dbm,
        hb_orders=hb_orders,
        results=results,
        notes=notes,
        out_path=out_path,
    )
    csv_path = out_path / "hb_convergence_ladder.csv"
    json_path = out_path / "hb_convergence_ladder.json"
    md_path = out_path / "hb_convergence_ladder.md"
    png_path = out_path / "hb_convergence_map.png"

    _write_csv(csv_path, results)
    plot_info = _write_plot(png_path, results, hb_orders=_normalize_positive_ints(hb_orders, label="HB order"))
    payload["paths"]["png"] = str(png_path)
    payload["plot"] = plot_info
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def simulation_runtime_available(ads_home: str | Path | None = None) -> bool:
    """Return true when the subprocess simulator helper can probably run."""

    env = _ads_environment(ads_home)
    if env.get("simulator_path"):
        return True
    hpeesof_dir = env.get("hpeesof_dir") or os.environ.get("HPEESOF_DIR")
    if hpeesof_dir:
        for name in SIMULATOR_BINARIES:
            if (Path(str(hpeesof_dir)) / "bin" / name).exists():
                return True
    return any(shutil.which(name) for name in SIMULATOR_BINARIES)


def _simulate_direct_point(
    point: LadderPoint,
    netlist: str,
    netlist_path: Path,
    datasets_dir: Path,
    logs_dir: Path,
    *,
    ads_environment: Mapping[str, Any],
    timeout_s: int,
    started: float,
) -> dict[str, Any]:
    case_dir = datasets_dir / _case_id(point)
    case_dir.mkdir(parents=True, exist_ok=True)
    sim_netlist_path = case_dir / netlist_path.name
    sim_netlist_path.write_text(netlist, encoding="utf-8")
    sim_result = run_hpeesofsim(
        sim_netlist_path,
        ads_environment=ads_environment or None,
        timeout_s=int(timeout_s),
    )
    elapsed = sim_result.get("wall_time_s")
    if elapsed is None:
        elapsed = time.monotonic() - started
    dataset_path = sim_result.get("dataset_path")
    dataset_exists = bool(dataset_path and Path(str(dataset_path)).exists())
    simulator_status = str(sim_result.get("status") or "")
    if simulator_status == "completed" and dataset_exists:
        status = CONVERGED
        error_summary = ""
    elif simulator_status == "blocked_missing_ads_runtime":
        status = ERROR
        error_summary = _error_summary(sim_result, case_dir)
    else:
        status = FAILED
        error_summary = _error_summary(sim_result, case_dir)
    log_path = _write_case_log(logs_dir, point, sim_result, error_summary)
    return {
        **point.to_dict(),
        "status": status,
        "dataset_exists": dataset_exists,
        "netlist_path": str(netlist_path),
        "simulation_netlist_path": str(sim_netlist_path),
        "dataset_path": str(dataset_path) if dataset_path else "",
        "log_path": str(log_path),
        "returncode": sim_result.get("returncode"),
        "exception_type": "",
        "error_summary": error_summary,
        "elapsed_seconds": round(float(elapsed), 6),
    }


def _dry_run_row(point: LadderPoint, netlist_path: Path, started: float) -> dict[str, Any]:
    return {
        **point.to_dict(),
        "status": SKIPPED,
        "dataset_exists": False,
        "netlist_path": str(netlist_path),
        "simulation_netlist_path": "",
        "dataset_path": "",
        "log_path": "",
        "returncode": None,
        "exception_type": "",
        "error_summary": "dry run; simulation not requested",
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _skipped_oa_row(
    point: LadderPoint,
    decks_dir: Path,
    logs_dir: Path,
    started: float,
) -> dict[str, Any]:
    log_path = logs_dir / (_case_id(point) + ".log")
    message = (
        "ADS-OA ladder not implemented for this sweep point; direct ladder "
        "is the required minimum."
    )
    log_path.write_text(message + "\n", encoding="utf-8")
    return {
        **point.to_dict(),
        "status": SKIPPED,
        "dataset_exists": False,
        "netlist_path": str(decks_dir / (_case_filename(point))),
        "simulation_netlist_path": "",
        "dataset_path": "",
        "log_path": str(log_path),
        "returncode": None,
        "exception_type": "",
        "error_summary": message,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _exception_row(
    point: LadderPoint,
    logs_dir: Path,
    started: float,
    exc: Exception,
) -> dict[str, Any]:
    summary = f"{type(exc).__name__}: {exc}"
    log_path = logs_dir / (_case_id(point) + ".log")
    log_path.write_text(summary + "\n", encoding="utf-8")
    return {
        **point.to_dict(),
        "status": ERROR,
        "dataset_exists": False,
        "netlist_path": "",
        "simulation_netlist_path": "",
        "dataset_path": "",
        "log_path": str(log_path),
        "returncode": None,
        "exception_type": type(exc).__name__,
        "error_summary": summary,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _payload(
    spec: Any,
    *,
    cells: Sequence[int],
    pump_dbm: Sequence[float],
    hb_orders: Sequence[int],
    results: Sequence[Mapping[str, Any]],
    notes: Sequence[str],
    out_path: Path,
) -> dict[str, Any]:
    first_failure = _first_failure(results)
    largest_converged = _largest_converged_case(results)
    return {
        "candidate": spec.candidate_name,
        "architecture": ARCHITECTURE,
        "fs_hz": float(spec.fs_hz),
        "pump_hz": float(spec.pump_hz),
        "sweep": {
            "cells": _normalize_positive_ints(cells, label="cell count"),
            "pump_dbm": _normalize_floats(pump_dbm, label="pump power"),
            "hb_orders": _normalize_positive_ints(hb_orders, label="HB order"),
        },
        "results": list(results),
        "first_failure": first_failure,
        "largest_converged_case": largest_converged,
        "notes": list(notes),
        "paths": {
            "json": str(out_path / "hb_convergence_ladder.json"),
            "csv": str(out_path / "hb_convergence_ladder.csv"),
            "markdown": str(out_path / "hb_convergence_ladder.md"),
            "png": str(out_path / "hb_convergence_map.png"),
            "decks_dir": str(out_path / "decks"),
            "datasets_dir": str(out_path / "datasets"),
            "logs_dir": str(out_path / "logs"),
        },
        "topology_audit": _topology_audit(results),
        "summaries": {
            "status_counts": _status_counts(results),
            "lowest_failing_pump_by_cell": _lowest_failing_pump_by_cell(results),
            "largest_converged_cells_by_pump": _largest_converged_cells_by_pump(results),
            "hb_order_summary": _group_summary(results, key="hb_order"),
            "cell_count_summary": _group_summary(results, key="n_cells"),
        },
    }


def _topology_audit(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    netlist_paths = [
        Path(str(row.get("netlist_path")))
        for row in results
        if row.get("deck_kind") == "direct" and row.get("netlist_path")
    ]
    texts = [
        path.read_text(encoding="utf-8", errors="replace")
        for path in netlist_paths
        if path.is_file()
    ]
    combined = "\n".join(texts)
    tokens = {token.strip(" ,;()") for token in combined.replace("\n", " ").split()}
    return {
        "PORT_SIG_on_RF_MAIN": f"Port:{SIGNAL_SOURCE_NAME} {RF_MAIN_NET} 0 " in combined,
        "PORT_PUMP_on_RF_MAIN": f"Port:{PUMP_SOURCE_NAME} {RF_MAIN_NET} 0 " in combined,
        "separate_pump_control_bias_lane_present": bool({"PUMP", "CONTROL", "BIAS"} & tokens),
        "LVIA_present": "LVIA" in combined,
        "MLOC_present": "MLOC:" in combined,
        "VIAGND_present": "VIAGND:" in combined,
    }


def _first_failure(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for row in results:
        if row.get("status") in {FAILED, ERROR}:
            return _small_case(row)
    return {}


def _largest_converged_case(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    converged = [row for row in results if row.get("status") == CONVERGED]
    if not converged:
        return {}
    best = max(
        converged,
        key=lambda row: (
            int(row.get("n_cells", 0)),
            float(row.get("pump_dbm", float("-inf"))),
            int(row.get("hb_order", 0)),
        ),
    )
    return _small_case(best)


def _small_case(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "deck_kind",
        "n_cells",
        "pump_dbm",
        "hb_order",
        "status",
        "dataset_exists",
        "netlist_path",
        "dataset_path",
        "error_summary",
        "elapsed_seconds",
    ]
    return {key: row.get(key) for key in keys if key in row}


def _status_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {CONVERGED: 0, FAILED: 0, ERROR: 0, SKIPPED: 0}
    for row in results:
        status = str(row.get("status") or "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _lowest_failing_pump_by_cell(results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    failures: dict[str, float] = {}
    for row in results:
        if row.get("status") not in {FAILED, ERROR}:
            continue
        key = str(int(row["n_cells"]))
        pump = float(row["pump_dbm"])
        if key not in failures or pump < failures[key]:
            failures[key] = pump
    return failures


def _largest_converged_cells_by_pump(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    largest: dict[str, int] = {}
    for row in results:
        if row.get("status") != CONVERGED:
            continue
        key = _float_key(float(row["pump_dbm"]))
        n_cells = int(row["n_cells"])
        if key not in largest or n_cells > largest[key]:
            largest[key] = n_cells
    return largest


def _group_summary(results: Sequence[Mapping[str, Any]], *, key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, dict[str, Any]] = {}
    for row in results:
        value = row.get(key)
        bucket = grouped.setdefault(value, {key: value, CONVERGED: 0, FAILED: 0, ERROR: 0, SKIPPED: 0})
        status = str(row.get("status") or "")
        bucket[status] = int(bucket.get(status, 0)) + 1
    return [grouped[key] for key in sorted(grouped)]


def _write_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _write_plot(
    path: Path,
    results: Sequence[Mapping[str, Any]],
    *,
    hb_orders: Sequence[int],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orders = list(hb_orders)
    fig, axes = plt.subplots(
        1,
        len(orders),
        figsize=(max(4.0, 3.6 * len(orders)), 4.2),
        sharey=True,
        squeeze=False,
    )
    colors = {
        CONVERGED: "tab:green",
        FAILED: "tab:red",
        ERROR: "tab:orange",
        SKIPPED: "0.55",
    }
    markers = {
        CONVERGED: "o",
        FAILED: "x",
        ERROR: "^",
        SKIPPED: "s",
    }
    plotted_statuses: set[str] = set()
    for ax, order in zip(axes[0], orders):
        rows = [row for row in results if int(row.get("hb_order", -1)) == int(order)]
        if rows:
            for status in (CONVERGED, FAILED, ERROR, SKIPPED):
                xs = [float(row["pump_dbm"]) for row in rows if row.get("status") == status]
                ys = [int(row["n_cells"]) for row in rows if row.get("status") == status]
                if not xs:
                    continue
                plotted_statuses.add(status)
                ax.scatter(
                    xs,
                    ys,
                    c=colors[status],
                    marker=markers[status],
                    label=status,
                    s=45,
                )
        else:
            ax.text(0.5, 0.5, "No points", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"HB order {order}")
        ax.set_xlabel("Pump power (dBm)")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("Cell count")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        for status in sorted(plotted_statuses, key=lambda value: STATUS_ORDER.get(value, 99)):
            axes[0][0].scatter([], [], c=colors[status], marker=markers[status], label=status)
        handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)))
    fig.suptitle("HB convergence ladder")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "path": str(path),
        "hb_orders": list(orders),
        "status_legend": sorted(plotted_statuses, key=lambda value: STATUS_ORDER.get(value, 99)),
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    counts = payload["summaries"]["status_counts"]
    first_failure = payload.get("first_failure") or {}
    largest = payload.get("largest_converged_case") or {}
    lines = [
        "# HB Convergence Ladder",
        "",
        "## Summary",
        "",
        f"- Candidate: `{payload['candidate']}`",
        f"- Architecture: `{payload['architecture']}`",
        f"- fs: `{payload['fs_hz']}` Hz",
        f"- Sweep cells: `{payload['sweep']['cells']}`",
        f"- Sweep pump dBm: `{payload['sweep']['pump_dbm']}`",
        f"- Sweep HB orders: `{payload['sweep']['hb_orders']}`",
        f"- Status counts: converged `{counts.get(CONVERGED, 0)}`, failed `{counts.get(FAILED, 0)}`, error `{counts.get(ERROR, 0)}`, skipped `{counts.get(SKIPPED, 0)}`",
        "",
        "## First Failure Point",
        "",
        _case_sentence(first_failure, empty="No failed or error point was recorded."),
        "",
        "## Largest Converged Case",
        "",
        _case_sentence(largest, empty="No converged point was recorded."),
        "",
        "## Per-HB-Order Table",
        "",
        "| hb_order | converged | failed | error | skipped |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summaries"]["hb_order_summary"]:
        lines.append(_summary_row(row, "hb_order"))
    lines.extend(
        [
            "",
            "## Per-Cell-Count Table",
            "",
            "| n_cells | converged | failed | error | skipped |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["summaries"]["cell_count_summary"]:
        lines.append(_summary_row(row, "n_cells"))
    lines.extend(["", "## Error Summaries", ""])
    error_rows = [
        row
        for row in payload["results"]
        if row.get("status") in {FAILED, ERROR} or row.get("error_summary")
    ]
    if not error_rows:
        lines.append("No simulator errors were recorded.")
    else:
        for row in error_rows[:40]:
            label = (
                f"{row['deck_kind']} cells={row['n_cells']} "
                f"pump={row['pump_dbm']} dBm order={row['hb_order']}"
            )
            lines.append(f"- `{label}`: {row.get('error_summary') or row.get('status')}")
        if len(error_rows) > 40:
            lines.append(f"- Additional rows omitted from Markdown: `{len(error_rows) - 40}`")
    lines.extend(
        [
            "",
            "## Recommended Next Actions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in _recommended_actions(payload))
    if payload.get("notes"):
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in payload["notes"])
    lines.extend(
        [
            "",
            "## Output Paths",
            "",
            f"- JSON: `{payload['paths']['json']}`",
            f"- CSV: `{payload['paths']['csv']}`",
            f"- Markdown: `{payload['paths']['markdown']}`",
            f"- PNG: `{payload['paths']['png']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _case_sentence(row: Mapping[str, Any], *, empty: str) -> str:
    if not row:
        return empty
    return (
        f"`{row.get('deck_kind')}` cells `{row.get('n_cells')}`, "
        f"pump `{row.get('pump_dbm')}` dBm, HB order `{row.get('hb_order')}`, "
        f"status `{row.get('status')}`."
    )


def _summary_row(row: Mapping[str, Any], key: str) -> str:
    return (
        f"| {row.get(key)} | {row.get(CONVERGED, 0)} | {row.get(FAILED, 0)} | "
        f"{row.get(ERROR, 0)} | {row.get(SKIPPED, 0)} |"
    )


def _recommended_actions(payload: Mapping[str, Any]) -> list[str]:
    counts = payload["summaries"]["status_counts"]
    if counts.get(CONVERGED, 0) == 0 and counts.get(FAILED, 0) == 0 and counts.get(ERROR, 0) == 0:
        return [
            "Run the quick ADS ladder with --simulate to collect the first numerical convergence boundary.",
            "Keep the direct ladder first; add ADS-OA only after the direct convergence boundary is clear.",
        ]
    actions = [
        "Use pump continuation: start from the lowest converged pump power and ramp upward.",
        "Try a lower HB order first, then restart the higher-order case from that solution if ADS supports it.",
        "Validate the 1-cell unit cell separately before scaling to the 15-cell target.",
    ]
    if counts.get(FAILED, 0) or counts.get(ERROR, 0):
        actions.extend(
            [
                "Check the varactor SDD operating domain at the first failing pump point.",
                "Add or audit realistic loss/Q limits if failures begin only at larger cascades or high pump power.",
            ]
        )
    return actions


def _error_summary(sim_result: Mapping[str, Any], output_dir: Path) -> str:
    parts = []
    for key in ("error_message", "stdout_tail", "stderr_tail"):
        value = sim_result.get(key)
        if value:
            parts.append(str(value).strip())
    out_tail = _latest_out_tail(output_dir)
    if out_tail:
        parts.append("Simulator .out tail:\n" + out_tail)
    if not parts:
        parts.append(str(sim_result.get("status") or "simulation failed"))
    return _truncate("\n".join(part for part in parts if part), 1200)


def _latest_out_tail(output_dir: Path, *, lines: int = 30) -> str:
    candidates = sorted(output_dir.glob("*.out"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return ""
    text = candidates[-1].read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:]).strip()


def _write_case_log(
    logs_dir: Path,
    point: LadderPoint,
    sim_result: Mapping[str, Any],
    error_summary: str,
) -> Path:
    log_path = logs_dir / (_case_id(point) + ".log")
    payload = {
        "point": point.to_dict(),
        "simulator_result": dict(sim_result),
        "error_summary": error_summary,
    }
    log_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return log_path


def _ads_environment(ads_home: str | Path | None) -> dict[str, str]:
    if ads_home is not None:
        return {"hpeesof_dir": str(Path(ads_home))}
    inferred = _infer_ads_home_from_python()
    if inferred is None:
        return {}
    return {"hpeesof_dir": str(inferred)}


def _infer_ads_home_from_python() -> Path | None:
    exe = Path(sys.executable)
    try:
        tools_dir = exe.parent.parent
        root = tools_dir.parent
    except IndexError:
        return None
    if tools_dir.name.lower() != "tools":
        return None
    if any((root / "bin" / name).exists() for name in SIMULATOR_BINARIES):
        return root
    return None


def _parse_positive_int_list(value: str, *, label: str) -> list[int]:
    values = _split_csv_values(value, label=label)
    out: list[int] = []
    for raw in values:
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {label} {raw!r}") from exc
        if parsed <= 0:
            raise ValueError(f"{label}s must be positive")
        out.append(parsed)
    return out


def _split_csv_values(value: str, *, label: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{label} list must not be empty")
    return values


def _normalize_positive_ints(values: Sequence[int], *, label: str) -> list[int]:
    out = [int(value) for value in values]
    if not out:
        raise ValueError(f"{label} list must not be empty")
    if any(value <= 0 for value in out):
        raise ValueError(f"{label}s must be positive")
    return out


def _normalize_floats(values: Sequence[float], *, label: str) -> list[float]:
    out = [float(value) for value in values]
    if not out:
        raise ValueError(f"{label} list must not be empty")
    if any(not math.isfinite(value) for value in out):
        raise ValueError(f"{label}s must be finite")
    return out


def _normalize_deck_kinds(values: Sequence[str]) -> list[str]:
    kinds = [str(value).lower() for value in values]
    unsupported = sorted(set(kinds) - {"direct", "oa"})
    if unsupported:
        raise ValueError(f"unsupported deck kind(s): {', '.join(unsupported)}")
    if not kinds:
        raise ValueError("deck kind list must not be empty")
    return kinds


def _case_id(point: LadderPoint) -> str:
    return (
        f"{point.deck_kind}_cells{int(point.n_cells):03d}_"
        f"pump{_number_token(point.pump_dbm)}_hb{int(point.hb_order)}"
    )


def _case_filename(point: LadderPoint) -> str:
    return _case_id(point) + ".ckt"


def _number_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _float_key(value: float) -> str:
    return f"{float(value):g}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
