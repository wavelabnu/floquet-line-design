"""Pump-power sweep reporting for self-pumped gain workflows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tvtl.optimization.stage1_per_cell import StubCandidate
from tvtl.optimization.stage2_main_tl import MainTLDesign
from tvtl.optimization.stage4_validation import CompleteDesign, analytical_characterization

from ..amplifier_spec import load_amplifier_circuit_spec
from ..gui_export import (
    ExportError,
    _matching_from_candidate,
    _params_from_candidate,
    _topology_from_candidate,
    _varactor_from_summary,
    load_optimizer_candidates,
)
from .direct_amplifier_hpeesofsim import run_direct_amplifier_hpeesofsim
from .gain_comparison import DEFAULT_HARMONICS_IN_SUM
from .gain_definitions import (
    gain_equations_markdown_section,
    gain_definitions_markdown_section,
    gain_definitions_payload,
)
from .validate_amplifier_hb import (
    STATUS_HB_CONVERGED,
    STATUS_HB_ERROR,
    STATUS_HB_NO_CONVERGENCE,
    STATUS_HB_SKIPPED,
    _metrics_from_run,
)


SUPPORTED_SWEEP_METRICS = (
    "coherent_gain_db",
    "voltage_gain_db",
    "transducer_gain_db",
)

CSV_COLUMNS = [
    "pump_dbm",
    "pump_hz",
    "candidate",
    "coherent_gain_db_analytical_raw",
    "coherent_gain_db_analytical_plot",
    "gain_point_status",
    "coherent_gain_db_analytical",
    "voltage_gain_db_analytical",
    "transducer_gain_db_analytical",
    "full_sum_gain_db_analytical",
    "coherent_gain_db_direct_hb",
    "coherent_gain_db_oa_hb",
    "direct_hb_status",
    "oa_hb_status",
    "status",
    "notes",
]


def run_pump_power_sweep(
    input_summary: str | Path,
    out_dir: str | Path,
    *,
    pump_start_dbm: float,
    pump_stop_dbm: float,
    pump_step_dbm: float,
    metrics: Sequence[str] | None = None,
    simulate: bool = False,
    draw_oa: bool = False,
    analytical_only: bool = False,
    plot_y_min: float = -20.0,
    plot_y_max: float = 50.0,
    clip_plot_db: bool = True,
    invalid_gain_floor_db: float = -100.0,
) -> dict[str, Any]:
    """Write pump-sweep CSV, JSON, Markdown, and PNG outputs."""

    requested_metrics = list(metrics or ["coherent_gain_db"])
    _validate_requested_metrics(requested_metrics)
    _validate_plot_options(
        plot_y_min=plot_y_min,
        plot_y_max=plot_y_max,
        invalid_gain_floor_db=invalid_gain_floor_db,
    )
    pump_values = _pump_points(pump_start_dbm, pump_stop_dbm, pump_step_dbm)
    summary_path = Path(input_summary).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    summary, candidates = load_optimizer_candidates(summary_path)
    candidate = candidates[0]
    fs_hz = _target_frequency_hz(summary)
    candidate_name = str(candidate.get("topology") or candidate.get("label") or "candidate")
    pump_hz = 2.0 * fs_hz
    notes = [
        (
            "Analytical coherent_gain_db is computed by re-running "
            "tvtl.optimization.stage4_validation.analytical_characterization "
            "at each requested pump power."
        ),
        (
            "voltage_gain_db, transducer_gain_db, and full_sum_gain_db remain "
            "null unless the required input-plane powers or harmonic powers "
            "are available from a validated analytical/numerical path."
        ),
        (
            "Analytical and HB models are compared, not expected to be exactly equal; "
            "differences are computed only when both values are finite."
        ),
    ]
    numerical_sweep_status = "analytical_only"
    if not analytical_only and simulate:
        numerical_sweep_status = "hb_attempted"
        notes.append(
            "Direct hpeesofsim HB is attempted per pump point; no-convergence "
            "points keep null numerical gain values."
        )
    if draw_oa:
        notes.append(
            "ADS-OA per-pump HB sweep is not attempted because OA pump/HB-order "
            "overrides need audit; OA HB status is recorded as hb_skipped."
        )

    rows = _analytical_rows(
        summary,
        candidate,
        candidate_name=candidate_name,
        fs_hz=fs_hz,
        pump_hz=pump_hz,
        pump_values=pump_values,
        numerical_sweep_status=numerical_sweep_status,
        invalid_gain_floor_db=invalid_gain_floor_db,
        clip_plot_db=clip_plot_db,
    )
    if simulate and not analytical_only:
        numerical_sweep_status = _attach_direct_hb_rows(summary_path, rows, out_path)
    flagged_points = _flagged_points(rows)
    if flagged_points:
        notes.append(
            f"{len(flagged_points)} analytical gain point(s) were flagged as null_or_singular "
            f"below {invalid_gain_floor_db:g} dB and do not control the plot scale."
        )
    if any(row["coherent_gain_db_analytical_raw"] is not None for row in rows):
        notes.append("At least one analytical coherent-gain point was computed.")
    else:
        notes.append("analytical_gain_unavailable: no point exposed coherent_gain_db.")

    csv_path = out_path / "pump_sweep.csv"
    json_path = out_path / "pump_sweep.json"
    md_path = out_path / "pump_power_sweep_report.md"
    png_path = out_path / "gain_vs_pump_power.png"

    _write_csv(csv_path, rows)
    plot_info = _write_plot(
        png_path,
        rows,
        candidate=candidate_name,
        fs_hz=fs_hz,
        numerical_sweep_status=numerical_sweep_status,
        plot_y_min=plot_y_min,
        plot_y_max=plot_y_max,
        invalid_gain_floor_db=invalid_gain_floor_db,
        flagged_points=flagged_points,
    )

    payload = {
        "candidate": candidate_name,
        "fs_hz": fs_hz,
        "architecture": "self_pumped_same_line",
        "pump_sweep": {
            "start_dbm": float(pump_start_dbm),
            "stop_dbm": float(pump_stop_dbm),
            "step_dbm": float(pump_step_dbm),
        },
        "metrics": requested_metrics,
        "gain_definitions": gain_definitions_payload(),
        "harmonics_in_sum": DEFAULT_HARMONICS_IN_SUM.copy(),
        "rows": rows,
        "best_gain_point": _best_gain_point(rows, metric="coherent_gain_db_analytical_raw"),
        "numerical_sweep_status": numerical_sweep_status,
        "notes": notes,
        "paths": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
            "png": str(png_path),
        },
        "plot": plot_info,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def _validate_requested_metrics(metrics: Sequence[str]) -> None:
    unsupported = sorted(set(metrics) - set(SUPPORTED_SWEEP_METRICS))
    if unsupported:
        raise ValueError(
            "unsupported sweep metric(s): "
            + ", ".join(unsupported)
            + "; supported metrics are "
            + ", ".join(SUPPORTED_SWEEP_METRICS)
        )


def _validate_plot_options(
    *,
    plot_y_min: float,
    plot_y_max: float,
    invalid_gain_floor_db: float,
) -> None:
    if not all(
        math.isfinite(float(value))
        for value in (plot_y_min, plot_y_max, invalid_gain_floor_db)
    ):
        raise ValueError("plot limits and invalid gain floor must be finite")
    if float(plot_y_max) <= float(plot_y_min):
        raise ValueError("plot_y_max must be greater than plot_y_min")


def _pump_points(start: float, stop: float, step: float) -> list[float]:
    start = float(start)
    stop = float(stop)
    step = float(step)
    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("pump sweep bounds must be finite")
    if step <= 0.0:
        raise ValueError("pump step must be positive")
    if stop < start:
        raise ValueError("pump stop must be greater than or equal to pump start")
    points: list[float] = []
    value = start
    epsilon = abs(step) * 1.0e-9
    while value <= stop + epsilon:
        points.append(round(value, 10))
        value += step
    return points


def _target_frequency_hz(summary: Mapping[str, Any]) -> float:
    if summary.get("target_frequency_Hz") is not None:
        return float(summary["target_frequency_Hz"])
    return float(summary["target_frequency_GHz"]) * 1.0e9


def _complete_design_from_summary(
    summary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    fs_hz: float,
) -> CompleteDesign:
    raw_summary = dict(summary)
    raw_candidate = dict(candidate)
    config = raw_summary.get("config")
    if not isinstance(config, Mapping):
        config = {}
    main_tl_raw = raw_candidate.get("main_tl")
    if not isinstance(main_tl_raw, Mapping):
        raise ExportError("candidate is missing main_tl")

    topology = _topology_from_candidate(raw_candidate)
    params = _params_from_candidate(raw_candidate)
    stub = StubCandidate(
        topology=topology,
        params=params,
        a1_score=float(raw_candidate.get("a1_score") or 0.0),
        f0_Hz=float(fs_hz),
    )
    bloch_z = _complex_from_json(main_tl_raw.get("bloch_z_ohm"))
    if bloch_z is None:
        bloch_z = complex(50.0, 0.0)
    residual = main_tl_raw.get("match_residual_ohm")
    if residual is None:
        residual = abs(bloch_z.real - 50.0)
    main_tl = MainTLDesign(
        stub=stub,
        z_tl_ohm=float(main_tl_raw["z_tl_ohm"]),
        w_main_m=float(main_tl_raw["width_m"]),
        bloch_z_ohm=complex(bloch_z),
        match_residual_ohm=float(residual),
        realizable=bool(main_tl_raw.get("realizable", True)),
        matched=bool(main_tl_raw.get("matched", float(residual) <= 1.0)),
        f0_Hz=float(fs_hz),
        note=str(main_tl_raw.get("note", "")),
    )
    return CompleteDesign(
        stub=stub,
        main_tl=main_tl,
        input_match=_matching_from_candidate(raw_candidate, "input_match"),
        output_match=_matching_from_candidate(raw_candidate, "output_match"),
        varactor=_varactor_from_summary(raw_summary),
        n_cells=int(config.get("n_cells", 15)),
        f0_Hz=float(fs_hz),
    )


def _complex_from_json(value: Any) -> complex | None:
    if isinstance(value, Mapping):
        real = value.get("real")
        imag = value.get("imag")
        if real is None and imag is None:
            magnitude = value.get("magnitude")
            return None if magnitude is None else complex(float(magnitude), 0.0)
        return complex(float(real or 0.0), float(imag or 0.0))
    if value is None:
        return None
    try:
        return complex(value)
    except (TypeError, ValueError):
        return None


def _analytical_rows(
    summary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    candidate_name: str,
    fs_hz: float,
    pump_hz: float,
    pump_values: Sequence[float],
    numerical_sweep_status: str,
    invalid_gain_floor_db: float,
    clip_plot_db: bool,
) -> list[dict[str, Any]]:
    rows = []
    try:
        design = _complete_design_from_summary(summary, candidate, fs_hz=fs_hz)
        design_error = None
    except Exception as exc:  # noqa: BLE001
        design = None
        design_error = str(exc)

    for pump_dbm in pump_values:
        row = {
            "pump_dbm": float(pump_dbm),
            "pump_hz": float(pump_hz),
            "candidate": candidate_name,
            "coherent_gain_db_analytical_raw": None,
            "coherent_gain_db_analytical_plot": None,
            "gain_point_status": "analytical_gain_unavailable",
            "coherent_gain_db_analytical": None,
            "voltage_gain_db_analytical": None,
            "transducer_gain_db_analytical": None,
            "full_sum_gain_db_analytical": None,
            "coherent_gain_db_direct_hb": None,
            "coherent_gain_db_oa_hb": None,
            "direct_hb_status": STATUS_HB_SKIPPED,
            "direct_hb_dataset_path": None,
            "direct_hb_error_summary": None,
            "oa_hb_status": STATUS_HB_SKIPPED,
            "oa_hb_dataset_path": None,
            "oa_hb_error_summary": "ADS-OA per-pump HB sweep not attempted",
            "status": "analytical_gain_unavailable",
            "notes": "",
        }
        point_notes = []
        if design is None:
            point_notes.append(
                "analytical_gain_unavailable: could not rebuild CompleteDesign "
                f"from optimizer summary ({design_error})"
            )
        else:
            try:
                report = analytical_characterization(
                    design,
                    pumps_dBm=(float(pump_dbm),),
                    phases_deg=(0.0,),
                    landscape_GHz=(),
                    gated_pump_dBm=float(pump_dbm),
                )
                analytical = report.analytical[float(pump_dbm)]
                raw_gain = _finite_or_none(
                    analytical.get("coherent_peak")
                )
                row["coherent_gain_db_analytical_raw"] = raw_gain
                row["coherent_gain_db_analytical"] = raw_gain
                plot_gain, point_status = _plot_gain_value(
                    raw_gain,
                    invalid_gain_floor_db=invalid_gain_floor_db,
                    clip_plot_db=clip_plot_db,
                )
                row["coherent_gain_db_analytical_plot"] = plot_gain
                row["gain_point_status"] = point_status
                point_notes.append(
                    "coherent_gain_db_analytical_raw uses Stage-4 coherent_peak "
                    "normalized to pump-off signal reference"
                )
                row["status"] = point_status
            except Exception as exc:  # noqa: BLE001
                point_notes.append(f"analytical_gain_unavailable: {exc}")
        point_notes.append(f"numerical_sweep_status={numerical_sweep_status}")
        row["notes"] = "; ".join(point_notes)
        rows.append(row)
    return rows


def _attach_direct_hb_rows(
    summary_path: Path,
    rows: list[dict[str, Any]],
    out_path: Path,
) -> str:
    statuses: list[str] = []
    direct_root = out_path / "direct_hb"
    direct_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        pump_dbm = float(row["pump_dbm"])
        case_dir = direct_root / f"pump_{_number_token(pump_dbm)}"
        try:
            spec = load_amplifier_circuit_spec(summary_path, pump_power_dbm=pump_dbm)
            run = run_direct_amplifier_hpeesofsim(spec, case_dir)
            metrics = _metrics_from_run(spec, run, output_net=spec.direct_output_net)
            hb_status = str(metrics.get("status") or STATUS_HB_ERROR)
            row["direct_hb_status"] = hb_status
            row["direct_hb_dataset_path"] = metrics.get("dataset_path")
            row["direct_hb_error_summary"] = _first_note(metrics)
            if hb_status == STATUS_HB_CONVERGED:
                row["coherent_gain_db_direct_hb"] = _finite_or_none(
                    metrics.get("coherent_gain_db")
                )
            else:
                row["coherent_gain_db_direct_hb"] = None
            row["notes"] = _append_note(
                str(row.get("notes", "")),
                f"direct_hb_status={hb_status}",
            )
        except Exception as exc:  # noqa: BLE001 - one pump point must not abort the sweep
            hb_status = STATUS_HB_ERROR
            row["direct_hb_status"] = hb_status
            row["direct_hb_error_summary"] = f"{type(exc).__name__}: {exc}"
            row["coherent_gain_db_direct_hb"] = None
            row["notes"] = _append_note(str(row.get("notes", "")), f"direct_hb_status={hb_status}")
        statuses.append(str(row["direct_hb_status"]))
    if any(status == STATUS_HB_CONVERGED for status in statuses):
        return "partial_hb_converged"
    if any(status == STATUS_HB_NO_CONVERGENCE for status in statuses):
        return STATUS_HB_NO_CONVERGENCE
    if any(status == STATUS_HB_ERROR for status in statuses):
        return STATUS_HB_ERROR
    return STATUS_HB_SKIPPED


def _append_note(existing: str, note: str) -> str:
    return f"{existing}; {note}" if existing else note


def _first_note(metrics: Mapping[str, Any]) -> str | None:
    notes = metrics.get("notes")
    if isinstance(notes, Sequence) and not isinstance(notes, str) and notes:
        return str(notes[0])
    return None


def _number_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _plot_gain_value(
    raw_gain: float | None,
    *,
    invalid_gain_floor_db: float,
    clip_plot_db: bool,
) -> tuple[float | None, str]:
    if raw_gain is None:
        return None, "analytical_gain_unavailable"
    if raw_gain < float(invalid_gain_floor_db):
        return (float(invalid_gain_floor_db) if clip_plot_db else None), "null_or_singular"
    return raw_gain, "analytical_ok"


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in CSV_COLUMNS})


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def _best_gain_point(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if row.get(metric) is not None
        and math.isfinite(float(row[metric]))
        and row.get("gain_point_status") != "null_or_singular"
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda row: float(row[metric]))
    return {
        "pump_dbm": best["pump_dbm"],
        "metric": metric,
        "gain_db": best[metric],
        "plot_gain_db": best.get("coherent_gain_db_analytical_plot"),
        "status": best["status"],
    }


def _flagged_points(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flagged = []
    for row in rows:
        if row.get("gain_point_status") != "null_or_singular":
            continue
        flagged.append(
            {
                "pump_dbm": row.get("pump_dbm"),
                "raw_gain_db": row.get("coherent_gain_db_analytical_raw"),
                "plot_gain_db": row.get("coherent_gain_db_analytical_plot"),
                "status": row.get("gain_point_status"),
            }
        )
    return flagged


def _write_plot(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    fs_hz: float,
    numerical_sweep_status: str,
    plot_y_min: float,
    plot_y_max: float,
    invalid_gain_floor_db: float,
    flagged_points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pump = [float(row["pump_dbm"]) for row in rows]
    analytical = [_finite_or_none(row.get("coherent_gain_db_analytical_plot")) for row in rows]
    direct = [_finite_or_none(row.get("coherent_gain_db_direct_hb")) for row in rows]
    oa = [_finite_or_none(row.get("coherent_gain_db_oa_hb")) for row in rows]
    has_analytical = any(value is not None for value in analytical)
    has_numerical = any(value is not None for value in direct + oa)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    if has_analytical:
        ax.plot(
            pump,
            [math.nan if value is None else value for value in analytical],
            marker="o",
            label="analytical coherent",
        )
        best = _best_gain_point(rows, metric="coherent_gain_db_analytical_raw")
        if best is not None and best.get("plot_gain_db") is not None:
            best_x = float(best["pump_dbm"])
            best_y = float(best["plot_gain_db"])
            ax.scatter([best_x], [best_y], s=70, color="tab:red", zorder=5, label="best point")
            ax.axvline(best_x, color="tab:red", alpha=0.25, linewidth=1.0)
            ax.axhline(best_y, color="tab:red", alpha=0.25, linewidth=1.0)
            ax.annotate(
                f"{float(best['gain_db']):.2f} dB @ {best_x:.2f} dBm",
                xy=(best_x, best_y),
                xytext=(8, -24),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.0},
                fontsize=8,
            )
    if any(value is not None for value in direct):
        ax.plot(
            pump,
            [math.nan if value is None else value for value in direct],
            marker="s",
            linestyle="--",
            label="direct HB coherent",
        )
    if any(value is not None for value in oa):
        ax.plot(
            pump,
            [math.nan if value is None else value for value in oa],
            marker="^",
            linestyle=":",
            label="ADS-OA HB coherent",
        )
    if not has_analytical and not has_numerical:
        ax.text(
            0.5,
            0.5,
            "No gain values available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    suffix = (
        "includes numerical HB"
        if has_numerical
        else "analytical-only; HB numerical points unavailable/no convergence"
    )
    ax.set_title(
        f"{candidate} analytical coherent gain vs pump power at fs = {fs_hz / 1.0e9:.3g} GHz\n"
        + suffix
    )
    ax.set_xlabel("Pump power (dBm)")
    ax.set_ylabel("Gain (dB)")
    ax.set_ylim(float(plot_y_min), float(plot_y_max))
    ax.grid(True, alpha=0.35)
    if has_analytical or has_numerical:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "path": str(path),
        "analytical_curve": has_analytical,
        "numerical_curve": has_numerical,
        "title_mode": suffix,
        "numerical_sweep_status": numerical_sweep_status,
        "plot_y_min": float(plot_y_min),
        "plot_y_max": float(plot_y_max),
        "invalid_gain_floor_db": float(invalid_gain_floor_db),
        "flagged_points": list(flagged_points),
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    best = payload.get("best_gain_point")
    lines = [
        "# Pump Power Sweep Report",
        "",
        f"- Candidate: `{payload['candidate']}`",
        f"- fs: `{payload['fs_hz']}` Hz",
        f"- Architecture: `{payload['architecture']}`",
        f"- Metrics requested: `{payload['metrics']}`",
        f"- Selected `K_out`: `{payload['harmonics_in_sum']}`",
        "",
        gain_equations_markdown_section(),
        "",
        "The analytical coherent gain column is Stage-4 coherent_peak normalized to pump-off signal reference.",
        "Analytical and HB models are compared, not expected to be exactly equal. HB no-convergence is recorded as status and null/NaN numerical values.",
        "",
        gain_definitions_markdown_section(),
        "",
        "## Sweep settings",
        "",
        f"- Start: `{payload['pump_sweep']['start_dbm']}` dBm",
        f"- Stop: `{payload['pump_sweep']['stop_dbm']}` dBm",
        f"- Step: `{payload['pump_sweep']['step_dbm']}` dB",
        f"- Numerical sweep status: `{payload['numerical_sweep_status']}`",
        "",
        "## Best gain point",
        "",
    ]
    if best is None:
        lines.append("No gain point is available from this sweep.")
    else:
        lines.append(
            f"`{best['metric']}` reaches `{float(best['gain_db']):.3f}` dB "
            f"at `{float(best['pump_dbm']):.3f}` dBm."
        )
    lines.extend(
        [
            "",
            "## Flagged/null points",
            "",
        ]
    )
    flagged = payload.get("plot", {}).get("flagged_points", [])
    if not flagged:
        lines.append("No analytical gain points were flagged as null/singular.")
    else:
        for point in flagged:
            lines.append(
                f"- Ppump `{float(point['pump_dbm']):.3f}` dBm: raw "
                f"`{float(point['raw_gain_db']):.3f}` dB, status `{point['status']}`."
            )
    lines.extend(
        [
            "",
            "## Pump power table",
            "",
            (
                "| pump_dBm | coherent analytical raw (dB) | coherent analytical plot (dB) | "
                "gain point status | voltage analytical (dB) | "
                "transducer analytical (dB) | direct HB coherent (dB) | "
                "direct HB status | ADS-OA HB coherent (dB) | ADS-OA HB status | status |"
            ),
            "| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['pump_dbm']):.3f}",
                    _format_optional_db(row.get("coherent_gain_db_analytical_raw")),
                    _format_optional_db(row.get("coherent_gain_db_analytical_plot")),
                    str(row.get("gain_point_status")),
                    _format_optional_db(row.get("voltage_gain_db_analytical")),
                    _format_optional_db(row.get("transducer_gain_db_analytical")),
                    _format_optional_db(row.get("coherent_gain_db_direct_hb")),
                    str(row.get("direct_hb_status")),
                    _format_optional_db(row.get("coherent_gain_db_oa_hb")),
                    str(row.get("oa_hb_status")),
                    str(row.get("status")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Data and plot",
            "",
            f"- CSV: `{payload['paths']['csv']}`",
            f"- JSON: `{payload['paths']['json']}`",
            f"- PNG plot: `{payload['paths']['png']}`",
            "",
            "## Interpretation",
            "",
            (
                "Analytical values come from the existing Stage-4 analytical "
                "Floquet/optimizer path. Numerical curves are added only when "
                "real hpeesofsim/ADS-OA data exists."
            ),
            (
                "ADS-OA numerical amplifier gain is unavailable/preliminary "
                "when the hierarchical amplifier export is not sim-ready."
            ),
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in payload.get("notes", []))
    return "\n".join(lines) + "\n"


def _format_optional_db(value: Any) -> str:
    value = _finite_or_none(value)
    return "" if value is None else f"{value:.3f}"
