"""End-to-end public amplifier HB validation workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

from ..amplifier_oa_export import export_amplifier_oa
from ..amplifier_spec import AmplifierCircuitSpec, RF_MAIN_NET, load_amplifier_circuit_spec
from ..analysis.hb_dataset import HBDataset, read_hb_dataset
from ..oa.psa_gain_artifact import degenerate_folding_mixes, full_family_phasor_sum
from .amplifier_netlist_compare import write_amplifier_netlist_comparison
from .direct_amplifier_hpeesofsim import (
    direct_amplifier_netlist,
    write_direct_amplifier_deck,
)
from .direct_hpeesofsim import HpeesofsimRunResult, _mtimes, _new_or_updated
from .gain_definitions import (
    dbm_to_watts,
    load_power_from_voltage,
    transducer_gain_db,
    voltage_gain_db,
)

GAIN_TOLERANCE_DB = 0.5
STATUS_ANALYTICAL_OK = "analytical_ok"
STATUS_HB_CONVERGED = "hb_converged"
STATUS_HB_NO_CONVERGENCE = "hb_no_convergence"
STATUS_HB_ERROR = "hb_error"
STATUS_HB_SKIPPED = "hb_skipped"
STATUS_METRIC_UNAVAILABLE = "metric_unavailable"
STATUS_COMPARISON_AVAILABLE = "comparison_available"
STATUS_COMPARISON_UNAVAILABLE = "comparison_unavailable"


@dataclass(frozen=True)
class DatasetMetrics:
    """Gain metrics extracted from one HB dataset."""

    status: str
    dataset_path: str | None
    input_net: str
    output_net: str | None
    voltage_gain_db: float | None = None
    coherent_gain_db: float | None = None
    load_power_gain_db: float | None = None
    transducer_gain_db: float | None = None
    full_sum_gain_db: float | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_validate_amplifier_hb(
    input_summary: str | Path,
    out_dir: str | Path,
    *,
    simulate: bool = False,
    ads_home: str | Path | None = None,
    candidate_index: int | None = None,
    candidate_name: str | None = None,
) -> dict[str, Any]:
    """Build direct/OA amplifier decks, compare, optionally simulate, and report."""

    summary_path = Path(input_summary).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    spec = load_amplifier_circuit_spec(
        summary_path,
        candidate_index=candidate_index,
        candidate_name=candidate_name,
    )
    direct_dir = _default_direct_dir(summary_path)
    direct_written = write_direct_amplifier_deck(
        summary_path,
        direct_dir,
        candidate_index=candidate_index,
        candidate_name=candidate_name,
    )
    direct_netlist_path = Path(direct_written["netlist_path"])

    oa_manifest: dict[str, Any] | None = None
    oa_error: str | None = None
    oa_netlist_path: Path | None = None
    try:
        oa_manifest = export_amplifier_oa(
            summary_path,
            out_path / "ads_oa_amplifier",
            candidate_index=candidate_index,
            candidate_name=candidate_name,
            simulate=False,
            ads_home=ads_home,
        )
        oa_netlist_path = Path(str(oa_manifest["generated_netlist_path"]))
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        oa_error = f"{type(exc).__name__}: {exc}"

    if oa_netlist_path is not None and oa_netlist_path.is_file():
        netlist_comparison = write_amplifier_netlist_comparison(
            direct_netlist_path,
            oa_netlist_path,
            spec,
            out_path,
        )
    else:
        netlist_comparison = _missing_oa_netlist_report(out_path, direct_netlist_path, oa_error)

    direct_run = _not_run_result(direct_netlist_path, direct_dir, "simulation not requested")
    oa_run = _not_run_result(oa_netlist_path, out_path / "ads_oa_hpeesofsim", "simulation not requested")
    if simulate:
        direct_run = _run_netlist_text(
            direct_amplifier_netlist(spec),
            direct_netlist_path,
            direct_dir,
        )
        if oa_netlist_path is None or not oa_netlist_path.is_file():
            oa_run = _not_run_result(
                oa_netlist_path,
                out_path / "ads_oa_hpeesofsim",
                "ADS OA netlist unavailable; cannot simulate OA path",
                status="error",
            )
        else:
            oa_run = _run_netlist_text(
                oa_netlist_path.read_text(encoding="utf-8", errors="replace"),
                oa_netlist_path,
                out_path / "ads_oa_hpeesofsim",
            )

    direct_metrics = _metrics_from_run(spec, direct_run, output_net=spec.direct_output_net)
    oa_output_net = (
        _load_net_from_netlist(oa_netlist_path.read_text(encoding="utf-8", errors="replace"))
        if oa_netlist_path is not None and oa_netlist_path.is_file()
        else None
    )
    oa_metrics = _metrics_from_run(spec, oa_run, output_net=oa_output_net)
    gain_report = _write_gain_report(
        out_path,
        spec,
        netlist_comparison,
        direct_run,
        oa_run,
        direct_metrics,
        oa_metrics,
        oa_manifest=oa_manifest,
        oa_error=oa_error,
    )
    manifest = {
        "candidate": spec.candidate_name,
        "public_spec": spec.to_dict(include_device_spec=False),
        "direct_deck": {
            "path": str(direct_netlist_path),
            "spec_path": direct_written["spec_path"],
            "output_dir": str(direct_dir),
        },
        "ads_oa": {
            "manifest": oa_manifest,
            "error": oa_error,
            "netlist_path": str(oa_netlist_path) if oa_netlist_path is not None else None,
        },
        "netlist_comparison": netlist_comparison,
        "simulation": {
            "direct_hpeesofsim": direct_run.to_dict(),
            "ads_oa_hpeesofsim": oa_run.to_dict(),
        },
        "gain_comparison_report": gain_report,
        "sim_ready": gain_report["sim_ready"],
        "ads_oa_amplifier_numerical_status": gain_report[
            "ads_oa_amplifier_numerical_status"
        ],
        "paths": {
            "netlist_comparison_json": str(out_path / "netlist_comparison.json"),
            "netlist_comparison_md": str(out_path / "netlist_comparison.md"),
            "gain_comparison_json": str(out_path / "gain_comparison_report.json"),
            "gain_comparison_md": str(out_path / "gain_comparison_report.md"),
        },
    }
    (out_path / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _default_direct_dir(summary_path: Path) -> Path:
    if summary_path.parent.parent.name.lower() == "runs":
        return summary_path.parent.parent / f"{summary_path.parent.name}_direct_hpeesofsim"
    return Path("runs") / "amp_2ghz_direct_hpeesofsim"


def _run_netlist_text(
    netlist: str,
    netlist_path: str | Path,
    output_dir: str | Path,
) -> HpeesofsimRunResult:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    net_path = Path(netlist_path)
    before = _mtimes(out_path)
    started = time.time()
    try:
        from keysight.edatoolbox import ads, util

        util.safe_makedirs(str(out_path))
        ads.CircuitSimulator().run_netlist(netlist, output_dir=str(out_path))
    except Exception as exc:  # noqa: BLE001
        output_files = _new_or_updated(out_path, before, started)
        return HpeesofsimRunResult(
            status="error",
            netlist_path=str(net_path),
            output_dir=str(out_path),
            dataset_files=[],
            output_files=output_files,
            error_message=_simulator_error_message(str(exc), out_path, output_files),
        )
    output_files = _new_or_updated(out_path, before, started)
    dataset_files = sorted(name for name in output_files if name.endswith(".ds"))
    return HpeesofsimRunResult(
        status="passed" if dataset_files else "error",
        netlist_path=str(net_path),
        output_dir=str(out_path),
        dataset_files=dataset_files,
        output_files=output_files,
        error_message=(
            None
            if dataset_files
            else _simulator_error_message("hpeesofsim wrote no .ds dataset", out_path, output_files)
        ),
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


def _not_run_result(
    netlist_path: Path | None,
    output_dir: Path,
    reason: str,
    *,
    status: str = "not_run",
) -> HpeesofsimRunResult:
    return HpeesofsimRunResult(
        status=status,
        netlist_path=str(netlist_path) if netlist_path is not None else "",
        output_dir=str(output_dir),
        dataset_files=[],
        output_files=[],
        error_message=reason,
    )


def _metrics_from_run(
    spec: AmplifierCircuitSpec,
    run: HpeesofsimRunResult,
    *,
    output_net: str | None,
) -> dict[str, Any]:
    if run.status != "passed" or not run.dataset_files:
        status = hb_status_from_run(run)
        return DatasetMetrics(
            status=status,
            dataset_path=None,
            input_net=RF_MAIN_NET,
            output_net=output_net,
            notes=(
                _hb_unavailable_note(status),
                run.error_message or "HB dataset unavailable",
            ),
        ).to_dict()
    dataset_path = Path(run.output_dir) / run.dataset_files[0]
    try:
        hb = read_hb_dataset(dataset_path, (spec.fs_hz, spec.pump_hz))
        metrics = _metrics_from_dataset(spec, hb, dataset_path, output_net=output_net)
        return metrics.to_dict()
    except Exception as exc:  # noqa: BLE001
        return DatasetMetrics(
            status=STATUS_METRIC_UNAVAILABLE,
            dataset_path=str(dataset_path),
            input_net=RF_MAIN_NET,
            output_net=output_net,
            notes=(f"dataset gain extraction failed: {exc}",),
        ).to_dict()


def _metrics_from_dataset(
    spec: AmplifierCircuitSpec,
    hb: HBDataset,
    dataset_path: Path,
    *,
    output_net: str | None,
) -> DatasetMetrics:
    out_net = output_net or spec.direct_output_net
    v_in = complex(hb.node_voltage(RF_MAIN_NET).at_mix((1, 0)))
    out = hb.node_voltage(out_net)
    v_out = complex(out.at_mix((1, 0)))
    try:
        v_idler = complex(out.at_mix((-1, 1)))
    except KeyError:
        v_idler = 0j
    family = degenerate_folding_mixes(out)
    full_sum = full_family_phasor_sum(out, family) if family else v_out + v_idler
    p_load = load_power_from_voltage(v_out, spec.load_z_ohm, "rms")
    p_avs = dbm_to_watts(spec.psig_dbm)
    return DatasetMetrics(
        status=STATUS_HB_CONVERGED,
        dataset_path=str(dataset_path),
        input_net=RF_MAIN_NET,
        output_net=out_net,
        voltage_gain_db=voltage_gain_db(v_out, v_in),
        coherent_gain_db=_safe_voltage_ratio_db(v_out + v_idler, v_in),
        load_power_gain_db=None,
        transducer_gain_db=transducer_gain_db(p_load, p_avs),
        full_sum_gain_db=_safe_voltage_ratio_db(full_sum, v_in),
        notes=(
            "coherent_gain_db is single-run output/input phasor gain for direct-vs-OA comparison",
            "load_power_gain_db is null because DUT input-plane power was not extracted",
        ),
    )


def hb_status_from_run(run: HpeesofsimRunResult | Mapping[str, Any]) -> str:
    """Map simulator run state into reporting status vocabulary."""

    status = str(_run_value(run, "status") or "")
    dataset_files = _run_value(run, "dataset_files") or []
    error_message = str(_run_value(run, "error_message") or "")
    if status == "passed" and dataset_files:
        return STATUS_HB_CONVERGED
    if status in {"not_run", "skipped"}:
        return STATUS_HB_SKIPPED
    if _is_no_convergence_message(error_message):
        return STATUS_HB_NO_CONVERGENCE
    return STATUS_HB_ERROR


def hb_status_is_strict_failure(status: str) -> bool:
    return status in {STATUS_HB_NO_CONVERGENCE, STATUS_HB_ERROR, STATUS_METRIC_UNAVAILABLE}


def _run_value(run: HpeesofsimRunResult | Mapping[str, Any], key: str) -> Any:
    if isinstance(run, Mapping):
        return run.get(key)
    return getattr(run, key)


def _is_no_convergence_message(message: str) -> bool:
    lowered = message.lower()
    return (
        "no convergence" in lowered
        or "did not converge" in lowered
        or "solution failed" in lowered
    )


def _hb_unavailable_note(status: str) -> str:
    if status == STATUS_HB_NO_CONVERGENCE:
        return "HB did not converge; numerical gain is unavailable/null."
    if status == STATUS_HB_SKIPPED:
        return "HB simulation was skipped; numerical gain is unavailable/null."
    return "HB simulation errored; numerical gain is unavailable/null."


def _safe_voltage_ratio_db(num: complex, den: complex) -> float | None:
    if abs(den) == 0.0 or abs(num) == 0.0:
        return None
    return 20.0 * math.log10(abs(num) / abs(den))


def _write_gain_report(
    out_path: Path,
    spec: AmplifierCircuitSpec,
    netlist_comparison: Mapping[str, Any],
    direct_run: HpeesofsimRunResult,
    oa_run: HpeesofsimRunResult,
    direct_metrics: Mapping[str, Any],
    oa_metrics: Mapping[str, Any],
    *,
    oa_manifest: Mapping[str, Any] | None,
    oa_error: str | None,
) -> dict[str, Any]:
    direct_gain = direct_metrics.get("coherent_gain_db")
    oa_gain = oa_metrics.get("coherent_gain_db")
    analytical_gain = spec.analytical_metrics.get("coherent_gain_db")
    direct_hb_status = str(direct_metrics.get("status") or hb_status_from_run(direct_run))
    oa_hb_status = str(oa_metrics.get("status") or hb_status_from_run(oa_run))
    gain_delta = (
        None
        if not (_is_finite_number(direct_gain) and _is_finite_number(oa_gain))
        else abs(float(direct_gain) - float(oa_gain))
    )
    gain_match = gain_delta is not None and gain_delta <= GAIN_TOLERANCE_DB
    sim_ready = (
        bool(netlist_comparison.get("match"))
        and direct_hb_status == STATUS_HB_CONVERGED
        and oa_hb_status == STATUS_HB_CONVERGED
        and gain_match
    )
    status = (
        "validated_against_direct_hpeesofsim"
        if sim_ready
        else (
            STATUS_HB_NO_CONVERGENCE
            if STATUS_HB_NO_CONVERGENCE in {direct_hb_status, oa_hb_status}
            else "not_validated"
        )
    )
    analytical_vs_hb = _analytical_vs_hb_comparison(
        analytical_gain,
        direct_gain,
        oa_gain,
        direct_status=direct_hb_status,
        oa_status=oa_hb_status,
    )
    report = {
        "candidate": spec.candidate_name,
        "architecture": "self_pumped_same_line",
        "analytical": {
            "status": (
                STATUS_ANALYTICAL_OK
                if _is_finite_number(analytical_gain)
                else STATUS_METRIC_UNAVAILABLE
            ),
            "coherent_gain_db": analytical_gain,
            "normalization": spec.analytical_metrics.get("normalization"),
        },
        "direct_hpeesofsim": {
            "run": direct_run.to_dict(),
            "hb_status": direct_hb_status,
            "metrics": dict(direct_metrics),
        },
        "ads_oa_hpeesofsim": {
            "run": oa_run.to_dict(),
            "hb_status": oa_hb_status,
            "metrics": dict(oa_metrics),
            "manifest_path": oa_manifest.get("project_dir") if oa_manifest else None,
            "error": oa_error,
        },
        "comparison": {
            "netlist_status": netlist_comparison.get("status"),
            "coherent_gain_delta_db": gain_delta,
            "coherent_gain_tolerance_db": GAIN_TOLERANCE_DB,
            "coherent_gain_match": gain_match,
            "direct_vs_oa_status": (
                STATUS_COMPARISON_AVAILABLE
                if gain_delta is not None
                else STATUS_COMPARISON_UNAVAILABLE
            ),
            "analytical_vs_hb": analytical_vs_hb,
        },
        "numerical_availability": {
            "direct_hb_status": direct_hb_status,
            "direct_hb_numerical_gain_available": _is_finite_number(direct_gain),
            "ads_oa_hb_status": oa_hb_status,
            "ads_oa_hb_numerical_gain_available": _is_finite_number(oa_gain),
        },
        "sim_ready": sim_ready,
        "ads_oa_amplifier_numerical_status": status,
        "notes": _gain_notes(netlist_comparison, direct_metrics, oa_metrics, oa_error),
    }
    (out_path / "gain_comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_path / "gain_comparison_report.md").write_text(
        _gain_report_markdown(report),
        encoding="utf-8",
    )
    return report


def _analytical_vs_hb_comparison(
    analytical_gain: Any,
    direct_gain: Any,
    oa_gain: Any,
    *,
    direct_status: str,
    oa_status: str,
) -> dict[str, Any]:
    direct_error = _signed_error(direct_gain, analytical_gain)
    oa_error = _signed_error(oa_gain, analytical_gain)
    return {
        "metric": "coherent_gain_db",
        "analytical": analytical_gain if _is_finite_number(analytical_gain) else None,
        "direct_hb": direct_gain if _is_finite_number(direct_gain) else None,
        "ads_oa_hb": oa_gain if _is_finite_number(oa_gain) else None,
        "direct_error_db": direct_error,
        "direct_abs_error_db": abs(direct_error) if direct_error is not None else None,
        "oa_error_db": oa_error,
        "oa_abs_error_db": abs(oa_error) if oa_error is not None else None,
        "direct_hb_status": direct_status,
        "ads_oa_hb_status": oa_status,
        "status": (
            STATUS_COMPARISON_AVAILABLE
            if direct_error is not None or oa_error is not None
            else STATUS_COMPARISON_UNAVAILABLE
        ),
        "note": (
            "Analytical and HB models are compared, not expected to be exactly equal. "
            "Differences are computed only when both values are finite."
        ),
    }


def _signed_error(value: Any, reference: Any) -> float | None:
    if not (_is_finite_number(value) and _is_finite_number(reference)):
        return None
    return float(value) - float(reference)


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _gain_notes(
    netlist_comparison: Mapping[str, Any],
    direct_metrics: Mapping[str, Any],
    oa_metrics: Mapping[str, Any],
    oa_error: str | None,
) -> list[str]:
    notes: list[str] = []
    if oa_error:
        notes.append(f"ADS OA generation unavailable: {oa_error}")
    if not netlist_comparison.get("match"):
        notes.append("direct-vs-OA netlist comparison did not pass")
    notes.extend(str(note) for note in direct_metrics.get("notes", []))
    notes.extend(str(note) for note in oa_metrics.get("notes", []))
    return notes


def _gain_report_markdown(report: Mapping[str, Any]) -> str:
    av = report.get("numerical_availability", {})
    avh = report.get("comparison", {}).get("analytical_vs_hb", {})
    lines = [
        "# Amplifier Gain Comparison Report",
        "",
        f"- Candidate: `{report['candidate']}`",
        f"- Architecture: `{report['architecture']}`",
        f"- sim_ready: `{report['sim_ready']}`",
        f"- ADS-OA numerical status: `{report['ads_oa_amplifier_numerical_status']}`",
        "",
        "| source | status | coherent_gain_db | voltage_gain_db | transducer_gain_db | full_sum_gain_db |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        _metric_row("analytical", report["analytical"], analytical=True),
        _metric_row("direct hpeesofsim", report["direct_hpeesofsim"]["metrics"]),
        _metric_row("ADS-OA hpeesofsim", report["ads_oa_hpeesofsim"]["metrics"]),
        "",
        "## Numerical Availability",
        "",
        f"- Direct HB: `{av.get('direct_hb_status')}`, numerical gain available: `{av.get('direct_hb_numerical_gain_available')}`",
        f"- ADS-OA HB: `{av.get('ads_oa_hb_status')}`, numerical gain available: `{av.get('ads_oa_hb_numerical_gain_available')}`",
        f"- sim_ready: `{report['sim_ready']}`",
        "",
        "HB no-convergence is recorded as status with `null` JSON metric values. Analytical and HB models are compared, not expected to be exactly equal.",
        "",
        "## Analytical vs Numerical Comparison",
        "",
        "| metric | analytical | direct HB | ADS-OA HB | direct error | OA error | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| {avh.get('metric', 'coherent_gain_db')} | "
            f"{_fmt(avh.get('analytical'))} | "
            f"{_fmt(avh.get('direct_hb'))} | "
            f"{_fmt(avh.get('ads_oa_hb'))} | "
            f"{_fmt(avh.get('direct_error_db'))} | "
            f"{_fmt(avh.get('oa_error_db'))} | "
            f"{avh.get('status', STATUS_COMPARISON_UNAVAILABLE)} |"
        ),
        "",
        "## Comparison",
        "",
        f"- Netlist comparison: `{report['comparison']['netlist_status']}`",
        f"- Coherent gain delta: `{_fmt(report['comparison']['coherent_gain_delta_db'])}` dB",
        f"- Tolerance: `{report['comparison']['coherent_gain_tolerance_db']}` dB",
        "",
    ]
    if report.get("notes"):
        lines.extend(["## Notes", ""])
        for note in report["notes"]:
            lines.extend(_note_markdown(str(note)))
        lines.append("")
    return "\n".join(lines)


def _note_markdown(note: str) -> list[str]:
    parts = note.splitlines()
    if not parts:
        return ["- "]
    return [f"- {parts[0]}", *(f"  {line}" if line else "" for line in parts[1:])]


def _metric_row(label: str, payload: Mapping[str, Any], *, analytical: bool = False) -> str:
    if analytical:
        status = payload.get("status")
        coherent = payload.get("coherent_gain_db")
        return f"| {label} | {status} | {_fmt(coherent)} |  |  |  |"
    return (
        f"| {label} | {payload.get('status')} | "
        f"{_fmt(payload.get('coherent_gain_db'))} | "
        f"{_fmt(payload.get('voltage_gain_db'))} | "
        f"{_fmt(payload.get('transducer_gain_db'))} | "
        f"{_fmt(payload.get('full_sum_gain_db'))} |"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "NaN"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _load_net_from_netlist(text: str) -> str | None:
    import re

    match = re.search(r"^(?:Port|TermG?):TERM_LOAD\s+(\S+)\s+0\b", text, flags=re.MULTILINE)
    if match:
        return match.group(1)
    match = re.search(r"^Port:P_OUT\s+(\S+)\s+0\b", text, flags=re.MULTILINE)
    if match:
        return match.group(1)
    return None


def _missing_oa_netlist_report(
    out_path: Path,
    direct_netlist_path: Path,
    error: str | None,
) -> dict[str, Any]:
    report = {
        "status": "failed",
        "match": False,
        "candidate": None,
        "topology_family": None,
        "n_cells": None,
        "checks": [
            {
                "name": "ads_oa.netlist_present",
                "status": "failed",
                "detail": error or "ADS OA netlist was not generated",
            }
        ],
        "failures": [
            {
                "name": "ads_oa.netlist_present",
                "status": "failed",
                "detail": error or "ADS OA netlist was not generated",
            }
        ],
        "warnings": [],
        "paths": {
            "direct_netlist": str(direct_netlist_path),
            "oa_netlist": None,
            "json": str(out_path / "netlist_comparison.json"),
            "markdown": str(out_path / "netlist_comparison.md"),
        },
    }
    (out_path / "netlist_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_path / "netlist_comparison.md").write_text(
        "# Amplifier Netlist Comparison\n\nADS OA netlist was not generated.\n",
        encoding="utf-8",
    )
    return report
