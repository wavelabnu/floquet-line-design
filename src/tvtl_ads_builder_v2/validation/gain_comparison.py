"""Gain comparison driver for analytical, direct, and ADS-OA paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..amplifier_oa_export import export_amplifier_oa
from ..ads_oa_writer import _open_or_create_workspace, import_ads_modules
from ..gui_export import load_optimizer_candidates
from ..amplifier_spec import load_amplifier_circuit_spec
from .direct_amplifier_hpeesofsim import (
    run_direct_amplifier_hpeesofsim,
    write_direct_amplifier_deck,
)
from .direct_hpeesofsim import HpeesofsimRunResult
from .gain_definitions import (
    GainMetricReport,
    dbm_to_watts,
    gain_equations_markdown_section,
    gain_definitions_markdown_section,
    gain_definitions_payload,
    load_power_from_voltage,
    transducer_gain_db,
    voltage_gain_db,
)
from .validate_amplifier_hb import (
    STATUS_ANALYTICAL_OK,
    STATUS_COMPARISON_AVAILABLE,
    STATUS_COMPARISON_UNAVAILABLE,
    STATUS_HB_CONVERGED,
    STATUS_HB_SKIPPED,
    STATUS_METRIC_UNAVAILABLE,
    _metrics_from_run,
)


FORBIDDEN_ADS_ELEMENT_MARKERS = ("L:LVIA", "LVIA", "MLOC:")
DEFAULT_HARMONICS_IN_SUM = [1]
GAIN_SUMMARY_METRICS = [
    "coherent_gain_db",
    "voltage_gain_db",
    "load_power_gain_db",
    "transducer_gain_db",
    "full_sum_gain_db",
]


@dataclass(frozen=True)
class AdsComponentInspection:
    """ADS schematic-symbol discovery for one required element."""

    library: str
    cell: str
    view: str
    cellview_exists: bool
    placed: bool
    terminal_count: int | None
    terminal_numbers: list[int]
    parameter_keys: list[str]
    pin_snap_points: list[dict[str, float]]
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_no_deprecated_ads_elements(text: str, *, label: str = "text") -> None:
    """Raise if active ADS text contains deprecated amplifier elements."""

    present = [marker for marker in FORBIDDEN_ADS_ELEMENT_MARKERS if marker in text]
    if present:
        raise ValueError(
            f"{label} contains deprecated ADS element marker(s): "
            + ", ".join(present)
        )


def audit_ads_element_text(text: str) -> dict[str, Any]:
    """Return a JSON-safe ADS element audit for source, plans, or netlists."""

    present = {
        marker: marker in text
        for marker in FORBIDDEN_ADS_ELEMENT_MARKERS
    }
    return {
        "LVIA_present": "LVIA" in text,
        "L_LVIA_present": "L:LVIA" in text,
        "MLOC_present": "MLOC:" in text,
        "MLEF_present": "MLEF" in text,
        "VIAGND_present": "VIAGND" in text,
        "forbidden_markers_present": [
            marker for marker, found in present.items() if found
        ],
        "passed": not any(present.values()),
    }


def validate_ads_termination_components(
    *,
    ads_home: str | Path | None = None,
) -> dict[str, Any]:
    """Validate MLEF and VIAGND by creating a temporary ADS workspace."""

    try:
        de, db_uu = import_ads_modules(ads_home)
    except Exception as exc:  # noqa: BLE001 - ADS is optional for normal Python
        return {
            "status": "unavailable",
            "MLEF_cellview_validated": False,
            "VIAGND_cellview_validated": False,
            "inspections": {},
            "notes": [f"ADS component discovery unavailable: {exc}"],
        }

    inspections: dict[str, AdsComponentInspection] = {}
    notes: list[str] = []
    cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="tvtl_ads_gain_components_") as tmp:
            workspace, library = _open_or_create_workspace(
                de, Path(tmp) / "workspace", "tvtl_ads_gain_probe_lib"
            )
            setup = getattr(library, "setup_schematic_tech", None)
            if setup is not None:
                try:
                    setup()
                except RuntimeError as exc:
                    if "file exists" not in str(exc).lower():
                        raise
                    notes.append(str(exc))

            design = db_uu.create_schematic(
                ("tvtl_ads_gain_probe_lib", "component_probe", "schematic")
            )
            for index, cell in enumerate(("MLEF", "VIAGND"), start=1):
                inspections[cell] = _inspect_component_instance(
                    de, design, "ads_tlines", cell, "symbol", index
                )
            design.save_design()
            if de.workspace_is_open():
                de.close_workspace()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"ADS component placement validation failed: {exc}")
    finally:
        try:
            if de.workspace_is_open():
                de.close_workspace()
        except Exception:  # noqa: BLE001
            pass
        try:
            os.chdir(cwd)
        except OSError as exc:
            notes.append(f"could not restore working directory after ADS validation: {exc}")

    mlef_ok = bool(inspections.get("MLEF") and inspections["MLEF"].placed)
    viagnd_ok = bool(inspections.get("VIAGND") and inspections["VIAGND"].placed)
    return {
        "status": "passed" if mlef_ok and viagnd_ok else "unavailable",
        "product_version": de.product_version().strip(),
        "api_version": de.version(),
        "MLEF_cellview_validated": mlef_ok,
        "VIAGND_cellview_validated": viagnd_ok,
        "inspections": {
            key: value.to_dict() for key, value in inspections.items()
        },
        "notes": notes,
    }


def _inspect_component_instance(
    de: Any,
    design: Any,
    library: str,
    cell: str,
    view: str,
    index: int,
) -> AdsComponentInspection:
    exists = bool(de.cellview_exists(library, cell, view))
    try:
        inst = design.add_instance(
            (library, cell, view),
            (float(index) * 2.0, 0.0),
            name=f"X_{cell}",
        )
        terminal_numbers: list[int] = []
        pin_snap_points: list[dict[str, float]] = []
        for term in inst.inst_terms:
            if term.is_numbered:
                terminal_numbers.append(int(term.term_number))
            for pin in term.inst_pins:
                snap = getattr(pin, "snap_point", None)
                if snap is not None:
                    pin_snap_points.append({"x": float(snap.x), "y": float(snap.y)})
        return AdsComponentInspection(
            library=library,
            cell=cell,
            view=view,
            cellview_exists=exists,
            placed=True,
            terminal_count=len(terminal_numbers),
            terminal_numbers=terminal_numbers,
            parameter_keys=sorted(str(key) for key in inst.parameters.keys()),
            pin_snap_points=pin_snap_points,
        )
    except Exception as exc:  # noqa: BLE001
        return AdsComponentInspection(
            library=library,
            cell=cell,
            view=view,
            cellview_exists=exists,
            placed=False,
            terminal_count=None,
            terminal_numbers=[],
            parameter_keys=[],
            pin_snap_points=[],
            error_message=str(exc),
        )


def run_gain_comparison(
    input_summary: str | Path,
    out_dir: str | Path,
    *,
    simulate: bool = False,
    draw_oa: bool = False,
    ads_home: str | Path | None = None,
    oa_project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write gain comparison reports and return the JSON payload."""

    summary_path = Path(input_summary).resolve()
    out_path = Path(out_dir).resolve()
    base_dir = Path.cwd().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    summary, candidates = load_optimizer_candidates(summary_path)
    if not candidates:
        raise ValueError(f"optimizer summary has no selected candidates: {summary_path}")
    candidate = candidates[0]
    fs_hz = _target_frequency_hz(summary)
    candidate_name = str(candidate.get("topology") or candidate.get("label") or "candidate")

    analytical = _write_analytical_gain(summary, candidate, out_path)
    numerical_direct = _write_direct_gain(summary_path, out_path, simulate=simulate)
    component_validation = (
        validate_ads_termination_components(ads_home=ads_home)
        if draw_oa or simulate
        else {
            "status": "not_run",
            "MLEF_cellview_validated": None,
            "VIAGND_cellview_validated": None,
            "inspections": {},
            "notes": ["ADS MLEF/VIAGND validation was not requested."],
        }
    )
    numerical_oa = _write_oa_gain(
        summary_path,
        out_path,
        draw_oa=draw_oa,
        simulate=simulate,
        ads_home=ads_home,
        component_validation=component_validation,
        oa_project_dir=oa_project_dir,
        base_dir=base_dir,
    )

    ads_element_audit = _combined_ads_element_audit(
        component_validation,
        numerical_oa.get("ads_element_audit", {}),
    )
    report = {
        "candidate": candidate_name,
        "fs_hz": fs_hz,
        "architecture": "self_pumped_same_line",
        "ads_element_audit": ads_element_audit,
        "gain_definitions": gain_definitions_payload(),
        "analytical": analytical,
        "numerical_direct": numerical_direct,
        "numerical_oa": numerical_oa,
        "comparison": _comparison_payload(analytical, numerical_direct, numerical_oa),
        "notes": _report_notes(component_validation, numerical_direct, numerical_oa),
    }
    report["minimal_validation"] = _minimal_validation_payload(analytical, numerical_direct)
    report["amplifier_analytical"] = _amplifier_analytical_payload(analytical)
    report["amplifier_numerical"] = _amplifier_numerical_payload(numerical_direct, numerical_oa)
    json_path = out_path / "gain_comparison_report.json"
    md_path = out_path / "gain_comparison_report.md"
    report["paths"] = {
        "analytical_gain": str(out_path / "analytical_gain.json"),
        "numerical_gain_direct": str(out_path / "numerical_gain_direct.json"),
        "numerical_gain_oa": str(out_path / "numerical_gain_oa.json"),
        "comparison_json": str(json_path),
        "comparison_md": str(md_path),
    }
    report["visible_summary"] = _visible_summary_payload(report)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _target_frequency_hz(summary: Mapping[str, Any]) -> float:
    if summary.get("target_frequency_Hz") is not None:
        return float(summary["target_frequency_Hz"])
    return float(summary["target_frequency_GHz"]) * 1.0e9


def _write_analytical_gain(
    summary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    metrics = candidate.get("metrics_at_drive")
    if not isinstance(metrics, Mapping):
        metrics = {}
    coherent_peak = _float_or_none(metrics.get("coherent_peak"))
    candidate_report = GainMetricReport(
        coherent_gain_db=coherent_peak,
        harmonics_in_sum=DEFAULT_HARMONICS_IN_SUM.copy(),
        notes=[
            "coherent_gain_db uses optimizer metrics_at_drive.coherent_peak when available.",
            "Other amplifier gain definitions require a validated matching analytical path or numerical dataset.",
            "Analytical and HB models are compared, not expected to be exactly equal.",
        ],
    )
    baseline_report = GainMetricReport(
        voltage_gain_db=0.0,
        load_power_gain_db=0.0,
        transducer_gain_db=None,
        coherent_gain_db=0.0,
        full_sum_gain_db=0.0,
        harmonics_in_sum=DEFAULT_HARMONICS_IN_SUM.copy(),
        notes=[
            "minimal passive baseline is unity from RF input plane to matched load at fs",
            "transducer gain is not asserted because source-power convention is verified numerically",
        ],
    )
    payload = {
        "status": STATUS_ANALYTICAL_OK if coherent_peak is not None else STATUS_METRIC_UNAVAILABLE,
        "candidate": str(candidate.get("topology") or candidate.get("label") or "candidate"),
        "fs_hz": _target_frequency_hz(summary),
        "optimizer_metrics_at_drive": dict(metrics),
        "baseline_minimal_matched": baseline_report.to_dict(),
        "amplifier_candidate": candidate_report.to_dict(),
    }
    (out_path / "analytical_gain.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_direct_gain(
    summary_path: Path,
    out_path: Path,
    *,
    simulate: bool,
) -> dict[str, Any]:
    spec = load_amplifier_circuit_spec(summary_path)
    direct_dir = out_path / "direct_amplifier_hpeesofsim"
    written = write_direct_amplifier_deck(summary_path, direct_dir)
    direct_netlist_path = Path(str(written["netlist_path"]))
    if not simulate:
        payload = {
            "status": STATUS_HB_SKIPPED,
            "scope": "amplifier_candidate",
            "run": HpeesofsimRunResult(
                status="not_run",
                netlist_path=str(direct_netlist_path),
                output_dir=str(direct_dir),
                dataset_files=[],
                output_files=[],
                error_message="simulation not requested",
            ).to_dict(),
            "metrics": {
                **GainMetricReport(
                harmonics_in_sum=DEFAULT_HARMONICS_IN_SUM.copy(),
                notes=[
                    "direct amplifier hpeesofsim gain metrics require --simulate",
                    "direct amplifier deck was written but not simulated",
                ],
                ).to_dict(),
                "status": STATUS_HB_SKIPPED,
            },
        }
    else:
        run = run_direct_amplifier_hpeesofsim(spec, direct_dir)
        metrics = _metrics_from_run(spec, run, output_net=spec.direct_output_net)
        payload = {
            "status": metrics.get("status"),
            "scope": "amplifier_candidate",
            "run": run.to_dict(),
            "metrics": metrics,
        }
    (out_path / "numerical_gain_direct.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_oa_gain(
    summary_path: Path,
    out_path: Path,
    *,
    draw_oa: bool,
    simulate: bool,
    ads_home: str | Path | None,
    component_validation: Mapping[str, Any],
    oa_project_dir: str | Path | None,
    base_dir: Path,
) -> dict[str, Any]:
    notes: list[str] = []
    audit: dict[str, Any] = {}
    manifest: dict[str, Any] | None = None
    status = STATUS_HB_SKIPPED
    if not draw_oa:
        notes.append("ADS OA amplifier export was not requested.")
    elif component_validation.get("status") != "passed":
        status = STATUS_METRIC_UNAVAILABLE
        notes.append(
            "amplifier numerical comparison unavailable: unit-cell MLEF/VIAGND "
            "mapping is not validated in this ADS installation"
        )
    else:
        project_dir = (
            Path(oa_project_dir)
            if oa_project_dir is not None
            else base_dir / "ads_projects" / "amp_2ghz_amplifier_gain_compare_oa"
        )
        manifest = export_amplifier_oa(
            summary_path,
            project_dir,
            simulate=False,
            ads_home=ads_home,
        )
        netlist_path = Path(manifest["generated_netlist_path"])
        netlist = netlist_path.read_text(encoding="utf-8")
        audit = audit_ads_element_text(netlist)
        if not audit["passed"]:
            notes.append("generated ADS OA amplifier netlist contains deprecated elements")
        notes.append(
            "Full amplifier gain comparison is preliminary or unavailable; "
            "unit-cell OA internals require direct-vs-OA hpeesofsim validation."
        )
        if simulate:
            notes.append(
                "ADS OA amplifier hpeesofsim metrics were not computed because "
                "the hierarchical amplifier export is still marked not sim-ready."
            )

    payload = {
        "status": status,
        "manifest": manifest,
        "ads_element_audit": audit,
        "metrics": {
            **GainMetricReport(
                harmonics_in_sum=DEFAULT_HARMONICS_IN_SUM.copy(),
                notes=notes,
            ).to_dict(),
            "status": status,
        },
        "simulation_ready": bool(manifest and manifest.get("sim_ready")),
    }
    (out_path / "numerical_gain_oa.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _combined_ads_element_audit(
    component_validation: Mapping[str, Any],
    netlist_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "MLEF_cellview_validated": component_validation.get("MLEF_cellview_validated"),
        "VIAGND_cellview_validated": component_validation.get("VIAGND_cellview_validated"),
        "component_validation": component_validation,
        "LVIA_present": bool(netlist_audit.get("LVIA_present", False)),
        "MLOC_present": bool(netlist_audit.get("MLOC_present", False)),
        "generated_netlist_audit": dict(netlist_audit),
    }


def _comparison_payload(
    analytical: Mapping[str, Any],
    numerical_direct: Mapping[str, Any],
    numerical_oa: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = analytical.get("baseline_minimal_matched", {})
    direct = (
        numerical_direct.get("metrics", {})
        if numerical_direct.get("scope") == "minimal_self_pumped_baseline"
        else {}
    )
    direct_amp = (
        numerical_direct.get("metrics", {})
        if numerical_direct.get("scope") == "amplifier_candidate"
        else {}
    )
    amplifier = analytical.get("amplifier_candidate", {})
    oa = numerical_oa.get("metrics", {})
    analytical_coherent = amplifier.get("coherent_gain_db")
    direct_coherent = direct_amp.get("coherent_gain_db")
    oa_coherent = oa.get("coherent_gain_db")
    direct_error = _signed_error(direct_coherent, analytical_coherent)
    oa_error = _signed_error(oa_coherent, analytical_coherent)
    absolute_error_db = {
        "minimal_voltage_gain_db": _absolute_error(
            baseline.get("voltage_gain_db"), direct.get("voltage_gain_db")
        ),
        "amplifier_analytical_vs_direct_coherent_gain_db": _absolute_error(
            analytical_coherent, direct_coherent
        ),
        "amplifier_coherent_gain_db": _absolute_error(
            analytical_coherent, oa_coherent
        ),
    }
    analytical_vs_hb_status = (
        STATUS_COMPARISON_AVAILABLE
        if direct_error is not None or oa_error is not None
        else STATUS_COMPARISON_UNAVAILABLE
    )
    return {
        "absolute_error_db": absolute_error_db,
        "analytical_vs_hb": {
            "metric": "coherent_gain_db",
            "analytical": analytical_coherent if _is_finite_number(analytical_coherent) else None,
            "direct_hb": direct_coherent if _is_finite_number(direct_coherent) else None,
            "ads_oa_hb": oa_coherent if _is_finite_number(oa_coherent) else None,
            "direct_error_db": direct_error,
            "direct_abs_error_db": abs(direct_error) if direct_error is not None else None,
            "oa_error_db": oa_error,
            "oa_abs_error_db": abs(oa_error) if oa_error is not None else None,
            "direct_hb_status": direct_amp.get("status") or numerical_direct.get("status"),
            "ads_oa_hb_status": oa.get("status") or numerical_oa.get("status"),
            "status": analytical_vs_hb_status,
            "note": (
                "Analytical and HB models are compared, not expected to be exactly equal. "
                "Differences are computed only when both values are finite."
            ),
        },
        "relative_error_linear": {},
        "pass_fail": {
            "minimal_voltage_gain_db": _pass_fail(
                absolute_error_db["minimal_voltage_gain_db"], tolerance=1.0e-9
            ),
            "amplifier_gain_db": {
                "status": "preliminary",
                "reason": (
                    "strict amplifier dB tolerance is not enforced until "
                    "analytical and ADS OA unit-cell model equivalence is validated"
                ),
            },
        },
        "tolerances": {
            "minimal_circuit_rf_main_phasor_relative_error": 1.0e-9,
            "topology_netlist_equivalence": "exact/topological match",
            "amplifier_gain_db": "preliminary, no strict tolerance",
        },
    }


def _absolute_error(left: Any, right: Any) -> float | None:
    if not (_is_finite_number(left) and _is_finite_number(right)):
        return None
    return abs(float(left) - float(right))


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


def _pass_fail(error: float | None, *, tolerance: float) -> dict[str, Any]:
    if error is None:
        return {"status": "skipped", "tolerance": tolerance}
    return {
        "status": "passed" if error <= tolerance else "error",
        "absolute_error": error,
        "tolerance": tolerance,
    }


def _report_notes(
    component_validation: Mapping[str, Any],
    numerical_direct: Mapping[str, Any],
    numerical_oa: Mapping[str, Any],
) -> list[str]:
    notes: list[str] = []
    notes.extend(str(note) for note in component_validation.get("notes", []))
    notes.extend(str(note) for note in numerical_direct.get("metrics", {}).get("notes", []))
    notes.extend(str(note) for note in numerical_oa.get("metrics", {}).get("notes", []))
    return notes


def _minimal_validation_payload(
    analytical: Mapping[str, Any],
    numerical_direct: Mapping[str, Any],
) -> dict[str, Any]:
    if numerical_direct.get("scope") != "minimal_self_pumped_baseline":
        return {
            "scope": "minimal_self_pumped_reference",
            "voltage_gain_db": analytical.get("baseline_minimal_matched", {}).get("voltage_gain_db"),
            "analytical_voltage_gain_db": analytical.get("baseline_minimal_matched", {}).get("voltage_gain_db"),
            "direct_hb_voltage_gain_db": None,
            "status": "not_run",
            "reason": "compare-gain now writes amplifier-scope direct HB metrics; minimal baseline was not run",
        }
    baseline = analytical.get("baseline_minimal_matched", {})
    direct_metrics = numerical_direct.get("metrics", {})
    direct_voltage = direct_metrics.get("voltage_gain_db")
    status = "validated" if numerical_direct.get("status") == "passed" and direct_voltage is not None else "not_run"
    return {
        "scope": "minimal_self_pumped_reference",
        "voltage_gain_db": direct_voltage if direct_voltage is not None else baseline.get("voltage_gain_db"),
        "analytical_voltage_gain_db": baseline.get("voltage_gain_db"),
        "direct_hb_voltage_gain_db": direct_voltage,
        "status": status,
        "reason": (
            "minimal self-pumped direct hpeesofsim baseline validated"
            if status == "validated"
            else "minimal self-pumped direct hpeesofsim baseline was not run"
        ),
    }


def _amplifier_analytical_payload(analytical: Mapping[str, Any]) -> dict[str, Any]:
    amplifier = analytical.get("amplifier_candidate", {})
    coherent = amplifier.get("coherent_gain_db")
    return {
        "scope": "amplifier_candidate",
        "candidate": analytical.get("candidate"),
        "coherent_peak_db": coherent,
        "coherent_gain_db": coherent,
        "coherent_gain_normalization": "Stage-4 coherent_peak normalized to pump-off signal reference",
        "status": STATUS_ANALYTICAL_OK if coherent is not None else STATUS_METRIC_UNAVAILABLE,
        "reason": (
            "optimizer metrics_at_drive.coherent_peak is available"
            if coherent is not None
            else "optimizer coherent_peak was not available"
        ),
    }


def _amplifier_numerical_payload(
    numerical_direct: Mapping[str, Any],
    numerical_oa: Mapping[str, Any],
) -> dict[str, Any]:
    sim_ready = bool(numerical_oa.get("simulation_ready"))
    direct_metrics = (
        numerical_direct.get("metrics", {})
        if numerical_direct.get("scope") == "amplifier_candidate"
        else {}
    )
    oa_metrics = numerical_oa.get("metrics", {})
    direct_coh = direct_metrics.get("coherent_gain_db")
    oa_coh = oa_metrics.get("coherent_gain_db")
    any_metric = direct_coh is not None or oa_coh is not None
    direct_status = str(direct_metrics.get("status") or numerical_direct.get("status"))
    oa_status = str(oa_metrics.get("status") or numerical_oa.get("status"))
    return {
        "scope": "amplifier_candidate",
        "direct_hb": numerical_direct.get("run"),
        "oa_hb": numerical_oa.get("run"),
        "direct_hb_status": direct_status,
        "oa_hb_status": oa_status,
        "coherent_gain_db_direct_hb": direct_coh,
        "coherent_gain_db_oa_hb": oa_coh,
        "status": (
            "available_partial"
            if any_metric and not sim_ready
            else (
                "available"
                if sim_ready
                else (
                    direct_status
                    if direct_status not in {"", STATUS_HB_SKIPPED}
                    else "unavailable_not_sim_ready"
                )
            )
        ),
        "reason": (
            "direct amplifier HB metric is available, but ADS-OA equivalence is not yet validated"
            if any_metric and not sim_ready
            else (
                "hierarchical amplifier export is not marked sim-ready / equivalence is not yet validated"
                if not sim_ready
                else "amplifier numerical hpeesofsim metric was extracted"
            )
        ),
        "direct_hb_scope": numerical_direct.get("scope"),
    }


def _visible_summary_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    analytical_metrics = report["analytical"].get("amplifier_candidate", {})
    direct_metrics = (
        report["numerical_direct"].get("metrics", {})
        if report["numerical_direct"].get("scope") == "amplifier_candidate"
        else {}
    )
    oa_metrics = report["numerical_oa"].get("metrics", {})
    rows = []
    for metric in GAIN_SUMMARY_METRICS:
        analytical = analytical_metrics.get(metric)
        direct = direct_metrics.get(metric)
        oa = oa_metrics.get(metric)
        if direct is not None or oa is not None:
            status = "available"
            reason = "one or more numerical amplifier metrics are available"
        elif analytical is not None:
            status = STATUS_COMPARISON_UNAVAILABLE
            reason = "analytical optimizer metric available; numerical amplifier gain unavailable"
        else:
            status = "unavailable"
            reason = "metric inputs are not available from the current analytical/numerical data"
        rows.append(
            {
                "metric": metric,
                "analytical": analytical,
                "direct_hb": direct,
                "ads_oa_hb": oa,
                "status": status,
                "reason": reason,
            }
        )

    fs_hz = float(report["fs_hz"])
    return {
        "title": "Gain comparison summary",
        "candidate": report["candidate"],
        "fs_hz": fs_hz,
        "fs_ghz": fs_hz / 1.0e9,
        "pump_hz": 2.0 * fs_hz,
        "pump_ghz": 2.0 * fs_hz / 1.0e9,
        "architecture": report["architecture"],
        "coherent_peak_from_optimizer": analytical_metrics.get("coherent_gain_db"),
        "minimal_validation": dict(report.get("minimal_validation", {})),
        "amplifier_analytical": dict(report.get("amplifier_analytical", {})),
        "amplifier_numerical": dict(report.get("amplifier_numerical", {})),
        "analytical_vs_hb": dict(report.get("comparison", {}).get("analytical_vs_hb", {})),
        "sim_ready": bool(report["numerical_oa"].get("simulation_ready")),
        "rows": rows,
        "report_paths": dict(report.get("paths", {})),
        "notes": [
            "Direct HB metrics are listed for the amplifier only when an amplifier-scope direct dataset exists.",
            "Analytical and HB models are compared, not expected to be exactly equal.",
        ],
    }


def format_visible_summary(report: Mapping[str, Any]) -> str:
    """Format the compact terminal summary table."""

    summary = report.get("visible_summary") or _visible_summary_payload(report)
    rows = list(summary["rows"])
    metric_w = max(len("metric"), *(len(str(row["metric"])) for row in rows))
    col_w = 15
    lines = [
        str(summary["title"]),
        "-" * len(str(summary["title"])),
        f"candidate: {summary['candidate']}",
        f"fs: {float(summary['fs_ghz']):.3f} GHz",
        f"pump frequency: {float(summary['pump_ghz']):.3f} GHz",
        f"architecture: {summary['architecture']}",
        f"sim_ready: {summary['sim_ready']}",
        "",
        "Minimal self-pumped validation:",
        f"  voltage_gain_db: {_format_db(summary.get('minimal_validation', {}).get('voltage_gain_db'))}",
        f"  status: {summary.get('minimal_validation', {}).get('status', 'unavailable')}",
        "",
        "Amplifier analytical prediction:",
        f"  candidate: {summary['candidate']}",
        f"  optimizer coherent_peak: {_format_db(summary.get('coherent_peak_from_optimizer'))}",
        f"  coherent_gain_db_analytical: {_format_db(summary.get('amplifier_analytical', {}).get('coherent_gain_db'))}",
        f"  status: {summary.get('amplifier_analytical', {}).get('status', 'unavailable')}",
        "",
        "Direct hpeesofsim amplifier:",
        f"  coherent_gain_db_direct_hb: {_format_db(summary.get('amplifier_numerical', {}).get('coherent_gain_db_direct_hb'))}",
        f"  status: {summary.get('amplifier_numerical', {}).get('direct_hb_status', 'unavailable')}",
        "",
        "Amplifier ADS-OA numerical:",
        f"  coherent_gain_db_oa_hb: {_format_db(summary.get('amplifier_numerical', {}).get('coherent_gain_db_oa_hb'))}",
        f"  status: {summary.get('amplifier_numerical', {}).get('oa_hb_status', 'unavailable')}",
        f"  reason: {summary.get('amplifier_numerical', {}).get('reason', 'not available')}",
        "",
        (
            f"{'metric':<{metric_w}}  "
            f"{'analytical':<{col_w}}  "
            f"{'direct HB':<{col_w}}  "
            f"{'ADS-OA HB':<{col_w}}  "
            "status"
        ),
    ]
    for row in rows:
        lines.append(
            f"{row['metric']:<{metric_w}}  "
            f"{_format_db(row.get('analytical')):<{col_w}}  "
            f"{_format_db(row.get('direct_hb')):<{col_w}}  "
            f"{_format_db(row.get('ads_oa_hb')):<{col_w}}  "
            f"{row['status']}"
        )
    paths = summary.get("report_paths", {})
    if paths:
        lines.extend(["", "Reports:"])
        for key in ("comparison_json", "comparison_md"):
            if paths.get(key):
                lines.append(f"  {paths[key]}")
    return "\n".join(lines)


def _format_db(value: Any) -> str:
    if value is None:
        return "NaN"
    try:
        return f"{float(value):.2f} dB"
    except (TypeError, ValueError):
        return "NaN"


def _markdown_summary_table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| metric | analytical | direct HB | ADS-OA HB | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["metric"]),
                    _format_db(row.get("analytical")),
                    _format_db(row.get("direct_hb")),
                    _format_db(row.get("ads_oa_hb")),
                    str(row["status"]),
                ]
            )
            + " |"
        )
    return lines


def _markdown_report(report: Mapping[str, Any]) -> str:
    audit = report["ads_element_audit"]
    summary = report["visible_summary"]
    avh = report.get("comparison", {}).get("analytical_vs_hb", {})
    numerical = report.get("amplifier_numerical", {})
    lines = [
        "# Gain Comparison Report",
        "",
        f"- Candidate: `{report['candidate']}`",
        f"- fs: `{report['fs_hz']}` Hz",
        f"- Architecture: `{report['architecture']}`",
        f"- MLEF validated: `{audit.get('MLEF_cellview_validated')}`",
        f"- VIAGND validated: `{audit.get('VIAGND_cellview_validated')}`",
        f"- LVIA present in generated OA netlist: `{audit.get('LVIA_present')}`",
        f"- MLOC present in generated OA netlist: `{audit.get('MLOC_present')}`",
        "",
        gain_equations_markdown_section(),
        "",
        "The available analytical coherent gain uses Stage-4 coherent_peak normalized to pump-off signal reference.",
        "",
        "## What gain is currently validated?",
        "",
        (
            "The minimal self-pumped reference circuit validates ADS OA/netlist/"
            "hpeesofsim equivalence when direct HB is run."
        ),
        (
            "The amplifier candidate currently has analytical gain predictions: "
            f"`{_format_db(report['amplifier_analytical'].get('coherent_gain_db'))}` "
            f"with status `{report['amplifier_analytical'].get('status')}`."
        ),
        (
            "Full ADS-OA HB amplifier gain remains preliminary/unavailable until "
            "the hierarchical OA amplifier is marked sim-ready and compared "
            "against a direct hpeesofsim amplifier deck."
        ),
        "",
        "## Status",
        "",
        f"- Analytical gain metrics: `{report['analytical']['status']}`",
        f"- Direct hpeesofsim gain metrics: `{report['numerical_direct']['status']}`",
        f"- ADS OA hpeesofsim gain metrics: `{report['numerical_oa']['status']}`",
        f"- Simulation-ready amplifier: `{report['numerical_oa'].get('simulation_ready')}`",
        f"- Selected `K_out`: `{DEFAULT_HARMONICS_IN_SUM}`",
        "",
        "Unavailable metrics are `null` in JSON with notes; they are not fabricated.",
        "",
        "## Numerical availability",
        "",
        f"- Direct HB: `{numerical.get('direct_hb_status')}`, numerical gain unavailable: `{numerical.get('coherent_gain_db_direct_hb') is None}`",
        f"- ADS-OA HB: `{numerical.get('oa_hb_status')}`, numerical gain unavailable: `{numerical.get('coherent_gain_db_oa_hb') is None}`",
        f"- sim_ready: `{report['numerical_oa'].get('simulation_ready')}`",
        "",
        "HB no-convergence is recorded as status and `null`/NaN metric values. Analytical and HB models are compared, not expected to be exactly equal.",
        "",
        "## Analytical vs numerical comparison",
        "",
        "| metric | analytical | direct HB | ADS-OA HB | direct error | OA error | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| {avh.get('metric', 'coherent_gain_db')} | "
            f"{_format_db(avh.get('analytical'))} | "
            f"{_format_db(avh.get('direct_hb'))} | "
            f"{_format_db(avh.get('ads_oa_hb'))} | "
            f"{_format_db(avh.get('direct_error_db'))} | "
            f"{_format_db(avh.get('oa_error_db'))} | "
            f"{avh.get('status', STATUS_COMPARISON_UNAVAILABLE)} |"
        ),
        "",
        "## Visible summary",
        "",
        *_markdown_summary_table(summary),
        "",
        gain_definitions_markdown_section(),
        "",
    ]
    if report.get("notes"):
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report["notes"])
        lines.append("")
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
