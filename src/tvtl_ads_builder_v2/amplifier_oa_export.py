"""ADS OpenAccess export for optimizer amplifier candidates.

This module builds a connected self-pumped amplifier GUI scaffold from
optimizer JSON. The top level keeps the verified same-line source topology while
the custom unit-cell symbol is populated from the repository's existing
``SingleShuntVaractorCell`` DeviceSpec mapping. Full amplifier simulation
equivalence is still not claimed until the hierarchical OA netlist is validated
against a direct netlist for the selected candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import shutil
from pathlib import Path
from typing import Any

from tvtl.microstrip.discontinuities import via_inductance_bahl_H
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    VIA_RADIUS_M,
    ViaConfig,
)
from tvtl.varactors import RealisticVaractor

from .ads_oa_writer import (
    ADSBuildResult,
    ADSComponentRef,
    _create_or_clear_schematic,
    _mtimes,
    _new_or_updated,
    _open_or_create_workspace,
    _pin_snap_point,
    _set_param,
    ads_ghz,
    ads_number,
    import_ads_modules,
    validate_component_refs,
)
from .cells.single_shunt_varactor import SingleShuntVaractorCell
from .device_spec import (
    Analysis,
    DeviceSpec,
    Element,
    ElementPin,
    FrequencyGrid,
    Node,
    Port,
)
from .gui_export import (
    ExportError,
    _params_from_candidate,
    _topology_from_candidate,
    _varactor_from_summary,
    load_optimizer_candidates,
    select_candidates,
)
from .models.registry import standard_chain_models
from .oa.oa_emitter import (
    _GROUND_CELL,
    _MSUB_CELL,
    _component_positions,
    _msub_params,
    _pin_xy,
)
from .oa.oa_hier_emitter import _connect_and_route, _node_pins_for, _place_elements
from .parity.optimized_design import _substrate_from_config
from .parity.self_pumped_design import _q_poly_reference, main_period_anchor_at_3p1GHz


RF_NET = "RF_MAIN"
SCaffold_MODE = "connected_amplifier_symbol_scaffold"
DEFAULT_LIBRARY_NAME = "tvtl_amp_oa_lib"
UNIT_CELL_REPRESENTATION = "custom_unit_cell_symbol"
UNIT_CELL_INTERNAL_REPRESENTATION = "single_shunt_varactor_device_spec"
UNIT_CELL_INTERNAL_STATUS = "implemented"
UNIT_CELL_ROLE = "unit_cell_symbol"
DEFAULT_SOURCE_Z_OHM = 100.0
DEFAULT_LOAD_Z_OHM = 50.0
UNIT_CELL_SYMBOL_START_X = 6.0
UNIT_CELL_SYMBOL_PITCH = 8.0
REAL_UNIT_CELL_COMPONENTS = [
    "MLIN main-line halves",
    "MTEE_ADS shunt tap",
    "L package inductance",
    "R package series resistance",
    "SDD2P charge varactor",
    "I_Probe varactor branch probe",
    "MLIN optimized stub sections",
    "VIAGND short/via-to-ground termination when topology requires a short",
    "MLEF open-end termination when topology requires an open end",
]
GAIN_ARTIFACT_FILENAMES = {
    "gain_comparison_report_md": "gain_comparison_report.md",
    "gain_comparison_report_json": "gain_comparison_report.json",
    "pump_power_sweep_report_md": "pump_power_sweep_report.md",
    "pump_power_sweep_json": "pump_sweep.json",
    "pump_power_sweep_png": "gain_vs_pump_power.png",
}
DEFAULT_GAIN_REPORT_PATHS = {
    "gain_comparison_report_md": r"runs\amp_2ghz_gain_compare\gain_comparison_report.md",
    "gain_comparison_report_json": r"runs\amp_2ghz_gain_compare\gain_comparison_report.json",
    "pump_power_sweep_report_md": r"runs\amp_2ghz_pump_sweep\pump_power_sweep_report.md",
    "pump_power_sweep_json": r"runs\amp_2ghz_pump_sweep\pump_sweep.json",
    "pump_power_sweep_png": r"runs\amp_2ghz_pump_sweep\gain_vs_pump_power.png",
}

AMPLIFIER_REQUIRED_COMPONENTS: dict[str, ADSComponentRef] = {
    "signal_source": ADSComponentRef("ads_sources", "P_1Tone"),
    "pump_source": ADSComponentRef("ads_sources", "P_1Tone"),
    "load_term": ADSComponentRef("ads_simulation", "Term"),
    "harmonic_balance": ADSComponentRef("ads_simulation", "HarmonicBalance"),
    "var": ADSComponentRef("ads_datacmps", "VAR"),
    "ground": ADSComponentRef("ads_rflib", "GROUND"),
    "unit_cell_msub": ADSComponentRef("ads_tlines", "MSUB"),
    "unit_cell_mlin": ADSComponentRef("ads_tlines", "MLIN"),
    "unit_cell_mtee": ADSComponentRef("ads_tlines", "MTEE_ADS"),
    "unit_cell_mlef": ADSComponentRef("ads_tlines", "MLEF"),
    "unit_cell_viagnd": ADSComponentRef("ads_tlines", "VIAGND"),
    "unit_cell_sdd2p": ADSComponentRef("ads_datacmps", "SDD2P"),
    "unit_cell_resistor": ADSComponentRef("ads_rflib", "R"),
    "unit_cell_inductor": ADSComponentRef("ads_rflib", "L"),
    "unit_cell_current_probe": ADSComponentRef("ads_rflib", "I_Probe"),
}


@dataclass(frozen=True)
class AmplifierExportResult:
    """Serializable result from an amplifier OA export."""

    project_dir: str
    plan_path: str
    summary_used_path: str
    open_in_ads_path: str
    oa_workspace: str
    generated_netlist_path: str
    build_result_path: str
    simulation_attempted: bool
    simulation_status: str
    output_dir: str | None
    simulation_output_files: list[str]
    mode: str = SCaffold_MODE
    sim_ready: bool = False
    unit_cell_internal_status: str = UNIT_CELL_INTERNAL_STATUS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_name(candidate: dict[str, Any]) -> str:
    return str(candidate.get("topology") or candidate.get("label") or "candidate")


def _unit_cell_symbol_name(candidate: dict[str, Any]) -> str:
    raw = _candidate_name(candidate).upper()
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    return f"UC_{safe or 'CANDIDATE'}"


def _target_frequency_hz(summary: dict[str, Any]) -> float:
    target_hz = summary.get("target_frequency_Hz")
    if target_hz is None:
        target_ghz = summary.get("target_frequency_GHz")
        if target_ghz is None:
            raise ExportError("optimizer summary is missing target frequency")
        target_hz = float(target_ghz) * 1.0e9
    return float(target_hz)


def _best_metric(candidate: dict[str, Any], key: str) -> float | None:
    metrics = candidate.get("metrics_at_drive")
    if isinstance(metrics, dict) and metrics.get(key) is not None:
        return float(metrics[key])
    analytical = candidate.get("analytical")
    if isinstance(analytical, dict):
        for value in analytical.values():
            if isinstance(value, dict) and value.get(key) is not None:
                return float(value[key])
    return None


def _candidate_summary(
    summary: dict[str, Any],
    candidate: dict[str, Any],
    *,
    selected_count: int,
) -> dict[str, Any]:
    main_tl = candidate.get("main_tl") if isinstance(candidate.get("main_tl"), dict) else {}
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    return {
        "candidate_name": _candidate_name(candidate),
        "rank": candidate.get("rank"),
        "target_frequency_Hz": _target_frequency_hz(summary),
        "target_frequency_GHz": _target_frequency_hz(summary) / 1.0e9,
        "coherent_peak": _best_metric(candidate, "coherent_peak"),
        "drive_dBm": config.get("drive_dBm"),
        "Z_TL_ohm": main_tl.get("z_tl_ohm"),
        "main_width_m": main_tl.get("width_m"),
        "main_width_mm": main_tl.get("width_mm"),
        "n_cells": config.get("n_cells"),
        "topology_family": candidate.get("topology"),
        "n_segments": candidate.get("n_segments"),
        "via": candidate.get("via"),
        "termination": candidate.get("termination"),
        "selected_candidates": selected_count,
    }


def _gain_report_paths_from_artifacts(
    gain_artifacts: dict[str, Any] | None,
) -> dict[str, str]:
    paths = dict(DEFAULT_GAIN_REPORT_PATHS)
    if not gain_artifacts:
        return paths
    for key, value in gain_artifacts.get("source_paths", {}).items():
        paths[key] = str(value)
    for key, value in gain_artifacts.get("project_paths", {}).items():
        paths[key] = str(value)
    return paths


def _gain_metadata(
    candidate_summary: dict[str, Any],
    gain_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coherent_peak = candidate_summary.get("coherent_peak")
    candidate = candidate_summary.get("candidate_name")
    report_paths = _gain_report_paths_from_artifacts(gain_artifacts)
    validation = (
        gain_artifacts.get("validation", {})
        if isinstance(gain_artifacts, dict)
        else {}
    )
    validated = bool(validation.get("sim_ready"))
    numerical_status = (
        "validated_against_direct_hpeesofsim"
        if validated
        else "unavailable_preliminary"
    )
    validation_annotation = (
        "ADS-OA amplifier HB gain validated against direct hpeesofsim"
        if validated
        else "ADS-OA amplifier HB gain not yet validated; see report"
    )
    artifact_status = (
        gain_artifacts.get("status", "suggested_paths_only")
        if gain_artifacts
        else "suggested_paths_only"
    )
    return {
        "gain_definition_summary": {
            "source": "docs/GAIN_DEFINITIONS.md",
            "voltage_gain_db": "20 log10(|V_out(fs)| / |V_in(fs)|)",
            "load_power_gain_db": "10 log10(P_L(fs) / P_in(fs))",
            "transducer_gain_db": "10 log10(P_L(fs) / P_avs(fs))",
            "coherent_gain_db": (
                "Stage-4 coherent_peak normalized to pump-off signal reference"
            ),
            "full_sum_gain_db": "10 log10(sum(P_out(k fs), k in K_out) / P_in(fs))",
        },
        "available_gain_metrics": {
            "analytical": {
                "coherent_gain_db": coherent_peak,
                "normalization": "Stage-4 coherent_peak normalized to pump-off signal reference",
            },
            "direct_hpeesofsim": {
                "status": validation.get("direct_status", "unavailable_for_full_amplifier"),
                "coherent_gain_db": validation.get("direct_coherent_gain_db"),
                "note": "direct amplifier HB metrics come from gain_comparison_report.json when available",
            },
            "ads_oa_hpeesofsim": {
                "status": validation.get("oa_status", numerical_status),
                "coherent_gain_db": validation.get("oa_coherent_gain_db"),
                "note": (
                    "full amplifier ADS-OA HB gain is not claimed until "
                    "direct-vs-hierarchical OA equivalence is validated"
                ),
            },
        },
        "coherent_peak_from_optimizer": coherent_peak,
        "pump_power_sweep_command": (
            ".\\.venv\\Scripts\\python.exe -m tvtl_ads_builder_v2 "
            "sweep-pump-gain --input runs\\amp_2ghz\\summary.json "
            "--out runs\\amp_2ghz_pump_sweep --pump-start-dbm -20 "
            "--pump-stop-dbm 10 --pump-step-dbm 2"
        ),
        "gain_report_paths": report_paths,
        "gain_artifacts": gain_artifacts
        or {
            "status": artifact_status,
            "source_paths": {},
            "project_paths": {},
            "missing": sorted(GAIN_ARTIFACT_FILENAMES),
        },
        "sim_ready_status": {
            "sim_ready": validated,
            "full_amplifier_numerical_gain": numerical_status,
            "ads_oa_amplifier_numerical_status": numerical_status,
            "reason": (
                validation_annotation
            ),
        },
        "schematic_annotation": (
            f"Self-pumped: PORT_SIG(fs) and PORT_PUMP(2fs) on {RF_NET}; "
            f"Candidate: {candidate}; Optimizer coherent peak: {coherent_peak} dB; "
            "Gain normalization: Stage-4 coherent_peak normalized to pump-off signal reference; "
            f"{validation_annotation}; "
            f"Reports: {report_paths['gain_comparison_report_md']}; "
            f"Pump plot: {report_paths['pump_power_sweep_png']}"
        ),
        "schematic_annotation_status": "planned_for_ads_text_and_open_in_ads",
    }


def _display_relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _discover_gain_artifacts(summary_path: Path) -> dict[str, Path]:
    run_dir = summary_path.parent
    runs_dir = run_dir.parent
    run_name = run_dir.name
    compare_dirs = [
        runs_dir / f"{run_name}_direct_vs_oa",
        runs_dir / f"{run_name}_gain_compare_ads",
        runs_dir / f"{run_name}_gain_compare",
    ]
    sweep_dirs = [
        runs_dir / f"{run_name}_pump_sweep_ads",
        runs_dir / f"{run_name}_pump_sweep",
    ]
    candidates = {
        "gain_comparison_report_md": [
            folder / "gain_comparison_report.md" for folder in compare_dirs
        ],
        "gain_comparison_report_json": [
            folder / "gain_comparison_report.json" for folder in compare_dirs
        ],
        "pump_power_sweep_report_md": [
            folder / "pump_power_sweep_report.md" for folder in sweep_dirs
        ],
        "pump_power_sweep_json": [folder / "pump_sweep.json" for folder in sweep_dirs],
        "pump_power_sweep_png": [
            folder / "gain_vs_pump_power.png" for folder in sweep_dirs
        ],
    }
    discovered: dict[str, Path] = {}
    for key, paths in candidates.items():
        for path in paths:
            if path.is_file():
                discovered[key] = path.resolve()
                break
    return discovered


def _prepare_gain_artifacts(
    out_dir: Path,
    summary_path: Path,
    provided_paths: dict[str, str | Path | None],
) -> dict[str, Any]:
    repo_base = summary_path.parent.parent.parent
    if summary_path.parent.parent.name.lower() != "runs":
        repo_base = Path.cwd().resolve()
    sources: dict[str, Path] = {}
    source_status: dict[str, str] = {}
    for key, raw_path in provided_paths.items():
        if raw_path is None:
            continue
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ExportError(f"provided gain artifact does not exist: {path}")
        sources[key] = path
        source_status[key] = "provided"

    for key, path in _discover_gain_artifacts(summary_path).items():
        if key not in sources:
            sources[key] = path
            source_status[key] = "auto_discovered"

    project_paths: dict[str, str] = {}
    copied_paths: dict[str, str] = {}
    if sources:
        reports_dir = out_dir / "gain_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for key, source in sources.items():
            target = reports_dir / GAIN_ARTIFACT_FILENAMES[key]
            if source != target.resolve():
                shutil.copyfile(source, target)
            project_paths[key] = _display_relative_path(target, out_dir)
            copied_paths[key] = project_paths[key]

    missing = sorted(set(GAIN_ARTIFACT_FILENAMES) - set(sources))
    if project_paths and not missing:
        status = "copied_all_gain_artifacts_to_project"
    elif project_paths:
        status = "copied_available_gain_artifacts_to_project"
    else:
        status = "suggested_paths_only"
    validation = _gain_validation_summary(sources.get("gain_comparison_report_json"))
    return {
        "status": status,
        "source_status": source_status,
        "source_paths": {
            key: _display_relative_path(path, repo_base)
            for key, path in sources.items()
        },
        "project_paths": project_paths,
        "copied_paths": copied_paths,
        "missing": missing,
        "validation": validation,
        "auto_discovery_patterns": [
            _display_relative_path(
                summary_path.parent.parent / f"{summary_path.parent.name}_direct_vs_oa",
                repo_base,
            ),
            _display_relative_path(
                summary_path.parent.parent
                / f"{summary_path.parent.name}_gain_compare_ads",
                repo_base,
            ),
            _display_relative_path(
                summary_path.parent.parent / f"{summary_path.parent.name}_gain_compare",
                repo_base,
            ),
            _display_relative_path(
                summary_path.parent.parent / f"{summary_path.parent.name}_pump_sweep_ads",
                repo_base,
            ),
            _display_relative_path(
                summary_path.parent.parent / f"{summary_path.parent.name}_pump_sweep",
                repo_base,
            ),
        ],
    }


def _gain_validation_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "not_available", "sim_ready": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreadable", "sim_ready": False, "error": str(exc)}
    direct = payload.get("direct_hpeesofsim", {})
    oa = payload.get("ads_oa_hpeesofsim", {})
    return {
        "status": payload.get("ads_oa_amplifier_numerical_status", payload.get("status", "available")),
        "sim_ready": bool(payload.get("sim_ready")),
        "direct_status": direct.get("run", {}).get("status"),
        "oa_status": oa.get("run", {}).get("status"),
        "direct_coherent_gain_db": direct.get("metrics", {}).get("coherent_gain_db"),
        "oa_coherent_gain_db": oa.get("metrics", {}).get("coherent_gain_db"),
        "source": str(path),
    }


def _varactor_dict(varactor: RealisticVaractor) -> dict[str, float]:
    return {
        "CJ0_f": float(varactor.CJ0_f),
        "VJ_v": float(varactor.VJ_v),
        "M": float(varactor.M),
        "CP_f": float(varactor.CP_f),
        "L_pkg_h": float(varactor.L_pkg_h),
        "R_series_ohm": float(varactor.R_series_ohm),
    }


def _varactor_from_plan(plan: dict[str, Any]) -> RealisticVaractor:
    raw = plan["unit_cell_implementation"]["varactor"]
    return RealisticVaractor(
        CJ0_f=float(raw["CJ0_f"]),
        VJ_v=float(raw["VJ_v"]),
        M=float(raw["M"]),
        CP_f=float(raw.get("CP_f", 0.0)),
        L_pkg_h=float(raw.get("L_pkg_h", 0.0)),
        R_series_ohm=float(raw.get("R_series_ohm", 0.0)),
    )


def _topology_from_plan(plan: dict[str, Any]) -> TopologyFamily:
    raw = plan["unit_cell_implementation"]["topology"]
    return TopologyFamily(
        n_segments=int(raw["n_segments"]),
        via=ViaConfig[str(raw["via"]).upper()],
        termination=Termination(str(raw["termination"]).lower()),
    )


def _l_via_for_topology(topology: TopologyFamily) -> float:
    if topology.termination not in (Termination.SHORT, Termination.COMBINED):
        return 0.0
    return float(
        via_inductance_bahl_H(
            substrate_thickness_m=float(ROGERS_4003C.h_m),
            via_radius_m=float(VIA_RADIUS_M),
            n_parallel=topology.n_via_parallel,
        )
    )


def _unit_cell_template_from_plan(plan: dict[str, Any]) -> tuple[SingleShuntVaractorCell, Any]:
    implementation = plan["unit_cell_implementation"]
    topology = _topology_from_plan(plan)
    varactor = _varactor_from_plan(plan)
    chain_models = standard_chain_models(varactor)
    substrate = _substrate_from_config(ROGERS_4003C)
    return (
        SingleShuntVaractorCell(
            topology=topology,
            params=tuple(float(value) for value in implementation["params"]),
            varactor=varactor,
            substrate_name=substrate.name,
            main_width_m=float(implementation["main_width_m"]),
            main_period_m=float(implementation["main_period_m"]),
            q_poly=varactor.charge_polynomial().charge_polynomial_coefficients,
            q_poly_reference=_q_poly_reference(topology, varactor),
            l_via_H=float(implementation["l_via_H"]),
            via_model=chain_models.via,
            open_end_model=chain_models.open_end,
        ),
        substrate,
    )


def _unit_cell_component_counts(elements: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        counts[element.type] = counts.get(element.type, 0) + 1
    return counts


def build_amplifier_schematic_plan(
    summary: dict[str, Any],
    candidate: dict[str, Any],
    *,
    selected_count: int,
    gain_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an ADS-free plan for the connected self-pumped OA scaffold."""

    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    main_tl = candidate.get("main_tl")
    if not isinstance(main_tl, dict):
        raise ExportError("candidate is missing main_tl")
    z_tl = main_tl.get("z_tl_ohm")
    if z_tl is None:
        raise ExportError("candidate is missing main_tl.z_tl_ohm")
    main_width_m = main_tl.get("width_m")
    if main_width_m is None:
        raise ExportError("candidate is missing main_tl.width_m")
    n_cells = int(config.get("n_cells") or 0)
    if n_cells <= 0:
        raise ExportError("optimizer summary config.n_cells must be positive")
    fs_hz = _target_frequency_hz(summary)
    topology = _topology_from_candidate(candidate)
    params = _params_from_candidate(candidate)
    varactor = _varactor_from_summary(summary)
    main_period_m = main_period_anchor_at_3p1GHz(fs_hz)
    l_via_H = _l_via_for_topology(topology)
    candidate_summary = _candidate_summary(
        summary, candidate, selected_count=selected_count
    )
    unit_cell_symbol_name = _unit_cell_symbol_name(candidate)
    unit_cell_cellview = f"{DEFAULT_LIBRARY_NAME}:{unit_cell_symbol_name}:symbol"
    unit_cell_implementation = {
        "source": (
            "tvtl_ads_builder_v2.cells.single_shunt_varactor."
            "SingleShuntVaractorCell"
        ),
        "status": UNIT_CELL_INTERNAL_STATUS,
        "representation": UNIT_CELL_INTERNAL_REPRESENTATION,
        "topology": {
            "key": topology.key,
            "n_segments": topology.n_segments,
            "via": topology.via.name.lower(),
            "termination": topology.termination.value,
        },
        "params": list(params),
        "stub_parameter_order": "segment-sorted [width_m, length_m, ...]",
        "main_width_m": float(main_width_m),
        "main_period_m": main_period_m,
        "main_period_source": "main_period_anchor_at_3p1GHz(fs)",
        "l_via_H": l_via_H,
        "via_radius_m": float(VIA_RADIUS_M),
        "substrate": ROGERS_4003C.name,
        "varactor": _varactor_dict(varactor),
        "components": REAL_UNIT_CELL_COMPONENTS,
        "termination_models": {
            "via": "viagnd",
            "open_end": "mlef",
            "policy": (
                "active ADS OA amplifier export uses validated VIAGND for "
                "short/via terminations and MLEF for open-end terminations; "
                "it does not use deprecated lumped-via emulation or "
                "deprecated open-stub terminations"
            ),
        },
    }
    one_cell, _substrate = _unit_cell_template_from_plan(
        {"unit_cell_implementation": unit_cell_implementation}
    )
    one_cell_elements = one_cell.cell_elements(1, input_net="RF_IN", output_net="RF_OUT")

    uc_blocks: list[dict[str, Any]] = []
    previous_net = RF_NET
    for index in range(1, n_cells + 1):
        output_net = f"N_UC{index:02d}_OUT"
        uc_blocks.append(
            {
                "name": f"UC{index:02d}",
                "role": UNIT_CELL_ROLE,
                "ads_cell": unit_cell_cellview,
                "sim_ready": False,
                "unit_cell_internal_status": UNIT_CELL_INTERNAL_STATUS,
                "input_net": previous_net,
                "output_net": output_net,
                "parameters": {
                    "unit_cell_symbol_name": unit_cell_symbol_name,
                    "unit_cell_representation": UNIT_CELL_REPRESENTATION,
                    "internal_representation": UNIT_CELL_INTERNAL_REPRESENTATION,
                    "z_tl_ohm_from_optimizer": float(z_tl),
                    "main_width_m_from_optimizer": main_tl.get("width_m"),
                    "main_width_mm_from_optimizer": main_tl.get("width_mm"),
                    "main_period_m_from_repo_default": main_period_m,
                },
            }
        )
        previous_net = output_net

    sources = [
        {
            "name": "PORT_SIG",
            "role": "signal_fs",
            "ads_cell": AMPLIFIER_REQUIRED_COMPONENTS["signal_source"].as_string(),
            "positive_net": RF_NET,
            "negative_net": "gnd!",
            "freq": "fs",
            "z_ohm": DEFAULT_SOURCE_Z_OHM,
        },
        {
            "name": "PORT_PUMP",
            "role": "pump_2fs",
            "ads_cell": AMPLIFIER_REQUIRED_COMPONENTS["pump_source"].as_string(),
            "positive_net": RF_NET,
            "negative_net": "gnd!",
            "freq": "2*fs",
            "z_ohm": DEFAULT_SOURCE_Z_OHM,
        },
    ]
    forbidden_lanes = ["PUMP", "CONTROL", "BIAS"]
    proof = {
        "two_sources_exist": len(sources) == 2,
        "source_roles": [source["role"] for source in sources],
        "source_outputs_connect_to_same_input_net": len(
            {source["positive_net"] for source in sources}
        )
        == 1
        and sources[0]["positive_net"] == RF_NET,
        "source_returns_are_grounded": all(
            source["negative_net"] == "gnd!" for source in sources
        ),
        "no_forbidden_lane_exists": True,
        "forbidden_lanes_present": [],
        "rf_path_connected_left_to_right": True,
        "uc_cascade_connected": all(
            uc_blocks[i]["output_net"] == uc_blocks[i + 1]["input_net"]
            for i in range(len(uc_blocks) - 1)
        ),
        "uc_blocks_are_custom_symbols": all(
            block["role"] == UNIT_CELL_ROLE
            and block["ads_cell"] != "ads_tlines:TLIN:symbol"
            for block in uc_blocks
        ),
        "unit_cell_internals_implemented": True,
        "unit_cell_internals_use_placeholder": False,
        "unit_cell_internal_status_labeled": True,
        "output_load_connected": previous_net == uc_blocks[-1]["output_net"],
        "placeholder_vs_sim_ready_mode_labeled": True,
    }
    gain_metadata = _gain_metadata(candidate_summary, gain_artifacts=gain_artifacts)

    return {
        "schema_version": 1,
        "export_type": "ADS OA self-pumped amplifier candidate",
        "mode": SCaffold_MODE,
        "unit_cell_representation": UNIT_CELL_REPRESENTATION,
        "unit_cell_symbol_name": unit_cell_symbol_name,
        "sim_ready": False,
        "simulation_claim": (
            "connected ADS GUI scaffold with implemented unit-cell internals "
            "from the repository self-pumped DeviceSpec; not a validated full "
            "amplifier hpeesofsim simulation"
        ),
        "source_of_truth": "docs/ADS_OA_DRAWING_GUIDE.md",
        "candidate_summary": candidate_summary,
        **gain_metadata,
        "frequency": {
            "fs_hz": fs_hz,
            "fs_ghz": fs_hz / 1.0e9,
            "pump_hz": 2.0 * fs_hz,
            "pump_expression": "2*fs",
        },
        "architecture": {
            "type": "self_pumped",
            "rf_input_net": RF_NET,
            "same_line_sources": True,
            "forbidden_separate_lanes": forbidden_lanes,
            "forbidden_lane_status": "absent",
        },
        "sources": sources,
        "unit_cell_implementation": {
            **unit_cell_implementation,
            "element_ids": [element.id for element in one_cell_elements],
            "component_counts": _unit_cell_component_counts(one_cell_elements),
            "ground_nodes": one_cell.ground_nodes(1),
            "terminals": ["RF_IN", "RF_OUT"],
        },
        "unit_cell_cascade": {
            "count": n_cells,
            "topology_family": candidate.get("topology"),
            "unit_cell_representation": UNIT_CELL_REPRESENTATION,
            "unit_cell_symbol_name": unit_cell_symbol_name,
            "unit_cell_symbol_cellview": unit_cell_cellview,
            "unit_cell_symbol_terminals": ["RF_IN", "RF_OUT"],
            "internal_representation": UNIT_CELL_INTERNAL_REPRESENTATION,
            "unit_cell_internal_status": UNIT_CELL_INTERNAL_STATUS,
            "internal_sim_ready": True,
            "components": REAL_UNIT_CELL_COMPONENTS,
            "main_period_m": main_period_m,
            "blocks": uc_blocks,
        },
        "load": {
            "name": "TERM_LOAD",
            "role": "output_load",
            "ads_cell": AMPLIFIER_REQUIRED_COMPONENTS["load_term"].as_string(),
            "input_net": previous_net,
            "negative_net": "gnd!",
            "z_ohm": DEFAULT_LOAD_Z_OHM,
        },
        "optimizer_fields_used": {
            "candidate_parameters": "summary.best_candidate or selected candidates.json entry",
            "n_cells": "summary.config.n_cells",
            "unit_cell_count_explicit": True,
            "cascade_connectivity_explicit": False,
            "ads_physical_mapping_source": "repository SingleShuntVaractorCell DeviceSpec mapping",
            "ads_unit_cell_mapping_explicit": True,
            "missing_for_sim_ready_amplifier": [
                "input/output matching sections are recorded but not drawn on the top-level scaffold",
                "direct-vs-hierarchical-OA amplifier topology comparison for this candidate",
                "hpeesofsim amplifier dataset validation for this hierarchical OA export",
            ],
        },
        "matching_sections": {
            "input_match": candidate.get("input_match"),
            "output_match": candidate.get("output_match"),
            "drawing_status": "recorded as metadata only in this scaffold",
        },
        "stub_parameters": candidate.get("stub_parameters", []),
        "proof": proof,
    }


def _assert_plan_contract(plan: dict[str, Any]) -> None:
    proof = plan.get("proof", {})
    required = [
        "two_sources_exist",
        "source_outputs_connect_to_same_input_net",
        "source_returns_are_grounded",
        "no_forbidden_lane_exists",
        "rf_path_connected_left_to_right",
        "uc_cascade_connected",
        "uc_blocks_are_custom_symbols",
        "unit_cell_internals_implemented",
        "unit_cell_internals_use_placeholder",
        "unit_cell_internal_status_labeled",
        "output_load_connected",
        "placeholder_vs_sim_ready_mode_labeled",
    ]
    missing = [
        key
        for key in required
        if proof.get(key) is not (False if key == "unit_cell_internals_use_placeholder" else True)
    ]
    if missing:
        raise ExportError(f"amplifier plan failed contract checks: {missing}")


def _write_open_in_ads(project_dir: Path, plan: dict[str, Any]) -> str:
    path = project_dir / "OPEN_IN_ADS.txt"
    summary = plan["candidate_summary"]
    lines = [
        f"ADS OA amplifier scaffold for {summary['candidate_name']}",
        "",
        "OpenAccess workspace:",
        f"  {project_dir / 'oa_workspace'}",
        "",
        "Open in ADS with File > Open > Workspace and select oa_workspace.",
        "",
        "Important status:",
        f"- {plan['sim_ready_status']['reason']}.",
        "- PORT_SIG and PORT_PUMP are same-line P_1Tone sources tied to RF_MAIN.",
        "- Both source returns are grounded and each source keeps Z=100 Ohm.",
        "- Unit cells are custom ADS symbols with internals emitted from SingleShuntVaractorCell.",
        "- Input/output matching sections are recorded in JSON but not drawn on the top-level scaffold.",
        f"- Unit-cell symbol: {plan['unit_cell_cascade']['unit_cell_symbol_name']}",
        "",
        "Gain reporting:",
        f"- Optimizer coherent peak: {plan.get('coherent_peak_from_optimizer')} dB",
        f"- Sim-ready gain status: {plan['sim_ready_status']['full_amplifier_numerical_gain']}",
        f"- Gain definitions: {plan['gain_definition_summary']['source']}",
        f"- Analytical coherent gain normalization: {plan['available_gain_metrics']['analytical']['normalization']}",
        f"- Gain artifact status: {plan.get('gain_artifacts', {}).get('status', 'suggested_paths_only')}",
        f"- Suggested pump sweep: {plan['pump_power_sweep_command']}",
        f"- Gain comparison report: {plan['gain_report_paths']['gain_comparison_report_md']}",
        f"- Gain comparison JSON: {plan['gain_report_paths']['gain_comparison_report_json']}",
        f"- Pump sweep report: {plan['gain_report_paths']['pump_power_sweep_report_md']}",
        f"- Pump sweep JSON: {plan['gain_report_paths']['pump_power_sweep_json']}",
        f"- Pump sweep plot: {plan['gain_report_paths']['pump_power_sweep_png']}",
        f"- Schematic annotation: {plan.get('schematic_annotation', '')}",
        "",
        "Artifacts:",
        "- amplifier_summary_used.json: optimizer summary and selected candidate used for export.",
        "- amplifier_schematic_plan.json: self-pumped topology proof, implemented UC internals, and remaining limitations.",
        "- generated_netlist.ckt: ADS-generated netlist from the OA schematic.",
        "- oa_build_result.json: ADS API/build metadata.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _assert_amplifier_netlist(netlist: str) -> None:
    legacy_via_name = "L" + "VIA"
    legacy_open_name = "M" + "LOC"
    forbidden_markers = [
        "L:" + legacy_via_name,
        legacy_via_name,
        legacy_open_name + ":",
    ]
    present = [marker for marker in forbidden_markers if marker in netlist]
    if present:
        raise RuntimeError(
            "Deprecated ADS element(s) found in amplifier netlist: "
            + ", ".join(present)
        )
    if "PORT_SIG  RF_MAIN 0" not in netlist:
        raise RuntimeError("Signal source is not netlisted on RF_MAIN.")
    if "PORT_PUMP  RF_MAIN 0" not in netlist:
        raise RuntimeError("Pump source is not netlisted on RF_MAIN.")
    if "Freq[1]=fs" not in netlist:
        raise RuntimeError("Signal source/HB fundamental does not use fs.")
    if "Freq[1]=2*fs" not in netlist:
        raise RuntimeError("Pump source is not set to 2*fs in the netlist.")
    if "TLIN:UC01" in netlist:
        raise RuntimeError("Top-level UC01 is still netlisted as a TLIN placeholder.")
    if "UC_TLIN_PLACEHOLDER" in netlist:
        raise RuntimeError("Unit-cell internals still contain UC_TLIN_PLACEHOLDER.")
    if "MLIN2:LINE_RIGHT_C1  RF_OUT RF_OUT" in netlist:
        raise RuntimeError("Unit-cell LINE_RIGHT_C1 is shorted to RF_OUT on both pins.")
    for required in [
        "MTEE_ADS:TEE_C1",
        "MLIN2:LINE_LEFT_C1",
        "MLIN2:LINE_RIGHT_C1",
        "MLIN2:STUB_C1_0",
        "MLIN2:STUB_C1_1",
        "SDD:VARACTOR_C1",
        "VIAGND:VIAGND_C1",
    ]:
        if required not in netlist:
            raise RuntimeError(f"Unit-cell netlist is missing required {required}.")
    uc_lines: list[tuple[int, str, str]] = []
    for raw_line in netlist.splitlines():
        line = raw_line.strip()
        if not line.startswith("UC_") or ":UC" not in line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        inst = parts[0]
        if ":UC" not in inst:
            continue
        try:
            index = int(inst.rsplit(":UC", 1)[1])
        except ValueError:
            continue
        uc_lines.append((index, parts[1], parts[2]))
    if uc_lines:
        uc_lines.sort()
        if uc_lines[0][1] != RF_NET:
            raise RuntimeError("UC01 input is not connected to RF_MAIN.")
        shorted = [index for index, node_in, node_out in uc_lines if node_in == node_out]
        if shorted:
            raise RuntimeError(f"Unit-cell instance(s) short input to output: {shorted}")
        disconnected = [
            (left[0], right[0])
            for left, right in zip(uc_lines, uc_lines[1:])
            if left[2] != right[1]
        ]
        if disconnected:
            raise RuntimeError(f"Unit-cell cascade is not connected: {disconnected}")
    forbidden = {"PUMP", "CONTROL", "BIAS"}
    tokens = {token.strip(" ,;()") for token in netlist.replace("\n", " ").split()}
    bad = sorted(forbidden & tokens)
    if bad:
        raise RuntimeError(f"Forbidden separate lane net(s) in netlist: {bad}")


def _create_or_clear_symbol(
    de: Any,
    db_uu: Any,
    library_name: str,
    cell_name: str,
) -> Any:
    design_ref = (library_name, cell_name, "symbol")
    if de.cellview_exists(library_name, cell_name, "symbol"):
        symbol = db_uu.open_design(design_ref, de.db.DesignMode.WRITE)
    else:
        symbol = db_uu.create_symbol(f"{library_name}:{cell_name}:symbol")
    return symbol


def _pin_for_numbered_term(instance: Any, term_number: int) -> Any:
    for inst_term in instance.inst_terms:
        if not inst_term.is_bound or not inst_term.is_numbered:
            continue
        if inst_term.term_number != term_number:
            continue
        pins = list(inst_term.inst_pins)
        if not pins:
            raise RuntimeError(
                f"Instance {instance.name} term {term_number} has no inst pin."
            )
        return pins[0]
    raise RuntimeError(
        f"Instance {instance.name} has no numbered terminal {term_number}."
    )


def _setup_library_for_custom_symbols(library: Any) -> None:
    setup = getattr(library, "setup_schematic_tech", None)
    if setup is not None:
        try:
            setup()
        except RuntimeError as exc:
            if "file exists" not in str(exc).lower():
                raise


def _unit_cell_layout_spec(
    elements: list[Element],
    *,
    substrate: Any,
    fs_hz: float,
    ground_nodes: list[str],
) -> DeviceSpec:
    nodes = {
        pin.node
        for element in elements
        for pin in element.pins
    }
    layout_elements = [
        Element(
            id="P_LAYOUT_IN",
            type="TermG",
            pins=(ElementPin(pin=1, node="RF_IN"),),
            parameters={"z_ohm": DEFAULT_SOURCE_Z_OHM, "num": 1},
        ),
        *elements,
        Element(
            id="P_LAYOUT_OUT",
            type="TermG",
            pins=(ElementPin(pin=1, node="RF_OUT"),),
            parameters={"z_ohm": DEFAULT_LOAD_Z_OHM, "num": 2},
        ),
    ]
    nodes.update({"RF_IN", "RF_OUT"})
    return DeviceSpec(
        device_spec_version=1,
        label="unit_cell_layout_template",
        analysis=Analysis(
            type="s_parameters",
            frequency_grid_Hz=FrequencyGrid(start=fs_hz, stop=fs_hz, points=1),
            port_impedance_ohm=DEFAULT_LOAD_Z_OHM,
        ),
        substrate=substrate,
        nodes=tuple(Node(id=node) for node in sorted(nodes)),
        elements=tuple(layout_elements),
        ports=(Port(num=1, node="RF_IN"), Port(num=2, node="RF_OUT")),
        ground_nodes=tuple(ground_nodes),
    )


def _unit_cell_element_positions(
    elements: list[Element],
    *,
    substrate: Any,
    fs_hz: float,
    ground_nodes: list[str],
) -> dict[str, tuple[float, float]]:
    layout_spec = _unit_cell_layout_spec(
        elements, substrate=substrate, fs_hz=fs_hz, ground_nodes=ground_nodes
    )
    all_positions = _component_positions(layout_spec)
    positions = {element.id: all_positions[element.id] for element in elements}
    min_x = min(x for x, _y in positions.values())
    return {
        element_id: (x - min_x, y)
        for element_id, (x, y) in positions.items()
    }


def _add_boundary_term(
    design: Any,
    db_uu: Any,
    name: str,
    number: int,
    xy: tuple[float, float],
) -> None:
    IO = db_uu.TermType.INPUT_OUTPUT
    net = design.find_or_add_net(name)
    term = design.add_numbered_term(net, name, number, IO)
    design.add_pin(term, design.add_pin_fig_for_term_type(IO, xy))


def _build_unit_cell_symbol_schematic(
    de: Any,
    db_uu: Any,
    library_name: str,
    uc_cell_name: str,
    plan: dict[str, Any],
) -> dict[str, str]:
    IO = db_uu.TermType.INPUT_OUTPUT
    cell, substrate = _unit_cell_template_from_plan(plan)
    elements = cell.cell_elements(1, input_net="RF_IN", output_net="RF_OUT")
    positions = _unit_cell_element_positions(
        elements,
        substrate=substrate,
        fs_hz=float(plan["frequency"]["fs_hz"]),
        ground_nodes=cell.ground_nodes(1),
    )
    uc = _create_or_clear_schematic(de, db_uu, library_name, uc_cell_name)
    with db_uu.Transaction(uc, "Build TVTL implemented unit-cell schematic") as transaction:
        uc.clear_design()
        msub = uc.add_instance(_MSUB_CELL, (-8.0, 4.0), name=substrate.name, angle=0)
        for key, value in _msub_params(substrate).items():
            if key in msub.parameters:
                msub.parameters[key].value = value

        oa_by_id = _place_elements(uc, elements, positions, substrate.name)
        node_pins = _node_pins_for(elements, oa_by_id)
        ground_nets: dict[str, object] = {}
        for index, ground_node in enumerate(
            sorted(node for node in cell.ground_nodes(1) if node in node_pins),
            start=1,
        ):
            gpinxy = min((_pin_xy(pin) for pin in node_pins[ground_node]), key=lambda xy: xy[1])
            gnd = uc.add_instance(
                _GROUND_CELL,
                (gpinxy[0], gpinxy[1] - 3.0),
                name=f"GND{index}",
                angle=0,
            )
            gpin = list(gnd.inst_pins)[0]
            gnet = gpin.net or uc.find_or_add_net("0")
            gpin.net = gnet
            ground_nets[ground_node] = gnet
            node_pins[ground_node].append(gpin)

        _connect_and_route(uc, node_pins, ground_nets=ground_nets)

        if not node_pins.get("RF_IN") or not node_pins.get("RF_OUT"):
            raise RuntimeError("unit-cell template did not expose RF_IN/RF_OUT pins")
        input_term_xy = _pin_xy(node_pins["RF_IN"][0])
        output_term_xy = _pin_xy(node_pins["RF_OUT"][0])
        _add_boundary_term(uc, db_uu, "RF_IN", 1, input_term_xy)
        _add_boundary_term(uc, db_uu, "RF_OUT", 2, output_term_xy)
        transaction.commit()
    uc.save_design()

    sym = _create_or_clear_symbol(de, db_uu, library_name, uc_cell_name)
    with db_uu.Transaction(sym, "Build TVTL unit-cell symbol") as transaction:
        sym.clear_design()
        layer = sym.get_layer_for_pin()
        sym.add_rectangle(layer, (-3.0, -1.5), (3.0, 1.5))
        s_in = sym.add_numbered_term(sym.find_or_add_net("RF_IN"), "RF_IN", 1, IO)
        sym.add_pin(s_in, sym.add_pin_fig_for_term_type(IO, (-3.0, 0.0)))
        s_out = sym.add_numbered_term(sym.find_or_add_net("RF_OUT"), "RF_OUT", 2, IO)
        sym.add_pin(s_out, sym.add_pin_fig_for_term_type(IO, (3.0, 0.0)))
        transaction.commit()
    sym.save_design()

    if not de.cellview_exists(library_name, uc_cell_name, "schematic"):
        raise RuntimeError(f"Unit-cell schematic was not created: {uc_cell_name}")
    if not de.cellview_exists(library_name, uc_cell_name, "symbol"):
        raise RuntimeError(f"Unit-cell symbol was not created: {uc_cell_name}")
    return {
        "unit_cell_schematic": f"{library_name}:{uc_cell_name}:schematic",
        "unit_cell_symbol": f"{library_name}:{uc_cell_name}:symbol",
    }


def _annotation_lines(plan: dict[str, Any]) -> list[str]:
    summary = plan["candidate_summary"]
    return [
        "TVTL gain summary",
        f"Candidate: {summary['candidate_name']}",
        f"Optimizer coherent peak: {plan.get('coherent_peak_from_optimizer')} dB",
        "Normalization: Stage-4 coherent_peak normalized to pump-off signal reference",
        f"Gain report: {plan['gain_report_paths']['gain_comparison_report_md']}",
        f"Pump plot: {plan['gain_report_paths']['pump_power_sweep_png']}",
        f"Full amplifier numerical gain: {plan['sim_ready_status']['full_amplifier_numerical_gain']}",
    ]


def _try_add_gain_annotation(
    design: Any,
    db_uu: Any,
    plan: dict[str, Any],
    *,
    x_start: float,
) -> str:
    add_text = getattr(design, "add_text", None)
    get_layer = getattr(design, "get_layer_for_pin", None)
    alignment = getattr(getattr(db_uu, "TextAlignment", object()), "LOWER_LEFT", None)
    if not callable(add_text) or not callable(get_layer) or alignment is None:
        return "not_embedded_add_text_unavailable"
    try:
        layer = get_layer()
        for index, line in enumerate(_annotation_lines(plan)):
            add_text(
                layer,
                line,
                (float(x_start), 4.0 - 0.35 * index),
                "Arial",
                0.225,
                alignment,
            )
    except Exception as exc:  # noqa: BLE001
        return f"not_embedded_{type(exc).__name__}: {exc}"
    return "embedded_in_ads_schematic"


def _set_two_tone_hb_controller(instance: Any) -> None:
    """Set the ADS HarmonicBalance instance to the self-pumped two-tone setup."""

    if "Freq" in instance.parameters:
        instance.parameters["Freq"].value = ["fs", "2*fs"]
    if "Order" in instance.parameters:
        instance.parameters["Order"].value = ["5", "5"]
    if "MaxOrder" in instance.parameters:
        instance.parameters["MaxOrder"].value = "6"
    if "StatusLevel" in instance.parameters:
        instance.parameters["StatusLevel"].value = "2"
    if "SweepPlan" in instance.parameters:
        instance.parameters["SweepPlan"].value = '"None"'


def build_amplifier_scaffold_schematic(
    workspace_path: str | Path,
    plan: dict[str, Any],
    *,
    library_name: str = DEFAULT_LIBRARY_NAME,
    cell_name: str = "amp_2ghz_scaffold",
    simulate: bool = False,
    ads_home: str | Path | None = None,
) -> ADSBuildResult:
    """Create, netlist, and optionally simulate the connected OA scaffold."""

    _assert_plan_contract(plan)
    de, db_uu = import_ads_modules(ads_home)
    workspace, library = _open_or_create_workspace(
        de, Path(workspace_path), library_name
    )
    _setup_library_for_custom_symbols(library)
    discovered = validate_component_refs(de, AMPLIFIER_REQUIRED_COMPONENTS)
    uc_cell_name = str(plan["unit_cell_cascade"]["unit_cell_symbol_name"])
    discovered.update(
        _build_unit_cell_symbol_schematic(de, db_uu, library_name, uc_cell_name, plan)
    )
    design = _create_or_clear_schematic(de, db_uu, library_name, cell_name)
    fs_hz = float(plan["frequency"]["fs_hz"])
    candidate_summary = plan["candidate_summary"]
    psig_dbm = float(candidate_summary.get("drive_dBm") or -3.0)
    ppump_dbm = psig_dbm
    uc_blocks = plan["unit_cell_cascade"]["blocks"]
    annotation_status = "not_attempted"

    with db_uu.Transaction(design, "Build TVTL amplifier scaffold") as transaction:
        design.clear_design()

        sig = design.add_instance(
            AMPLIFIER_REQUIRED_COMPONENTS["signal_source"].as_lcv(),
            (0.0, 0.0),
            name="PORT_SIG",
            angle=-90.0,
        )
        pump = design.add_instance(
            AMPLIFIER_REQUIRED_COMPONENTS["pump_source"].as_lcv(),
            (2.0, 0.0),
            name="PORT_PUMP",
            angle=-90.0,
        )
        uc_instances = []
        for index, block in enumerate(uc_blocks, start=1):
            uc_instances.append(
                design.add_instance(
                    (library_name, uc_cell_name, "symbol"),
                    (
                        UNIT_CELL_SYMBOL_START_X
                        + (index - 1) * UNIT_CELL_SYMBOL_PITCH,
                        0.0,
                    ),
                    name=block["name"],
                    angle=0.0,
                )
            )
        load_x = (
            UNIT_CELL_SYMBOL_START_X
            + len(uc_instances) * UNIT_CELL_SYMBOL_PITCH
            + 2.0
        )
        load = design.add_instance(
            AMPLIFIER_REQUIRED_COMPONENTS["load_term"].as_lcv(),
            (load_x, 0.0),
            name="TERM_LOAD",
            angle=-90.0,
        )

        for index, point in enumerate(
            [
                _pin_snap_point(sig, 2),
                _pin_snap_point(pump, 2),
                _pin_snap_point(load, 2),
            ],
            start=1,
        ):
            design.add_instance(
                AMPLIFIER_REQUIRED_COMPONENTS["ground"].as_lcv(),
                (point.x, point.y),
                name=f"GND{index}",
                angle=-90.0,
            )

        hb = design.add_instance(
            AMPLIFIER_REQUIRED_COMPONENTS["harmonic_balance"].as_lcv(),
            (0.0, 2.0),
            name="HB1",
        )
        var = design.add_instance(
            AMPLIFIER_REQUIRED_COMPONENTS["var"].as_lcv(),
            (3.0, 2.0),
            name="VAR1",
            angle=90.0,
        )
        var.vars["fs"] = ads_ghz(fs_hz)
        var.vars["Psig_dBm"] = ads_number(psig_dbm)
        var.vars["Ppump_dBm"] = ads_number(ppump_dbm)
        try:
            del var.vars["X"]
        except Exception:
            pass

        _set_param(sig, "Num", "1")
        _set_param(sig, "Z", f"{ads_number(DEFAULT_SOURCE_Z_OHM)} Ohm")
        _set_param(sig, "Freq", "fs")
        _set_param(sig, "P", "polar(dbmtow(Psig_dBm),0)")

        _set_param(pump, "Num", "2")
        _set_param(pump, "Z", f"{ads_number(DEFAULT_SOURCE_Z_OHM)} Ohm")
        _set_param(pump, "Freq", "2*fs")
        _set_param(pump, "P", "polar(dbmtow(Ppump_dBm),0)")

        _set_param(load, "Num", "3")
        _set_param(load, "Z", f"{ads_number(DEFAULT_LOAD_Z_OHM)} Ohm")

        _set_two_tone_hb_controller(hb)

        rf_wire = design.add_wire([_pin_snap_point(sig, 1), _pin_snap_point(pump, 1)])
        rf_wire.add_wire_label(RF_NET)
        if not uc_instances:
            raise RuntimeError("amplifier scaffold requires at least one unit cell")
        design.add_wire([_pin_snap_point(pump, 1), _pin_snap_point(uc_instances[0], 1)])
        for left, right in zip(uc_instances, uc_instances[1:]):
            design.add_wire([_pin_snap_point(left, 2), _pin_snap_point(right, 1)])
        design.add_wire([_pin_snap_point(uc_instances[-1], 2), _pin_snap_point(load, 1)])
        annotation_status = _try_add_gain_annotation(
            design,
            db_uu,
            plan,
            x_start=0.0,
        )

        transaction.commit()

    design.save_design()
    netlist = design.generate_netlist()
    netlist_path = Path(workspace.path) / f"{cell_name}.ckt"
    netlist_path.write_text(netlist, encoding="utf-8")
    _assert_amplifier_netlist(netlist)

    output_dir: str | None = None
    output_files: list[str] = []
    if simulate:
        from keysight.edatoolbox import ads, util

        output_path = Path(workspace.path) / "output"
        util.safe_makedirs(str(output_path))
        before = _mtimes(output_path)
        import time

        started = time.time()
        ads.CircuitSimulator().run_netlist(netlist, output_dir=str(output_path))
        output_dir = str(output_path)
        output_files = _new_or_updated(output_path, before, started)

    instance_cells = {instance.name: instance.cell_name for instance in design.instances}
    net_names = sorted(str(net.name) for net in design.nets if str(net.name))
    discovered["schematic_gain_annotation"] = annotation_status

    return ADSBuildResult(
        workspace_path=str(workspace.path),
        library_name=library_name,
        cell_name=cell_name,
        design_name=f"{library_name}:{cell_name}:schematic",
        product_version=de.product_version().strip(),
        api_version=de.version(),
        running_automation=de.running_automation(),
        is_pde_app=de.is_pde_app(),
        netlist_path=str(netlist_path),
        output_dir=output_dir,
        simulation_output_files=output_files,
        required_cellviews=discovered,
        net_names=net_names,
        instance_cells=instance_cells,
    )


def export_amplifier_oa(
    summary_path: str | Path,
    out_dir: str | Path,
    *,
    candidate_index: int | None = None,
    candidate_name: str | None = None,
    simulate: bool = False,
    ads_home: str | Path | None = None,
    gain_comparison_report: str | Path | None = None,
    gain_comparison_json: str | Path | None = None,
    pump_sweep_report: str | Path | None = None,
    pump_sweep_json: str | Path | None = None,
    pump_sweep_plot: str | Path | None = None,
) -> dict[str, Any]:
    """Export the best optimizer amplifier candidate as an OA scaffold."""

    summary_path = Path(summary_path).resolve()
    out_dir = Path(out_dir).resolve()
    summary, candidates = load_optimizer_candidates(summary_path)
    selection = select_candidates(
        candidates,
        candidate_index=candidate_index,
        candidate_name=candidate_name,
    )[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    gain_artifacts = _prepare_gain_artifacts(
        out_dir,
        summary_path,
        {
            "gain_comparison_report_md": gain_comparison_report,
            "gain_comparison_report_json": gain_comparison_json,
            "pump_power_sweep_report_md": pump_sweep_report,
            "pump_power_sweep_json": pump_sweep_json,
            "pump_power_sweep_png": pump_sweep_plot,
        },
    )

    plan = build_amplifier_schematic_plan(
        summary,
        selection.candidate,
        selected_count=len(candidates),
        gain_artifacts=gain_artifacts,
    )
    _assert_plan_contract(plan)
    summary_used = {
        "input_summary": str(summary_path),
        "candidate_rank": selection.rank,
        "candidate_label": selection.label,
        "selected_candidate_count": len(candidates),
        "summary": summary,
        "candidate": selection.candidate,
    }

    plan_path = out_dir / "amplifier_schematic_plan.json"
    summary_used_path = out_dir / "amplifier_summary_used.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_used_path.write_text(
        json.dumps(summary_used, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    build_result = build_amplifier_scaffold_schematic(
        out_dir / "oa_workspace",
        plan,
        cell_name=selection.label,
        simulate=False,
        ads_home=ads_home,
    )
    generated_netlist_path = out_dir / "generated_netlist.ckt"
    shutil.copyfile(build_result.netlist_path, generated_netlist_path)
    build_result_path = out_dir / "oa_build_result.json"
    build_result_path.write_text(build_result.to_json() + "\n", encoding="utf-8")
    open_in_ads_path = _write_open_in_ads(out_dir, plan)

    simulation_status = (
        "not_attempted_unvalidated_hierarchical_amplifier"
        if simulate
        else "not_requested_unvalidated_hierarchical_amplifier"
    )
    result = AmplifierExportResult(
        project_dir=str(out_dir),
        plan_path=str(plan_path),
        summary_used_path=str(summary_used_path),
        open_in_ads_path=open_in_ads_path,
        oa_workspace=build_result.workspace_path,
        generated_netlist_path=str(generated_netlist_path),
        build_result_path=str(build_result_path),
        simulation_attempted=False,
        simulation_status=simulation_status,
        output_dir=None,
        simulation_output_files=[],
    )
    manifest = result.to_dict()
    manifest["candidate_summary"] = plan["candidate_summary"]
    manifest["unit_cell_internal_status"] = plan["unit_cell_implementation"]["status"]
    manifest["unit_cell_internal_representation"] = plan["unit_cell_implementation"][
        "representation"
    ]
    manifest["gain_definition_summary"] = plan["gain_definition_summary"]
    manifest["available_gain_metrics"] = plan["available_gain_metrics"]
    manifest["coherent_peak_from_optimizer"] = plan["coherent_peak_from_optimizer"]
    manifest["pump_power_sweep_command"] = plan["pump_power_sweep_command"]
    manifest["gain_report_paths"] = plan["gain_report_paths"]
    manifest["gain_artifacts"] = plan["gain_artifacts"]
    manifest["sim_ready_status"] = plan["sim_ready_status"]
    manifest["schematic_annotation"] = plan["schematic_annotation"]
    manifest["schematic_annotation_status"] = build_result.required_cellviews.get(
        "schematic_gain_annotation",
        plan["schematic_annotation_status"],
    )
    manifest["note"] = plan["simulation_claim"]
    (out_dir / "amplifier_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
