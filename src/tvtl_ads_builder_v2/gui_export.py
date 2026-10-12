"""Public ADS GUI export workflow for optimizer candidates."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tvtl.optimization.stage3_matching import MatchingSection
from tvtl.optimization.topology_generator import Termination, TopologyFamily, ViaConfig
from tvtl.varactors import RealisticVaractor

from .audit import audit_circuit
from .device_spec import DeviceSpec, dump_device_spec, load_device_spec
from .netlist import emit_ads_native_netlist, emit_hpeesofsim_netlist
from .parity.optimized_design import optimized_control_port_spec
from .topology import render_topology_svg

Backend = Literal["auto", "scaffold", "oa"]


class ExportError(RuntimeError):
    """User-facing export failure."""


@dataclass(frozen=True)
class CandidateSelection:
    """One optimizer candidate selected for export."""

    candidate: dict[str, Any]
    rank: int
    label: str


def _safe_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return out or "candidate"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportError(f"input summary not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExportError(f"input summary is not valid JSON: {path}") from exc


def _candidate_json_path(summary_path: Path, summary: dict[str, Any]) -> Path | None:
    direct = summary_path.parent / "candidates.json"
    if direct.is_file():
        return direct
    files = summary.get("files", {})
    raw = files.get("candidates_json") if isinstance(files, dict) else None
    if not raw:
        return None
    path = Path(str(raw))
    probes = [path]
    if not path.is_absolute():
        probes.append(summary_path.parent / path)
        probes.append(summary_path.parent / path.name)
    for probe in probes:
        if probe.is_file():
            return probe
    return None


def load_optimizer_candidates(summary_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load optimizer summary plus candidate list.

    ``summary.json`` always carries the best candidate. When the adjacent
    ``candidates.json`` is available, this returns the full list so ``--all``
    and ``--candidate-index`` can address every selected design.
    """

    path = Path(summary_path)
    summary = _read_json(path)
    if not isinstance(summary, dict):
        raise ExportError(f"input summary must be a JSON object: {path}")

    candidates_path = _candidate_json_path(path, summary)
    candidates: list[dict[str, Any]]
    if candidates_path is not None:
        payload = _read_json(candidates_path)
        if not isinstance(payload, list):
            raise ExportError(f"candidate file must contain a JSON list: {candidates_path}")
        candidates = [c for c in payload if isinstance(c, dict)]
    else:
        best = summary.get("best_candidate")
        candidates = [best] if isinstance(best, dict) else []

    if not candidates:
        raise ExportError("optimizer summary contains no exportable candidates")
    return summary, candidates


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    candidate_index: int | None = None,
    candidate_name: str | None = None,
    all_candidates: bool = False,
) -> list[CandidateSelection]:
    """Apply candidate selection options."""

    specified = sum(
        1 for value in (candidate_index, candidate_name, all_candidates) if value
    )
    if specified > 1:
        raise ExportError("choose only one of --candidate-index, --candidate-name, or --all")

    if all_candidates:
        chosen = candidates
    elif candidate_index is not None:
        if candidate_index < 1 or candidate_index > len(candidates):
            raise ExportError(
                f"candidate index {candidate_index} out of range; "
                f"valid range is 1..{len(candidates)}"
            )
        chosen = [candidates[candidate_index - 1]]
    elif candidate_name is not None:
        matches = [
            c for c in candidates
            if str(c.get("topology", "")) == candidate_name
            or str(c.get("label", "")) == candidate_name
        ]
        if not matches:
            raise ExportError(f"no candidate named {candidate_name!r}")
        chosen = matches
    else:
        chosen = [candidates[0]]

    selections: list[CandidateSelection] = []
    for fallback_rank, candidate in enumerate(chosen, start=1):
        rank = int(candidate.get("rank") or fallback_rank)
        topology = str(candidate.get("topology") or "candidate")
        label = f"rank{rank:02d}_{_safe_name(topology)}"
        selections.append(CandidateSelection(candidate=candidate, rank=rank, label=label))
    return selections


def _topology_from_candidate(candidate: dict[str, Any]) -> TopologyFamily:
    try:
        n_segments = int(candidate["n_segments"])
        via = ViaConfig[str(candidate["via"]).upper()]
        termination = Termination(str(candidate["termination"]).lower())
    except (KeyError, ValueError) as exc:
        raise ExportError("candidate is missing topology fields") from exc
    return TopologyFamily(n_segments=n_segments, via=via, termination=termination)


def _params_from_candidate(candidate: dict[str, Any]) -> tuple[float, ...]:
    raw = candidate.get("stub_parameters")
    if not isinstance(raw, list) or not raw:
        raise ExportError("candidate is missing stub_parameters")
    params: list[float] = []
    for entry in sorted(raw, key=lambda p: int(p.get("segment", 0))):
        try:
            params.extend([float(entry["width_m"]), float(entry["length_m"])])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExportError("candidate stub_parameters must include width_m and length_m") from exc
    return tuple(params)


def _varactor_from_summary(summary: dict[str, Any]) -> RealisticVaractor:
    raw = summary.get("varactor")
    if not isinstance(raw, dict):
        raise ExportError("optimizer summary is missing varactor settings")
    try:
        return RealisticVaractor(
            CJ0_f=float(raw["CJ0_f"]),
            VJ_v=float(raw["VJ_v"]),
            M=float(raw["M"]),
            CP_f=float(raw.get("CP_f", 0.0)),
            L_pkg_h=float(raw.get("L_pkg_h", 0.0)),
            R_series_ohm=float(raw.get("R_series_ohm", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportError("optimizer summary has incomplete varactor settings") from exc


def _matching_from_candidate(candidate: dict[str, Any], key: str) -> MatchingSection | None:
    raw = candidate.get(key)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ExportError(f"{key} must be an object or null")
    elements = []
    for entry in raw.get("elements", []):
        try:
            elements.append(
                (
                    str(entry["kind"]),
                    float(entry["width_m"]),
                    float(entry["length_m"]),
                    float(entry["z0_ohm"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExportError(f"{key} contains an incomplete matching element") from exc
    return MatchingSection(
        topology=str(raw["topology"]),
        elements=tuple(elements),
        s11_before_dB=float(raw.get("s11_before_dB", 0.0)),
        s11_after_dB=float(raw.get("s11_after_dB", 0.0)),
        frac_bandwidth_pct=float(raw.get("frac_bandwidth_pct", 0.0)),
        realizable=bool(raw.get("realizable", True)),
        note=str(raw.get("note", "")),
    )


def spec_from_optimizer_candidate(
    summary: dict[str, Any],
    candidate: dict[str, Any],
    *,
    label: str,
) -> DeviceSpec:
    """Rebuild an ADS-emittable DeviceSpec from optimizer JSON output."""

    config = summary.get("config", {})
    if not isinstance(config, dict):
        config = {}
    target_hz = summary.get("target_frequency_Hz")
    if target_hz is None:
        target_hz = float(summary["target_frequency_GHz"]) * 1.0e9
    main_tl = candidate.get("main_tl")
    if not isinstance(main_tl, dict) or main_tl.get("width_m") is None:
        raise ExportError("candidate is missing main_tl.width_m")
    return optimized_control_port_spec(
        topology=_topology_from_candidate(candidate),
        params=_params_from_candidate(candidate),
        varactor=_varactor_from_summary(summary),
        n_cells=int(config.get("n_cells", 15)),
        main_width_m=float(main_tl["width_m"]),
        f_signal_Hz=float(target_hz),
        f_pump_Hz=2.0 * float(target_hz),
        pump_power_dBm=float(config.get("drive_dBm", -3.0)),
        input_match=_matching_from_candidate(candidate, "input_match"),
        output_match=_matching_from_candidate(candidate, "output_match"),
        spec_id=label,
    )


def _write_text_artifacts(spec: DeviceSpec, project_dir: Path) -> dict[str, str]:
    project_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "device_spec": project_dir / "device_spec.device_spec.json",
        "hpeesofsim_netlist": project_dir / "hpeesofsim.ckt",
        "ads_native_netlist": project_dir / "ads_native.net",
        "topology_svg": project_dir / "topology.svg",
        "adjacency": project_dir / "adjacency.txt",
    }
    dump_device_spec(spec, paths["device_spec"])
    hpeesofsim_text = emit_hpeesofsim_netlist(spec)
    paths["hpeesofsim_netlist"].write_text(hpeesofsim_text, encoding="utf-8")
    try:
        ads_native_text = emit_ads_native_netlist(spec)
    except Exception as exc:  # noqa: BLE001
        ads_native_text = (
            "# ADS-native netlist text was not emitted for this full HB spec.\n"
            "# Use hpeesofsim.ckt as the authoritative flat ADS/HB deck, or\n"
            "# materialize oa_workspace with ADS Python and generate a GUI netlist.\n"
            f"# Reason: {exc}\n"
        )
    paths["ads_native_netlist"].write_text(ads_native_text, encoding="utf-8")
    try:
        render_topology_svg(spec, paths["topology_svg"])
    except Exception as exc:  # noqa: BLE001
        paths["topology_svg"].write_text(
            f"Topology SVG generation skipped: {exc}\n", encoding="utf-8"
        )
    try:
        audit = audit_circuit(spec, hpeesofsim_text)
        paths["adjacency"].write_text(audit.report_text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        paths["adjacency"].write_text(
            f"Adjacency audit skipped: {exc}\n", encoding="utf-8"
        )
    return {key: str(value) for key, value in paths.items()}


def _write_build_script(project_dir: Path) -> dict[str, str]:
    script = project_dir / "build_oa_workspace.py"
    batch = project_dir / "build_oa_workspace.bat"
    script.write_text(
        """from __future__ import annotations

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for parent in (root, *root.parents):
    src = parent / "src"
    if (src / "tvtl_ads_builder_v2").is_dir():
        sys.path.insert(0, str(src))
        break

ads_install = os.environ.get("ADS_INSTALL_DIR")
if ads_install and not os.environ.get("HPEESOF_DIR"):
    os.environ["HPEESOF_DIR"] = ads_install

from tvtl_ads_builder_v2.device_spec import load_device_spec
from tvtl_ads_builder_v2.oa.oa_emitter import emit_workspace_from_spec

spec = load_device_spec(root / "device_spec.device_spec.json")
info = emit_workspace_from_spec(
    spec,
    root / "oa_workspace",
    library_name="floquet_line_design",
    cell_name=spec.label,
)
print("ADS OpenAccess workspace:", info["workspace_path"])
print("ADS-generated netlist:", info["ads_netlist_path"])
print("Flat hpeesofsim deck:", info["hpeesofsim_netlist_path"])
""",
        encoding="utf-8",
    )
    batch.write_text(
        """@echo off
setlocal
if "%ADS_PYTHON%"=="" (
  if not "%ADS_INSTALL_DIR%"=="" set "ADS_PYTHON=%ADS_INSTALL_DIR%\\tools\\python\\python.exe"
)
if "%ADS_PYTHON%"=="" (
  echo Set ADS_PYTHON or ADS_INSTALL_DIR to your Keysight ADS installation.
  exit /b 2
)
"%ADS_PYTHON%" "%~dp0build_oa_workspace.py"
""",
        encoding="utf-8",
    )
    return {"build_script": str(script), "build_batch": str(batch)}


def _configure_ads_env_from_install_dir() -> None:
    ads_install = os.environ.get("ADS_INSTALL_DIR")
    if ads_install and not os.environ.get("HPEESOF_DIR"):
        os.environ["HPEESOF_DIR"] = ads_install


def _ads_de_available() -> tuple[bool, str | None]:
    _configure_ads_env_from_install_dir()
    try:
        import keysight.ads.de  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, None


def _emit_oa_workspace(spec: DeviceSpec, project_dir: Path) -> dict[str, str]:
    from .oa.oa_emitter import emit_workspace_from_spec

    return emit_workspace_from_spec(
        spec,
        project_dir / "oa_workspace",
        library_name="floquet_line_design",
        cell_name=spec.label,
    )


def _write_open_in_ads(project_dir: Path, *, label: str, backend_status: str) -> str:
    path = project_dir / "OPEN_IN_ADS.txt"
    path.write_text(
        "\n".join(
            [
                f"ADS GUI export for {label}",
                "",
                "Files in this directory:",
                "- device_spec.device_spec.json: public DeviceSpec rebuilt from optimizer output.",
                "- hpeesofsim.ckt: flat HB deck for command-line ADS simulation.",
                "- ads_native.net: ADS-native netlist text when the spec subset supports it.",
                "- topology.svg and adjacency.txt: ADS-free inspection aids.",
                "- build_oa_workspace.py / .bat: run with ADS Python to create oa_workspace.",
                "",
                f"OpenAccess workspace status: {backend_status}",
                "",
                "To create or refresh the GUI workspace:",
                "  1. Set ADS_PYTHON or ADS_INSTALL_DIR for your local Keysight ADS install.",
                "  2. Run build_oa_workspace.bat, or run ADS Python on build_oa_workspace.py.",
                "  3. In ADS GUI, use File > Open > Workspace and select the oa_workspace directory.",
                "",
                "The pure Python optimizer and this portable scaffold do not require a .env file.",
                "ADS/HB/OpenAccess verification requires Keysight ADS and the ADS Python bridge.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def export_gui_project(
    summary_path: str | Path,
    out_dir: str | Path,
    *,
    candidate_index: int | None = None,
    candidate_name: str | None = None,
    all_candidates: bool = False,
    backend: Backend = "auto",
) -> dict[str, Any]:
    """Export optimizer candidates as public ADS GUI project packages."""

    if backend not in {"auto", "scaffold", "oa"}:
        raise ExportError(f"unknown backend {backend!r}")
    summary_path = Path(summary_path).resolve()
    out_dir = Path(out_dir).resolve()
    summary, candidates = load_optimizer_candidates(summary_path)
    selections = select_candidates(
        candidates,
        candidate_index=candidate_index,
        candidate_name=candidate_name,
        all_candidates=all_candidates,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    materialize_oa = False
    ads_error: str | None = None
    if backend in {"auto", "oa"}:
        materialize_oa, ads_error = _ads_de_available()
        if backend == "oa" and not materialize_oa:
            raise ExportError(
                "ADS OpenAccess export requires keysight.ads.de. "
                f"Run with ADS Python or set ADS_INSTALL_DIR/HPEESOF_DIR. Detail: {ads_error}"
            )

    exported = []
    multiple = len(selections) > 1
    for selection in selections:
        project_dir = out_dir / selection.label if multiple else out_dir
        spec = spec_from_optimizer_candidate(
            summary, selection.candidate, label=selection.label
        )
        files = _write_text_artifacts(spec, project_dir)
        files.update(_write_build_script(project_dir))
        oa_info = None
        status = "portable scaffold only; run build_oa_workspace with ADS Python"
        if materialize_oa:
            oa_info = _emit_oa_workspace(spec, project_dir)
            status = f"created at {oa_info['workspace_path']}"
        files["open_in_ads"] = _write_open_in_ads(
            project_dir, label=selection.label, backend_status=status
        )
        exported.append(
            {
                "rank": selection.rank,
                "label": selection.label,
                "topology": selection.candidate.get("topology"),
                "project_dir": str(project_dir),
                "backend": "oa" if oa_info else "scaffold",
                "oa_workspace": None if oa_info is None else oa_info["workspace_path"],
                "files": files,
                "ads_error": ads_error if oa_info is None else None,
            }
        )

    manifest = {
        "input_summary": str(summary_path),
        "output_dir": str(out_dir),
        "backend_requested": backend,
        "backend_materialized": "oa" if any(e["backend"] == "oa" for e in exported) else "scaffold",
        "ads_python_required_for_oa": True,
        "env_required_for_optimizer": False,
        "candidates": exported,
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_workspace_from_project(project_dir: str | Path) -> dict[str, str]:
    """Materialize an OA workspace from a previously exported project package."""

    project_dir = Path(project_dir)
    spec_path = project_dir / "device_spec.device_spec.json"
    if not spec_path.is_file():
        raise ExportError(f"device spec not found: {spec_path}")
    _configure_ads_env_from_install_dir()
    return _emit_oa_workspace(load_device_spec(spec_path), project_dir)
