"""Compare analytical, direct hpeesofsim, and ADS OA self-pumped paths."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..ads_oa_writer import build_self_pumped_minimal_schematic
from ..circuit_spec import SelfPumpedCircuitSpec
from .analytical_self_pumped import evaluate_analytical, write_analytical_result
from .direct_hpeesofsim import (
    HpeesofsimRunResult,
    direct_self_pumped_netlist,
    run_direct_hpeesofsim,
    write_direct_netlist,
)


FORBIDDEN_NETS = frozenset({"PUMP", "CONTROL", "BIAS"})
_ELEMENT_RE = re.compile(r"^(?P<kind>[A-Za-z_][\w]*):(?P<name>\S+)\s+(?P<body>.*)$")


def run_self_pumped_validation(
    spec: SelfPumpedCircuitSpec,
    out_dir: str | Path,
    *,
    simulate: bool = False,
    draw_oa: bool = False,
    oa_workspace_dir: str | Path | None = None,
    ads_home: str | Path | None = None,
) -> dict[str, Any]:
    """Run the requested validation paths and write comparison reports."""

    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    analytical = evaluate_analytical(spec)
    analytical_path = write_analytical_result(spec, out_path)

    direct_dir = out_path / "direct_hpeesofsim"
    direct_netlist_path = write_direct_netlist(spec, direct_dir)
    direct_netlist = direct_netlist_path.read_text(encoding="utf-8")
    if simulate:
        direct_run = run_direct_hpeesofsim(spec, direct_dir)
    else:
        direct_run = HpeesofsimRunResult(
            status="not_run",
            netlist_path=str(direct_netlist_path),
            output_dir=str(direct_dir),
            dataset_files=[],
            output_files=sorted(path.name for path in direct_dir.glob("*")),
            error_message="simulation requires --simulate and ADS Python",
        )

    oa: dict[str, Any] = {"status": "not_run"}
    oa_netlist: str | None = None
    if draw_oa:
        workspace = (
            Path(oa_workspace_dir).resolve()
            if oa_workspace_dir is not None
            else (Path.cwd() / "ads_projects" / "validation_self_pumped_oa").resolve()
        )
        try:
            build = build_self_pumped_minimal_schematic(
                workspace,
                spec=spec,
                simulate=simulate,
                ads_home=ads_home,
            )
            oa_netlist = Path(build.netlist_path).read_text(encoding="utf-8")
            dataset_files = sorted(
                name for name in build.simulation_output_files if name.endswith(".ds")
            )
            oa = {
                "status": "passed",
                "build": build.to_dict(),
                "dataset_files": dataset_files,
                "dataset_status": (
                    "passed"
                    if dataset_files
                    else ("not_run" if not simulate else "error")
                ),
            }
        except Exception as exc:  # noqa: BLE001 - keep report actionable
            oa = {"status": "error", "error_message": str(exc)}

    topology = compare_topology(
        direct_netlist,
        oa_netlist,
        expected_rf_net=spec.expected_rf_main_net,
    )
    simulation = {
        "direct_hpeesofsim": direct_run.to_dict(),
        "ads_oa_hpeesofsim": oa,
    }
    numerical = _try_numerical_comparison(spec, direct_run, oa)

    report = {
        "spec": spec.to_dict(),
        "paths": {
            "analytical_json": str(analytical_path),
            "direct_netlist": str(direct_netlist_path),
            "oa_workspace": (
                oa.get("build", {}).get("workspace_path") if isinstance(oa, dict) else None
            ),
            "oa_netlist": (
                oa.get("build", {}).get("netlist_path") if isinstance(oa, dict) else None
            ),
        },
        "analytical": analytical,
        "simulation": simulation,
        "topology_comparison": topology,
        "numerical_comparison": numerical,
    }
    _write_reports(report, out_path)
    return report


def compare_topology(
    direct_netlist: str,
    oa_netlist: str | None,
    *,
    expected_rf_net: str = "RF_MAIN",
) -> dict[str, Any]:
    """Compare direct and OA netlist topology, or direct topology alone."""

    direct = _inspect_netlist_topology(direct_netlist, expected_rf_net)
    oa = (
        _inspect_netlist_topology(oa_netlist, expected_rf_net)
        if oa_netlist is not None
        else {"status": "not_run", "passed": False, "checks": []}
    )
    checks = [
        {
            "name": "direct_netlist_topology",
            "passed": bool(direct["passed"]),
            "details": direct,
        }
    ]
    if oa_netlist is not None:
        checks.append(
            {
                "name": "oa_netlist_topology",
                "passed": bool(oa["passed"]),
                "details": oa,
            }
        )
        checks.append(
            {
                "name": "direct_and_oa_required_instances_match",
                "passed": _required_instance_set(direct) == _required_instance_set(oa),
                "details": {
                    "direct": sorted(_required_instance_set(direct)),
                    "oa": sorted(_required_instance_set(oa)),
                },
            }
        )
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "error",
        "scope": (
            "topology-only equivalence; generated dataset existence is checked "
            "separately"
        ),
        "checks": checks,
    }


def _inspect_netlist_topology(netlist: str, expected_rf_net: str) -> dict[str, Any]:
    elements = _parse_elements(netlist)
    nets = _component_nets(elements)
    forbidden = sorted(FORBIDDEN_NETS & nets)
    checks = [
        _check("contains PORT_SIG", "PORT_SIG" in elements),
        _check("contains PORT_PUMP", "PORT_PUMP" in elements),
        _check(
            "PORT_SIG uses Freq[1]=fs",
            _param(elements, "PORT_SIG", "Freq[1]") == "fs",
        ),
        _check(
            "PORT_PUMP uses Freq[1]=2*fs",
            _param(elements, "PORT_PUMP", "Freq[1]") == "2*fs",
        ),
        _check(
            "PORT_SIG output connects to RF_MAIN",
            _node(elements, "PORT_SIG", 0) == expected_rf_net,
        ),
        _check(
            "PORT_PUMP output connects to RF_MAIN",
            _node(elements, "PORT_PUMP", 0) == expected_rf_net,
        ),
        _check(
            "source returns are grounded",
            _node(elements, "PORT_SIG", 1) == "0"
            and _node(elements, "PORT_PUMP", 1) == "0",
        ),
        _check("no PUMP/CONTROL/BIAS net", not forbidden),
        _check(
            "TL_RF starts from RF_MAIN",
            _node(elements, "TL_RF", 0) == expected_rf_net,
        ),
        _check(
            "TERM_LOAD is downstream of TL_RF",
            _node(elements, "TL_RF", 1) == _node(elements, "TERM_LOAD", 0),
        ),
        _check(
            "HB fundamental is fs",
            _param(elements, "HB1", "Freq[1]") == "fs",
        ),
        _check(
            "HB order is present",
            _param(elements, "HB1", "Order[1]") is not None,
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "forbidden_nets": forbidden,
        "elements": {
            name: {
                "kind": data["kind"],
                "nodes": data["nodes"],
                "params": data["params"],
            }
            for name, data in elements.items()
        },
        "checks": checks,
    }


def _parse_elements(netlist: str) -> dict[str, dict[str, Any]]:
    elements: dict[str, dict[str, Any]] = {}
    for raw in netlist.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        match = _ELEMENT_RE.match(line)
        if not match:
            continue
        body = match.group("body")
        tokens = body.split()
        nodes = []
        params = {}
        for token in tokens:
            if "=" in token:
                key, value = token.split("=", 1)
                params[key] = value
            elif not params:
                nodes.append(token)
        elements[match.group("name")] = {
            "kind": match.group("kind"),
            "body": body,
            "nodes": nodes,
            "params": params,
        }
    return elements


def _component_nets(elements: dict[str, dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for data in elements.values():
        out.update(str(node) for node in data.get("nodes", []))
    return out


def _required_instance_set(inspected: dict[str, Any]) -> set[str]:
    elements = inspected.get("elements", {})
    return {name for name in ("PORT_SIG", "PORT_PUMP", "TL_RF", "TERM_LOAD", "HB1") if name in elements}


def _param(
    elements: dict[str, dict[str, Any]],
    name: str,
    key: str,
) -> str | None:
    return elements.get(name, {}).get("params", {}).get(key)


def _node(
    elements: dict[str, dict[str, Any]],
    name: str,
    index: int,
) -> str | None:
    nodes = elements.get(name, {}).get("nodes", [])
    return nodes[index] if index < len(nodes) else None


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _try_numerical_comparison(
    spec: SelfPumpedCircuitSpec,
    direct_run: HpeesofsimRunResult,
    oa: dict[str, Any],
) -> dict[str, Any]:
    if direct_run.status != "passed":
        return {
            "status": "not_available",
            "reason": "direct hpeesofsim dataset is not available",
        }
    oa_build = oa.get("build") if isinstance(oa, dict) else None
    if not oa_build or oa.get("dataset_status") != "passed":
        return {
            "status": "not_available",
            "reason": "ADS OA hpeesofsim dataset is not available",
        }
    direct_ds = _first_dataset_path(direct_run.output_dir, direct_run.dataset_files)
    oa_ds = _first_dataset_path(oa_build.get("output_dir"), oa.get("dataset_files", []))
    if direct_ds is None or oa_ds is None:
        return {"status": "not_available", "reason": "dataset path missing"}
    try:
        from ..analysis.hb_dataset import read_hb_dataset

        direct_hb = read_hb_dataset(direct_ds, [spec.fs_hz])
        oa_hb = read_hb_dataset(oa_ds, [spec.fs_hz])
        direct_v = direct_hb.node_voltage("RF_MAIN").at_mix((1,))
        oa_v = oa_hb.node_voltage("RF_MAIN").at_mix((1,))
    except Exception as exc:  # noqa: BLE001 - parser availability is optional
        return {
            "status": "not_available",
            "reason": f"RF_MAIN HB dataset comparison unavailable: {exc}",
        }
    denom = max(abs(direct_v), abs(oa_v), 1.0e-30)
    rel_error = abs(direct_v - oa_v) / denom
    return {
        "status": "passed" if rel_error <= 1.0e-9 else "error",
        "metric": "RF_MAIN voltage phasor at HB mix (1,)",
        "direct": [direct_v.real, direct_v.imag],
        "oa": [oa_v.real, oa_v.imag],
        "relative_error": rel_error,
        "tolerance": 1.0e-9,
    }


def _first_dataset_path(output_dir: str | None, names: list[str]) -> Path | None:
    if not output_dir or not names:
        return None
    return Path(output_dir) / names[0]


def _write_reports(report: dict[str, Any], out_path: Path) -> None:
    json_path = out_path / "comparison_report.json"
    md_path = out_path / "comparison_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    topology = report["topology_comparison"]
    direct = report["simulation"]["direct_hpeesofsim"]
    oa = report["simulation"]["ads_oa_hpeesofsim"]
    numerical = report["numerical_comparison"]
    lines = [
        "# Self-Pumped Validation Report",
        "",
        "## Scope",
        "",
        (
            "Analytical -> direct hpeesofsim -> ADS OA schematic -> ADS OA "
            "hpeesofsim -> comparison."
        ),
        "",
        "This minimal validation checks topology/connectivity equivalence for "
        "a passive RF path at fs. It does not claim nonlinear parametric gain.",
        "",
        "## Summary",
        "",
        f"- Analytical matched path: {report['analytical']['expected_matched']}",
        f"- Direct hpeesofsim: {direct['status']}",
        f"- ADS OA schematic generation: {oa.get('status', 'not_run')}",
        f"- ADS OA hpeesofsim: {oa.get('dataset_status', 'not_run')}",
        f"- Topology comparison: {topology['status']}",
        f"- Numerical comparison: {numerical['status']}",
        "",
        "## Required Topology",
        "",
        "- `PORT_SIG` and `PORT_PUMP` are both ADS `P_1Tone`/`Port:` sources.",
        "- Both source outputs connect to `RF_MAIN`.",
        "- The pump source uses `2*fs` on the same RF/main line.",
        "- No `PUMP`, `CONTROL`, or `BIAS` net/lane is used.",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in report["paths"].items():
        lines.append(f"- {key}: `{value}`")
    if numerical["status"] == "not_available":
        lines.extend(["", f"Numerical comparison reason: {numerical['reason']}"])
    return "\n".join(lines) + "\n"
