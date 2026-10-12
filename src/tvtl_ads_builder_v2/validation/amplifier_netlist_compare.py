"""Public amplifier direct-vs-ADS-OA netlist comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..amplifier_spec import (
    AmplifierCircuitSpec,
    PUMP_SOURCE_NAME,
    RF_MAIN_NET,
    SIGNAL_SOURCE_NAME,
)

FORBIDDEN_LANE_NETS = {"PUMP", "CONTROL", "BIAS"}
DEPRECATED_MARKERS = ("L:LVIA", "LVIA", "MLOC:")


def compare_amplifier_netlists(
    direct_netlist_text: str,
    oa_netlist_text: str,
    spec: AmplifierCircuitSpec,
) -> dict[str, Any]:
    """Compare direct and ADS-OA amplifier netlists for public policy parity."""

    checks: list[dict[str, Any]] = []
    checks.extend(_source_checks(direct_netlist_text, label="direct"))
    checks.extend(_source_checks(oa_netlist_text, label="ads_oa"))
    checks.extend(_forbidden_checks(direct_netlist_text, label="direct"))
    checks.extend(_forbidden_checks(oa_netlist_text, label="ads_oa"))
    checks.extend(_element_model_checks(direct_netlist_text, oa_netlist_text, spec))
    checks.extend(_unit_cell_checks(direct_netlist_text, oa_netlist_text, spec))
    checks.extend(_hb_checks(direct_netlist_text, oa_netlist_text, spec))
    checks.extend(_matching_checks(spec))

    failures = [check for check in checks if check["status"] == "failed"]
    warnings = [check for check in checks if check["status"] == "warning"]
    return {
        "status": "passed" if not failures else "failed",
        "match": not failures,
        "candidate": spec.candidate_name,
        "topology_family": spec.topology_family,
        "n_cells": spec.n_cells,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "direct_cell_count": _direct_cell_count(direct_netlist_text),
            "oa_unit_cell_instances": _oa_unit_cell_instance_count(oa_netlist_text),
            "direct_viagnd_count": direct_netlist_text.count("VIAGND:"),
            "oa_viagnd_present": "VIAGND" in oa_netlist_text,
            "direct_mlef_count": _count_model_token(direct_netlist_text, "MLEF"),
            "oa_mlef_present": "MLEF" in oa_netlist_text,
            "matching_status": spec.matching_status,
        },
    }


def write_amplifier_netlist_comparison(
    direct_netlist_path: str | Path,
    oa_netlist_path: str | Path,
    spec: AmplifierCircuitSpec,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Write JSON and Markdown comparison reports."""

    direct_path = Path(direct_netlist_path)
    oa_path = Path(oa_netlist_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report = compare_amplifier_netlists(
        direct_path.read_text(encoding="utf-8", errors="replace"),
        oa_path.read_text(encoding="utf-8", errors="replace"),
        spec,
    )
    report["paths"] = {
        "direct_netlist": str(direct_path),
        "oa_netlist": str(oa_path),
        "json": str(out_path / "netlist_comparison.json"),
        "markdown": str(out_path / "netlist_comparison.md"),
    }
    (out_path / "netlist_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_path / "netlist_comparison.md").write_text(
        format_netlist_comparison_markdown(report),
        encoding="utf-8",
    )
    return report


def format_netlist_comparison_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise Markdown netlist-comparison report."""

    lines = [
        "# Amplifier Netlist Comparison",
        "",
        f"- Candidate: `{report['candidate']}`",
        f"- Topology: `{report['topology_family']}`",
        f"- Status: `{report['status']}`",
        f"- Unit cells: `{report['n_cells']}`",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in report.get("checks", []):
        lines.append(
            f"| {check['name']} | {check['status']} | "
            f"{str(check.get('detail', '')).replace('|', '/') } |"
        )
    if report.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item['name']}: {item.get('detail', '')}" for item in report["failures"])
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item['name']}: {item.get('detail', '')}" for item in report["warnings"])
    return "\n".join(lines) + "\n"


def _source_checks(text: str, *, label: str) -> list[dict[str, Any]]:
    checks = []
    for name in (SIGNAL_SOURCE_NAME, PUMP_SOURCE_NAME):
        source = _source_line(text, name)
        if source is None:
            checks.append(_check(f"{label}.{name}.present", False, "source line not found"))
            continue
        positive, negative, line = source
        checks.append(
            _check(
                f"{label}.{name}.same_rf_main",
                positive == RF_MAIN_NET and negative == "0",
                line,
            )
        )
    return checks


def _forbidden_checks(text: str, *, label: str) -> list[dict[str, Any]]:
    tokens = {token.strip(" ,;()") for token in text.replace("\n", " ").split()}
    lanes = sorted(FORBIDDEN_LANE_NETS & tokens)
    deprecated = [marker for marker in DEPRECATED_MARKERS if marker in text]
    return [
        _check(f"{label}.no_forbidden_lanes", not lanes, f"forbidden lanes: {lanes}"),
        _check(f"{label}.no_deprecated_elements", not deprecated, f"deprecated markers: {deprecated}"),
    ]


def _element_model_checks(
    direct: str,
    oa: str,
    spec: AmplifierCircuitSpec,
) -> list[dict[str, Any]]:
    viagnd_expected = spec.termination_type in {"short", "combined"}
    mlef_expected = spec.termination_type in {"open", "combined"}
    checks = [
        _check(
            "direct.viagnd_expected",
            (not viagnd_expected) or "VIAGND:" in direct,
            f"expected={viagnd_expected}",
        ),
        _check(
            "ads_oa.viagnd_expected",
            (not viagnd_expected) or "VIAGND" in oa,
            f"expected={viagnd_expected}",
        ),
        _check(
            "direct.mlef_expected",
            (not mlef_expected) or "MLEF:" in direct,
            f"expected={mlef_expected}",
        ),
        _check(
            "ads_oa.mlef_expected",
            (not mlef_expected) or "MLEF" in oa,
            f"expected={mlef_expected}",
        ),
    ]
    if not mlef_expected:
        checks.append(_check("mlef.not_applicable", True, "topology has no open-end termination"))
    return checks


def _unit_cell_checks(
    direct: str,
    oa: str,
    spec: AmplifierCircuitSpec,
) -> list[dict[str, Any]]:
    direct_cells = _direct_cell_count(direct)
    oa_instances = _oa_unit_cell_instance_count(oa)
    family_markers = {
        "MTEE_ADS": "MTEE_ADS:TEE_C1" in direct and "TEE_C1" in oa,
        "MLIN": "MLIN:LINE_LEFT_C1" in direct and "LINE_LEFT_C1" in oa,
        "SDD": "SDD:VARACTOR_C1" in direct and "VARACTOR_C1" in oa,
    }
    return [
        _check(
            "unit_cells.direct_count",
            direct_cells == spec.n_cells,
            f"direct={direct_cells}, expected={spec.n_cells}",
        ),
        _check(
            "unit_cells.ads_oa_instance_count",
            oa_instances == spec.n_cells,
            f"ads_oa={oa_instances}, expected={spec.n_cells}",
        ),
        _check(
            "unit_cells.element_family_present",
            all(family_markers.values()),
            str(family_markers),
        ),
    ]


def _hb_checks(direct: str, oa: str, spec: AmplifierCircuitSpec) -> list[dict[str, Any]]:
    direct_hb = _hb_line(direct)
    oa_hb = _hb_line(oa)
    checks = [
        _check("direct.hb_two_tone", "Freq[1]=" in (direct_hb or "") and "Freq[2]=" in (direct_hb or ""), direct_hb),
        _check("ads_oa.hb_two_tone", "Freq[1]=" in (oa_hb or "") and "Freq[2]=" in (oa_hb or ""), oa_hb),
        _check("direct.hb_order2", "Order[2]=" in (direct_hb or ""), direct_hb),
        _check("ads_oa.hb_order2", "Order[2]=" in (oa_hb or ""), oa_hb),
    ]
    checks.append(
        _check(
            "hb.frequency_ratio",
            _has_frequency_ratio(direct_hb, spec) and _oa_mentions_two_tone_frequency(oa),
            f"direct={direct_hb}; oa={oa_hb}",
        )
    )
    return checks


def _matching_checks(spec: AmplifierCircuitSpec) -> list[dict[str, Any]]:
    if spec.matching_status == "none":
        return [_check("matching.consistent", True, "no matching sections recorded")]
    return [
        {
            "name": "matching.consistent",
            "status": "warning",
            "detail": (
                "optimizer matching sections are recorded but excluded from both "
                "direct and OA decks to match the current public OA scaffold"
            ),
        }
    ]


def _source_line(text: str, name: str) -> tuple[str, str, str] | None:
    pattern = re.compile(rf"^(?:Port|P_1Tone):{re.escape(name)}\s+(\S+)\s+(\S+).*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(0)


def _hb_line(text: str) -> str | None:
    match = re.search(r"^HB:HB1\b.*$", text, flags=re.MULTILINE)
    return match.group(0) if match else None


def _direct_cell_count(text: str) -> int:
    return len(set(re.findall(r"MTEE_ADS:TEE_C(\d+)\b", text)))


def _oa_unit_cell_instance_count(text: str) -> int:
    return len(re.findall(r"^UC_[A-Z0-9_]+:UC\d+\b", text, flags=re.MULTILINE))


def _count_model_token(text: str, token: str) -> int:
    return len(re.findall(rf"\b{re.escape(token)}\w*:", text))


def _has_frequency_ratio(hb_line: str | None, spec: AmplifierCircuitSpec) -> bool:
    if hb_line is None:
        return False
    f1 = _extract_freq_ghz(hb_line, 1)
    f2 = _extract_freq_ghz(hb_line, 2)
    if f1 is None or f2 is None:
        return False
    return abs(f1 - spec.fs_hz / 1.0e9) < 1.0e-6 and abs(f2 - spec.pump_hz / 1.0e9) < 1.0e-6


def _extract_freq_ghz(line: str, index: int) -> float | None:
    match = re.search(rf"Freq\[{index}\]=([0-9.eE+-]+)\s*GHz", line)
    if not match:
        return None
    return float(match.group(1))


def _oa_mentions_two_tone_frequency(text: str) -> bool:
    return (
        ("Freq[1]=fs" in text or "Freq[1]=\"fs\"" in text or "Freq[1]=2" in text)
        and ("Freq[2]=2*fs" in text or "Freq[2]=\"2*fs\"" in text or "Freq[2]=4" in text)
    )


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", "detail": detail}
