"""Amplifier direct-vs-OA netlist comparison tests."""

from __future__ import annotations

import json
from pathlib import Path

from tvtl_ads_builder_v2.amplifier_oa_export import build_amplifier_schematic_plan
from tvtl_ads_builder_v2.amplifier_spec import load_amplifier_circuit_spec
from tvtl_ads_builder_v2.gui_export import load_optimizer_candidates
from tvtl_ads_builder_v2.validation.amplifier_netlist_compare import (
    compare_amplifier_netlists,
)
from tvtl_ads_builder_v2.validation.direct_amplifier_hpeesofsim import (
    direct_amplifier_netlist,
)


def _summary() -> dict:
    return {
        "workflow": "optimize-amplifier",
        "target_frequency_GHz": 2.0,
        "target_frequency_Hz": 2.0e9,
        "config": {"n_cells": 15, "drive_dBm": -3.0},
        "varactor": {
            "CJ0_f": 3.0e-13,
            "VJ_v": 3.5,
            "M": 1.7,
            "CP_f": 0.0,
            "L_pkg_h": 3.0e-10,
            "R_series_ohm": 0.5,
        },
        "best_candidate": {
            "rank": 1,
            "topology": "n2_double_short",
            "n_segments": 2,
            "via": "double",
            "termination": "short",
            "metrics_at_drive": {"coherent_peak": 10.6},
            "main_tl": {
                "z_tl_ohm": 61.3,
                "width_m": 2.49e-3,
                "width_mm": 2.49,
            },
            "stub_parameters": [
                {"segment": 1, "width_m": 2.0e-3, "length_m": 7.0e-3},
                {"segment": 2, "width_m": 4.0e-3, "length_m": 14.0e-3},
            ],
            "input_match": None,
            "output_match": None,
        },
    }


def _write_summary(tmp_path: Path) -> Path:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    (tmp_path / "candidates.json").write_text(
        json.dumps([_summary()["best_candidate"]]),
        encoding="utf-8",
    )
    return summary_path


def _fake_oa_netlist(n_cells: int = 15, extra: str = "") -> str:
    uc_lines = "\n".join(
        f"UC_N2_DOUBLE_SHORT:UC{index:02d} N{index} N{index + 1}"
        for index in range(1, n_cells + 1)
    )
    return "\n".join(
        [
            "Port:PORT_SIG RF_MAIN 0 Num=1 Freq[1]=fs",
            "Port:PORT_PUMP RF_MAIN 0 Num=2 Freq[1]=2*fs",
            "HB:HB1 Freq[1]=fs Freq[2]=2*fs Order[1]=5 Order[2]=5 MaxOrder=6",
            uc_lines,
            "MTEE_ADS:TEE_C1 A B C",
            "MLIN2:LINE_LEFT_C1 A B",
            "SDD:VARACTOR_C1 A B C C",
            "VIAGND:VIAGND_C1 C",
            extra,
        ]
    )


def test_shared_amplifier_spec_matches_oa_plan(tmp_path):
    summary_path = _write_summary(tmp_path)
    summary, candidates = load_optimizer_candidates(summary_path)
    spec = load_amplifier_circuit_spec(summary_path)
    plan = build_amplifier_schematic_plan(summary, candidates[0], selected_count=1)

    assert spec.candidate_name == plan["candidate_summary"]["candidate_name"]
    assert spec.n_cells == plan["candidate_summary"]["n_cells"]
    assert spec.topology_family == plan["unit_cell_implementation"]["topology"]["key"]
    assert spec.main_width_m == plan["unit_cell_implementation"]["main_width_m"]
    assert spec.via_model == plan["unit_cell_implementation"]["termination_models"]["via"]
    assert spec.open_end_model == plan["unit_cell_implementation"]["termination_models"]["open_end"]


def test_netlist_compare_detects_mismatched_cell_counts(tmp_path):
    summary_path = _write_summary(tmp_path)
    spec = load_amplifier_circuit_spec(summary_path)
    direct = direct_amplifier_netlist(spec)

    report = compare_amplifier_netlists(direct, _fake_oa_netlist(n_cells=14), spec)

    assert report["status"] == "failed"
    assert any(check["name"] == "unit_cells.ads_oa_instance_count" for check in report["failures"])


def test_netlist_compare_detects_deprecated_elements(tmp_path):
    summary_path = _write_summary(tmp_path)
    spec = load_amplifier_circuit_spec(summary_path)
    direct = direct_amplifier_netlist(spec)

    report = compare_amplifier_netlists(direct, _fake_oa_netlist(extra="MLOC:BAD X"), spec)

    assert report["status"] == "failed"
    assert any(check["name"] == "ads_oa.no_deprecated_elements" for check in report["failures"])
