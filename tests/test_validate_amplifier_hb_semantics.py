from __future__ import annotations

import json
from pathlib import Path

from tvtl_ads_builder_v2 import cli
from tvtl_ads_builder_v2.validation.direct_hpeesofsim import HpeesofsimRunResult
from tvtl_ads_builder_v2.validation.validate_amplifier_hb import (
    STATUS_COMPARISON_UNAVAILABLE,
    STATUS_HB_NO_CONVERGENCE,
    run_validate_amplifier_hb,
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


def test_validate_amplifier_hb_records_no_convergence_as_report_status(tmp_path, monkeypatch):
    import tvtl_ads_builder_v2.validation.validate_amplifier_hb as validate

    def fake_export(_summary_path, out_dir, **_kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        netlist = out / "amplifier_oa.ckt"
        netlist.write_text("Port:TERM_LOAD N_LOAD 0 Num=3 Z=50 Ohm\n", encoding="utf-8")
        return {
            "project_dir": str(out),
            "generated_netlist_path": str(netlist),
        }

    def fake_compare(_direct, _oa, _spec, out_path):
        report = {"status": "passed", "match": True, "failures": [], "warnings": []}
        Path(out_path, "netlist_comparison.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        Path(out_path, "netlist_comparison.md").write_text("passed\n", encoding="utf-8")
        return report

    def fake_run(_netlist, netlist_path, output_dir):
        return HpeesofsimRunResult(
            status="error",
            netlist_path=str(netlist_path),
            output_dir=str(output_dir),
            dataset_files=[],
            output_files=[],
            error_message="Error detected by hpeesofsim during HB analysis `HB1'. No convergence.",
        )

    monkeypatch.setattr(validate, "export_amplifier_oa", fake_export)
    monkeypatch.setattr(validate, "write_amplifier_netlist_comparison", fake_compare)
    monkeypatch.setattr(validate, "_run_netlist_text", fake_run)

    report = run_validate_amplifier_hb(
        _write_summary(tmp_path),
        tmp_path / "direct_vs_oa",
        simulate=True,
    )

    gain = report["gain_comparison_report"]
    assert report["sim_ready"] is False
    assert gain["numerical_availability"]["direct_hb_status"] == STATUS_HB_NO_CONVERGENCE
    assert gain["numerical_availability"]["ads_oa_hb_status"] == STATUS_HB_NO_CONVERGENCE
    assert gain["direct_hpeesofsim"]["metrics"]["coherent_gain_db"] is None
    assert gain["comparison"]["analytical_vs_hb"]["status"] == STATUS_COMPARISON_UNAVAILABLE
    report_md = Path(report["paths"]["gain_comparison_md"]).read_text(encoding="utf-8")
    assert "Analytical and HB models are compared, not expected to be exactly equal" in report_md
    assert "hb_no_convergence" in report_md


def test_validate_amplifier_hb_cli_default_allows_no_convergence(monkeypatch, tmp_path):
    def fake_report(*_args, **_kwargs):
        return {
            "candidate": "n2_double_short",
            "direct_deck": {"path": "direct.ckt"},
            "ads_oa": {"netlist_path": "oa.ckt"},
            "netlist_comparison": {"status": "passed", "match": True},
            "simulation": {
                "direct_hpeesofsim": {"status": "error"},
                "ads_oa_hpeesofsim": {"status": "error"},
            },
            "gain_comparison_report": {
                "numerical_availability": {
                    "direct_hb_status": STATUS_HB_NO_CONVERGENCE,
                    "ads_oa_hb_status": STATUS_HB_NO_CONVERGENCE,
                }
            },
            "sim_ready": False,
            "paths": {"gain_comparison_json": "gain.json"},
        }

    monkeypatch.setattr(cli, "run_validate_amplifier_hb", fake_report)

    args = [
        "validate-amplifier-hb",
        "--input",
        str(tmp_path / "summary.json"),
        "--out",
        str(tmp_path / "out"),
        "--simulate",
    ]
    assert cli.main(args) == 0
    assert cli.main([*args, "--strict-convergence"]) == 2
