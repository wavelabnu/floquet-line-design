from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tvtl_ads_builder_v2.validation.gain_comparison import (
    assert_no_deprecated_ads_elements,
    audit_ads_element_text,
    run_gain_comparison,
)
from tvtl_ads_builder_v2.validation.direct_hpeesofsim import HpeesofsimRunResult
from tvtl_ads_builder_v2.validation.validate_amplifier_hb import (
    STATUS_ANALYTICAL_OK,
    STATUS_COMPARISON_UNAVAILABLE,
    STATUS_HB_NO_CONVERGENCE,
    STATUS_HB_SKIPPED,
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


def test_ads_element_audit_fails_on_deprecated_elements():
    with pytest.raises(ValueError):
        assert_no_deprecated_ads_elements("L:LVIA_C1  0 N1 L=0.2 nH")
    with pytest.raises(ValueError):
        assert_no_deprecated_ads_elements("MLOC:OPEN1 N1 Subst=\"sub\"")
    audit = audit_ads_element_text("VIAGND:VIAGND_C1 N1\nMLEF:MLEF_C1 N2")
    assert audit["passed"] is True
    assert audit["VIAGND_present"] is True
    assert audit["MLEF_present"] is True


def test_gain_comparison_reports_missing_numerical_metrics_honestly(tmp_path):
    summary_path = _write_summary(tmp_path)
    report = run_gain_comparison(summary_path, tmp_path / "gain_compare")

    out = tmp_path / "gain_compare"
    assert (out / "analytical_gain.json").is_file()
    assert (out / "numerical_gain_direct.json").is_file()
    assert (out / "numerical_gain_oa.json").is_file()
    assert (out / "gain_comparison_report.json").is_file()
    assert (out / "gain_comparison_report.md").is_file()
    assert report["candidate"] == "n2_double_short"
    assert report["analytical"]["amplifier_candidate"]["coherent_gain_db"] == 10.6
    assert report["analytical"]["status"] == STATUS_ANALYTICAL_OK
    assert report["numerical_direct"]["status"] == STATUS_HB_SKIPPED
    assert report["numerical_direct"]["scope"] == "amplifier_candidate"
    assert report["numerical_direct"]["metrics"]["voltage_gain_db"] is None
    assert report["numerical_oa"]["status"] == STATUS_HB_SKIPPED
    assert "require --simulate" in " ".join(report["numerical_direct"]["metrics"]["notes"])
    assert "visible_summary" in report
    assert report["visible_summary"]["rows"][0]["metric"] == "coherent_gain_db"
    assert report["minimal_validation"]["scope"] == "minimal_self_pumped_reference"
    assert report["minimal_validation"]["status"] in {"not_run", "validated"}
    assert report["amplifier_analytical"]["scope"] == "amplifier_candidate"
    assert report["amplifier_analytical"]["coherent_gain_db"] == 10.6
    assert (
        report["amplifier_analytical"]["coherent_gain_normalization"]
        == "Stage-4 coherent_peak normalized to pump-off signal reference"
    )
    assert report["amplifier_numerical"]["scope"] == "amplifier_candidate"
    assert report["amplifier_numerical"]["status"] == "unavailable_not_sim_ready"
    assert report["comparison"]["analytical_vs_hb"]["status"] == STATUS_COMPARISON_UNAVAILABLE


def test_gain_comparison_records_hb_no_convergence_without_losing_analytical_gain(
    tmp_path, monkeypatch
):
    def fake_run(_spec, output_dir, **_kwargs):
        return HpeesofsimRunResult(
            status="error",
            netlist_path=str(Path(output_dir) / "amplifier_direct.ckt"),
            output_dir=str(output_dir),
            dataset_files=[],
            output_files=["amplifier_direct.out"],
            error_message="Error detected by hpeesofsim during HB analysis `HB1'. No convergence.",
        )

    import tvtl_ads_builder_v2.validation.gain_comparison as gain_comparison

    monkeypatch.setattr(gain_comparison, "run_direct_amplifier_hpeesofsim", fake_run)
    summary_path = _write_summary(tmp_path)
    report = run_gain_comparison(summary_path, tmp_path / "gain_compare", simulate=True)

    assert report["amplifier_analytical"]["status"] == STATUS_ANALYTICAL_OK
    assert report["amplifier_analytical"]["coherent_gain_db"] == 10.6
    assert report["numerical_direct"]["status"] == STATUS_HB_NO_CONVERGENCE
    assert report["numerical_direct"]["metrics"]["coherent_gain_db"] is None
    comparison = report["comparison"]["analytical_vs_hb"]
    assert comparison["direct_error_db"] is None
    assert comparison["status"] == STATUS_COMPARISON_UNAVAILABLE
    report_md = (tmp_path / "gain_compare" / "gain_comparison_report.md").read_text(
        encoding="utf-8"
    )
    assert "Analytical and HB models are compared, not expected to be exactly equal" in report_md
    assert "hb_no_convergence" in report_md


def test_gain_comparison_markdown_includes_equations(tmp_path):
    summary_path = _write_summary(tmp_path)
    run_gain_comparison(summary_path, tmp_path / "gain_compare")
    report_md = (tmp_path / "gain_compare" / "gain_comparison_report.md").read_text(
        encoding="utf-8"
    )

    assert "## Gain equations used" in report_md
    assert "## What gain is currently validated?" in report_md
    assert report_md.index("## Gain equations used") < report_md.index(
        "## What gain is currently validated?"
    )
    assert "Gain definitions used in this report" in report_md
    assert "G_V(fs) = |V_out(fs)| / |V_in(fs)|" in report_md
    assert "G_sum = (sum_{k in K_out} P_out(k fs)) / P_in(fs)" in report_md
    assert "Stage-4 coherent_peak normalized to pump-off signal reference" in report_md
    assert "| metric | analytical | direct HB | ADS-OA HB | status |" in report_md


def test_compare_gain_cli_prints_visible_summary_table(tmp_path):
    summary_path = _write_summary(tmp_path)
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tvtl_ads_builder_v2",
            "compare-gain",
            "--input",
            str(summary_path),
            "--out",
            str(tmp_path / "gain_compare_cli"),
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert proc.returncode == 0
    assert "Gain comparison summary" in proc.stdout
    assert "Minimal self-pumped validation" in proc.stdout
    assert "Amplifier analytical prediction" in proc.stdout
    assert "Direct hpeesofsim amplifier" in proc.stdout
    assert "coherent_gain_db" in proc.stdout
    assert "Reports:" in proc.stdout
