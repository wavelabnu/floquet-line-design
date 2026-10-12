from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from tvtl_ads_builder_v2.validation import pump_power_sweep
from tvtl_ads_builder_v2.validation.direct_hpeesofsim import HpeesofsimRunResult
from tvtl_ads_builder_v2.validation.pump_power_sweep import run_pump_power_sweep
from tvtl_ads_builder_v2.validation.validate_amplifier_hb import STATUS_HB_NO_CONVERGENCE


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
            "a1_score": 1.0,
            "metrics_at_drive": {"coherent_peak": 10.6},
            "main_tl": {
                "z_tl_ohm": 61.3,
                "width_m": 2.49e-3,
                "width_mm": 2.49,
                "bloch_z_ohm": {"real": 50.0, "imag": 0.0},
                "match_residual_ohm": 0.0,
                "realizable": True,
                "matched": True,
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


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_sweep_pump_gain_help_exits_successfully():
    proc = subprocess.run(
        [sys.executable, "-m", "tvtl_ads_builder_v2", "sweep-pump-gain", "--help"],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
    )

    assert proc.returncode == 0
    assert "--pump-start-dbm" in proc.stdout
    assert "--metric" in proc.stdout
    assert "--plot-y-min" in proc.stdout
    assert "--clip-plot-db" in proc.stdout
    assert "--invalid-gain-floor-db" in proc.stdout


def test_pump_power_sweep_writes_reports_plot_and_honest_numerics(
    tmp_path, monkeypatch
):
    class FakeAnalyticalReport:
        def __init__(self, pump_dbm: float) -> None:
            gain = -312.6096 if pump_dbm == -10.0 else 34.0 + pump_dbm
            self.analytical = {pump_dbm: {"coherent_peak": gain}}

    def fake_analytical_characterization(_design, *, pumps_dBm, **_kwargs):
        return FakeAnalyticalReport(float(tuple(pumps_dBm)[0]))

    monkeypatch.setattr(
        pump_power_sweep,
        "analytical_characterization",
        fake_analytical_characterization,
    )

    def fake_run_direct(_spec, output_dir, **_kwargs):
        return HpeesofsimRunResult(
            status="error",
            netlist_path=str(Path(output_dir) / "amplifier_direct.ckt"),
            output_dir=str(output_dir),
            dataset_files=[],
            output_files=[],
            error_message="HB1: No convergence",
        )

    monkeypatch.setattr(
        pump_power_sweep,
        "run_direct_amplifier_hpeesofsim",
        fake_run_direct,
    )
    summary_path = _write_summary(tmp_path)
    out = tmp_path / "pump_sweep"
    report = run_pump_power_sweep(
        summary_path,
        out,
        pump_start_dbm=-10,
        pump_stop_dbm=-6,
        pump_step_dbm=2,
        simulate=True,
    )

    csv_path = out / "pump_sweep.csv"
    json_path = out / "pump_sweep.json"
    md_path = out / "pump_power_sweep_report.md"
    png_path = out / "gain_vs_pump_power.png"
    assert csv_path.is_file()
    assert json_path.is_file()
    assert md_path.is_file()
    assert png_path.is_file()
    assert png_path.stat().st_size > 0
    assert report["numerical_sweep_status"] == STATUS_HB_NO_CONVERGENCE
    assert report["architecture"] == "self_pumped_same_line"
    assert report["best_gain_point"] is not None
    assert report["best_gain_point"]["pump_dbm"] == -6.0

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows
    assert rows[0]["pump_dbm"] == "-10.0"
    assert any(row["coherent_gain_db_analytical"] for row in rows)
    assert "coherent_gain_db_analytical_raw" in rows[0]
    assert "coherent_gain_db_analytical_plot" in rows[0]
    assert "gain_point_status" in rows[0]
    assert "direct_hb_status" in rows[0]
    assert "oa_hb_status" in rows[0]
    assert float(rows[0]["coherent_gain_db_analytical_raw"]) == -312.6096
    assert float(rows[0]["coherent_gain_db_analytical_plot"]) == -100.0
    assert rows[0]["gain_point_status"] == "null_or_singular"
    assert all(row["coherent_gain_db_direct_hb"] == "" for row in rows)
    assert all(row["coherent_gain_db_oa_hb"] == "" for row in rows)
    assert all(row["direct_hb_status"] == STATUS_HB_NO_CONVERGENCE for row in rows)
    assert all("direct_hb_status=hb_no_convergence" in row["notes"] for row in rows)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["coherent_gain_db_direct_hb"] is None
    assert payload["rows"][0]["coherent_gain_db_oa_hb"] is None
    assert payload["rows"][0]["direct_hb_status"] == STATUS_HB_NO_CONVERGENCE
    assert payload["rows"][0]["coherent_gain_db_analytical_raw"] == -312.6096
    assert payload["rows"][0]["coherent_gain_db_analytical_plot"] == -100.0
    assert payload["rows"][0]["gain_point_status"] == "null_or_singular"
    assert payload["plot"]["analytical_curve"] is True
    assert payload["plot"]["numerical_curve"] is False
    assert payload["plot"]["title_mode"] == "analytical-only; HB numerical points unavailable/no convergence"
    assert payload["plot"]["plot_y_min"] == -20.0
    assert payload["plot"]["plot_y_max"] == 50.0
    assert payload["plot"]["invalid_gain_floor_db"] == -100.0
    assert payload["plot"]["flagged_points"] == [
        {
            "pump_dbm": -10.0,
            "plot_gain_db": -100.0,
            "raw_gain_db": -312.6096,
            "status": "null_or_singular",
        }
    ]

    report_md = md_path.read_text(encoding="utf-8")
    assert "## Gain equations used" in report_md
    assert "## Best gain point" in report_md
    assert "## Flagged/null points" in report_md
    assert "## Interpretation" in report_md
    assert "Analytical and HB models are compared, not expected to be exactly equal" in report_md
    assert "hb_no_convergence" in report_md
    assert "Gain definitions used in this report" in report_md
    assert "Pump power table" in report_md
    assert "G_T(fs) = P_L(fs) / P_avs(fs)" in report_md
    assert "gain_vs_pump_power.png" in report_md


def test_sweep_pump_gain_cli_prints_best_and_flagged_count(tmp_path):
    summary_path = _write_summary(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tvtl_ads_builder_v2",
            "sweep-pump-gain",
            "--input",
            str(summary_path),
            "--out",
            str(tmp_path / "pump_sweep_cli"),
            "--pump-start-dbm",
            "-4",
            "--pump-stop-dbm",
            "0",
            "--pump-step-dbm",
            "2",
        ],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
    )

    assert proc.returncode == 0
    assert "metric: analytical coherent gain" in proc.stdout
    assert "best point:" in proc.stdout
    assert "flagged points:" in proc.stdout
    assert "plot:" in proc.stdout
    assert "numerical sweep:" in proc.stdout
