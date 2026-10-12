from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from tvtl_ads_builder_v2.validation import hb_convergence_ladder
from tvtl_ads_builder_v2.validation.hb_convergence_ladder import (
    build_ladder_plan,
    parse_cell_list,
    parse_hb_order_list,
    parse_pump_dbm_list,
    run_hb_convergence_ladder,
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


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_hb_convergence_ladder_help_exits_successfully():
    proc = subprocess.run(
        [sys.executable, "-m", "tvtl_ads_builder_v2", "hb-convergence-ladder", "--help"],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
    )

    assert proc.returncode == 0
    assert "--cells" in proc.stdout
    assert "--pump-dbm" in proc.stdout
    assert "--hb-orders" in proc.stdout
    assert "--simulate" in proc.stdout


def test_hb_convergence_ladder_list_parsers():
    assert parse_cell_list("1,3,5,10,15") == [1, 3, 5, 10, 15]
    assert parse_pump_dbm_list("-30,-20,-10,0,3,6") == [-30.0, -20.0, -10.0, 0.0, 3.0, 6.0]
    assert parse_hb_order_list("1,3,5") == [1, 3, 5]


def test_hb_convergence_ladder_plan_contains_requested_combinations():
    plan = build_ladder_plan(cells=[1, 3], pump_dbm=[-30, -20], hb_orders=[1, 3])

    assert len(plan) == 8
    assert plan[0].to_dict() == {
        "deck_kind": "direct",
        "n_cells": 1,
        "pump_dbm": -30.0,
        "hb_order": 1,
    }
    assert plan[-1].to_dict() == {
        "deck_kind": "direct",
        "n_cells": 3,
        "pump_dbm": -20.0,
        "hb_order": 3,
    }


def test_hb_convergence_ladder_dry_run_writes_reports_and_policy_decks(tmp_path):
    summary_path = _write_summary(tmp_path)
    out = tmp_path / "hb_ladder"

    report = run_hb_convergence_ladder(
        summary_path,
        out,
        cells=[1, 3],
        pump_dbm=[-30, -20, -10],
        hb_orders=[1, 3],
        dry_run=True,
    )

    json_path = out / "hb_convergence_ladder.json"
    csv_path = out / "hb_convergence_ladder.csv"
    md_path = out / "hb_convergence_ladder.md"
    png_path = out / "hb_convergence_map.png"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert md_path.is_file()
    assert png_path.is_file()
    assert png_path.stat().st_size > 0
    assert (out / "decks").is_dir()
    assert (out / "datasets").is_dir()
    assert (out / "logs").is_dir()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["sweep"]["cells"] == [1, 3]
    assert payload["sweep"]["pump_dbm"] == [-30.0, -20.0, -10.0]
    assert payload["sweep"]["hb_orders"] == [1, 3]
    assert len(payload["results"]) == 12
    assert all(row["status"] == "skipped" for row in payload["results"])
    assert report["topology_audit"]["PORT_SIG_on_RF_MAIN"] is True
    assert report["topology_audit"]["PORT_PUMP_on_RF_MAIN"] is True
    assert report["topology_audit"]["separate_pump_control_bias_lane_present"] is False
    assert report["topology_audit"]["LVIA_present"] is False
    assert report["topology_audit"]["MLOC_present"] is False
    assert report["topology_audit"]["VIAGND_present"] is True

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 12
    assert rows[0]["deck_kind"] == "direct"
    assert rows[0]["dataset_exists"] == "False"

    one_cell_deck = (out / "decks" / "direct_cells001_pumpm30_hb1.ckt").read_text(
        encoding="utf-8"
    )
    assert "Port:PORT_SIG RF_MAIN 0" in one_cell_deck
    assert "Port:PORT_PUMP RF_MAIN 0" in one_cell_deck
    assert " PUMP " not in f" {one_cell_deck} "
    assert " CONTROL " not in f" {one_cell_deck} "
    assert " BIAS " not in f" {one_cell_deck} "
    assert "LVIA" not in one_cell_deck
    assert "MLOC:" not in one_cell_deck
    assert one_cell_deck.count("VIAGND:") == 2  # n2_double_short: two vias per cell since v1.0.1 (was 1)
    assert "Order[1]=1" in one_cell_deck
    assert "Order[2]=1" in one_cell_deck


def test_hb_convergence_ladder_records_failures_without_aborting(tmp_path, monkeypatch):
    def fake_run_hpeesofsim(netlist_path, **_kwargs):
        return {
            "status": "error",
            "dataset_path": None,
            "log_path": str(Path(netlist_path).with_name(Path(netlist_path).stem + "_stdout.txt")),
            "returncode": 1,
            "stdout_tail": "HB1: No convergence",
            "stderr_tail": "",
            "wall_time_s": 0.01,
            "error_message": "hpeesofsim returned non-zero",
        }

    monkeypatch.setattr(hb_convergence_ladder, "run_hpeesofsim", fake_run_hpeesofsim)
    summary_path = _write_summary(tmp_path)
    out = tmp_path / "hb_ladder_failed"

    report = run_hb_convergence_ladder(
        summary_path,
        out,
        cells=[1, 3],
        pump_dbm=[-30],
        hb_orders=[1],
        simulate=True,
    )

    assert len(report["results"]) == 2
    assert [row["status"] for row in report["results"]] == ["failed", "failed"]
    assert report["first_failure"]["n_cells"] == 1
    assert report["summaries"]["status_counts"]["failed"] == 2
    assert all("HB1: No convergence" in row["error_summary"] for row in report["results"])
