"""Public ADS GUI export CLI tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src) if not existing else str(src) + os.pathsep + existing
    return env


def _minimal_summary(path: Path) -> Path:
    summary = {
        "workflow": "optimize-amplifier",
        "target_frequency_GHz": 2.0,
        "target_frequency_Hz": 2.0e9,
        "config": {"n_cells": 1, "drive_dBm": -3.0},
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
            "topology": "n1_none_open",
            "n_segments": 1,
            "via": "none",
            "termination": "open",
            "stub_parameters": [
                {
                    "segment": 1,
                    "width_m": 1.0e-3,
                    "length_m": 5.0e-3,
                }
            ],
            "main_tl": {"width_m": 2.0e-3},
            "input_match": None,
            "output_match": None,
        },
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_tvtl_ads_builder_module_help_exits_successfully():
    proc = subprocess.run(
        [sys.executable, "-m", "tvtl_ads_builder_v2", "--help"],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
    )
    assert proc.returncode == 0
    assert "export-gui-project" in proc.stdout


def test_export_gui_project_help_exits_successfully():
    proc = subprocess.run(
        [sys.executable, "-m", "tvtl_ads_builder_v2", "export-gui-project", "--help"],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
    )
    assert proc.returncode == 0
    assert "--candidate-index" in proc.stdout
    assert "--backend" in proc.stdout


def test_top_level_ads_export_help_exits_successfully(capsys):
    from tvtl.optimization.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["ads-export", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "ads-export" in out
    assert "--input" in out


def test_export_missing_input_is_controlled(tmp_path, capsys):
    from tvtl_ads_builder_v2.cli import main

    rc = main(
        [
            "export-gui-project",
            "--input",
            str(tmp_path / "missing_summary.json"),
            "--out",
            str(tmp_path / "ads_project"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "input summary not found" in err


def test_export_scaffold_from_minimal_summary_without_ads_or_env(tmp_path, monkeypatch):
    from tvtl_ads_builder_v2.cli import main

    monkeypatch.delenv("ADS_PYTHON", raising=False)
    monkeypatch.delenv("ADS_INSTALL_DIR", raising=False)
    monkeypatch.delenv("HPEESOF_DIR", raising=False)

    summary_path = _minimal_summary(tmp_path / "summary.json")
    out_dir = tmp_path / "ads_project"
    rc = main(
        [
            "export-gui-project",
            "--input",
            str(summary_path),
            "--out",
            str(out_dir),
            "--backend",
            "scaffold",
        ]
    )

    assert rc == 0
    assert (out_dir / "device_spec.device_spec.json").is_file()
    assert (out_dir / "hpeesofsim.ckt").is_file()
    assert (out_dir / "ads_native.net").is_file()
    assert (out_dir / "build_oa_workspace.py").is_file()
    assert (out_dir / "build_oa_workspace.bat").is_file()
    assert (out_dir / "OPEN_IN_ADS.txt").is_file()
    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend_materialized"] == "scaffold"
    assert manifest["env_required_for_optimizer"] is False
    assert manifest["ads_python_required_for_oa"] is True
    assert not (tmp_path / ".env").exists()


def test_export_real_oa_workspace_when_ads_python_available(tmp_path):
    pytest.importorskip("keysight.ads.de")
    from tvtl_ads_builder_v2.cli import main

    summary_path = _minimal_summary(tmp_path / "summary.json")
    out_dir = tmp_path / "ads_project"
    rc = main(
        [
            "export-gui-project",
            "--input",
            str(summary_path),
            "--out",
            str(out_dir),
            "--backend",
            "oa",
        ]
    )

    assert rc == 0
    assert (out_dir / "oa_workspace").exists()
    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend_materialized"] == "oa"
