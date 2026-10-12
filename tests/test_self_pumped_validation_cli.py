from __future__ import annotations

import json

from tvtl_ads_builder_v2.cli import main


def test_validate_self_pumped_cli_writes_topology_report_without_ads(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HPEESOF_DIR", raising=False)
    monkeypatch.delenv("ADS_INSTALL_DIR", raising=False)
    monkeypatch.delenv("ADS_PYTHON", raising=False)

    out_dir = tmp_path / "validation_self_pumped_no_ads"
    rc = main(
        [
            "validate-self-pumped",
            "--freq-ghz",
            "2.0",
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 0
    report_path = out_dir / "comparison_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["analytical"]["expected_matched"] is True
    assert report["simulation"]["direct_hpeesofsim"]["status"] == "not_run"
    assert report["topology_comparison"]["status"] == "passed"
