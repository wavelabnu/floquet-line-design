from __future__ import annotations

from pathlib import Path

import pytest

from tvtl_ads_builder_v2.circuit_spec import SelfPumpedCircuitSpec
from tvtl_ads_builder_v2.validation.compare_self_pumped import run_self_pumped_validation


def test_ads_oa_self_pumped_full_validation(ads_tmp_path: Path) -> None:
    pytest.importorskip("keysight.ads.de")
    pytest.importorskip("keysight.edatoolbox")

    report = run_self_pumped_validation(
        SelfPumpedCircuitSpec(),
        ads_tmp_path / "validation_self_pumped_ads",
        simulate=True,
        draw_oa=True,
        oa_workspace_dir=ads_tmp_path / "validation_self_pumped_oa",
    )

    assert report["analytical"]["expected_matched"] is True
    assert report["simulation"]["direct_hpeesofsim"]["status"] == "passed"
    assert report["simulation"]["direct_hpeesofsim"]["dataset_files"]
    assert report["simulation"]["ads_oa_hpeesofsim"]["status"] == "passed"
    assert report["simulation"]["ads_oa_hpeesofsim"]["dataset_status"] == "passed"
    assert report["topology_comparison"]["status"] == "passed"
    assert Path(report["paths"]["oa_netlist"]).is_file()
