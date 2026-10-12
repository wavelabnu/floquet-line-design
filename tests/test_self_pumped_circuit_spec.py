from __future__ import annotations

import pytest

from tvtl_ads_builder_v2.circuit_spec import SelfPumpedCircuitSpec
from tvtl_ads_builder_v2.validation.analytical_self_pumped import evaluate_analytical


def test_two_100_ohm_sources_are_50_ohm_parallel() -> None:
    spec = SelfPumpedCircuitSpec(source_z_ohm=100.0)

    assert spec.source_z_parallel_ohm == pytest.approx(50.0)


def test_minimal_passive_rf_path_is_matched_at_fs() -> None:
    result = evaluate_analytical(SelfPumpedCircuitSpec())

    assert result["expected_rf_main_net"] == "RF_MAIN"
    assert result["expected_matched"] is True
    assert "nonlinear parametric gain" in result["not_compared"]
