from __future__ import annotations

import inspect

from tvtl_ads_builder_v2 import ads_oa_writer
from tvtl_ads_builder_v2.ads_oa_writer import (
    REQUIRED_COMPONENTS,
    self_pumped_schematic_contract,
)


def test_required_component_refs_are_explicit_lcvs() -> None:
    assert REQUIRED_COMPONENTS["signal_source"].as_string() == "ads_sources:P_1Tone:symbol"
    assert REQUIRED_COMPONENTS["pump_source"].as_string() == "ads_sources:P_1Tone:symbol"
    assert REQUIRED_COMPONENTS["harmonic_balance"].as_string() == "ads_simulation:HarmonicBalance:symbol"
    assert REQUIRED_COMPONENTS["main_line"].as_string() == "ads_tlines:TLIN:symbol"
    assert REQUIRED_COMPONENTS["ground"].as_string() == "ads_rflib:GROUND:symbol"


def test_self_pumped_contract_has_one_rf_net_and_no_lane() -> None:
    contract = self_pumped_schematic_contract()

    assert contract["architecture"] == "self-pumped"
    assert contract["rf_net"] == "RF_MAIN"
    assert contract["source_instances"] == ["PORT_SIG", "PORT_PUMP"]
    assert contract["signal_frequency"] == "fs"
    assert contract["pump_frequency"] == "2*fs"
    assert set(contract["forbidden_separate_lanes"]) == {"PUMP", "CONTROL", "BIAS"}


def test_oa_writer_uses_snap_point_wiring_and_committed_transaction() -> None:
    source = inspect.getsource(ads_oa_writer)

    assert "de.cellview_exists" in source
    assert "design.add_instance" in source
    assert "inst_terms" in source
    assert "inst_pins" in source
    assert "snap_point" in source
    assert "design.add_wire" in source
    assert "RF_MAIN" in source
    assert "db_uu.Transaction" in source
    assert "transaction.commit()" in source
    assert "design.save_design()" in source
    assert "design.generate_netlist()" in source
