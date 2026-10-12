from __future__ import annotations

from tvtl_ads_builder_v2.circuit_spec import SelfPumpedCircuitSpec
from tvtl_ads_builder_v2.validation.compare_self_pumped import compare_topology
from tvtl_ads_builder_v2.validation.direct_hpeesofsim import direct_self_pumped_netlist


def test_direct_netlist_has_both_sources_on_rf_main() -> None:
    netlist = direct_self_pumped_netlist(SelfPumpedCircuitSpec())

    assert "Port:PORT_SIG  RF_MAIN 0 Num=1 Z=100 Ohm" in netlist
    assert "P[1]=polar(dbmtow(Psig_dBm),0) Freq[1]=fs" in netlist
    assert "Port:PORT_PUMP  RF_MAIN 0 Num=2 Z=100 Ohm" in netlist
    assert "P[1]=polar(dbmtow(Ppump_dBm),0) Freq[1]=2*fs" in netlist


def test_direct_netlist_has_no_separate_pump_control_or_bias_lane() -> None:
    result = compare_topology(
        direct_self_pumped_netlist(SelfPumpedCircuitSpec()),
        None,
    )

    direct = result["checks"][0]["details"]
    assert direct["passed"] is True
    assert direct["forbidden_nets"] == []


def test_direct_netlist_hb_fundamental_is_fs_and_pump_is_second_harmonic() -> None:
    netlist = direct_self_pumped_netlist(SelfPumpedCircuitSpec())

    assert "HB:HB1 MaxOrder=4 Freq[1]=fs Order[1]=5" in netlist
    assert "Freq[1]=2*fs" in netlist
    assert "TLIN:TL_RF  RF_MAIN N_LOAD Z=50 Ohm E=90 F=fs" in netlist
