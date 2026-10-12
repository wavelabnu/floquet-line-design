"""Emitter tests for the single-tone HB controller.

The hpeesofsim ``HB:`` emitter is tested directly here (ADS-free). The
OA emitter is exercised against the stub ``keysight.ads.de`` in
``test_oa_canonical.py::test_level6_hb_oa_emitter_runs``, where the OA
stub machinery lives.
"""

from __future__ import annotations

from tvtl_ads_builder_v2.device_spec import HBSingleToneSpec
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.netlist.elements.hb_controller import (
    emit_hb_single_tone,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    make_single_tone_hb_controller,
)


def test_emit_hb_single_tone_syntax():
    line = emit_hb_single_tone(HBSingleToneSpec(3.0e9, 5, 5))
    assert line == "HB:HB1 Freq[1]=3 GHz Order[1]=5 MaxOrder=5 StatusLevel=2"


def test_emit_hb_single_tone_frequency_in_ghz():
    line = emit_hb_single_tone(HBSingleToneSpec(3.1e9, 7, 9))
    assert "Freq[1]=3.1 GHz" in line
    assert "Order[1]=7" in line
    assert "MaxOrder=9" in line


def test_hpeesofsim_netlist_carries_hb_directive():
    netlist = emit_hpeesofsim_netlist(make_single_tone_hb_controller(label="t"))
    assert (
        "HB:HB1 Freq[1]=3 GHz Order[1]=5 MaxOrder=5 StatusLevel=2" in netlist
    )
    # the S-parameter directive is still present alongside the HB line.
    assert "S_Param:SP1" in netlist


def test_hpeesofsim_netlist_emits_no_hb_when_absent():
    netlist = emit_hpeesofsim_netlist(LEVEL_1_CANONICAL_CIRCUITS[0])
    assert not any(
        line.startswith("HB:") for line in netlist.splitlines()
    )


def test_emit_is_deterministic():
    spec = make_single_tone_hb_controller(label="t")
    assert emit_hpeesofsim_netlist(spec) == emit_hpeesofsim_netlist(spec)


