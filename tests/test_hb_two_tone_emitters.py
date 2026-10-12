"""Emitter tests for the two-tone HB controller.

The hpeesofsim ``HB:`` emitter is tested directly here (ADS-free). The
OA emitter is exercised against the stub ``keysight.ads.de`` in
``test_oa_canonical.py::test_level6_two_tone_hb_oa_emitter_runs``.
"""

from __future__ import annotations

from tvtl_ads_builder_v2.device_spec import HBTwoToneSpec
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.netlist.elements.hb_controller import emit_hb_two_tone
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_two_tone_hb_controller,
)


def test_emit_hb_two_tone_syntax():
    line = emit_hb_two_tone(HBTwoToneSpec(3.0e9, 5, 6.0e9, 5, 10))
    assert line == (
        "HB:HB1 Freq[1]=3 GHz Freq[2]=6 GHz "
        "Order[1]=5 Order[2]=5 MaxOrder=10 StatusLevel=2"
    )


def test_emit_hb_two_tone_distinct_tones():
    line = emit_hb_two_tone(HBTwoToneSpec(3.1e9, 7, 6.2e9, 4, 11))
    assert "Freq[1]=3.1 GHz" in line
    assert "Freq[2]=6.2 GHz" in line
    assert "Order[1]=7" in line
    assert "Order[2]=4" in line
    assert "MaxOrder=11" in line


def test_hpeesofsim_netlist_carries_two_tone_hb_directive():
    netlist = emit_hpeesofsim_netlist(make_two_tone_hb_controller(label="t"))
    assert (
        "HB:HB1 Freq[1]=3 GHz Freq[2]=6 GHz "
        "Order[1]=5 Order[2]=5 MaxOrder=10 StatusLevel=2" in netlist
    )
    # the S-parameter directive is still present alongside the HB line.
    assert "S_Param:SP1" in netlist


def test_emit_is_deterministic():
    spec = make_two_tone_hb_controller(label="t")
    assert emit_hpeesofsim_netlist(spec) == emit_hpeesofsim_netlist(spec)


