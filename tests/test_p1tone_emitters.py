"""Emitter tests for the ``P_1Tone`` power source.

The hpeesofsim ``P_1Tone:`` emitter is tested directly here (ADS-free).
The OA emitter is exercised against the stub ``keysight.ads.de`` in
``test_oa_canonical.py::test_level7_oa_emitters_run``.
"""

from __future__ import annotations

import math

from tvtl_ads_builder_v2.device_spec import P1ToneSource
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.netlist.elements.p1tone import emit_p1tone, format_p1tone_power
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_tvtl_unit_cell_two_tone_hb,
    make_two_tone_p1tone_sources,
)


def _src(**overrides) -> P1ToneSource:
    base = dict(
        name="SIGNAL_SRC",
        port_node="n_in",
        num=1,
        freq=3.1e9,
        power_dBm=-30.0,
        z_ohm=50.0,
    )
    base.update(overrides)
    return P1ToneSource(**base)


def test_emit_p1tone_matches_v1_confirmed_syntax():
    """The emitted line matches v1's archived P_1Tone: form."""

    line = emit_p1tone(_src())
    assert line == (
        "P_1Tone:SIGNAL_SRC n_in 0 Num=1 Z=50 Ohm "
        "Freq=3.1 GHz P=polar(dbmtow(-30),0) Noise=no"
    )


def test_emit_p1tone_negative_terminal_is_ground():
    """The source's return terminal is the literal ground 0."""

    line = emit_p1tone(_src(port_node="n_input"))
    # tokens: P_1Tone:NAME <pos> <neg> ...
    tokens = line.split()
    assert tokens[1] == "n_input"
    assert tokens[2] == "0"


def test_emit_p1tone_power_uses_dbmtow_wrapper():
    """Power is emitted inside ADS's polar(dbmtow(<dBm>),0) wrapper."""

    assert "P=polar(dbmtow(5),0)" in emit_p1tone(_src(power_dBm=5.0))
    assert "P=polar(dbmtow(-12.5),0)" in emit_p1tone(_src(power_dBm=-12.5))


def test_emit_p1tone_pump_tone():
    line = emit_p1tone(_src(name="PUMP_SRC", freq=6.2e9, power_dBm=10.0))
    assert line.startswith("P_1Tone:PUMP_SRC ")
    assert "Freq=6.2 GHz" in line
    assert "P=polar(dbmtow(10),0)" in line


def test_format_p1tone_power_phase_zero_is_byte_identical():
    """phase_rad defaults to 0.0 -> the historical fixed-zero ',0' form."""

    assert format_p1tone_power(-30.0) == "polar(dbmtow(-30),0)"
    assert format_p1tone_power(5.0, 0.0) == "polar(dbmtow(5),0)"
    assert "P=polar(dbmtow(-30),0)" in emit_p1tone(_src())  # phase_rad default 0.0


def test_emit_p1tone_bakes_nonzero_phase_in_degrees():
    """A non-zero source phase_rad reaches the polar() angle in DEGREES.

    This is what lets an OA workspace bake the degenerate-PSA optimal pump phase
    (e.g. 120 deg for the A.3.1 anchor) into the P_1Tone source instead of
    silently dropping it."""

    assert format_p1tone_power(1.6697, math.radians(120.0)) == "polar(dbmtow(1.6697),120)"
    line = emit_p1tone(_src(name="PUMP_SRC", freq=6.2e9, power_dBm=1.6697,
                            phase_rad=math.radians(120.0)))
    assert "P=polar(dbmtow(1.6697),120)" in line


def test_hpeesofsim_netlist_carries_source_lines():
    """The full deck carries both P_1Tone: sources in a sources section.

    Signal source = Num 1, pump source = Num 2; output port = Num 3.
    Each P_1Tone source carries its own ADS port number so hpeesofsim
    does not abort with "Duplicate Port or Term number" end-to-end.
    """

    netlist = emit_hpeesofsim_netlist(
        make_two_tone_p1tone_sources(label="t")
    )
    assert "; --- sources ---" in netlist
    assert "P_1Tone:SIGNAL_SRC n_in 0 Num=1" in netlist
    assert "P_1Tone:PUMP_SRC n_in 0 Num=2" in netlist
    # the S-parameter directive is still present alongside the sources.
    assert "S_Param:SP1" in netlist


def test_emit_is_deterministic():
    spec = make_two_tone_p1tone_sources(label="t")
    assert emit_hpeesofsim_netlist(spec) == emit_hpeesofsim_netlist(spec)


def test_resolver_substitutes_named_source_references():
    """Named freq / power references resolve to their VAR literals.

    The Level-7 source canonical declares ``f_signal = 3 GHz``,
    ``f_pump = 6 GHz``, ``signal_power_dBm = -30``, ``pump_power_dBm = 5``
    and references all four by name; the emitted deck must carry the
    substituted literals, not the variable names.
    """

    netlist = emit_hpeesofsim_netlist(
        make_two_tone_p1tone_sources(label="t")
    )
    assert "P_1Tone:SIGNAL_SRC n_in 0 Num=1 Z=50 Ohm Freq=3 GHz " \
           "P=polar(dbmtow(-30),0) Noise=no" in netlist
    assert "P_1Tone:PUMP_SRC n_in 0 Num=2 Z=50 Ohm Freq=6 GHz " \
           "P=polar(dbmtow(5),0) Noise=no" in netlist
    # the variable names never reach the netlist.
    assert "f_signal" not in netlist
    assert "pump_power_dBm" not in netlist


def test_integrated_cell_shares_one_var_across_source_and_hb():
    """One VAR declaration drives both the HB controller and the source.

    In the integrated Level-7 cell ``f_signal`` drives both
    ``HB:`` ``Freq[1]`` and ``SIGNAL_SRC``'s ``Freq``; ``f_pump`` drives
    ``Freq[2]`` and ``PUMP_SRC``'s ``Freq`` -- both resolve to the same
    literal from one declaration.
    """

    netlist = emit_hpeesofsim_netlist(
        make_tvtl_unit_cell_two_tone_hb(label="t")
    )
    assert "HB:HB1 Freq[1]=3 GHz Freq[2]=6 GHz" in netlist
    assert "P_1Tone:SIGNAL_SRC n_input 0 Num=1 Z=50 Ohm Freq=3 GHz" in netlist
    assert "P_1Tone:PUMP_SRC n_input 0 Num=2 Z=50 Ohm Freq=6 GHz" in netlist


