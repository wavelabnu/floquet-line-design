"""Unit tests for the OA schematic-netlist CONNECTIVITY check (base Python).

These run everywhere (no ADS): they exercise the parser and the checker on
captured good / broken netlists so the check itself is pinned. The companion
ADSPY-gated test ``tests/parity/test_oa_schematic_connectivity.py`` asserts
the live emitter produces a clean schematic netlist.

The captured "broken" source line is the exact historical break: both source
terminals merged onto ``n_input`` (``Port:SIGNAL_SRC n_input n_input``).
"""

from __future__ import annotations

from tvtl_ads_builder_v2.oa.connectivity_check import (
    check_connectivity,
    parse_oa_netlist,
    topology_floating_nets,
)

# A minimal but representative schematic netlist: substrate model line, a
# series MLIN, a shunt-to-ground L, the output Port, and the two sources.
_GOOD = """\
; Top Design: "x:y:schematic"
Options ResourceUsage=yes
model msub1 MSUB H=1.54 mm Er=3.38
MLIN2:MATCH_IN_SERIES  N__392 n_input Subst="msub1" W=3.5 mm L=3.9 mm
L:LVIA_C1  0 N__405 L=0.57 nH Noise=yes
MLIN2:LINE  n_output N__392 Subst="msub1" W=4.5 mm L=3.0 mm
Port:P_OUT  n_output 0 Num=3 Z=50 Ohm Noise=no
Port:SIGNAL_SRC  n_input 0 Num=1 Z=50 Ohm P[1]=polar(dbmtow(-30),0) Freq[1]=3.1 GHz
Port:PUMP_SRC  n_input 0 Num=2 Z=50 Ohm P[1]=polar(dbmtow(1.6),0) Freq[1]=6.2 GHz
"""

# The break: both sources' return terminal merged onto n_input (SHORTED).
_SHORTED = _GOOD.replace("n_input 0 Num=1", "n_input n_input Num=1").replace(
    "n_input 0 Num=2", "n_input n_input Num=2")

_SRC = {"SIGNAL_SRC": 1, "PUMP_SRC": 2}
_NAMED = {"n_input", "n_output"}


def test_parse_extracts_nets_before_first_param() -> None:
    insts = parse_oa_netlist(_GOOD)
    by_name = {i.name: i for i in insts}
    assert by_name["SIGNAL_SRC"].model == "Port"
    assert by_name["SIGNAL_SRC"].nets == ("n_input", "0")
    assert by_name["P_OUT"].nets == ("n_output", "0")
    # the `model msub1 ...` and `Options ...` lines are not instances
    assert "msub1" not in by_name


def test_parse_handles_sdd_self_tie_and_continuation() -> None:
    text = (
        "SDD:VARACTOR_C1  N__400 n_c1_v N__402 N__402  I[1,1]=(3e-13)*_v1 \\\n"
        "  I[2,1]=(-3e-13)*_v1\n"
    )
    insts = parse_oa_netlist(text)
    assert len(insts) == 1
    # 4 nets parsed (the I[1,1]=... body is a param, not a net)
    assert insts[0].nets == ("N__400", "n_c1_v", "N__402", "N__402")


def test_good_netlist_passes() -> None:
    rep = check_connectivity(
        _GOOD, source_nums=_SRC, named_nets=_NAMED,
        output_port_name="P_OUT", output_port_num=3)
    assert rep.ok, rep.failures
    assert rep.sources["SIGNAL_SRC"]["nets"] == ["n_input", "0"]
    assert not rep.sources["SIGNAL_SRC"]["shorted"]


def test_shorted_sources_are_caught() -> None:
    rep = check_connectivity(
        _SHORTED, source_nums=_SRC, named_nets=_NAMED,
        output_port_name="P_OUT", output_port_num=3)
    assert not rep.ok
    assert rep.sources["SIGNAL_SRC"]["shorted"]
    assert rep.sources["PUMP_SRC"]["shorted"]
    assert any("SIGNAL_SRC" in f and "SHORTED" in f for f in rep.failures)
    assert any("PUMP_SRC" in f and "SHORTED" in f for f in rep.failures)


def test_missing_source_is_caught() -> None:
    # Drop the SIGNAL_SRC line entirely (ADS would do this if it floated).
    dropped = "\n".join(
        ln for ln in _GOOD.splitlines() if "SIGNAL_SRC" not in ln)
    rep = check_connectivity(dropped, source_nums=_SRC)
    assert not rep.ok
    assert rep.sources["SIGNAL_SRC"]["present"] is False
    assert any("SIGNAL_SRC" in f and "MISSING" in f for f in rep.failures)


def test_source_return_off_ground_is_caught() -> None:
    # Return terminal on a non-ground net (e.g. a stray N__99) -> fault.
    bad = _GOOD.replace("n_input 0 Num=1", "n_input N__99 Num=1")
    rep = check_connectivity(bad, source_nums=_SRC)
    assert not rep.ok
    assert any("return terminal" in f for f in rep.failures)


def test_absent_named_net_is_caught() -> None:
    rep = check_connectivity(
        _GOOD, source_nums=_SRC, named_nets={"n_input", "n_missing"})
    assert not rep.ok
    assert any("n_missing" in f and "ABSENT" in f for f in rep.failures)


def test_controller_pseudo_net_not_floating() -> None:
    text = _GOOD + "S_Param:SP1  SP1_stim CalcS=yes\n"
    # SP1_stim is degree-1 but only on S_Param -> excluded from floating.
    floating = topology_floating_nets(parse_oa_netlist(text))
    assert "SP1_stim" not in floating


def test_new_floating_terminal_caught_against_baseline() -> None:
    # An extra series MLIN whose far end N__999 dangles is a NEW floater.
    extra = _GOOD + "MLIN2:DANGLE  N__999 n_output Subst=\"msub1\" W=1 mm L=1 mm\n"
    rep = check_connectivity(
        extra, source_nums=_SRC, allowed_floating=set())
    assert not rep.ok
    assert any("N__999" in f for f in rep.failures)


