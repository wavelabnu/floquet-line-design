"""Tests for the ``P_1Tone`` power-source content cross-check.

The content check verifies a circuit's ``P_1Tone:`` source lines match
its ``sources`` spec (with VAR named references resolved) and that both
backends emit them identically. All tests are ADS-free -- the OA
backend is stood in for by a second hpeesofsim netlist (deterministic
from the same spec), and mismatch cases are produced by tampering with
that text.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa.netlist_compare import (
    compare_circuits,
    parse_netlist_text,
)
from tvtl_ads_builder_v2.oa.p1tone_content_check import (
    check_p1tone_content,
    find_p1tone_lines,
    format_content_report,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_tvtl_unit_cell_two_tone_hb,
    make_two_tone_p1tone_sources,
)


def _spec_and_netlist():
    spec = make_two_tone_p1tone_sources(label="p1tone_content_test")
    return spec, emit_hpeesofsim_netlist(spec)


def _oa_form(hpeesofsim_netlist: str) -> str:
    """Rewrite each hpeesofsim ``P_1Tone:`` line into ADS's OA form.

    ADS 2026 normalises the ``ads_sources/P_1Tone`` OA cell to a
    ``Port:`` line: the element token becomes ``Port:``, the tone
    parameters become array-indexed (``Freq`` -> ``Freq[1]``, ``P`` ->
    ``P[1]``), and a default ``Pac`` is appended -- exactly the form a
    live emission probe captured.
    Every non-source line passes through unchanged, so the result
    stands in for the OA-generated netlist of the same circuit.
    """

    lines: list[str] = []
    for line in hpeesofsim_netlist.splitlines():
        if line.strip().startswith("P_1Tone:"):
            lines.append(
                line.replace("P_1Tone:", "Port:", 1)
                .replace(" Freq=", " Freq[1]=")
                .replace(" P=", " P[1]=")
                .rstrip()
                + " Pac=polar(dbmtow(0),0)"
            )
        else:
            lines.append(line)
    return "\n".join(lines)


def test_find_p1tone_lines_parses_both_sources():
    _spec, netlist = _spec_and_netlist()
    found = find_p1tone_lines(netlist)
    assert set(found) == {"SIGNAL_SRC", "PUMP_SRC"}
    signal = found["SIGNAL_SRC"]
    assert signal.pos_node == "n_in"
    assert signal.neg_node == "0"
    assert signal.num == 1.0
    assert signal.z_ohm == 50.0
    assert signal.freq_Hz == 3.0e9
    assert signal.power_dBm == -30.0


def test_content_check_passes_for_wellformed_circuit():
    spec, netlist = _spec_and_netlist()
    result = check_p1tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=netlist
    )
    assert result.match, result.failures
    assert result.checked == ["PUMP_SRC", "SIGNAL_SRC"]


@pytest.mark.parametrize(
    "old,new,token",
    [
        ("Num=1 Z=50 Ohm Freq=3 GHz", "Num=2 Z=50 Ohm Freq=3 GHz", "Num"),
        ("Num=1 Z=50 Ohm Freq=3 GHz", "Num=1 Z=75 Ohm Freq=3 GHz", "Z"),
        ("Freq=3 GHz", "Freq=9 GHz", "Freq"),
        ("P=polar(dbmtow(-30),0)", "P=polar(dbmtow(-9),0)", "P"),
    ],
)
def test_content_check_detects_per_parameter_mismatch(old, new, token):
    spec, netlist = _spec_and_netlist()
    tampered = netlist.replace(old, new)
    assert tampered != netlist
    result = check_p1tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=tampered
    )
    assert not result.match
    assert any(token in failure for failure in result.failures)


def test_content_check_detects_wrong_driven_node():
    spec, netlist = _spec_and_netlist()
    tampered = netlist.replace(
        "P_1Tone:SIGNAL_SRC n_in 0", "P_1Tone:SIGNAL_SRC n_wrong 0"
    )
    assert tampered != netlist
    result = check_p1tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=tampered
    )
    assert not result.match
    assert any("n_wrong" in failure for failure in result.failures)


def test_content_check_detects_missing_source():
    spec, netlist = _spec_and_netlist()
    dropped = "\n".join(
        line for line in netlist.splitlines()
        if not line.startswith("P_1Tone:PUMP_SRC")
    )
    result = check_p1tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=dropped
    )
    assert not result.match
    assert any("PUMP_SRC" in failure for failure in result.failures)


def test_content_check_cross_backend_consistency():
    spec, netlist = _spec_and_netlist()
    other = emit_hpeesofsim_netlist(spec)
    result = check_p1tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=other
    )
    assert result.match, result.failures


def test_content_check_no_sources_is_noop():
    spec = replace(make_two_tone_p1tone_sources(label="t"), sources=())
    result = check_p1tone_content(spec)
    assert result.match


def test_content_check_named_references_resolve_identically():
    """Both backends emit the VAR-resolved literal for every source field.

    The Level-7 source canonical references ``f_signal``, ``f_pump``,
    ``signal_power_dBm`` and ``pump_power_dBm`` by name; the content
    check resolves the spec internally, so the parsed netlist literals
    must match the substituted VAR values on both backends.
    """

    spec, netlist = _spec_and_netlist()
    # both backends emit from the same resolved spec -> identical literals.
    result = check_p1tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=netlist
    )
    assert result.match, result.failures
    found = find_p1tone_lines(netlist)
    assert found["SIGNAL_SRC"].freq_Hz == 3.0e9
    assert found["SIGNAL_SRC"].power_dBm == -30.0
    assert found["PUMP_SRC"].freq_Hz == 6.0e9
    assert found["PUMP_SRC"].power_dBm == 5.0


def test_content_check_named_reference_divergence_is_caught():
    """A backend that resolved a named freq differently is flagged.

    Tampering one backend's ``Freq`` simulates a resolver that put a
    different literal on the two backends -- the cross-backend
    invariant the named-reference lift must hold.
    """

    spec, netlist = _spec_and_netlist()
    diverged = netlist.replace("Freq=6 GHz", "Freq=7 GHz")
    result = check_p1tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=diverged
    )
    assert not result.match
    assert any(
        "cross-check" in failure or "Freq" in failure
        for failure in result.failures
    )


def test_content_check_integrated_cell_sources():
    """The integrated Level-7 cell's sources content-check MATCH."""

    spec = make_tvtl_unit_cell_two_tone_hb(label="t")
    netlist = emit_hpeesofsim_netlist(spec)
    result = check_p1tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=netlist
    )
    assert result.match, result.failures
    assert result.checked == ["PUMP_SRC", "SIGNAL_SRC"]


def test_format_content_report_renders_verdict():
    spec, netlist = _spec_and_netlist()
    report = format_content_report(
        check_p1tone_content(spec, hpeesofsim_netlist_text=netlist)
    )
    assert "P_1Tone source content check: MATCH" in report


# --- ADS Port: normalisation -- the OA backend's source form ------------


def test_parser_recognizes_p1tone_token_form():
    """The parser reads the hpeesofsim ``P_1Tone:`` source form."""

    _spec, netlist = _spec_and_netlist()
    assert "P_1Tone:SIGNAL_SRC" in netlist
    found = find_p1tone_lines(netlist)
    assert set(found) == {"SIGNAL_SRC", "PUMP_SRC"}


def test_parser_recognizes_oa_port_source_form():
    """The parser reads ADS's OA-normalised ``Port:`` source form.

    A ``Port:`` line carrying ``P[1]`` / ``Freq[1]`` (and a ``Pac``) is
    a P_1Tone source; its fields parse identically to the hpeesofsim
    ``P_1Tone:`` form.
    """

    _spec, netlist = _spec_and_netlist()
    oa = _oa_form(netlist)
    assert "Port:SIGNAL_SRC" in oa and "P[1]=" in oa and "Freq[1]=" in oa
    assert "P_1Tone:" not in oa
    found = find_p1tone_lines(oa)
    assert set(found) == {"SIGNAL_SRC", "PUMP_SRC"}
    signal = found["SIGNAL_SRC"]
    assert signal.pos_node == "n_in"
    assert signal.neg_node == "0"
    assert signal.num == 1.0
    assert signal.z_ohm == 50.0
    assert signal.freq_Hz == 3.0e9
    assert signal.power_dBm == -30.0


def test_parser_distinguishes_source_port_from_sink_port():
    """A sink ``Port:`` (output termination) is NOT read as a source.

    The OA-form netlist carries both: two source ``Port:`` lines (with
    ``P[1]``) and the output ``Port:P_OUT`` sink (``Num`` / ``Z`` /
    ``Noise`` only). The parser must return just the two sources.
    """

    _spec, netlist = _spec_and_netlist()
    oa = _oa_form(netlist)
    assert "Port:P_OUT" in oa  # the sink output port is present
    found = find_p1tone_lines(oa)
    assert set(found) == {"SIGNAL_SRC", "PUMP_SRC"}
    assert "P_OUT" not in found


def test_content_check_cross_backend_p1tone_vs_port():
    """A ``P_1Tone:`` emission and an equivalent ``Port:`` emission MATCH.

    hpeesofsim writes ``P_1Tone:``; ADS normalises the OA cell to
    ``Port:``. The content check must treat the two token forms as the
    same source and report MATCH when their parameters agree.
    """

    spec, netlist = _spec_and_netlist()
    result = check_p1tone_content(
        spec,
        hpeesofsim_netlist_text=netlist,
        ads_netlist_text=_oa_form(netlist),
    )
    assert result.match, result.failures
    assert result.checked == ["PUMP_SRC", "SIGNAL_SRC"]


def test_content_check_detects_mismatch_across_token_forms():
    """A real parameter mismatch is still caught across the token forms.

    Token-asymmetry is accepted, but a genuine value disagreement
    between a ``P_1Tone:`` backend and a ``Port:`` backend must still
    fail.
    """

    spec, netlist = _spec_and_netlist()
    diverged = _oa_form(netlist).replace("Freq[1]=6 GHz", "Freq[1]=7 GHz")
    result = check_p1tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=diverged
    )
    assert not result.match
    assert any("Freq" in failure for failure in result.failures)


def test_content_check_integrated_cell_cross_token():
    """The integrated Level-7 cell content-checks MATCH across token forms."""

    spec = make_tvtl_unit_cell_two_tone_hb(label="t")
    netlist = emit_hpeesofsim_netlist(spec)
    result = check_p1tone_content(
        spec,
        hpeesofsim_netlist_text=netlist,
        ads_netlist_text=_oa_form(netlist),
    )
    assert result.match, result.failures


def test_netlist_compare_merges_p1tone_and_port_source():
    """The structural comparator does not count the source token forms.

    hpeesofsim emits ``P_1Tone:`` sources, ADS emits ``Port:`` sources;
    both must be excluded from the element inventory so the two
    backends' element counts stay equal (the element-count audit). A
    ``P_1Tone:`` deck and its OA-form equivalent compare as MATCH.
    """

    _spec, netlist = _spec_and_netlist()
    result = compare_circuits(
        parse_netlist_text(netlist),
        parse_netlist_text(_oa_form(netlist)),
    )
    assert result.match, result.differences


def test_netlist_compare_sink_port_stays_a_counted_element():
    """A sink ``Port:`` is still a counted element after the merge.

    Only source-form ``Port:`` / ``P_1Tone:`` lines are excluded; the
    output-port termination stays in the element inventory.
    """

    _spec, netlist = _spec_and_netlist()
    parsed = parse_netlist_text(_oa_form(netlist))
    port_elements = [c for c in parsed.elements if c.ctype == "PORT"]
    # exactly the one sink output port -- the two sources are excluded.
    assert [c.name for c in port_elements] == ["P_OUT"]


