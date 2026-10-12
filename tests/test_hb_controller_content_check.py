"""Tests for the single-tone HB controller content cross-check.

The content check verifies a circuit's ``HB:`` directive matches its
``hb_single_tone`` spec and that both backends emit it identically.
All tests are ADS-free -- the OA backend is stood in for by a second
hpeesofsim netlist (deterministically emitted from the same spec), and
mismatch cases are produced by tampering with that text.
"""

from __future__ import annotations

from dataclasses import replace

from tvtl_ads_builder_v2.device_spec import (
    HBSingleToneSpec,
    VarBlockSpec,
    VarEntry,
)
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa.hb_controller_content_check import (
    check_hb_controller_content,
    format_content_report,
    parse_hb_line,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_single_tone_hb_controller,
)


def _spec_and_netlist():
    spec = make_single_tone_hb_controller(label="hb_content_test")
    return spec, emit_hpeesofsim_netlist(spec)


def _named_freq_spec():
    """A single-tone HB spec whose fundamental_freq names a VAR variable."""

    base = make_single_tone_hb_controller(label="hb_named_content_test")
    var_block = VarBlockSpec(
        name="VAR1", variables=(VarEntry(name="f_signal", value=3.0e9),)
    )
    hb = HBSingleToneSpec(
        fundamental_freq="f_signal", fundamental_order=5, max_order=5
    )
    return replace(base, var_block=var_block, hb_single_tone=hb)


def test_parse_hb_line_normalizes_freq_to_hz():
    params = parse_hb_line(
        "HB:HB1 Freq[1]=3 GHz Order[1]=5 MaxOrder=5 StatusLevel=2"
    )
    assert params["Freq[1]"] == 3.0e9
    assert params["Order[1]"] == 5.0
    assert params["MaxOrder"] == 5.0


def test_content_check_passes_for_wellformed_circuit():
    spec, netlist = _spec_and_netlist()
    result = check_hb_controller_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=netlist
    )
    assert result.match, result.failures
    assert result.checked == ["HB1"]


def test_content_check_detects_value_mismatch():
    spec, netlist = _spec_and_netlist()
    tampered = netlist.replace("MaxOrder=5", "MaxOrder=8")
    result = check_hb_controller_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=tampered
    )
    assert not result.match
    assert any("MaxOrder" in failure for failure in result.failures)


def test_content_check_detects_missing_hb_line():
    spec, netlist = _spec_and_netlist()
    dropped = "\n".join(
        line for line in netlist.splitlines() if not line.startswith("HB:")
    )
    result = check_hb_controller_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=dropped
    )
    assert not result.match
    assert any("no HB" in failure for failure in result.failures)


def test_content_check_detects_second_tone_artifact():
    spec, netlist = _spec_and_netlist()
    two_tone = netlist.replace(
        "Order[1]=5", "Order[1]=5 Freq[2]=6 GHz Order[2]=5"
    )
    result = check_hb_controller_content(
        spec, hpeesofsim_netlist_text=two_tone
    )
    assert not result.match
    assert any(
        "Freq[2]" in failure or "Order[2]" in failure
        for failure in result.failures
    )


def test_content_check_cross_backend_consistency():
    spec, netlist = _spec_and_netlist()
    other = emit_hpeesofsim_netlist(spec)
    result = check_hb_controller_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=other
    )
    assert result.match, result.failures


def test_content_check_no_hb_controller_is_noop():
    spec = replace(make_single_tone_hb_controller(label="t"), hb_single_tone=None)
    result = check_hb_controller_content(spec)
    assert result.match


def test_format_content_report_renders_verdict():
    spec, netlist = _spec_and_netlist()
    report = format_content_report(
        check_hb_controller_content(spec, hpeesofsim_netlist_text=netlist)
    )
    assert "HB controller content check: MATCH" in report


# --- Named-reference fundamental_freq -----------------------------------


def test_content_check_resolves_named_freq():
    """A named fundamental_freq resolves to the VAR literal on both backends.

    The spec carries ``fundamental_freq="f_signal"`` and a VAR block
    declaring ``f_signal = 3 GHz``; the emitted deck therefore carries
    ``Freq[1]=3 GHz``, and the content check -- which resolves the spec
    internally -- must MATCH it.
    """

    spec = _named_freq_spec()
    netlist = emit_hpeesofsim_netlist(spec)
    assert "Freq[1]=3 GHz" in netlist
    result = check_hb_controller_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=netlist
    )
    assert result.match, result.failures


def test_content_check_named_freq_detects_mismatch():
    """A tampered Freq[1] still fails against a named-reference spec."""

    spec = _named_freq_spec()
    netlist = emit_hpeesofsim_netlist(spec)
    tampered = netlist.replace("Freq[1]=3 GHz", "Freq[1]=9 GHz")
    assert tampered != netlist
    result = check_hb_controller_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=tampered
    )
    assert not result.match
    assert any("Freq[1]" in failure for failure in result.failures)


