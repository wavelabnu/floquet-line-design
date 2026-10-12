"""Tests for the two-tone HB controller content cross-check.

The content check verifies a circuit's two-tone ``HB:`` directive
matches its ``hb_two_tone`` spec and that both backends emit it
identically. All tests are ADS-free -- the OA backend is stood in for
by a second hpeesofsim netlist, and mismatch cases are produced by
tampering with that text.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tvtl_ads_builder_v2.device_spec import (
    HBTwoToneSpec,
    VarBlockSpec,
    VarEntry,
)
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa.hb_controller_content_check import (
    check_hb_two_tone_content,
    format_content_report,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_two_tone_hb_controller,
)


def _spec_and_netlist():
    spec = make_two_tone_hb_controller(label="hb2_content_test")
    return spec, emit_hpeesofsim_netlist(spec)


def _named_freq_spec():
    """A two-tone HB spec whose both fundamentals name VAR variables."""

    base = make_two_tone_hb_controller(label="hb2_named_content_test")
    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name="f_signal", value=3.0e9),
            VarEntry(name="f_pump", value=6.0e9),
        ),
    )
    hb = HBTwoToneSpec(
        fundamental_freq_1="f_signal",
        fundamental_order_1=5,
        fundamental_freq_2="f_pump",
        fundamental_order_2=5,
        max_order=10,
    )
    return replace(base, var_block=var_block, hb_two_tone=hb)


def test_content_check_passes_for_wellformed_circuit():
    spec, netlist = _spec_and_netlist()
    result = check_hb_two_tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=netlist
    )
    assert result.match, result.failures
    assert result.checked == ["HB1"]


@pytest.mark.parametrize(
    "old,new,parameter",
    [
        ("Freq[1]=3 GHz", "Freq[1]=9 GHz", "Freq[1]"),
        ("Freq[2]=6 GHz", "Freq[2]=9 GHz", "Freq[2]"),
        ("Order[1]=5", "Order[1]=8", "Order[1]"),
        ("Order[2]=5", "Order[2]=8", "Order[2]"),
        ("MaxOrder=10", "MaxOrder=3", "MaxOrder"),
    ],
)
def test_content_check_detects_per_parameter_mismatch(old, new, parameter):
    spec, netlist = _spec_and_netlist()
    tampered = netlist.replace(old, new)
    assert tampered != netlist
    result = check_hb_two_tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=tampered
    )
    assert not result.match
    assert any(parameter in failure for failure in result.failures)


def test_content_check_detects_missing_hb_line():
    spec, netlist = _spec_and_netlist()
    dropped = "\n".join(
        line for line in netlist.splitlines() if not line.startswith("HB:")
    )
    result = check_hb_two_tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=dropped
    )
    assert not result.match
    assert any("no HB" in failure for failure in result.failures)


def test_content_check_cross_backend_consistency():
    spec, netlist = _spec_and_netlist()
    other = emit_hpeesofsim_netlist(spec)
    result = check_hb_two_tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=other
    )
    assert result.match, result.failures


def test_content_check_no_two_tone_is_noop():
    spec = replace(make_two_tone_hb_controller(label="t"), hb_two_tone=None)
    result = check_hb_two_tone_content(spec)
    assert result.match


def test_format_content_report_renders_verdict():
    spec, netlist = _spec_and_netlist()
    report = format_content_report(
        check_hb_two_tone_content(spec, hpeesofsim_netlist_text=netlist)
    )
    assert "HB controller content check: MATCH" in report


# --- Named-reference fundamental frequencies ----------------------------


def test_content_check_resolves_named_freqs():
    """Both named fundamentals resolve to their VAR literals on each backend.

    ``fundamental_freq_1="f_signal"`` / ``fundamental_freq_2="f_pump"``
    with a VAR block declaring ``3 GHz`` / ``6 GHz``; the emitted deck
    carries ``Freq[1]=3 GHz Freq[2]=6 GHz``, which the content check --
    resolving the spec internally -- must MATCH.
    """

    spec = _named_freq_spec()
    netlist = emit_hpeesofsim_netlist(spec)
    assert "Freq[1]=3 GHz Freq[2]=6 GHz" in netlist
    result = check_hb_two_tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=netlist
    )
    assert result.match, result.failures


@pytest.mark.parametrize(
    "old,new,parameter",
    [
        ("Freq[1]=3 GHz", "Freq[1]=9 GHz", "Freq[1]"),
        ("Freq[2]=6 GHz", "Freq[2]=9 GHz", "Freq[2]"),
    ],
)
def test_content_check_named_freqs_detect_mismatch(old, new, parameter):
    """A tampered fundamental still fails against a named-reference spec."""

    spec = _named_freq_spec()
    netlist = emit_hpeesofsim_netlist(spec)
    tampered = netlist.replace(old, new)
    assert tampered != netlist
    result = check_hb_two_tone_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=tampered
    )
    assert not result.match
    assert any(parameter in failure for failure in result.failures)


