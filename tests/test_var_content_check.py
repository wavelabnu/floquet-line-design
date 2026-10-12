"""Tests for the VAR named-parameter content cross-check.

The content check verifies two invariants for a circuit with a VAR
block: (1) the resolver substitutes each variable reference with the
variable's declared literal, and (2) both backends emit those literals
identically. All tests are ADS-free -- the OA backend is stood in for
by a second hpeesofsim netlist (deterministically emitted from the same
resolved spec), and mismatch cases are produced by tampering with that
text.
"""

from __future__ import annotations

from dataclasses import replace

from tvtl_ads_builder_v2.device_spec import resolve_spec
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa.var_content_check import (
    check_var_content,
    find_var_references,
    format_content_report,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import make_single_var_block


def _spec_and_netlist():
    spec = make_single_var_block(label="var_content_test")
    return spec, emit_hpeesofsim_netlist(spec)


def test_find_var_references():
    spec, _ = _spec_and_netlist()
    refs = {(r.element_id, r.parameter): r for r in find_var_references(spec)}
    # R, L and Cvar are referenced by RLC1; f0 is a declared-but-unused
    # named constant, so it is not a reference.
    assert len(refs) == 3
    assert refs[("RLC1", "r_ohm")].variable == "R"
    assert refs[("RLC1", "r_ohm")].expected_value == 50.0
    assert refs[("RLC1", "c_f")].expected_value == 7.0e-12


def test_content_check_passes_for_wellformed_circuit():
    spec, netlist = _spec_and_netlist()
    result = check_var_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=netlist
    )
    assert result.match, result.failures
    assert result.checked == ["RLC1"]


def test_content_check_resolver_invariant_only():
    # With no netlist text, only invariant 1 (resolver correctness) runs.
    spec, _ = _spec_and_netlist()
    result = check_var_content(spec)
    assert result.match, result.failures
    assert "RLC1" in result.checked


def test_content_check_detects_value_mismatch():
    spec, netlist = _spec_and_netlist()
    tampered = netlist.replace("R=50 Ohm", "R=999 Ohm")
    result = check_var_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=tampered
    )
    assert not result.match
    assert any("RLC1" in failure for failure in result.failures)


def test_content_check_detects_missing_element_parameter():
    spec, netlist = _spec_and_netlist()
    # Drop RLC1's resistor sub-element line from one backend.
    dropped = "\n".join(
        line for line in netlist.splitlines() if not line.startswith("R:RLC1_R")
    )
    result = check_var_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=dropped
    )
    assert not result.match
    assert any("RLC1" in failure for failure in result.failures)


def test_content_check_cross_backend_consistency():
    # Both backends emitted from the same resolved spec -> identical.
    spec, netlist = _spec_and_netlist()
    other = emit_hpeesofsim_netlist(resolve_spec(spec))
    result = check_var_content(
        spec, hpeesofsim_netlist_text=netlist, ads_netlist_text=other
    )
    assert result.match, result.failures


def test_content_check_no_var_block_is_noop():
    spec = replace(make_single_var_block(label="t"), var_block=None)
    result = check_var_content(spec)
    assert result.match
    assert result.checked == []


def test_format_content_report_renders_verdict():
    spec, netlist = _spec_and_netlist()
    report = format_content_report(
        check_var_content(spec, hpeesofsim_netlist_text=netlist)
    )
    assert "VAR named-parameter content check: MATCH" in report


