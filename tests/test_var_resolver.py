"""Tests for the VAR named-parameter resolver.

The re-scoped design replaces a VAR *emitter* with a spec-level
resolver: ADS VAR/VarEqn is non-functional in ADS 2026 Update 1.2 (see
``docs/LIMITATIONS.md``), so a VAR block is never emitted -- the
resolver substitutes each variable-name reference in an element
parameter with the variable's literal value before emission. These
tests cover the resolver in isolation and confirm the emitters consume
its output deterministically. All tests are ADS-free.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tvtl_ads_builder_v2.device_spec import (
    DeviceSpecError,
    VarBlockSpec,
    VarEntry,
    resolve_spec,
)
from tvtl_ads_builder_v2.device_spec.var_resolver import (
    resolve_parameter_value,
    resolved_namespace,
)
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.parity.canonical_circuits import make_single_var_block


# --- the pure substitution core -----------------------------------------


def test_resolved_namespace():
    block = VarBlockSpec("VAR1", (VarEntry("R", 50.0), VarEntry("Cvar", 7.0e-12)))
    assert resolved_namespace(block) == {"R": 50.0, "Cvar": 7.0e-12}


def test_resolve_parameter_value_variable_name():
    assert resolve_parameter_value("R", {"R": 50.0}) == 50.0


def test_resolve_parameter_value_non_variable_string_passthrough():
    assert resolve_parameter_value("smv1245_ref", {"R": 50.0}) == "smv1245_ref"


def test_resolve_parameter_value_number_passthrough():
    assert resolve_parameter_value(7.0e-12, {"R": 50.0}) == 7.0e-12


# --- resolve_spec --------------------------------------------------------


def test_resolve_spec_no_var_block_is_identity():
    spec = replace(make_single_var_block(label="t"), var_block=None)
    assert resolve_spec(spec) is spec


def test_resolve_spec_substitutes_literals_and_drops_block():
    resolved = resolve_spec(make_single_var_block(label="t"))
    assert resolved.var_block is None
    rlc = next(e for e in resolved.elements if e.id == "RLC1")
    assert rlc.parameters == {"r_ohm": 50.0, "l_h": 1.0e-9, "c_f": 7.0e-12}


def test_resolve_spec_is_idempotent():
    once = resolve_spec(make_single_var_block(label="t"))
    twice = resolve_spec(once)
    assert twice is once


def test_resolve_spec_rejects_malformed_var_block():
    spec = make_single_var_block(label="t")
    bad = replace(
        spec, var_block=VarBlockSpec("VAR1", (VarEntry("1bad", 1.0),))
    )
    with pytest.raises(DeviceSpecError):
        resolve_spec(bad)


# --- emitter consumption (deterministic, literal-only) ------------------


def test_emit_is_deterministic():
    spec = make_single_var_block(label="t")
    assert emit_hpeesofsim_netlist(spec) == emit_hpeesofsim_netlist(spec)


def test_emit_matches_pre_resolved_spec():
    spec = make_single_var_block(label="t")
    resolved = resolve_spec(spec)
    assert emit_hpeesofsim_netlist(spec) == emit_hpeesofsim_netlist(resolved)


def test_emitted_netlist_carries_literals_not_a_var_block():
    netlist = emit_hpeesofsim_netlist(make_single_var_block(label="t"))
    assert "R=50 Ohm" in netlist
    assert "L=1 nH" in netlist
    assert "C=7 pF" in netlist
    # VAR is resolved away -- no VAR block / VarEqn element is emitted.
    assert "VAR1" not in netlist
    assert "VarEqn" not in netlist


