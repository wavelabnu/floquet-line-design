"""Schema + validator tests for the VAR equations block.

VAR is a spec-level construct: a :class:`VarBlockSpec` on
``DeviceSpec.var_block`` holding named scalar parameters. It is
resolved away before emission (see ``test_var_resolver.py``); these
tests cover the dataclasses' round-trip serialization and the
``validate_var_block`` invariants.

Scope: a variable's value is a literal number. Symbolic cross-variable
expressions are not supported, so a ``VarEntry`` value is numeric; a
*reference* to a variable is a string element-parameter value naming
it, which ``validate_device_spec`` accepts.
"""

from __future__ import annotations

import json

import pytest

from tvtl_ads_builder_v2.device_spec import (
    Analysis,
    DeviceSpec,
    DeviceSpecError,
    Element,
    ElementPin,
    FrequencyGrid,
    Node,
    Port,
    Substrate,
    VarBlockSpec,
    VarEntry,
    validate_device_spec,
    validate_var_block,
)
from tvtl_ads_builder_v2.device_spec.schema import DEVICE_SPEC_VERSION

_SUBSTRATE = Substrate(
    name="msub1",
    model="hammerstad_kirschning",
    parameters={
        "eps_r": 3.38,
        "h_m": 1.54e-3,
        "tan_delta": 0.0027,
        "sigma_S_per_m": 5.8e7,
        "t_m": 17.5e-6,
    },
)


def _var_block(variables=None, name="VAR1") -> VarBlockSpec:
    if variables is None:
        variables = (
            VarEntry("R", 50.0),
            VarEntry("L", 1.0e-9),
            VarEntry("Cvar", 7.0e-12),
            VarEntry("f0", 3.0e9),
        )
    return VarBlockSpec(name=name, variables=tuple(variables))


def _var_spec(*, var_block="default", rlc_params=None) -> DeviceSpec:
    """A ``P1 -- RLC_SERIES -- P2`` spec, by default with a VAR block."""

    if var_block == "default":
        var_block = _var_block()
    if rlc_params is None:
        rlc_params = {"r_ohm": "R", "l_h": "L", "c_f": "Cvar"}
    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(start=1.0e9, stop=10.0e9, points=91),
        port_impedance_ohm=50.0,
    )
    nodes = (Node("n_in"), Node("n_out"))
    p1 = Element("P1", "TermG", (ElementPin(1, "n_in"),), {"z_ohm": 50.0, "num": 1})
    rlc = Element(
        "RLC1",
        "RLC_SERIES",
        (ElementPin(1, "n_in"), ElementPin(2, "n_out")),
        dict(rlc_params),
    )
    p2 = Element("P2", "TermG", (ElementPin(1, "n_out"),), {"z_ohm": 50.0, "num": 2})
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=_SUBSTRATE,
        nodes=nodes,
        elements=(p1, rlc, p2),
        ports=(Port(1, "n_in"), Port(2, "n_out")),
        label="var_test",
        var_block=var_block,
    )


# --- round-trip serialization -------------------------------------------


def test_varentry_roundtrip():
    entry = VarEntry("Cvar", 7.0e-12)
    assert VarEntry.from_dict(entry.to_dict()) == entry


def test_varblockspec_roundtrip():
    block = _var_block()
    assert VarBlockSpec.from_dict(block.to_dict()) == block


def test_devicespec_var_block_dict_roundtrip():
    spec = _var_spec()
    assert DeviceSpec.from_dict(spec.to_dict()) == spec


def test_devicespec_var_block_json_roundtrip():
    spec = _var_spec()
    restored = DeviceSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.var_block == spec.var_block


def test_devicespec_without_var_block_roundtrip():
    spec = _var_spec(
        var_block=None, rlc_params={"r_ohm": 50.0, "l_h": 1.0e-9, "c_f": 7.0e-12}
    )
    assert spec.var_block is None
    assert "var_block" not in spec.to_dict()
    assert DeviceSpec.from_dict(spec.to_dict()) == spec


# --- validate_var_block --------------------------------------------------


def test_validate_var_block_accepts_wellformed():
    validate_var_block(_var_block())


def test_validate_var_block_rejects_digit_leading_name():
    with pytest.raises(DeviceSpecError):
        validate_var_block(_var_block(variables=(VarEntry("1R", 1.0),)))


def test_validate_var_block_rejects_non_identifier_name():
    with pytest.raises(DeviceSpecError):
        validate_var_block(_var_block(variables=(VarEntry("R-1", 1.0),)))


def test_validate_var_block_rejects_duplicate_names():
    with pytest.raises(DeviceSpecError):
        validate_var_block(
            _var_block(variables=(VarEntry("R", 1.0), VarEntry("R", 2.0)))
        )


def test_validate_var_block_rejects_non_numeric_value():
    with pytest.raises(DeviceSpecError):
        validate_var_block(_var_block(variables=(VarEntry("R", "fifty"),)))


def test_validate_var_block_rejects_non_finite_value():
    with pytest.raises(DeviceSpecError):
        validate_var_block(_var_block(variables=(VarEntry("R", float("inf")),)))


def test_validate_var_block_rejects_bad_block_name():
    with pytest.raises(DeviceSpecError):
        validate_var_block(_var_block(name="1bad"))


# --- validate_device_spec with a VAR block ------------------------------


def test_validate_device_spec_accepts_var_references():
    validate_device_spec(_var_spec())


def test_validate_device_spec_accepts_numeric_params_with_var_block():
    validate_device_spec(
        _var_spec(rlc_params={"r_ohm": 50.0, "l_h": 1.0e-9, "c_f": 7.0e-12})
    )


def test_validate_device_spec_rejects_undeclared_var_reference():
    spec = _var_spec(
        rlc_params={"r_ohm": "Rmissing", "l_h": "L", "c_f": "Cvar"}
    )
    with pytest.raises(DeviceSpecError):
        validate_device_spec(spec)


def test_validate_device_spec_rejects_malformed_var_block():
    spec = _var_spec(var_block=_var_block(variables=(VarEntry("1R", 1.0),)))
    with pytest.raises(DeviceSpecError):
        validate_device_spec(spec)


