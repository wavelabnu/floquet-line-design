"""Schema + validator tests for the single-tone HB controller.

:class:`HBSingleToneSpec` lives on ``DeviceSpec.hb_single_tone`` -- a
simulation analysis directive, not a connected element. These tests
cover the dataclass round-trip serialization and the
``validate_hb_single_tone`` invariants.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tvtl_ads_builder_v2.device_spec import (
    Analysis,
    DeviceSpec,
    DeviceSpecError,
    Element,
    ElementPin,
    FrequencyGrid,
    HBSingleToneSpec,
    Node,
    Port,
    Substrate,
    VarBlockSpec,
    VarEntry,
    resolve_spec,
    validate_device_spec,
    validate_hb_single_tone,
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


def _hb(fundamental_freq=3.0e9, fundamental_order=5, max_order=5):
    return HBSingleToneSpec(
        fundamental_freq=fundamental_freq,
        fundamental_order=fundamental_order,
        max_order=max_order,
    )


def _hb_spec(*, hb="default") -> DeviceSpec:
    """A ``P1 -- R -- P2`` spec, by default with an HB controller."""

    if hb == "default":
        hb = _hb()
    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(start=1.0e9, stop=10.0e9, points=91),
        port_impedance_ohm=50.0,
    )
    nodes = (Node("n_in"), Node("n_out"))
    p1 = Element("P1", "TermG", (ElementPin(1, "n_in"),), {"z_ohm": 50.0, "num": 1})
    load = Element(
        "LOAD", "R",
        (ElementPin(1, "n_in"), ElementPin(2, "n_out")),
        {"r_ohm": 50.0},
    )
    p2 = Element("P2", "TermG", (ElementPin(1, "n_out"),), {"z_ohm": 50.0, "num": 2})
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=_SUBSTRATE,
        nodes=nodes,
        elements=(p1, load, p2),
        ports=(Port(1, "n_in"), Port(2, "n_out")),
        label="hb_test",
        hb_single_tone=hb,
    )


# --- round-trip serialization -------------------------------------------


def test_hbsingletonespec_roundtrip():
    hb = _hb()
    assert HBSingleToneSpec.from_dict(hb.to_dict()) == hb


def test_devicespec_hb_dict_roundtrip():
    spec = _hb_spec()
    assert DeviceSpec.from_dict(spec.to_dict()) == spec


def test_devicespec_hb_json_roundtrip():
    spec = _hb_spec()
    restored = DeviceSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.hb_single_tone == spec.hb_single_tone


def test_devicespec_without_hb_roundtrip():
    spec = _hb_spec(hb=None)
    assert spec.hb_single_tone is None
    assert "hb_single_tone" not in spec.to_dict()
    assert DeviceSpec.from_dict(spec.to_dict()) == spec


# --- validate_hb_single_tone --------------------------------------------


def test_validate_hb_accepts_wellformed():
    validate_hb_single_tone(_hb())


def test_validate_hb_rejects_nonpositive_freq():
    with pytest.raises(DeviceSpecError):
        validate_hb_single_tone(_hb(fundamental_freq=0.0))
    with pytest.raises(DeviceSpecError):
        validate_hb_single_tone(_hb(fundamental_freq=-3.0e9))


def test_validate_hb_rejects_nonfinite_freq():
    with pytest.raises(DeviceSpecError):
        validate_hb_single_tone(_hb(fundamental_freq=float("inf")))


def test_validate_hb_rejects_order_below_one():
    with pytest.raises(DeviceSpecError):
        validate_hb_single_tone(_hb(fundamental_order=0))


def test_validate_hb_rejects_non_integer_order():
    with pytest.raises(DeviceSpecError):
        validate_hb_single_tone(_hb(fundamental_order=5.5))


def test_validate_hb_rejects_max_order_below_fundamental():
    with pytest.raises(DeviceSpecError):
        validate_hb_single_tone(_hb(fundamental_order=5, max_order=4))


def test_validate_hb_accepts_max_order_above_fundamental():
    validate_hb_single_tone(_hb(fundamental_order=5, max_order=7))


# --- validate_device_spec with an HB controller -------------------------


def test_validate_device_spec_accepts_hb():
    validate_device_spec(_hb_spec())


def test_validate_device_spec_rejects_malformed_hb():
    with pytest.raises(DeviceSpecError):
        validate_device_spec(_hb_spec(hb=_hb(fundamental_order=0)))


# --- Named-reference fundamental_freq -----------------------------------


def test_hbsingletonespec_named_freq_roundtrip():
    """A string-valued fundamental_freq round-trips verbatim."""

    hb = _hb(fundamental_freq="f_signal")
    restored = HBSingleToneSpec.from_dict(json.loads(json.dumps(hb.to_dict())))
    assert restored == hb
    assert restored.fundamental_freq == "f_signal"


def test_validate_hb_accepts_named_freq_in_var_block():
    """A named fundamental_freq validates when the VAR block declares it."""

    validate_hb_single_tone(
        _hb(fundamental_freq="f_signal"), frozenset({"f_signal"})
    )


def test_validate_hb_rejects_named_freq_not_declared():
    """A named fundamental_freq with no matching VAR variable is rejected."""

    with pytest.raises(DeviceSpecError):
        validate_hb_single_tone(
            _hb(fundamental_freq="f_signal"), frozenset({"other"})
        )
    # an empty var-name set (no VAR block) rejects a string freq too.
    with pytest.raises(DeviceSpecError):
        validate_hb_single_tone(_hb(fundamental_freq="f_signal"))


def test_validate_device_spec_accepts_hb_named_freq():
    """validate_device_spec resolves an HB named freq against the VAR block."""

    spec = _hb_spec(hb=_hb(fundamental_freq="f_signal"))
    var_block = VarBlockSpec(
        name="VAR1", variables=(VarEntry(name="f_signal", value=3.0e9),)
    )
    spec = replace(spec, var_block=var_block)
    validate_device_spec(spec)


def test_validate_device_spec_rejects_hb_named_freq_undeclared():
    """An HB named freq that no VAR variable declares is rejected."""

    spec = _hb_spec(hb=_hb(fundamental_freq="missing"))
    var_block = VarBlockSpec(
        name="VAR1", variables=(VarEntry(name="f_signal", value=3.0e9),)
    )
    spec = replace(spec, var_block=var_block)
    with pytest.raises(DeviceSpecError):
        validate_device_spec(spec)


def test_resolve_spec_substitutes_hb_named_freq():
    """resolve_spec substitutes a named HB fundamental_freq to its literal."""

    spec = _hb_spec(hb=_hb(fundamental_freq="f_signal"))
    var_block = VarBlockSpec(
        name="VAR1", variables=(VarEntry(name="f_signal", value=3.1e9),)
    )
    spec = replace(spec, var_block=var_block)
    resolved = resolve_spec(spec)
    assert resolved.var_block is None
    assert resolved.hb_single_tone.fundamental_freq == 3.1e9


def test_resolve_spec_literal_hb_freq_passthrough():
    """A literal HB fundamental_freq is unchanged by the resolver."""

    spec = _hb_spec(hb=_hb(fundamental_freq=4.0e9))
    var_block = VarBlockSpec(
        name="VAR1", variables=(VarEntry(name="f_signal", value=3.1e9),)
    )
    resolved = resolve_spec(replace(spec, var_block=var_block))
    assert resolved.hb_single_tone.fundamental_freq == 4.0e9


