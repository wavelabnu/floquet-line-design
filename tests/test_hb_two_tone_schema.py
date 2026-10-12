"""Schema + validator tests for the two-tone HB controller.

:class:`HBTwoToneSpec` lives on ``DeviceSpec.hb_two_tone`` and is
mutually exclusive with ``hb_single_tone``. These tests cover the
dataclass round-trip and the ``validate_hb_two_tone`` invariants,
including the mutual-exclusivity rule.
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
    HBTwoToneSpec,
    Node,
    Port,
    Substrate,
    VarBlockSpec,
    VarEntry,
    resolve_spec,
    validate_device_spec,
    validate_hb_two_tone,
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


def _hb2(f1=3.0e9, o1=5, f2=6.0e9, o2=5, max_order=10):
    return HBTwoToneSpec(
        fundamental_freq_1=f1,
        fundamental_order_1=o1,
        fundamental_freq_2=f2,
        fundamental_order_2=o2,
        max_order=max_order,
    )


def _hb2_spec(*, hb_two_tone="default", hb_single_tone=None) -> DeviceSpec:
    """A ``P1 -- R -- P2`` spec, by default with a two-tone HB controller."""

    if hb_two_tone == "default":
        hb_two_tone = _hb2()
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
        label="hb2_test",
        hb_single_tone=hb_single_tone,
        hb_two_tone=hb_two_tone,
    )


# --- round-trip serialization -------------------------------------------


def test_hbtwotonespec_roundtrip():
    hb = _hb2()
    assert HBTwoToneSpec.from_dict(hb.to_dict()) == hb


def test_devicespec_hb2_dict_roundtrip():
    spec = _hb2_spec()
    assert DeviceSpec.from_dict(spec.to_dict()) == spec


def test_devicespec_hb2_json_roundtrip():
    spec = _hb2_spec()
    restored = DeviceSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.hb_two_tone == spec.hb_two_tone


# --- validate_hb_two_tone -----------------------------------------------


def test_validate_hb2_accepts_wellformed():
    validate_hb_two_tone(_hb2())


def test_validate_hb2_rejects_nonpositive_freq():
    with pytest.raises(DeviceSpecError):
        validate_hb_two_tone(_hb2(f1=0.0))
    with pytest.raises(DeviceSpecError):
        validate_hb_two_tone(_hb2(f2=-6.0e9))


def test_validate_hb2_rejects_nonfinite_freq():
    with pytest.raises(DeviceSpecError):
        validate_hb_two_tone(_hb2(f2=float("inf")))


def test_validate_hb2_rejects_order_below_one():
    with pytest.raises(DeviceSpecError):
        validate_hb_two_tone(_hb2(o1=0))
    with pytest.raises(DeviceSpecError):
        validate_hb_two_tone(_hb2(o2=0))


def test_validate_hb2_rejects_non_integer_order():
    with pytest.raises(DeviceSpecError):
        validate_hb_two_tone(_hb2(o1=5.5))


def test_validate_hb2_rejects_max_order_below_per_tone():
    with pytest.raises(DeviceSpecError):
        validate_hb_two_tone(_hb2(o1=5, o2=7, max_order=6))


def test_validate_hb2_accepts_max_order_at_per_tone_min():
    # max_order == max(order_1, order_2) is the accepted loose bound.
    validate_hb_two_tone(_hb2(o1=5, o2=7, max_order=7))


# --- validate_device_spec: mutual exclusivity ---------------------------


def test_validate_device_spec_accepts_two_tone_only():
    validate_device_spec(_hb2_spec())


def test_validate_device_spec_rejects_both_hb_controllers():
    spec = _hb2_spec(hb_single_tone=HBSingleToneSpec(3.0e9, 5, 5))
    with pytest.raises(DeviceSpecError):
        validate_device_spec(spec)


def test_validate_device_spec_rejects_malformed_hb2():
    with pytest.raises(DeviceSpecError):
        validate_device_spec(_hb2_spec(hb_two_tone=_hb2(o1=0)))


# --- Named-reference fundamental frequencies ----------------------------


def test_hbtwotonespec_named_freq_roundtrip():
    """String-valued fundamental frequencies round-trip verbatim."""

    hb = _hb2(f1="f_signal", f2="f_pump")
    restored = HBTwoToneSpec.from_dict(json.loads(json.dumps(hb.to_dict())))
    assert restored == hb
    assert restored.fundamental_freq_1 == "f_signal"
    assert restored.fundamental_freq_2 == "f_pump"


def test_validate_hb2_accepts_named_freqs_in_var_block():
    """Named fundamental frequencies validate against a VAR-name set."""

    validate_hb_two_tone(
        _hb2(f1="f_signal", f2="f_pump"),
        frozenset({"f_signal", "f_pump"}),
    )


def test_validate_hb2_rejects_named_freq_not_declared():
    """A named fundamental frequency with no VAR variable is rejected."""

    with pytest.raises(DeviceSpecError):
        validate_hb_two_tone(
            _hb2(f1="f_signal", f2="f_pump"), frozenset({"f_signal"})
        )
    with pytest.raises(DeviceSpecError):
        validate_hb_two_tone(_hb2(f1="f_signal"))


def test_validate_device_spec_accepts_hb2_named_freqs():
    """validate_device_spec resolves HB2 named freqs against the VAR block."""

    spec = _hb2_spec(hb_two_tone=_hb2(f1="f_signal", f2="f_pump"))
    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name="f_signal", value=3.0e9),
            VarEntry(name="f_pump", value=6.0e9),
        ),
    )
    validate_device_spec(replace(spec, var_block=var_block))


def test_validate_device_spec_rejects_hb2_named_freq_undeclared():
    """An HB2 named freq that no VAR variable declares is rejected."""

    spec = _hb2_spec(hb_two_tone=_hb2(f1="f_signal", f2="missing"))
    var_block = VarBlockSpec(
        name="VAR1", variables=(VarEntry(name="f_signal", value=3.0e9),)
    )
    with pytest.raises(DeviceSpecError):
        validate_device_spec(replace(spec, var_block=var_block))


def test_resolve_spec_substitutes_hb2_named_freqs():
    """resolve_spec substitutes both named HB2 frequencies to literals."""

    spec = _hb2_spec(hb_two_tone=_hb2(f1="f_signal", f2="f_pump"))
    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name="f_signal", value=3.1e9),
            VarEntry(name="f_pump", value=6.2e9),
        ),
    )
    resolved = resolve_spec(replace(spec, var_block=var_block))
    assert resolved.var_block is None
    assert resolved.hb_two_tone.fundamental_freq_1 == 3.1e9
    assert resolved.hb_two_tone.fundamental_freq_2 == 6.2e9


def test_resolve_spec_mixed_literal_and_named_hb2_freqs():
    """A literal freq passes through while a named one is substituted."""

    spec = _hb2_spec(hb_two_tone=_hb2(f1=3.0e9, f2="f_pump"))
    var_block = VarBlockSpec(
        name="VAR1", variables=(VarEntry(name="f_pump", value=6.2e9),)
    )
    resolved = resolve_spec(replace(spec, var_block=var_block))
    assert resolved.hb_two_tone.fundamental_freq_1 == 3.0e9
    assert resolved.hb_two_tone.fundamental_freq_2 == 6.2e9


