"""Schema + validator tests for the ``P1ToneSource``.

:class:`P1ToneSource` is an ADS ``P_1Tone`` power source -- a port-like
stimulus stored on ``DeviceSpec.sources``. These tests cover the
dataclass round-trip and the ``validate_p1tone_source`` /
``validate_device_spec`` invariants, including the ``float | str``
named-reference path for ``freq`` / ``power_dBm``.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tvtl_ads_builder_v2.device_spec import (
    DeviceSpec,
    DeviceSpecError,
    P1ToneSource,
    resolve_spec,
    validate_device_spec,
    validate_p1tone_source,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_two_tone_p1tone_sources,
)


def _src(**overrides) -> P1ToneSource:
    """A well-formed literal-valued P1ToneSource, with optional overrides."""

    base = dict(
        name="SIGNAL_SRC",
        port_node="n_in",
        num=1,
        freq=3.0e9,
        power_dBm=-30.0,
        z_ohm=50.0,
    )
    base.update(overrides)
    return P1ToneSource(**base)


# --- round-trip serialization -------------------------------------------


def test_p1tonesource_literal_roundtrip():
    src = _src()
    assert P1ToneSource.from_dict(src.to_dict()) == src


def test_p1tonesource_named_roundtrip():
    """A string-valued freq / power_dBm round-trips verbatim."""

    src = _src(freq="f_signal", power_dBm="signal_power_dBm")
    restored = P1ToneSource.from_dict(json.loads(json.dumps(src.to_dict())))
    assert restored == src
    assert restored.freq == "f_signal"
    assert restored.power_dBm == "signal_power_dBm"


def test_p1tonesource_default_z_ohm():
    """z_ohm defaults to v1's 50 Ohm when omitted."""

    src = P1ToneSource(
        name="S", port_node="n_in", num=1, freq=3.0e9, power_dBm=-30.0
    )
    assert src.z_ohm == 50.0


def test_devicespec_sources_dict_roundtrip():
    spec = make_two_tone_p1tone_sources(label="p1tone_schema_test")
    assert DeviceSpec.from_dict(spec.to_dict()) == spec


def test_devicespec_sources_json_roundtrip():
    spec = make_two_tone_p1tone_sources(label="p1tone_schema_test")
    restored = DeviceSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.sources == spec.sources


def test_devicespec_without_sources_omits_key():
    spec = make_two_tone_p1tone_sources(label="t")
    no_sources = replace(spec, sources=())
    assert "sources" not in no_sources.to_dict()
    assert DeviceSpec.from_dict(no_sources.to_dict()) == no_sources


# --- validate_p1tone_source (intrinsic rules) ---------------------------


def test_validate_source_accepts_literal_form():
    validate_p1tone_source(_src())


def test_validate_source_accepts_named_form():
    validate_p1tone_source(
        _src(freq="f_signal", power_dBm="signal_power_dBm"),
        frozenset({"f_signal", "signal_power_dBm"}),
    )


def test_validate_source_rejects_nonpositive_impedance():
    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(z_ohm=0.0))
    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(z_ohm=-50.0))


def test_validate_source_rejects_nonpositive_literal_freq():
    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(freq=0.0))
    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(freq=-3.0e9))


def test_validate_source_rejects_nonfinite_freq():
    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(freq=float("inf")))


def test_validate_source_rejects_nonfinite_power():
    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(power_dBm=float("nan")))


def test_validate_source_accepts_negative_power_dbm():
    """A negative power in dBm is ordinary -- dBm is a logarithmic scale."""

    validate_p1tone_source(_src(power_dBm=-30.0))


def test_validate_source_rejects_num_below_one():
    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(num=0))


def test_validate_source_rejects_bad_name():
    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(name="1bad"))


def test_validate_source_rejects_named_ref_to_undefined_variable():
    """A string freq naming no declared variable is rejected."""

    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(freq="f_signal"), frozenset({"other"}))
    # an empty var-name set (no VAR block) rejects a string freq too.
    with pytest.raises(DeviceSpecError):
        validate_p1tone_source(_src(freq="f_signal"))


# --- validate_device_spec with sources ----------------------------------


def test_validate_device_spec_accepts_wellformed_sources():
    validate_device_spec(make_two_tone_p1tone_sources(label="t"))


def test_validate_device_spec_rejects_duplicate_source_name():
    spec = make_two_tone_p1tone_sources(label="t")
    dup = replace(spec.sources[1], name=spec.sources[0].name)
    with pytest.raises(DeviceSpecError):
        validate_device_spec(replace(spec, sources=(spec.sources[0], dup)))


def test_validate_device_spec_rejects_source_port_node_undeclared():
    spec = make_two_tone_p1tone_sources(label="t")
    bad = replace(spec.sources[0], port_node="n_missing")
    with pytest.raises(DeviceSpecError):
        validate_device_spec(replace(spec, sources=(bad, spec.sources[1])))


def test_validate_device_spec_rejects_source_on_ground_node():
    """A source may not drive a ground node."""

    spec = make_two_tone_p1tone_sources(label="t")
    bad = replace(spec.sources[0], port_node="n_out")
    grounded = replace(
        spec, sources=(bad, spec.sources[1]), ground_nodes=("n_out",)
    )
    with pytest.raises(DeviceSpecError):
        validate_device_spec(grounded)


def test_validate_device_spec_rejects_source_named_ref_undeclared():
    """A source freq naming no VAR variable is rejected by the spec check."""

    spec = make_two_tone_p1tone_sources(label="t")
    bad = replace(spec.sources[0], freq="f_missing")
    with pytest.raises(DeviceSpecError):
        validate_device_spec(replace(spec, sources=(bad, spec.sources[1])))


def test_validate_device_spec_accepts_resolved_sources():
    """After resolution every source field is a literal and still valid."""

    resolved = resolve_spec(make_two_tone_p1tone_sources(label="t"))
    assert resolved.var_block is None
    validate_device_spec(resolved)
    for source in resolved.sources:
        assert isinstance(source.freq, float)
        assert isinstance(source.power_dBm, float)


