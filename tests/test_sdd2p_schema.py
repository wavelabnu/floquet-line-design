"""Schema + validator tests for the ``SDD2P`` element type.

The ``SDD2P`` symbolically-defined device follows the existing
type-string + parameters-dict pattern: ``Element.type == "SDD2P"`` with
``q_poly_coefficients`` (required) and ``q_poly_reference`` (optional) in
the parameters dict. These tests cover round-trip serialization and the
new validator invariants.
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
    dump_device_spec,
    load_device_spec,
    validate_device_spec,
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
        "rough_m": 0.0,
        "mur": 1.0,
    },
)

_SAMPLE_POLY = (0.0, 7.0e-12, -1.0e-12, 0.0, 0.0, 0.0, 0.0)


def _sdd2p_spec(
    *,
    q_poly_coefficients=_SAMPLE_POLY,
    q_poly_reference: str | None = "sample_at_vdc_0",
) -> DeviceSpec:
    """Build a minimal ``P1 -- SDD2P -- GND`` device spec."""

    parameters: dict[str, object] = {"q_poly_coefficients": q_poly_coefficients}
    if q_poly_reference is not None:
        parameters["q_poly_reference"] = q_poly_reference
    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(start=1.0e9, stop=10.0e9, points=91),
        port_impedance_ohm=50.0,
    )
    nodes = (
        Node(id="n_in"),
        Node(id="n_gnd"),
        Node(id="n_ret_p"),
        Node(id="n_ret_n"),
    )
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": 50.0, "num": 1},
    )
    sdd = Element(
        id="VAR1",
        type="SDD2P",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_gnd"),
            ElementPin(pin=3, node="n_ret_p"),
            ElementPin(pin=4, node="n_ret_n"),
        ),
        parameters=parameters,
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=_SUBSTRATE,
        nodes=nodes,
        elements=(p1, sdd),
        ports=(Port(num=1, node="n_in"),),
        label="sdd2p_schema_sample",
        ground_nodes=("n_gnd",),
        open_terminations=("n_ret_p", "n_ret_n"),
    )


def test_sdd2p_spec_validates() -> None:
    """A well-formed SDD2P spec passes the validator."""

    validate_device_spec(_sdd2p_spec())


def test_sdd2p_round_trip_serialization() -> None:
    """An SDD2P spec round-trips through to_dict/from_dict and JSON."""

    spec = _sdd2p_spec()
    rebuilt = DeviceSpec.from_dict(spec.to_dict())
    validate_device_spec(rebuilt)

    # The element survives the round-trip with its polynomial intact.
    sdd = next(e for e in rebuilt.elements if e.type == "SDD2P")
    assert tuple(float(x) for x in sdd.parameters["q_poly_coefficients"]) == (
        _SAMPLE_POLY
    )
    assert sdd.parameters["q_poly_reference"] == "sample_at_vdc_0"

    # A full JSON round-trip is byte-stable.
    payload = json.dumps(spec.to_dict(), indent=2, sort_keys=True)
    from_json = DeviceSpec.from_dict(json.loads(payload))
    assert json.dumps(from_json.to_dict(), indent=2, sort_keys=True) == payload


def test_sdd2p_round_trip_through_disk(tmp_path) -> None:
    """An SDD2P spec round-trips through dump_device_spec/load_device_spec."""

    spec = _sdd2p_spec()
    path = dump_device_spec(spec, tmp_path / "sample.device_spec.json")
    loaded = load_device_spec(path)
    sdd = next(e for e in loaded.elements if e.type == "SDD2P")
    coeffs = tuple(float(x) for x in sdd.parameters["q_poly_coefficients"])
    assert coeffs == _SAMPLE_POLY


def test_sdd2p_disk_round_trip_preserves_equality(tmp_path) -> None:
    """A dumped-then-loaded SDD2P spec compares equal to the original.

    A JSON round-trip turns the tuple-valued ``q_poly_coefficients``
    into a list; ``Element.from_dict`` normalises it back to a tuple,
    so the loaded spec is an exact ``==`` match for the original and
    not merely coefficient-wise equal.
    """

    spec = _sdd2p_spec()
    path = dump_device_spec(spec, tmp_path / "sample.device_spec.json")
    assert load_device_spec(path) == spec


def test_sdd2p_reference_is_optional() -> None:
    """q_poly_reference may be omitted entirely."""

    spec = _sdd2p_spec(q_poly_reference=None)
    validate_device_spec(spec)
    sdd = next(e for e in spec.elements if e.type == "SDD2P")
    assert "q_poly_reference" not in sdd.parameters


def test_sdd2p_rejects_missing_coefficients() -> None:
    """An SDD2P with no q_poly_coefficients is rejected."""

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(start=1.0e9, stop=10.0e9, points=91),
        port_impedance_ohm=50.0,
    )
    nodes = (
        Node(id="n_in"),
        Node(id="n_gnd"),
        Node(id="n_ret_p"),
        Node(id="n_ret_n"),
    )
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": 50.0, "num": 1},
    )
    sdd = Element(
        id="VAR1",
        type="SDD2P",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_gnd"),
            ElementPin(pin=3, node="n_ret_p"),
            ElementPin(pin=4, node="n_ret_n"),
        ),
        parameters={},
    )
    spec = DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=_SUBSTRATE,
        nodes=nodes,
        elements=(p1, sdd),
        ports=(Port(num=1, node="n_in"),),
        label="sdd2p_missing_coeffs",
        ground_nodes=("n_gnd",),
        open_terminations=("n_ret_p", "n_ret_n"),
    )
    with pytest.raises(DeviceSpecError, match="q_poly_coefficients"):
        validate_device_spec(spec)


def test_sdd2p_rejects_empty_coefficients() -> None:
    """An empty q_poly_coefficients tuple is rejected."""

    with pytest.raises(DeviceSpecError, match="non-empty"):
        validate_device_spec(_sdd2p_spec(q_poly_coefficients=()))


def test_sdd2p_rejects_nonzero_c0() -> None:
    """A charge polynomial whose c0 is not 0.0 is rejected (Q(0) != 0)."""

    with pytest.raises(DeviceSpecError, match="c0"):
        validate_device_spec(
            _sdd2p_spec(q_poly_coefficients=(1.0e-13, 7.0e-12, -1.0e-12))
        )


def test_sdd2p_rejects_non_sequence_coefficients() -> None:
    """A scalar / string q_poly_coefficients is rejected -- must be a tuple."""

    with pytest.raises(DeviceSpecError, match="tuple/list"):
        validate_device_spec(_sdd2p_spec(q_poly_coefficients="0,7e-12"))


def test_sdd2p_rejects_wrong_pin_count() -> None:
    """An SDD2P with other than four pins is rejected (it is a 2-port)."""

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(start=1.0e9, stop=10.0e9, points=91),
        port_impedance_ohm=50.0,
    )
    nodes = (Node(id="n_in"), Node(id="n_gnd"))
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": 50.0, "num": 1},
    )
    sdd = Element(
        id="VAR1",
        type="SDD2P",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_gnd"),
        ),
        parameters={"q_poly_coefficients": _SAMPLE_POLY},
    )
    spec = DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=_SUBSTRATE,
        nodes=nodes,
        elements=(p1, sdd),
        ports=(Port(num=1, node="n_in"),),
        label="sdd2p_two_pin",
        ground_nodes=("n_gnd",),
    )
    with pytest.raises(DeviceSpecError, match="four pins"):
        validate_device_spec(spec)


