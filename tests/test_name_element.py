"""Tests for the NAME element platform support.

The NAME element is a 1-pin spec-level annotation declaring "expose
this net's voltage in the HB ``.ds`` dataset." Validator, hpeesofsim
emitter, and (transparent) interaction with layout / TVTL simulator
are all ADS-free and covered here. The OA-side rename pass and its
MATCH-preserving behaviour against the Level-9 canonical are exercised
by the integration test ``tests/parity/test_level9_name_match.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from tvtl_ads_builder_v2.device_spec import Element, ElementPin, Node
from tvtl_ads_builder_v2.device_spec.validator import (
    DeviceSpecError,
    validate_device_spec,
)
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.netlist.elements import emit_name
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_9_CANONICAL_CIRCUITS,
)


_LEVEL_9_SPEC_WITH_NAMES = LEVEL_9_CANONICAL_CIRCUITS[0]

# The Level-9 canonicals carry NAME annotations.
# The unit tests below construct in-memory NAME variants and would
# duplicate NAMEs if they appended to the annotated spec; strip the
# canonical's NAMEs to get a clean baseline for the helpers.
_LEVEL_9_SPEC = dataclasses.replace(
    _LEVEL_9_SPEC_WITH_NAMES,
    elements=tuple(
        e for e in _LEVEL_9_SPEC_WITH_NAMES.elements if e.type != "NAME"
    ),
)


def _spec_with_names(target_nets: list[str]):
    """Return a Level-9 spec with one NAME element per target net.

    The base spec is the canonical with its own NAMEs
    stripped (see ``_LEVEL_9_SPEC``); the tests then add the NAMEs
    they want to exercise without colliding with the canonical's own.
    """

    new_elems = list(_LEVEL_9_SPEC.elements)
    for idx, net in enumerate(target_nets):
        new_elems.append(
            Element(
                id=f"NAME_TEST_{idx:02d}_{net}",
                type="NAME",
                pins=(ElementPin(pin=1, node=net),),
                parameters={},
            )
        )
    return dataclasses.replace(_LEVEL_9_SPEC, elements=tuple(new_elems))


# --------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------


def test_name_element_validates_with_single_pin() -> None:
    spec = _spec_with_names(["n_input"])
    validate_device_spec(spec)  # no raise


def test_name_element_rejected_with_zero_pins() -> None:
    bad = Element(id="BAD", type="NAME", pins=(), parameters={})
    spec = dataclasses.replace(
        _LEVEL_9_SPEC,
        elements=(*_LEVEL_9_SPEC.elements, bad),
    )
    with pytest.raises(DeviceSpecError, match="must contain at least one"):
        validate_device_spec(spec)


def test_name_element_rejected_with_two_pins() -> None:
    bad = Element(
        id="BAD", type="NAME",
        pins=(ElementPin(pin=1, node="n_input"), ElementPin(pin=2, node="n_pin")),
        parameters={},
    )
    spec = dataclasses.replace(
        _LEVEL_9_SPEC,
        elements=(*_LEVEL_9_SPEC.elements, bad),
    )
    with pytest.raises(DeviceSpecError, match="must declare exactly one pin"):
        validate_device_spec(spec)


def test_name_element_pin_node_must_be_declared() -> None:
    """A NAME pin pointing at an undeclared net is rejected by the
    generic pin-node-declared check (same rule as every other element).
    """

    bad = Element(
        id="BAD", type="NAME",
        pins=(ElementPin(pin=1, node="n_does_not_exist"),),
        parameters={},
    )
    spec = dataclasses.replace(
        _LEVEL_9_SPEC,
        elements=(*_LEVEL_9_SPEC.elements, bad),
    )
    with pytest.raises(DeviceSpecError, match="not declared in nodes"):
        validate_device_spec(spec)


def test_name_uniqueness_rejects_two_names_on_one_net() -> None:
    """At most one NAME element per net -- two NAMEs on n_input fail."""

    twin1 = Element(
        id="NAME_A", type="NAME",
        pins=(ElementPin(pin=1, node="n_input"),), parameters={},
    )
    twin2 = Element(
        id="NAME_B", type="NAME",
        pins=(ElementPin(pin=1, node="n_input"),), parameters={},
    )
    spec = dataclasses.replace(
        _LEVEL_9_SPEC,
        elements=(*_LEVEL_9_SPEC.elements, twin1, twin2),
    )
    with pytest.raises(
        DeviceSpecError,
        match="at most one NAME element is allowed per net",
    ):
        validate_device_spec(spec)


def test_name_uniqueness_allows_one_name_per_net() -> None:
    """Different NAMEs on different nets are fine -- the multi case."""

    spec = _spec_with_names([
        "n_input", "n_pin", "n_c1_v", "n_c1_b", "n_c1_out",
        "n_pc1c2", "n_c2_v", "n_c2_b", "n_c2_out", "n_output",
    ])
    validate_device_spec(spec)  # no raise


# --------------------------------------------------------------------------
# Hpeesofsim flat-deck emitter -- comment-only
# --------------------------------------------------------------------------


def test_emit_name_returns_a_comment_line() -> None:
    """The hpeesofsim NAME emitter writes a single `;`-prefixed comment."""

    elem = Element(
        id="NAME_INPUT", type="NAME",
        pins=(ElementPin(pin=1, node="n_input"),), parameters={},
    )
    line = emit_name(elem)
    assert line.startswith(";")
    assert "NAME_INPUT" in line
    assert "n_input" in line


def test_emit_name_resolves_ground_node_via_resolver() -> None:
    """The resolver translates a ground node to the literal ``0``."""

    elem = Element(
        id="NAME_GND", type="NAME",
        pins=(ElementPin(pin=1, node="n_c1_g"),), parameters={},
    )
    line = emit_name(elem, resolve_node=lambda n: "0" if n == "n_c1_g" else n)
    assert "NAME_GND" in line
    assert " 0" in line  # the resolved ground


def test_emit_name_rejects_wrong_type() -> None:
    elem = Element(
        id="WRONG", type="MLIN",
        pins=(ElementPin(pin=1, node="n_input"),), parameters={},
    )
    with pytest.raises(ValueError, match="emit_name called"):
        emit_name(elem)


def test_emit_name_rejects_wrong_pin_count() -> None:
    elem = Element(
        id="WRONG", type="NAME",
        pins=(ElementPin(pin=1, node="n_input"), ElementPin(pin=2, node="n_pin")),
        parameters={},
    )
    with pytest.raises(ValueError, match="exactly one pin"):
        emit_name(elem)


# --------------------------------------------------------------------------
# Flat-deck integration -- NAME comments appear, electrical content
# unchanged
# --------------------------------------------------------------------------


def test_flat_deck_with_names_carries_comments_and_no_electrical_lines() -> None:
    """Adding NAME elements adds comment lines only -- no element-type
    line per NAME. The Level-9 baseline electrical content is unchanged.
    """

    targets = ["n_input", "n_pin", "n_c1_v"]
    spec_with = _spec_with_names(targets)
    deck_with = emit_hpeesofsim_netlist(spec_with)
    deck_base = emit_hpeesofsim_netlist(_LEVEL_9_SPEC)

    # The with-NAME deck has exactly one extra "; NAME annotation"
    # comment per NAME element.
    base_comments = [
        l for l in deck_base.splitlines() if l.startswith("; NAME annotation")
    ]
    with_comments = [
        l for l in deck_with.splitlines() if l.startswith("; NAME annotation")
    ]
    assert len(base_comments) == 0
    assert len(with_comments) == len(targets)

    # Stripping NAME comments, the two decks have identical electrical
    # content (same non-comment lines in the same order).
    def _strip_name_comments(text: str) -> list[str]:
        return [
            l for l in text.splitlines()
            if not l.startswith("; NAME annotation")
        ]
    assert _strip_name_comments(deck_with) == _strip_name_comments(deck_base)


# --------------------------------------------------------------------------
# Layout / topology -- NAME is invisible to the layout pass
# --------------------------------------------------------------------------


def test_layout_skips_name_elements() -> None:
    """``build_layout`` treats NAME elements as having no footprint."""

    from tvtl_ads_builder_v2.layout import build_layout

    spec_with = _spec_with_names(["n_c1_b", "n_c1_v", "n_c2_b", "n_c2_v"])
    layout = build_layout(spec_with)
    placed_ids = {p.element.id for p in layout.placements}
    # No NAME id ends up in the layout.
    assert not any(pid.startswith("NAME_") for pid in placed_ids)
    # Every non-NAME spec element is placed.
    expected = {e.id for e in _LEVEL_9_SPEC.elements}
    assert expected.issubset(placed_ids)


def test_topology_svg_renderer_handles_name() -> None:
    """The topology SVG renderer does not warn or fail on NAME elements."""

    from tvtl_ads_builder_v2.topology import render_topology_svg
    import tempfile

    spec_with = _spec_with_names(["n_input", "n_pin"])
    with tempfile.NamedTemporaryFile(
        suffix=".svg", delete=False, mode="w"
    ) as tmp:
        tmp_path = tmp.name
    try:
        render_topology_svg(spec_with, tmp_path)
        # The renderer must produce a non-empty SVG.
        from pathlib import Path
        assert Path(tmp_path).read_text().startswith("<?xml") or \
            "<svg" in Path(tmp_path).read_text()
    finally:
        from pathlib import Path
        Path(tmp_path).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# RECOGNIZED_ELEMENT_TYPES surface
# --------------------------------------------------------------------------


def test_name_in_recognized_element_types() -> None:
    from tvtl_ads_builder_v2.device_spec.validator import (
        _RECOGNIZED_ELEMENT_TYPES,
    )
    assert "NAME" in _RECOGNIZED_ELEMENT_TYPES


def test_name_in_element_emitters() -> None:
    """NAME is registered in the hpeesofsim emitter dispatch table."""

    from tvtl_ads_builder_v2.netlist.emit import ELEMENT_EMITTERS
    assert "NAME" in ELEMENT_EMITTERS


