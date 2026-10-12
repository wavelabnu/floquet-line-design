"""Tests that the OA emitter places components explicitly and routes
shunt branches orthogonally.

``emit_workspace_from_spec`` must give every component an explicit ADS
coordinate -- otherwise the ADS schematic is the "correct but
uninspectable" overlap the placement work set out to fix -- and it must
align each shunt branch so ADS auto-routes the tap and ground drop
wires as clean vertical segments instead of diagonals.

These tests run with **no ADS installation**: a fake ``keysight.ads.de``
module is injected into ``sys.modules``, and a recorder design captures
every ``add_instance`` call. Pin coordinates are reconstructed from the
recorded origins plus the measured ``_PIN_OFFSETS`` table, so alignment
can be checked exactly without a real ADS workspace.
"""

from __future__ import annotations

import sys
import types

import pytest

from tvtl_ads_builder_v2.layout import compute_layout
from tvtl_ads_builder_v2.oa.oa_emitter import (
    ADS_HORIZONTAL_OFFSET,
    ADS_UNIT_PER_LAYOUT_UNIT,
    ADS_VERTICAL_OFFSET,
    _pin_offset,
    emit_workspace_from_spec,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    LEVEL_3_CANONICAL_CIRCUITS,
    LEVEL_4_CANONICAL_CIRCUITS,
)

# ADS cell name -> terminal count, so the fake instance exposes the
# right number of pins for the emitter's connectivity walk.
_PIN_COUNT = {
    "TermG": 1, "MLIN": 2, "MTEE_ADS": 3, "R": 2, "L": 2, "C": 2,
    "GROUND": 1, "MSUB": 1, "S_Param": 2,
}


# ---------------------------------------------------------------------------
# Fake keysight.ads.de -- just enough surface for emit_workspace_from_spec
# ---------------------------------------------------------------------------


class _FakePin:
    def __init__(self) -> None:
        self.net = None                    # GROUND pin starts unbound


class _FakeNet:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeInstance:
    """Records the coordinate it was placed at."""

    def __init__(self, cell, position, name) -> None:
        self.cell = cell
        self.position = position           # the add_instance (x, y) argument
        self.name = name
        self.parameters: dict = {}         # empty -> setp() is a no-op
        self.inst_pins = [
            _FakePin() for _ in range(_PIN_COUNT.get(cell[1], 2))
        ]


class _RecorderDesign:
    """A fake schematic design that records every placed instance."""

    # the emitter reads design.default_wire_layer when placing labels.
    default_wire_layer = "fake_wire_layer"

    def __init__(self) -> None:
        self.instances: list[_FakeInstance] = []
        self._nets: dict[str, _FakeNet] = {}

    def add_instance(self, cell, position, *, name, angle):
        inst = _FakeInstance(cell, position, name)
        self.instances.append(inst)
        return inst

    def find_or_add_net(self, name):
        return self._nets.setdefault(name, _FakeNet(name))

    def add_wire(self, _points):
        pass

    def add_text(self, *_args):
        pass

    def save_design(self):
        pass

    def generate_netlist(self):
        return "* fake ADS netlist\n"

    def by_name(self, name: str) -> _FakeInstance:
        return next(i for i in self.instances if i.name == name)


class _FakeWorkspace:
    def open(self):
        pass

    def add_library(self, *_a):
        pass

    def open_library(self, *_a):
        return _FakeLibrary()


class _FakeLibrary:
    def setup_schematic_tech(self):
        pass


def _emit_with_fakes(spec, tmp_path, monkeypatch) -> _RecorderDesign:
    """Run ``emit_workspace_from_spec`` against a fake ADS; return the design."""

    design = _RecorderDesign()

    db_uu = types.ModuleType("keysight.ads.de.db_uu")
    db_uu.create_schematic = lambda _name: design

    de = types.ModuleType("keysight.ads.de")
    de.db_uu = db_uu
    de.workspace_is_open = lambda: False
    de.close_workspace = lambda: None
    de.create_workspace = lambda _path: _FakeWorkspace()
    de.create_new_library = lambda *_a: None
    de.LibraryMode = types.SimpleNamespace(NON_SHARED="non_shared")

    ads = types.ModuleType("keysight.ads")
    ads.de = de
    keysight = types.ModuleType("keysight")
    keysight.ads = ads

    monkeypatch.setitem(sys.modules, "keysight", keysight)
    monkeypatch.setitem(sys.modules, "keysight.ads", ads)
    monkeypatch.setitem(sys.modules, "keysight.ads.de", de)
    monkeypatch.setitem(sys.modules, "keysight.ads.de.db_uu", db_uu)

    emit_workspace_from_spec(spec, tmp_path / "oa_workspace")
    return design


def _expected_ads(layout_xy: tuple[float, float]) -> tuple[float, float]:
    """Independent layout-unit -> ADS-coordinate conversion."""

    x, y = layout_xy
    return (
        ADS_HORIZONTAL_OFFSET + x * ADS_UNIT_PER_LAYOUT_UNIT,
        ADS_VERTICAL_OFFSET - y * ADS_UNIT_PER_LAYOUT_UNIT,
    )


def _pin_x(design: _RecorderDesign, name: str, etype: str, device_pin: int) -> float:
    """ADS x of one device pin: recorded origin + measured pin offset."""

    return design.by_name(name).position[0] + _pin_offset(etype, device_pin)[0]


def _by_label(circuits, label):
    return next(c for c in circuits if c.label == label)


# ---------------------------------------------------------------------------
# Explicit placement
# ---------------------------------------------------------------------------


def test_components_placed_at_compute_layout_coordinates(tmp_path, monkeypatch):
    """Every Level-1 component lands at its compute_layout coordinate.

    A Level-1 two-port has no shunt branch, so no alignment shift
    applies and the placed coordinates are exactly the scaled layout.
    """

    spec = LEVEL_1_CANONICAL_CIRCUITS[0]
    design = _emit_with_fakes(spec, tmp_path, monkeypatch)
    coords = compute_layout(spec)

    for element in spec.elements:
        inst = design.by_name(element.id)
        assert inst.position == pytest.approx(_expected_ads(coords[element.id]))


def test_shunt_component_sits_below_the_through_row(tmp_path, monkeypatch):
    """A Level-3 stub is placed below (negative ADS y) the through row."""

    spec = _by_label(LEVEL_3_CANONICAL_CIRCUITS, "tee_short_zero_stub")
    design = _emit_with_fakes(spec, tmp_path, monkeypatch)

    for through_id in ("LINE_LEFT", "TEE1", "LINE_RIGHT"):
        assert design.by_name(through_id).position[1] == ADS_VERTICAL_OFFSET
    # the shunt stub hangs below the through-path row.
    assert design.by_name("STUB").position[1] < ADS_VERTICAL_OFFSET


def test_support_instances_keep_fixed_positions(tmp_path, monkeypatch):
    """MSUB and the S_Param controller are parked, not laid out."""

    spec = LEVEL_1_CANONICAL_CIRCUITS[0]
    design = _emit_with_fakes(spec, tmp_path, monkeypatch)

    assert design.by_name(spec.substrate.name).position == (-8.0, 4.0)
    assert design.by_name("SP1").position == (-8.0, -5.0)


# ---------------------------------------------------------------------------
# Orthogonal shunt routing
# ---------------------------------------------------------------------------


def test_shunt_input_pin_aligns_under_the_tap_pin(tmp_path, monkeypatch):
    """A single-MLIN shunt stub's input pin is x-aligned with the tap pin.

    The tap is the MTEE branch pin (device pin 3); the shunt MLIN's
    input pin is its device pin 1. Equal x means ADS routes the tap
    drop as a clean vertical wire.
    """

    spec = _by_label(LEVEL_3_CANONICAL_CIRCUITS, "tee_short_zero_stub")
    design = _emit_with_fakes(spec, tmp_path, monkeypatch)

    tap_pin_x = _pin_x(design, "TEE1", "MTEE", 3)
    shunt_in_x = _pin_x(design, "STUB", "MLIN", 1)
    assert shunt_in_x == pytest.approx(tap_pin_x)


def test_ground_aligns_under_the_shunt_exit_pin(tmp_path, monkeypatch):
    """The GROUND symbol is x-aligned with the shunt's real exit pin.

    The shunt MLIN's exit pin is its device pin 2; the GROUND symbol's
    pin sits at its placed origin. Equal x means the ground drop wire
    is a clean vertical, and GROUND sits below the shunt.
    """

    spec = _by_label(LEVEL_3_CANONICAL_CIRCUITS, "tee_short_zero_stub")
    design = _emit_with_fakes(spec, tmp_path, monkeypatch)

    exit_pin_x = _pin_x(design, "STUB", "MLIN", 2)
    ground = design.by_name("GND1")
    assert ground.position[0] == pytest.approx(exit_pin_x)
    assert ground.position[1] < design.by_name("STUB").position[1]


def test_rlc_shunt_branch_is_horizontal_and_aligned(tmp_path, monkeypatch):
    """An RLC_SERIES shunt branch routes orthogonally end to end.

    The branch expands to R/L/C. They share one y (horizontal row) at
    uniform spacing, the first primitive's input pin is x-aligned with
    the MTEE tap pin, and GROUND is x-aligned with the last primitive's
    exit pin.
    """

    spec = _by_label(LEVEL_4_CANONICAL_CIRCUITS, "mtee_with_rlc_stub")
    design = _emit_with_fakes(spec, tmp_path, monkeypatch)
    primitives = [design.by_name(f"STUB_{kind}") for kind in ("R", "L", "C")]

    # horizontal row -- one shared y, distinct xs, uniform spacing.
    ys = {p.position[1] for p in primitives}
    assert len(ys) == 1
    xs = [p.position[0] for p in primitives]
    assert len(set(xs)) == 3
    assert (xs[1] - xs[0]) == pytest.approx(xs[2] - xs[1])

    # the branch's first primitive (R, device pin 1) is under the tap.
    tap_pin_x = _pin_x(design, "TEE1", "MTEE", 3)
    assert _pin_x(design, "STUB_R", "R", 1) == pytest.approx(tap_pin_x)

    # GROUND is under the last primitive's exit pin (C, device pin 2).
    exit_pin_x = _pin_x(design, "STUB_C", "C", 2)
    assert design.by_name("GND1").position[0] == pytest.approx(exit_pin_x)


