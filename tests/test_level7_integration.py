"""Level-7 integration test: the full two-tone-HB TVTL unit cell.

``level7_tvtl_unit_cell_two_tone_hb`` composes every element layer at
once -- a VAR block, two ``P_1Tone`` sources, a two-tone HB controller
and the SDD2P unit-cell topology. This test builds that spec and emits
it through both backends, ADS-free, verifying the var-resolver
substitutes every named reference to one consistent literal that both
backends then carry identically.

The OA backend is exercised against a stub ``keysight.ads.de``; the
real ADS cross-check (Path A on ADS) is the user's verification step.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from tvtl_ads_builder_v2.device_spec import resolve_spec, validate_device_spec
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa import oa_emitter
from tvtl_ads_builder_v2.oa.hb_controller_content_check import (
    check_hb_two_tone_content,
)
from tvtl_ads_builder_v2.oa.p1tone_content_check import check_p1tone_content
from tvtl_ads_builder_v2.oa.sdd2p_content_check import check_sdd2p_content
from tvtl_ads_builder_v2.oa.var_content_check import check_var_content
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_tvtl_unit_cell_two_tone_hb,
)


def _spec():
    return make_tvtl_unit_cell_two_tone_hb(
        label="level7_tvtl_unit_cell_two_tone_hb"
    )


# --- the spec composes and validates ------------------------------------


def test_level7_spec_validates():
    """The integrated Level-7 spec passes the device-spec validator."""

    validate_device_spec(_spec())


def test_level7_spec_carries_every_phase5_layer():
    """One spec carries the VAR block, sources, HB controller and SDD2P."""

    spec = _spec()
    assert spec.var_block is not None
    assert len(spec.sources) == 2
    assert spec.hb_two_tone is not None
    assert any(e.type == "SDD2P" for e in spec.elements)


# --- the resolver substitutes every named reference ---------------------


def test_level7_resolver_substitutes_every_named_reference():
    """resolve_spec turns every named reference into a literal.

    A single ``f_signal`` declaration drives both the HB controller's
    ``fundamental_freq_1`` and the signal source's ``freq``; ``f_pump``
    drives ``fundamental_freq_2`` and the pump source's ``freq``;
    ``pump_power_dBm`` drives the pump source's ``power_dBm``; and the
    RLC stub's R/L/C name VAR variables. After resolution the VAR block
    is gone and every one of those fields is a literal number.
    """

    resolved = resolve_spec(_spec())
    assert resolved.var_block is None

    # HB controller frequencies -> literals.
    assert resolved.hb_two_tone.fundamental_freq_1 == 3.0e9
    assert resolved.hb_two_tone.fundamental_freq_2 == 6.0e9

    # source freq / power -> literals.
    by_name = {s.name: s for s in resolved.sources}
    assert by_name["SIGNAL_SRC"].freq == 3.0e9
    assert by_name["SIGNAL_SRC"].power_dBm == -30.0
    assert by_name["PUMP_SRC"].freq == 6.0e9
    assert by_name["PUMP_SRC"].power_dBm == 5.0

    # the RLC stub's element parameters -> literals.
    stub = next(e for e in resolved.elements if e.id == "STUB")
    assert stub.parameters["r_ohm"] == 2.0
    assert stub.parameters["l_h"] == 2.0e-9
    assert stub.parameters["c_f"] == 4.0e-12


def test_level7_one_var_drives_controller_and_source_together():
    """f_signal / f_pump reach the HB controller and the sources alike."""

    resolved = resolve_spec(_spec())
    by_name = {s.name: s for s in resolved.sources}
    assert (
        resolved.hb_two_tone.fundamental_freq_1
        == by_name["SIGNAL_SRC"].freq
    )
    assert (
        resolved.hb_two_tone.fundamental_freq_2
        == by_name["PUMP_SRC"].freq
    )


# --- emission through the hpeesofsim backend ----------------------------


def test_level7_hpeesofsim_deck_carries_only_literals():
    """The emitted deck carries the resolved literals, no VAR names."""

    netlist = emit_hpeesofsim_netlist(_spec())
    for name in (
        "f_signal", "f_pump", "signal_power_dBm", "pump_power_dBm",
        "R_var", "L_par", "C_var",
    ):
        assert name not in netlist, f"unresolved VAR name {name!r} in deck"
    # the composed directives are all present.
    assert "P_1Tone:SIGNAL_SRC n_input 0" in netlist
    assert "P_1Tone:PUMP_SRC n_input 0" in netlist
    assert "HB:HB1 Freq[1]=3 GHz Freq[2]=6 GHz" in netlist
    assert "SDD:VARACTOR" in netlist


def test_level7_emission_is_deterministic():
    spec = _spec()
    assert emit_hpeesofsim_netlist(spec) == emit_hpeesofsim_netlist(spec)


# --- every content cross-check matches ----------------------------------


def test_level7_all_content_checks_match():
    """Every element-layer content check MATCHes on the resolved Level-7 deck.

    Feeding the resolved hpeesofsim deck as both backends exercises the
    cross-backend invariant: a resolver that substituted a named
    reference to two different literals would make at least one check
    diverge.
    """

    spec = _spec()
    netlist = emit_hpeesofsim_netlist(spec)
    kw = dict(hpeesofsim_netlist_text=netlist, ads_netlist_text=netlist)

    assert check_p1tone_content(spec, **kw).match
    assert check_hb_two_tone_content(spec, **kw).match
    assert check_var_content(spec, **kw).match
    assert check_sdd2p_content(spec, **kw).match


# --- emission through the OA backend (stubbed) --------------------------


def _install_fake_keysight(monkeypatch):
    """Install a minimal recording stub for ``keysight.ads.de``."""

    pin_count = {
        "TermG": 1, "MLIN": 2, "MTEE_ADS": 3, "MSUB": 0, "GROUND": 1,
        "S_Param": 0, "R": 2, "L": 2, "C": 2, "SDD2P": 4,
        "HarmonicBalance": 0, "P_1Tone": 2,
    }

    class _Bbox:
        x1 = y1 = -0.1
        x2 = y2 = 0.1

    class _Pin:
        def __init__(self):
            self.net = None
            self.bbox = _Bbox()

    class _Inst:
        def __init__(self, n):
            self.parameters = {}
            self.inst_pins = [_Pin() for _ in range(n)]

    class _Design:
        default_wire_layer = "fake"

        def __init__(self):
            self.instances = {}

        def add_instance(self, cell, coord, name=None, angle=0):
            inst = _Inst(pin_count.get(cell[1], 0))
            self.instances[name] = inst
            return inst

        def find_or_add_net(self, name):
            return f"net::{name}"

        def add_wire(self, points):
            return None

        def save_design(self):
            return None

        def generate_netlist(self):
            return "; stub ADS netlist\n"

    class _Library:
        def setup_schematic_tech(self):
            return None

    class _Workspace:
        def open(self):
            return None

        def add_library(self, *a, **k):
            return None

        def open_library(self, *a, **k):
            return _Library()

    class _DbUu:
        def create_schematic(self, ref):
            return _Design()

    de = types.ModuleType("keysight.ads.de")
    de.workspace_is_open = lambda: False
    de.close_workspace = lambda: None
    de.create_workspace = lambda path: _Workspace()
    de.create_new_library = lambda name, path: None
    de.LibraryMode = types.SimpleNamespace(NON_SHARED="NON_SHARED")
    de.db_uu = _DbUu()
    ks = types.ModuleType("keysight")
    ads = types.ModuleType("keysight.ads")
    ks.ads = ads
    ads.de = de
    monkeypatch.setitem(sys.modules, "keysight", ks)
    monkeypatch.setitem(sys.modules, "keysight.ads", ads)
    monkeypatch.setitem(sys.modules, "keysight.ads.de", de)


def test_level7_oa_backend_builds_and_emits_the_hpeesofsim_deck(
    tmp_path: Path, monkeypatch
) -> None:
    """The OA backend builds the Level-7 workspace against the stub.

    ``emit_workspace_from_spec`` places every element, both ``P_1Tone``
    sources, the HB controller and the SDD2P varactor without an ADS
    runtime, and writes the same resolved hpeesofsim deck the flat
    backend produces.
    """

    _install_fake_keysight(monkeypatch)
    spec = _spec()
    info = oa_emitter.emit_workspace_from_spec(
        spec, tmp_path / spec.label / "oa_workspace"
    )
    deck = Path(info["hpeesofsim_netlist_path"]).read_text(encoding="utf-8")
    assert deck == emit_hpeesofsim_netlist(spec)


