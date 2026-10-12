"""Cross-check tests for the OA-Python D0-D4 backend.

Every test runs without ADS. The netlist parser and comparator are
exercised with fixtures; the five OA emitters are exercised against a
stub ``keysight.ads.de`` (no ADS install needed); the CLI is exercised
with the emitter stubbed out. The real OA workspace construction is
verified by the user running the CLI on a machine with ADS.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from tvtl_ads_builder_v2.netlist.emit import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa import oa_diagnostic_cli, oa_emitter
from tvtl_ads_builder_v2.oa.netlist_compare import (
    compare_circuits,
    parse_ads_netlist,
    parse_hpeesofsim_deck,
    parse_netlist_text,
)
from tvtl_ads_builder_v2.parity.ael_diagnostics.specs import (
    DIAGNOSTIC_ORDER,
    DIAGNOSTIC_SPECS,
    make_d1_spec,
    make_d2_spec,
    make_d3_spec,
)

# The real ADS "Generate Netlist" output for the spec-valued D1 OA
# workspace -- used as the parser fixture (line continuations, the
# Options/OutputPlan/#load/Component directives) and, since it is the
# spec circuit, as the equivalent-to-hpeesofsim fixture.
ADS_D1_NETLIST = '''\
; Top Design: "d1_oa_lib:d1_oa_test:schematic"
; Netlisted using Hierarchy Policy: "Standard"

Options ResourceUsage=yes UseNutmegFormat=no TopDesignName="d1_oa_lib:d1_oa_test:schematic"
model msub1 MSUB H=1.54 mm Er=3.38 Mur=1 Cond=5.8e7 Hu=3.93701e+34 mil T=0.0175 mm TanD=0.0027 Rough=0 mm DielectricLossModel=1 RoughnessModel=2
Port:P1  n_in 0 Num=1 Z=50 Ohm Noise=no
MLIN2:LINE  n_out n_in Subst="msub1" W=2.95 mm L=2.5 mm Wall1=1.0E+30 mil Wall2=1.0E+30 mil Mod=1
Port:P2  n_out 0 Num=2 Z=50 Ohm Noise=no
S_Param:SP1 CalcS=yes CalcY=no CalcZ=no GroupDelayAperture=1e-4 StatusLevel=2 \\
SweepVar="freq" SweepPlan="SP1_stim" OutputPlan="SP1_Output"

SweepPlan: SP1_stim Start=1 GHz Stop=10 GHz Step=0.1 GHz

OutputPlan:SP1_Output \\
      Type="Output" \\
      UseEquationNestLevel=yes

#load "python","LinearCollapse"
Component Module="LinearCollapse" Type="ModelExtractor" NetworkRepresentation=2
'''


# --------------------------------------------------------------------------
# A stub keysight.ads.de so the OA emitter can run ADS-free.
# --------------------------------------------------------------------------

# inst_pins count by ADS cell name -- TermG 1, MLIN 2, MTEE_ADS 3, etc.
_FAKE_PIN_COUNT = {
    "TermG": 1, "MLIN": 2, "MTEE_ADS": 3, "MSUB": 0, "GROUND": 1, "S_Param": 0,
}


class _FakeBbox:
    x1 = y1 = -0.1
    x2 = y2 = 0.1


class _FakePin:
    def __init__(self) -> None:
        self.net = None
        self.bbox = _FakeBbox()


class _FakeInstance:
    def __init__(self, n_pins: int) -> None:
        self.parameters: dict = {}  # empty -> setp() is a no-op (no error)
        self.inst_pins = [_FakePin() for _ in range(n_pins)]


class _FakeDesign:
    default_wire_layer = "fake_wire_layer"

    def __init__(self) -> None:
        self.instances: dict = {}  # name -> _FakeInstance, for inspection

    def add_instance(self, cell, coord, name=None, angle=0):
        inst = _FakeInstance(_FAKE_PIN_COUNT.get(cell[1], 0))
        self.instances[name] = inst
        return inst

    def find_or_add_net(self, name):
        return f"net::{name}"

    def add_wire(self, points):
        return None

    def add_text(self, *args):
        return None

    def save_design(self):
        return None

    def generate_netlist(self):
        return "; stub ADS netlist\n"


class _FakeLibrary:
    def setup_schematic_tech(self):
        return None


class _FakeWorkspace:
    def open(self):
        return None

    def add_library(self, *args, **kwargs):
        return None

    def open_library(self, *args, **kwargs):
        return _FakeLibrary()


class _FakeDbUu:
    def __init__(self) -> None:
        self.last_design = None  # the most recent _FakeDesign, for inspection

    def create_schematic(self, ref):
        self.last_design = _FakeDesign()
        return self.last_design


def _install_fake_keysight(monkeypatch):
    """Install a stub keysight.ads.de into sys.modules; return the stub."""

    de = types.ModuleType("keysight.ads.de")
    de.workspace_is_open = lambda: False
    de.close_workspace = lambda: None
    de.create_workspace = lambda path: _FakeWorkspace()
    de.create_new_library = lambda name, path: None
    de.LibraryMode = types.SimpleNamespace(NON_SHARED="NON_SHARED")
    de.db_uu = _FakeDbUu()
    ks = types.ModuleType("keysight")
    ads = types.ModuleType("keysight.ads")
    ks.ads = ads
    ads.de = de
    monkeypatch.setitem(sys.modules, "keysight", ks)
    monkeypatch.setitem(sys.modules, "keysight.ads", ads)
    monkeypatch.setitem(sys.modules, "keysight.ads.de", de)
    return de


def _stub_emit(full_id, workspace_dir, **kwargs):
    """Emitter stub: write the hpeesofsim deck as both netlist files.

    The ADS netlist is the hpeesofsim deck verbatim, so the CLI's
    cross-check compares the circuit against itself -> MATCH. This
    exercises the CLI loop ADS-free without faking an ADS netlist.
    """

    circuit_dir = Path(workspace_dir).parent
    circuit_dir.mkdir(parents=True, exist_ok=True)
    deck = emit_hpeesofsim_netlist(DIAGNOSTIC_SPECS[full_id]())
    hpeesofsim_path = circuit_dir / "hpeesofsim.ckt"
    ads_path = circuit_dir / "ads_generated_netlist.txt"
    hpeesofsim_path.write_text(deck, encoding="utf-8")
    ads_path.write_text(deck, encoding="utf-8")
    return {
        "workspace_path": str(workspace_dir),
        "ads_netlist_path": str(ads_path),
        "hpeesofsim_netlist_path": str(hpeesofsim_path),
    }


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Parser + comparator (ADS-free, fixture-based)
# --------------------------------------------------------------------------

def test_parse_ads_netlist_handles_real_format(tmp_path: Path) -> None:
    """parse_ads_netlist reads real ADS Generate-Netlist output."""

    circuit = parse_ads_netlist(_write(tmp_path, "ads.txt", ADS_D1_NETLIST))
    assert sorted(c.ctype for c in circuit.elements) == [
        "MLIN", "MSUB", "PORT", "PORT"
    ]
    mlin = next(c for c in circuit.elements if c.ctype == "MLIN")
    assert mlin.name == "LINE"  # MLIN2 normalized to MLIN
    assert set(mlin.terminals) == {"n_in", "n_out"}
    assert {c.ctype for c in circuit.controllers} == {"S_PARAM", "SWEEPPLAN"}
    assert circuit.ignored  # Options / OutputPlan / #load / Component


def test_parse_hpeesofsim_deck_handles_d1(tmp_path: Path) -> None:
    """parse_hpeesofsim_deck reads the D1 flat deck."""

    deck = emit_hpeesofsim_netlist(make_d1_spec())
    circuit = parse_hpeesofsim_deck(_write(tmp_path, "d1.ckt", deck))
    assert sorted(c.ctype for c in circuit.elements) == [
        "MLIN", "MSUB", "PORT", "PORT"
    ]
    assert {c.ctype for c in circuit.controllers} == {"S_PARAM", "SWEEPPLAN"}


def test_compare_match_for_equivalent_d1(tmp_path: Path) -> None:
    """The ADS and hpeesofsim D1 netlists compare as the same circuit."""

    ads = parse_ads_netlist(_write(tmp_path, "ads.txt", ADS_D1_NETLIST))
    hpeesofsim = parse_hpeesofsim_deck(
        _write(tmp_path, "d1.ckt", emit_hpeesofsim_netlist(make_d1_spec()))
    )
    result = compare_circuits(ads, hpeesofsim)
    assert result.match, result.differences
    assert result.differences == []


def test_compare_mismatch_for_different_mlin_width(tmp_path: Path) -> None:
    """A genuine D1 difference (MLIN width) is caught."""

    altered = ADS_D1_NETLIST.replace("W=2.95 mm", "W=0.5 mm")
    ads = parse_ads_netlist(_write(tmp_path, "ads.txt", altered))
    hpeesofsim = parse_hpeesofsim_deck(
        _write(tmp_path, "d1.ckt", emit_hpeesofsim_netlist(make_d1_spec()))
    )
    result = compare_circuits(ads, hpeesofsim)
    assert not result.match
    assert any("W" in diff for diff in result.differences), result.differences


def test_compare_mismatch_for_altered_d2(tmp_path: Path) -> None:
    """Negative test: a deliberately mismatched D2 is reported."""

    deck = emit_hpeesofsim_netlist(make_d2_spec())
    altered = deck.replace("W=2.95 mm", "W=1 mm", 1)  # one MLIN widened
    good = parse_hpeesofsim_deck(_write(tmp_path, "d2.ckt", deck))
    bad = parse_hpeesofsim_deck(_write(tmp_path, "d2_bad.ckt", altered))
    result = compare_circuits(good, bad)
    assert not result.match
    assert any("W" in diff for diff in result.differences), result.differences


def test_ground_components_are_not_counted() -> None:
    """A GROUND line is a net reference, not a counted circuit element."""

    text = (
        'model m1 MSUB H=1 mm Er=3\n'
        'Port:P1 n_in 0 Num=1 Z=50 Ohm\n'
        'GROUND:G1 0\n'
        'MLIN:L1 n_in n_out Subst="m1" W=1 mm L=1 mm\n'
        'Port:P2 n_out 0 Num=2 Z=50 Ohm\n'
    )
    circuit = parse_netlist_text(text)
    assert not any(c.ctype == "GROUND" for c in circuit.elements)
    assert any("GROUND" in line for line in circuit.ignored)


# --------------------------------------------------------------------------
# OA emitter (stub keysight.ads.de)
# --------------------------------------------------------------------------

def test_all_five_oa_emitters_run(tmp_path: Path, monkeypatch) -> None:
    """Every D0-D4 OA emitter builds a workspace without raising."""

    _install_fake_keysight(monkeypatch)
    for diagnostic_id in DIAGNOSTIC_ORDER:
        workspace_dir = tmp_path / diagnostic_id / "oa_workspace"
        info = oa_emitter.emit_diagnostic_workspace(diagnostic_id, workspace_dir)
        assert Path(info["hpeesofsim_netlist_path"]).is_file()
        assert Path(info["ads_netlist_path"]).is_file()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_all_iterates_five_diagnostics(tmp_path: Path, monkeypatch, capsys) -> None:
    """`oa_diagnostic_cli all` processes every diagnostic d0-d4."""

    monkeypatch.setattr(oa_emitter, "emit_diagnostic_workspace", _stub_emit)
    monkeypatch.setattr(oa_diagnostic_cli, "_leg_b", lambda full_id, d: "n/a")
    rc = oa_diagnostic_cli.main(["all", "--output_root", str(tmp_path)])
    out = capsys.readouterr().out
    for full_id in DIAGNOSTIC_ORDER:
        assert full_id in out
    assert out.count("MATCH") >= 5
    assert rc == 0


def test_cli_single_circuit(tmp_path: Path, monkeypatch, capsys) -> None:
    """`oa_diagnostic_cli d3` processes just d3."""

    monkeypatch.setattr(oa_emitter, "emit_diagnostic_workspace", _stub_emit)
    monkeypatch.setattr(oa_diagnostic_cli, "_leg_b", lambda full_id, d: "n/a")
    rc = oa_diagnostic_cli.main(["d3", "--output_root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "d3_mtee_short_stub" in out
    assert "d0_bare_ports" not in out  # only d3 was requested
    assert rc == 0


def test_cli_manual_mode_prints_instructions(tmp_path: Path, monkeypatch, capsys) -> None:
    """`oa_diagnostic_cli d3 --manual` builds and prints manual steps."""

    monkeypatch.setattr(oa_emitter, "emit_diagnostic_workspace", _stub_emit)
    rc = oa_diagnostic_cli.main(["d3", "--manual", "--output_root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Manual verification" in out
    assert "Generate Netlist" in out
    assert rc == 0


def test_mtee_pin_convention_matches_hpeesofsim(tmp_path: Path, monkeypatch) -> None:
    """The OA emitter's MTEE pin-to-net assignment matches hpeesofsim.

    The hpeesofsim emitter (netlist/elements/mtee.py) is the validated
    reference: it emits the MTEE terminals positionally as pin1.node,
    pin2.node, pin3.node (ADS convention -- pin 1 input, pin 2 output,
    pin 3 branch). ADS lists an MTEE_ADS instance's netlist terminals
    in the order [inst_pins[0], inst_pins[2], inst_pins[1]]; the OA
    emitter's inst_pins assignment must make that order equal the
    hpeesofsim one. This guards against reverting the _PIN_INDEX swap.
    """

    de = _install_fake_keysight(monkeypatch)
    oa_emitter.emit_diagnostic_workspace(
        "d3_mtee_short_stub", tmp_path / "d3" / "oa_workspace"
    )
    mtee = de.db_uu.last_design.instances["TEE1"]

    mtee_element = next(e for e in make_d3_spec().elements if e.type == "MTEE")
    node_of = {ep.pin: ep.node for ep in mtee_element.pins}

    # ADS netlist terminal order for an MTEE_ADS instance ...
    oa_netlist_order = [
        mtee.inst_pins[0].net,
        mtee.inst_pins[2].net,
        mtee.inst_pins[1].net,
    ]
    # ... must equal the hpeesofsim positional order pin1/pin2/pin3.
    hpeesofsim_order = [f"net::{node_of[pin]}" for pin in (1, 2, 3)]
    assert oa_netlist_order == hpeesofsim_order


