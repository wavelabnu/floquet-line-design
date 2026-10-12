"""Cross-check tests for the OA-Python canonical Levels 1-3.

Every test runs without ADS. The 12 canonical OA emitters are exercised
against a stub ``keysight.ads.de``; the canonical CLI is exercised with
the emitter stubbed out; the comparator is exercised with fixtures.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
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
    validate_device_spec,
)
from tvtl_ads_builder_v2.device_spec.schema import DEVICE_SPEC_VERSION
from tvtl_ads_builder_v2.netlist.emit import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa import oa_canonical_cli, oa_emitter
from tvtl_ads_builder_v2.oa.netlist_compare import (
    compare_circuits,
    parse_hpeesofsim_deck,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    LEVEL_2_CANONICAL_CIRCUITS,
    LEVEL_3_CANONICAL_CIRCUITS,
    LEVEL_4_CANONICAL_CIRCUITS,
    LEVEL_5_CANONICAL_CIRCUITS,
    LEVEL_6_CANONICAL_CIRCUITS,
    LEVEL_7_CANONICAL_CIRCUITS,
    LEVEL_9_CANONICAL_CIRCUITS,
)
from tvtl_ads_builder_v2.tvtl_simulate import simulate_s_parameters

_ALL_CANONICAL = (
    LEVEL_1_CANONICAL_CIRCUITS
    + LEVEL_2_CANONICAL_CIRCUITS
    + LEVEL_3_CANONICAL_CIRCUITS
)


# --------------------------------------------------------------------------
# Stub keysight.ads.de (recording variant -- instances kept for inspection)
# --------------------------------------------------------------------------

_FAKE_PIN_COUNT = {
    "TermG": 1, "MLIN": 2, "MTEE_ADS": 3, "MSUB": 0, "GROUND": 1, "S_Param": 0,
    "R": 2, "L": 2, "C": 2, "SDD2P": 4, "HarmonicBalance": 0, "P_1Tone": 2,
    "I_Probe": 2,
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
        self.parameters: dict = {}  # empty -> setp() is a no-op
        self.inst_pins = [_FakePin() for _ in range(n_pins)]


class _FakeDesign:
    default_wire_layer = "fake_wire_layer"

    def __init__(self) -> None:
        self.instances: dict = {}

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
        self.last_design = None

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


def _stub_emit(spec, workspace_dir, **kwargs):
    """Emitter stub: write the hpeesofsim deck as both netlist files."""

    circuit_dir = Path(workspace_dir).parent
    circuit_dir.mkdir(parents=True, exist_ok=True)
    deck = emit_hpeesofsim_netlist(spec)
    (circuit_dir / "hpeesofsim.ckt").write_text(deck, encoding="utf-8")
    (circuit_dir / "ads_generated_netlist.txt").write_text(deck, encoding="utf-8")
    return {
        "workspace_path": str(workspace_dir),
        "ads_netlist_path": str(circuit_dir / "ads_generated_netlist.txt"),
        "hpeesofsim_netlist_path": str(circuit_dir / "hpeesofsim.ckt"),
    }


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_all_canonical_oa_emitters_run(tmp_path: Path, monkeypatch) -> None:
    """Every Level 1/2/3 canonical OA emitter builds without raising."""

    _install_fake_keysight(monkeypatch)
    assert len(_ALL_CANONICAL) == 12
    for spec in _ALL_CANONICAL:
        info = oa_emitter.emit_workspace_from_spec(
            spec, tmp_path / spec.label / "oa_workspace"
        )
        assert Path(info["hpeesofsim_netlist_path"]).is_file()
        assert Path(info["ads_netlist_path"]).is_file()


def test_level5_sdd2p_oa_emitter_runs(tmp_path: Path, monkeypatch) -> None:
    """The Level-5 SDD2P canonical OA emitter builds without raising.

    Exercised against the stub ``keysight.ads.de`` -- the OA emitter
    must place the 4-terminal SDD2P, wire it through ``_PIN_INDEX`` and
    set its body without an ADS runtime. The hpeesofsim deck written
    alongside carries the charge polynomial.
    """

    _install_fake_keysight(monkeypatch)
    spec = next(
        c for c in LEVEL_5_CANONICAL_CIRCUITS
        if c.label == "level5_single_sdd2p_varactor"
    )
    info = oa_emitter.emit_workspace_from_spec(
        spec, tmp_path / spec.label / "oa_workspace"
    )
    hpeesofsim = Path(info["hpeesofsim_netlist_path"]).read_text(
        encoding="utf-8"
    )
    assert "SDD:VARACTOR" in hpeesofsim
    assert "I[1,1]=" in hpeesofsim
    # port 2 is the KCL-closure mirror at weighting index 1, not I[2,0].
    assert "I[2,1]=" in hpeesofsim
    assert "I[2,0]" not in hpeesofsim
    # port 1 -- the SMV1245 charge polynomial: q1 = C0 = 7e-12,
    # q2 = C1/2 = -1e-12; port 2 is its exact negation.
    assert "I[1,1]=(7e-12)*_v1+(-1e-12)*_v1**2" in hpeesofsim
    assert "I[2,1]=(-7e-12)*_v1+(1e-12)*_v1**2" in hpeesofsim


def test_level6_hb_oa_emitter_runs(tmp_path: Path, monkeypatch) -> None:
    """The Level-6 single-tone HB controller OA emitter builds without raising.

    Exercised against the stub ``keysight.ads.de`` -- the OA emitter
    must place the HarmonicBalance controller and set its single-tone
    parameters without an ADS runtime. The hpeesofsim deck written
    alongside carries the HB: analysis directive.
    """

    _install_fake_keysight(monkeypatch)
    spec = next(
        c for c in LEVEL_6_CANONICAL_CIRCUITS
        if c.label == "level6_single_tone_hb_controller"
    )
    info = oa_emitter.emit_workspace_from_spec(
        spec, tmp_path / spec.label / "oa_workspace"
    )
    hpeesofsim = Path(info["hpeesofsim_netlist_path"]).read_text(
        encoding="utf-8"
    )
    assert "HB:HB1" in hpeesofsim
    assert "Freq[1]=3 GHz" in hpeesofsim
    assert "Order[1]=5" in hpeesofsim
    assert "MaxOrder=5" in hpeesofsim


def test_level6_two_tone_hb_oa_emitter_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """The Level-6 two-tone HB controller OA emitter builds without raising.

    Exercised against the stub ``keysight.ads.de`` -- the OA emitter
    must place the HarmonicBalance controller and set both tones'
    parameters without an ADS runtime. The hpeesofsim deck written
    alongside carries the two-tone HB: directive.
    """

    _install_fake_keysight(monkeypatch)
    spec = next(
        c for c in LEVEL_6_CANONICAL_CIRCUITS
        if c.label == "level6_two_tone_hb_controller"
    )
    info = oa_emitter.emit_workspace_from_spec(
        spec, tmp_path / spec.label / "oa_workspace"
    )
    hpeesofsim = Path(info["hpeesofsim_netlist_path"]).read_text(
        encoding="utf-8"
    )
    assert "HB:HB1" in hpeesofsim
    assert "Freq[1]=3 GHz" in hpeesofsim
    assert "Freq[2]=6 GHz" in hpeesofsim
    assert "Order[1]=5" in hpeesofsim
    assert "Order[2]=5" in hpeesofsim
    assert "MaxOrder=10" in hpeesofsim


def test_level7_oa_emitters_run(tmp_path: Path, monkeypatch) -> None:
    """Both Level-7 canonical OA emitters build without raising.

    Exercised against the stub ``keysight.ads.de`` -- the OA emitter
    must place every ``P_1Tone`` source instance, wire its positive
    terminal to its port node and its return terminal to a GROUND
    symbol, all without an ADS runtime. The hpeesofsim deck written
    alongside carries the ``P_1Tone:`` source lines.
    """

    _install_fake_keysight(monkeypatch)
    assert len(LEVEL_7_CANONICAL_CIRCUITS) == 2
    for spec in LEVEL_7_CANONICAL_CIRCUITS:
        info = oa_emitter.emit_workspace_from_spec(
            spec, tmp_path / spec.label / "oa_workspace"
        )
        hpeesofsim = Path(info["hpeesofsim_netlist_path"]).read_text(
            encoding="utf-8"
        )
        assert "P_1Tone:SIGNAL_SRC" in hpeesofsim
        assert "P_1Tone:PUMP_SRC" in hpeesofsim
        # named references resolved to literals in the emitted deck.
        assert "Freq=3 GHz" in hpeesofsim
        assert "Freq=6 GHz" in hpeesofsim
        assert "polar(dbmtow(-30),0)" in hpeesofsim


def test_level7_integrated_cell_oa_emitter_composes_all_layers(
    tmp_path: Path, monkeypatch
) -> None:
    """The integrated Level-7 cell emits VAR + sources + HB + SDD2P.

    The single integration circuit exercises every element layer at
    once; the OA emitter must place its sources, the HB controller and
    the SDD2P varactor against the stub without raising, and the
    hpeesofsim deck must carry the resolved literals.
    """

    _install_fake_keysight(monkeypatch)
    spec = next(
        c for c in LEVEL_7_CANONICAL_CIRCUITS
        if c.label == "level7_tvtl_unit_cell_two_tone_hb"
    )
    info = oa_emitter.emit_workspace_from_spec(
        spec, tmp_path / spec.label / "oa_workspace"
    )
    hpeesofsim = Path(info["hpeesofsim_netlist_path"]).read_text(
        encoding="utf-8"
    )
    # two P_1Tone sources, the two-tone HB controller, the SDD2P.
    assert "P_1Tone:SIGNAL_SRC n_input 0" in hpeesofsim
    assert "P_1Tone:PUMP_SRC n_input 0" in hpeesofsim
    assert "HB:HB1 Freq[1]=3 GHz Freq[2]=6 GHz" in hpeesofsim
    assert "SDD:VARACTOR" in hpeesofsim
    # the RLC stub's VAR references resolved (R_var/L_par/C_var).
    assert "R:STUB_R n_branch_top STUB_RL_node R=2 Ohm" in hpeesofsim
    assert "L:STUB_L STUB_RL_node STUB_LC_node L=2 nH" in hpeesofsim
    assert "C:STUB_C STUB_LC_node n_var_mid C=4 pF" in hpeesofsim


def test_cli_all_iterates_every_canonical_circuit(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """`oa_canonical_cli all` processes every canonical circuit, Levels 1-4."""

    monkeypatch.setattr(oa_emitter, "emit_workspace_from_spec", _stub_emit)
    monkeypatch.setattr(
        oa_canonical_cli, "_leg_b",
        lambda spec, d, label, tol: {
            "reason": "within_tolerance", "max_abs_delta_dB": 0.01
        },
    )
    rc = oa_canonical_cli.main(["all", "--output_root", str(tmp_path)])
    out = capsys.readouterr().out
    every_circuit = (
        _ALL_CANONICAL
        + LEVEL_4_CANONICAL_CIRCUITS
        + LEVEL_5_CANONICAL_CIRCUITS
        + LEVEL_6_CANONICAL_CIRCUITS
        + LEVEL_7_CANONICAL_CIRCUITS
        + LEVEL_9_CANONICAL_CIRCUITS
    )
    assert len(every_circuit) == 25
    for spec in every_circuit:
        assert spec.label in out
    assert out.count("MATCH") >= len(every_circuit)
    assert rc == 0


def test_cli_single_level(tmp_path: Path, monkeypatch, capsys) -> None:
    """`oa_canonical_cli level2` processes only Level 2."""

    monkeypatch.setattr(oa_emitter, "emit_workspace_from_spec", _stub_emit)
    monkeypatch.setattr(
        oa_canonical_cli, "_leg_b",
        lambda spec, d, label, tol: {
            "reason": "within_tolerance", "max_abs_delta_dB": 0.01
        },
    )
    rc = oa_canonical_cli.main(["level2", "--output_root", str(tmp_path)])
    out = capsys.readouterr().out
    for spec in LEVEL_2_CANONICAL_CIRCUITS:
        assert spec.label in out
    assert "matched_short" not in out  # a Level-1 circuit -- not requested
    assert rc == 0


def test_negative_broken_level2_circuit(tmp_path: Path) -> None:
    """A deliberately broken Level-2 circuit is reported as a mismatch."""

    spec = LEVEL_2_CANONICAL_CIRCUITS[0]  # two_identical_5mm_each
    deck = emit_hpeesofsim_netlist(spec)
    broken = deck.replace("W=2.95 mm", "W=1 mm", 1)  # widen one MLIN
    good_path = tmp_path / "good.ckt"
    bad_path = tmp_path / "bad.ckt"
    good_path.write_text(deck, encoding="utf-8")
    bad_path.write_text(broken, encoding="utf-8")
    result = compare_circuits(
        parse_hpeesofsim_deck(good_path),
        parse_hpeesofsim_deck(bad_path),
    )
    assert not result.match
    assert any("W" in diff for diff in result.differences), result.differences


def test_mtee_pin_convention_canonical(tmp_path: Path, monkeypatch) -> None:
    """The MTEE pin convention holds for a canonical Level-3 circuit.

    Regression guard: ADS lists an MTEE_ADS instance's
    netlist terminals as [inst_pins[0], inst_pins[2], inst_pins[1]];
    that order must equal the hpeesofsim positional order pin1/pin2/pin3.
    """

    spec = next(
        s for s in LEVEL_3_CANONICAL_CIRCUITS if s.label == "tee_short_paper_dims"
    )
    de = _install_fake_keysight(monkeypatch)
    oa_emitter.emit_workspace_from_spec(spec, tmp_path / spec.label / "oa_workspace")
    mtee = de.db_uu.last_design.instances["TEE1"]

    mtee_element = next(e for e in spec.elements if e.type == "MTEE")
    node_of = {ep.pin: ep.node for ep in mtee_element.pins}

    oa_netlist_order = [
        mtee.inst_pins[0].net,
        mtee.inst_pins[2].net,
        mtee.inst_pins[1].net,
    ]
    hpeesofsim_order = [f"net::{node_of[pin]}" for pin in (1, 2, 3)]
    assert oa_netlist_order == hpeesofsim_order


def test_open_stub_reversed_mlin_terminals_match(tmp_path: Path) -> None:
    """An open-stub MLIN compares equal whichever terminal order it lists.

    Regression guard for the net-isomorphism fix: the open stub's two
    nets are both internal (its far net is anchored only by the MTEE,
    which is compared afterwards), so the comparator must defer the
    stub's terminal-order decision -- otherwise ADS's reversed MLIN
    terminal order is bound the wrong way and conflicts at the MTEE.
    """

    spec = next(
        s for s in LEVEL_3_CANONICAL_CIRCUITS if s.label == "tee_open_zero_stub"
    )
    deck = emit_hpeesofsim_netlist(spec)
    # ADS may netlist an MLIN's two terminals in either order.
    reversed_stub = deck.replace(
        "n_branch_top n_branch_end", "n_branch_end n_branch_top", 1
    )
    assert reversed_stub != deck
    good_path = tmp_path / "stub.ckt"
    rev_path = tmp_path / "stub_rev.ckt"
    good_path.write_text(deck, encoding="utf-8")
    rev_path.write_text(reversed_stub, encoding="utf-8")
    result = compare_circuits(
        parse_hpeesofsim_deck(good_path),
        parse_hpeesofsim_deck(rev_path),
    )
    assert result.match, result.differences


# --------------------------------------------------------------------------
# Level 4: lumped R / L / C and the composite RLC_SERIES
# --------------------------------------------------------------------------

_SUBSTRATE = LEVEL_1_CANONICAL_CIRCUITS[0].substrate


def _two_port_single_element(element: Element) -> DeviceSpec:
    """Wrap one two-terminal element between two 50-Ohm ports.

    Used by the parameter-validation test: the element under test spans
    ``n_in`` -> ``n_out`` and the rest of the spec is a minimal valid
    2-port, so ``validate_device_spec`` exercises only the element.
    """

    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=Analysis(
            type="s_parameters",
            frequency_grid_Hz=FrequencyGrid(start=1.0e9, stop=10.0e9, points=91),
            port_impedance_ohm=50.0,
        ),
        substrate=_SUBSTRATE,
        nodes=(Node(id="n_in"), Node(id="n_out")),
        elements=(
            Element(
                id="P1", type="TermG",
                pins=(ElementPin(pin=1, node="n_in"),),
                parameters={"z_ohm": 50.0, "num": 1},
            ),
            element,
            Element(
                id="P2", type="TermG",
                pins=(ElementPin(pin=1, node="n_out"),),
                parameters={"z_ohm": 50.0, "num": 2},
            ),
        ),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        label="level4_validation_probe",
    )


def _lumped(element_type: str, parameters: dict) -> Element:
    """A two-terminal lumped element spanning ``n_in`` -> ``n_out``."""

    return Element(
        id="DUT", type=element_type,
        pins=(ElementPin(pin=1, node="n_in"), ElementPin(pin=2, node="n_out")),
        parameters=parameters,
    )


def test_level4_specs_build_and_validate() -> None:
    """The four Level-4 canonical specs construct and pass validation."""

    assert len(LEVEL_4_CANONICAL_CIRCUITS) == 4
    labels = [s.label for s in LEVEL_4_CANONICAL_CIRCUITS]
    assert labels == [
        "rlc_pure_50ohm", "rlc_lossless", "mlin_then_rlc", "mtee_with_rlc_stub"
    ]
    for spec in LEVEL_4_CANONICAL_CIRCUITS:
        validate_device_spec(spec)  # raises DeviceSpecError on any violation


def test_lumped_element_parameter_validation() -> None:
    """R accepts R >= 0; L and C demand strictly positive values."""

    # Valid: R = 0 is a legal zero-Ohm resistor; L and C positive.
    validate_device_spec(_two_port_single_element(_lumped("R", {"r_ohm": 0.0})))
    validate_device_spec(_two_port_single_element(_lumped("R", {"r_ohm": 50.0})))
    validate_device_spec(_two_port_single_element(_lumped("L", {"l_h": 1.0e-9})))
    validate_device_spec(_two_port_single_element(_lumped("C", {"c_f": 1.0e-12})))
    validate_device_spec(
        _two_port_single_element(
            _lumped("RLC_SERIES", {"r_ohm": 0.0, "l_h": 1.0e-9, "c_f": 1.0e-12})
        )
    )

    # Invalid: negative R, non-positive L, non-positive C.
    with pytest.raises(DeviceSpecError):
        validate_device_spec(
            _two_port_single_element(_lumped("R", {"r_ohm": -1.0}))
        )
    with pytest.raises(DeviceSpecError):
        validate_device_spec(
            _two_port_single_element(_lumped("L", {"l_h": 0.0}))
        )
    with pytest.raises(DeviceSpecError):
        validate_device_spec(
            _two_port_single_element(_lumped("C", {"c_f": -1.0e-12}))
        )
    with pytest.raises(DeviceSpecError):
        validate_device_spec(
            _two_port_single_element(
                _lumped("RLC_SERIES", {"r_ohm": 50.0, "l_h": 0.0, "c_f": 1.0e-12})
            )
        )
    # A missing required parameter is also rejected.
    with pytest.raises(DeviceSpecError):
        validate_device_spec(_two_port_single_element(_lumped("C", {})))


def test_tvtl_rlc_series_impedance_matches_closed_form() -> None:
    """The analytical S21 of the pure RLC equals 2 / (2 + Z/Z0).

    With ABCD = [[1, Z], [0, 1]] between two Z0 ports, the reciprocal
    S-parameter conversion gives S21 = 2 / (2 + Z/Z0). ``Z`` here is the
    independently evaluated series impedance R + jwL + 1/(jwC).
    """

    spec = next(
        s for s in LEVEL_4_CANONICAL_CIRCUITS if s.label == "rlc_pure_50ohm"
    )
    result = simulate_s_parameters(spec)
    freqs = np.asarray(result["frequencies_Hz"], dtype=float)
    omega = 2.0 * np.pi * freqs
    r_ohm, l_h, c_f, z0 = 50.0, 1.0e-9, 1.0e-12, 50.0
    z = r_ohm + 1j * omega * l_h + 1.0 / (1j * omega * c_f)
    expected_s21 = 2.0 / (2.0 + z / z0)
    assert np.allclose(result["S21"], expected_s21, rtol=1e-9, atol=1e-12)

    # The series resonance ~5.03 GHz is where |S21| is largest.
    f0 = 1.0 / (2.0 * np.pi * np.sqrt(l_h * c_f))
    peak_freq = freqs[int(np.argmax(np.abs(result["S21"])))]
    assert abs(peak_freq - f0) <= 0.15e9


def test_tvtl_lossless_rlc_resonance_reaches_full_transmission() -> None:
    """The lossless (R = 0) RLC reaches ~0 dB near its series resonance."""

    spec = next(
        s for s in LEVEL_4_CANONICAL_CIRCUITS if s.label == "rlc_lossless"
    )
    result = simulate_s_parameters(spec)
    s21_mag = np.abs(result["S21"])
    # At resonance Z = 0 -- a through short -- so |S21| -> 1.
    assert s21_mag.max() > 0.999
    # Far below resonance the capacitive reactance dominates and rejects.
    assert s21_mag[0] < 0.95


def test_hpeesofsim_rlc_series_three_line_expansion() -> None:
    """An RLC_SERIES emits R/L/C lines with deterministic internal nodes."""

    spec = next(
        s for s in LEVEL_4_CANONICAL_CIRCUITS if s.label == "rlc_pure_50ohm"
    )
    deck = emit_hpeesofsim_netlist(spec)
    lines = deck.splitlines()
    r_line = next(ln for ln in lines if ln.startswith("R:RLC1_R "))
    l_line = next(ln for ln in lines if ln.startswith("L:RLC1_L "))
    c_line = next(ln for ln in lines if ln.startswith("C:RLC1_C "))

    # The three primitives chain pin 1 -> RL_node -> LC_node -> pin 2.
    assert r_line == "R:RLC1_R n_in RLC1_RL_node R=50 Ohm"
    assert l_line.startswith("L:RLC1_L RLC1_RL_node RLC1_LC_node L=")
    assert l_line.endswith("nH")
    assert c_line.startswith("C:RLC1_C RLC1_LC_node n_out C=")
    assert c_line.endswith("pF")


def test_oa_emitter_runs_for_all_level4_circuits(
    tmp_path: Path, monkeypatch
) -> None:
    """Every Level-4 canonical OA emitter builds without raising."""

    _install_fake_keysight(monkeypatch)
    for spec in LEVEL_4_CANONICAL_CIRCUITS:
        info = oa_emitter.emit_workspace_from_spec(
            spec, tmp_path / spec.label / "oa_workspace"
        )
        assert Path(info["hpeesofsim_netlist_path"]).is_file()
        assert Path(info["ads_netlist_path"]).is_file()


def test_oa_rlc_pin_convention_matches_hpeesofsim(
    tmp_path: Path, monkeypatch
) -> None:
    """The OA emitter places the expanded R/L/C with hpeesofsim's net order.

    RLC_SERIES expands to a series R -> L -> C chain. The hpeesofsim
    emitter writes each primitive's terminals positionally as
    ``pin1.node pin2.node``; the OA emitter must land spec pin 1 on
    ``inst_pins[0]`` and pin 2 on ``inst_pins[1]`` so the two netlists
    carry the identical R-L-C topology.
    """

    spec = next(
        s for s in LEVEL_4_CANONICAL_CIRCUITS if s.label == "rlc_pure_50ohm"
    )
    de = _install_fake_keysight(monkeypatch)
    oa_emitter.emit_workspace_from_spec(spec, tmp_path / spec.label / "oa_workspace")
    design = de.db_uu.last_design

    resistor = design.instances["RLC1_R"]
    inductor = design.instances["RLC1_L"]
    capacitor = design.instances["RLC1_C"]
    assert [p.net for p in resistor.inst_pins] == [
        "net::n_in", "net::RLC1_RL_node"
    ]
    assert [p.net for p in inductor.inst_pins] == [
        "net::RLC1_RL_node", "net::RLC1_LC_node"
    ]
    assert [p.net for p in capacitor.inst_pins] == [
        "net::RLC1_LC_node", "net::n_out"
    ]


def test_cli_level4_runs_end_to_end(tmp_path: Path, monkeypatch, capsys) -> None:
    """`oa_canonical_cli level4` processes only the four Level-4 circuits."""

    monkeypatch.setattr(oa_emitter, "emit_workspace_from_spec", _stub_emit)
    monkeypatch.setattr(
        oa_canonical_cli, "_leg_b",
        lambda spec, d, label, tol: {
            "reason": "within_tolerance", "max_abs_delta_dB": 0.01
        },
    )
    rc = oa_canonical_cli.main(["level4", "--output_root", str(tmp_path)])
    out = capsys.readouterr().out
    for spec in LEVEL_4_CANONICAL_CIRCUITS:
        assert spec.label in out
    assert "matched_short" not in out  # a Level-1 circuit -- not requested
    assert rc == 0


