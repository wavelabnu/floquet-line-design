"""Tests for the Level-8 canonical circuits.

ADS-free: the canonicals build and emit without an ADS runtime. The OA
emitter is exercised against the stub ``keysight.ads.de`` reused from
``test_oa_canonical``; the content checks (SDD2P / VAR / two-tone-HB /
P_1Tone) run against the emitted hpeesofsim deck.
"""

from __future__ import annotations

from pathlib import Path

from test_oa_canonical import _install_fake_keysight, _stub_emit

from tvtl_ads_builder_v2.design import canonical_cli
from tvtl_ads_builder_v2.design.canonicals import (
    LEVEL_8_CANONICAL_CIRCUITS,
    LEVEL_8_LABEL,
    level8_five_cell_tvtl_cascade,
    level8_three_cell_tvtl_cascade,
)
from tvtl_ads_builder_v2.netlist.emit import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa import oa_canonical_cli, oa_emitter
from tvtl_ads_builder_v2.oa.hb_controller_content_check import (
    check_hb_two_tone_content,
)
from tvtl_ads_builder_v2.oa.p1tone_content_check import check_p1tone_content
from tvtl_ads_builder_v2.oa.sdd2p_content_check import check_sdd2p_content
from tvtl_ads_builder_v2.oa.var_content_check import check_var_content
from tvtl_ads_builder_v2.topology import render_topology_svg


# --------------------------------------------------------------------------
# The canonical tuple
# --------------------------------------------------------------------------


def test_level_8_tuple_holds_both_cascades() -> None:
    """LEVEL_8_CANONICAL_CIRCUITS is the three-cell and five-cell cascades."""

    assert len(LEVEL_8_CANONICAL_CIRCUITS) == 2
    labels = [spec.label for spec in LEVEL_8_CANONICAL_CIRCUITS]
    assert labels == [
        "level8_three_cell_tvtl_cascade",
        "level8_five_cell_tvtl_cascade",
    ]


def test_both_canonicals_build() -> None:
    """Both canonical builders produce a labelled device spec."""

    three = level8_three_cell_tvtl_cascade()
    five = level8_five_cell_tvtl_cascade()
    assert three.label == "level8_three_cell_tvtl_cascade"
    assert five.label == "level8_five_cell_tvtl_cascade"


# --------------------------------------------------------------------------
# Element count per cell
# --------------------------------------------------------------------------


def test_three_cell_element_count() -> None:
    """The three-cell cascade has 6 elements/cell + 1 output port = 19."""

    spec = level8_three_cell_tvtl_cascade()
    assert len(spec.elements) == 6 * 3 + 1
    assert sum(1 for e in spec.elements if e.type == "MLIN") == 3
    assert sum(1 for e in spec.elements if e.type == "MTEE") == 3
    assert sum(1 for e in spec.elements if e.type == "SDD2P") == 3
    assert sum(1 for e in spec.elements if e.type == "TermG") == 1


def test_five_cell_element_count() -> None:
    """The five-cell cascade has 6 elements/cell + 1 output port = 31."""

    spec = level8_five_cell_tvtl_cascade()
    assert len(spec.elements) == 6 * 5 + 1
    assert sum(1 for e in spec.elements if e.type == "MTEE") == 5
    assert sum(1 for e in spec.elements if e.type == "SDD2P") == 5


def test_canonicals_share_one_cell_template() -> None:
    """Three-cell and five-cell cells are identical -- only n_cells differs.

    Cell 0 of each canonical has byte-identical element types, so the
    five-cell canonical genuinely reuses the three-cell builder pattern.
    """

    three = level8_three_cell_tvtl_cascade()
    five = level8_five_cell_tvtl_cascade()
    cell0_three = [e.type for e in three.elements[:6]]
    cell0_five = [e.type for e in five.elements[:6]]
    assert cell0_three == cell0_five == ["MLIN", "MTEE", "R", "L", "C", "SDD2P"]


# --------------------------------------------------------------------------
# hpeesofsim deck
# --------------------------------------------------------------------------


def test_canonicals_emit_a_hpeesofsim_deck() -> None:
    """Both canonicals emit a hpeesofsim flat netlist."""

    for spec in LEVEL_8_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        assert "P_1Tone:SIGNAL_SRC n_input 0" in deck
        assert "P_1Tone:PUMP_SRC n_input 0" in deck
        assert "HB:HB1 Freq[1]=3 GHz Freq[2]=6 GHz" in deck
        # generic test values resolved from the VAR block.
        assert "R=2 Ohm" in deck
        assert "L=2 nH" in deck
        assert "C=4 pF" in deck


# --------------------------------------------------------------------------
# Content checks against the emitted deck
# --------------------------------------------------------------------------


def test_canonicals_pass_the_sdd2p_content_check() -> None:
    """Every cell's SDD2P carries the SMV1245 charge polynomial."""

    for spec in LEVEL_8_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        result = check_sdd2p_content(spec, hpeesofsim_netlist_text=deck)
        assert result.match, result.failures
        # one SDD2P checked per cell.
        assert len(result.checked) == sum(
            1 for e in spec.elements if e.type == "SDD2P"
        )


def test_canonicals_pass_the_var_content_check() -> None:
    """The VAR named references resolve to the declared literals."""

    for spec in LEVEL_8_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        result = check_var_content(spec, hpeesofsim_netlist_text=deck)
        assert result.match, result.failures


def test_canonicals_pass_the_p1tone_content_check() -> None:
    """Both P_1Tone sources are emitted with the resolved tone parameters."""

    for spec in LEVEL_8_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        result = check_p1tone_content(spec, hpeesofsim_netlist_text=deck)
        assert result.match, result.failures


def test_canonicals_pass_the_hb_two_tone_content_check() -> None:
    """The two-tone HB controller is emitted as the spec declares it."""

    for spec in LEVEL_8_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        result = check_hb_two_tone_content(spec, hpeesofsim_netlist_text=deck)
        assert result.match, result.failures


# --------------------------------------------------------------------------
# OA emitter accepts the spec (stub keysight -- no ADS runtime)
# --------------------------------------------------------------------------


def test_oa_emitter_accepts_both_canonicals(tmp_path: Path, monkeypatch) -> None:
    """The OA emitter builds a workspace for each canonical without raising.

    Exercised against the stub ``keysight.ads.de``: the emitter must
    place every per-cell MLIN, MTEE, R, L, C and SDD2P instance and
    route the cascade, all without an ADS runtime. The 5-cell cascade is
    the longer-through-path stress for the polyline wire router.
    """

    _install_fake_keysight(monkeypatch)
    for spec in LEVEL_8_CANONICAL_CIRCUITS:
        info = oa_emitter.emit_workspace_from_spec(
            spec, tmp_path / spec.label / "oa_workspace"
        )
        assert Path(info["hpeesofsim_netlist_path"]).is_file()
        assert Path(info["ads_netlist_path"]).is_file()


def test_topology_svg_renders_for_both_canonicals(tmp_path: Path) -> None:
    """The topology renderer draws the cascade without grid fallback."""

    for spec in LEVEL_8_CANONICAL_CIRCUITS:
        out = tmp_path / f"{spec.label}.svg"
        render_topology_svg(spec, out)
        text = out.read_text(encoding="utf-8")
        # one SDD2P box per cell appears in the drawing.
        n_cells = sum(1 for e in spec.elements if e.type == "MTEE")
        assert text.count('data-type="SDD2P"') == n_cells


# --------------------------------------------------------------------------
# The Level-8 CLI
# --------------------------------------------------------------------------


def test_canonical_cli_registers_level_8() -> None:
    """Importing the Level-8 CLI registers it in the shared per-level table."""

    assert "level8" in oa_canonical_cli._LEVELS
    circuits, label, _tol = oa_canonical_cli._LEVELS["level8"]
    assert label == LEVEL_8_LABEL
    assert len(circuits) == 2


def test_canonical_cli_runs_end_to_end(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """`design.canonical_cli` processes both Level-8 canonicals."""

    monkeypatch.setattr(oa_emitter, "emit_workspace_from_spec", _stub_emit)
    monkeypatch.setattr(
        oa_canonical_cli, "_leg_b",
        lambda spec, d, label, tol: {
            "reason": "blocked", "max_abs_delta_dB": float("nan")
        },
    )
    rc = canonical_cli.main(["--output_root", str(tmp_path)])
    out = capsys.readouterr().out
    for spec in LEVEL_8_CANONICAL_CIRCUITS:
        assert spec.label in out
    assert out.count("MATCH") >= 2  # both canonicals cross-check MATCH
    assert rc == 0


