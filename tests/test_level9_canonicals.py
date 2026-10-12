"""Tests for the Level-9 probed cascades.

ADS-free: the two Level-9 canonicals build, validate, emit through the
hpeesofsim and OA paths (the OA path against the stub
``keysight.ads.de``), and pass every content check across the multiple
SDD2P / P_1Tone / I_Probe instances. Live ADS HB validation is an
ADS-dependent reproduction step and is not part of this base test.

Two refinements over Level-7/8 that these tests pin down:

* the two P_1Tone sources carry DISTINCT ``Num`` values (signal Num=1,
  pump Num=2) -- the Level-7/8 specs share Num=1 and fail HB with
  "Duplicate Port or Term number for SIGNAL_SRC"; the Level-9 specs
  must not regress that bug;
* every cell carries a varactor I_Probe (``IPR_VAR_C{k}``) plus a pair
  of boundary I_Probes (``IPR_IN`` / ``IPR_C{k}C{k+1}`` / ``IPR_OUT``)
  -- the energy-conservation cross-check vehicle. The count
  per canonical is 2*N + 1 (N boundary + N+1 inclusive) -- five for
  the 2-cell, seven for the 3-cell.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test_oa_canonical import _install_fake_keysight, _stub_emit

from tvtl_ads_builder_v2.device_spec import resolve_spec, validate_device_spec
from tvtl_ads_builder_v2.netlist.emit import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa import oa_canonical_cli, oa_emitter
from tvtl_ads_builder_v2.oa.hb_controller_content_check import (
    check_hb_two_tone_content,
)
from tvtl_ads_builder_v2.oa.netlist_compare import (
    compare_circuits,
    parse_netlist_text,
)
from tvtl_ads_builder_v2.oa.p1tone_content_check import check_p1tone_content
from tvtl_ads_builder_v2.oa.sdd2p_content_check import check_sdd2p_content
from tvtl_ads_builder_v2.oa.var_content_check import check_var_content
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_9_CANONICAL_CIRCUITS,
    LEVEL_9_LABEL,
    make_probed_tvtl_cascade,
)
from tvtl_ads_builder_v2.topology import render_topology_svg


# --------------------------------------------------------------------------
# The canonical tuple + per-canonical inventory
# --------------------------------------------------------------------------


def test_level_9_tuple_holds_two_and_three_cell_cascades() -> None:
    """LEVEL_9_CANONICAL_CIRCUITS is exactly the 2-cell and 3-cell canonicals."""

    assert len(LEVEL_9_CANONICAL_CIRCUITS) == 2
    labels = [spec.label for spec in LEVEL_9_CANONICAL_CIRCUITS]
    assert labels == [
        "level9_two_cell_probed_tvtl_cascade",
        "level9_three_cell_probed_tvtl_cascade",
    ]


@pytest.mark.parametrize(
    "spec_index,n_cells,n_probes",
    [(0, 2, 5), (1, 3, 7)],
)
def test_i_probe_count_per_canonical(spec_index, n_cells, n_probes) -> None:
    """I_Probe count = (N+1) boundary + N varactor probes."""

    spec = LEVEL_9_CANONICAL_CIRCUITS[spec_index]
    iprobes = [e for e in spec.elements if e.type == "I_Probe"]
    assert len(iprobes) == n_probes == 2 * n_cells + 1


def test_element_counts_scale_with_cell_count() -> None:
    """A probed cell contributes 6 elements + 1 boundary probe between cells.

    NAME annotations sit on every per-cell observable net,
    contributing ``4*N + 2`` additional 1-pin NAME elements (the
    cascade endpoints ``n_input`` / ``n_output`` plus per cell the
    branch top, the varactor-pin-2 net, the cell output, and the
    post-inter-cell-probe net for k < N). Excluding NAME elements,
    the original ``7*N + 2`` formula still holds; including them, the
    total is ``11*N + 4``.
    """

    two_cell = LEVEL_9_CANONICAL_CIRCUITS[0]
    three_cell = LEVEL_9_CANONICAL_CIRCUITS[1]
    # 7*N + 2 electrical elements: 6 per cell (LINE_LEFT, TEE,
    # LINE_RIGHT, VARACTOR, IPR_VAR, STUB) + (N-1) inter-cell I_Probes
    # + IPR_IN + IPR_OUT + P_OUT.
    two_cell_non_name = [e for e in two_cell.elements if e.type != "NAME"]
    three_cell_non_name = [e for e in three_cell.elements if e.type != "NAME"]
    assert len(two_cell_non_name) == 7 * 2 + 2
    assert len(three_cell_non_name) == 7 * 3 + 2
    # 4*N + 2 NAME annotations -- see
    # ``make_probed_tvtl_cascade`` for the per-net enumeration.
    two_cell_names = [e for e in two_cell.elements if e.type == "NAME"]
    three_cell_names = [e for e in three_cell.elements if e.type == "NAME"]
    assert len(two_cell_names) == 4 * 2 + 2
    assert len(three_cell_names) == 4 * 3 + 2


def test_per_cell_topology_uses_two_mlin_halves_and_an_mtee() -> None:
    """Every cell is LINE_LEFT + TEE + LINE_RIGHT + VARACTOR + IPR_VAR + STUB."""

    spec = LEVEL_9_CANONICAL_CIRCUITS[0]
    ids = {e.id for e in spec.elements}
    for k in (1, 2):
        for prefix in ("LINE_LEFT", "TEE", "LINE_RIGHT", "VARACTOR",
                       "IPR_VAR", "STUB"):
            assert f"{prefix}_C{k}" in ids, f"missing {prefix}_C{k}"


def test_distinct_num_on_two_sources_and_output_port() -> None:
    """Signal Num=1, pump Num=2, output Port Num=3 -- no duplicates.

    The Level-7/8 canonicals both put the two P_1Tones at Num=1, which
    makes hpeesofsim's S_Param controller error with "Duplicate Port or
    Term number for SIGNAL_SRC" and aborts HB end-to-end (verified by
    direct live simulation). Level-9 must
    avoid that trap; this test pins it down.
    """

    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        nums = {s.name: s.num for s in spec.sources}
        assert nums["SIGNAL_SRC"] == 1
        assert nums["PUMP_SRC"] == 2
        # The output Port carries the third distinct Num so the three
        # Port: lines in the OA-generated netlist do not collide.
        assert {p.num for p in spec.ports} == {3}


def test_per_cell_ground_node_per_canonical() -> None:
    """One ground node per cell (the stub-end termination)."""

    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        n_cells = sum(1 for e in spec.elements if e.type == "MTEE")
        assert len(spec.ground_nodes) == n_cells
        # The grounded nets are the n_c{k}_g stub ends.
        assert set(spec.ground_nodes) == {
            f"n_c{k}_g" for k in range(1, n_cells + 1)
        }


# --------------------------------------------------------------------------
# Validation + emit
# --------------------------------------------------------------------------


def test_canonicals_validate() -> None:
    """Every Level-9 canonical passes the device-spec validator (resolved)."""

    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        validate_device_spec(resolve_spec(spec))


def test_hpeesofsim_deck_carries_every_i_probe_short_line() -> None:
    """Every I_Probe instance emits its ``Short:`` line."""

    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        iprobes = [e for e in spec.elements if e.type == "I_Probe"]
        for probe in iprobes:
            line_prefix = f"Short:{probe.id} "
            matching = [
                line for line in deck.splitlines()
                if line.startswith(line_prefix)
            ]
            assert matching, f"no Short line for {probe.id} in {spec.label}"
            assert "Mode=0 SaveCurrent=yes" in matching[0]


def test_make_probed_tvtl_cascade_rejects_zero_cells() -> None:
    """The builder refuses an empty cascade."""

    with pytest.raises(ValueError, match="n_cells must be >= 1"):
        make_probed_tvtl_cascade(n_cells=0, label="x")


def test_make_probed_tvtl_cascade_single_cell_degenerate() -> None:
    """n_cells=1 is a valid degenerate case (no inter-cell probes)."""

    spec = make_probed_tvtl_cascade(n_cells=1, label="lv9_one_cell")
    validate_device_spec(resolve_spec(spec))
    iprobes = [e.id for e in spec.elements if e.type == "I_Probe"]
    # Three boundary probes (IPR_IN / IPR_OUT) + one varactor probe.
    assert set(iprobes) == {"IPR_IN", "IPR_VAR_C1", "IPR_OUT"}


# --------------------------------------------------------------------------
# Content checks against the emitted deck
# --------------------------------------------------------------------------


def test_canonicals_pass_the_sdd2p_content_check() -> None:
    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        result = check_sdd2p_content(spec, hpeesofsim_netlist_text=deck)
        assert result.match, result.failures
        # One SDD2P checked per cell.
        n_cells = sum(1 for e in spec.elements if e.type == "MTEE")
        assert len(result.checked) == n_cells


def test_canonicals_pass_the_var_content_check() -> None:
    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        result = check_var_content(spec, hpeesofsim_netlist_text=deck)
        assert result.match, result.failures


def test_canonicals_pass_the_p1tone_content_check() -> None:
    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        result = check_p1tone_content(spec, hpeesofsim_netlist_text=deck)
        assert result.match, result.failures


def test_canonicals_pass_the_hb_two_tone_content_check() -> None:
    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        result = check_hb_two_tone_content(spec, hpeesofsim_netlist_text=deck)
        assert result.match, result.failures


def test_self_compare_match_on_emitted_deck() -> None:
    """The emitted deck cross-checks MATCH against itself (I_Probes paired)."""

    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        deck = emit_hpeesofsim_netlist(spec)
        result = compare_circuits(
            parse_netlist_text(deck), parse_netlist_text(deck)
        )
        assert result.match, result.differences
        # And every I_PROBE element is a counted element under the
        # netlist_compare normalisation (Short:... Mode=0 -> I_PROBE).
        circuit = parse_netlist_text(deck)
        n_iprobes_emitted = sum(1 for c in circuit.elements if c.ctype == "I_PROBE")
        n_iprobes_spec = sum(1 for e in spec.elements if e.type == "I_Probe")
        assert n_iprobes_emitted == n_iprobes_spec


# --------------------------------------------------------------------------
# OA emitter (stub keysight)
# --------------------------------------------------------------------------


def test_oa_emitter_accepts_both_canonicals(tmp_path: Path, monkeypatch) -> None:
    """The OA emitter builds a workspace for each canonical without raising.

    Exercised against the stub ``keysight.ads.de`` -- the emitter must
    place every per-cell MLIN / MTEE / SDD2P / I_Probe instance and
    route the cascade (including the shunt-branch I_Probe at every
    varactor), all without an ADS runtime. The hpeesofsim deck written
    alongside carries every Short:IPR_* line.
    """

    _install_fake_keysight(monkeypatch)
    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        info = oa_emitter.emit_workspace_from_spec(
            spec, tmp_path / spec.label / "oa_workspace"
        )
        hpeesofsim = Path(info["hpeesofsim_netlist_path"]).read_text(
            encoding="utf-8"
        )
        # every I_Probe shows up as a Short line.
        for probe in (e for e in spec.elements if e.type == "I_Probe"):
            assert f"Short:{probe.id} " in hpeesofsim


def test_topology_svg_renders_for_both_canonicals(tmp_path: Path) -> None:
    """The topology renderer draws the probed cascade without grid fallback."""

    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        out = tmp_path / f"{spec.label}.svg"
        render_topology_svg(spec, out)
        text = out.read_text(encoding="utf-8")
        # One I_Probe box per probe element.
        n_iprobes = sum(1 for e in spec.elements if e.type == "I_Probe")
        assert text.count('data-type="I_Probe"') == n_iprobes


# --------------------------------------------------------------------------
# CLI registration
# --------------------------------------------------------------------------


def test_oa_canonical_cli_registers_level_9() -> None:
    """oa_canonical_cli has a level9 entry pointing at the two canonicals."""

    assert "level9" in oa_canonical_cli._LEVELS
    circuits, label, _tol = oa_canonical_cli._LEVELS["level9"]
    assert label == LEVEL_9_LABEL
    assert circuits is LEVEL_9_CANONICAL_CIRCUITS


def test_cli_level9_processes_both_canonicals(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """`oa_canonical_cli level9` processes the two Level-9 canonicals."""

    monkeypatch.setattr(oa_emitter, "emit_workspace_from_spec", _stub_emit)
    monkeypatch.setattr(
        oa_canonical_cli, "_leg_b",
        lambda spec, d, label, tol: {
            "reason": "blocked", "max_abs_delta_dB": float("nan"),
        },
    )
    rc = oa_canonical_cli.main(["level9", "--output_root", str(tmp_path)])
    out = capsys.readouterr().out
    for spec in LEVEL_9_CANONICAL_CIRCUITS:
        assert spec.label in out
    # Cross-check + content rows both report MATCH for each canonical.
    assert out.count("MATCH") >= 2 * len(LEVEL_9_CANONICAL_CIRCUITS)
    assert rc == 0


