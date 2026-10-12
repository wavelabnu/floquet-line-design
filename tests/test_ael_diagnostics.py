"""Tests for the AEL diagnostic ladder D0-D4.

All tests run with no ADS installed: the generation tests force the
no-ADS path (Path 1 reported BLOCKED) so they are fast and
environment-independent.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from tvtl_ads_builder_v2.device_spec import validate_device_spec
from tvtl_ads_builder_v2.inspection.ads_workspace import (
    REFERENCE_LIBRARY_NAME,
    REFERENCE_WORKSPACE_PATH,
)
from tvtl_ads_builder_v2.inspection.ael_templates import render_ael_script
from tvtl_ads_builder_v2.inspection.schematic_layout import build_schematic_plan
from tvtl_ads_builder_v2.inspection.verified_ael_functions import (
    VERIFIED_AEL_FUNCTIONS,
)
from tvtl_ads_builder_v2.parity.ael_diagnostics import (
    DIAGNOSTIC_ORDER,
    DIAGNOSTIC_SPECS,
)
from tvtl_ads_builder_v2.parity.ael_diagnostics import generate as generate_mod
from tvtl_ads_builder_v2.parity.ael_diagnostics.expected import (
    compute_tvtl_expected,
)
from tvtl_ads_builder_v2.parity.ael_diagnostics.generate import (
    _path1_status,
    generate_one,
)

_EXPECTED_FILES = (
    "hpeesofsim.ckt",
    "build_workspace.ael",
    "open_workspace.bat",
    "tvtl_expected.csv",
    "hpeesofsim_result.csv",
    "path1_parity_report.txt",
    "README.md",
)


def _render_diagnostic_deck(name: str) -> str:
    """Render the AEL deck for one diagnostic, without touching disk."""

    spec = DIAGNOSTIC_SPECS[name]()
    plan = build_schematic_plan(spec)
    return render_ael_script(
        REFERENCE_LIBRARY_NAME, name, plan, "2026-01-01T00:00:00",
        REFERENCE_WORKSPACE_PATH,
    )


def _force_no_ads(monkeypatch, tmp_path: Path) -> None:
    """Force the no-ADS path so generation is fast and deterministic."""

    monkeypatch.setattr(generate_mod, "resolve_ads_environment", lambda: None)
    monkeypatch.delenv("HPEESOF_DIR", raising=False)
    empty = tmp_path / "_empty_path"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", str(empty))


_ELEMENT_HEADER = re.compile(r"^//   (\S+) \((\w+)\)$")


def _element_blocks(deck: str) -> list[tuple[str, str, list[str]]]:
    """Parse a deck's place section into per-element blocks.

    Returns one ``(item_id, element_type, code_lines)`` triple per
    placed element, in deck order. ``code_lines`` are the element's
    AEL statements (de_init_item / de_rotate_image / de_place_item /
    de_free_item), with the header comment and blank lines dropped.
    """

    blocks: list[tuple[str, str, list[str]]] = []
    current: tuple[str, str, list[str]] | None = None
    in_place = False
    for line in deck.splitlines():
        if line.startswith("// --- place elements"):
            in_place = True
            continue
        if line.startswith("// --- wire connections"):
            break
        if not in_place:
            continue
        header = _ELEMENT_HEADER.match(line)
        if header:
            current = (header.group(1), header.group(2), [])
            blocks.append(current)
        elif current is not None and line.strip():
            if not line.lstrip().startswith("//"):
                current[2].append(line.strip())
    return blocks


def test_all_diagnostic_specs_build() -> None:
    """All five diagnostic specs build and validate without exception."""

    assert DIAGNOSTIC_ORDER == (
        "d0_bare_ports",
        "d1_single_mlin",
        "d2_two_mlins",
        "d3_mtee_short_stub",
        "d4_full_level3",
    )
    for name in DIAGNOSTIC_ORDER:
        spec = DIAGNOSTIC_SPECS[name]()
        validate_device_spec(spec)
        assert spec.label == name


def test_all_tvtl_expected_finite() -> None:
    """Every diagnostic's TVTL expected |S21| is finite (no NaN/Inf)."""

    for name in DIAGNOSTIC_ORDER:
        spec = DIAGNOSTIC_SPECS[name]()
        expected = compute_tvtl_expected(spec)
        assert expected["freq_Hz"].size == 91
        assert np.all(np.isfinite(expected["S21_dB"])), name
        assert np.all(np.isfinite(expected["S21_phase_deg"])), name


def test_generate_one_produces_all_files(tmp_path: Path, monkeypatch) -> None:
    """generate_one writes every expected per-circuit file."""

    _force_no_ads(monkeypatch, tmp_path)
    diagnostics_root = tmp_path / "ael_diagnostics"
    info = generate_one("d1_single_mlin", diagnostics_root)

    circuit_dir = diagnostics_root / "d1_single_mlin"
    assert info["circuit_dir"] == circuit_dir
    for filename in _EXPECTED_FILES:
        assert (circuit_dir / filename).is_file(), f"missing {filename}"
    # With no ADS runtime, Path 1 is blocked (not a crash).
    assert info["path1_status"] == "BLOCKED"
    report = (circuit_dir / "path1_parity_report.txt").read_text(encoding="utf-8")
    assert "BLOCKED" in report


def test_path1_status_mapping() -> None:
    """`_path1_status` maps parity-check reasons to status strings."""

    assert _path1_status({"reason": "within_tolerance"}) == "PASS"
    assert _path1_status({"reason": "exceeds_tolerance"}) == "FAIL"
    assert _path1_status({"reason": "ads_blocked"}) == "BLOCKED"
    assert _path1_status({"reason": "tvtl_error"}) == "ERROR"


def test_generated_ael_decks_use_only_verified_functions() -> None:
    """Every `de_*` call in a diagnostic deck is evidence-pinned."""

    for name in DIAGNOSTIC_ORDER:
        deck = _render_diagnostic_deck(name)
        code = "\n".join(
            ln for ln in deck.splitlines() if not ln.lstrip().startswith("//")
        )
        for function_name in set(re.findall(r"\b(de_\w+)\s*\(", code)):
            assert function_name in VERIFIED_AEL_FUNCTIONS, (
                f"{name}: deck calls unverified function {function_name!r}"
            )


def test_generated_ael_decks_open_reference_workspace() -> None:
    """Each diagnostic deck self-bootstraps via de_open_workspace."""

    for name in DIAGNOSTIC_ORDER:
        deck = _render_diagnostic_deck(name)
        code_lines = [
            ln for ln in deck.splitlines()
            if ln.strip() and not ln.lstrip().startswith("//")
        ]
        assert code_lines[0] == "/* Begin */"
        assert code_lines[1].startswith("de_open_workspace("), code_lines[1]
        assert "<ADS_REFERENCE_WORKSPACE>" in code_lines[1]


def test_d1_deck_has_two_wire_blocks() -> None:
    """D1 (single MLIN) emits one wire per device-spec connection node.

    D1's spec has two connection nodes -- n_in (port 1 <-> MLIN pin 1)
    and n_out (MLIN pin 2 <-> port 2) -- so the deck must contain two
    de_connect/de_add_wire/de_end blocks.
    """

    deck = _render_diagnostic_deck("d1_single_mlin")
    assert deck.count("de_connect();") == 2
    assert deck.count("de_end();") == 2
    assert "node n_in" in deck
    assert "node n_out" in deck


def test_termg_placements_rotated_down() -> None:
    """Every TermG block rotates DOWN before its de_place_item.

    The ADS Command Line log captured a TermG placed only after
    de_rotate_image("DOWN"); the deck must reproduce that so the port
    pin lands on the placement origin the wiring plan assumes.
    """

    for name in DIAGNOSTIC_ORDER:
        blocks = _element_blocks(_render_diagnostic_deck(name))
        terms = [b for b in blocks if b[1] == "TermG"]
        assert terms, f"{name}: no TermG block found"
        for item_id, _type, code in terms:
            label = f"{name}/{item_id}"
            assert "de_init_item(" in code[0], label
            assert code[1] == 'de_rotate_image("DOWN");', label
            assert code[2].startswith("de_place_item("), label


def test_controllers_rotated_right() -> None:
    """The S_Param and SweepPlan controllers rotate RIGHT before placement.

    The Command Line log placed both analysis controllers after
    de_rotate_image("RIGHT"); every deck carries both controllers.
    """

    for name in DIAGNOSTIC_ORDER:
        blocks = _element_blocks(_render_diagnostic_deck(name))
        seen = set()
        for item_id, etype, code in blocks:
            if etype not in ("S_Param", "SweepPlan"):
                continue
            seen.add(etype)
            assert code[1] == 'de_rotate_image("RIGHT");', f"{name}/{item_id}"
        assert seen == {"S_Param", "SweepPlan"}, f"{name}: missing controller"


def test_default_orientation_elements_emit_no_rotation() -> None:
    """MLIN / MTEE / MLOC / VIAGND / GROUND / MSUB carry no de_rotate_image.

    The Command Line log placed these in ADS's default orientation;
    emitting a rotation for them would be wrong and would risk leaking
    onto the next element.
    """

    unrotated = {"MLIN", "MTEE", "MLOC", "VIAGND", "GROUND", "MSUB"}
    for name in DIAGNOSTIC_ORDER:
        for item_id, etype, code in _element_blocks(_render_diagnostic_deck(name)):
            if etype not in unrotated:
                continue
            for stmt in code:
                assert not stmt.startswith("de_rotate_image("), (
                    f"{name}/{item_id} ({etype}) emits a rotation"
                )


def test_blocks_end_at_free_item_with_no_rotation_reset() -> None:
    """No element block emits a de_rotate_image after its de_free_item.

    The ADS Command Line log went straight from one element's
    de_free_item to the next element's de_init_item -- no rotation
    reset between blocks. A fresh de_init_item starts in the default
    orientation, and de_rotate_image("UP") is itself a rotation, not a
    neutral reset, so emitting one would mis-orient the next element.
    Every block therefore ends at de_free_item: a rotated block is
    exactly init / rotate / place / free, an unrotated block is
    init / place / free.
    """

    for name in DIAGNOSTIC_ORDER:
        blocks = _element_blocks(_render_diagnostic_deck(name))
        rotated_seen = False
        for item_id, etype, code in blocks:
            label = f"{name}/{item_id} ({etype})"
            assert "de_free_item(" in code[-1], (
                f"{label}: block must end at de_free_item -- got {code}"
            )
            rotates = [s for s in code if s.startswith("de_rotate_image(")]
            assert len(rotates) <= 1, f"{label}: more than one rotation"
            if rotates:
                rotated_seen = True
                assert len(code) == 4, f"{label}: {code}"
                assert "de_init_item(" in code[0], label
                assert code[1].startswith("de_rotate_image("), label
                assert code[2].startswith("de_place_item("), label
                assert "de_free_item(" in code[3], label
            else:
                assert len(code) == 3, f"{label}: {code}"
                assert "de_init_item(" in code[0], label
                assert code[1].startswith("de_place_item("), label
                assert "de_free_item(" in code[2], label
        assert rotated_seen, f"{name}: expected at least one rotated block"


def test_diagnostic_decks_no_coincident_placements_or_wires() -> None:
    """No two elements share a placement coordinate, and every wire has
    two distinct endpoints (a zero-length wire is rejected by ADS)."""

    place_re = re.compile(
        r"de_place_item\([^,]+,\s*(-?[\d.]+),\s*(-?[\d.]+)\)"
    )
    wire_re = re.compile(r"de_add_wire\(\s*(-?[\d.]+),\s*(-?[\d.]+)\)")
    for name in DIAGNOSTIC_ORDER:
        deck = _render_diagnostic_deck(name)
        placements = [
            (float(x), float(y)) for x, y in place_re.findall(deck)
        ]
        assert placements, f"{name}: no placements found"
        assert len(placements) == len(set(placements)), (
            f"{name}: two elements share a placement coordinate"
        )
        wire_points = [
            (float(x), float(y)) for x, y in wire_re.findall(deck)
        ]
        # de_add_wire calls come in (start, end) pairs.
        assert len(wire_points) % 2 == 0, f"{name}: odd de_add_wire count"
        for index in range(0, len(wire_points), 2):
            start, end = wire_points[index], wire_points[index + 1]
            assert start != end, (
                f"{name}: zero-length wire at {start}"
            )


def test_cli_generate_runs(tmp_path: Path, monkeypatch) -> None:
    """The CLI `generate` runs without exception and builds all workspaces."""

    _force_no_ads(monkeypatch, tmp_path)
    from tvtl_ads_builder_v2.parity import ael_diagnostics_cli

    rc = ael_diagnostics_cli.main(["generate", "--output_root", str(tmp_path)])
    # No ADS runtime -> every Path 1 blocked -> exit 4 (none failed).
    assert rc == 4
    for name in DIAGNOSTIC_ORDER:
        circuit_dir = tmp_path / "ael_diagnostics" / name
        assert (circuit_dir / "README.md").is_file()
        assert (circuit_dir / "build_workspace.ael").is_file()


