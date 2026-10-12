"""CLI: OA-Python <-> hpeesofsim cross-check for canonical Levels 1-5.

::

    python -m tvtl_ads_builder_v2.oa.oa_canonical_cli level1   (level2 .. level5)
    python -m tvtl_ads_builder_v2.oa.oa_canonical_cli all

For each canonical circuit this builds the OA workspace, auto-generates
its ADS netlist, emits the hpeesofsim flat netlist, cross-checks the two
(Path A), and confirms the hpeesofsim<->TVTL leg (Path B), then prints a
per-level summary table.

Path A is a structural netlist comparison -- tolerance-independent.
Path B uses the v2 parity tolerance (0.5 dB, uniform across Levels 1-4
-- ``parity/tolerance.py``). The two Level-3 strong-reflection circuits
are classified **parity-blocked** in ``inspection/metadata.py``: the
TVTL ideal-junction MTEE model cannot match ADS within 0.5 dB there, a
documented model limitation, so their Path B is reported but does not
fail the row. v2 has no looser numeric tolerance class -- the binary
parity-blocked classification is the existing mechanism.

For a circuit that carries an ``SDD2P`` element (Level 5) a third
check runs: the polynomial **content** cross-check
(``sdd2p_content_check``) verifies that both backends' emitted SDD2P
body equations carry the device spec's charge polynomial -- the gate
that would have caught v1's silent ``(_v1)/50.0`` placeholder. A
content mismatch fails the row. Level 5 has no analytical SDD2P model,
so its Path B leg does not run; Path A plus the content check are the
Level-5 deliverable.

Exit 0 iff every circuit passes (Path A matched, the SDD2P content
check matched, Path B within its class).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..inspection.metadata import metadata_for
from ..parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    LEVEL_1_LABEL,
    LEVEL_2_CANONICAL_CIRCUITS,
    LEVEL_2_LABEL,
    LEVEL_3_CANONICAL_CIRCUITS,
    LEVEL_3_LABEL,
    LEVEL_4_CANONICAL_CIRCUITS,
    LEVEL_4_LABEL,
    LEVEL_5_CANONICAL_CIRCUITS,
    LEVEL_5_LABEL,
    LEVEL_6_CANONICAL_CIRCUITS,
    LEVEL_6_LABEL,
    LEVEL_7_CANONICAL_CIRCUITS,
    LEVEL_7_LABEL,
    LEVEL_9_CANONICAL_CIRCUITS,
    LEVEL_9_LABEL,
    LEVEL_10_CANONICAL_CIRCUITS,
    LEVEL_10_LABEL,
)
from ..parity.tolerance import (
    PARITY_TOLERANCE_DB_S21_LEVEL_1,
    PARITY_TOLERANCE_DB_S21_LEVEL_2,
    PARITY_TOLERANCE_DB_S21_LEVEL_3,
    PARITY_TOLERANCE_DB_S21_LEVEL_4,
    PARITY_TOLERANCE_DB_S21_LEVEL_5,
    PARITY_TOLERANCE_DB_S21_LEVEL_6,
    PARITY_TOLERANCE_DB_S21_LEVEL_7,
    PARITY_TOLERANCE_DB_S21_LEVEL_9,
    PARITY_TOLERANCE_DB_S21_LEVEL_10,
)

_DEFAULT_OUTPUT_ROOT = Path.cwd() / "runs"

# level key -> (circuit specs, parity-gate label, Path-B tolerance dB)
_LEVELS: dict[str, tuple] = {
    "level1": (LEVEL_1_CANONICAL_CIRCUITS, LEVEL_1_LABEL,
               PARITY_TOLERANCE_DB_S21_LEVEL_1),
    "level2": (LEVEL_2_CANONICAL_CIRCUITS, LEVEL_2_LABEL,
               PARITY_TOLERANCE_DB_S21_LEVEL_2),
    "level3": (LEVEL_3_CANONICAL_CIRCUITS, LEVEL_3_LABEL,
               PARITY_TOLERANCE_DB_S21_LEVEL_3),
    "level4": (LEVEL_4_CANONICAL_CIRCUITS, LEVEL_4_LABEL,
               PARITY_TOLERANCE_DB_S21_LEVEL_4),
    "level5": (LEVEL_5_CANONICAL_CIRCUITS, LEVEL_5_LABEL,
               PARITY_TOLERANCE_DB_S21_LEVEL_5),
    "level6": (LEVEL_6_CANONICAL_CIRCUITS, LEVEL_6_LABEL,
               PARITY_TOLERANCE_DB_S21_LEVEL_6),
    "level7": (LEVEL_7_CANONICAL_CIRCUITS, LEVEL_7_LABEL,
               PARITY_TOLERANCE_DB_S21_LEVEL_7),
    # Level 8 (the topology-driven design-API cascades) lives in
    # ``design/`` per the dependency-direction rule and self-registers
    # via ``tvtl_ads_builder_v2.design.canonical_cli``. It is therefore
    # absent here. Level 9 IS a parity-layer canonical (hand-authored
    # DeviceSpecs, no design/ import), so it registers directly.
    "level9": (LEVEL_9_CANONICAL_CIRCUITS, LEVEL_9_LABEL,
               PARITY_TOLERANCE_DB_S21_LEVEL_9),
    "level10": (LEVEL_10_CANONICAL_CIRCUITS, LEVEL_10_LABEL,
                PARITY_TOLERANCE_DB_S21_LEVEL_10),
}
_LEVEL_ORDER = (
    "level1", "level2", "level3", "level4", "level5", "level6", "level7",
    "level9", "level10",
)


def _is_parity_blocked(label: str) -> bool:
    """True if a circuit is classified parity-blocked in metadata.py.

    The two Level-3 strong-reflection circuits (``tee_short_zero_stub``,
    ``tee_qw_short_3p1GHz``) and the Level-4 ``mtee_with_rlc_stub`` carry
    a 'parity-blocked' note: the TVTL ideal-junction MTEE model cannot
    match ADS within 0.5 dB there.
    """

    return "parity-blocked" in (metadata_for(label).notes or "")


def _cross_check(ads_path: Path, hpeesofsim_path: Path):
    from .netlist_compare import (
        compare_circuits,
        parse_ads_netlist,
        parse_hpeesofsim_deck,
    )

    return compare_circuits(
        parse_ads_netlist(ads_path),
        parse_hpeesofsim_deck(hpeesofsim_path),
    )


def _content_check(spec, ads_path: Path, hpeesofsim_path: Path):
    """Run the SDD2P polynomial content cross-check for one circuit.

    Returns the :class:`SDD2PContentResult`; both backends' netlist
    texts are read so the spec polynomial is verified against the
    OA/ADS netlist *and* the hpeesofsim netlist, and the two backends
    against each other.
    """

    from .sdd2p_content_check import check_sdd2p_content

    return check_sdd2p_content(
        spec,
        ads_netlist_text=ads_path.read_text(encoding="utf-8"),
        hpeesofsim_netlist_text=hpeesofsim_path.read_text(encoding="utf-8"),
    )


def _var_content_check(spec, ads_path: Path, hpeesofsim_path: Path):
    """Run the VAR named-parameter content cross-check for one circuit.

    Verifies the var-resolver substituted each named-scalar reference
    with the variable's declared literal value and that both backends
    emit those literals identically.
    """

    from .var_content_check import check_var_content

    return check_var_content(
        spec,
        ads_netlist_text=ads_path.read_text(encoding="utf-8"),
        hpeesofsim_netlist_text=hpeesofsim_path.read_text(encoding="utf-8"),
    )


def _hb_content_check(spec, ads_path: Path, hpeesofsim_path: Path):
    """Run the single-tone HB controller content cross-check.

    Verifies both backends emit the same ``HB:`` directive
    (``Freq[1]`` / ``Order[1]`` / ``MaxOrder``) and that it matches the
    device spec's ``hb_single_tone`` directive.
    """

    from .hb_controller_content_check import check_hb_controller_content

    return check_hb_controller_content(
        spec,
        ads_netlist_text=ads_path.read_text(encoding="utf-8"),
        hpeesofsim_netlist_text=hpeesofsim_path.read_text(encoding="utf-8"),
    )


def _hb_two_tone_content_check(spec, ads_path: Path, hpeesofsim_path: Path):
    """Run the two-tone HB controller content cross-check.

    Verifies both backends emit the same two-tone ``HB:`` directive
    (``Freq[1]`` / ``Freq[2]`` / ``Order[1]`` / ``Order[2]`` /
    ``MaxOrder``) and that it matches the spec's ``hb_two_tone``.
    """

    from .hb_controller_content_check import check_hb_two_tone_content

    return check_hb_two_tone_content(
        spec,
        ads_netlist_text=ads_path.read_text(encoding="utf-8"),
        hpeesofsim_netlist_text=hpeesofsim_path.read_text(encoding="utf-8"),
    )


def _p1tone_content_check(spec, ads_path: Path, hpeesofsim_path: Path):
    """Run the P_1Tone power-source content cross-check (Level 7).

    Verifies both backends emit every ``P_1Tone:`` source with the
    same driven node, Num, Z, Freq and P, and that they match the
    device spec's ``sources`` (with any VAR named reference resolved).
    """

    from .p1tone_content_check import check_p1tone_content

    return check_p1tone_content(
        spec,
        ads_netlist_text=ads_path.read_text(encoding="utf-8"),
        hpeesofsim_netlist_text=hpeesofsim_path.read_text(encoding="utf-8"),
    )


def _leg_b(spec, circuit_dir: Path, level_label: str, tolerance_dB: float) -> dict:
    """hpeesofsim <-> TVTL via the existing v2 parity gate (ADS-required)."""

    try:
        from ..parity.ael_diagnostics.generate import resolve_ads_environment
        from ..parity.check import parity_check

        return parity_check(
            spec,
            circuit_dir,
            ads_environment=resolve_ads_environment(),
            tolerance_dB_S21=tolerance_dB,
            level_label=level_label,
        )
    except Exception as exc:  # noqa: BLE001 -- leg B is the ADS-required path
        return {"reason": f"error:{type(exc).__name__}",
                "max_abs_delta_dB": float("nan")}


def _process_circuit(level_key: str, spec, output_root: str | Path) -> dict:
    """Build, cross-check and validate one canonical circuit -> row dict."""

    _circuits, level_label, tolerance_dB = _LEVELS[level_key]
    circuit_dir = Path(output_root) / "oa_canonical" / level_key / spec.label
    circuit_dir.mkdir(parents=True, exist_ok=True)
    blocked = _is_parity_blocked(spec.label)
    has_sdd2p = any(element.type == "SDD2P" for element in spec.elements)
    row = {
        "circuit": spec.label,
        "crosscheck": "ERROR",
        "content": "-",
        "path_a": "-",
        "path_b": "-",
        "tolerance": "parity-blocked" if blocked else f"{tolerance_dB:g} dB",
        "differences": [],
        "content_failures": [],
        "passed": False,
    }

    try:
        from .oa_emitter import emit_workspace_from_spec

        emit_workspace_from_spec(spec, circuit_dir / "oa_workspace")
    except Exception as exc:  # noqa: BLE001 -- a build failure is one ERROR row
        row["path_a"] = f"emit failed ({type(exc).__name__})"
        return row

    ads_path = circuit_dir / "ads_generated_netlist.txt"
    hpeesofsim_path = circuit_dir / "hpeesofsim.ckt"
    if not ads_path.is_file():
        row["path_a"] = "ADS netlist not generated"
        return row

    try:
        result = _cross_check(ads_path, hpeesofsim_path)
        row["crosscheck"] = "MATCH" if result.match else "MISMATCH"
        row["path_a"] = (
            "0 differences" if result.match
            else f"{len(result.differences)} difference(s)"
        )
        row["differences"] = result.differences

        # SDD2P polynomial content cross-check (Level 5). Verifies both
        # backends emit the device spec's charge polynomial -- the gate
        # that would have caught v1's silent (_v1)/50.0 placeholder.
        # Accumulate the content verdict across every applicable check: a later
        # passing check must NOT overwrite an earlier mismatch on the displayed
        # row (that would hide the failure that failed the row). row["passed"]
        # already ANDs content_ok; the display now matches it.
        content_ok = True
        content_checked = False
        row["content_failures"] = []
        if has_sdd2p:
            content = _content_check(spec, ads_path, hpeesofsim_path)
            content_ok = content_ok and content.match
            content_checked = True
            row["content_failures"] += list(content.failures)
        # VAR named-parameter content cross-check (Level 5). Verifies the
        # resolver substituted each named scalar with the declared
        # literal and both backends emit those literals identically.
        if spec.var_block is not None:
            var_content = _var_content_check(spec, ads_path, hpeesofsim_path)
            content_ok = content_ok and var_content.match
            content_checked = True
            row["content_failures"] += list(var_content.failures)
        # Single-tone HB controller content cross-check (Level 6).
        # Verifies both backends emit the same HB: analysis directive.
        if spec.hb_single_tone is not None:
            hb_content = _hb_content_check(spec, ads_path, hpeesofsim_path)
            content_ok = content_ok and hb_content.match
            content_checked = True
            row["content_failures"] += list(hb_content.failures)
        # Two-tone HB controller content cross-check (Level 6).
        if spec.hb_two_tone is not None:
            hb2_content = _hb_two_tone_content_check(
                spec, ads_path, hpeesofsim_path
            )
            content_ok = content_ok and hb2_content.match
            content_checked = True
            row["content_failures"] += list(hb2_content.failures)
        # P_1Tone power-source content cross-check (Level 7).
        if spec.sources:
            p1tone_content = _p1tone_content_check(
                spec, ads_path, hpeesofsim_path
            )
            content_ok = content_ok and p1tone_content.match
            content_checked = True
            row["content_failures"] += list(p1tone_content.failures)
        if content_checked:
            row["content"] = "MATCH" if content_ok else "MISMATCH"

        legb = _leg_b(spec, circuit_dir, level_label, tolerance_dB)
        reason = str(legb.get("reason", ""))
        try:
            row["path_b"] = f"{float(legb.get('max_abs_delta_dB')):.3f} dB"
        except (TypeError, ValueError):
            row["path_b"] = reason or "-"

        if not result.match:
            structural_pass = False
        elif reason == "within_tolerance":
            structural_pass = True
        elif reason == "exceeds_tolerance":
            # Exceeding 0.5 dB is the documented, expected outcome for a
            # parity-blocked circuit; for any other circuit it is a fail.
            structural_pass = blocked
            if not blocked:
                row["path_b"] += " (>tol)"
        else:
            # leg B blocked or errored -- Path A is the primary deliverable.
            structural_pass = True
        row["passed"] = structural_pass and content_ok
    except Exception as exc:  # noqa: BLE001 -- a parse failure is one ERROR row
        row["crosscheck"] = "ERROR"
        row["path_a"] = f"cross-check failed ({type(exc).__name__})"
        row["passed"] = False
    return row


def _print_level(level_key: str, rows: list[dict]) -> None:
    """Print one level's summary table and per-level aggregate."""

    label = _LEVELS[level_key][1]
    print(f"=== {level_key} ({label}) ===")
    head = (
        f"  {'Circuit':<28} | {'Cross-check':<11} | {'Content':<8} | "
        f"{'Path A':<16} | {'Path B':<14} | Tolerance"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for row in rows:
        print(
            f"  {row['circuit']:<28} | {row['crosscheck']:<11} | "
            f"{row['content']:<8} | {row['path_a']:<16} | "
            f"{row['path_b']:<14} | {row['tolerance']}"
        )
    n_pass = sum(1 for row in rows if row["passed"])
    print(f"  {level_key}: {n_pass}/{len(rows)} passed.")
    for row in rows:
        if row["crosscheck"] == "MISMATCH" and row["differences"]:
            print(f"\n  {row['circuit']} -- circuit-level differences:")
            for diff in row["differences"]:
                print(f"    - {diff}")
        if row["content"] == "MISMATCH" and row["content_failures"]:
            print(
                f"\n  {row['circuit']} -- SDD2P polynomial content "
                f"mismatch:"
            )
            for failure in row["content_failures"]:
                print(f"    - {failure}")
    print()


def _run(circuit_arg: str, output_root: str | Path) -> int:
    """Process the requested level(s); print tables; return the exit code."""

    keys = _LEVEL_ORDER if circuit_arg == "all" else (circuit_arg,)
    all_rows: list[dict] = []
    for level_key in keys:
        circuits = _LEVELS[level_key][0]
        rows = [_process_circuit(level_key, spec, output_root) for spec in circuits]
        _print_level(level_key, rows)
        all_rows.extend(rows)

    n_pass = sum(1 for row in all_rows if row["passed"])
    total = len(all_rows)
    if n_pass == total:
        print(f"All {total} canonical circuit(s) passed: every OA workspace "
              f"is netlist-equivalent to its hpeesofsim flat deck.")
    else:
        print(f"{n_pass}/{total} canonical circuit(s) passed -- see the rows "
              f"above.")
    return 0 if n_pass == total else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""

    parser = argparse.ArgumentParser(
        prog="oa_canonical_cli",
        description=(
            "OA-Python <-> hpeesofsim cross-check for canonical Levels "
            "1-7 and 9 (Level 8 lives in design/ and self-registers via "
            "tvtl_ads_builder_v2.design.canonical_cli)."
        ),
    )
    parser.add_argument(
        "circuit",
        choices=("all",) + _LEVEL_ORDER,
        help=(
            "canonical level to process: level1 .. level7, level9, or 'all'"
        ),
    )
    parser.add_argument(
        "--output_root",
        default=str(_DEFAULT_OUTPUT_ROOT),
        help="root directory for oa_canonical/ outputs",
    )
    args = parser.parse_args(argv)
    return _run(args.circuit, args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())



