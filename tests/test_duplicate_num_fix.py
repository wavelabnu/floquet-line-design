"""Regression test for the Level-7/Level-8 duplicate-Num=1 bug.

The Level-7 ``make_two_tone_p1tone_sources`` and
``make_tvtl_unit_cell_two_tone_hb`` canonicals -- plus the Level-8
``_cascade_design`` expanded via the design API -- historically emitted
both ``P_1Tone:`` sources with ``Num=1``. hpeesofsim's S_Param controller
then aborted with ``Duplicate Port or Term number for SIGNAL_SRC`` and
the HB run never started. The structural cross-check (Path A) tolerated
the bug because both sources were present in both the OA and the
hpeesofsim deck; it surfaced only when the deck was actually simulated.

The fix renumbers the canonicals so signal = ``Num=1``, pump = ``Num=2``,
output TermG = ``Num=3``. These tests guard against the bug returning.

Two checks per canonical:

* **structural (ADS-free, always runs)** -- emit the deck and assert
  that every ``P_1Tone:`` and ``Port:`` line carries a distinct
  ``Num=`` value;
* **HB-runtime (ADS-bound)** -- when hpeesofsim is reachable, run the
  deck end-to-end and assert returncode 0 with no
  ``"Duplicate Port or Term number"`` text in the simulator's stderr.
  Skipped automatically on hosts without an ADS install.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from tvtl_ads_builder_v2.design.canonicals import (
    level8_five_cell_tvtl_cascade,
    level8_three_cell_tvtl_cascade,
)
from tvtl_ads_builder_v2.device_spec import resolve_spec
from tvtl_ads_builder_v2.netlist import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_tvtl_unit_cell_two_tone_hb,
    make_two_tone_p1tone_sources,
)
from tvtl_ads_builder_v2.simulate import run_hpeesofsim


_NUM_PATTERN = re.compile(r"\bNum=(\d+)\b")
_PORTLINE_PATTERN = re.compile(r"^(P_1Tone:|Port:)", re.MULTILINE)


def _spec_factories():
    """Return the four canonicals that historically carried the bug.

    Level-7 isolated source canonical, Level-7 integrated unit cell,
    Level-8 three-cell cascade, Level-8 five-cell cascade.
    """

    return [
        ("level7_two_tone_p1tone_sources",
         lambda: make_two_tone_p1tone_sources(label="t")),
        ("level7_tvtl_unit_cell_two_tone_hb",
         lambda: make_tvtl_unit_cell_two_tone_hb(label="t")),
        ("level8_three_cell_tvtl_cascade",
         level8_three_cell_tvtl_cascade),
        ("level8_five_cell_tvtl_cascade",
         level8_five_cell_tvtl_cascade),
    ]


def _emit_netlist(spec_factory) -> str:
    spec = spec_factory()
    # Level-8 designs carry an unresolved VAR block; the netlist emitter
    # expects literals, so resolve first (mirrors the parity-gate path).
    return emit_hpeesofsim_netlist(resolve_spec(spec))


@pytest.mark.parametrize(
    "label,spec_factory",
    _spec_factories(),
    ids=[f[0] for f in _spec_factories()],
)
def test_no_duplicate_num_on_port_or_source_lines(label, spec_factory):
    """Every ``P_1Tone:`` / ``Port:`` line carries a unique ``Num=``."""

    netlist = _emit_netlist(spec_factory)
    nums = []
    for line in netlist.splitlines():
        if not _PORTLINE_PATTERN.match(line):
            continue
        match = _NUM_PATTERN.search(line)
        assert match is not None, (
            f"{label}: source/port line is missing a Num= field: {line!r}"
        )
        nums.append((int(match.group(1)), line))
    assert nums, f"{label}: no P_1Tone/Port lines found in the netlist"
    seen: dict[int, str] = {}
    for num, line in nums:
        prior = seen.get(num)
        assert prior is None, (
            f"{label}: duplicate Num={num} on two distinct port/source "
            f"lines (this is the pre-existing duplicate-Num=1 bug):\n"
            f"  {prior!r}\n  {line!r}"
        )
        seen[num] = line


def _ads_available() -> bool:
    """Heuristic: hpeesofsim is reachable via $HPEESOF_DIR or PATH."""

    import os
    import shutil

    if os.environ.get("HPEESOF_DIR"):
        return True
    for name in ("adssim_win.exe", "adssim", "hpeesofsim.exe", "hpeesofsim"):
        if shutil.which(name):
            return True
    return False


@pytest.mark.skipif(
    not _ads_available(),
    reason="hpeesofsim not on PATH and HPEESOF_DIR is unset",
)
@pytest.mark.parametrize(
    "label,spec_factory",
    _spec_factories(),
    ids=[f[0] for f in _spec_factories()],
)
def test_hpeesofsim_does_not_report_duplicate_port_error(label, spec_factory):
    """End-to-end: the duplicate-Num=1 error does not appear in stderr.

    The decks may still fail to run for other reasons (Level-7/8 decks
    have pre-existing issues unrelated to this bug, such as missing
    ``P_1Tone`` model registration in the S_Param-only path), so this
    test does *not* assert returncode 0. The strict assertion is that
    the *specific* duplicate-Num error from this bug never appears.
    """

    netlist = _emit_netlist(spec_factory)
    with tempfile.TemporaryDirectory() as tmp:
        netlist_path = Path(tmp) / f"{label}.ckt"
        netlist_path.write_text(netlist, encoding="utf-8")
        result = run_hpeesofsim(netlist_path)

    assert result["status"] != "blocked_missing_ads_runtime", (
        "hpeesofsim was advertised but did not start; check $HPEESOF_DIR"
    )
    stderr = result.get("stderr_tail") or ""
    stdout = result.get("stdout_tail") or ""
    combined = stderr + "\n" + stdout
    assert "Duplicate Port or Term number" not in combined, (
        f"{label}: hpeesofsim reported the duplicate-Num=1 error in its "
        f"output:\nstdout tail: {stdout!r}\nstderr tail: {stderr!r}"
    )


