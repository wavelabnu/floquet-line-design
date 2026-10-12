"""Cross-level inspection index for ADS Builder v2.

:func:`update_global_index` rebuilds ``ads_builder_v2_inspection_index.md``
at the output root by scanning every ``ads_builder_v2_level_<N>/``
directory for an ``inspection_status.json``. The rebuild is
*monotonic*: it reads whatever per-level status files are on disk, so
running one level's parity CLI refreshes that level's row while every
other level's row is reconstructed unchanged from its own (untouched)
status file.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

GLOBAL_INDEX_FILENAME = "ads_builder_v2_inspection_index.md"
_LEVEL_DIR_RE = re.compile(r"^ads_builder_v2_level_(\d+)$")


def update_global_index(output_root: str | Path) -> Path:
    """Rebuild the cross-level inspection index from on-disk status files.

    Parameters
    ----------
    output_root : str or Path
        The directory holding the per-level parity roots
        (``ads_builder_v2_level_1``, ``ads_builder_v2_level_2``, ...).

    Returns
    -------
    Path
        The path of the written ``ads_builder_v2_inspection_index.md``.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    levels = _collect_level_status(output_root)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    lines: list[str] = []
    lines.append("# ADS Builder v2 -- Inspection Index (all levels)\n\n")
    lines.append(
        "This index aggregates the inspection status of every ADS "
        "Builder v2 ladder level. It is rebuilt monotonically: each "
        "level's parity CLI refreshes only its own `inspection_status.json`, "
        "and this index is then reconstructed from whatever per-level "
        "status files are present -- so running one level never erases "
        "another level's row.\n\n"
    )
    lines.append(f"**Last updated**: {timestamp}\n\n")

    if not levels:
        lines.append(
            "_No level inspection status found yet. Run a parity CLI "
            "(`ads_builder_v2_parity_level_<N>_cli`) to populate this "
            "index._\n"
        )
        path = output_root / GLOBAL_INDEX_FILENAME
        path.write_text("".join(lines), encoding="utf-8")
        return path

    lines.append(
        "| Level | Description | Circuits | Pass | Fail | Blocked | "
        "Index |\n"
    )
    lines.append(
        "|-------|-------------|----------|------|------|---------|"
        "-------|\n"
    )
    for level in levels:
        number = level["level_number"]
        link = (
            f"[level {number}]"
            f"(ads_builder_v2_level_{number}/inspection_index.md)"
        )
        lines.append(
            f"| {number} | {_escape_cell(str(level['level_description']))} "
            f"| {level['n_circuits']} | {level['n_pass']} | "
            f"{level['n_fail']} | {level['n_blocked']} | {link} |\n"
        )
    lines.append("\n")

    total_circuits = sum(int(lvl["n_circuits"]) for lvl in levels)
    total_pass = sum(int(lvl["n_pass"]) for lvl in levels)
    total_fail = sum(int(lvl["n_fail"]) for lvl in levels)
    total_blocked = sum(int(lvl["n_blocked"]) for lvl in levels)
    lines.append(
        f"**Total**: {total_pass} pass, {total_fail} fail, "
        f"{total_blocked} blocked (of {total_circuits} circuits across "
        f"{len(levels)} levels).\n\n"
    )
    lines.append(
        "Open a level's `inspection_index.md` for the per-circuit table "
        "and failure summary; each circuit there links to its own "
        "`inspection_README.md`.\n"
    )

    path = output_root / GLOBAL_INDEX_FILENAME
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _collect_level_status(output_root: Path) -> list[dict[str, Any]]:
    """Read every ``ads_builder_v2_level_<N>/inspection_status.json``.

    Returns the parsed status dicts sorted by level number. Directories
    without a readable status file are skipped, so a partially-run
    ladder still produces a coherent index.
    """

    levels: list[dict[str, Any]] = []
    for entry in sorted(output_root.iterdir()):
        if not entry.is_dir():
            continue
        match = _LEVEL_DIR_RE.match(entry.name)
        if match is None:
            continue
        status_path = entry / "inspection_status.json"
        if not status_path.is_file():
            continue
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        data.setdefault("level_number", int(match.group(1)))
        for key in ("n_circuits", "n_pass", "n_fail", "n_blocked"):
            data.setdefault(key, 0)
        data.setdefault("level_description", "(no description)")
        levels.append(data)
    levels.sort(key=lambda d: int(d["level_number"]))
    return levels


def _escape_cell(text: str) -> str:
    """Escape ``|`` so a description never breaks a markdown table row."""

    return text.replace("|", r"\|")


