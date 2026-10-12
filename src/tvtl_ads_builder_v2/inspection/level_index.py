"""Per-level inspection index for ADS Builder v2.

Each ladder level's parity run produces one ``inspection_index.md`` and
one ``inspection_status.json`` in its parity-root directory
(``ads_builder_v2_level_<N>/``):

* ``inspection_index.md`` -- a human-readable circuit table, a failure
  summary, and a navigation pointer to the per-circuit READMEs and the
  cross-level index.
* ``inspection_status.json`` -- the machine-readable status the
  cross-level index (:mod:`tvtl_ads_builder_v2.inspection.global_index`)
  aggregates. It is the per-level record on disk, so re-running one
  level never erases another level's row in the global index.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metadata import LEVEL_DESCRIPTIONS, metadata_for


def write_level_index(
    level_dir: str | Path,
    level_number: int,
    results: Sequence[Mapping[str, Any]],
    *,
    level_title: str | None = None,
) -> dict[str, Path]:
    """Write ``inspection_index.md`` and ``inspection_status.json``.

    Parameters
    ----------
    level_dir : str or Path
        The level's parity-root directory
        (``<output_root>/ads_builder_v2_level_<N>``).
    level_number : int
        The ladder level number.
    results : sequence of Mapping
        The per-circuit parity-check results for the level. Each result
        must carry the circuit's ``label`` (the device-spec label),
        ``pass``, ``reason``, ``max_abs_delta_dB`` and ``tolerance_dB``.
    level_title : str or None
        The level's human-readable gate title; falls back to a generic
        title built from the level description.

    Returns
    -------
    dict
        ``{"index_path": Path, "status_path": Path}``.
    """

    level_dir = Path(level_dir)
    level_dir.mkdir(parents=True, exist_ok=True)
    results = list(results)

    description = LEVEL_DESCRIPTIONS.get(level_number, "ADS Builder v2 circuit")
    if level_title is None:
        level_title = f"ADS Builder v2 Level-{level_number} parity gate"
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    tolerance_dB = _tolerance(results)

    counts = {"pass": 0, "fail": 0, "blocked": 0}
    circuits: list[dict[str, Any]] = []
    for result in results:
        verdict = _classify(result)
        counts[verdict] += 1
        circuits.append(
            {
                "label": str(result.get("label", "")),
                "verdict": verdict,
                "pass": bool(result.get("pass", False)),
                "reason": str(result.get("reason", "")),
                "max_abs_delta_dB": _finite_or_none(result.get("max_abs_delta_dB")),
            }
        )

    status = {
        "level_number": level_number,
        "level_description": description,
        "level_title": level_title,
        "tolerance_dB": _finite_or_none(tolerance_dB),
        "generated": timestamp,
        "n_circuits": len(circuits),
        "n_pass": counts["pass"],
        "n_fail": counts["fail"],
        "n_blocked": counts["blocked"],
        "circuits": circuits,
    }
    status_path = level_dir / "inspection_status.json"
    status_path.write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )

    index_path = level_dir / "inspection_index.md"
    index_path.write_text(
        _render_index(level_number, description, level_title, timestamp,
                      tolerance_dB, results, counts),
        encoding="utf-8",
    )
    return {"index_path": index_path, "status_path": status_path}


def _render_index(
    level_number: int,
    description: str,
    level_title: str,
    timestamp: str,
    tolerance_dB: float,
    results: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
) -> str:
    """Render the ``inspection_index.md`` body."""

    lines: list[str] = []
    lines.append(f"# ADS Builder v2 Level {level_number} -- Inspection Index\n\n")
    lines.append(f"**Level**: {level_number} ({description})\n\n")
    lines.append(f"**Gate**: {level_title}\n\n")
    lines.append(f"**Parity tolerance**: {tolerance_dB:g} dB\n\n")
    lines.append(f"**Generated**: {timestamp}\n\n")
    lines.append(
        f"**Result**: {counts['pass']} pass, {counts['fail']} fail, "
        f"{counts['blocked']} blocked (of {len(list(results))} circuits)\n\n"
    )

    lines.append("## Circuits\n\n")
    lines.append("| Circuit | Parity | max abs delta_S21 (dB) | Topology |\n")
    lines.append("|---------|--------|------------------------|----------|\n")
    for result in results:
        label = str(result.get("label", ""))
        verdict = _classify(result).upper()
        delta = _fmt_delta(result.get("max_abs_delta_dB"))
        topology = _escape_cell(metadata_for(label).topology_description)
        link = f"[`{label}`]({label}/inspection_README.md)"
        lines.append(f"| {link} | {verdict} | {delta} | {topology} |\n")
    lines.append("\n")

    lines.append("## Failure summary\n\n")
    failed = [r for r in results if _classify(r) != "pass"]
    if not failed:
        lines.append("All circuits passed the parity gate.\n\n")
    else:
        for result in failed:
            label = str(result.get("label", ""))
            verdict = _classify(result).upper()
            reason = str(result.get("reason", ""))
            delta = _fmt_delta(result.get("max_abs_delta_dB"))
            lines.append(
                f"- `{label}` -- {verdict} (reason: `{reason}`, "
                f"max abs delta_S21 = {delta} dB)\n"
            )
            note = metadata_for(label).notes
            if note:
                lines.append(f"  - {note}\n")
        lines.append("\n")

    lines.append("## Navigation\n\n")
    lines.append(
        "Each circuit has its own workspace directory next to this file. "
        "Open the circuit's `inspection_README.md` first -- it carries the "
        "topology, the device summary, the parity verdict, and the "
        "ADS-import steps. `comparison_plot.png` (or `comparison_plot.csv` "
        "when matplotlib is unavailable) shows the TVTL-vs-ADS |S21| "
        "overlay.\n\n"
    )
    lines.append(
        "Up: [`../ads_builder_v2_inspection_index.md`]"
        "(../ads_builder_v2_inspection_index.md) -- the cross-level "
        "index covering every ADS Builder v2 level.\n"
    )
    return "".join(lines)


def _classify(result: Mapping[str, Any]) -> str:
    """Bucket a parity result into ``pass`` / ``fail`` / ``blocked``."""

    reason = str(result.get("reason", ""))
    if reason in ("ads_blocked", "tvtl_error"):
        return "blocked"
    return "pass" if bool(result.get("pass", False)) else "fail"


def _tolerance(results: Sequence[Mapping[str, Any]]) -> float:
    """Return the parity tolerance shared by the level's circuits."""

    for result in results:
        try:
            return float(result["tolerance_dB"])
        except (KeyError, TypeError, ValueError):
            continue
    return float("nan")


def _finite_or_none(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` (JSON has no NaN literal)."""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt_delta(value: Any) -> str:
    """Format a max-delta value for a markdown table cell."""

    finite = _finite_or_none(value)
    return "n/a" if finite is None else f"{finite:.3f}"


def _escape_cell(text: str) -> str:
    """Escape ``|`` so a description never breaks a markdown table row."""

    return text.replace("|", r"\|")


