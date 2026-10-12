"""Light, presentational polish for emitted hpeesofsim ``.ckt`` files.

The netlist emitter already groups the deck (substrate, then elements,
then the analysis controller) and uses the meaningful node names from
the device spec. Polish therefore only enriches the **leading comment
block**: it rewrites every leading ``;``-comment line with a header
carrying the circuit label, ladder level, a generation timestamp, and a
one-line level description.

Polish is strictly presentational: it touches only the leading comment
lines. Every non-comment line -- the substrate ``model`` line, the
element lines, the analysis directives -- is preserved byte-for-byte and
in its original order, so the simulated circuit is unchanged.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from .metadata import LEVEL_DESCRIPTIONS


def polish_netlist(netlist_path: str | Path, *, label: str, level: int) -> Path:
    """Rewrite the leading comment block of an emitted ``.ckt`` deck.

    Parameters
    ----------
    netlist_path : str or Path
        Path to a flat hpeesofsim netlist produced by the emitter.
    label : str
        Circuit label, written into the header.
    level : int
        Ladder level number, written into the header.

    Returns
    -------
    Path
        The polished netlist path (same path; rewritten in place).

    Notes
    -----
    Only the contiguous run of leading ``;`` comment lines (and blank
    lines within it) is replaced. The first non-comment, non-blank line
    and everything after it are kept verbatim, so a parse of the
    element/connection graph is identical before and after polish.
    """

    path = Path(netlist_path)
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    # Find the first line that is neither a comment nor blank -- that is
    # where the simulated circuit begins.
    first_body = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(";"):
            first_body = index
            break
    else:
        first_body = len(lines)

    body = "".join(lines[first_body:])
    description = LEVEL_DESCRIPTIONS.get(level, "ADS Builder v2 circuit")
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    header = (
        "; ===================================================================\n"
        "; tvtl_ads_builder_v2 -- flat hpeesofsim netlist (ADS-importable)\n"
        f";   circuit label : {label}\n"
        f";   ladder level  : {level} ({description})\n"
        f";   generated     : {timestamp}\n"
        ";   note          : presentational header only; the simulated\n"
        ";                   circuit below is emitter output, unmodified.\n"
        "; ===================================================================\n"
        "\n"
    )
    path.write_text(header + body, encoding="utf-8")
    return path


def netlist_element_graph(netlist_text: str) -> list[tuple[str, ...]]:
    """Parse a netlist's non-comment lines into a comparable token graph.

    Each element / directive line is split into a tuple of whitespace-
    separated tokens. Comment lines (``;``) and blank lines are dropped.
    Two netlists with the same return value here describe the same
    circuit to the simulator -- this is the equivalence the polish step
    must preserve.

    Parameters
    ----------
    netlist_text : str
        The contents of a ``.ckt`` file.

    Returns
    -------
    list of tuple of str
        One token tuple per non-comment, non-blank line, in order.
    """

    graph: list[tuple[str, ...]] = []
    for line in netlist_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        graph.append(tuple(stripped.split()))
    return graph


