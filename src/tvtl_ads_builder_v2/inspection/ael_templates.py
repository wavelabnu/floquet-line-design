"""AEL deck / launcher templates for the ADS inspection schematic.

The deck builds an ADS schematic *by element placement*, the way ADS
itself does it -- the function vocabulary is taken verbatim from two
macros recorded in ADS 2026 (``recorded.dem`` / ``recorded2.dem``; see
:mod:`.verified_ael_functions`). For each device-spec element the deck
emits ``de_init_item`` / ``de_place_item`` / ``de_free_item``, preceded
by a ``de_rotate_image`` call when the element needs a non-default
orientation (see :data:`.schematic_layout.ROTATION_FOR_ELEMENT`); for
each connection it emits ``de_connect`` / ``de_add_wire`` / ``de_end``.

Element *parameter values* (MLIN ``W``/``L``, substrate, sweep range)
are **not** set by this deck -- the schematic shows the correct
topology with ADS default values. Parameter substitution
(``de_edit_item`` / ``de_set_item_parameters``) is a separate
follow-up. The deck never emits the manual-reposition functions
(``db_select_item`` / ``de_move_and_adjust_edges``) -- auto-layout
places every element at its final coordinate.

AEL lexical rules respected: ``//`` and ``/* */`` comments (a leading
``;`` is the statement terminator, not a comment); double-quoted
strings; plain ASCII, no BOM.
"""

from __future__ import annotations

from .element_library_mapping import ael_path_for_element
from .schematic_layout import SchematicPlan

# Timestamp markers; idempotency tests strip the line containing them.
AEL_TIMESTAMP_MARKER = "// generated     :"
README_TIMESTAMP_MARKER = "<!-- generated:"


OPENER_BAT = """\
@echo off
REM ====================================================================
REM open_workspace.bat
REM tvtl_ads_builder_v2 inspection schematic launcher (generated file).
REM
REM Launches ADS and runs build_workspace.ael with the documented -m
REM startup-AEL flag. The deck builds an ADS schematic cellview for
REM this circuit by placing and wiring its elements.
REM ====================================================================
setlocal
cd /d "%~dp0"

if "%HPEESOF_DIR%"=="" (
    echo ERROR: HPEESOF_DIR is not set.
    echo Run scripts\\setup_ads_environment.ps1 to configure the ADS
    echo environment, then double-click this file again.
    exit /b 1
)
if not exist "%HPEESOF_DIR%\\bin\\ads.exe" (
    echo ERROR: ads.exe not found under "%HPEESOF_DIR%\\bin".
    echo Check that HPEESOF_DIR points at a valid ADS installation.
    exit /b 1
)

echo Launching ADS to build the inspection schematic...
"%HPEESOF_DIR%\\bin\\ads.exe" -m build_workspace.ael
endlocal
"""


def _num(value: float) -> str:
    """Format a coordinate for AEL (compact, no trailing zeros)."""

    return f"{float(value):g}"


def render_ael_script(
    library: str,
    cell: str,
    plan: SchematicPlan,
    timestamp: str,
    workspace_path: str,
) -> str:
    """Render the ``build_workspace.ael`` element-placement deck.

    Parameters
    ----------
    library : str
        ADS library the cellview is created in (the reference
        workspace's design library).
    cell : str
        Cell name -- the canonical circuit's label.
    plan : SchematicPlan
        Placement + wiring plan from
        :func:`.schematic_layout.build_schematic_plan`.
    timestamp : str
        Generation timestamp; emitted on one marked AEL comment line.
    workspace_path : str
        Absolute path of the reference ADS workspace directory. The
        deck opens it first (``de_open_workspace``) so the deck
        self-bootstraps when ADS is launched with no workspace open;
        backslashes are converted to forward slashes for AEL.

    Returns
    -------
    str
        The complete AEL deck -- plain ASCII, ``//`` comments, and only
        evidence-pinned function calls.
    """

    view = f"{library}:{cell}:schematic"
    workspace_ael_path = workspace_path.replace("\\", "/")
    lines: list[str] = []
    lines.append("// ===================================================================")
    lines.append("// build_workspace.ael")
    lines.append("// ===================================================================")
    lines.append("// tvtl_ads_builder_v2 inspection schematic deck (generated file).")
    lines.append(f"{AEL_TIMESTAMP_MARKER} {timestamp}")
    lines.append(f"//   circuit (cell) : {cell}")
    lines.append("//")
    lines.append("// Builds an ADS schematic by placing every device-spec element and")
    lines.append("// wiring its connections. Element parameter values are NOT set --")
    lines.append("// the schematic shows the correct topology with ADS defaults;")
    lines.append("// parameter substitution is a separate follow-up. Every function")
    lines.append("// call is evidence-pinned to a recorded ADS macro -- see")
    lines.append("// src/tvtl_ads_builder_v2/inspection/verified_ael_functions.py.")
    lines.append("// ===================================================================")
    lines.append("")
    lines.append("/* Begin */")
    lines.append("// Bootstrap: open the reference workspace so the cellview's")
    lines.append("// library exists when ADS is launched with no workspace open.")
    lines.append(f'de_open_workspace("{workspace_ael_path}");')
    lines.append("")
    lines.append(
        f'decl macroContext = de_create_new_schematic_view("{library}", '
        f'"{cell}", "schematic");'
    )
    lines.append("de_show_context_in_new_window(macroContext);")
    lines.append(
        f'macroContext = de_get_design_context_from_name("{view}");'
    )
    lines.append("de_bring_context_to_top_or_open_new_window(macroContext);")
    lines.append("")

    lines.append("// --- place elements --------------------------------------------------")
    for index, item in enumerate(plan.items):
        path = ael_path_for_element(item.element_type)
        handle = f"itemInfo{index}"
        lines.append(f"//   {item.item_id} ({item.element_type})")
        lines.append(f'decl {handle} = de_init_item("{path}");')
        # Rotation is per-block: a fresh de_init_item starts in the
        # default orientation, so no reset is emitted after placement.
        if item.rotation is not None:
            lines.append(f'de_rotate_image("{item.rotation}");')
        lines.append(
            f"de_place_item({handle}, {_num(item.x)}, {_num(item.y)});"
        )
        lines.append(f"{handle} = de_free_item({handle});")
    lines.append("")

    lines.append("// --- wire connections ------------------------------------------------")
    for wire in plan.wires:
        sx, sy = wire.start
        ex, ey = wire.end
        lines.append(f"//   node {wire.node}")
        lines.append("de_connect();")
        lines.append(f"de_add_wire({_num(sx)}, {_num(sy)});")
        lines.append(f"de_add_wire({_num(ex)}, {_num(ey)});")
        lines.append("de_end();")
    lines.append("")

    lines.append("// --- simulate (Path 3) -----------------------------------------------")
    lines.append("// Run the schematic's S-parameter analysis. ADS writes the dataset")
    lines.append("// to <workspace>/data/<cell>.ds; the parity CLI's --with-schematic-sim")
    lines.append("// flag reads it for the three-way comparison against TVTL + Path 1.")
    lines.append("de_analyze();")
    lines.append("")
    lines.append(f'de_save_oa_design("{view}");')
    lines.append("de_close_window();")
    lines.append("/* End */")
    return "\n".join(lines) + "\n"


