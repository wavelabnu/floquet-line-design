"""Evidence registry for AEL functions used by the inspection deck.

Every AEL function the generated ``build_workspace.ael`` deck calls
must appear in :data:`VERIFIED_AEL_FUNCTIONS` with an ``evidence_path``
and ``evidence_line`` -- the *recorded macro* in which ADS itself
emitted that call. A function with no such evidence does not go in
the deck.

Evidence source: two macros recorded in ADS 2026 Update 1.2 by
performing the actions in the GUI and saving the generated AEL.

* ``recorded.dem``  -- a minimal schematic: create a cellview, place an
  MLIN and a TermG, wire them, save.
* ``recorded2.dem`` -- a comprehensive schematic exercising MLIN,
  MTEE_ADS, MLOC, VIAGND, MSUB, TermG, S_Param, SweepPlan, GROUND, R,
  L, SDD2P and VAR placement, wiring and parameter editing.

These macros correct an earlier wrong conclusion: the
netlist-import functions searched for earlier (``de_create_cellview`` /
``de_import_netlist``) genuinely do not exist, but ADS *does* build
schematics by *element placement* -- ``de_create_new_schematic_view`` +
``de_init_item`` / ``de_place_item`` / ``de_connect`` / ``de_add_wire``.
The macros are the authoritative evidence for those names.
"""

from __future__ import annotations

import os
from pathlib import Path

_REFERENCE_WORKSPACE_ENV = "TVTL_AEL_REFERENCE_WORKSPACE"
_ADS_ABL_MENU_ENV = "TVTL_ADS_ABL_MENU_AEL"

_REF = os.environ.get(_REFERENCE_WORKSPACE_ENV)
_REC1 = str(Path(_REF) / "recorded.dem") if _REF else "<TVTL_AEL_REFERENCE_WORKSPACE>/recorded.dem"
_REC2 = str(Path(_REF) / "recorded2.dem") if _REF else "<TVTL_AEL_REFERENCE_WORKSPACE>/recorded2.dem"
# de_open_workspace is not in the recorded macros; it is evidenced in a shipped
# ADS script when the optional ADS install path is available.
_SHIPPED_ABL_MENU = os.environ.get(
    _ADS_ABL_MENU_ENV,
    "<ADS_INSTALL_DIR>/pcb_connect/eedfi/ael/abl_menu.ael",
)

def _ev(path: str, line: int, purpose: str) -> dict[str, object]:
    return {"evidence_path": path, "evidence_line": line, "purpose": purpose}


# AEL functions confirmed by a recorded ADS 2026 macro. The inspection
# deck calls a subset of these (cellview + placement + wiring + save);
# the parameter-editing entries are recorded for later use.
VERIFIED_AEL_FUNCTIONS: dict[str, dict[str, object]] = {
    # --- cellview / window management ---
    "de_open_workspace": _ev(
        _SHIPPED_ABL_MENU, 130,
        "Open an ADS workspace by directory path (deck bootstrap).",
    ),
    "de_create_new_schematic_view": _ev(
        _REC1, 2, "Create a new schematic cellview; returns a context."
    ),
    "de_show_context_in_new_window": _ev(
        _REC1, 3, "Show a design context in a new window."
    ),
    "de_get_design_context_from_name": _ev(
        _REC1, 4, "Look up a design context by 'lib:cell:view' name."
    ),
    "de_bring_context_to_top_or_open_new_window": _ev(
        _REC1, 6, "Bring a design context's window to the front."
    ),
    "de_save_oa_design": _ev(
        _REC1, 18, "Save the OpenAccess design 'lib:cell:view'."
    ),
    "de_close_window": _ev(_REC1, 19, "Close the active design window."),
    # --- element placement ---
    "de_init_item": _ev(
        _REC1, 7, "Initialise a placement handle for 'lib:Model:symbol'."
    ),
    "de_rotate_image": _ev(
        _REC1, 11, 'Set placement rotation ("UP"/"DOWN"/"LEFT"/"RIGHT").'
    ),
    "de_rotate_inc": _ev(_REC2, 14, "Increment placement rotation by 90 deg."),
    "de_place_item": _ev(
        _REC1, 8, "Place the initialised item at (x, y)."
    ),
    "de_free_item": _ev(_REC1, 9, "Release a placement / edit handle."),
    # --- wiring ---
    "de_connect": _ev(_REC1, 14, "Begin a wire."),
    "de_add_wire": _ev(_REC1, 15, "Add a wire vertex at (x, y)."),
    "de_end": _ev(_REC1, 17, "Finish the wire begun by de_connect()."),
    # --- simulation ---
    "de_analyze": _ev(
        _REC2, 191, "Run the analysis (simulation) on the active design."
    ),
    # --- parameter editing (recorded for later use) ---
    "de_edit_item": _ev(_REC2, 112, "Open an item by name for parameter edit."),
    "de_set_item_parameters": _ev(
        _REC2, 113, "Set an item's parameters (with prm_ex values)."
    ),
    "de_end_edit_item": _ev(_REC2, 114, "End a parameter-edit session."),
    "prm_ex": _ev(_REC2, 113, "Construct a parameter value for de_set_item_parameters."),
}

# Recorded in the macros but deliberately NOT emitted by our deck:
# db_select_item / de_move_and_adjust_edges / db_deselect_all /
# de_rotate_center appear in recorded2.dem only because the user moved
# elements with the mouse after placing them. Our auto-layout places
# every element at its final coordinate, so no repositioning is needed.
MANUAL_REPOSITION_FUNCTIONS: tuple[str, ...] = (
    "db_select_item",
    "de_move_and_adjust_edges",
    "db_deselect_all",
    "de_rotate_center",
)


def is_verified(function_name: str) -> bool:
    """Return whether ``function_name`` is an evidence-pinned AEL function."""

    return function_name in VERIFIED_AEL_FUNCTIONS



