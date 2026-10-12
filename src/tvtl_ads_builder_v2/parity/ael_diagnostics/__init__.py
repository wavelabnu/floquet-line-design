"""AEL diagnostic ladder D0-D4.

A small ladder of minimal circuits, each isolating exactly one wiring
pattern the AEL schematic builder must get right (port-to-element,
element-to-element, MTEE pins, stub-to-ground). Each circuit is
packaged into its own workspace directory with the flat hpeesofsim
deck, the AEL deck, the TVTL expected response, the hpeesofsim result
and a Path-1 (TVTL vs hpeesofsim) parity report.

The diagnostic generator does the headless work (generate, validate Path 1,
package); the user opens each ``open_workspace.bat`` in ADS and reports whether
the AEL-built schematic is wired correctly. The lowest-numbered
diagnostic that fails localises the broken wiring pattern.

The diagnostics are deliberately separate from the main parity ladder
(``parity/canonical_circuits.py``) -- they reuse its spec constructors
but never modify it.
"""

from .specs import DIAGNOSTIC_SPECS, DIAGNOSTIC_ORDER

__all__ = ["DIAGNOSTIC_SPECS", "DIAGNOSTIC_ORDER"]


