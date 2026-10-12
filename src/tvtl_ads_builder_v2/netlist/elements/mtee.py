"""Emit an ``MTEE_ADS`` microstrip T-junction element line for hpeesofsim.

The framework's device-spec element type remains ``MTEE`` (the abstract
junction in the device-spec layer); this emitter writes the simulator
directive ``MTEE_ADS``, the wider-validity ADS native model. The basic
``MTEE`` model is out of validity at f_pump = 6.2 GHz on the 1.54 mm
Rogers 4003C substrate used in this project (hpeesofsim emits
``freq*H > 0.4*Z_LOWEST`` warnings on every TEE element). Public
validation uses ``MTEE_ADS`` for the wider-validity native model, and
the OA workspace emitter already maps ``MTEE`` device-spec
elements to the ``MTEE_ADS`` library cell. The flat-deck emitter is
hard-switched to the same model so the two ADS-output paths (flat
hpeesofsim deck and OA workspace) agree.

Two pins are the through line (main, widths ``W1``/``W2``); the third
is the perpendicular branch (width ``W3``). The element models the
junction parasitics only -- the connecting line lengths are separate
``MLIN`` elements. The parameter list (``W1``, ``W2``, ``W3``,
``Subst``) is identical between the two models, so the emission
template only changes the directive token.

Emitted form::

    MTEE_ADS:TEE1 n_main_left n_main_right n_branch Subst="msub1" W1=4.27 mm W2=4.27 mm W3=1.93 mm
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import Element


def emit_mtee(
    element: Element,
    substrate_name: str,
    resolve_node: Callable[[str], str],
) -> str:
    """Return a single-line MTEE directive for ``element``.

    Parameters
    ----------
    element : Element
        Must have ``type == "MTEE"`` and three pins (1, 2, 3).
    substrate_name : str
        Name of the substrate declared by the netlist's MSUB line.
    resolve_node : callable
        Maps a device-spec node id to the netlist token (translates
        ground nodes to the literal ``0``).
    """

    if element.type != "MTEE":
        raise ValueError(f"emit_mtee called on element of type '{element.type}'.")
    if len(element.pins) != 3:
        raise ValueError(
            f"MTEE '{element.id}' must declare exactly three pins "
            f"(got {len(element.pins)})."
        )
    substrate_ref = str(element.parameters.get("substrate_ref", substrate_name))
    if substrate_ref != substrate_name:
        raise ValueError(
            f"MTEE '{element.id}' references substrate '{substrate_ref}' but the "
            f"netlist declares only '{substrate_name}'."
        )
    pins = {pin.pin: resolve_node(pin.node) for pin in element.pins}
    w1_mm = float(element.parameters["w1_m"]) * 1.0e3
    w2_mm = float(element.parameters["w2_m"]) * 1.0e3
    w3_mm = float(element.parameters["w3_m"]) * 1.0e3
    return (
        f"MTEE_ADS:{element.id} {pins[1]} {pins[2]} {pins[3]} "
        f'Subst="{substrate_name}" '
        f"W1={w1_mm:.12g} mm "
        f"W2={w2_mm:.12g} mm "
        f"W3={w3_mm:.12g} mm"
    )


