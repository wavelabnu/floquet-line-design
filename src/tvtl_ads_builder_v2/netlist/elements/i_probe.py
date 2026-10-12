"""Emit an ``I_Probe`` current-probe line for hpeesofsim.

``I_Probe`` is the native ADS current probe -- an electrically
transparent (0 ohm) two-terminal element that reports the current
through itself.

ADS conventions used by the public emitter:

* the OA cell is ``ads_rflib:I_Probe:symbol`` (NOT ``ads_simulation``);
* ADS netlists the cell as a ``Short`` element -- the netlist token is
  ``Short``, NOT ``I_Probe`` -- carrying ``Mode=0`` and
  ``SaveCurrent=yes``. ``Mode=0`` selects the ideal-short / current-
  probe behaviour (``Mode=1`` with a ``C`` is the DC-block behaviour of
  the same multi-mode ``Short`` element).
  ``SaveCurrent=yes`` is what records the probe current into the
  dataset.

Both backends therefore emit the same ``Short`` token so the netlist
cross-check pairs them directly; ``Mode=0`` is the key the cross-check
keys on to recognise a ``Short`` line as an ``I_Probe`` rather than a
DC block (see :mod:`tvtl_ads_builder_v2.oa.netlist_compare`).

The emitted line is::

    Short:<id> <n1> <n2> Mode=0 SaveCurrent=yes
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import Element

# The netlist element token ADS normalises an I_Probe to, and the two
# fixed parameters that select the current-probe behaviour. These are
# emitter constants, not device-spec fields: an I_Probe carries no
# user-set parameter (it is a transparent probe), so the spec-level
# Element is parameterless and both emitters supply these verbatim.
I_PROBE_NETLIST_TOKEN = "Short"
I_PROBE_MODE = "0"
I_PROBE_SAVE_CURRENT = "yes"


def emit_i_probe(
    element: Element, resolve_node: Callable[[str], str] = lambda node: node
) -> str:
    """Return a single-line ``Short`` directive for an ``I_Probe`` element.

    Parameters
    ----------
    element : Element
        Must have ``type == "I_Probe"`` and exactly two pins.
    resolve_node : callable
        Maps a device-spec node id to the netlist token (translates a
        ground node to the literal ``0``). Defaults to the identity.
    """

    if element.type != "I_Probe":
        raise ValueError(
            f"emit_i_probe called on element of type '{element.type}'."
        )
    if len(element.pins) != 2:
        raise ValueError(
            f"I_Probe '{element.id}' must declare exactly two pins "
            f"(got {len(element.pins)})."
        )
    by_pin = {pin.pin: pin.node for pin in element.pins}
    if set(by_pin) != {1, 2}:
        raise ValueError(
            f"I_Probe '{element.id}' must declare pins 1 and 2 "
            f"(got {sorted(by_pin)})."
        )
    node1 = resolve_node(by_pin[1])
    node2 = resolve_node(by_pin[2])
    return (
        f"{I_PROBE_NETLIST_TOKEN}:{element.id} {node1} {node2} "
        f"Mode={I_PROBE_MODE} SaveCurrent={I_PROBE_SAVE_CURRENT}"
    )


