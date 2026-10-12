"""Emit a ``NAME`` node-annotation element -- hpeesofsim no-op.

``NAME`` is a 1-pin spec-level
annotation declaring "this net should appear as a voltage column in
the HB ``.ds`` dataset." A live probe confirmed that ADS HB only
writes voltage columns for explicitly-named nets; the OA emitter
implements that by walking NAME elements after placement and renaming
the matching OA ``Net`` via ``net.name = <spec_net_name>``. See
:mod:`tvtl_ads_builder_v2.oa.oa_emitter` (the ``_rename_named_nets``
pass) for the OA side.

The hpeesofsim flat deck already uses the spec's net names directly
(the spec is the source of truth on the flat-deck path), so NAME has
nothing substantive to emit there. A single comment line records the
annotation in the inspected deck for human review -- the line carries
no simulator semantics and the netlist comparator's tokenizer treats
``;``-prefixed lines as comments and ignores them, so the comment
adds no false positive to the OA-vs-hpeesofsim structural cross-check.
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import Element


def emit_name(
    element: Element, resolve_node: Callable[[str], str] = lambda node: node
) -> str:
    """Return a single comment line documenting a ``NAME`` annotation.

    The line is a hpeesofsim-side audit trail: the net the spec marks
    as observable is recorded in the deck so a reader can match the
    OA-side rename without cross-referencing the spec. The comment
    carries no simulator semantics; hpeesofsim ignores every ``;``-
    prefixed line.

    Parameters
    ----------
    element : Element
        Must have ``type == "NAME"`` and exactly one pin.
    resolve_node : callable
        Maps a spec node id to its netlist token. For NAME the
        annotation is "this is a node *name*", so a ground-node
        resolution to ``0`` is presented verbatim -- naming the ground
        net is harmless but informational.

    Raises
    ------
    ValueError
        If the element is not a NAME or does not have exactly one pin.
    """

    if element.type != "NAME":
        raise ValueError(
            f"emit_name called on element of type '{element.type}'."
        )
    if len(element.pins) != 1:
        raise ValueError(
            f"NAME '{element.id}' must declare exactly one pin "
            f"(got {len(element.pins)})."
        )
    node = element.pins[0].node
    return f"; NAME annotation: {element.id} -> {resolve_node(node)}"


