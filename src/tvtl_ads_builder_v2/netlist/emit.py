"""Compose a flat hpeesofsim ``.ckt`` deck from a device spec.

The emitter is element-type-dispatching: it walks ``spec.elements`` in
order and calls the appropriate per-type emitter from the
:mod:`.elements` package. The S-parameter analysis directive is
synthesised from ``spec.analysis``; the substrate is emitted once near
the top of the deck.

The recognised element types are ``MSUB``, ``MLIN``, ``TermG``,
``MTEE``, the lumped ``R`` / ``L`` / ``C`` / ``RLC_SERIES``, the
``SDD2P`` symbolically-defined device, and the ``I_Probe`` current
probe. Anything outside that vocabulary raises ``ValueError`` --
unsupported element types are caught loudly rather than silently
degenerating.

``I_Probe`` is a transparent two-terminal current probe; ADS netlists
it as a ``Short`` element (``Short:<id> <n1> <n2> Mode=0
SaveCurrent=yes``), so its emitter writes that token on both backends.

``RLC_SERIES`` has no single hpeesofsim primitive: its emitter expands
it inline to three lines (an ``R``, an ``L`` and a ``C``) joined by two
auto-generated internal nodes.

Any node listed in ``spec.ground_nodes`` is translated to the literal
``0`` (hpeesofsim's ground reference) wherever it appears as an
element pin.
"""

from __future__ import annotations

import math
from typing import Callable

from ..device_spec import (
    DeviceSpec,
    Element,
    Substrate,
    resolve_spec,
    validate_device_spec,
)
from .elements import (
    emit_c,
    emit_hb_single_tone,
    emit_hb_two_tone,
    emit_i_probe,
    emit_l,
    emit_mlef,
    emit_mlin,
    emit_mloc,
    emit_msub,
    emit_mtee,
    emit_name,
    emit_p1tone,
    emit_r,
    emit_rlc_series,
    emit_sdd2p,
    emit_termg,
    emit_viagnd,
)


# Each emitter is called as ``emitter(element, substrate, resolve_node)``.
# A microstrip-family emitter consumes ``substrate``; the lumped R/L/C
# emitters ignore it. ``RLC_SERIES`` returns three newline-joined lines.
ELEMENT_EMITTERS: dict[str, Callable[..., str]] = {
    "TermG": lambda element, substrate, resolve: emit_termg(element, resolve),
    "MLIN": lambda element, substrate, resolve: emit_mlin(
        element, substrate.name, resolve
    ),
    "MLEF": lambda element, substrate, resolve: emit_mlef(
        element, substrate.name, resolve
    ),
    "MLOC": lambda element, substrate, resolve: emit_mloc(
        element, substrate.name, resolve
    ),
    "VIAGND": lambda element, substrate, resolve: emit_viagnd(
        element, substrate.name, resolve
    ),
    "MTEE": lambda element, substrate, resolve: emit_mtee(
        element, substrate.name, resolve
    ),
    "R": lambda element, substrate, resolve: emit_r(element, resolve),
    "L": lambda element, substrate, resolve: emit_l(element, resolve),
    "C": lambda element, substrate, resolve: emit_c(element, resolve),
    "RLC_SERIES": lambda element, substrate, resolve: emit_rlc_series(
        element, resolve
    ),
    "SDD2P": lambda element, substrate, resolve: emit_sdd2p(element, resolve),
    "I_Probe": lambda element, substrate, resolve: emit_i_probe(
        element, resolve
    ),
    "NAME": lambda element, substrate, resolve: emit_name(element, resolve),
}


def _make_node_resolver(spec: DeviceSpec) -> Callable[[str], str]:
    """Return a function mapping a node id to its netlist token.

    Nodes listed in ``spec.ground_nodes`` resolve to the literal ``0``;
    every other node keeps its declared name.
    """

    ground = set(spec.ground_nodes)

    def resolve(node_id: str) -> str:
        return "0" if node_id in ground else node_id

    return resolve


def emit_hpeesofsim_netlist(spec: DeviceSpec) -> str:
    """Emit the full ``.ckt`` text for ``spec``.

    Parameters
    ----------
    spec : DeviceSpec
        Device spec describing a Level-1/2/3 circuit. Re-validated here.

    Returns
    -------
    str
        The complete netlist contents: a leading provenance comment
        block, the substrate model, every element in order (with
        ground nodes translated to ``0``), the S-parameter analysis
        directive, and a trailing newline.

    Raises
    ------
    ValueError
        If any element in ``spec`` has a ``type`` not registered in
        :data:`ELEMENT_EMITTERS`.
    """

    spec = resolve_spec(spec)
    validate_device_spec(spec)
    resolve = _make_node_resolver(spec)

    lines: list[str] = []
    lines.append(
        "; tvtl_ads_builder_v2 hpeesofsim deck (S-parameter analysis)\n"
    )
    if spec.label:
        lines.append(f"; label = {spec.label}\n")
    lines.append(f"; device_spec_version = {int(spec.device_spec_version)}\n")
    lines.append(
        f"; port_impedance_ohm = {float(spec.analysis.port_impedance_ohm):.6g}\n"
    )
    if spec.ground_nodes:
        lines.append(
            f"; ground_nodes = {', '.join(spec.ground_nodes)} (emitted as 0)\n"
        )
    lines.append("\n")

    lines.append("; --- substrate ---\n")
    lines.append(emit_msub(spec.substrate) + "\n")
    lines.append("\n")

    lines.append("; --- elements ---\n")
    for element in spec.elements:
        emitter = ELEMENT_EMITTERS.get(element.type)
        if emitter is None:
            raise ValueError(
                f"Element '{element.id}' has unsupported type '{element.type}'. "
                f"The emitter handles only {sorted(ELEMENT_EMITTERS)}."
            )
        lines.append(emitter(element, spec.substrate, resolve) + "\n")
    lines.append("\n")

    if spec.sources:
        lines.append("; --- sources ---\n")
        for source in spec.sources:
            lines.append(emit_p1tone(source) + "\n")
        lines.append("\n")

    lines.append("; --- analysis ---\n")
    lines.append(_emit_sp_directive(spec) + "\n")
    if spec.hb_single_tone is not None:
        lines.append(emit_hb_single_tone(spec.hb_single_tone) + "\n")
    if spec.hb_two_tone is not None:
        lines.append(emit_hb_two_tone(spec.hb_two_tone) + "\n")
    return "".join(lines)


def emit_flat_hb_netlist(spec: DeviceSpec) -> str:
    """Emit a flat hpeesofsim **HB** deck with built-in ``Port`` sources.

    For HB designs whose sources drive *distinct* nets -- e.g. an
    externally-pumped ideal-C device with the pump on a control net rather than
    the signal input -- the OA layout engine's single-source-net assumption
    trips a (broken-on-this-install) grid fallback. This flat emitter sidesteps
    OA layout entirely.

    Differences from :func:`emit_hpeesofsim_netlist`:

    * sources are emitted as the built-in ``Port`` HB source
      (``P[1]=polar(dbmtow(P),0) Freq[1]=F GHz``) -- the form the OA path itself
      uses; bare ``P_1Tone`` is an *undefined model* in a flat deck;
    * no ``S_Param`` controller (it errors on a source-driven deck);
    * the two-tone HB controller is the only analysis directive.

    hpeesofsim saves node voltages and I_Probe currents by default, so the
    resulting ``.ds`` is readable with ``read_hb_dataset``. Requires
    ``spec.hb_two_tone`` and at least one source.
    """

    spec = resolve_spec(spec)
    validate_device_spec(spec)
    if spec.hb_two_tone is None:
        raise ValueError(
            "emit_flat_hb_netlist requires a two-tone HB controller "
            "(spec.hb_two_tone is None)."
        )
    if not spec.sources:
        raise ValueError(
            "emit_flat_hb_netlist requires at least one source (spec.sources "
            "is empty)."
        )
    # Every varactor terminal needs a DC path to ground (reason no_dc_return_for_varactor).
    from tvtl_ads_builder_v2.validation.dc_return import check_dc_return
    check_dc_return(spec)
    resolve = _make_node_resolver(spec)

    lines: list[str] = ["; tvtl_ads_builder_v2 flat hpeesofsim HB deck\n"]
    if spec.label:
        lines.append(f"; label = {spec.label}\n")
    if spec.ground_nodes:
        lines.append(
            f"; ground_nodes = {', '.join(spec.ground_nodes)} (emitted as 0)\n"
        )
    lines.append("\n; --- substrate ---\n")
    lines.append(emit_msub(spec.substrate) + "\n")

    lines.append("\n; --- elements ---\n")
    for element in spec.elements:
        emitter = ELEMENT_EMITTERS.get(element.type)
        if emitter is None:
            raise ValueError(
                f"Element '{element.id}' has unsupported type '{element.type}'."
            )
        lines.append(emitter(element, spec.substrate, resolve) + "\n")

    lines.append("\n; --- HB sources (built-in Port) ---\n")
    for source in spec.sources:
        freq = float(source.freq)
        power = float(source.power_dBm)
        # ADS polar(mag, angle) takes the angle in DEGREES; the spec carries
        # the phase in radians. A 0.0 phase emits "...,0", byte-identical to
        # the historical fixed-zero decks.
        phase_deg = math.degrees(float(source.phase_rad))
        lines.append(
            f"Port:{source.name} {resolve(source.port_node)} 0 "
            f"Num={int(source.num)} Z={float(source.z_ohm):.10g} Ohm "
            f"P[1]=polar(dbmtow({power:.10g}),{phase_deg:.10g}) "
            f"Freq[1]={freq / 1.0e9:.10g} GHz Noise=no\n"
        )

    lines.append("\n; --- analysis ---\n")
    lines.append(emit_hb_two_tone(spec.hb_two_tone) + "\n")
    return "".join(lines)


def _emit_sp_directive(spec: DeviceSpec) -> str:
    """Emit the S-parameter controller plus its companion SweepPlan.

    hpeesofsim's S-parameter analysis is a two-directive arrangement:
    the ``S_Param`` controller references a named ``SweepPlan`` for
    the frequency grid (``Start``/``Stop``/``Lin`` belong on the
    SweepPlan, not on S_Param itself). Both lines are returned as a
    single string, separated by a newline.
    """

    grid = spec.analysis.frequency_grid_Hz
    start_GHz = float(grid.start) * 1.0e-9
    stop_GHz = float(grid.stop) * 1.0e-9
    sweep_plan_name = "SP1_stim"
    controller = (
        f"S_Param:SP1 CalcS=yes CalcY=no CalcZ=no "
        f'SweepVar="freq" SweepPlan="{sweep_plan_name}"'
    )
    plan = (
        f"SweepPlan:{sweep_plan_name} "
        f"Start={start_GHz:.12g} GHz "
        f"Stop={stop_GHz:.12g} GHz "
        f"Lin={int(grid.points)}"
    )
    return controller + "\n" + plan


def emit_element(
    element: Element,
    substrate: Substrate,
    resolve_node: Callable[[str], str] = lambda node: node,
) -> str:
    """Dispatch helper used by tests and external callers.

    Parameters
    ----------
    element : Element
        The element to emit.
    substrate : Substrate
        Substrate the element's MSUB reference must match (only
        relevant for microstrip-family elements).
    resolve_node : callable
        Maps a device-spec node id to the netlist token. Defaults to
        the identity.

    Returns
    -------
    str
        The single netlist line for ``element``.

    Raises
    ------
    ValueError
        On an unsupported element type.
    """

    emitter = ELEMENT_EMITTERS.get(element.type)
    if emitter is None:
        raise ValueError(
            f"emit_element: unsupported element type '{element.type}'."
        )
    return emitter(element, substrate, resolve_node)


