"""Expand a :class:`TVTLDesign` into a fully-explicit ``DeviceSpec``.

:func:`to_device_spec` is the design API's one transform. It replicates
the design's single unit cell ``n_cells`` times, generating per-cell
element instances and per-cell net names by a deterministic rule, wires
each cell's output net to the next cell's input net, and attaches the
spec-level singletons (the ``P_1Tone`` sources, the VAR block, the HB
controller, the substrate, the S-parameter analysis).

The expansion is mechanical and pure: the same design always yields the
same device spec -- element ids and net names are derived from the cell
index, the branch index and the element position, never from a counter
with hidden state.

Naming convention (see ``V1_MULTI_CELL_PATTERN.txt``)
-----------------------------------------------------
Cell index ``k`` runs ``0 .. n_cells-1``; branch index ``b`` and branch
element index ``i`` are likewise 0-based.

* through-path nets: ``n_input`` (design input), ``n_output`` (design
  output), ``n_cell{k}_out`` (cell k output = cell k+1 input);
* per-cell nets: ``n_cell{k}_junc`` (after the through-line MLIN),
  ``n_cell{k}_seg{b}`` (between chained MTEEs), ``n_cell{k}_br{b}_top``
  (branch tap), ``n_cell{k}_br{b}_n{i}`` (branch chain node),
  ``n_cell{k}_br{b}_ret`` (SDD2P port-2 self-tie), ``n_cell{k}_br{b}_gnd``
  (a grounded branch terminus), ``n_cell{k}_br{b}_open`` (a floating
  branch terminus).

Each grounded branch gets its OWN ground net rather than one shared
``n_gnd``: every ground net still emits as the literal ``0`` (the netlist
emitter translates every ``ground_nodes`` entry to ``0``), so the cells
are electrically identical, but a per-branch ground net keeps each
shunt branch a distinct chain the layout engine can trace -- a single
shared ground net would make every branch fork at it. This is v1's
pattern too: each v1 unit cell's via grounds to its own per-cell net.
* element ids: ``MLIN_c{k}``, ``MTEE_c{k}_{b}``, and per branch element
  ``{R,L,C,MLIN,SDD}_c{k}_b{b}_{i}``; the lone output port is ``P_OUT``.
"""

from __future__ import annotations

from typing import Callable

from tvtl.design.nonlinear_capacitor import smv1245_default_polynomial

from ..device_spec import (
    Analysis,
    DeviceSpec,
    Element,
    ElementPin,
    FrequencyGrid,
    Node,
    P1ToneSource,
    Port,
    Substrate,
    VarBlockSpec,
    VarEntry,
)
from ..device_spec.schema import DEVICE_SPEC_VERSION
from .elements import BranchSpec, CellSpec, TVTLDesign
from .validator import DesignError, validate_design

# The design input / output nets.
_NET_INPUT = "n_input"
_NET_OUTPUT = "n_output"

# The lone output-port element id and ADS port number. Like the Level-7
# integrated cell, a source-driven design has no input port: the
# P_1Tone sources are the input, and the only TermG is the output. The
# output TermG sits at ADS Num=3 so a two-tone HB design (signal Num=1,
# pump Num=2, output Num=3) does not collide on the same port number;
# single-tone designs reuse the same convention for consistency.
_OUTPUT_PORT_ID = "P_OUT"
_OUTPUT_PORT_NUM = 3

# Design varactor type and its polynomial-reference parameter key.
_VARACTOR_TYPE = "SDD2P_varactor"
_Q_POLY_REFERENCE_KEY = "q_poly_reference"

# Design branch element type -> (device-spec element type, id token,
# needs a substrate_ref). The varactor is handled out of band.
_BRANCH_TYPE_MAP: dict[str, tuple[str, str, bool]] = {
    "R": ("R", "R", False),
    "L": ("L", "L", False),
    "C": ("C", "C", False),
    "MLIN": ("MLIN", "MLIN", True),
}

# Charge-polynomial registry: a q_poly_reference string -> a callable
# returning the coefficient tuple (q0, q1, q2, ...). The single entry is
# the SMV1245 model from the analytical stack -- the same source of
# truth the Level-5/7 canonicals use.
_Q_POLY_REGISTRY: dict[str, Callable[[], tuple[float, ...]]] = {
    "smv1245_default_polynomial": lambda: tuple(
        smv1245_default_polynomial().charge_polynomial_coefficients
    ),
}


def to_device_spec(design: TVTLDesign) -> DeviceSpec:
    """Expand ``design`` into a fully-explicit :class:`DeviceSpec`.

    The cell is replicated ``design.n_cells`` times; cell ``k``'s output
    net is cell ``k+1``'s input net. The returned spec carries the VAR
    block unresolved -- exactly like the Level-7 integrated cell -- so
    the existing emitters resolve it; it is re-validated here (in its
    resolved form) so a malformed design fails fast.

    Raises
    ------
    DesignError
        If the design itself is malformed, or a branch carries an
        element type the linear-chain expander does not support, or an
        ``SDD2P_varactor`` names an unknown charge-polynomial reference.
    """

    validate_design(design)

    elements: list[Element] = []
    ground_nodes: set[str] = set()
    open_terminations: list[str] = []

    for k in range(design.n_cells):
        _expand_cell(
            design,
            k,
            elements,
            ground_nodes,
            open_terminations,
        )

    # The lone output port TermG terminates the cascade.
    port_impedance = float(design.analysis.s_param.port_impedance_ohm)
    elements.append(
        Element(
            id=_OUTPUT_PORT_ID,
            type="TermG",
            pins=(ElementPin(pin=1, node=_NET_OUTPUT),),
            parameters={"z_ohm": port_impedance, "num": _OUTPUT_PORT_NUM},
        )
    )

    sources = tuple(
        P1ToneSource(
            name=source.name,
            port_node=_NET_INPUT,
            num=int(source.num),
            freq=source.freq,
            power_dBm=source.power_dBm,
            z_ohm=float(source.z_ohm),
        )
        for source in design.sources
    )

    var_block = (
        VarBlockSpec(
            name="VAR1",
            variables=tuple(
                VarEntry(name=str(name), value=float(value))
                for name, value in design.parameters.items()
            ),
        )
        if design.parameters
        else None
    )

    sp = design.analysis.s_param
    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(sp.f_start_Hz),
            stop=float(sp.f_stop_Hz),
            points=int(sp.n_points),
        ),
        port_impedance_ohm=port_impedance,
    )
    substrate = _to_substrate(design)
    nodes = _collect_nodes(elements, sources)

    spec = DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=tuple(elements),
        ports=(Port(num=_OUTPUT_PORT_NUM, node=_NET_OUTPUT),),
        label="",
        ground_nodes=tuple(sorted(ground_nodes)),
        open_terminations=tuple(open_terminations),
        var_block=var_block,
        hb_single_tone=design.analysis.hb_single_tone,
        hb_two_tone=design.analysis.hb_two_tone,
        sources=sources,
    )

    # Fail fast: the resolved spec must pass the device-spec validator.
    # Resolving first means a named MTEE/MLIN width is a literal by the
    # time the (literal-only) MTEE checks run.
    from ..device_spec import resolve_spec, validate_device_spec

    validate_device_spec(resolve_spec(spec))
    return spec


def _cell_input_net(k: int) -> str:
    """The through-path net feeding cell ``k``."""

    return _NET_INPUT if k == 0 else f"n_cell{k - 1}_out"


def _cell_output_net(k: int, n_cells: int) -> str:
    """The through-path net leaving cell ``k`` (the design output if last)."""

    return _NET_OUTPUT if k == n_cells - 1 else f"n_cell{k}_out"


def _expand_cell(
    design: TVTLDesign,
    k: int,
    elements: list[Element],
    ground_nodes: set[str],
    open_terminations: list[str],
) -> None:
    """Expand cell ``k`` -- its through-line, MTEEs and shunt branches."""

    cell: CellSpec = design.cell
    substrate_name = design.substrate.name
    in_net = _cell_input_net(k)
    out_net = _cell_output_net(k, design.n_cells)
    width = cell.through_line.width
    length = cell.through_line.length

    if not cell.branches:
        # A pure transmission-line cell: the through-line spans the cell.
        elements.append(
            _mlin(f"MLIN_c{k}", in_net, out_net, width, length, substrate_name)
        )
        return

    # The through-line MLIN feeds the first MTEE's pin 1; the MTEEs are
    # chained pin-2 -> pin-1, the last MTEE's pin 2 being the cell output.
    junc = f"n_cell{k}_junc"
    elements.append(
        _mlin(f"MLIN_c{k}", in_net, junc, width, length, substrate_name)
    )
    seg_in = junc
    for b, branch in enumerate(cell.branches):
        is_last = b == len(cell.branches) - 1
        seg_out = out_net if is_last else f"n_cell{k}_seg{b}"
        br_top = f"n_cell{k}_br{b}_top"
        elements.append(
            Element(
                id=f"MTEE_c{k}_{b}",
                type="MTEE",
                pins=(
                    ElementPin(pin=1, node=seg_in),
                    ElementPin(pin=2, node=seg_out),
                    ElementPin(pin=3, node=br_top),
                ),
                parameters={
                    "w1_m": width,
                    "w2_m": width,
                    "w3_m": width,
                    "substrate_ref": substrate_name,
                },
            )
        )
        _expand_branch(
            design, k, b, branch, br_top, elements, ground_nodes,
            open_terminations,
        )
        seg_in = seg_out


def _expand_branch(
    design: TVTLDesign,
    k: int,
    b: int,
    branch: BranchSpec,
    tap_net: str,
    elements: list[Element],
    ground_nodes: set[str],
    open_terminations: list[str],
) -> None:
    """Expand one shunt branch -- the series chain from tap to terminus."""

    if branch.termination == "ground":
        # A per-branch ground net (still emitted as the literal 0). One
        # shared ground net would make every branch fork at it -- the
        # layout shunt tracer needs each branch to be a distinct chain.
        terminus = f"n_cell{k}_br{b}_gnd"
        ground_nodes.add(terminus)
    else:  # "floating" -- the validator has already checked the value.
        terminus = f"n_cell{k}_br{b}_open"
        open_terminations.append(terminus)

    substrate_name = design.substrate.name
    current = tap_net
    last = len(branch.elements) - 1
    for i, elem in enumerate(branch.elements):
        downstream = terminus if i == last else f"n_cell{k}_br{b}_n{i + 1}"
        if elem.type == _VARACTOR_TYPE:
            elements.append(
                _varactor(f"SDD_c{k}_b{b}_{i}", elem, current, downstream, k, b)
            )
        else:
            device_type, _token, needs_substrate = _BRANCH_TYPE_MAP.get(
                elem.type, ("", "", False)
            )
            if not device_type:
                raise DesignError(
                    f"cell.branches[{b}].elements[{i}] type {elem.type!r} is "
                    f"not expandable as a linear branch element; linear "
                    f"branches chain R / L / C / MLIN and terminate in an "
                    f"{_VARACTOR_TYPE}."
                )
            params = dict(elem.params)
            if needs_substrate:
                params["substrate_ref"] = substrate_name
            elements.append(
                Element(
                    id=f"{_BRANCH_TYPE_MAP[elem.type][1]}_c{k}_b{b}_{i}",
                    type=device_type,
                    pins=(
                        ElementPin(pin=1, node=current),
                        ElementPin(pin=2, node=downstream),
                    ),
                    parameters=params,
                )
            )
        current = downstream


def _varactor(
    element_id: str,
    elem,
    signal_net: str,
    return_net: str,
    k: int,
    b: int,
) -> Element:
    """Build the device-spec ``SDD2P`` element for an ``SDD2P_varactor``.

    Pin 1 is the branch-chain (signal) terminal; pin 2 is the branch
    terminus (ground or floating); pins 3 and 4 are tied to one per-cell,
    per-branch return net -- the port-2 self-tie the Level-5/7 canonicals
    use so the KCL-closure mirror current circulates within the port.
    """

    reference = elem.params.get(_Q_POLY_REFERENCE_KEY)
    builder = _Q_POLY_REGISTRY.get(str(reference))
    if builder is None:
        raise DesignError(
            f"SDD2P_varactor {_Q_POLY_REFERENCE_KEY}={reference!r} is not a "
            f"known charge-polynomial reference (known: "
            f"{sorted(_Q_POLY_REGISTRY)})."
        )
    ret_net = f"n_cell{k}_br{b}_ret"
    return Element(
        id=element_id,
        type="SDD2P",
        pins=(
            ElementPin(pin=1, node=signal_net),
            ElementPin(pin=2, node=return_net),
            ElementPin(pin=3, node=ret_net),
            ElementPin(pin=4, node=ret_net),
        ),
        parameters={
            "q_poly_coefficients": builder(),
            "q_poly_reference": str(reference),
        },
    )


def _mlin(
    element_id: str,
    node1: str,
    node2: str,
    width,
    length,
    substrate_name: str,
) -> Element:
    """Build an ``MLIN`` element spanning ``node1`` -> ``node2``."""

    return Element(
        id=element_id,
        type="MLIN",
        pins=(
            ElementPin(pin=1, node=node1),
            ElementPin(pin=2, node=node2),
        ),
        parameters={
            "w_m": width,
            "l_m": length,
            "substrate_ref": substrate_name,
        },
    )


def _to_substrate(design: TVTLDesign) -> Substrate:
    """Map the design's :class:`MSUBSpec` to a device-spec ``Substrate``."""

    msub = design.substrate
    return Substrate(
        name=msub.name,
        model=msub.model,
        parameters={
            "eps_r": float(msub.eps_r),
            "h_m": float(msub.h_m),
            "tan_delta": float(msub.tan_delta),
            "sigma_S_per_m": float(msub.sigma_S_per_m),
            "t_m": float(msub.t_m),
            "rough_m": float(msub.rough_m),
            "mur": float(msub.mur),
        },
    )


def _collect_nodes(
    elements: list[Element], sources: tuple[P1ToneSource, ...]
) -> tuple[Node, ...]:
    """Collect every net referenced by an element pin or a source.

    Order is first-seen so the node list is deterministic; the
    device-spec validator only needs uniqueness, not a particular order.
    """

    seen: dict[str, None] = {}
    for source in sources:
        seen.setdefault(source.port_node, None)
    for element in elements:
        for pin in element.pins:
            seen.setdefault(pin.node, None)
    return tuple(Node(id=node_id) for node_id in seen)


