"""Device-spec generator for optimized stub-loaded TVTL designs.

Emits a Level-10-shaped, ADS-emittable control-port cascade for an **arbitrary
stub topology** (the :mod:`tvtl.optimization.topology_generator` families),
in the paper structural class (per cell: main microstrip line split by an MTEE,
the optimized shunt stub, a control-port time-varying-capacitor SDD with the
realistic varactor's package ``L_pkg`` / ``R_series`` in the signal path). This
is the device-spec foundation the design workflow cross-validates: the
**unchanged** analytical leg
:func:`tvtl_ads_builder_v2.analytical.level10_realistic_param_gain` consumes the
spec this generator builds (same paper main TL, MTEE, control-port pump on a
zero-current net), so the design freedom lives entirely in the stub topology and
the main-line width + matching sections.

Scope
-----
* **Core:** OPEN and SHORT linear topology families, paper
  main-line width, no matching network. Used by the Stage-1 SingleCellFloquetA2
  fallback and the A1-correlation calibration cascade.
* **Extension:** ``main_width_m`` override (Stage 2 Bloch match) and
  ``input_match`` / ``output_match`` distributed-microstrip sections (Stage 3).
  Defaults (``None`` -> paper width / no matching) keep the signature stable.
* COMBINED families are analytical-screening-only; this generator
  raises for them (their ADS layout is a future tree-aware extension).

The generated spec is validator-clean (``validate_device_spec``) for OPEN/SHORT.
"""

from __future__ import annotations

from tvtl.microstrip.discontinuities import via_inductance_bahl_H
from tvtl.optimization.substrate import ROGERS_4003C, SubstrateConfig
from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    VIA_RADIUS_M,
    stub_segments,
)
from tvtl.varactors import RealisticVaractor

from ..device_spec import (
    Analysis,
    DeviceSpec,
    Element,
    ElementPin,
    FrequencyGrid,
    HBTwoToneSpec,
    Node,
    P1ToneSource,
    Port,
    Substrate,
    VarBlockSpec,
    VarEntry,
)
from ..device_spec.schema import DEVICE_SPEC_VERSION
from .canonical_circuits import _DEFAULT_LEVEL10_VAR_NAMES

__all__ = ["optimized_control_port_spec"]

# Paper structural-class main-line geometry (matches Level10IdealC /
# Level10RealisticC; the design freedom is the stub + the main-width override).
_MAIN_WIDTH_M = 4.27e-3
_MAIN_PERIOD_M = 6.11e-3
_BRANCH_WIDTH_M = 1.93e-3


def _match_elements_by_kind(section) -> tuple[tuple | None, tuple | None]:
    """Return ``(series_element, shunt_element)`` from a matching section.

    Order-independent: looks up the ``series_line`` and the ``shunt_*`` element
    by kind from ``section.elements`` (each ``(kind, w_m, l_m, z0_ohm)``).
    """

    series = shunt = None
    for el in section.elements:
        if el[0] == "series_line":
            series = el
        elif el[0] in ("shunt_open_stub", "shunt_short_stub"):
            shunt = el
    return series, shunt


def _emit_match_network(
    section,
    *,
    port_ref_net: str,
    port_casc_net: str,
    tag: str,
    substrate_name: str,
    l_via_H: float,
) -> tuple[list[Node], list[Element], list[str]]:
    """Emit a distributed-microstrip matching network as a 2-port.

    ``port_ref_net`` is the 50-ohm-reference-side node (source for input
    matching, load for output matching); ``port_casc_net`` is the cascade-side
    node. The single-stub network places the shunt stub at ``port_ref_net`` and
    a series line ``port_ref_net -> port_casc_net``; the quarter-wave network is
    a single series line ``port_ref_net -> port_casc_net``. Both reference nodes
    are assumed already declared by the caller.
    """

    nodes: list[Node] = []
    elements: list[Element] = []
    grounds: list[str] = []
    topology = section.topology

    if topology == "quarter_wave":
        series, _ = _match_elements_by_kind(section)
        if series is None:
            raise ValueError(f"quarter_wave section {tag} has no series_line element.")
        _, w, l, _z0 = series
        elements.append(Element(
            id=f"MATCH_{tag}_QW", type="MLIN",
            pins=(ElementPin(pin=1, node=port_ref_net), ElementPin(pin=2, node=port_casc_net)),
            parameters={"w_m": float(w), "l_m": float(l), "substrate_ref": substrate_name},
        ))
        return nodes, elements, grounds

    if topology == "single_stub":
        series, shunt = _match_elements_by_kind(section)
        if series is None or shunt is None:
            raise ValueError(f"single_stub section {tag} missing series/shunt element.")
        # series line: port_ref -> port_casc
        _, ws, ls, _zs = series
        elements.append(Element(
            id=f"MATCH_{tag}_SERIES", type="MLIN",
            pins=(ElementPin(pin=1, node=port_ref_net), ElementPin(pin=2, node=port_casc_net)),
            parameters={"w_m": float(ws), "l_m": float(ls), "substrate_ref": substrate_name},
        ))
        # shunt stub at port_ref
        kind, wst, lst, _zst = shunt
        stub_end = f"n_{tag}_stub_end"
        nodes.append(Node(id=stub_end))
        elements.append(Element(
            id=f"MATCH_{tag}_STUB", type="MLIN",
            pins=(ElementPin(pin=1, node=port_ref_net), ElementPin(pin=2, node=stub_end)),
            parameters={"w_m": float(wst), "l_m": float(lst), "substrate_ref": substrate_name},
        ))
        if kind == "shunt_short_stub":
            gnd = f"n_{tag}_stub_g"
            nodes.append(Node(id=gnd))
            elements.append(Element(
                id=f"MATCH_{tag}_STUB_VIA", type="L",
                pins=(ElementPin(pin=1, node=stub_end), ElementPin(pin=2, node=gnd)),
                parameters={"l_h": float(l_via_H)},
            ))
            grounds.append(gnd)
        # shunt_open_stub: stub_end floats (open) -- no via.
        return nodes, elements, grounds

    raise ValueError(f"unknown matching topology {topology!r} for section {tag}.")


def _substrate_from_config(substrate: SubstrateConfig) -> Substrate:
    params = substrate.stub_substrate_mapping()
    return Substrate(
        name=substrate.name, model="hammerstad_kirschning", parameters=params
    )


def _tvc_reference(topology: TopologyFamily, varactor: RealisticVaractor) -> str:
    return (
        f"optimized_control_port[{topology.key}]("
        f"CJ0={varactor.CJ0_f * 1e12:.3g}pF,VJ={varactor.VJ_v:.3g}V,"
        f"M={varactor.M:.3g},L_pkg={varactor.L_pkg_h * 1e9:.3g}nH,"
        f"R={varactor.R_series_ohm:.3g}ohm)"
    )


def _cell_nodes(
    k: int,
    n_segments: int,
    *,
    emit_lpkg: bool,
    emit_rseries: bool,
    end_node: str,
) -> list[Node]:
    nodes = [
        Node(id=f"n_c{k}_l"),
        Node(id=f"n_c{k}_r"),
        Node(id=f"n_c{k}_out"),
        Node(id=f"n_c{k}_b"),
        Node(id=f"n_c{k}_v"),
    ]
    if emit_lpkg:
        nodes.append(Node(id=f"n_c{k}_lp"))
    if emit_rseries:
        nodes.append(Node(id=f"n_c{k}_rp"))
    for i in range(n_segments):
        nodes.append(Node(id=f"n_c{k}_s{i}"))
    nodes.append(Node(id=end_node))
    nodes.append(Node(id=f"n_c{k}_g"))
    return nodes


def _cell_elements(
    *,
    k: int,
    input_net: str,
    output_net: str,
    control_net: str,
    topology: TopologyFamily,
    params,
    substrate_name: str,
    main_width_m: float,
    varactor: RealisticVaractor,
    tvc_poly: tuple[float, ...],
    tvc_reference: str,
    l_via_H: float,
    end_node: str,
) -> list[Element]:
    half_period = _MAIN_PERIOD_M / 2.0
    emit_lpkg = float(varactor.L_pkg_h) > 0.0
    emit_rseries = float(varactor.R_series_ohm) > 0.0
    els: list[Element] = []

    els.append(Element(
        id=f"LINE_LEFT_C{k}", type="MLIN",
        pins=(ElementPin(pin=1, node=input_net), ElementPin(pin=2, node=f"n_c{k}_l")),
        parameters={"w_m": float(main_width_m), "l_m": half_period, "substrate_ref": substrate_name},
    ))
    els.append(Element(
        id=f"TEE_C{k}", type="MTEE",
        pins=(ElementPin(pin=1, node=f"n_c{k}_l"), ElementPin(pin=2, node=f"n_c{k}_r"),
              ElementPin(pin=3, node=f"n_c{k}_b")),
        parameters={"w1_m": float(main_width_m), "w2_m": float(main_width_m),
                    "w3_m": _BRANCH_WIDTH_M, "substrate_ref": substrate_name},
    ))
    els.append(Element(
        id=f"LINE_RIGHT_C{k}", type="MLIN",
        pins=(ElementPin(pin=1, node=f"n_c{k}_r"), ElementPin(pin=2, node=output_net)),
        parameters={"w_m": float(main_width_m), "l_m": half_period, "substrate_ref": substrate_name},
    ))

    # Signal-path package elements (control port bypasses them only for pump delivery).
    cur = f"n_c{k}_b"
    if emit_lpkg:
        els.append(Element(
            id=f"LPKG_C{k}", type="L",
            pins=(ElementPin(pin=1, node=cur), ElementPin(pin=2, node=f"n_c{k}_lp")),
            parameters={"l_h": float(varactor.L_pkg_h)},
        ))
        cur = f"n_c{k}_lp"
    if emit_rseries:
        els.append(Element(
            id=f"RPKG_C{k}", type="R",
            pins=(ElementPin(pin=1, node=cur), ElementPin(pin=2, node=f"n_c{k}_rp")),
            parameters={"r_ohm": float(varactor.R_series_ohm)},
        ))
        cur = f"n_c{k}_rp"

    # Control-port SDD: port1 = (cur, n_c{k}_v) in-line shunt; port2 =
    # (control_net, n_c{k}_g) zero-current control driven by the pump.
    els.append(Element(
        id=f"VARACTOR_C{k}", type="SDD2P",
        pins=(ElementPin(pin=1, node=cur), ElementPin(pin=2, node=f"n_c{k}_v"),
              ElementPin(pin=3, node=control_net), ElementPin(pin=4, node=f"n_c{k}_g")),
        parameters={"tvc_poly_coefficients": tuple(tvc_poly), "tvc_reference": tvc_reference},
    ))
    els.append(Element(
        id=f"IPR_VAR_C{k}", type="I_Probe",
        pins=(ElementPin(pin=1, node=f"n_c{k}_v"), ElementPin(pin=2, node=f"n_c{k}_s0")),
        parameters={},
    ))

    segs = stub_segments(topology, params)
    n_seg = len(segs)
    for i, seg in enumerate(segs):
        seg_in = f"n_c{k}_s{i}"
        seg_out = end_node if i == n_seg - 1 else f"n_c{k}_s{i + 1}"
        els.append(Element(
            id=f"STUB_C{k}_{i}", type="MLIN",
            pins=(ElementPin(pin=1, node=seg_in), ElementPin(pin=2, node=seg_out)),
            parameters={"w_m": float(seg.width_m), "l_m": float(seg.length_m),
                        "substrate_ref": substrate_name},
        ))

    # Termination: SHORT -> via(s) to ground; OPEN -> floating end (no element).
    if topology.termination is Termination.SHORT:
        els.append(Element(
            id=f"LVIA_C{k}", type="L",
            pins=(ElementPin(pin=1, node=end_node), ElementPin(pin=2, node=f"n_c{k}_g")),
            parameters={"l_h": float(l_via_H)},
        ))
    return els


def optimized_control_port_spec(
    *,
    topology: TopologyFamily,
    params,
    varactor: RealisticVaractor,
    substrate: SubstrateConfig = ROGERS_4003C,
    n_cells: int = 15,
    main_width_m: float | None = None,
    f_signal_Hz: float = 3.1e9,
    f_pump_Hz: float = 6.2e9,
    signal_power_dBm: float = -30.0,
    pump_power_dBm: float = -3.0,
    pump_phase_rad: float = 0.0,
    via_radius_m: float = VIA_RADIUS_M,
    spec_id: str | None = None,
    f_start_Hz: float = 2.0e9,
    f_stop_Hz: float = 17.0e9,
    n_points: int = 151,
    port_impedance_ohm: float = 50.0,
    hb_order_signal: int = 8,
    hb_order_pump: int = 8,
    hb_max_order: int = 8,
    input_match=None,
    output_match=None,
) -> DeviceSpec:
    """Build a control-port realistic cascade for an optimized stub topology.

    Parameters
    ----------
    topology, params : TopologyFamily, sequence of float
        The stub topology (OPEN or SHORT linear family) and its continuous
        per-segment ``(width, length)`` parameter vector.
    varactor : RealisticVaractor
        Curved realistic varactor (control-port path requires C-V curvature).
    substrate : SubstrateConfig
        Substrate configuration; default Rogers 4003C.
    n_cells : int
        Cell count (15 for the design cascade; 1 for the A2 single-cell screen).
    main_width_m : float | None
        Main-line width override (Stage-2 Bloch match). ``None`` ->
        paper width ``4.27 mm``.
    input_match, output_match : matching section | None
        Stage-3 distributed-microstrip matching sections (objects exposing
        ``.topology`` in ``{"quarter_wave", "single_stub"}`` and ``.elements``,
        e.g. :class:`tvtl.optimization.stage3_matching.MatchingSection`).
        ``None`` emits no matching on that side (conditional emission, Q4).

    Returns
    -------
    DeviceSpec
        Validator-clean Level-10-shaped control-port cascade.
    """

    if int(n_cells) < 1:
        raise ValueError("n_cells must be >= 1.")
    if varactor.ideal_linear:
        raise ValueError(
            "control-port realisation requires a curved varactor; an ideal "
            "constant-C device has no pump response."
        )
    if topology.termination is Termination.COMBINED:
        raise NotImplementedError(
            "COMBINED-termination families are analytical-screening-only; "
            "their ADS layout (a parallel via||open "
            "stublet tree) is not implemented. Use topology_y_stub_fn for "
            "Stage-1 screening of COMBINED candidates."
        )
    width_main = _MAIN_WIDTH_M if main_width_m is None else float(main_width_m)
    substrate_obj = _substrate_from_config(substrate)
    tvc_poly = tuple(varactor.control_port_capacitance_coeffs())
    tvc_reference = _tvc_reference(topology, varactor)
    emit_lpkg = float(varactor.L_pkg_h) > 0.0
    emit_rseries = float(varactor.R_series_ohm) > 0.0
    short = topology.termination is Termination.SHORT
    l_via_H = (
        via_inductance_bahl_H(
            substrate_thickness_m=float(substrate.h_m),
            via_radius_m=float(via_radius_m),
            n_parallel=topology.n_via_parallel,
        )
        if short
        else 0.0
    )
    label = spec_id if spec_id is not None else f"optimized_{topology.key}_n{n_cells}"
    control_net = "n_pump_ctrl"

    nodes: list[Node] = [Node(id="n_input"), Node(id="n_pin"), Node(id=control_net)]
    elements: list[Element] = []
    ground_nodes: list[str] = []
    l_via_match_H = via_inductance_bahl_H(
        substrate_thickness_m=float(substrate.h_m),
        via_radius_m=float(via_radius_m), n_parallel=1,
    )

    # Input matching network (Stage 3) between the source node and the cascade
    # feed, when supplied; else the source feeds the cascade directly.
    if input_match is not None:
        feed = "n_mi_feed"
        nodes.append(Node(id=feed))
        mn_nodes, mn_els, mn_g = _emit_match_network(
            input_match, port_ref_net="n_input", port_casc_net=feed, tag="IN",
            substrate_name=substrate_obj.name, l_via_H=l_via_match_H,
        )
        nodes.extend(mn_nodes)
        elements.extend(mn_els)
        ground_nodes.extend(mn_g)
        elements.append(Element(
            id="IPR_IN", type="I_Probe",
            pins=(ElementPin(pin=1, node=feed), ElementPin(pin=2, node="n_pin")),
            parameters={},
        ))
    else:
        elements.append(Element(
            id="IPR_IN", type="I_Probe",
            pins=(ElementPin(pin=1, node="n_input"), ElementPin(pin=2, node="n_pin")),
            parameters={},
        ))

    nc = int(n_cells)
    for k in range(1, nc + 1):
        cell_in = "n_pin" if k == 1 else f"n_pc{k - 1}c{k}"
        cell_out = f"n_c{k}_out"
        end_node = f"n_c{k}_via_top" if short else f"n_c{k}_open"
        nodes.extend(_cell_nodes(
            k, topology.n_segments, emit_lpkg=emit_lpkg, emit_rseries=emit_rseries,
            end_node=end_node,
        ))
        elements.extend(_cell_elements(
            k=k, input_net=cell_in, output_net=cell_out, control_net=control_net,
            topology=topology, params=params, substrate_name=substrate_obj.name,
            main_width_m=width_main, varactor=varactor, tvc_poly=tvc_poly,
            tvc_reference=tvc_reference, l_via_H=l_via_H, end_node=end_node,
        ))
        ground_nodes.append(f"n_c{k}_g")
        if k < nc:
            inter = f"n_pc{k}c{k + 1}"
            nodes.append(Node(id=inter))
            elements.append(Element(
                id=f"IPR_C{k}C{k + 1}", type="I_Probe",
                pins=(ElementPin(pin=1, node=cell_out), ElementPin(pin=2, node=inter)),
                parameters={},
            ))

    nodes.append(Node(id="n_output"))
    last_out = f"n_c{nc}_out"
    # Output matching network (Stage 3) between the cascade output and the load,
    # when supplied; else the cascade feeds the load directly.
    if output_match is not None:
        casc = "n_mo_casc"
        nodes.append(Node(id=casc))
        elements.append(Element(
            id="IPR_OUT", type="I_Probe",
            pins=(ElementPin(pin=1, node=last_out), ElementPin(pin=2, node=casc)),
            parameters={},
        ))
        mn_nodes, mn_els, mn_g = _emit_match_network(
            output_match, port_ref_net="n_output", port_casc_net=casc, tag="OUT",
            substrate_name=substrate_obj.name, l_via_H=l_via_match_H,
        )
        nodes.extend(mn_nodes)
        elements.extend(mn_els)
        ground_nodes.extend(mn_g)
    else:
        elements.append(Element(
            id="IPR_OUT", type="I_Probe",
            pins=(ElementPin(pin=1, node=last_out), ElementPin(pin=2, node="n_output")),
            parameters={},
        ))
    elements.append(Element(
        id="P_OUT", type="TermG",
        pins=(ElementPin(pin=1, node="n_output"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 3},
    ))

    name_counter = 0

    def _add_name(net: str) -> None:
        nonlocal name_counter
        elements.append(Element(
            id=f"NAME_{name_counter:02d}_{net}", type="NAME",
            pins=(ElementPin(pin=1, node=net),), parameters={},
        ))
        name_counter += 1

    _add_name("n_input")
    _add_name(control_net)
    for k in range(1, nc + 1):
        _add_name(f"n_c{k}_b")
        _add_name(f"n_c{k}_v")
    _add_name("n_output")

    vn = _DEFAULT_LEVEL10_VAR_NAMES
    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name=vn["f_signal"], value=float(f_signal_Hz)),
            VarEntry(name=vn["f_pump"], value=float(f_pump_Hz)),
            VarEntry(name=vn["signal_power_dBm"], value=float(signal_power_dBm)),
            VarEntry(name=vn["pump_power_dBm"], value=float(pump_power_dBm)),
            VarEntry(name=vn["pump_phase_rad"], value=float(pump_phase_rad)),
        ),
    )
    sources = (
        P1ToneSource(name="SIGNAL_SRC", port_node="n_input", num=1,
                     freq=vn["f_signal"], power_dBm=vn["signal_power_dBm"]),
        P1ToneSource(name="PUMP_SRC", port_node=control_net, num=2,
                     freq=vn["f_pump"], power_dBm=vn["pump_power_dBm"],
                     phase_rad=vn["pump_phase_rad"]),
    )
    hb = HBTwoToneSpec(
        fundamental_freq_1=vn["f_signal"], fundamental_order_1=int(hb_order_signal),
        fundamental_freq_2=vn["f_pump"], fundamental_order_2=int(hb_order_pump),
        max_order=int(hb_max_order),
    )
    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate_obj,
        nodes=tuple(nodes),
        elements=tuple(elements),
        ports=(Port(num=3, node="n_output"),),
        label=str(label),
        ground_nodes=tuple(ground_nodes),
        var_block=var_block,
        hb_two_tone=hb,
        sources=sources,
    )


