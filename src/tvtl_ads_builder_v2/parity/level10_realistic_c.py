"""Level-10 *realistic-varactor* stub-loaded TVTL.

A parameterised, ADS-emittable cascade in the paper's stub-loaded structural
class, with the **realistic** SMV1245-class varactor branch:

    n_c{k}_b --[L_pkg]--[R_series]--> SDD2P(charge-driven Q(v), CP folded)
             --> IPR_VAR --> stub MLIN cascade --> L_via --> ground

This is the same physical branch the paper-fidelity canonical
(:func:`...canonical_circuits.make_tvtl_level10_cascade`) emits, but driven by
a reusable :class:`tvtl.varactors.RealisticVaractor` (so CJ0/VJ/M/CP, L_pkg,
R_series and the DC bias are all one object) and parameterised by a
stub-length scale ``stub_scale``. The varactor is **self-pumped**: signal and
pump ``P_1Tone`` sources at ``n_input`` propagate down the line, and the pump
divides through ``L_pkg`` before swinging the junction -- the only realisation
that reproduces the L_pkg pump-shielding finding (a control-port pump
would bypass the divider). The charge polynomial is the SMV1245 SPICE Taylor
expansion **around the device's DC bias**, with ``CP`` folded into ``q[1]`` --
both handled inside :meth:`RealisticVaractor.charge_polynomial`, so this
generator never re-derives the convention.

Package-element emission convention: ``L_pkg`` is emitted
only when ``L_pkg > 0`` (a zero inductor is validator-illegal) and
``R_series`` only when ``R_series > 0``; an ideal (zero-parasitic) varactor
therefore produces a bare-junction branch with no ``L``/``R`` elements.

The analytical cross-validation leg
(:func:`...analytical.level10_realistic_param_gain`) consumes the same
``RealisticVaractor`` and the geometry of the spec this generator builds.
"""

from __future__ import annotations

import math

from tvtl.microstrip.discontinuities import via_inductance_bahl_H
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
    VarBlockSpec,
    VarEntry,
)
from ..device_spec.schema import DEVICE_SPEC_VERSION
from .canonical_circuits import (
    Level10SubstrateSpec,
    Level10ViaSpec,
    _DEFAULT_LEVEL10_VAR_NAMES,
)
from .level10_ideal_c import open_circuit_pump_voltage

__all__ = [
    "PAPER_STUB_SEGMENTS_MM",
    "realistic_q_poly_reference",
    "level10_realistic_c_canonical",
    "level10_realistic_c_control_port_canonical",
]

# Paper stub: (width_mm, length_mm) from the varactor outward to the via --
# the same geometry the paper-fidelity canonical and Level10IdealC use.
PAPER_STUB_SEGMENTS_MM: tuple[tuple[float, float], ...] = (
    (1.93, 6.83),
    (1.41, 5.25),
    (1.98, 1.04),
)
_MAIN_WIDTH_M = 4.27e-3
_MAIN_PERIOD_M = 6.11e-3
_BRANCH_WIDTH_M = 1.93e-3


def realistic_q_poly_reference(varactor: RealisticVaractor) -> str:
    """Stable audit string labelling the realistic varactor's Q(v) in datasets."""

    return (
        f"realistic_varactor(CJ0={varactor.CJ0_f * 1e12:.3g}pF,"
        f"VJ={varactor.VJ_v:.3g}V,M={varactor.M:.3g},"
        f"CP={varactor.CP_f * 1e12:.3g}pF,Vbias={varactor.V_bias_v:.3g}V,"
        f"order={varactor.taylor_order})"
    )


def realistic_control_port_reference(varactor: RealisticVaractor) -> str:
    """Audit string for the control-port realistic SDD (capacitance Taylor)."""

    return (
        f"realistic_varactor_control_port(CJ0={varactor.CJ0_f * 1e12:.3g}pF,"
        f"VJ={varactor.VJ_v:.3g}V,M={varactor.M:.3g},"
        f"CP={varactor.CP_f * 1e12:.3g}pF,Vbias={varactor.V_bias_v:.3g}V,"
        f"order={varactor.taylor_order})"
    )


def _realistic_cell_nodes(
    k: int, n_stub_segments: int, *, emit_lpkg: bool, emit_rseries: bool
) -> tuple[Node, ...]:
    nodes = [
        Node(id=f"n_c{k}_l"),       # MTEE pin 1 / LINE_LEFT pin 2
        Node(id=f"n_c{k}_r"),       # MTEE pin 2 / LINE_RIGHT pin 1
        Node(id=f"n_c{k}_out"),     # cell output
        Node(id=f"n_c{k}_b"),       # MTEE pin 3 (branch tap)
        Node(id=f"n_c{k}_v"),       # SDD2P pin 2 / IPR_VAR pin 1
        Node(id=f"n_c{k}_ret"),     # SDD2P pin 3/4 self-tie
    ]
    if emit_lpkg:
        nodes.append(Node(id=f"n_c{k}_lp"))   # after L_pkg
    if emit_rseries:
        nodes.append(Node(id=f"n_c{k}_rp"))   # after R_series
    for i in range(n_stub_segments):
        nodes.append(Node(id=f"n_c{k}_s{i}"))
    nodes.append(Node(id=f"n_c{k}_via_top"))
    nodes.append(Node(id=f"n_c{k}_g"))
    return tuple(nodes)


def _realistic_cell_elements(
    *,
    k: int,
    input_net: str,
    output_net: str,
    stub_segments_m: tuple[tuple[float, float], ...],
    substrate_name: str,
    via_inductance_H: float,
    varactor: RealisticVaractor,
    q_poly: tuple[float, ...],
    q_poly_reference: str,
) -> tuple[Element, ...]:
    half_period = _MAIN_PERIOD_M / 2.0
    emit_lpkg = float(varactor.L_pkg_h) > 0.0
    emit_rseries = float(varactor.R_series_ohm) > 0.0
    elements: list[Element] = []

    # Main-line halves split by the MTEE.
    elements.append(Element(
        id=f"LINE_LEFT_C{k}", type="MLIN",
        pins=(ElementPin(pin=1, node=input_net), ElementPin(pin=2, node=f"n_c{k}_l")),
        parameters={"w_m": _MAIN_WIDTH_M, "l_m": half_period, "substrate_ref": substrate_name},
    ))
    elements.append(Element(
        id=f"TEE_C{k}", type="MTEE",
        pins=(
            ElementPin(pin=1, node=f"n_c{k}_l"),
            ElementPin(pin=2, node=f"n_c{k}_r"),
            ElementPin(pin=3, node=f"n_c{k}_b"),
        ),
        parameters={"w1_m": _MAIN_WIDTH_M, "w2_m": _MAIN_WIDTH_M,
                    "w3_m": _BRANCH_WIDTH_M, "substrate_ref": substrate_name},
    ))
    elements.append(Element(
        id=f"LINE_RIGHT_C{k}", type="MLIN",
        pins=(ElementPin(pin=1, node=f"n_c{k}_r"), ElementPin(pin=2, node=output_net)),
        parameters={"w_m": _MAIN_WIDTH_M, "l_m": half_period, "substrate_ref": substrate_name},
    ))

    # Branch: [L_pkg] -> [R_series] -> SDD2P -> IPR -> stub -> L_via -> gnd.
    # Package elements are emitted only when non-zero (Q5).
    cur = f"n_c{k}_b"
    if emit_lpkg:
        elements.append(Element(
            id=f"LPKG_C{k}", type="L",
            pins=(ElementPin(pin=1, node=cur), ElementPin(pin=2, node=f"n_c{k}_lp")),
            parameters={"l_h": float(varactor.L_pkg_h)},
        ))
        cur = f"n_c{k}_lp"
    if emit_rseries:
        elements.append(Element(
            id=f"RPKG_C{k}", type="R",
            pins=(ElementPin(pin=1, node=cur), ElementPin(pin=2, node=f"n_c{k}_rp")),
            parameters={"r_ohm": float(varactor.R_series_ohm)},
        ))
        cur = f"n_c{k}_rp"
    # SDD2P: pin 1 = signal-side (cur), pin 2 = per-varactor measurement node
    # (n_c{k}_v), pins 3/4 tied to a shared per-cell return net (KCL closure).
    elements.append(Element(
        id=f"VARACTOR_C{k}", type="SDD2P",
        pins=(
            ElementPin(pin=1, node=cur),
            ElementPin(pin=2, node=f"n_c{k}_v"),
            ElementPin(pin=3, node=f"n_c{k}_ret"),
            ElementPin(pin=4, node=f"n_c{k}_ret"),
        ),
        parameters={
            "q_poly_coefficients": tuple(q_poly),
            "q_poly_reference": q_poly_reference,
        },
    ))
    elements.append(Element(
        id=f"IPR_VAR_C{k}", type="I_Probe",
        pins=(ElementPin(pin=1, node=f"n_c{k}_v"), ElementPin(pin=2, node=f"n_c{k}_s0")),
        parameters={},
    ))
    n_seg = len(stub_segments_m)
    for i, (w, l) in enumerate(stub_segments_m):
        seg_in = f"n_c{k}_s{i}"
        seg_out = f"n_c{k}_via_top" if i == n_seg - 1 else f"n_c{k}_s{i + 1}"
        elements.append(Element(
            id=f"STUB_C{k}_{i}", type="MLIN",
            pins=(ElementPin(pin=1, node=seg_in), ElementPin(pin=2, node=seg_out)),
            parameters={"w_m": float(w), "l_m": float(l), "substrate_ref": substrate_name},
        ))
    elements.append(Element(
        id=f"LVIA_C{k}", type="L",
        pins=(ElementPin(pin=1, node=f"n_c{k}_via_top"), ElementPin(pin=2, node=f"n_c{k}_g")),
        parameters={"l_h": float(via_inductance_H)},
    ))
    return tuple(elements)


def level10_realistic_c_canonical(
    *,
    varactor: RealisticVaractor,
    stub_scale: float = 1.0,
    n_cells: int = 15,
    spec_id: str | None = None,
    f_signal_Hz: float = 3.1e9,
    f_pump_Hz: float = 6.2e9,
    signal_power_dBm: float = -30.0,
    pump_power_dBm: float = 16.0,
    pump_phase_rad: float = 0.0,
    substrate: Level10SubstrateSpec = Level10SubstrateSpec(),
    via: Level10ViaSpec = Level10ViaSpec(),
    f_start_Hz: float = 2.0e9,
    f_stop_Hz: float = 17.0e9,
    n_points: int = 151,
    port_impedance_ohm: float = 50.0,
    hb_order_signal: int = 5,
    hb_order_pump: int = 5,
    hb_max_order: int = 6,
) -> DeviceSpec:
    """Build a Level-10 stub-loaded TVTL canonical with a realistic varactor.

    Parameters
    ----------
    varactor : RealisticVaractor
        The device. Supplies the charge polynomial (SMV1245 Taylor around the
        DC bias, CP folded -- via :meth:`RealisticVaractor.charge_polynomial`)
        and the package ``L_pkg`` / ``R_series`` branch elements (emitted only
        when non-zero).
    stub_scale : float
        Multiplies the paper stub-segment lengths (widths unchanged). 1.0 is
        the paper geometry; the Level10IdealC finalists used 0.4-0.6.
    n_cells : int
        Cell count (default 15).
    spec_id : str | None
        ``DeviceSpec.label`` override; default
        ``level10_realistic_c_k{stub_scale}``.
    f_signal_Hz, f_pump_Hz, signal_power_dBm, pump_power_dBm, pump_phase_rad :
        Two-tone operating point. The pump is on the line (self-pumped),
        Num=2; the signal Num=1; both at ``n_input``.
    """

    if int(n_cells) < 1:
        raise ValueError("n_cells must be >= 1.")
    substrate_obj = substrate.to_substrate()
    stub_segments_m = tuple(
        (w * 1e-3, l * 1e-3 * float(stub_scale)) for (w, l) in PAPER_STUB_SEGMENTS_MM
    )
    q_poly = varactor.charge_polynomial().charge_polynomial_coefficients
    q_poly_reference = realistic_q_poly_reference(varactor)
    l_via_H = via_inductance_bahl_H(
        substrate_thickness_m=float(substrate.h_m),
        via_radius_m=float(via.radius_m), n_parallel=int(via.n_parallel),
    )
    emit_lpkg = float(varactor.L_pkg_h) > 0.0
    emit_rseries = float(varactor.R_series_ohm) > 0.0

    label = spec_id if spec_id is not None else f"level10_realistic_c_k{stub_scale:g}"

    nodes: list[Node] = [Node(id="n_input"), Node(id="n_pin")]
    elements: list[Element] = []
    ground_nodes: list[str] = []

    elements.append(Element(
        id="IPR_IN", type="I_Probe",
        pins=(ElementPin(pin=1, node="n_input"), ElementPin(pin=2, node="n_pin")),
        parameters={},
    ))

    nc = int(n_cells)
    for k in range(1, nc + 1):
        cell_in = "n_pin" if k == 1 else f"n_pc{k - 1}c{k}"
        cell_out = f"n_c{k}_out"
        nodes.extend(_realistic_cell_nodes(
            k, len(stub_segments_m), emit_lpkg=emit_lpkg, emit_rseries=emit_rseries,
        ))
        elements.extend(_realistic_cell_elements(
            k=k, input_net=cell_in, output_net=cell_out,
            stub_segments_m=stub_segments_m, substrate_name=substrate_obj.name,
            via_inductance_H=l_via_H, varactor=varactor,
            q_poly=q_poly, q_poly_reference=q_poly_reference,
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
        P1ToneSource(name="PUMP_SRC", port_node="n_input", num=2,
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


# =========================================================================== #
# Control-port realistic realisation                                           #
# =========================================================================== #
# Mirrors the Level10IdealC control-port topology (the pump on a separate
# zero-current control net imposes the modulation directly, bypassing the
# package divider for pump *delivery*) but with (i) the realistic varactor's
# capacitance-vs-pump Taylor body on the SDD (tvc_poly mode -> full C-V
# curvature) and (ii) L_pkg / R_series kept as explicit series elements in the
# *signal* path (n_c{k}_b -> L -> R -> SDD port1), so realistic loading is
# preserved while the pump is cleanly imposed. This is the positive-gain
# cross-validation realisation.


def _control_port_cell_nodes(
    k: int, n_stub_segments: int, *, emit_lpkg: bool, emit_rseries: bool
) -> tuple[Node, ...]:
    nodes = [
        Node(id=f"n_c{k}_l"), Node(id=f"n_c{k}_r"), Node(id=f"n_c{k}_out"),
        Node(id=f"n_c{k}_b"), Node(id=f"n_c{k}_v"),
    ]
    if emit_lpkg:
        nodes.append(Node(id=f"n_c{k}_lp"))
    if emit_rseries:
        nodes.append(Node(id=f"n_c{k}_rp"))
    for i in range(n_stub_segments):
        nodes.append(Node(id=f"n_c{k}_s{i}"))
    nodes.append(Node(id=f"n_c{k}_via_top"))
    nodes.append(Node(id=f"n_c{k}_g"))
    return tuple(nodes)


def _control_port_cell_elements(
    *,
    k: int,
    input_net: str,
    output_net: str,
    control_net: str,
    stub_segments_m: tuple[tuple[float, float], ...],
    substrate_name: str,
    via_inductance_H: float,
    varactor: RealisticVaractor,
    tvc_poly: tuple[float, ...],
    tvc_reference: str,
) -> tuple[Element, ...]:
    half_period = _MAIN_PERIOD_M / 2.0
    emit_lpkg = float(varactor.L_pkg_h) > 0.0
    emit_rseries = float(varactor.R_series_ohm) > 0.0
    elements: list[Element] = []
    elements.append(Element(
        id=f"LINE_LEFT_C{k}", type="MLIN",
        pins=(ElementPin(pin=1, node=input_net), ElementPin(pin=2, node=f"n_c{k}_l")),
        parameters={"w_m": _MAIN_WIDTH_M, "l_m": half_period, "substrate_ref": substrate_name},
    ))
    elements.append(Element(
        id=f"TEE_C{k}", type="MTEE",
        pins=(ElementPin(pin=1, node=f"n_c{k}_l"), ElementPin(pin=2, node=f"n_c{k}_r"),
              ElementPin(pin=3, node=f"n_c{k}_b")),
        parameters={"w1_m": _MAIN_WIDTH_M, "w2_m": _MAIN_WIDTH_M,
                    "w3_m": _BRANCH_WIDTH_M, "substrate_ref": substrate_name},
    ))
    elements.append(Element(
        id=f"LINE_RIGHT_C{k}", type="MLIN",
        pins=(ElementPin(pin=1, node=f"n_c{k}_r"), ElementPin(pin=2, node=output_net)),
        parameters={"w_m": _MAIN_WIDTH_M, "l_m": half_period, "substrate_ref": substrate_name},
    ))
    # Signal-path package elements (kept; the control port bypasses them only
    # for pump delivery).
    cur = f"n_c{k}_b"
    if emit_lpkg:
        elements.append(Element(
            id=f"LPKG_C{k}", type="L",
            pins=(ElementPin(pin=1, node=cur), ElementPin(pin=2, node=f"n_c{k}_lp")),
            parameters={"l_h": float(varactor.L_pkg_h)},
        ))
        cur = f"n_c{k}_lp"
    if emit_rseries:
        elements.append(Element(
            id=f"RPKG_C{k}", type="R",
            pins=(ElementPin(pin=1, node=cur), ElementPin(pin=2, node=f"n_c{k}_rp")),
            parameters={"r_ohm": float(varactor.R_series_ohm)},
        ))
        cur = f"n_c{k}_rp"
    # Control-port SDD: port1 = (cur, n_c{k}_v) in-line shunt; port2 =
    # (control_net, n_c{k}_g) zero-current control driven by the pump.
    elements.append(Element(
        id=f"VARACTOR_C{k}", type="SDD2P",
        pins=(ElementPin(pin=1, node=cur), ElementPin(pin=2, node=f"n_c{k}_v"),
              ElementPin(pin=3, node=control_net), ElementPin(pin=4, node=f"n_c{k}_g")),
        parameters={"tvc_poly_coefficients": tuple(tvc_poly), "tvc_reference": tvc_reference},
    ))
    elements.append(Element(
        id=f"IPR_VAR_C{k}", type="I_Probe",
        pins=(ElementPin(pin=1, node=f"n_c{k}_v"), ElementPin(pin=2, node=f"n_c{k}_s0")),
        parameters={},
    ))
    n_seg = len(stub_segments_m)
    for i, (w, l) in enumerate(stub_segments_m):
        seg_in = f"n_c{k}_s{i}"
        seg_out = f"n_c{k}_via_top" if i == n_seg - 1 else f"n_c{k}_s{i + 1}"
        elements.append(Element(
            id=f"STUB_C{k}_{i}", type="MLIN",
            pins=(ElementPin(pin=1, node=seg_in), ElementPin(pin=2, node=seg_out)),
            parameters={"w_m": float(w), "l_m": float(l), "substrate_ref": substrate_name},
        ))
    elements.append(Element(
        id=f"LVIA_C{k}", type="L",
        pins=(ElementPin(pin=1, node=f"n_c{k}_via_top"), ElementPin(pin=2, node=f"n_c{k}_g")),
        parameters={"l_h": float(via_inductance_H)},
    ))
    return tuple(elements)


def level10_realistic_c_control_port_canonical(
    *,
    varactor: RealisticVaractor,
    stub_scale: float = 0.5,
    n_cells: int = 15,
    spec_id: str | None = None,
    f_signal_Hz: float = 3.1e9,
    f_pump_Hz: float = 6.2e9,
    signal_power_dBm: float = -30.0,
    pump_power_dBm: float = 10.0,
    pump_phase_rad: float = 0.0,
    substrate: Level10SubstrateSpec = Level10SubstrateSpec(),
    via: Level10ViaSpec = Level10ViaSpec(),
    f_start_Hz: float = 2.0e9,
    f_stop_Hz: float = 17.0e9,
    n_points: int = 151,
    port_impedance_ohm: float = 50.0,
    hb_order_signal: int = 8,
    hb_order_pump: int = 8,
    hb_max_order: int = 8,
) -> DeviceSpec:
    """Build a control-port realistic-varactor Level-10 cascade.

    The pump drives a shared zero-current control net (Num=2) and imposes the
    realistic varactor's capacitance-vs-pump law directly on each junction
    (``C(t) = C(V_bias + Voc*cos)`` via the SDD ``tvc_poly`` body), bypassing
    the package divider for pump delivery; ``L_pkg`` / ``R_series`` remain in
    the signal branch. The signal drives ``n_input`` (Num=1). This is the
    realistic positive-gain cross-validation device. The matching analytical
    call is ``level10_realistic_param_gain(spec, varactor,
    control_port_vpump_v=open_circuit_pump_voltage(pump_power_dBm))``.
    """

    if int(n_cells) < 1:
        raise ValueError("n_cells must be >= 1.")
    if varactor.ideal_linear:
        raise ValueError(
            "control-port realistic realisation requires a curved varactor; "
            "an ideal constant-C device has no pump response (use "
            "make_level10_ideal_c_* for the ideal control-port path)."
        )
    substrate_obj = substrate.to_substrate()
    stub_segments_m = tuple(
        (w * 1e-3, l * 1e-3 * float(stub_scale)) for (w, l) in PAPER_STUB_SEGMENTS_MM
    )
    tvc_poly = tuple(varactor.control_port_capacitance_coeffs())
    tvc_reference = realistic_control_port_reference(varactor)
    l_via_H = via_inductance_bahl_H(
        substrate_thickness_m=float(substrate.h_m),
        via_radius_m=float(via.radius_m), n_parallel=int(via.n_parallel),
    )
    emit_lpkg = float(varactor.L_pkg_h) > 0.0
    emit_rseries = float(varactor.R_series_ohm) > 0.0
    label = spec_id if spec_id is not None else f"level10_realistic_c_ctrlport_k{stub_scale:g}"
    control_net = "n_pump_ctrl"

    nodes: list[Node] = [Node(id="n_input"), Node(id="n_pin"), Node(id=control_net)]
    elements: list[Element] = []
    ground_nodes: list[str] = []
    elements.append(Element(
        id="IPR_IN", type="I_Probe",
        pins=(ElementPin(pin=1, node="n_input"), ElementPin(pin=2, node="n_pin")),
        parameters={},
    ))

    nc = int(n_cells)
    for k in range(1, nc + 1):
        cell_in = "n_pin" if k == 1 else f"n_pc{k - 1}c{k}"
        cell_out = f"n_c{k}_out"
        nodes.extend(_control_port_cell_nodes(
            k, len(stub_segments_m), emit_lpkg=emit_lpkg, emit_rseries=emit_rseries,
        ))
        elements.extend(_control_port_cell_elements(
            k=k, input_net=cell_in, output_net=cell_out, control_net=control_net,
            stub_segments_m=stub_segments_m, substrate_name=substrate_obj.name,
            via_inductance_H=l_via_H, varactor=varactor,
            tvc_poly=tvc_poly, tvc_reference=tvc_reference,
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


