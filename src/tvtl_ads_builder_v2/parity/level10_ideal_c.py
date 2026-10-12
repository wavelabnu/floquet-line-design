"""Level-10 *ideal-C* stub-loaded TVTL: a design-cross-validation variant.

This is a **design variant for TVTL/ADS cross-validation**, not a reproduction
of the paper's varactor. It preserves the paper's stub-loaded structural class
(per cell: main microstrip line split by an MTEE, a 3-segment shunt stub, and
2 vias to ground) but replaces the SMV1245 SDD2P + package parasitics with an
**ideal externally-pumped linear time-varying capacitor**:

    C(t) = C0 * (1 + depth*cos(w_pump*t + phi))

realised by a control-port bilinear-charge SDD (port 1 the in-line shunt, port 2
a zero-current control port driven by a shared pump source -- validated to run
in ADS HB).

The ideal-C screening found that the paper-magnitude C0 = 7 pF cannot parametrically amplify
in this class (the branch shorts the line when well coupled); positive gain
needs a smaller C0 (~0.3 pF) and the stub shortened by a scale factor ``k`` to
move the branch resonance in-band. The factory therefore parameterises by the
stub-length scale ``k`` and ``C0``; the k0.4 / k0.5 / k0.6 instances are the
screened finalists.

The analytical cross-validation leg uses the capacitance-linear branch
admittance :func:`tvtl.floquet.harmonic_abcd.tv_capacitor_harmonic_admittance`
with the same C0 / depth / stub geometry -- an apples-to-apples match to this
ADS device.
"""

from __future__ import annotations

import math

from tvtl.microstrip.discontinuities import via_inductance_bahl_H

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
from .canonical_circuits import (
    Level10SubstrateSpec,
    _DEFAULT_LEVEL10_VAR_NAMES,
)

__all__ = [
    "PAPER_STUB_SEGMENTS_MM",
    "control_port_k_from_depth",
    "open_circuit_pump_voltage",
    "make_level10_ideal_c_cascade",
    "make_level10_ideal_c_family",
]

# Paper stub: (width_mm, length_mm) from the varactor outward to the via.
PAPER_STUB_SEGMENTS_MM: tuple[tuple[float, float], ...] = (
    (1.93, 6.83),
    (1.41, 5.25),
    (1.98, 1.04),
)
_MAIN_WIDTH_M = 4.27e-3
_MAIN_PERIOD_M = 6.11e-3
_BRANCH_WIDTH_M = 1.93e-3


def open_circuit_pump_voltage(power_dBm: float, z_ohm: float = 50.0) -> float:
    """Open-circuit peak voltage of an ADS ``Port`` power source.

    An ADS ``Port`` with ``P[1]=polar(dbmtow(P),0)`` delivers available power
    ``P`` into a matched ``z_ohm`` load; into the zero-current control port it
    presents its open-circuit voltage ``Voc = 2*sqrt(2*z*P_watts)``. For
    ``z = 50`` this is ``20 * 10**((P_dBm - 30)/20)`` V (peak) -- verified
    against ADS HB (10 dBm -> 2.000 V).
    """

    p_watts = 10.0 ** ((float(power_dBm) - 30.0) / 10.0)
    return 2.0 * math.sqrt(2.0 * float(z_ohm) * p_watts)


def control_port_k_from_depth(
    depth: float, c0_f: float, pump_power_dBm: float, z_ohm: float = 50.0
) -> float:
    """Return the SDD control slope ``K`` (F/V) for a target modulation depth.

    The realised depth is ``m = K*Voc/C0`` where ``Voc`` is the pump's
    open-circuit control voltage (:func:`open_circuit_pump_voltage`). So
    ``K = depth*C0/Voc``.
    """

    voc = open_circuit_pump_voltage(pump_power_dBm, z_ohm)
    if voc <= 0.0:
        raise ValueError("pump open-circuit voltage must be positive.")
    return float(depth) * float(c0_f) / voc


def _ideal_c_cell_nodes(k: int, n_stub_segments: int) -> tuple[Node, ...]:
    nodes = [
        Node(id=f"n_c{k}_l"),
        Node(id=f"n_c{k}_r"),
        Node(id=f"n_c{k}_out"),
        Node(id=f"n_c{k}_b"),       # MTEE branch tap / SDD port-1 +
        Node(id=f"n_c{k}_v"),       # SDD port-1 - / into stub
    ]
    for i in range(n_stub_segments):
        nodes.append(Node(id=f"n_c{k}_s{i}"))
    nodes.append(Node(id=f"n_c{k}_via_top"))
    nodes.append(Node(id=f"n_c{k}_g"))
    return tuple(nodes)


def _ideal_c_cell_elements(
    *,
    k: int,
    input_net: str,
    output_net: str,
    control_net: str,
    stub_segments_m: tuple[tuple[float, float], ...],
    substrate: Substrate,
    via_inductance_H: float,
    c0_f: float,
    k_f_per_v: float,
    tvc_reference: str,
) -> tuple[Element, ...]:
    half_period = _MAIN_PERIOD_M / 2.0
    elements: list[Element] = []
    elements.append(Element(
        id=f"LINE_LEFT_C{k}", type="MLIN",
        pins=(ElementPin(pin=1, node=input_net), ElementPin(pin=2, node=f"n_c{k}_l")),
        parameters={"w_m": _MAIN_WIDTH_M, "l_m": half_period, "substrate_ref": substrate.name},
    ))
    elements.append(Element(
        id=f"TEE_C{k}", type="MTEE",
        pins=(
            ElementPin(pin=1, node=f"n_c{k}_l"),
            ElementPin(pin=2, node=f"n_c{k}_r"),
            ElementPin(pin=3, node=f"n_c{k}_b"),
        ),
        parameters={"w1_m": _MAIN_WIDTH_M, "w2_m": _MAIN_WIDTH_M,
                    "w3_m": _BRANCH_WIDTH_M, "substrate_ref": substrate.name},
    ))
    elements.append(Element(
        id=f"LINE_RIGHT_C{k}", type="MLIN",
        pins=(ElementPin(pin=1, node=f"n_c{k}_r"), ElementPin(pin=2, node=output_net)),
        parameters={"w_m": _MAIN_WIDTH_M, "l_m": half_period, "substrate_ref": substrate.name},
    ))
    # Ideal time-varying capacitor (control-port SDD). No L_pkg / R_pkg.
    # port1 = (n_c{k}_b, n_c{k}_v) in-line shunt; port2 = (control_net, gnd).
    elements.append(Element(
        id=f"VARACTOR_C{k}", type="SDD2P",
        pins=(
            ElementPin(pin=1, node=f"n_c{k}_b"),
            ElementPin(pin=2, node=f"n_c{k}_v"),
            ElementPin(pin=3, node=control_net),
            ElementPin(pin=4, node=f"n_c{k}_g"),
        ),
        parameters={
            "tvc_c0_f": float(c0_f),
            "tvc_k_f_per_v": float(k_f_per_v),
            "tvc_reference": tvc_reference,
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
            parameters={"w_m": float(w), "l_m": float(l), "substrate_ref": substrate.name},
        ))
    elements.append(Element(
        id=f"LVIA_C{k}", type="L",
        pins=(ElementPin(pin=1, node=f"n_c{k}_via_top"), ElementPin(pin=2, node=f"n_c{k}_g")),
        parameters={"l_h": float(via_inductance_H)},
    ))
    return tuple(elements)


def make_level10_ideal_c_cascade(
    *,
    stub_scale: float,
    c0_f: float = 0.3e-12,
    depth: float = 0.3,
    n_cells: int = 15,
    label: str,
    f_signal_Hz: float = 3.1e9,
    f_pump_Hz: float = 6.2e9,
    signal_power_dBm: float = -30.0,
    pump_power_dBm: float = 10.0,
    pump_phase_rad: float = 0.0,
    substrate: Level10SubstrateSpec = Level10SubstrateSpec(),
    f_start_Hz: float = 2.0e9,
    f_stop_Hz: float = 17.0e9,
    n_points: int = 151,
    port_impedance_ohm: float = 50.0,
    hb_order_signal: int = 8,
    hb_order_pump: int = 8,
    hb_max_order: int = 8,
) -> DeviceSpec:
    """Build a stub-loaded ideal-C TVTL cascade (design-cross-validation variant).

    Parameters
    ----------
    stub_scale : float
        Multiplies the paper stub-segment lengths (widths unchanged). The
        screened finalists are 0.4 / 0.5 / 0.6.
    c0_f : float
        Ideal static capacitance [F] (default 0.3 pF -- the screened value).
    depth : float
        Target modulation depth ``dC/C0``. The control-slope ``K`` is computed
        from ``depth``, ``c0_f`` and ``pump_power_dBm`` so the realised depth
        equals this value (control port draws zero current).
    n_cells : int
        Cell count (default 15).
    pump_power_dBm : float
        Pump power on the shared control net. With the zero-current control
        port this sets ``Voc`` and hence (with ``K``) the depth.
    """

    if int(n_cells) < 1:
        raise ValueError("n_cells must be >= 1.")
    substrate_obj = substrate.to_substrate()
    stub_segments_m = tuple(
        (w * 1e-3, l * 1e-3 * float(stub_scale)) for (w, l) in PAPER_STUB_SEGMENTS_MM
    )
    k_f_per_v = control_port_k_from_depth(
        depth, c0_f, pump_power_dBm, port_impedance_ohm
    )
    tvc_reference = (
        f"ideal_tv_capacitor(C0={c0_f * 1e12:.4g}pF,depth={depth:.3g},"
        f"stub_scale={stub_scale:.3g})"
    )
    l_via_H = via_inductance_bahl_H(
        substrate_thickness_m=float(substrate.h_m), via_radius_m=0.8e-3, n_parallel=2
    )

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
        nodes.extend(_ideal_c_cell_nodes(k, len(stub_segments_m)))
        elements.extend(_ideal_c_cell_elements(
            k=k, input_net=cell_in, output_net=cell_out, control_net=control_net,
            stub_segments_m=stub_segments_m, substrate=substrate_obj,
            via_inductance_H=l_via_H, c0_f=c0_f, k_f_per_v=k_f_per_v,
            tvc_reference=tvc_reference,
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


# Screened finalists (stub_scale, C0, label).
_FAMILY = {
    "k0p4": dict(stub_scale=0.4, c0_f=0.3e-12),
    "k0p5": dict(stub_scale=0.5, c0_f=0.3e-12),
    "k0p6": dict(stub_scale=0.6, c0_f=0.3e-12),
}


def make_level10_ideal_c_family(
    name: str, *, depth: float = 0.3, n_cells: int = 15, **kwargs
) -> DeviceSpec:
    """Build a named screened finalist: ``"k0p4"`` / ``"k0p5"`` / ``"k0p6"``."""

    if name not in _FAMILY:
        raise ValueError(f"unknown family member {name!r}; have {sorted(_FAMILY)}.")
    cfg = _FAMILY[name]
    return make_level10_ideal_c_cascade(
        stub_scale=cfg["stub_scale"], c0_f=cfg["c0_f"], depth=depth,
        n_cells=n_cells, label=f"level10_ideal_c_{name}", **kwargs
    )


