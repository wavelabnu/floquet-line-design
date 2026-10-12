"""Canonical parity circuits for the ADS Builder v2 ladder.

Level 1 circuits are ``TermG -- MLIN -- TermG`` 2-ports. Level 2
circuits cascade two or three MLINs through shared internal nodes.
Level 3 circuits add one ``MTEE`` T-junction with a shunt branch MLIN
that is either grounded (short stub) or left open. Level 4 circuits
exercise the lumped ``R`` / ``L`` / ``C`` / ``RLC_SERIES`` elements: a
series RLC alone between the ports, the lossless (R = 0) RLC, an MLIN
in series with an RLC, and an MTEE with a grounded RLC shunt stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tvtl.design.nonlinear_capacitor import (
    smv1245_default_polynomial,
    smv1245_skyworks_spice_polynomial,
)
from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
from tvtl.microstrip.discontinuities import via_inductance_bahl_H
from tvtl.microstrip.models import hammerstad_kirschning

from ..device_spec import (
    Analysis,
    DeviceSpec,
    Element,
    ElementPin,
    FrequencyGrid,
    HBSingleToneSpec,
    HBTwoToneSpec,
    Node,
    P1ToneSource,
    Port,
    Substrate,
    VarBlockSpec,
    VarEntry,
)
from ..device_spec.schema import DEVICE_SPEC_VERSION


LEVEL_1_LABEL = "parity_level_1"
LEVEL_2_LABEL = "parity_level_2"
LEVEL_3_LABEL = "parity_level_3"
LEVEL_4_LABEL = "parity_level_4"
LEVEL_5_LABEL = "parity_level_5"
LEVEL_6_LABEL = "parity_level_6"
LEVEL_7_LABEL = "parity_level_7"
LEVEL_9_LABEL = "parity_level_9"
LEVEL_10_LABEL = "parity_level_10"

# Rogers 4003C-like substrate used throughout the repo's TVTL stack.
_LEVEL_1_SUBSTRATE = Substrate(
    name="msub1",
    model="hammerstad_kirschning",
    parameters={
        "eps_r": 3.38,
        "h_m": 1.54e-3,
        "tan_delta": 0.0027,
        "sigma_S_per_m": 5.8e7,
        "t_m": 17.5e-6,
        "rough_m": 0.0,
        "mur": 1.0,
    },
)


def make_termg_mlin_termg(
    *,
    w_m: float,
    l_m: float,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build a Level-1 ``TermG -- MLIN -- TermG`` device spec.

    Parameters
    ----------
    w_m, l_m : float
        Microstrip width and length in metres.
    label : str
        Free-text label written into ``DeviceSpec.label`` and used as
        the workspace directory name by the parity CLI.
    f_start_Hz, f_stop_Hz, n_points : float, float, int
        S-parameter frequency grid.
    port_impedance_ohm : float
        TermG reference impedance.
    substrate : Substrate
        Optional substrate override. Defaults to the repo-standard
        Rogers 4003C-like stack-up.
    """

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (Node(id="n_in"), Node(id="n_out"))
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    line = Element(
        id="LINE",
        type="MLIN",
        pins=(ElementPin(pin=1, node="n_in"), ElementPin(pin=2, node="n_out")),
        parameters={"w_m": float(w_m), "l_m": float(l_m), "substrate_ref": substrate.name},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    ports = (Port(num=1, node="n_in"), Port(num=2, node="n_out"))
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, line, p2),
        ports=ports,
        label=label,
    )


LEVEL_1_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    make_termg_mlin_termg(w_m=2.95e-3, l_m=1.00e-3, label="matched_short"),
    make_termg_mlin_termg(w_m=2.95e-3, l_m=10.00e-3, label="matched_long"),
    make_termg_mlin_termg(w_m=5.00e-3, l_m=5.00e-3, label="mismatched_50ohm_port"),
    make_termg_mlin_termg(w_m=4.27e-3, l_m=6.10e-3, label="paper_main_line_one_period"),
)


def _mlin_cascade_spec(
    *,
    widths_m: tuple[float, ...],
    lengths_m: tuple[float, ...],
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build a series cascade of N MLINs joined at internal nodes.

    The node chain is ``n_in -- n_seg1 -- n_seg2 -- ... -- n_out``: the
    cascade of ``len(widths_m)`` MLINs shares one internal node per
    junction. The internal nodes are named ``n_seg{k}`` for the node
    after segment ``k`` (1-indexed); the final segment ends on
    ``n_out``.

    Parameters
    ----------
    widths_m, lengths_m : tuple of float
        Per-segment microstrip width and length in metres. Must be the
        same length and contain at least two segments.
    label : str
        Workspace label.
    f_start_Hz, f_stop_Hz, n_points : float, float, int
        S-parameter frequency grid.
    port_impedance_ohm : float
        Port reference impedance.
    substrate : Substrate
        Substrate stack-up.
    """

    if len(widths_m) != len(lengths_m):
        raise ValueError("widths_m and lengths_m must have the same length.")
    n_segments = len(widths_m)
    if n_segments < 2:
        raise ValueError("a cascade spec needs at least two MLIN segments.")

    # Node chain: n_in, n_seg1, ..., n_seg{N-1}, n_out.
    internal_nodes = [f"n_seg{k}" for k in range(1, n_segments)]
    node_ids = ["n_in", *internal_nodes, "n_out"]
    nodes = tuple(Node(id=nid) for nid in node_ids)

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    lines: list[Element] = []
    for index in range(n_segments):
        in_node = node_ids[index]
        out_node = node_ids[index + 1]
        lines.append(
            Element(
                id=f"LINE{index + 1}",
                type="MLIN",
                pins=(
                    ElementPin(pin=1, node=in_node),
                    ElementPin(pin=2, node=out_node),
                ),
                parameters={
                    "w_m": float(widths_m[index]),
                    "l_m": float(lengths_m[index]),
                    "substrate_ref": substrate.name,
                },
            )
        )
    ports = (Port(num=1, node="n_in"), Port(num=2, node="n_out"))
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, *lines, p2),
        ports=ports,
        label=label,
    )


def make_two_mlins_series(
    *,
    w1_m: float,
    l1_m: float,
    w2_m: float,
    l2_m: float,
    label: str,
) -> DeviceSpec:
    """Build a two-MLIN cascade sharing one internal node ``n_seg1``.

    Topology::

        Port:P1   n_in   ...
        MLIN:LINE1 n_in   n_seg1  w=w1 l=l1
        MLIN:LINE2 n_seg1 n_out   w=w2 l=l2
        Port:P2   n_out  ...

    Parameters
    ----------
    w1_m, l1_m : float
        First segment width and length in metres.
    w2_m, l2_m : float
        Second segment width and length in metres.
    label : str
        Workspace label.
    """

    return _mlin_cascade_spec(
        widths_m=(w1_m, w2_m),
        lengths_m=(l1_m, l2_m),
        label=label,
    )


def make_three_mlins_series(
    *,
    widths_m: tuple[float, float, float],
    lengths_m: tuple[float, float, float],
    label: str,
) -> DeviceSpec:
    """Build a three-MLIN cascade sharing internal nodes ``n_seg1``/``n_seg2``.

    Topology::

        Port:P1   n_in   ...
        MLIN:LINE1 n_in   n_seg1  w=widths[0] l=lengths[0]
        MLIN:LINE2 n_seg1 n_seg2  w=widths[1] l=lengths[1]
        MLIN:LINE3 n_seg2 n_out   w=widths[2] l=lengths[2]
        Port:P2   n_out  ...

    Parameters
    ----------
    widths_m, lengths_m : tuple of three float
        Per-segment widths and lengths in metres.
    label : str
        Workspace label.
    """

    return _mlin_cascade_spec(
        widths_m=tuple(widths_m),
        lengths_m=tuple(lengths_m),
        label=label,
    )


LEVEL_2_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    # Circuit 1: two identical 5 mm MLINs in series. Splitting the
    # Level-1 'matched_long' 10 mm line into two collinear halves must
    # be electrically identical -- this is the cascade cross-check.
    make_two_mlins_series(
        w1_m=2.95e-3, l1_m=5.0e-3,
        w2_m=2.95e-3, l2_m=5.0e-3,
        label="two_identical_5mm_each",
    ),
    # Circuit 2: same width, asymmetric length split.
    make_two_mlins_series(
        w1_m=2.95e-3, l1_m=3.0e-3,
        w2_m=2.95e-3, l2_m=7.0e-3,
        label="same_width_split_3_7mm",
    ),
    # Circuit 3: step-width cascade (the MSTEP discovery circuit).
    make_two_mlins_series(
        w1_m=4.27e-3, l1_m=3.0e-3,   # paper main-line width
        w2_m=1.93e-3, l2_m=3.0e-3,   # paper stub-segment width
        label="step_width_4p27_to_1p93mm",
    ),
    # Circuit 4: three MLINs of mixed widths (two width steps).
    make_three_mlins_series(
        widths_m=(4.27e-3, 2.95e-3, 4.27e-3),
        lengths_m=(2.0e-3, 4.0e-3, 2.0e-3),
        label="three_segment_step_step",
    ),
)


# ===========================================================================
# Level 3: MTEE T-junction with a shunt branch MLIN
# ===========================================================================


def quarter_wave_length_m(
    width_m: float,
    fs_Hz: float,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> float:
    """Return the physical length of a quarter-wave microstrip line.

    The electrical length of one quarter wavelength is
    ``l = c / (4 * fs * sqrt(eps_eff))``, where ``eps_eff`` is the
    Hammerstad-Kirschning effective permittivity of a line of the given
    width on the given substrate, evaluated at ``fs_Hz``.

    Parameters
    ----------
    width_m : float
        Microstrip width in metres.
    fs_Hz : float
        Frequency at which the line is one quarter wavelength.
    substrate : Substrate
        Substrate stack-up.

    Returns
    -------
    float
        Physical length in metres.
    """

    _z0, eps_eff = hammerstad_kirschning(
        width_m=float(width_m),
        thickness_m=float(substrate.parameters["t_m"]),
        height_m=float(substrate.parameters["h_m"]),
        epsilon_r=float(substrate.parameters["eps_r"]),
        frequency_hz=np.asarray([float(fs_Hz)]),
    )
    eps = float(eps_eff[0])
    return SPEED_OF_LIGHT_M_PER_S / (4.0 * float(fs_Hz) * np.sqrt(eps))


def _make_tee_circuit(
    *,
    main_w_m: float,
    left_l_m: float,
    right_l_m: float,
    branch_w_m: float,
    branch_l_m: float,
    branch_grounded: bool,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build an MTEE T-junction circuit with one shunt branch MLIN.

    Topology::

        Port:P1    n_in
        MLIN:LINE_LEFT   n_in        -> n_junc_left
        MTEE:TEE1        n_junc_left, n_junc_right, n_branch_top
        MLIN:LINE_RIGHT  n_junc_right -> n_out
        MLIN:STUB        n_branch_top -> n_branch_end
        Port:P2    n_out

    When ``branch_grounded`` is true, ``n_branch_end`` is declared in
    ``ground_nodes`` (a short-circuited shunt stub); otherwise it is
    left dangling (an open stub).

    Parameters
    ----------
    main_w_m : float
        Width of both main-line MLINs and MTEE pins 1/2, in metres.
    left_l_m, right_l_m : float
        Lengths of the left/right main-line MLINs, in metres.
    branch_w_m : float
        Width of the branch MLIN and MTEE pin 3, in metres.
    branch_l_m : float
        Length of the branch MLIN, in metres.
    branch_grounded : bool
        Whether the branch terminus is grounded (short stub) or open.
    label : str
        Workspace label.
    f_start_Hz, f_stop_Hz, n_points : float, float, int
        S-parameter frequency grid.
    port_impedance_ohm : float
        Port reference impedance.
    substrate : Substrate
        Substrate stack-up.
    """

    nodes = (
        Node(id="n_in"),
        Node(id="n_junc_left"),
        Node(id="n_junc_right"),
        Node(id="n_branch_top"),
        Node(id="n_branch_end"),
        Node(id="n_out"),
    )
    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    line_left = Element(
        id="LINE_LEFT",
        type="MLIN",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_junc_left"),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(left_l_m),
            "substrate_ref": substrate.name,
        },
    )
    line_right = Element(
        id="LINE_RIGHT",
        type="MLIN",
        pins=(
            ElementPin(pin=1, node="n_junc_right"),
            ElementPin(pin=2, node="n_out"),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(right_l_m),
            "substrate_ref": substrate.name,
        },
    )
    tee = Element(
        id="TEE1",
        type="MTEE",
        pins=(
            ElementPin(pin=1, node="n_junc_left"),
            ElementPin(pin=2, node="n_junc_right"),
            ElementPin(pin=3, node="n_branch_top"),
        ),
        parameters={
            "w1_m": float(main_w_m),
            "w2_m": float(main_w_m),
            "w3_m": float(branch_w_m),
            "substrate_ref": substrate.name,
        },
    )
    stub = Element(
        id="STUB",
        type="MLIN",
        pins=(
            ElementPin(pin=1, node="n_branch_top"),
            ElementPin(pin=2, node="n_branch_end"),
        ),
        parameters={
            "w_m": float(branch_w_m),
            "l_m": float(branch_l_m),
            "substrate_ref": substrate.name,
        },
    )
    ports = (Port(num=1, node="n_in"), Port(num=2, node="n_out"))
    ground_nodes = ("n_branch_end",) if branch_grounded else ()
    # A non-grounded shunt-stub branch ends in an open circuit; declaring
    # it lets the adjacency audit (and the topology renderer) tell that
    # one-pin terminus net apart from an accidental dangling pin.
    open_terminations = () if branch_grounded else ("n_branch_end",)
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, line_left, tee, line_right, stub, p2),
        ports=ports,
        label=label,
        ground_nodes=ground_nodes,
        open_terminations=open_terminations,
    )


def make_tee_with_open_zero_length_stub(*, label: str) -> DeviceSpec:
    """MTEE whose branch is a near-zero-length open stub.

    The branch is electrically inactive (an open stub of 0.1 mm), so
    this circuit is near-equivalent to two cascaded MLINs through a
    T-junction. It tests whether the MTEE element introduces parasitic
    behaviour even when the branch carries no shunt load. The main line
    matches the Level-2 ``same_width_split_3_7mm`` cascade so the two
    can be compared directly.

    A literally zero-length MLIN is avoided because hpeesofsim may
    reject ``L=0``; 0.1 mm of open stub contributes a negligible
    sub-femtofarad shunt.
    """

    return _make_tee_circuit(
        main_w_m=2.95e-3,
        left_l_m=3.0e-3,
        right_l_m=7.0e-3,
        branch_w_m=1.93e-3,
        branch_l_m=0.1e-3,
        branch_grounded=False,
        label=label,
    )


def make_tee_with_short_zero_length_stub(*, label: str) -> DeviceSpec:
    """MTEE whose branch is a short grounded stub (near short at the junction).

    The branch is a 1.0 mm grounded MLIN -- a low-impedance shunt that
    produces strong reflection. A literally zero-length grounded stub
    would be an exact short (``|S21| = -inf``), a degenerate point
    useless for a dB-domain parity comparison; 1.0 mm gives a strong
    but finite shunt that both TVTL and ADS evaluate cleanly.
    """

    return _make_tee_circuit(
        main_w_m=4.27e-3,
        left_l_m=3.0e-3,
        right_l_m=3.0e-3,
        branch_w_m=1.93e-3,
        branch_l_m=1.0e-3,
        branch_grounded=True,
        label=label,
    )


def make_tee_with_short_stub_paper_dims(*, label: str) -> DeviceSpec:
    """MTEE with paper-relevant main-line and shunt-stub geometry.

    Main line: two 4.27 mm-wide, 3.05 mm-long MLINs meeting at the
    MTEE. Branch: a 1.93 mm-wide, 6.83 mm-long MLIN grounded at its
    far end (a realistic short-circuited shunt stub).
    """

    return _make_tee_circuit(
        main_w_m=4.27e-3,
        left_l_m=3.05e-3,
        right_l_m=3.05e-3,
        branch_w_m=1.93e-3,
        branch_l_m=6.83e-3,
        branch_grounded=True,
        label=label,
    )


def make_tee_with_quarter_wave_short_stub_at_3p1GHz(*, label: str) -> DeviceSpec:
    """MTEE with a shorted branch stub one quarter-wave long at 3.1 GHz.

    A quarter-wave shorted stub presents an open circuit at its branch
    port at the design frequency, so the main line transmits almost
    completely at 3.1 GHz and reflects strongly off-frequency. The
    stub length is computed from the Hammerstad-Kirschning effective
    permittivity of the branch microstrip at 3.1 GHz.

    The main line uses the 50-Ohm-matched microstrip width (2.95 mm on
    this substrate) and short 0.2 mm arms. Both choices isolate the
    stub resonance: a mismatched or electrically long main line carries
    its own reactive mismatch, which combines with the stub and shifts
    the whole-circuit |S21| transmission peak away from the stub's own
    quarter-wave frequency. With short matched arms the circuit reduces
    to the shunt stub alone, and the transmission peak lands cleanly on
    the 3.1 GHz grid bin where the shorted stub is open. (At 0.5 mm
    arms the peak already drifts to the 3.2 GHz bin -- the shorted-shunt
    transmission feature is only ~0.01 dB deep, so its argmax bin is
    sensitive to any residual main-line reactance.)
    """

    branch_w_m = 1.93e-3
    branch_l_m = quarter_wave_length_m(branch_w_m, fs_Hz=3.1e9)
    return _make_tee_circuit(
        main_w_m=2.95e-3,
        left_l_m=0.2e-3,
        right_l_m=0.2e-3,
        branch_w_m=branch_w_m,
        branch_l_m=branch_l_m,
        branch_grounded=True,
        label=label,
    )


LEVEL_3_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    make_tee_with_open_zero_length_stub(label="tee_open_zero_stub"),
    make_tee_with_short_zero_length_stub(label="tee_short_zero_stub"),
    make_tee_with_short_stub_paper_dims(label="tee_short_paper_dims"),
    make_tee_with_quarter_wave_short_stub_at_3p1GHz(label="tee_qw_short_3p1GHz"),
)


# ===========================================================================
# Level 4: lumped R / L / C and the composite RLC_SERIES
# ===========================================================================


def make_rlc_series_two_port(
    *,
    r_ohm: float,
    l_h: float,
    c_f: float,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build a Level-4 ``TermG -- RLC_SERIES -- TermG`` device spec.

    The composite series-RLC element connects the two ports directly.
    The circuit has no microstrip; the substrate is still declared (the
    schema requires one) but no line references it.

    Parameters
    ----------
    r_ohm, l_h, c_f : float
        Series resistance (Ohm), inductance (H) and capacitance (F).
    label : str
        Workspace label, also the ``DeviceSpec.label``.
    f_start_Hz, f_stop_Hz, n_points : float, float, int
        S-parameter frequency grid (default 1-10 GHz, 0.1 GHz step).
    port_impedance_ohm : float
        TermG reference impedance.
    substrate : Substrate
        Substrate stack-up (declared, but electrically unused here).
    """

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (Node(id="n_in"), Node(id="n_out"))
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    rlc = Element(
        id="RLC1",
        type="RLC_SERIES",
        pins=(ElementPin(pin=1, node="n_in"), ElementPin(pin=2, node="n_out")),
        parameters={"r_ohm": float(r_ohm), "l_h": float(l_h), "c_f": float(c_f)},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    ports = (Port(num=1, node="n_in"), Port(num=2, node="n_out"))
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, rlc, p2),
        ports=ports,
        label=label,
    )


def make_mlin_then_rlc(
    *,
    w_m: float,
    l_m: float,
    r_ohm: float,
    l_h: float,
    c_f: float,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build a Level-4 ``TermG -- MLIN -- RLC_SERIES -- TermG`` device spec.

    A distributed MLIN and a lumped series-RLC in cascade through the
    shared internal node ``n_mid``.

    Parameters
    ----------
    w_m, l_m : float
        MLIN width and length in metres.
    r_ohm, l_h, c_f : float
        Series-RLC resistance (Ohm), inductance (H) and capacitance (F).
    label : str
        Workspace label.
    """

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (Node(id="n_in"), Node(id="n_mid"), Node(id="n_out"))
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    line = Element(
        id="LINE",
        type="MLIN",
        pins=(ElementPin(pin=1, node="n_in"), ElementPin(pin=2, node="n_mid")),
        parameters={
            "w_m": float(w_m),
            "l_m": float(l_m),
            "substrate_ref": substrate.name,
        },
    )
    rlc = Element(
        id="RLC1",
        type="RLC_SERIES",
        pins=(ElementPin(pin=1, node="n_mid"), ElementPin(pin=2, node="n_out")),
        parameters={"r_ohm": float(r_ohm), "l_h": float(l_h), "c_f": float(c_f)},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    ports = (Port(num=1, node="n_in"), Port(num=2, node="n_out"))
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, line, rlc, p2),
        ports=ports,
        label=label,
    )


def make_tee_with_rlc_stub(
    *,
    main_w_m: float,
    left_l_m: float,
    right_l_m: float,
    branch_w_m: float,
    r_ohm: float,
    l_h: float,
    c_f: float,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build an MTEE T-junction with a grounded series-RLC shunt branch.

    Topology::

        Port:P1          n_in
        MLIN:LINE_LEFT   n_in         -> n_junc_left
        MTEE:TEE1        n_junc_left, n_junc_right, n_branch_top
        MLIN:LINE_RIGHT  n_junc_right -> n_out
        RLC_SERIES:STUB  n_branch_top -> n_branch_end   (grounded)
        Port:P2          n_out

    The shunt branch is a lumped series R-L-C from the MTEE branch pin
    to ground -- the Level-4 analogue of the Level-3 shorted microstrip
    stub, and the structural shape of the paper's varactor unit cell
    (the SMV1245 varactor is itself modelled as a series R + L + C(V)).

    ``r_ohm`` is deliberately non-zero: a finite series resistance damps
    the shunt branch, so this circuit is not a near-lossless
    strong-reflection degeneracy. It is parity-blocked nonetheless --
    the TVTL ideal-junction MTEE model leaves a ~1 dB |S21| gap against
    ADS's MTEE element (the Level-3 model-fidelity limitation), so the
    Path B gate records it parity-blocked rather than loosening the
    0.5 dB tolerance.

    Parameters
    ----------
    main_w_m : float
        Width of both main-line MLINs and MTEE pins 1/2, in metres.
    left_l_m, right_l_m : float
        Lengths of the left/right main-line MLINs, in metres.
    branch_w_m : float
        Width of MTEE pin 3 (the branch arm), in metres.
    r_ohm, l_h, c_f : float
        Shunt series-RLC resistance (Ohm), inductance (H), capacitance (F).
    label : str
        Workspace label.
    """

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (
        Node(id="n_in"),
        Node(id="n_junc_left"),
        Node(id="n_junc_right"),
        Node(id="n_branch_top"),
        Node(id="n_branch_end"),
        Node(id="n_out"),
    )
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    line_left = Element(
        id="LINE_LEFT",
        type="MLIN",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_junc_left"),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(left_l_m),
            "substrate_ref": substrate.name,
        },
    )
    line_right = Element(
        id="LINE_RIGHT",
        type="MLIN",
        pins=(
            ElementPin(pin=1, node="n_junc_right"),
            ElementPin(pin=2, node="n_out"),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(right_l_m),
            "substrate_ref": substrate.name,
        },
    )
    tee = Element(
        id="TEE1",
        type="MTEE",
        pins=(
            ElementPin(pin=1, node="n_junc_left"),
            ElementPin(pin=2, node="n_junc_right"),
            ElementPin(pin=3, node="n_branch_top"),
        ),
        parameters={
            "w1_m": float(main_w_m),
            "w2_m": float(main_w_m),
            "w3_m": float(branch_w_m),
            "substrate_ref": substrate.name,
        },
    )
    stub = Element(
        id="STUB",
        type="RLC_SERIES",
        pins=(
            ElementPin(pin=1, node="n_branch_top"),
            ElementPin(pin=2, node="n_branch_end"),
        ),
        parameters={"r_ohm": float(r_ohm), "l_h": float(l_h), "c_f": float(c_f)},
    )
    ports = (Port(num=1, node="n_in"), Port(num=2, node="n_out"))
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, line_left, tee, line_right, stub, p2),
        ports=ports,
        label=label,
        ground_nodes=("n_branch_end",),
    )


LEVEL_4_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    # Circuit 1: a lossy series RLC alone between the ports. At the
    # series resonance f0 = 1/(2*pi*sqrt(LC)) ~ 5.03 GHz the reactance
    # cancels and Z = R -- the transmission peak; off-resonance the
    # growing net reactance reflects.
    make_rlc_series_two_port(
        r_ohm=50.0,
        l_h=1.0e-9,
        c_f=1.0e-12,
        label="rlc_pure_50ohm",
    ),
    # Circuit 2: the lossless (R = 0) series RLC. At resonance Z = 0 --
    # a through short -- so |S21| peaks at 0 dB; the resonance is sharp
    # but lands between the 5.0 and 5.1 GHz grid bins.
    make_rlc_series_two_port(
        r_ohm=0.0,
        l_h=1.0e-9,
        c_f=1.0e-12,
        label="rlc_lossless",
    ),
    # Circuit 3: a 50-Ohm-matched MLIN (2.95 mm wide) in series with the
    # lossy RLC -- a lumped element composed with a distributed one.
    make_mlin_then_rlc(
        w_m=2.95e-3,
        l_m=5.0e-3,
        r_ohm=50.0,
        l_h=1.0e-9,
        c_f=1.0e-12,
        label="mlin_then_rlc",
    ),
    # Circuit 4: MTEE T-junction with a grounded series-RLC shunt stub
    # -- the structural shape of the paper's varactor unit cell. R is
    # 50 Ohm (non-zero) so the shunt branch is damped; the circuit is
    # still parity-blocked, inheriting the Level-3 MTEE ideal-junction
    # model gap (observed Path B ~0.997 dB).
    make_tee_with_rlc_stub(
        main_w_m=4.27e-3,
        left_l_m=3.05e-3,
        right_l_m=3.05e-3,
        branch_w_m=1.93e-3,
        r_ohm=50.0,
        l_h=1.0e-9,
        c_f=1.0e-12,
        label="mtee_with_rlc_stub",
    ),
)


# ===========================================================================
# Level 5: the SDD2P symbolically-defined-device varactor
# ===========================================================================


def make_single_sdd2p_varactor(
    *,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build the Level-5 ``P1 -- SDD2P -- GND`` diagnostic circuit.

    Topology::

        Port:P1   n_in
        SDD2P:VARACTOR  n_in n_gnd n_return n_return   (Q(_v1) charge poly)

    A single ``SDD2P`` shunts the input port to ground -- the simplest
    possible end-to-end exercise of the SDD2P infrastructure: no
    through-path, no MTEE, no other elements. Port 1 of the device
    spans ``n_in`` (the port node) and ``n_gnd`` (ground) and carries
    the charge polynomial; port 2 is the KCL closure -- its two
    terminals (pins 3 and 4) are tied to a single shared ``n_return``
    net so the mirror current ``I[2,1]=-Q(_v1)`` circulates within the
    shorted port. ``n_return`` therefore carries two pins of one
    element (an intentional self-tie, recognised by the adjacency
    audit), so it needs no open-termination declaration.

    The charge polynomial is the actual SMV1245 model from
    ``tvtl.design.nonlinear_capacitor`` -- the single source of truth
    for the polynomial coefficients shared with the analytical stack.
    """

    charge_model = smv1245_default_polynomial()
    q_poly = charge_model.charge_polynomial_coefficients

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (
        Node(id="n_in"),
        Node(id="n_gnd"),
        Node(id="n_return"),
    )
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    varactor = Element(
        id="VARACTOR",
        type="SDD2P",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_gnd"),
            ElementPin(pin=3, node="n_return"),
            ElementPin(pin=4, node="n_return"),
        ),
        parameters={
            "q_poly_coefficients": q_poly,
            "q_poly_reference": "smv1245_default_polynomial",
        },
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, varactor),
        ports=(Port(num=1, node="n_in"),),
        label=label,
        ground_nodes=("n_gnd",),
    )


def make_tee_with_sdd2p_varactor(
    *,
    main_w_m: float = 4.27e-3,
    left_l_m: float = 3.05e-3,
    right_l_m: float = 3.05e-3,
    branch_w_m: float = 1.93e-3,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build a TVTL unit cell: an MTEE T-junction with an SDD2P varactor
    shunt branch.

    Topology::

        Port:P1          n_in
        MLIN:LINE_LEFT   n_in         -> n_junc_left
        MTEE:TEE1        n_junc_left, n_junc_right, n_branch_top
        MLIN:LINE_RIGHT  n_junc_right -> n_out
        SDD2P:VARACTOR   n_branch_top n_gnd n_return n_return
        Port:P2          n_out

    This is the smallest meaningful integration of the
    ``SDD2P`` element in an ADS-relevant topology: the through path
    is two ``MLIN`` arms across an ``MTEE``, and the shunt branch is the
    SMV1245 nonlinear-capacitor varactor.

    The main-line / MTEE geometry defaults match the Level-4
    ``mtee_with_rlc_stub`` canonical circuit (the RLC-stub structural
    analogue of this unit cell). The SDD2P shunt wiring follows the
    public SDD2P convention: port 1 spans ``n_branch_top`` (the MTEE branch
    tap) and ``n_gnd`` (ground) and carries the charge polynomial
    ``I[1,1]=+Q(_v1)``; port 2 is the KCL closure, its two terminals
    tied to one shared ``n_return`` net carrying ``I[2,1]=-Q(_v1)``.

    The charge polynomial is the SMV1245 model from
    ``tvtl.design.nonlinear_capacitor`` -- the single source of truth
    shared with the analytical stack.

    Parameters
    ----------
    main_w_m : float
        Width of both main-line MLINs and MTEE pins 1/2, in metres.
    left_l_m, right_l_m : float
        Lengths of the left/right main-line MLINs, in metres.
    branch_w_m : float
        Width of MTEE pin 3 (the branch arm), in metres.
    label : str
        Workspace label.
    """

    charge_model = smv1245_default_polynomial()
    q_poly = charge_model.charge_polynomial_coefficients

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (
        Node(id="n_in"),
        Node(id="n_junc_left"),
        Node(id="n_junc_right"),
        Node(id="n_branch_top"),
        Node(id="n_gnd"),
        Node(id="n_out"),
        Node(id="n_return"),
    )
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    line_left = Element(
        id="LINE_LEFT",
        type="MLIN",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_junc_left"),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(left_l_m),
            "substrate_ref": substrate.name,
        },
    )
    tee = Element(
        id="TEE1",
        type="MTEE",
        pins=(
            ElementPin(pin=1, node="n_junc_left"),
            ElementPin(pin=2, node="n_junc_right"),
            ElementPin(pin=3, node="n_branch_top"),
        ),
        parameters={
            "w1_m": float(main_w_m),
            "w2_m": float(main_w_m),
            "w3_m": float(branch_w_m),
            "substrate_ref": substrate.name,
        },
    )
    line_right = Element(
        id="LINE_RIGHT",
        type="MLIN",
        pins=(
            ElementPin(pin=1, node="n_junc_right"),
            ElementPin(pin=2, node="n_out"),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(right_l_m),
            "substrate_ref": substrate.name,
        },
    )
    varactor = Element(
        id="VARACTOR",
        type="SDD2P",
        pins=(
            ElementPin(pin=1, node="n_branch_top"),
            ElementPin(pin=2, node="n_gnd"),
            ElementPin(pin=3, node="n_return"),
            ElementPin(pin=4, node="n_return"),
        ),
        parameters={
            "q_poly_coefficients": q_poly,
            "q_poly_reference": "smv1245_default_polynomial",
        },
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, line_left, tee, line_right, varactor, p2),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        label=label,
        ground_nodes=("n_gnd",),
    )


def make_single_var_block(
    *,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build the Level-5 ``P1 -- RLC_SERIES -- P2`` VAR-block circuit.

    Topology::

        Port:P1            n_in
        RLC_SERIES:RLC1    n_in -> n_out   (r_ohm/l_h/c_f via VAR refs)
        Port:P2            n_out
        VAR1               R, L, Cvar, f0

    A series RLC between the two ports whose three element parameters
    -- ``r_ohm``, ``l_h``, ``c_f`` -- are not literals but the names
    ``R``, ``L`` and ``Cvar`` of scalars declared in the circuit's VAR
    block. ``f0`` is a fourth declared scalar that no element
    references -- a VAR block may hold named constants nothing consumes.

    VAR is not an emitted ADS element (see ``docs/LIMITATIONS.md``): the
    var-resolver substitutes each reference with the variable's literal
    value before emission, so both backends emit an ordinary numeric
    R/L/C expansion. This circuit is the end-to-end exercise of that
    resolver -- a connected component whose parameters come entirely
    from the VAR block.

    The variable values are generic -- a 50 Ohm resistance, a 1 nH
    inductance, a 7 pF capacitance, a 3 GHz frequency -- not tied to
    any specific device model or paper geometry.
    """

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (Node(id="n_in"), Node(id="n_out"))
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    rlc = Element(
        id="RLC1",
        type="RLC_SERIES",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_out"),
        ),
        parameters={"r_ohm": "R", "l_h": "L", "c_f": "Cvar"},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name="R", value=50.0),
            VarEntry(name="L", value=1.0e-9),
            VarEntry(name="Cvar", value=7.0e-12),
            VarEntry(name="f0", value=3.0e9),
        ),
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, rlc, p2),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        label=label,
        var_block=var_block,
    )


LEVEL_5_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    make_single_sdd2p_varactor(label="level5_single_sdd2p_varactor"),
    make_tee_with_sdd2p_varactor(label="level5_tvtl_unit_cell"),
    make_single_var_block(label="level5_single_var_block"),
)


def make_single_tone_hb_controller(
    *,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build the Level-6 single-tone HB controller circuit.

    Topology::

        Port:P1   n_in
        R:LOAD    n_in -> n_out   (50 Ohm)
        Port:P2   n_out
        HB:HB1    Freq[1] / Order[1] / MaxOrder   (single-tone HB)

    A 50-Ohm series resistor between the two ports gives the
    harmonic-balance controller a connected component to analyse; the
    :class:`HBSingleToneSpec` carries the single-tone HB directive. This
    is the first Level-6 circuit -- it exercises the HB controller
    schema, dual emission and content check end to end.

    The HB parameters are generic test values -- a 3 GHz fundamental
    and 5 harmonics -- not tied to any device model or paper band. HB
    *sources* and the SDD2P-driven nonlinear path are exercised by the
    later canonicals; this circuit exercises the controller element type only.
    """

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (Node(id="n_in"), Node(id="n_out"))
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    load = Element(
        id="LOAD",
        type="R",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_out"),
        ),
        parameters={"r_ohm": 50.0},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    hb = HBSingleToneSpec(
        fundamental_freq=3.0e9,
        fundamental_order=5,
        max_order=5,
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, load, p2),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        label=label,
        hb_single_tone=hb,
    )


def make_two_tone_hb_controller(
    *,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build the Level-6 two-tone HB controller circuit.

    Topology::

        Port:P1   n_in
        R:LOAD    n_in -> n_out   (50 Ohm)
        Port:P2   n_out
        HB:HB1    Freq[1]/Freq[2], Order[1]/Order[2], MaxOrder

    A 50-Ohm series resistor between the two ports gives the two-tone
    harmonic-balance controller a connected component to analyse. The
    :class:`HBTwoToneSpec` carries a signal tone and a 2x pump tone --
    the arrangement a parametric-amplifier study drives the circuit
    with.

    The HB parameters are generic test values -- a 3 GHz signal, a
    6 GHz pump, 5 harmonics each, MaxOrder 10 -- not tied to any device
    model or paper band. HB *sources* and the SDD2P-driven nonlinear
    path are exercised by the later canonicals; this circuit exercises
    the two-tone controller element type only.
    """

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (Node(id="n_in"), Node(id="n_out"))
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 1},
    )
    load = Element(
        id="LOAD",
        type="R",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_out"),
        ),
        parameters={"r_ohm": 50.0},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 2},
    )
    hb = HBTwoToneSpec(
        fundamental_freq_1=3.0e9,
        fundamental_order_1=5,
        fundamental_freq_2=6.0e9,
        fundamental_order_2=5,
        max_order=10,
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(p1, load, p2),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        label=label,
        hb_two_tone=hb,
    )


def make_two_tone_p1tone_sources(
    *,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build the Level-7 isolated ``P_1Tone`` source circuit.

    Topology::

        P_1Tone:SIGNAL_SRC  n_in (0)   Freq=f_signal P=signal_power_dBm
        P_1Tone:PUMP_SRC    n_in (0)   Freq=f_pump   P=pump_power_dBm
        R:LOAD              n_in -> n_out   (50 Ohm)
        Port:P_OUT          n_out
        VAR1                f_signal, f_pump, signal_power_dBm, pump_power_dBm

    Two ``P_1Tone`` power sources on one input net -- v1's two-tone HB
    stimulus -- driving a 50-Ohm series load to the output port. The
    input carries no S-parameter port: the two sources are the input,
    exactly as v1's HB workspaces drop the input ``TermG`` in favour of
    the sources.

    Every source parameter is a named reference into the VAR block --
    ``freq`` and ``power_dBm`` for both tones -- so this circuit is the
    isolated end-to-end exercise of the source emission and
    its named-reference resolution. The variable values are generic
    test values, not tied to any device model or paper band.
    """

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (Node(id="n_in"), Node(id="n_out"))
    load = Element(
        id="LOAD",
        type="R",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_out"),
        ),
        parameters={"r_ohm": 50.0},
    )
    # Signal source = Num 1, pump source = Num 2, output TermG = Num 3.
    # Each P_1Tone source carries its own ADS port number so hpeesofsim
    # does not abort with "Duplicate Port or Term number" when the deck
    # runs end-to-end.
    p_out = Element(
        id="P_OUT",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 3},
    )
    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name="f_signal", value=3.0e9),
            VarEntry(name="f_pump", value=6.0e9),
            VarEntry(name="signal_power_dBm", value=-30.0),
            VarEntry(name="pump_power_dBm", value=5.0),
        ),
    )
    signal_src = P1ToneSource(
        name="SIGNAL_SRC",
        port_node="n_in",
        num=1,
        freq="f_signal",
        power_dBm="signal_power_dBm",
    )
    pump_src = P1ToneSource(
        name="PUMP_SRC",
        port_node="n_in",
        num=2,
        freq="f_pump",
        power_dBm="pump_power_dBm",
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(load, p_out),
        ports=(Port(num=3, node="n_out"),),
        label=label,
        var_block=var_block,
        sources=(signal_src, pump_src),
    )


LEVEL_6_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    make_single_tone_hb_controller(label="level6_single_tone_hb_controller"),
    make_two_tone_hb_controller(label="level6_two_tone_hb_controller"),
)


# ===========================================================================
# Level 7: the P_1Tone power source + the integrated two-tone-HB TVTL cell
# ===========================================================================


def make_tvtl_unit_cell_two_tone_hb(
    *,
    main_w_m: float = 4.27e-3,
    left_l_m: float = 3.05e-3,
    right_l_m: float = 3.05e-3,
    branch_w_m: float = 1.93e-3,
    label: str,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_1_SUBSTRATE,
) -> DeviceSpec:
    """Build the integrated Level-7 TVTL unit cell under two-tone HB.

    Topology::

        P_1Tone:SIGNAL_SRC  n_input (0)   Freq=f_signal P=signal_power_dBm
        P_1Tone:PUMP_SRC    n_input (0)   Freq=f_pump   P=pump_power_dBm
        MLIN:LINE_LEFT      n_input       -> n_junc_left
        MTEE:TEE1           n_junc_left, n_junc_right, n_branch_top
        MLIN:LINE_RIGHT     n_junc_right  -> n_output
        RLC_SERIES:STUB     n_branch_top  -> n_var_mid   (R_var/L_par/C_var)
        SDD2P:VARACTOR      n_var_mid n_gnd n_return n_return
        Port:P2 / TermG     n_output
        HB:HB1              Freq[1]=f_signal Freq[2]=f_pump
        VAR1                f_signal, f_pump, signal/pump_power_dBm,
                            R_var, L_par, C_var

    This is the canonical that, emitted and run in ADS, simulates
    parametric amplification in a TVTL unit cell from a single device
    spec. It composes the four element layers:

    * a VAR block centrally declaring the design scalars;
    * two ``P_1Tone`` sources -- v1's two-tone HB stimulus;
    * a two-tone HB controller;
    * the TVTL unit-cell topology of ``level5_tvtl_unit_cell``
      -- two MLIN main-line arms across an MTEE -- with the shunt
      branch a series RLC (the varactor's R + L + linear-C package
      parasitics) feeding the ``SDD2P`` nonlinear-charge varactor.

    A single named ``f_signal`` drives both ``HB:`` ``Freq[1]`` and the
    signal source's ``Freq``; ``f_pump`` likewise drives ``Freq[2]``
    and the pump source's ``Freq``; ``pump_power_dBm`` drives the pump
    source's ``P`` -- the integration point where one VAR declaration
    feeds the controller and the source together.

    The input carries no S-parameter port: the two sources are the
    input (v1 drops the input ``TermG`` in favour of the sources). The
    only port is the output ``TermG`` -- the through-line termination.

    The RLC and source values are generic test values: no SMV1245
    specifics. The ``SDD2P`` charge polynomial is the SMV1245 model
    from ``tvtl.design.nonlinear_capacitor``, the single source of
    truth shared with ``level5_tvtl_unit_cell`` and the analytical
    stack.
    """

    charge_model = smv1245_default_polynomial()
    q_poly = charge_model.charge_polynomial_coefficients

    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    nodes = (
        Node(id="n_input"),
        Node(id="n_junc_left"),
        Node(id="n_junc_right"),
        Node(id="n_branch_top"),
        Node(id="n_var_mid"),
        Node(id="n_gnd"),
        Node(id="n_return"),
        Node(id="n_output"),
    )
    line_left = Element(
        id="LINE_LEFT",
        type="MLIN",
        pins=(
            ElementPin(pin=1, node="n_input"),
            ElementPin(pin=2, node="n_junc_left"),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(left_l_m),
            "substrate_ref": substrate.name,
        },
    )
    tee = Element(
        id="TEE1",
        type="MTEE",
        pins=(
            ElementPin(pin=1, node="n_junc_left"),
            ElementPin(pin=2, node="n_junc_right"),
            ElementPin(pin=3, node="n_branch_top"),
        ),
        parameters={
            "w1_m": float(main_w_m),
            "w2_m": float(main_w_m),
            "w3_m": float(branch_w_m),
            "substrate_ref": substrate.name,
        },
    )
    line_right = Element(
        id="LINE_RIGHT",
        type="MLIN",
        pins=(
            ElementPin(pin=1, node="n_junc_right"),
            ElementPin(pin=2, node="n_output"),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(right_l_m),
            "substrate_ref": substrate.name,
        },
    )
    # The varactor package parasitics -- series R + L + linear C --
    # carried by named VAR references; generic test values.
    stub = Element(
        id="STUB",
        type="RLC_SERIES",
        pins=(
            ElementPin(pin=1, node="n_branch_top"),
            ElementPin(pin=2, node="n_var_mid"),
        ),
        parameters={"r_ohm": "R_var", "l_h": "L_par", "c_f": "C_var"},
    )
    varactor = Element(
        id="VARACTOR",
        type="SDD2P",
        pins=(
            ElementPin(pin=1, node="n_var_mid"),
            ElementPin(pin=2, node="n_gnd"),
            ElementPin(pin=3, node="n_return"),
            ElementPin(pin=4, node="n_return"),
        ),
        parameters={
            "q_poly_coefficients": q_poly,
            "q_poly_reference": "smv1245_default_polynomial",
        },
    )
    # Signal source = Num 1, pump source = Num 2, output TermG = Num 3.
    # Each P_1Tone source carries its own ADS port number so hpeesofsim
    # does not abort with "Duplicate Port or Term number" when the deck
    # runs end-to-end.
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_output"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 3},
    )
    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name="f_signal", value=3.0e9),
            VarEntry(name="f_pump", value=6.0e9),
            VarEntry(name="signal_power_dBm", value=-30.0),
            VarEntry(name="pump_power_dBm", value=5.0),
            VarEntry(name="R_var", value=2.0),
            VarEntry(name="L_par", value=2.0e-9),
            VarEntry(name="C_var", value=4.0e-12),
        ),
    )
    signal_src = P1ToneSource(
        name="SIGNAL_SRC",
        port_node="n_input",
        num=1,
        freq="f_signal",
        power_dBm="signal_power_dBm",
    )
    pump_src = P1ToneSource(
        name="PUMP_SRC",
        port_node="n_input",
        num=2,
        freq="f_pump",
        power_dBm="pump_power_dBm",
    )
    hb = HBTwoToneSpec(
        fundamental_freq_1="f_signal",
        fundamental_order_1=5,
        fundamental_freq_2="f_pump",
        fundamental_order_2=5,
        max_order=6,
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=nodes,
        elements=(line_left, tee, line_right, stub, varactor, p2),
        ports=(Port(num=3, node="n_output"),),
        label=label,
        ground_nodes=("n_gnd",),
        var_block=var_block,
        hb_two_tone=hb,
        sources=(signal_src, pump_src),
    )


LEVEL_7_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    make_two_tone_p1tone_sources(label="level7_two_tone_p1tone_sources"),
    make_tvtl_unit_cell_two_tone_hb(label="level7_tvtl_unit_cell_two_tone_hb"),
)


# ===========================================================================
# Level 9: probed TVTL cascades -- I_Probes at cell boundaries AND at each
# varactor. Sets up the per-cell gain analysis (boundary voltages +
# currents -> per-cell signal power; energy-conservation cross-check at the
# varactor between boundary-side and varactor-side power injection).
#
# Hand-authored DeviceSpecs (no design/ import) per the dependency rule.
# Topology mirrors level7_tvtl_unit_cell_two_tone_hb but with two
# refinements that make the cascade HB-RUNNABLE:
#
#   1. distinct ``Num`` on the two P_1Tone sources (signal Num=1, pump
#      Num=2, output Port Num=3). The Level-7/8 specs both share Num=1
#      across the two sources, which makes hpeesofsim's S_Param controller
#      fail with "Duplicate Port or Term number for SIGNAL_SRC" (verified
#      by direct live-HB probe). Distinct Num
#      avoids the duplicate-port trap so HB completes end-to-end.
#   2. an I_Probe in series with each varactor -- between SDD2P pin 2 and
#      the stub MLIN -- exposing per-varactor current. This is the
#      energy-conservation cross-check vehicle.
#
# Boundary I_Probe naming: IPR_IN / IPR_C{k}C{k+1} / IPR_OUT.
# Varactor probe naming:   IPR_VAR_C{k}.
# A live probe confirmed each I_Probe's saved
# current appears in the .ds HB1.HB varblock as ``<id>.i``; the analysis
# helpers read e.g. ``IPR_VAR_C1.i`` and ``IPR_C1C2.i`` directly.
#
# Named-node mechanism (NAME / nlabel): the Level-9 canonicals predate
# the NAME element type; per-cell node voltages are reached through the
# existing bijection-aware net mapping the structural cross-check uses.
# ===========================================================================


# Substrate -- mirrors the existing Level-8 design-API scheme
# (eps_r=3.4, h=1.5 mm, sigma=58 MS/m, tan_delta=0.003), so a Level-9
# canonical built and emitted reads identically to a Level-8-shaped
# device on the same dielectric.
_LEVEL_9_SUBSTRATE = Substrate(
    name="msub1",
    model="hammerstad_kirschning",
    parameters={
        "eps_r": 3.4,
        "h_m": 1.5e-3,
        "tan_delta": 0.003,
        "sigma_S_per_m": 58.0e6,
        "t_m": 17.5e-6,
        "rough_m": 0.0,
        "mur": 1.0,
    },
)


def _probed_cell_elements(
    *,
    cell_index: int,
    input_net: str,
    output_net: str,
    main_w_m: float,
    main_l_m: float,
    branch_w_m: float,
    stub_w_m: float,
    stub_l_m: float,
    substrate_name: str,
    q_poly_coefficients: tuple[float, ...],
    q_poly_reference: str,
) -> tuple[Element, ...]:
    """Return the six elements of one probed unit cell, in cascade order.

    Through path: ``input_net -> LINE_LEFT -> n_c{k}_l -> MTEE ->
    n_c{k}_r -> LINE_RIGHT -> output_net``. Branch from MTEE pin 3
    (``n_c{k}_b``): ``VARACTOR -> n_c{k}_v -> IPR_VAR -> n_c{k}_s ->
    STUB -> n_c{k}_g`` (the grounded stub end).
    """

    k = cell_index
    line_left = Element(
        id=f"LINE_LEFT_C{k}", type="MLIN",
        pins=(
            ElementPin(pin=1, node=input_net),
            ElementPin(pin=2, node=f"n_c{k}_l"),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(main_l_m),
            "substrate_ref": substrate_name,
        },
    )
    tee = Element(
        id=f"TEE_C{k}", type="MTEE",
        pins=(
            ElementPin(pin=1, node=f"n_c{k}_l"),
            ElementPin(pin=2, node=f"n_c{k}_r"),
            ElementPin(pin=3, node=f"n_c{k}_b"),
        ),
        parameters={
            "w1_m": float(main_w_m),
            "w2_m": float(main_w_m),
            "w3_m": float(branch_w_m),
            "substrate_ref": substrate_name,
        },
    )
    line_right = Element(
        id=f"LINE_RIGHT_C{k}", type="MLIN",
        pins=(
            ElementPin(pin=1, node=f"n_c{k}_r"),
            ElementPin(pin=2, node=output_net),
        ),
        parameters={
            "w_m": float(main_w_m),
            "l_m": float(main_l_m),
            "substrate_ref": substrate_name,
        },
    )
    varactor = Element(
        id=f"VARACTOR_C{k}", type="SDD2P",
        pins=(
            ElementPin(pin=1, node=f"n_c{k}_b"),
            ElementPin(pin=2, node=f"n_c{k}_v"),
            ElementPin(pin=3, node=f"n_c{k}_ret"),
            ElementPin(pin=4, node=f"n_c{k}_ret"),
        ),
        parameters={
            "q_poly_coefficients": tuple(q_poly_coefficients),
            "q_poly_reference": q_poly_reference,
        },
    )
    # IPR_VAR_C{k}: the varactor probe. Pin 1 sits on the varactor's
    # port-1- net (``n_c{k}_v``, the SDD2P pin-2 net); pin 2 enters the
    # stub. The analysis reads its current as ``IPR_VAR_C{k}.i`` -- the
    # per-varactor signal-frequency current for the energy-conservation
    # cross-check against the boundary-side power balance.
    ipr_var = Element(
        id=f"IPR_VAR_C{k}", type="I_Probe",
        pins=(
            ElementPin(pin=1, node=f"n_c{k}_v"),
            ElementPin(pin=2, node=f"n_c{k}_s"),
        ),
        parameters={},
    )
    stub = Element(
        id=f"STUB_C{k}", type="MLIN",
        pins=(
            ElementPin(pin=1, node=f"n_c{k}_s"),
            ElementPin(pin=2, node=f"n_c{k}_g"),
        ),
        parameters={
            "w_m": float(stub_w_m),
            "l_m": float(stub_l_m),
            "substrate_ref": substrate_name,
        },
    )
    return (line_left, tee, line_right, varactor, ipr_var, stub)


def _probed_cell_nodes(cell_index: int) -> tuple[Node, ...]:
    """The eight per-cell nodes of one probed unit cell."""

    k = cell_index
    return (
        Node(id=f"n_c{k}_l"),       # LINE_LEFT pin 2 / MTEE pin 1
        Node(id=f"n_c{k}_r"),       # MTEE pin 2 / LINE_RIGHT pin 1
        Node(id=f"n_c{k}_out"),     # LINE_RIGHT pin 2 -- the cell output
        Node(id=f"n_c{k}_b"),       # MTEE pin 3 -- branch tap
        Node(id=f"n_c{k}_v"),       # VARACTOR pin 2 / IPR_VAR pin 1
        Node(id=f"n_c{k}_s"),       # IPR_VAR pin 2 / STUB pin 1
        Node(id=f"n_c{k}_g"),       # STUB pin 2 -- grounded stub end
        Node(id=f"n_c{k}_ret"),     # SDD2P port-2 self-tie
    )


def make_probed_tvtl_cascade(
    *,
    n_cells: int,
    label: str,
    main_w_m: float = 4.0e-3,
    main_l_m: float = 3.0e-3,
    branch_w_m: float = 1.93e-3,
    stub_w_m: float = 1.93e-3,
    stub_l_m: float = 5.0e-3,
    f_start_Hz: float = 1.0e9,
    f_stop_Hz: float = 10.0e9,
    n_points: int = 91,
    port_impedance_ohm: float = 50.0,
    substrate: Substrate = _LEVEL_9_SUBSTRATE,
) -> DeviceSpec:
    """Build a hand-authored, HB-runnable, probed N-cell TVTL cascade.

    Topology (cell index k = 1 .. n_cells):

      n_input
        -> SIGNAL_SRC (P_1Tone, Num=1, Freq=f_signal)   |
        -> PUMP_SRC   (P_1Tone, Num=2, Freq=f_pump)     |  shared input
      n_input
        -> IPR_IN (I_Probe)
      n_pin
        -> cell 1: LINE_LEFT -> TEE -> LINE_RIGHT
                                  |__> VARACTOR -> IPR_VAR_C1 -> STUB -> ground
        -> IPR_C1C2 (I_Probe)
      ... repeated for cells 2 .. n_cells ...
        -> IPR_OUT (I_Probe)
      n_output
        -> P_OUT (TermG, Num=3, 50 Ohm)

    ``n_cells >= 1``. n_cells=1 is a valid degenerate single-cell case
    (no inter-cell probes, just IPR_IN and IPR_OUT). The two Level-9
    canonicals use n_cells=2 and n_cells=3.

    The two P_1Tone sources carry **distinct ``Num`` values** (1 and 2)
    -- the Level-7/8 specs both use Num=1, which makes hpeesofsim's
    S_Param controller fail with "Duplicate Port or Term number" and
    aborts HB end-to-end. The output port is Num=3 so all three Port:
    lines in the OA-generated netlist carry distinct port numbers and
    nothing collides.
    """

    if int(n_cells) < 1:
        raise ValueError(f"n_cells must be >= 1 (got {n_cells}).")

    charge_model = smv1245_default_polynomial()
    q_poly = tuple(charge_model.charge_polynomial_coefficients)
    q_poly_ref = "smv1245_default_polynomial"

    elements: list[Element] = []
    nodes: list[Node] = [Node(id="n_input"), Node(id="n_pin")]
    ground_nodes: list[str] = []

    # IPR_IN spans n_input -> n_pin. The two P_1Tone sources drive
    # n_input directly; cell 1's LINE_LEFT pin 1 sits on n_pin (the
    # post-probe net).
    elements.append(
        Element(
            id="IPR_IN", type="I_Probe",
            pins=(
                ElementPin(pin=1, node="n_input"),
                ElementPin(pin=2, node="n_pin"),
            ),
            parameters={},
        )
    )

    # Walk the cells, building each one's elements + nodes, and splicing
    # inter-cell I_Probes between adjacent cell boundaries.
    for k in range(1, int(n_cells) + 1):
        cell_in = "n_pin" if k == 1 else f"n_pc{k - 1}c{k}"
        cell_out = f"n_c{k}_out"  # LINE_RIGHT pin 2 -- the cell's output net
        nodes.extend(_probed_cell_nodes(k))
        elements.extend(
            _probed_cell_elements(
                cell_index=k,
                input_net=cell_in,
                output_net=cell_out,
                main_w_m=main_w_m,
                main_l_m=main_l_m,
                branch_w_m=branch_w_m,
                stub_w_m=stub_w_m,
                stub_l_m=stub_l_m,
                substrate_name=substrate.name,
                q_poly_coefficients=q_poly,
                q_poly_reference=q_poly_ref,
            )
        )
        ground_nodes.append(f"n_c{k}_g")
        # Inter-cell probe between cell k and cell k+1 (skipped after the
        # last cell -- that boundary is taken by IPR_OUT below).
        if k < int(n_cells):
            inter_net = f"n_pc{k}c{k + 1}"
            nodes.append(Node(id=inter_net))
            elements.append(
                Element(
                    id=f"IPR_C{k}C{k + 1}", type="I_Probe",
                    pins=(
                        ElementPin(pin=1, node=cell_out),
                        ElementPin(pin=2, node=inter_net),
                    ),
                    parameters={},
                )
            )

    # IPR_OUT spans the last cell's output to n_output.
    nodes.append(Node(id="n_output"))
    last_cell_out = f"n_c{int(n_cells)}_out"
    elements.append(
        Element(
            id="IPR_OUT", type="I_Probe",
            pins=(
                ElementPin(pin=1, node=last_cell_out),
                ElementPin(pin=2, node="n_output"),
            ),
            parameters={},
        )
    )
    # Output port: Num=3 (distinct from the two sources' 1 and 2).
    elements.append(
        Element(
            id="P_OUT", type="TermG",
            pins=(ElementPin(pin=1, node="n_output"),),
            parameters={"z_ohm": float(port_impedance_ohm), "num": 3},
        )
    )

    # NAME annotations. ADS HB writes a ``.ds``
    # voltage column for every net with an explicit schematic name; an
    # un-named net stays ``N__nn`` and its voltage stays out of the
    # dataset. The Level-9 cascade names every per-cell observable
    # net so the power and gain helpers, the
    # energy-conservation cross-check, and (later) the Floquet
    # analytical-vs-ADS comparison all have the V they need to run.
    #
    # Naming convention -- the NAME pin's spec ``node`` IS the
    # dataset column name. The set is:
    #
    # * design-input boundary -- ``n_input`` (source side) and
    #   ``n_pin`` (cell-0 cell-facing side); both sides of IPR_IN
    #   named so the boundary I_Probe's V is reachable from either
    #   side.
    # * per-cell output -- ``n_c{k}_out`` (LINE_RIGHT pin 2). For
    #   k < N this also doubles as the upstream side of IPR_C{k}C{k+1}.
    # * inter-cell post-probe -- ``n_pc{k-1}c{k}`` (LINE_LEFT_C{k}
    #   pin 1 / IPR_C{k-1}C{k} downstream side), for k = 2 .. N.
    # * design-output boundary -- ``n_output`` (load side; the cell-
    #   facing side ``n_c{N}_out`` is already named above).
    # * per-cell varactor branch top -- ``n_c{k}_b`` (MTEE pin 3 /
    #   SDD2P pin 1, the branch-input observable
    #   ``varactor_power`` reads V from).
    # * per-cell SDD2P-pin-2 -- ``n_c{k}_v`` (between the varactor
    #   and IPR_VAR_C{k}, the natural V observable when the
    #   energy-conservation check reads the energy across the varactor itself).
    #
    # Total: 2 endpoints + 2 endpoint-cell-facing + (N-1) inter-cell
    # + 2*N varactor + N cell-output = 4*N + 2 NAMEs (e.g. 10 for
    # N=2, 14 for N=3). Each is a 1-pin element targeting an already-
    # declared spec node; the validator enforces uniqueness per net.
    name_id_counter = 0

    def _add_name(net: str) -> None:
        nonlocal name_id_counter
        elements.append(
            Element(
                id=f"NAME_{name_id_counter:02d}_{net}",
                type="NAME",
                pins=(ElementPin(pin=1, node=net),),
                parameters={},
            )
        )
        name_id_counter += 1

    _add_name("n_input")
    _add_name("n_pin")
    for k in range(1, int(n_cells) + 1):
        _add_name(f"n_c{k}_b")
        _add_name(f"n_c{k}_v")
        _add_name(f"n_c{k}_out")
        if k < int(n_cells):
            _add_name(f"n_pc{k}c{k + 1}")
    _add_name("n_output")

    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name="f_signal", value=3.0e9),
            VarEntry(name="f_pump", value=6.0e9),
            VarEntry(name="signal_power_dBm", value=-30.0),
            VarEntry(name="pump_power_dBm", value=5.0),
        ),
    )
    sources = (
        P1ToneSource(
            name="SIGNAL_SRC", port_node="n_input", num=1,
            freq="f_signal", power_dBm="signal_power_dBm",
        ),
        P1ToneSource(
            name="PUMP_SRC", port_node="n_input", num=2,
            freq="f_pump", power_dBm="pump_power_dBm",
        ),
    )
    hb = HBTwoToneSpec(
        fundamental_freq_1="f_signal", fundamental_order_1=5,
        fundamental_freq_2="f_pump", fundamental_order_2=5,
        max_order=6,
    )
    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz),
            points=int(n_points),
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate,
        nodes=tuple(nodes),
        elements=tuple(elements),
        ports=(Port(num=3, node="n_output"),),
        label=label,
        ground_nodes=tuple(ground_nodes),
        var_block=var_block,
        hb_two_tone=hb,
        sources=sources,
    )


LEVEL_9_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    make_probed_tvtl_cascade(
        n_cells=2, label="level9_two_cell_probed_tvtl_cascade",
    ),
    make_probed_tvtl_cascade(
        n_cells=3, label="level9_three_cell_probed_tvtl_cascade",
    ),
)


# ===========================================================================
# Level 10: paper-geometry TVTL cascade
# ===========================================================================
# The paper-fidelity canonical that the publication's quantitative claims
# depend on. Unlike Levels 7-9 (deliberately abstract structural tests),
# Level-10 represents the fabricated structure -- N cells, Rogers 4003C,
# SMV1245-011LF varactor (Skyworks SPICE Taylor expansion), microstrip
# dimensions per Figure 3b. The 15-cell paper-fidelity instance returned by
# :func:`make_paper_geometry_tvtl_cascade` is the publication's reference
# device. The generic generator :func:`make_tvtl_level10_cascade` accepts
# the per-cell geometry, varactor parameters, via configuration, and
# substrate as data so the same code path produces any paper-shaped device.
#
# Per-cell topology (cell index k = 1..n_cells):
#
#   n_in_k -> LINE_LEFT_C{k}(MLIN)  -> n_c{k}_l  -> TEE_C{k}(MTEE)
#                                                  | pin 1 -> n_c{k}_l (main left)
#                                                  | pin 2 -> n_c{k}_r (main right)
#                                                  | pin 3 -> n_c{k}_b (branch tap)
#                                  -> n_c{k}_r -> LINE_RIGHT_C{k}(MLIN) -> n_c{k}_out
#
#   n_c{k}_b
#     -> LPKG_C{k}(L_pkg)             -> n_c{k}_lp
#     -> RPKG_C{k}(R_pkg=Skyworks RS) -> n_c{k}_rp
#     -> VARACTOR_C{k}(SDD2P, Skyworks Taylor Q(v))    } parallel
#     -> CPAR_C{k}(C=CP), pin 1=n_c{k}_rp, pin 2=n_c{k}_v   } across SDD2P
#     -> IPR_VAR_C{k}(I_Probe)        -> n_c{k}_s0
#     -> STUB_C{k}_0(MLIN seg 0)      -> n_c{k}_s1
#     -> STUB_C{k}_1(MLIN seg 1)      -> n_c{k}_s2
#     ...                                 (n stub segments per cell)
#     -> STUB_C{k}_{S-1}(MLIN last)   -> n_c{k}_via_top
#     -> LVIA_C{k}(L_via parallel-combined)
#     -> n_c{k}_g (grounded — emitted as the literal ``0``)
#
# Boundary I_Probes: IPR_IN at n_input → n_pin, IPR_C{k}C{k+1} between
# adjacent cells, IPR_OUT at the last cell's output → n_output. P_OUT is
# the lone output TermG (Num=3); two P_1Tone sources (signal Num=1, pump
# Num=2) drive n_input. The VAR block carries f_signal, f_pump,
# signal_power_dBm, pump_power_dBm, pump_phase_rad defaults; a future
# pump-sweep API can replace the VAR block to drive operating-point
# sweeps without rebuilding the spec.
#
# NAME annotations (locked: minimum-needed): n_input, n_output, and one
# per varactor (n_c{k}_v). The energy-conservation and
# Floquet-comparison harnesses read voltages off the n_c{k}_v columns;
# the input/output boundaries are needed for the dataset's port-level
# response. No per-cell inter-cell NAMEs (a deliberate lean dataset).
# ===========================================================================


@dataclass(frozen=True)
class Level10SubstrateSpec:
    """Substrate parameters for a Level-10 device.

    Default values are Rogers 4003C as fabricated for the paper
    (``eps_r = 3.38``, ``h = 1.54 mm``, ``tan_delta = 0.0027``,
    copper ``17.5 μm = 1/2 oz``). The conductivity is the standard
    copper figure ``5.8e7 S/m``.
    """

    eps_r: float = 3.38
    h_m: float = 1.54e-3
    tan_delta: float = 0.0027
    sigma_S_per_m: float = 5.8e7
    t_m: float = 17.5e-6
    rough_m: float = 0.0
    mur: float = 1.0
    name: str = "msub1"
    model: str = "hammerstad_kirschning"

    def to_substrate(self) -> Substrate:
        return Substrate(
            name=self.name,
            model=self.model,
            parameters={
                "eps_r": float(self.eps_r),
                "h_m": float(self.h_m),
                "tan_delta": float(self.tan_delta),
                "sigma_S_per_m": float(self.sigma_S_per_m),
                "t_m": float(self.t_m),
                "rough_m": float(self.rough_m),
                "mur": float(self.mur),
            },
        )


@dataclass(frozen=True)
class Level10StubSegmentSpec:
    """One microstrip stub segment in a Level-10 cell.

    A cell carries a series chain of these segments between the varactor
    branch tap and the via to ground. The paper has three segments per
    cell (``l_2/w_2``, ``l_3/w_3``, ``l_4/w_4`` in Figure 3b).
    """

    width_m: float
    length_m: float


@dataclass(frozen=True)
class Level10ViaSpec:
    """Via-to-ground configuration for the end of one cell's stub.

    The Bahl simple inductance formula
    (:func:`tvtl.microstrip.discontinuities.via_inductance_bahl_H`) is
    applied to each via with the design's substrate thickness; for
    ``n_parallel > 1`` the result is divided by ``n_parallel``. The
    paper's fabricated structure has two parallel vias per cell.
    """

    n_parallel: int = 2
    radius_m: float = 0.8e-3


@dataclass(frozen=True)
class Level10VaractorSpec:
    """Skyworks SMV1245-011LF varactor + package parasitics.

    The locked Level-10 defaults are:

    * ``cj0_f = 6.9 pF``  — Skyworks CJ0
    * ``vj_v  = 3.5 V``   — Skyworks VJ
    * ``m_grade = 1.7``    — Skyworks M
    * ``cp_f  = 0.47 pF`` — package parasitic capacitance (parallel)
    * ``rs_ohm = 2 Ω``    — Skyworks RS (series)
    * ``l_pkg_h = 1.7 nH`` — external package L (series)
    * ``analytical_c0_f = 7.0 pF`` — paper-stated static C0 (slightly
      different from CJ0; used by the analytical Floquet solver via the
      ``q_poly_coefficients`` parameter at index 1).
    * ``taylor_order = 5`` — Q(v) Taylor expansion order for SDD2P
      emission; capped at 5 to fit the
      :class:`ChargePolynomialModel` schema.
    """

    cj0_f: float = 6.9e-12
    vj_v: float = 3.5
    m_grade: float = 1.7
    cp_f: float = 0.47e-12
    rs_ohm: float = 2.0
    l_pkg_h: float = 1.7e-9
    analytical_c0_f: float | None = 7.0e-12
    taylor_order: int = 5


@dataclass(frozen=True)
class Level10CellSpec:
    """Geometry of one Level-10 unit cell.

    Captures the per-cell main microstrip line width and the cell
    period (length along the main line; ``LINE_LEFT`` and
    ``LINE_RIGHT`` each span ``main_period_m / 2``), the branch
    microstrip width at the MTEE tap, the ordered stub segment chain
    feeding the via, and the via configuration. The paper has 15
    identical cells; the generic generator accepts a tuple so future
    non-uniform cascades remain expressible.
    """

    main_width_m: float
    main_period_m: float
    branch_width_m: float
    stub_segments: tuple[Level10StubSegmentSpec, ...]
    via: Level10ViaSpec = field(default_factory=Level10ViaSpec)


# Default VAR-block variable names. A sweep consumer can override these
# to drive operating-point sweeps without rebuilding the spec.
_DEFAULT_LEVEL10_VAR_NAMES: dict[str, str] = {
    "f_signal": "f_signal",
    "f_pump": "f_pump",
    "signal_power_dBm": "signal_power_dBm",
    "pump_power_dBm": "pump_power_dBm",
    "pump_phase_rad": "pump_phase_rad",
}

# Default VAR-block values (the locked first-run operating point). All
# five carry sensible defaults so a Level-10
# build runs HB end-to-end without any sweep configuration; a future
# sweep API replaces the VAR block to drive an operating-point sweep.
_DEFAULT_LEVEL10_VAR_VALUES: dict[str, float] = {
    "f_signal": 3.1e9,
    "f_pump": 6.2e9,
    "signal_power_dBm": -30.0,
    "pump_power_dBm": 16.0,
    "pump_phase_rad": 0.0,
}


def _q_poly_from_varactor(varactor: Level10VaractorSpec) -> tuple[float, ...]:
    """Return the SDD2P ``q_poly_coefficients`` tuple for ``varactor``.

    Builds the Skyworks SMV1245 Taylor polynomial of ``Q(v)`` around
    ``v = 0``, overrides ``q[1]`` with ``analytical_c0_f`` if the user
    supplied one (the paper-stated ``C0 = 7.0 pF`` replaces the Skyworks
    ``CJ0 = 6.9 pF`` on the static term only), and folds ``cp_f`` into
    the same ``q[1]`` so the SDD2P element represents the combined
    junction + parasitic package capacitance.

    Folding ``CP`` into ``q[1]`` is mathematically equivalent to a
    parallel ``C = CP`` across the SDD2P: the total displacement
    current into the parallel pair is
    ``i = dQ_junction/dt + CP·dv/dt = d(Q_junction + CP·v)/dt``, so
    augmenting ``Q(v) -> Q(v) + CP·v`` produces the same current at
    every voltage. Higher-order ``Q(v)`` coefficients (``q[2..]``) are
    unchanged because ``CP`` is voltage-independent.

    The fold is a deliberate workaround for the v2 OA layout engine's
    single-chain shunt-branch constraint: a literal parallel ``C``
    element across the SDD2P creates a fork (two elements at the same
    chain net) that triggers a grid-layout fallback. Emitting the
    combined polynomial preserves the physics, the analytical-side
    static stub ABCD (which reads ``q[1]`` as the static varactor
    capacitance) sees the same combined value, and ADS HB sees the
    same total dQ/dt at the SDD2P pin -- so no information about CP is
    lost.
    """

    model = smv1245_skyworks_spice_polynomial(
        cj0_f=float(varactor.cj0_f),
        vj_v=float(varactor.vj_v),
        m_grade=float(varactor.m_grade),
        order=int(varactor.taylor_order),
    )
    q_poly = list(model.charge_polynomial_coefficients)
    if varactor.analytical_c0_f is not None:
        q_poly[1] = float(varactor.analytical_c0_f)
    # Fold the parasitic package capacitance ``CP`` into ``q[1]`` so the
    # SDD2P element represents (junction || CP). See the docstring above
    # for the equivalence proof and the layout-engine context.
    q_poly[1] = float(q_poly[1]) + float(varactor.cp_f)
    return tuple(float(value) for value in q_poly)


def _q_poly_reference_for_varactor(varactor: Level10VaractorSpec) -> str:
    """Return a stable text reference labelling the polynomial in datasets.

    The string is recorded on the SDD2P element's
    ``q_poly_reference`` parameter so reading the spec back surfaces
    the model lineage. It mirrors the
    ``smv1245_default_polynomial`` token Level-5/7/8/9 use, with the
    suffix carrying the Skyworks parameters and Taylor order.
    """

    cj0 = float(varactor.cj0_f) * 1.0e12  # display in pF
    vj = float(varactor.vj_v)
    m = float(varactor.m_grade)
    order = int(varactor.taylor_order)
    return (
        f"smv1245_skyworks_spice_taylor_polynomial(cj0={cj0:.3g}pF,"
        f"vj={vj:.3g}V,m={m:.3g},order={order})"
    )


def _level10_cell_elements(
    *,
    cell_index: int,
    input_net: str,
    output_net: str,
    cell: Level10CellSpec,
    varactor: Level10VaractorSpec,
    substrate: Substrate,
    via_inductance_H: float,
    q_poly: tuple[float, ...],
    q_poly_reference: str,
) -> tuple[Element, ...]:
    """Build the explicit Element instances of one Level-10 unit cell."""

    k = cell_index
    half_period = float(cell.main_period_m) / 2.0
    elements: list[Element] = []

    # Main-line halves split by the MTEE.
    elements.append(
        Element(
            id=f"LINE_LEFT_C{k}", type="MLIN",
            pins=(
                ElementPin(pin=1, node=input_net),
                ElementPin(pin=2, node=f"n_c{k}_l"),
            ),
            parameters={
                "w_m": float(cell.main_width_m),
                "l_m": float(half_period),
                "substrate_ref": substrate.name,
            },
        )
    )
    elements.append(
        Element(
            id=f"TEE_C{k}", type="MTEE",
            pins=(
                ElementPin(pin=1, node=f"n_c{k}_l"),
                ElementPin(pin=2, node=f"n_c{k}_r"),
                ElementPin(pin=3, node=f"n_c{k}_b"),
            ),
            parameters={
                "w1_m": float(cell.main_width_m),
                "w2_m": float(cell.main_width_m),
                "w3_m": float(cell.branch_width_m),
                "substrate_ref": substrate.name,
            },
        )
    )
    elements.append(
        Element(
            id=f"LINE_RIGHT_C{k}", type="MLIN",
            pins=(
                ElementPin(pin=1, node=f"n_c{k}_r"),
                ElementPin(pin=2, node=output_net),
            ),
            parameters={
                "w_m": float(cell.main_width_m),
                "l_m": float(half_period),
                "substrate_ref": substrate.name,
            },
        )
    )

    # Branch: L_pkg -> R_pkg -> (SDD2P || CP) -> IPR_VAR -> stub segs ->
    # L_via -> ground.
    elements.append(
        Element(
            id=f"LPKG_C{k}", type="L",
            pins=(
                ElementPin(pin=1, node=f"n_c{k}_b"),
                ElementPin(pin=2, node=f"n_c{k}_lp"),
            ),
            parameters={"l_h": float(varactor.l_pkg_h)},
        )
    )
    elements.append(
        Element(
            id=f"RPKG_C{k}", type="R",
            pins=(
                ElementPin(pin=1, node=f"n_c{k}_lp"),
                ElementPin(pin=2, node=f"n_c{k}_rp"),
            ),
            parameters={"r_ohm": float(varactor.rs_ohm)},
        )
    )
    # SDD2P: pin 1 = signal-side (n_c{k}_rp), pin 2 = port-1- net
    # (n_c{k}_v -- the per-varactor measurement node), pins 3/4 tied to
    # a shared per-cell return net for the KCL closure mirror current.
    elements.append(
        Element(
            id=f"VARACTOR_C{k}", type="SDD2P",
            pins=(
                ElementPin(pin=1, node=f"n_c{k}_rp"),
                ElementPin(pin=2, node=f"n_c{k}_v"),
                ElementPin(pin=3, node=f"n_c{k}_ret"),
                ElementPin(pin=4, node=f"n_c{k}_ret"),
            ),
            parameters={
                "q_poly_coefficients": tuple(q_poly),
                "q_poly_reference": q_poly_reference,
            },
        )
    )
    # CP parasitic capacitance is folded into the SDD2P polynomial
    # (``q_poly_coefficients[1] += cp_f`` in :func:`_q_poly_from_varactor`)
    # rather than emitted as a literal parallel ``C`` element. The
    # mathematical equivalence is direct -- ``Q_total = Q_junction + CP·v``
    # produces the same dQ/dt at every voltage as a literal parallel
    # ``CP`` capacitor -- and the fold sidesteps the v2 OA layout
    # engine's single-chain shunt-branch constraint that a literal
    # parallel element would violate (the layout tracer rejects the
    # fork at ``n_c{k}_rp`` between SDD2P and a parallel C). See
    # :func:`_q_poly_from_varactor` for the equivalence proof.
    # IPR_VAR_C{k}: varactor current probe, pin 1 on n_c{k}_v (SDD2P
    # pin 2 net), pin 2 enters the stub.
    elements.append(
        Element(
            id=f"IPR_VAR_C{k}", type="I_Probe",
            pins=(
                ElementPin(pin=1, node=f"n_c{k}_v"),
                ElementPin(pin=2, node=f"n_c{k}_s0"),
            ),
            parameters={},
        )
    )
    # Stub MLIN cascade.
    n_segments = len(cell.stub_segments)
    if n_segments == 0:
        raise ValueError(
            f"Level-10 cell {k} has no stub segments; at least one segment "
            f"is required between the varactor and the via-to-ground."
        )
    for i, seg in enumerate(cell.stub_segments):
        seg_in = f"n_c{k}_s{i}"
        seg_out = (
            f"n_c{k}_via_top" if i == n_segments - 1
            else f"n_c{k}_s{i + 1}"
        )
        elements.append(
            Element(
                id=f"STUB_C{k}_{i}", type="MLIN",
                pins=(
                    ElementPin(pin=1, node=seg_in),
                    ElementPin(pin=2, node=seg_out),
                ),
                parameters={
                    "w_m": float(seg.width_m),
                    "l_m": float(seg.length_m),
                    "substrate_ref": substrate.name,
                },
            )
        )
    # L_via: the parallel-combined Bahl inductance between the stub end
    # and the per-cell ground net.
    elements.append(
        Element(
            id=f"LVIA_C{k}", type="L",
            pins=(
                ElementPin(pin=1, node=f"n_c{k}_via_top"),
                ElementPin(pin=2, node=f"n_c{k}_g"),
            ),
            parameters={"l_h": float(via_inductance_H)},
        )
    )
    return tuple(elements)


def _level10_cell_nodes(
    cell_index: int, n_stub_segments: int
) -> tuple[Node, ...]:
    """Return the per-cell nodes of one Level-10 unit cell."""

    k = cell_index
    nodes: list[Node] = [
        Node(id=f"n_c{k}_l"),       # MTEE pin 1 / LINE_LEFT pin 2
        Node(id=f"n_c{k}_r"),       # MTEE pin 2 / LINE_RIGHT pin 1
        Node(id=f"n_c{k}_out"),     # LINE_RIGHT pin 2 (cell output)
        Node(id=f"n_c{k}_b"),       # MTEE pin 3 (branch tap)
        Node(id=f"n_c{k}_lp"),      # after L_pkg
        Node(id=f"n_c{k}_rp"),      # after R_pkg / SDD2P pin 1
        Node(id=f"n_c{k}_v"),       # SDD2P pin 2 / CP pin 2 / IPR_VAR pin 1
        Node(id=f"n_c{k}_ret"),     # SDD2P pin 3/4 self-tie
    ]
    for i in range(n_stub_segments):
        nodes.append(Node(id=f"n_c{k}_s{i}"))
    nodes.append(Node(id=f"n_c{k}_via_top"))
    nodes.append(Node(id=f"n_c{k}_g"))
    return tuple(nodes)


def make_tvtl_level10_cascade(
    *,
    cells: tuple[Level10CellSpec, ...],
    varactor: Level10VaractorSpec = Level10VaractorSpec(),
    substrate: Level10SubstrateSpec = Level10SubstrateSpec(),
    label: str,
    f_start_Hz: float = 2.0e9,
    f_stop_Hz: float = 17.0e9,
    n_points: int = 151,
    port_impedance_ohm: float = 50.0,
    source_var_names: dict[str, str] | None = None,
    source_var_values: dict[str, float] | None = None,
    hb_order_signal: int = 5,
    hb_order_pump: int = 5,
    hb_max_order: int = 6,
) -> DeviceSpec:
    """Build a Level-10 N-cell TVTL cascade with paper-shape topology.

    Each cell is a through-line split by an MTEE with a shunt branch
    carrying ``L_pkg → R_pkg → SDD2P(Skyworks Taylor Q(v)) || CP →
    IPR_VAR → stub MLIN cascade → L_via → ground``. Boundary I_Probes
    (``IPR_IN``, ``IPR_C{k}C{k+1}``, ``IPR_OUT``) splice between cells
    and at the design boundaries. Two ``P_1Tone`` sources (signal Num=1,
    pump Num=2) drive ``n_input``; the lone output ``TermG`` is Num=3.

    Parameters
    ----------
    cells : tuple of Level10CellSpec
        One entry per cell. The paper has 15 identical cells; passing a
        non-uniform tuple cascades arbitrary cell shapes.
    varactor : Level10VaractorSpec, optional
        Skyworks SMV1245 parameters + package parasitics. Defaults to
        the locked paper values (CJ0=6.9 pF, VJ=3.5 V, M=1.7, CP=0.47 pF,
        RS=2 Ω, L_pkg=1.7 nH, analytical_c0=7.0 pF, Taylor order 5).
    substrate : Level10SubstrateSpec, optional
        Substrate stack-up. Defaults to Rogers 4003C as fabricated.
    label : str
        ``DeviceSpec.label`` for the produced spec.
    f_start_Hz, f_stop_Hz, n_points : float, float, int
        S-parameter sweep grid; default 2-17 GHz, 151 points (Figure 4
        resolution).
    port_impedance_ohm : float
        TermG reference impedance.
    source_var_names : dict, optional
        Variable-name overrides keyed by canonical name
        (``"f_signal"`` / ``"f_pump"`` / ``"signal_power_dBm"`` /
        ``"pump_power_dBm"`` / ``"pump_phase_rad"``). A future
        pump-sweep API supplies its own names so the VAR block is
        replaceable without rebuilding the spec.
    source_var_values : dict, optional
        Numeric defaults for the VAR block. Overrides
        :data:`_DEFAULT_LEVEL10_VAR_VALUES` per-key.
    hb_order_signal, hb_order_pump, hb_max_order : int
        Two-tone HB truncation orders. Defaults match the Level-9
        canonical (5, 5, 6).
    """

    if not cells:
        raise ValueError("Level-10 cascade requires at least one cell.")

    substrate_obj = substrate.to_substrate()
    q_poly = _q_poly_from_varactor(varactor)
    q_poly_reference = _q_poly_reference_for_varactor(varactor)

    # Resolve VAR-block variable names + numeric defaults; the
    # canonical-name → spec-name remap lets a sweep consumer swap in a
    # custom VAR namespace.
    var_names: dict[str, str] = dict(_DEFAULT_LEVEL10_VAR_NAMES)
    if source_var_names is not None:
        for canonical, spec_name in source_var_names.items():
            if canonical not in var_names:
                raise ValueError(
                    f"unknown source variable {canonical!r}; accepted: "
                    f"{sorted(var_names)}."
                )
            var_names[canonical] = str(spec_name)
    var_values: dict[str, float] = dict(_DEFAULT_LEVEL10_VAR_VALUES)
    if source_var_values is not None:
        for canonical, value in source_var_values.items():
            if canonical not in var_values:
                raise ValueError(
                    f"unknown source variable {canonical!r}; accepted: "
                    f"{sorted(var_values)}."
                )
            var_values[canonical] = float(value)

    nodes: list[Node] = [Node(id="n_input"), Node(id="n_pin")]
    elements: list[Element] = []
    ground_nodes: list[str] = []

    elements.append(
        Element(
            id="IPR_IN", type="I_Probe",
            pins=(
                ElementPin(pin=1, node="n_input"),
                ElementPin(pin=2, node="n_pin"),
            ),
            parameters={},
        )
    )

    n_cells = len(cells)
    for k in range(1, n_cells + 1):
        cell_spec = cells[k - 1]
        cell_in = "n_pin" if k == 1 else f"n_pc{k - 1}c{k}"
        cell_out = f"n_c{k}_out"
        l_via_H = via_inductance_bahl_H(
            substrate_thickness_m=float(substrate.h_m),
            via_radius_m=float(cell_spec.via.radius_m),
            n_parallel=int(cell_spec.via.n_parallel),
        )
        nodes.extend(
            _level10_cell_nodes(k, len(cell_spec.stub_segments))
        )
        elements.extend(
            _level10_cell_elements(
                cell_index=k,
                input_net=cell_in,
                output_net=cell_out,
                cell=cell_spec,
                varactor=varactor,
                substrate=substrate_obj,
                via_inductance_H=l_via_H,
                q_poly=q_poly,
                q_poly_reference=q_poly_reference,
            )
        )
        ground_nodes.append(f"n_c{k}_g")
        if k < n_cells:
            inter_net = f"n_pc{k}c{k + 1}"
            nodes.append(Node(id=inter_net))
            elements.append(
                Element(
                    id=f"IPR_C{k}C{k + 1}", type="I_Probe",
                    pins=(
                        ElementPin(pin=1, node=cell_out),
                        ElementPin(pin=2, node=inter_net),
                    ),
                    parameters={},
                )
            )

    # Output boundary I_Probe + port.
    nodes.append(Node(id="n_output"))
    last_cell_out = f"n_c{n_cells}_out"
    elements.append(
        Element(
            id="IPR_OUT", type="I_Probe",
            pins=(
                ElementPin(pin=1, node=last_cell_out),
                ElementPin(pin=2, node="n_output"),
            ),
            parameters={},
        )
    )
    elements.append(
        Element(
            id="P_OUT", type="TermG",
            pins=(ElementPin(pin=1, node="n_output"),),
            parameters={"z_ohm": float(port_impedance_ohm), "num": 3},
        )
    )

    # NAME annotations (minimum-needed for the Floquet comparison +
    # energy-conservation harnesses):
    #
    # - n_input / n_output: cascade boundary observables for the dataset's
    #   port-level response.
    # - n_c{k}_b: MTEE pin 3 (the branch top). V_0 for the analytical
    #   Floquet solver is read here -- the input to the "local RLC"
    #   (R_pkg + L_pkg + time-varying C, per Q5's lock). Level-9
    #   collapses n_c{k}_b == SDD2P pin 1 because Level-9 has no
    #   L_pkg/R_pkg in the branch chain, so the V at SDD2P pin 1
    #   doubles as V_0; Level-10 separates the two and V_0 must come
    #   from n_c{k}_b.
    # - n_c{k}_v: SDD2P pin 2 (the time-varying capacitor's other
    #   terminal). The energy-conservation cross-check reads
    #   the per-varactor V here.
    #
    # Total NAMEs: 2 + 2*N. For the paper's N=15 that is 32; deliberately
    # leaner than Level-9's 4*N+2 = 14 (for N=3) by skipping the per-cell
    # inter-cell boundary nets and the per-cell output net (no
    # per-cell power calculation runs on Level-10 yet -- the
    # publication's Figure 4 reproduction needs only the boundary
    # observables plus the per-varactor V_0 and V_pump).
    name_counter = 0

    def _add_name(net: str) -> None:
        nonlocal name_counter
        elements.append(
            Element(
                id=f"NAME_{name_counter:02d}_{net}",
                type="NAME",
                pins=(ElementPin(pin=1, node=net),),
                parameters={},
            )
        )
        name_counter += 1

    _add_name("n_input")
    for k in range(1, n_cells + 1):
        _add_name(f"n_c{k}_b")
        _add_name(f"n_c{k}_v")
    _add_name("n_output")

    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(
                name=var_names["f_signal"], value=float(var_values["f_signal"]),
            ),
            VarEntry(
                name=var_names["f_pump"], value=float(var_values["f_pump"]),
            ),
            VarEntry(
                name=var_names["signal_power_dBm"],
                value=float(var_values["signal_power_dBm"]),
            ),
            VarEntry(
                name=var_names["pump_power_dBm"],
                value=float(var_values["pump_power_dBm"]),
            ),
            VarEntry(
                name=var_names["pump_phase_rad"],
                value=float(var_values["pump_phase_rad"]),
            ),
        ),
    )
    sources = (
        P1ToneSource(
            name="SIGNAL_SRC", port_node="n_input", num=1,
            freq=var_names["f_signal"],
            power_dBm=var_names["signal_power_dBm"],
        ),
        P1ToneSource(
            name="PUMP_SRC", port_node="n_input", num=2,
            freq=var_names["f_pump"],
            power_dBm=var_names["pump_power_dBm"],
        ),
    )
    hb = HBTwoToneSpec(
        fundamental_freq_1=var_names["f_signal"],
        fundamental_order_1=int(hb_order_signal),
        fundamental_freq_2=var_names["f_pump"],
        fundamental_order_2=int(hb_order_pump),
        max_order=int(hb_max_order),
    )
    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz),
            points=int(n_points),
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
        label=label,
        ground_nodes=tuple(ground_nodes),
        var_block=var_block,
        hb_two_tone=hb,
        sources=sources,
    )


# ===========================================================================
# Paper-fidelity Level-10 instance
# ===========================================================================
# These are the locked paper parameters per Figure 3b and the SI
# Methodology. The factory exists so a caller (a test, a CLI, a report
# script) can grab the paper-fidelity DeviceSpec in one call without
# repeating the parameter list.
# ===========================================================================

_PAPER_GEOMETRY_CELL = Level10CellSpec(
    main_width_m=4.27e-3,
    main_period_m=6.11e-3,
    branch_width_m=1.93e-3,
    stub_segments=(
        Level10StubSegmentSpec(width_m=1.93e-3, length_m=6.83e-3),
        Level10StubSegmentSpec(width_m=1.41e-3, length_m=5.25e-3),
        Level10StubSegmentSpec(width_m=1.98e-3, length_m=1.04e-3),
    ),
    via=Level10ViaSpec(n_parallel=2, radius_m=0.8e-3),
)


def make_paper_geometry_tvtl_cascade(
    *,
    n_cells: int = 15,
    label: str = "level10_paper_geometry_tvtl_cascade",
) -> DeviceSpec:
    """Return the paper-fidelity Level-10 DeviceSpec.

    The cascade has ``n_cells`` (default 15, the paper's fabricated
    cell count) of one identical cell template
    (:data:`_PAPER_GEOMETRY_CELL`) on Rogers 4003C with the SMV1245
    Skyworks SPICE Taylor-expansion varactor. The substrate, varactor,
    main microstrip dimensions, stub segment dimensions, and via
    configuration are all locked to the paper's published values
    (Figure 3b in the main text; the SI Methodology Tables 1-2).

    Parameters
    ----------
    n_cells : int, optional
        Cell count. Default 15 matches the fabricated device; passing a
        smaller integer (2-3) produces a paper-shape mini-cascade for
        regression testing on smaller cascades.
    label : str, optional
        ``DeviceSpec.label`` override; default
        ``"level10_paper_geometry_tvtl_cascade"``.

    References
    ----------
    Abzaliyeva & Orazbayev, "Physics-Guided Design of Varactor-Loaded
    Time-Varying Transmission Lines," Scientific Reports. Figure 3b and
    SI Methodology.
    """

    if int(n_cells) < 1:
        raise ValueError(f"n_cells must be >= 1 (got {n_cells}).")
    return make_tvtl_level10_cascade(
        cells=tuple([_PAPER_GEOMETRY_CELL] * int(n_cells)),
        varactor=Level10VaractorSpec(),         # locked Skyworks defaults
        substrate=Level10SubstrateSpec(),       # locked Rogers 4003C defaults
        label=str(label),
    )


LEVEL_10_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    make_paper_geometry_tvtl_cascade(
        n_cells=15,
        label="level10_paper_geometry_tvtl_cascade",
    ),
)


def make_paper_geometry_sparam_only(
    *,
    n_cells: int = 15,
    label: str = "level10_paper_geometry_sparam_only",
    f_start_Hz: float = 2.0e9,
    f_stop_Hz: float = 17.0e9,
    n_points: int = 151,
) -> DeviceSpec:
    """Passive S-parameter-only variant of the Level-10 paper canonical.

    The HB-driven Level-10 canonical (:func:`make_paper_geometry_tvtl_cascade`)
    carries two ``P_1Tone`` sources at ``n_input`` plus a two-tone HB
    controller. That configuration is the right shape for Floquet /
    per-cell cross-checks but fails the hpeesofsim S_Param leg: the
    S_Param controller does not recognise ``P_1Tone`` (only the HB
    controller does), so an S-parameter sweep over the HB-driven
    canonical aborts with the documented ``P_1Tone' is an instance of
    an undefined model`` error.

    This factory builds the same Level-10 paper geometry but replaces
    the input ``P_1Tone`` sources with a 50 Ω ``TermG`` port (Num=1)
    and drops the HB controller, so the resulting spec is a pure
    passive 2-port suitable for an S-parameter sweep. The SDD2P
    elements remain; ADS's S_Param controller small-signal-linearises
    them around the DC operating point ``v = 0``, presenting an
    effective static capacitance equal to the SDD2P polynomial's
    ``q[1]`` coefficient (= ``analytical_c0_f + cp_f`` with the
    CP-fold). The analytical Path-B :func:`level10_paper_sparams`
    uses the same effective C so the two paths are apples-to-apples.

    The S-parameter sweep grid defaults to 2 GHz → 17 GHz × 151 points,
    matching the publication's Figure 4 resolution.

    Parameters mirror :func:`make_paper_geometry_tvtl_cascade` except
    for the analysis grid (which is the primary use case for this
    factory).
    """

    if int(n_cells) < 1:
        raise ValueError(f"n_cells must be >= 1 (got {n_cells}).")

    # Build the HB-driven spec and rewrite it: drop sources / HB / VAR,
    # add a TermG at n_input, retarget the existing output TermG to
    # Num=2.
    hb_spec = make_paper_geometry_tvtl_cascade(
        n_cells=int(n_cells),
        label=str(label),
    )

    # Filter out NAME elements that reference nets we keep; replace the
    # output TermG (Num=3) with a Num=2 port and add a Num=1 TermG at
    # n_input. Drop both P_1Tone sources and the HB controller.
    new_elements: list[Element] = []
    for element in hb_spec.elements:
        if element.id == "P_OUT":
            new_elements.append(
                Element(
                    id="P_OUT",
                    type="TermG",
                    pins=element.pins,
                    parameters={
                        "z_ohm": float(element.parameters["z_ohm"]),
                        "num": 2,
                    },
                )
            )
        else:
            new_elements.append(element)
    # Insert the input TermG at n_input -- ADS expects a port at the
    # source side for S-parameter analysis.
    new_elements.insert(
        0,
        Element(
            id="P_IN",
            type="TermG",
            pins=(ElementPin(pin=1, node="n_input"),),
            parameters={
                "z_ohm": float(hb_spec.analysis.port_impedance_ohm),
                "num": 1,
            },
        ),
    )

    new_analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz),
            stop=float(f_stop_Hz),
            points=int(n_points),
        ),
        port_impedance_ohm=float(hb_spec.analysis.port_impedance_ohm),
    )

    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=new_analysis,
        substrate=hb_spec.substrate,
        nodes=hb_spec.nodes,
        elements=tuple(new_elements),
        ports=(
            Port(num=1, node="n_input"),
            Port(num=2, node="n_output"),
        ),
        label=str(label),
        ground_nodes=hb_spec.ground_nodes,
        # No sources, no HB, no VAR -- pure passive S-parameter spec.
    )


