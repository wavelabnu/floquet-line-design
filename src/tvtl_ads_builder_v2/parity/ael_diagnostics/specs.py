"""Device specs for the AEL diagnostic ladder D0-D4.

Each diagnostic isolates exactly one wiring pattern beyond the
previous one:

* **D0** -- bare ports. The TVTL analytical simulator and hpeesofsim
  both require at least one element, so a literally element-free
  port-to-port spec has no Path-1 reference. D0 is therefore realised
  as a *near-zero-length* (0.1 mm) MLIN: electrically a direct through
  (|S21| ~ 0 dB), it is the minimal port<->element<->port circuit.
  (A documented limitation of the D0 near-zero-length stand-in.)
* **D1** -- a single 50-ohm MLIN between two ports.
* **D2** -- two MLINs cascaded (tests element-to-element wiring).
* **D3** -- an MTEE wired directly between the two ports with a
  grounded shunt stub (tests MTEE pins 1/2/3 and stub-to-ground).
* **D4** -- the full Level-3 topology: MLIN + MTEE + MLIN with a
  grounded shunt stub.

D1, D2 and D4 reuse the main-ladder spec constructors verbatim (with
diagnostic labels); D0 and D3 are degenerate / non-standard topologies
the constructors do not cover and are built here directly. All five
share one substrate and a fixed 1-10 GHz sweep.
"""

from __future__ import annotations

from typing import Callable

from ...device_spec import (
    Analysis,
    DeviceSpec,
    Element,
    ElementPin,
    FrequencyGrid,
    Node,
    Port,
)
from ...device_spec.schema import DEVICE_SPEC_VERSION
from ..canonical_circuits import (
    _LEVEL_1_SUBSTRATE,
    make_termg_mlin_termg,
    make_tee_with_short_stub_paper_dims,
    make_two_mlins_series,
)

# ~50-ohm microstrip width on the Rogers-4003C-like substrate, and a
# trivial body length. The diagnostics validate wiring, not parameter
# values, so geometry is kept fixed and simple.
_WIDTH_50OHM_M = 2.95e-3
_BODY_LENGTH_M = 2.5e-3
_STUB_WIDTH_M = 1.93e-3
_STUB_LENGTH_M = 6.83e-3
_D0_LENGTH_M = 0.1e-3  # near-zero through-line stand-in for "bare ports"


def _analysis() -> Analysis:
    """The shared 1-10 GHz S-parameter sweep, 50-ohm ports."""

    return Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(start=1.0e9, stop=10.0e9, points=91),
        port_impedance_ohm=50.0,
    )


def make_d0_spec() -> DeviceSpec:
    """D0 -- bare ports, realised as a near-zero-length MLIN.

    A genuinely element-free port-to-port spec cannot be analysed (the
    TVTL simulator and hpeesofsim both need an element). The 0.1 mm
    MLIN is electrically a direct wire and gives D0 a valid Path-1
    reference.
    """

    return make_termg_mlin_termg(
        w_m=_WIDTH_50OHM_M, l_m=_D0_LENGTH_M, label="d0_bare_ports"
    )


def make_d1_spec() -> DeviceSpec:
    """D1 -- a single 50-ohm MLIN between two ports."""

    return make_termg_mlin_termg(
        w_m=_WIDTH_50OHM_M, l_m=_BODY_LENGTH_M, label="d1_single_mlin"
    )


def make_d2_spec() -> DeviceSpec:
    """D2 -- two MLINs cascaded through a shared internal node."""

    return make_two_mlins_series(
        w1_m=_WIDTH_50OHM_M,
        l1_m=_BODY_LENGTH_M,
        w2_m=_WIDTH_50OHM_M,
        l2_m=_BODY_LENGTH_M,
        label="d2_two_mlins",
    )


def make_d3_spec() -> DeviceSpec:
    """D3 -- an MTEE wired straight between the ports + grounded stub.

    No main-line MLINs: the MTEE's through pins (1, 2) connect directly
    to the ports, and pin 3 feeds a grounded shunt stub. This isolates
    the MTEE pin wiring and the stub-to-ground connection.
    """

    nodes = (
        Node(id="n_in"),
        Node(id="n_out"),
        Node(id="n_branch_top"),
        Node(id="n_branch_end"),
    )
    p1 = Element(
        id="P1",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_in"),),
        parameters={"z_ohm": 50.0, "num": 1},
    )
    p2 = Element(
        id="P2",
        type="TermG",
        pins=(ElementPin(pin=1, node="n_out"),),
        parameters={"z_ohm": 50.0, "num": 2},
    )
    tee = Element(
        id="TEE1",
        type="MTEE",
        pins=(
            ElementPin(pin=1, node="n_in"),
            ElementPin(pin=2, node="n_out"),
            ElementPin(pin=3, node="n_branch_top"),
        ),
        parameters={
            "w1_m": _WIDTH_50OHM_M,
            "w2_m": _WIDTH_50OHM_M,
            "w3_m": _STUB_WIDTH_M,
            "substrate_ref": _LEVEL_1_SUBSTRATE.name,
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
            "w_m": _STUB_WIDTH_M,
            "l_m": _STUB_LENGTH_M,
            "substrate_ref": _LEVEL_1_SUBSTRATE.name,
        },
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=_analysis(),
        substrate=_LEVEL_1_SUBSTRATE,
        nodes=nodes,
        elements=(p1, tee, stub, p2),
        ports=(Port(num=1, node="n_in"), Port(num=2, node="n_out")),
        label="d3_mtee_short_stub",
        ground_nodes=("n_branch_end",),
    )


def make_d4_spec() -> DeviceSpec:
    """D4 -- the full Level-3 topology (MLIN + MTEE + MLIN + grounded stub)."""

    return make_tee_with_short_stub_paper_dims(label="d4_full_level3")


# Diagnostic ladder, lowest pattern first.
DIAGNOSTIC_ORDER: tuple[str, ...] = (
    "d0_bare_ports",
    "d1_single_mlin",
    "d2_two_mlins",
    "d3_mtee_short_stub",
    "d4_full_level3",
)

DIAGNOSTIC_SPECS: dict[str, Callable[[], DeviceSpec]] = {
    "d0_bare_ports": make_d0_spec,
    "d1_single_mlin": make_d1_spec,
    "d2_two_mlins": make_d2_spec,
    "d3_mtee_short_stub": make_d3_spec,
    "d4_full_level3": make_d4_spec,
}


