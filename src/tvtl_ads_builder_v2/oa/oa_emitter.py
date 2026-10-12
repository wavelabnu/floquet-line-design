"""Spec-driven OA-Python emitter for the D0-D4 diagnostic ladder.

:func:`emit_diagnostic_workspace` builds an ADS workspace for any
diagnostic D0-D4 by walking its :class:`DeviceSpec`:

* the ``MSUB`` substrate from ``spec.substrate``;
* one OA instance per ``spec.element`` (``TermG`` / ``MLIN`` / ``MTEE``
  / the lumped ``R`` / ``L`` / ``C`` / the ``SDD2P`` symbolically-defined
  device; an ``RLC_SERIES`` expands to three primitive R/L/C instances);
* a ``GROUND`` symbol per ``spec.ground_nodes`` entry;
* an ``S_Param`` controller from ``spec.analysis``;
* connectivity *by node* -- every pin on a device-spec node is put on
  one net (``pin.net = net``) with a reinforcing ``add_wire``.

It then calls ``design.generate_netlist()`` (v1's
``hb_workspace_worker.py:664`` pattern) to write the ADS netlist, and
emits the hpeesofsim flat deck -- so the netlist cross-check needs no
manual ADS step.

This is the spec-driven generalization of the earlier single-design
``d1_emitter``, scoped to the diagnostic suite; the
full canonical ladder is wired through :mod:`.oa_canonical_cli`.

``keysight.ads.de`` is imported inside :func:`emit_diagnostic_workspace`,
so importing this module never requires ADS.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from ..audit import audit_circuit
from ..device_spec import (
    DeviceSpec,
    Element,
    HBSingleToneSpec,
    HBTwoToneSpec,
    P1ToneSource,
    resolve_spec,
)
from ..layout import CircuitLayout, build_layout
from ..netlist.elements.hb_controller import HB_CONTROLLER_NAME, HB_STATUS_LEVEL
from ..netlist.elements.i_probe import I_PROBE_MODE, I_PROBE_SAVE_CURRENT
from ..netlist.elements.p1tone import (
    P1TONE_NOISE,
    format_p1tone_freq,
    format_p1tone_power,
    format_p1tone_z,
)
from ..netlist.elements.rlc import (
    expand_rlc_series,
    format_c_value,
    format_l_value,
    format_r_value,
)
from ..netlist.elements.sdd2p import sdd2p_body_rows
from ..netlist.emit import emit_hpeesofsim_netlist
from ..parity.ael_diagnostics.specs import DIAGNOSTIC_SPECS
from ..topology import render_topology_svg

_LOG = logging.getLogger(__name__)

# Device-spec element type -> ADS OpenAccess (library, cell, view).
# RLC_SERIES has no entry: it is expanded to R/L/C before placement.
_OA_CELL = {
    "TermG": ("ads_simulation", "TermG", "symbol"),
    "MLIN": ("ads_tlines", "MLIN", "symbol"),
    "MTEE": ("ads_tlines", "MTEE_ADS", "symbol"),
    "R": ("ads_rflib", "R", "symbol"),
    "L": ("ads_rflib", "L", "symbol"),
    "C": ("ads_rflib", "C", "symbol"),
    "SDD2P": ("ads_datacmps", "SDD2P", "symbol"),
    # I_Probe is the native ADS current probe. A live OA
    # probe found it resolves from ads_rflib (NOT ads_simulation /
    # ads_sources) and netlists as a "Short" element with Mode=0.
    "I_Probe": ("ads_rflib", "I_Probe", "symbol"),
    # Emitter-accuracy additions: the distributed termination elements the
    # reference ADS design uses (its circuit2 cell), so a design that opts
    # into the chain model registry's VIAGND via / MLEF (or MLOC) open-end model
    # renders GUI-openable. Library:cell paths are from
    # inspection.element_library_mapping (ads_tlines). Each is a 1-port (a single
    # device-spec pin) -- VIAGND grounds internally, MLEF/MLOC are open-end stubs.
    # Inert for the byte-identical anchor (its spec carries none of these).
    "VIAGND": ("ads_tlines", "VIAGND", "symbol"),
    "MLEF": ("ads_tlines", "MLEF", "symbol"),
    "MLOC": ("ads_tlines", "MLOC", "symbol"),
}
_MSUB_CELL = ("ads_tlines", "MSUB", "symbol")
_GROUND_CELL = ("ads_rflib", "GROUND", "symbol")
_SPARAM_CELL = ("ads_simulation", "S_Param", "symbol")
_HB_CELL = ("ads_simulation", "HarmonicBalance", "symbol")
# ADS P_1Tone single-tone power source -- v1's _BUILTIN_TO_OA registers
# this exact (library, cell, view) tuple (hb_workspace_worker.py:47).
_P1TONE_CELL = ("ads_sources", "P_1Tone", "symbol")

# Device-spec pin number -> OA inst_pins index. Most symbols -- TermG,
# MLIN and the lumped R / L / C -- expose their pins in spec-pin order,
# so the index is just (pin - 1) and they need no entry here. (A
# two-terminal element is doubly safe: the netlist comparator treats
# its terminal order as symmetric.)
#
# MTEE is one exception. The ADS MTEE convention is pin 1 = input,
# pin 2 = output, pin 3 = branch; the hpeesofsim emitter
# (netlist/elements/mtee.py) is the validated reference and emits the
# three terminals positionally as pin1.node, pin2.node, pin3.node.
# Comparing the OA-generated and hpeesofsim D3/D4 netlists shows ADS
# lists an MTEE_ADS instance's terminals as [inst_pins[0],
# inst_pins[2], inst_pins[1]] -- so to land the device-spec MTEE pins
# in the hpeesofsim order, spec pin 2 goes to inst_pins index 2 and
# spec pin 3 to index 1.
#
# SDD2P is the other exception. It is a 2-port (4 terminals); ADS lists
# the four terminals in inst_pins order [0, 3, 2, 1]. So device-spec SDD2P pin 1
# -> inst_pins[0], pin 2 -> inst_pins[3], pin 3 -> inst_pins[2], pin 4
# -> inst_pins[1], which lands the device pins in the hpeesofsim
# positional order pin1/pin2/pin3/pin4.
_PIN_INDEX: dict[str, dict[int, int]] = {
    "MTEE": {1: 0, 2: 2, 3: 1},
    "SDD2P": {1: 0, 2: 3, 3: 2, 4: 1},
}

# --- explicit component placement ------------------------------------
# ADS 2026 keeps two connectivity layers: the database-level net
# assignments this emitter writes (always correct), and a geometry-level
# layer ADS recomputes from pin coordinates. Components left at
# near-identical coordinates overlap visually, and a manual rearrange in
# the GUI can then rebind pins off the overlap -- a "correct but
# uninspectable" workspace. Every component is therefore placed at an
# explicit position taken from the shared layout engine
# (``tvtl_ads_builder_v2.layout``), so the OA schematic and the topology
# SVG agree on where each component sits.
#
# ``compute_layout`` returns dimensionless *layout units*: x is one unit
# per through-path column, y is one unit per shunt row and increases
# downward. These constants map a layout coordinate to an ADS schematic
# coordinate::
#
#     ads_x = ADS_HORIZONTAL_OFFSET + x_layout * ADS_UNIT_PER_LAYOUT_UNIT
#     ads_y = ADS_VERTICAL_OFFSET   - y_layout * ADS_UNIT_PER_LAYOUT_UNIT
#
# The y term is subtracted because layout y grows downward while the ADS
# schematic y axis grows upward -- so a shunt branch hangs at negative
# ADS y, below the through-path row (where ADS also expects the GROUND
# symbols).
ADS_UNIT_PER_LAYOUT_UNIT = 5.0
ADS_HORIZONTAL_OFFSET = 0.0
ADS_VERTICAL_OFFSET = 0.0

# RLC_SERIES has no single OA symbol; its three expanded R/L/C
# primitives share one layout slot and are spread by this much (ADS
# units) so they neither overlap each other nor a neighbouring column.
_RLC_CLUSTER_DX = 1.4

# A GROUND symbol sits this many ADS units below its shunt branch's
# exit pin, so it reads as the foot of the shunt branch.
_GROUND_DROP = 3.0

# --- orthogonal shunt routing ----------------------------------------
# ADS schematic-symbol pin offsets. Each maps a device-spec pin number to the
# (dx, dy) offset of that pin from the instance's add_instance origin.
#
# A horizontal MLIN / MTEE / R / L / C symbol places its pins away from
# the origin, so two components placed at the same origin x do NOT have
# their connecting pins at the same x. That is the cause of the diagonal
# shunt-branch wires (see the routing-diagnosis
# probe). The table is keyed by device-spec pin number -- the _PIN_INDEX
# MTEE remapping is already folded in -- so it reads straight off
# ``element.pins``.
# SDD2P offsets are keyed by device-spec pin -- the _PIN_INDEX SDD2P
# remapping {1:0, 2:3, 3:2, 4:1} is already folded in, so device pin 1
# carries inst_pins[0]'s measured offset, pin 2 carries inst_pins[3]'s,
# and so on.
_PIN_OFFSETS: dict[str, dict[int, tuple[float, float]]] = {
    "TermG": {1: (0.0, 0.0)},
    "MLIN": {1: (1.0, 0.0), 2: (0.0, 0.0)},
    "MTEE": {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.5, -0.5)},
    "R": {1: (0.0, 0.0), 2: (1.0, 0.0)},
    "L": {1: (1.0, 0.0), 2: (0.0, 0.0)},
    "C": {1: (0.0, 0.0), 2: (1.0, 0.0)},
    "SDD2P": {
        1: (0.0, 0.0),
        2: (0.0, -0.25),
        3: (1.0, 0.0),
        4: (1.0, -0.25),
    },
    # I_Probe -- a live pin probe measured the
    # two pins at (0.0, 0.0) and (0.25, 0.0). The probe also found the
    # inst_pins positional order is REVERSED vs the ADS logical pin
    # order: inst_pins[0] is logical pin .2 (offset 0.25,0), inst_pins[1]
    # is logical pin .1 (offset 0,0). This table is keyed by device-spec
    # pin number, and device-spec pin N maps to ADS logical pin .N, so
    # device pin 1 carries the (0,0) offset and pin 2 the (0.25,0).
    "I_Probe": {1: (0.0, 0.0), 2: (0.25, 0.0)},
    # 1-port termination elements (emitter-accuracy additions): a single
    # pin at the symbol origin -- VIAGND grounds internally, MLEF/MLOC are
    # open-end stubs. The shunt-branch aligner places them by this pin.
    "VIAGND": {1: (0.0, 0.0)},
    "MLEF": {1: (0.0, 0.0)},
    "MLOC": {1: (0.0, 0.0)},
}

# Element types whose inst_pins are resolved by inst_pin_id, NOT by the
# positional inst_pins[N] index. A live probe confirmed an
# I_Probe's inst_pins array is in reversed order vs its logical pin
# numbering, so the connectivity pass must look pin .N up by id rather than assume inst_pins[0]
# is device pin 1.
_ID_RESOLVED_PIN_TYPES = frozenset({"I_Probe"})


def _pin_offset(element_type: str, device_pin: int) -> tuple[float, float]:
    """(dx, dy) of a device-spec pin from its instance's add_instance origin."""

    return _PIN_OFFSETS[element_type][device_pin]


def _ensure_keysight_on_path() -> None:
    """Add ``<HPEESOF_DIR>/tools/python/packages`` to ``sys.path``."""

    hpeesof = os.environ.get("HPEESOF_DIR")
    if not hpeesof:
        return
    packages = Path(hpeesof) / "tools" / "python" / "packages"
    if packages.is_dir() and str(packages) not in sys.path:
        sys.path.insert(0, str(packages))


def _mm(value_m) -> str:
    """Format an SI-metres length as an ADS ``<value> mm`` string."""

    return f"{float(value_m) * 1.0e3:.12g} mm"


def _g(value) -> str:
    return f"{float(value):.12g}"


def _msub_params(substrate) -> dict[str, str]:
    """ADS MSUB parameters from a device-spec substrate.

    Mirrors :func:`netlist.elements.emit_msub` exactly so the OA MSUB
    and the hpeesofsim MSUB describe the same substrate.
    """

    p = substrate.parameters
    return {
        "H": _mm(p["h_m"]),
        "Er": _g(p["eps_r"]),
        "Mur": _g(p.get("mur", 1.0)),
        "Cond": _g(p["sigma_S_per_m"]),
        "T": _mm(p["t_m"]),
        "TanD": _g(p["tan_delta"]),
        "Rough": _mm(p.get("rough_m", 0.0)),
    }


def _element_params(element, substrate_name: str) -> dict[str, str]:
    """ADS instance parameters for one device-spec element."""

    p = element.parameters
    if element.type == "TermG":
        return {
            "Num": str(int(p["num"])),
            "Z": f'{_g(p["z_ohm"])} Ohm',
            "Noise": "no",
        }
    if element.type == "MLIN":
        return {
            "Subst": f'"{substrate_name}"',
            "W": _mm(p["w_m"]),
            "L": _mm(p["l_m"]),
        }
    if element.type == "MTEE":
        return {
            "Subst": f'"{substrate_name}"',
            "W1": _mm(p["w1_m"]),
            "W2": _mm(p["w2_m"]),
            "W3": _mm(p["w3_m"]),
        }
    if element.type == "R":
        return {"R": format_r_value(element)}
    if element.type == "L":
        return {"L": format_l_value(element)}
    if element.type == "C":
        return {"C": format_c_value(element)}
    if element.type == "I_Probe":
        # The native I_Probe cell is a multi-mode "Short": Mode=0 selects
        # the ideal-short / current-probe behaviour and SaveCurrent=yes
        # records its current. Both are emitter constants (the spec-level
        # I_Probe is parameterless) and match netlist/elements/i_probe.py
        # so the OA and hpeesofsim "Short" lines agree.
        return {
            "Mode": I_PROBE_MODE,
            "SaveCurrent": I_PROBE_SAVE_CURRENT,
        }
    if element.type == "VIAGND":
        # Physical via-to-ground (1-port, grounds internally). Mirrors
        # netlist/elements/viagnd.py: Subst / D / [W] / T / [Rho]. W and Rho
        # are emitted only when the model supplied them (it degenerates to the
        # required D / T form otherwise). A parameter the OA symbol does not
        # expose is skipped by setp (a no-op), so emitting all known names is
        # safe.
        params = {
            "Subst": f'"{substrate_name}"',
            "D": _mm(p["d_m"]),
            "T": _mm(p["t_m"]),
        }
        if p.get("w_m") is not None:
            params["W"] = _mm(p["w_m"])
        if p.get("rho") is not None:
            params["Rho"] = _g(p["rho"])
        return params
    if element.type in ("MLEF", "MLOC"):
        # Open-end-effect microstrip stub (MLEF, with fringe) / open-circuited
        # stub (MLOC, ideal open), both 1-port. Mirrors netlist/elements/
        # {mlef,mloc}.py: Subst / W / L.
        return {
            "Subst": f'"{substrate_name}"',
            "W": _mm(p["w_m"]),
            "L": _mm(p["l_m"]),
        }
    # SDD2P has no scalar instance parameters; its body equations are a
    # repeated parameter set by _set_sdd2p_body in a separate step.
    return {}


def _set_sdd2p_body(inst, element: Element) -> None:
    """Set an SDD2P instance's body equations from its device spec.

    The SDD body is a single *repeated* parameter named ``"SDD"`` whose
    value is a list of ``[port, weight, expression]`` rows; the ADS
    netlister emits each row as ``I[port,weight]=expression``. Port 1
    carries the charge-driven varactor (``I[1,1]=+Q(_v1)``, weight 1 ->
    ``dQ/dt``); port 2 is the KCL-closure mirror (``I[2,1]=-Q(_v1)``).
    The rows come from the shared :func:`sdd2p_body_rows`, so the OA
    instance carries the byte-identical body the hpeesofsim emitter
    writes.

    Set defensively: like :func:`setp`, if the instance has no ``"SDD"``
    parameter (the ADS-free test stub) this is a no-op.
    """

    if "SDD" in inst.parameters:
        inst.parameters["SDD"].value = sdd2p_body_rows(element)


def _set_hb_controller(inst, hb: HBSingleToneSpec) -> None:
    """Set a HarmonicBalance controller instance from a single-tone spec.

    Mirrors v1's ``hb_workspace_worker.py::_add_hb_controller``: ``Freq``
    and ``Order`` are array-valued, so assigning a one-element list
    populates index 1 (the single tone); ``MaxOrder`` and
    ``StatusLevel`` are scalars. ``SweepPlan`` is forced to a non-empty
    placeholder -- the OA HarmonicBalance cell defaults it empty and a
    bare ``SweepPlan=`` is a netlist parser error; v1 documents and
    applies the same workaround in ``_add_hb_controller``.

    Set defensively: like :func:`setp`, a parameter absent from the
    instance (the ADS-free test stub) is skipped.
    """

    freq_GHz = float(hb.fundamental_freq) * 1.0e-9
    values: dict[str, object] = {
        "Freq": [f"{freq_GHz:.12g} GHz"],
        "Order": [str(int(hb.fundamental_order))],
        "MaxOrder": str(int(hb.max_order)),
        "StatusLevel": str(HB_STATUS_LEVEL),
        "SweepPlan": '"None"',
    }
    for key, value in values.items():
        if key in inst.parameters:
            inst.parameters[key].value = value


def _set_hb_two_tone_controller(inst, hb: HBTwoToneSpec) -> None:
    """Set a HarmonicBalance controller instance from a two-tone spec.

    Like :func:`_set_hb_controller`, but ``Freq`` and ``Order`` carry
    two elements -- assigning a two-element list populates index 1 and
    index 2, the two-tone pattern v1's ``_add_hb_controller`` uses
    (``Freq=[f_signal, f_pump]``). ``SweepPlan`` is the same non-empty
    placeholder workaround. Set defensively: a parameter absent from
    the instance (the ADS-free test stub) is skipped.
    """

    freq1_GHz = float(hb.fundamental_freq_1) * 1.0e-9
    freq2_GHz = float(hb.fundamental_freq_2) * 1.0e-9
    values: dict[str, object] = {
        "Freq": [f"{freq1_GHz:.12g} GHz", f"{freq2_GHz:.12g} GHz"],
        "Order": [
            str(int(hb.fundamental_order_1)),
            str(int(hb.fundamental_order_2)),
        ],
        "MaxOrder": str(int(hb.max_order)),
        "StatusLevel": str(HB_STATUS_LEVEL),
        "SweepPlan": '"None"',
    }
    for key, value in values.items():
        if key in inst.parameters:
            inst.parameters[key].value = value


def _p1tone_params(source: P1ToneSource) -> dict[str, str]:
    """ADS ``P_1Tone`` instance parameters for one resolved source.

    Mirrors v1's ``_add_signal_source`` / ``_add_pump_source``
    (``hb_workspace_worker.py``): ``Num`` / ``Z`` / ``Freq`` / ``P`` /
    ``Noise``. The value strings come from the shared format helpers in
    ``netlist/elements/p1tone.py``, so the OA instance carries the
    byte-identical parameter set the hpeesofsim emitter writes -- the
    invariant the ``p1tone_content_check`` relies on. ``source`` is
    post-resolver, so ``freq`` / ``power_dBm`` are literal numbers.
    """

    return {
        "Num": str(int(source.num)),
        "Z": format_p1tone_z(source.z_ohm),
        "Freq": format_p1tone_freq(source.freq),
        "P": format_p1tone_power(source.power_dBm, source.phase_rad),
        "Noise": P1TONE_NOISE,
    }


def _source_terminal_pins(inst: object) -> tuple[object, object]:
    """Return a ``P_1Tone`` instance's ``(signal_pin, ground_pin)``.

    The two terminals are identified by ``inst_pin_id`` -- the ``.1``
    pin is the signal/"+" terminal (the driven port node), the ``.2``
    pin is the ground/"-" terminal -- mirroring v1's
    ``hb_workspace_worker._source_pin_terminals``.

    ADS does not guarantee the positional order of ``inst.inst_pins``
    matches the netlist terminal order, so indexing ``inst_pins[0]`` /
    ``inst_pins[1]`` directly can assign nets to the wrong pin objects
    and leave the source's real terminals unconnected -- which makes
    ADS's ``generate_netlist()`` drop the instance entirely (the
    P_1Tone-OA emission failure). The
    id-based lookup is the fix.

    Falls back to bbox-position ordering -- lowest ``(x, y)`` is the
    signal terminal, highest the ground terminal -- when ``inst_pin_id``
    is absent or unparseable (e.g. the ADS-free test stub, whose fake
    pins carry no id and identical bboxes, so the order is preserved).
    """

    pins = list(inst.inst_pins)
    if len(pins) < 2:
        raise ValueError(
            f"P_1Tone instance exposes {len(pins)} pin(s); a source is a "
            f"two-terminal device and needs at least two."
        )
    by_id: dict[int, object] = {}
    for pin in pins:
        ident = str(getattr(pin, "inst_pin_id", "") or "")
        if not ident:
            continue
        try:
            number = int(ident.rsplit(".", 1)[-1])
        except (IndexError, ValueError):
            continue
        by_id[number] = pin
    if 1 in by_id and 2 in by_id:
        return by_id[1], by_id[2]
    ordered = sorted(pins, key=_pin_xy)
    return ordered[0], ordered[-1]


def _pin_by_logical_id(inst: object, logical_number: int):
    """Return the inst_pin whose ADS ``inst_pin_id`` ends with ``.N``.

    ADS does not guarantee the positional ``inst.inst_pins`` order
    matches the logical pin numbering -- ADS reports an
    I_Probe's inst_pins array is reversed (inst_pins[0] is logical pin
    .2). For an element type that must be wired by logical pin, this
    looks the pin up by id (the same pattern
    :func:`_source_terminal_pins` uses for P_1Tone).

    Returns ``None`` when no pin carries a parseable ``inst_pin_id`` --
    the ADS-free test stub's fake pins have none, so the caller falls
    back to the positional index.
    """

    for pin in getattr(inst, "inst_pins", []):
        ident = str(getattr(pin, "inst_pin_id", "") or "")
        if not ident:
            continue
        try:
            number = int(ident.rsplit(".", 1)[-1])
        except (IndexError, ValueError):
            continue
        if number == logical_number:
            return pin
    return None


def _add_wire_label(net: object, name: str) -> None:
    """Place a GUI wire-label on ``net``'s wire so the ADS GUI exposes the node.

    Renaming the OA ``Net`` (``net.name = ...``) names it in the *netlist* --
    enough for headless ``hpeesofsim`` to write the ``.ds`` voltage column (the
    ``test_level9_named_nets_appear_in_hb_dataset`` guarantee) -- but it does
    NOT place the schematic wire-label the ADS GUI's node picker scans. Without
    that label the GUI reports "no named nodes found" and node voltages are not
    plottable, which breaks Data-Display equations that reference the nets.

    ``Line.add_wire_label`` is the OA equivalent of the GUI Wire/Pin-Label
    (NAME) tool; it attaches a net-name ``AttrDisplay`` to one of the net's wire
    (``Line``) shapes. Duck-typed (only wire shapes expose ``add_wire_label``),
    so no ``db_uu`` type import is needed here. Idempotent: skips a net that
    already carries a label shape. Best-effort: a net with no wire shape (pins
    coincident, no routed segment) is left unlabeled rather than raising -- the
    netlist name still carries it for the headless path.
    """

    try:
        shapes = list(net.shapes)
    except Exception:  # noqa: BLE001
        return
    # Already labeled? (a wire-label is an AttrDisplay shape on the net.)
    for shape in shapes:
        if type(shape).__name__ == "AttrDisplay":
            return
    for shape in shapes:
        adder = getattr(shape, "add_wire_label", None)
        if callable(adder):
            try:
                adder(name)
            except Exception:  # noqa: BLE001
                continue
            return


def _apply_name_annotations(
    spec: DeviceSpec,
    oa_by_id: dict[str, object],
    name_elements_by_net: dict[str, str],
    ground_nodes: set[str],
) -> None:
    """Rename OA Nets to expose spec net voltages in the HB ``.ds``.

    For each spec net that has a ``NAME`` annotation, find any non-NAME
    element pin on that net (via ``oa_by_id``), look up the inst_pin
    on the post-save design, and set its ``.net.name`` to the spec net
    id. ADS HB writes a voltage column for every named net; un-named
    nets keep their auto-generated ``N__nn`` form and stay out of the
    dataset, keeping it lean.

    Post-save-only OA mutation pattern
    -----------------------------------
    This function is the canonical example of an OA-side mutation that
    MUST run AFTER ``design.save_design()`` AND resolve Net / pin
    handles FRESHLY each time. Development surfaced two
    related failure modes future post-emission OA passes should avoid:

    * **Pre-save handles are stale.** Before save, ``inst_pin.net``
      can return a Net object whose ``.name`` reports a value that
      does not appear in the final netlist (a live probe saw
      ``TEE_C2.3.net.name == "gnd!"`` pre-save while the same pin
      in the post-save netlist references a unique floating
      ``N__54``). Mutations on those pre-save handles silently
      strand on a net the netlister no longer uses.
    * **Cached pin lists go stale too.** ``node_pins`` (the
      spec-net -> pin-list map the connectivity pass builds) is fine
      to *iterate* post-save, but the recipe is "re-read
      ``inst_pin.net`` now," not "use the Net handle we cached
      pre-save."

    Future OA-side mutations -- net-attribute setting, instance
    parameter overrides, post-route adjustments -- should run after
    ``design.save_design()`` and use the same fresh-handle pattern.

    Pattern: this is the probe-established working pattern -- rename via
    ``inst_pin.net.name`` on the post-save design. The pre-save Net
    handle is unstable; ``design.save_design()`` reconciles
    connectivity and the post-save Net is the one the netlister
    consumes.

    Idempotent: when two NAME elements happen to point at the same
    physical OA net (the validator forbids it at the spec level, but
    ADS may merge two spec nets into one OA net), the second rename
    finds ``net.name`` already equal to the target and skips.

    Parameters
    ----------
    spec
        The validated, resolved DeviceSpec being emitted.
    oa_by_id
        Map from spec element id to the OA inst object returned by
        ``add_instance``. Built by the main emit loop.
    name_elements_by_net
        Map from target spec net id to the NAME element id annotating
        that net (one entry per NAME element, validator-enforced).
    ground_nodes
        Spec ground-node set; NAMEs on ground are silently skipped --
        the global ``0`` net is special in ADS and we leave it alone.
    """

    # Build a map: spec net -> (element_id, pin_number) of any
    # non-NAME element that touches that net. The first such anchor
    # wins; subsequent anchors produce the same OA net by KCL.
    anchors: dict[str, tuple[str, int]] = {}
    for element in spec.elements:
        if element.type == "NAME":
            continue
        for pin in element.pins:
            anchors.setdefault(pin.node, (element.id, pin.pin))

    for target_net in name_elements_by_net:
        if target_net in ground_nodes:
            continue
        anchor = anchors.get(target_net)
        if anchor is None:
            continue
        anchor_id, anchor_pin_number = anchor
        inst = oa_by_id.get(anchor_id)
        if inst is None:
            continue
        # Resolve the inst_pin for the anchor's device-spec pin
        # number. The OA emitter's existing pin-resolution conventions
        # apply: I_Probe (and other ``_ID_RESOLVED_PIN_TYPES``)
        # resolve by ``inst_pin_id`` ending in ``.N``; MTEE / SDD2P
        # use the ``_PIN_INDEX`` positional remap; other element
        # types use plain ``device_pin - 1`` positional indexing.
        inst_pin = None
        anchor_element = next(
            (e for e in spec.elements if e.id == anchor_id), None
        )
        if anchor_element is None:
            continue
        if anchor_element.type in _ID_RESOLVED_PIN_TYPES:
            inst_pin = _pin_by_logical_id(inst, anchor_pin_number)
        if inst_pin is None:
            pins = list(getattr(inst, "inst_pins", []))
            pin_index_map = _PIN_INDEX.get(anchor_element.type)
            index = (
                pin_index_map[anchor_pin_number]
                if pin_index_map is not None
                else anchor_pin_number - 1
            )
            if not 0 <= index < len(pins):
                continue
            inst_pin = pins[index]
        net = getattr(inst_pin, "net", None)
        if net is None:
            continue
        try:
            current = net.name
        except AttributeError:
            continue
        if current != target_net:
            try:
                net.name = target_net
            except Exception:  # noqa: BLE001
                # ``Net with name X already exists`` arises when ADS has
                # already given another OA net the target name (two spec
                # nets merged into one OA net by the connectivity
                # engine). Silent skip is safe; downstream tests
                # cross-check named-net presence in the .ds and surface
                # genuine misses.
                continue
        # Place the GUI wire-label so the ADS GUI exposes this node for
        # plotting -- the netlist name alone leaves the GUI reporting "no named
        # nodes". Runs on BOTH the just-renamed and the already-named (the
        # connectivity pass's find_or_add_net(node_id)) paths. Additive: it
        # leaves net.name and the generated netlist byte-unchanged.
        _add_wire_label(net, target_net)


def _effective_elements(spec: DeviceSpec) -> list[Element]:
    """Return ``spec.elements`` with each RLC_SERIES expanded to R, L, C.

    The OA backend has no single ``RLC_SERIES`` symbol, so the composite
    is placed as three primitive instances joined by the same internal
    nodes the hpeesofsim emitter uses (:func:`expand_rlc_series` is the
    shared source of those names). Every other element passes through
    unchanged. The result is the element list the schematic walker
    actually places and wires.

    ``NAME`` annotations are filtered out: they have no OA cell, no
    placement, no wires. The OA-side handling is the post-emission
    :func:`_rename_named_nets` pass, not an ``add_instance`` call.
    """

    effective: list[Element] = []
    for element in spec.elements:
        if element.type == "RLC_SERIES":
            effective.extend(expand_rlc_series(element))
        elif element.type == "NAME":
            continue
        else:
            effective.append(element)
    return effective


def _ads_xy(layout_x: float, layout_y: float) -> tuple[float, float]:
    """Convert a layout-unit coordinate to an ADS schematic coordinate.

    See the ``ADS_*`` constants. Layout y grows
    downward, so it is subtracted: a shunt branch lands at negative ADS
    y, below the through-path row.
    """

    return (
        ADS_HORIZONTAL_OFFSET + layout_x * ADS_UNIT_PER_LAYOUT_UNIT,
        ADS_VERTICAL_OFFSET - layout_y * ADS_UNIT_PER_LAYOUT_UNIT,
    )


def _component_positions(spec: DeviceSpec) -> dict[str, tuple[float, float]]:
    """Map every effective-element id to an explicit ADS coordinate.

    The shared layout engine places the spec's components in layout
    units; each is scaled to ADS units via :func:`_ads_xy`. A composite
    ``RLC_SERIES`` occupies a single layout slot, so its three expanded
    R/L/C primitives are spread across that slot's column -- the keys
    are then the *primitive* ids, matching :func:`_effective_elements`.

    Each shunt branch is then rigidly shifted by :func:`_align_shunt_branches`
    so its tap-facing pin sits directly under the tap pin, which makes
    ADS auto-route the tap and ground drop wires as clean vertical
    segments instead of diagonals.
    """

    layout = build_layout(spec)
    layout_coords = {p.element.id: (p.x, p.y) for p in layout.placements}
    positions: dict[str, tuple[float, float]] = {}
    for element in spec.elements:
        # NAME annotations have no layout footprint -- skip.
        if element.type == "NAME":
            continue
        base_x, base_y = _ads_xy(*layout_coords[element.id])
        if element.type == "RLC_SERIES":
            primitives = expand_rlc_series(element)
            centre = (len(primitives) - 1) / 2.0
            for offset, primitive in enumerate(primitives):
                positions[primitive.id] = (
                    base_x + (offset - centre) * _RLC_CLUSTER_DX,
                    base_y,
                )
        else:
            positions[element.id] = (base_x, base_y)
    _align_shunt_branches(spec, layout, positions)
    return positions


def _source_positions(spec: DeviceSpec) -> dict[str, tuple[float, float]]:
    """Map every ``P_1Tone`` source name to its ADS schematic coordinate.

    The layout engine places each source next to the net it drives
    (:func:`tvtl_ads_builder_v2.layout._place_sources`); this scales
    those layout-unit positions to ADS units via :func:`_ads_xy`. The
    coordinate is therefore derived from the topology -- there is no
    hardcoded source position -- so a source sits beside its target
    pin and its wire stays short and axis-aligned. Empty for a spec
    with no sources.
    """

    return {
        placed.name: _ads_xy(placed.x, placed.y)
        for placed in build_layout(spec).sources
    }


def _align_shunt_branches(
    spec: DeviceSpec,
    layout: CircuitLayout,
    positions: dict[str, tuple[float, float]],
) -> None:
    """Shift each shunt branch so its tap-facing pin is under the tap pin.

    A shunt component is a horizontal ADS symbol whose pins sit at fixed
    offsets from the add_instance origin (:data:`_PIN_OFFSETS`). Placing
    the tap component and the shunt component at the same origin x does
    not put their *connecting* pins at the same x, so ADS auto-routes a
    diagonal wire between them.

    This pass rigidly shifts every effective component of a shunt branch
    by a single ``dx`` so the branch's first (tap-facing) component's
    input pin lands exactly at the tap pin's x -- a clean vertical drop.
    Shifting the branch as one block preserves the already-horizontal
    inter-component wires of a multi-component branch (an ``RLC_SERIES``
    expands to R/L/C). ``positions`` is modified in place.
    """

    net_to_effective: dict[str, list[Element]] = {}
    for element in _effective_elements(spec):
        for pin in element.pins:
            net_to_effective.setdefault(pin.node, []).append(element)

    for placed in layout.placements:
        if placed.role != "shunt":
            continue
        shunt_element = placed.element
        # The "top" pin faces the tap; the "bottom" pin the terminus.
        tap_facing = [p for p, side in placed.pin_sides.items() if side == "top"]
        if not tap_facing:
            continue
        in_net = next(
            p.node for p in shunt_element.pins if p.pin == tap_facing[0]
        )

        # The effective components of this branch (an RLC_SERIES expands).
        if shunt_element.type == "RLC_SERIES":
            branch = list(expand_rlc_series(shunt_element))
        else:
            branch = [shunt_element]
        branch_ids = {e.id for e in branch}

        # The first component carries a pin on the tap net; the tap
        # component is the other (non-branch) component on that net.
        first = next(
            (e for e in branch if any(p.node == in_net for p in e.pins)), None
        )
        tap = next(
            (e for e in net_to_effective.get(in_net, ())
             if e.id not in branch_ids), None
        )
        if first is None or tap is None:
            continue
        first_in_pin = next(p.pin for p in first.pins if p.node == in_net)
        tap_pin = next(p.pin for p in tap.pins if p.node == in_net)

        tap_pin_x = positions[tap.id][0] + _pin_offset(tap.type, tap_pin)[0]
        first_in_x = (
            positions[first.id][0] + _pin_offset(first.type, first_in_pin)[0]
        )
        dx = tap_pin_x - first_in_x
        for element in branch:
            x, y = positions[element.id]
            positions[element.id] = (x + dx, y)


def _ground_symbol_xy(
    ground_node: str,
    effective_elements: list[Element],
    component_positions: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """ADS coordinate for the GROUND symbol on one ground node.

    The symbol is placed directly below the real exit pin of the shunt
    component that terminates on the node, so ADS auto-routes the wire
    to ground as a clean vertical drop. The exit pin's position is the
    component's placed origin plus its measured pin offset
    (:data:`_PIN_OFFSETS`); the GROUND symbol's own pin sits at its
    origin, so the origin is placed at that exit-pin x, ``_GROUND_DROP``
    units below.
    """

    for element in effective_elements:
        if element.id not in component_positions:
            continue
        for pin in element.pins:
            if pin.node != ground_node:
                continue
            origin_x, origin_y = component_positions[element.id]
            off_x, off_y = _pin_offset(element.type, pin.pin)
            return (origin_x + off_x, origin_y + off_y - _GROUND_DROP)
    return (ADS_HORIZONTAL_OFFSET, ADS_VERTICAL_OFFSET - _GROUND_DROP)


def _sweep_params(analysis) -> dict[str, str]:
    """ADS S_Param sweep parameters from the device-spec analysis grid."""

    grid = analysis.frequency_grid_Hz
    start_GHz = float(grid.start) * 1.0e-9
    stop_GHz = float(grid.stop) * 1.0e-9
    points = int(grid.points)
    step_GHz = (stop_GHz - start_GHz) / (points - 1) if points > 1 else 0.0
    return {
        "Start": f"{start_GHz:.12g} GHz",
        "Stop": f"{stop_GHz:.12g} GHz",
        "Step": f"{step_GHz:.12g} GHz",
    }


def _pin_xy(pin) -> tuple[float, float]:
    """Bounding-box centre of a pin -- the coordinate ``add_wire`` needs."""

    bbox = getattr(pin, "bbox", None)
    if bbox is None:
        return (0.0, 0.0)
    return (
        (float(bbox.x1) + float(bbox.x2)) / 2.0,
        (float(bbox.y1) + float(bbox.y2)) / 2.0,
    )


# --- off-row polyline wire routing -----------------------------------
# ADS 2026 recomputes a design's connectivity from wire geometry when a
# P_1Tone source is present (the M1/M2 minimal experiment isolated this
# trigger -- PIN_OFFSET_PROBE.txt). A straight wire between two pins on a
# shared pin row then runs collinearly THROUGH every unrelated pin
# between its endpoints and shorts those nets together (the Level-7
# merges -- PIN_COINCIDENCE_DIAGNOSTIC.txt).
#
# Every wire is therefore routed as an axis-aligned polyline: it leaves
# each pin perpendicular to the pin row (a vertical stub off a
# horizontal-axis symbol), runs horizontally on a track that clears
# every pin row, then drops onto the far pin. No wire segment is ever
# collinear with a pin row, so no wire can pass through an unrelated
# pin -- the merge mechanism is removed deterministically. A wire whose
# two pins already share a column is left as the direct vertical
# segment: it lies on no pin row and crosses nothing.

# A routed wire's horizontal run stays at least this many ADS units
# clear of every pin row.
_TRACK_MARGIN = 1.0
# Successive horizontal tracks are offset by this much, so two distinct
# nets never share one collinear horizontal line.
_TRACK_STEP = 0.5
# Coordinates closer than this are treated as identical.
_ROUTE_EPS = 1.0e-6


def _pin_rows(pin_xys: list[tuple[float, float]]) -> list[float]:
    """Sorted, de-duplicated Y coordinates of every placed pin.

    These are the horizontal lines a routed wire's horizontal run must
    avoid: a run sharing a pin row's Y passes through the pins on it.
    """

    return sorted({round(xy[1], 3) for xy in pin_xys})


def _clear_track(
    y_lo: float,
    y_hi: float,
    pin_rows: list[float],
    used: set[float],
) -> float | None:
    """Pick a horizontal-run Y clear of every pin row and of ``used``.

    ``y_lo`` / ``y_hi`` bound the wire's pins. A track strictly between
    two distinct pin levels is preferred -- its perpendicular stubs then
    never leave the pin-free gap. Otherwise the track sits outside the
    pins, on whichever side has the larger clear gap (the through path
    routes up into open space; a shunt row routes down). ``used`` holds
    every track already handed out, so two different nets never get the
    same collinear line. Returns None when no clear track exists -- the
    caller then falls back to a direct wire.
    """

    def is_clear(y: float) -> bool:
        return all(abs(y - row) >= _TRACK_MARGIN - 1.0e-9 for row in pin_rows)

    candidates: list[float] = []
    # Between the pins, fanning out from the midpoint -- only positions
    # that keep a full margin off both bounding pins.
    if y_hi - y_lo >= 2.0 * _TRACK_MARGIN:
        mid = (y_lo + y_hi) / 2.0
        for k in range(80):
            for cand in (mid + k * _TRACK_STEP, mid - k * _TRACK_STEP):
                if y_lo + _TRACK_MARGIN <= cand <= y_hi - _TRACK_MARGIN:
                    candidates.append(cand)
    # Outside the pins -- the side whose nearest pin row is farther off
    # is offered first.
    rows_above = [r for r in pin_rows if r > y_hi + _ROUTE_EPS]
    rows_below = [r for r in pin_rows if r < y_lo - _ROUTE_EPS]
    gap_above = (min(rows_above) - y_hi) if rows_above else float("inf")
    gap_below = (y_lo - max(rows_below)) if rows_below else float("inf")
    above = [y_hi + _TRACK_MARGIN + k * _TRACK_STEP for k in range(160)]
    below = [y_lo - _TRACK_MARGIN - k * _TRACK_STEP for k in range(160)]
    candidates += (above + below) if gap_above >= gap_below else (below + above)

    for track in candidates:
        key = round(track, 3)
        if key in used:
            continue
        if is_clear(track):
            used.add(key)
            return track
    return None


def _simplify_path(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Drop coincident and collinear interior vertices from a polyline."""

    out: list[tuple[float, float]] = []
    for pt in points:
        if (
            out
            and abs(pt[0] - out[-1][0]) < _ROUTE_EPS
            and abs(pt[1] - out[-1][1]) < _ROUTE_EPS
        ):
            continue
        out.append(pt)
    index = 1
    while index < len(out) - 1:
        (x0, y0), (x1, y1), (x2, y2) = out[index - 1], out[index], out[index + 1]
        same_x = abs(x0 - x1) < _ROUTE_EPS and abs(x1 - x2) < _ROUTE_EPS
        same_y = abs(y0 - y1) < _ROUTE_EPS and abs(y1 - y2) < _ROUTE_EPS
        if same_x or same_y:
            del out[index]
        else:
            index += 1
    return out


def _route_wire(
    start: tuple[float, float],
    end: tuple[float, float],
    track: float | None,
) -> list[tuple[float, float]]:
    """Axis-aligned polyline connecting ``start`` to ``end``.

    A wire whose pins already share a column is the direct vertical
    segment -- it lies on no pin row. Otherwise the wire is a Z-route:
    a perpendicular stub off ``start`` to ``track``, a horizontal run
    along ``track`` (clear of every pin row), and a perpendicular stub
    onto ``end``. With no clear track the wire falls back to a direct
    segment so a build is never aborted by a routing gap.
    """

    if abs(start[0] - end[0]) < _ROUTE_EPS or track is None:
        return [start, end]
    return _simplify_path(
        [start, (start[0], track), (end[0], track), end]
    )


def _add_polyline(design, points: list[tuple[float, float]]) -> None:
    """Emit one polyline wire; a routing fault must not abort the build."""

    if len(points) < 2:
        return
    try:
        design.add_wire(list(points))
    except Exception:  # noqa: BLE001 -- the wire only reinforces pin.net
        pass


def _generate_netlist_text(design) -> str | None:
    """Return the ADS netlist text via the design's netlister.

    Mirrors v1's ``hb_workspace_worker.py`` -- ``generate_netlist()``
    returns the netlist as text; ``create_netlist`` is the fallback name.
    """

    for method in ("generate_netlist", "create_netlist"):
        try:
            return str(getattr(design, method)())
        except Exception:  # noqa: BLE001 -- try the next method name
            continue
    return None


def _emit_build_artifacts(
    spec: DeviceSpec, circuit_dir: Path, netlist_text: str
) -> None:
    """Write the ADS-free per-circuit build artifacts.

    Produces ``<label>_topology.svg`` (an ADS-independent topology view)
    and ``<label>_adjacency.txt`` (the per-component adjacency audit) in
    ``circuit_dir``. A renderer or audit fault is logged and swallowed:
    a drawing or audit problem must never abort the workspace build.
    """

    try:
        render_topology_svg(spec, circuit_dir / f"{spec.label}_topology.svg")
    except Exception as exc:  # noqa: BLE001 -- a drawing must not fail a build
        _LOG.warning("topology SVG skipped for %r: %s", spec.label, exc)
    try:
        report = audit_circuit(spec, netlist_text)
        (circuit_dir / f"{spec.label}_adjacency.txt").write_text(
            report.report_text, encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001 -- an audit must not fail a build
        _LOG.warning("adjacency audit skipped for %r: %s", spec.label, exc)


def emit_workspace_from_spec(
    spec: DeviceSpec,
    workspace_dir: str | Path,
    *,
    library_name: str = "oa_lib",
    cell_name: str | None = None,
    auto_netlist: bool = True,
) -> dict[str, str]:
    """Build an ADS workspace for any device spec.

    Walks ``spec.elements`` and ``spec.nodes`` and builds the schematic
    via ``keysight.ads.de``: the MSUB substrate, one instance per
    element, a GROUND symbol per ground node, the S_Param controller,
    and connectivity by node. This is the spec-driven core shared by
    the diagnostic ladder and the canonical Level 1-3 circuits.

    Every build also writes two ADS-free artifacts next to the netlist
    -- ``<label>_topology.svg`` (an ADS-independent topology view) and
    ``<label>_adjacency.txt`` (the per-component adjacency audit).

    Parameters
    ----------
    spec : DeviceSpec
        Any device spec built from TermG / MLIN / MTEE elements.
    workspace_dir : str or Path
        Directory the ADS workspace is created at (removed first if it
        exists). The netlist files are written next to it.
    library_name : str
        Schematic library name.
    cell_name : str, optional
        Schematic cell name; defaults to ``"<spec.label>_oa_test"``.
    auto_netlist : bool
        When True (default), generate the ADS netlist via the Python
        API. When False (the ``--manual`` path), skip it -- the user
        runs ``Tools -> Generate Netlist`` themselves.

    Returns
    -------
    dict
        ``{"workspace_path", "ads_netlist_path", "hpeesofsim_netlist_path"}``.

    Notes
    -----
    Requires an ADS installation: ``keysight.ads.de`` is imported here.
    """

    # Resolve any VAR block to literals before anything reads element
    # parameters -- a no-op for every spec with no VAR block.
    spec = resolve_spec(spec)
    cell_name = cell_name or f"{spec.label}_oa_test"
    workspace_dir = Path(workspace_dir)
    circuit_dir = workspace_dir.parent
    circuit_dir.mkdir(parents=True, exist_ok=True)

    # hpeesofsim flat deck -- ADS-free; emit it first so it always exists.
    netlist_text = emit_hpeesofsim_netlist(spec)
    hpeesofsim_path = circuit_dir / "hpeesofsim.ckt"
    hpeesofsim_path.write_text(netlist_text, encoding="utf-8")

    # Topology SVG + adjacency report -- ADS-free build artifacts, written
    # before the OA build so they land on every build even when no ADS
    # runtime is present.
    _emit_build_artifacts(spec, circuit_dir, netlist_text)

    _ensure_keysight_on_path()
    import keysight.ads.de as de
    from keysight.ads.de import db_uu as db

    # --- workspace + schematic library --------------------------------
    if de.workspace_is_open():
        de.close_workspace()
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace = de.create_workspace(str(workspace_dir))
    workspace.open()
    lib_path = workspace_dir / library_name
    de.create_new_library(library_name, str(lib_path))
    workspace.add_library(library_name, str(lib_path), de.LibraryMode.NON_SHARED)
    library = workspace.open_library(
        library_name, str(lib_path), de.LibraryMode.NON_SHARED
    )
    library.setup_schematic_tech()
    design = db.create_schematic(f"{library_name}:{cell_name}:schematic")

    def setp(inst, name: str, value: str) -> None:
        if name in inst.parameters:
            inst.parameters[name].value = value

    # --- substrate ----------------------------------------------------
    msub = design.add_instance(
        _MSUB_CELL, (-8.0, 4.0), name=spec.substrate.name, angle=0
    )
    for key, value in _msub_params(spec.substrate).items():
        setp(msub, key, value)

    # --- device-spec elements -----------------------------------------
    # RLC_SERIES has no single OA symbol; expand it to its R/L/C
    # primitives so every placed instance is a library cell, and so the
    # OA and hpeesofsim netlists carry the identical R-L-C topology.
    #
    # Every component is placed at an explicit, non-overlapping ADS
    # coordinate from the shared layout engine (compute_layout).
    # Position-setting (the add_instance coordinate) is kept separate
    # from the parameter-setting loop below.
    effective_elements = _effective_elements(spec)
    component_positions = _component_positions(spec)
    oa_by_id: dict[str, object] = {}
    for element in effective_elements:
        cell = _OA_CELL.get(element.type)
        if cell is None:
            raise ValueError(
                f"spec {spec.label!r}: element {element.id!r} has "
                f"unsupported type {element.type!r}"
            )
        inst = design.add_instance(
            cell, component_positions[element.id], name=element.id, angle=0
        )
        for key, value in _element_params(element, spec.substrate.name).items():
            setp(inst, key, value)
        if element.type == "SDD2P":
            _set_sdd2p_body(inst, element)
        oa_by_id[element.id] = inst

    # --- analysis controller ------------------------------------------
    sparam = design.add_instance(
        _SPARAM_CELL, (-8.0, -5.0), name="SP1", angle=0
    )
    for key, value in _sweep_params(spec.analysis).items():
        setp(sparam, key, value)

    # --- HB controller (single-tone) ----------------------------------
    # An extra HarmonicBalance controller, parked below S_Param, when the
    # spec carries an hb_single_tone directive. Mirrors v1's
    # _add_hb_controller; a no-op for every spec without an HB analysis.
    if spec.hb_single_tone is not None:
        hb_ctrl = design.add_instance(
            _HB_CELL, (-8.0, -8.0), name=HB_CONTROLLER_NAME, angle=0
        )
        _set_hb_controller(hb_ctrl, spec.hb_single_tone)
    if spec.hb_two_tone is not None:
        hb_ctrl = design.add_instance(
            _HB_CELL, (-8.0, -8.0), name=HB_CONTROLLER_NAME, angle=0
        )
        _set_hb_two_tone_controller(hb_ctrl, spec.hb_two_tone)

    # --- P_1Tone power sources ----------------------------------------
    # Each P1ToneSource is placed at the coordinate the layout engine
    # derives from the net it drives (_source_positions -> the layout's
    # _place_sources) -- next to its target pin, so the source's wire
    # is short and axis-aligned, not a long diagonal across the
    # topology. The two terminals are identified by inst_pin_id
    # (_source_terminal_pins) -- NOT positionally: ADS does not
    # guarantee inst_pins order matches the netlist terminal order, and
    # a wrong assignment leaves the source unconnected so ADS drops it
    # from the generated netlist. The signal terminal joins the source's
    # port_node net in the connectivity pass below; the ground terminal
    # is anchored to a GROUND symbol after wiring. A no-op for every
    # spec without sources.
    source_positions = _source_positions(spec)
    source_pos_pins: dict[str, list] = {}
    source_neg_pins: list = []
    for source in spec.sources:
        src_inst = design.add_instance(
            _P1TONE_CELL,
            source_positions[source.name],
            name=source.name,
            angle=0,
        )
        for key, value in _p1tone_params(source).items():
            setp(src_inst, key, value)
        signal_pin, ground_pin = _source_terminal_pins(src_inst)
        source_pos_pins.setdefault(source.port_node, []).append(signal_pin)
        source_neg_pins.append(ground_pin)

    # --- connectivity by node -----------------------------------------
    ground_nodes = set(spec.ground_nodes)
    node_pins: dict[str, list] = {}
    for element in effective_elements:
        inst = oa_by_id[element.id]
        pins = list(inst.inst_pins)
        pin_index = _PIN_INDEX.get(element.type)
        id_resolved = element.type in _ID_RESOLVED_PIN_TYPES
        for element_pin in element.pins:
            pin_obj = None
            if id_resolved:
                # device-spec pin N maps to ADS logical pin .N; resolve
                # by inst_pin_id, not by the positional inst_pins index
                # _pin_by_logical_id returns None for
                # the ADS-free stub's id-less fake pins, so the loop
                # then falls through to the positional index.
                pin_obj = _pin_by_logical_id(inst, element_pin.pin)
            if pin_obj is None:
                index = (
                    pin_index[element_pin.pin]
                    if pin_index is not None
                    else element_pin.pin - 1
                )
                if not 0 <= index < len(pins):
                    raise ValueError(
                        f"spec {spec.label!r}: element {element.id!r} "
                        f"pin {element_pin.pin} out of range ({len(pins)} pins)"
                    )
                pin_obj = pins[index]
            node_pins.setdefault(element_pin.node, []).append(pin_obj)

    # Each source's positive terminal drives its port_node net, on the
    # same net as whatever element pins already sit there.
    for port_node, pins in source_pos_pins.items():
        node_pins.setdefault(port_node, []).extend(pins)

    # A GROUND symbol anchors each ground node. ADS places a GROUND
    # symbol's pin on the global-ground net; that net object is read
    # back and reused for every pin on the node. Assigning ground pins
    # to find_or_add_net("0") instead leaves them on a floating
    # "_net0" net -- the failure mode v1's hb_workspace_worker.py
    # documents and avoids the same way.
    # A ground node no element pin touches is skipped -- e.g. a cell ground
    # left orphaned when the via grounds internally (VIAGND): the spec still
    # declares ``n_c{k}_g`` but no element connects to it, so placing a GROUND
    # symbol there would strand a floating symbol at the layout-fallback origin.
    # Every via that uses an explicit ground net (the lumped ``L`` LVIA, RLC via)
    # keeps a pin on its ground node, so the default/anchor path is unaffected.
    touched_nodes = {
        pin.node for element in effective_elements for pin in element.pins
    }
    ground_net: dict[str, object] = {}
    for g_index, ground_node in enumerate(
        sorted(n for n in ground_nodes if n in touched_nodes)
    ):
        gnd = design.add_instance(
            _GROUND_CELL,
            _ground_symbol_xy(ground_node, effective_elements, component_positions),
            name=f"GND{g_index + 1}",
            angle=0,
        )
        gnd_pin = list(gnd.inst_pins)[0]
        net_obj = gnd_pin.net
        if net_obj is None:
            net_obj = design.find_or_add_net("0")
            gnd_pin.net = net_obj
        ground_net[ground_node] = net_obj
        node_pins.setdefault(ground_node, []).append(gnd_pin)

    # Off-row routing needs every placed pin's Y -- the pin rows a
    # horizontal run must clear. node_pins now holds every element,
    # source-signal and ground pin; the source return pins are folded
    # in so a return wire's track clears them too.
    all_pin_xy = [_pin_xy(pin) for pins in node_pins.values() for pin in pins]
    all_pin_xy += [_pin_xy(pin) for pin in source_neg_pins]
    pin_rows = _pin_rows(all_pin_xy)
    used_tracks: set[float] = set()

    for node_id, pin_list in node_pins.items():
        if node_id in ground_nodes:
            net = ground_net[node_id]
        else:
            net = design.find_or_add_net(node_id)
        for pin in pin_list:
            pin.net = net
        coords = [_pin_xy(pin) for pin in pin_list]
        # One horizontal track for the whole net: every wire of the net
        # shares it (same net, so any overlap is electrically a no-op),
        # and it is distinct from every other net's track so no two nets
        # ever share a collinear line. A track is allocated only when a
        # wire needs one -- a wire whose pins share a column routes as a
        # straight vertical and crosses no pin row.
        net_track: float | None = None
        if any(
            abs(coords[0][0] - other[0]) >= _ROUTE_EPS for other in coords[1:]
        ):
            net_ys = [xy[1] for xy in coords]
            net_track = _clear_track(
                min(net_ys), max(net_ys), pin_rows, used_tracks
            )
        for other in coords[1:]:
            _add_polyline(design, _route_wire(coords[0], other, net_track))

    # A P_1Tone source's negative (return) terminal is global ground.
    # Anchor EACH return pin with its OWN GROUND symbol placed COINCIDENT
    # with that return pin. A GROUND symbol's pin sits exactly at its
    # add_instance origin (measured: offset (0,0)), so placing the symbol at
    # ``_pin_xy(return)`` lands the GROUND pin and the return pin on the same
    # coordinate. Coincident pins are the STRONGEST connectivity primitive --
    # ADS treats two pins at one point as a single node unconditionally, and
    # a GROUND pin forces global ground -- so the return terminal stays
    # grounded even when ADS RECOMPUTES connectivity from geometry (which a
    # schematic wire-label, or a user pressing Save/Simulate in the GUI,
    # triggers whenever a P_1Tone source is present).
    #
    # The earlier scheme -- one shared ``GND_SRC`` below the leftmost source,
    # reached by off-row Z-routed return wires -- left each return pin a free
    # node on the ``n_input`` pin row (both source pins are on that row). The
    # wire-label recompute then merged every return pin into ``n_input`` and
    # SHORTED both sources: the generated netlist read ``Port:SIGNAL_SRC
    # n_input n_input`` instead of ``n_input 0`` (the connectivity
    # break; ``Port 1 missing`` / ``sources shorted`` in the GUI). Coincident
    # grounding removes that fragile reach entirely. (The headless flat HB
    # deck is emitted from the spec, not the schematic, so it is unchanged --
    # the anchor still hashes ``D135A120...``.)
    for s_index, pin in enumerate(source_neg_pins):
        gnd_src = design.add_instance(
            _GROUND_CELL, _pin_xy(pin), name=f"GND_SRC{s_index + 1}", angle=0,
        )
        gnd_src_pin = list(gnd_src.inst_pins)[0]
        src_ground_net = gnd_src_pin.net
        if src_ground_net is None:
            src_ground_net = design.find_or_add_net("0")
            gnd_src_pin.net = src_ground_net
        pin.net = src_ground_net

    design.save_design()

    # --- NAME annotations ----------------------------------------------
    # ADS HB only writes voltage columns to the ``.ds`` for nets the
    # schematic gives explicit names. ``NAME`` is a spec-level
    # annotation declaring "expose this net's voltage." The rename
    # pass below assigns spec net names to the corresponding OA Nets,
    # which the next ``generate_netlist`` then emits.
    #
    # The rename runs AFTER ``design.save_design()`` because the
    # connectivity engine reconciles its net assignments during save;
    # any pre-save ``net.name = ...`` operation targets a net handle
    # the netlister no longer uses. The post-save design is the only
    # stable point at which the rename lands.
    #
    # Net handles are reached via ``inst_pin.net`` on the *post-save*
    # design -- the probe-established
    # working pattern. For each NAME element, find any non-NAME
    # element pin on the same spec net, look up its OA inst_pin's
    # current ``.net`` (which post-save references the
    # netlister-visible Net), and set its ``.name``. Per-net
    # deduplication is implicit: once the net is renamed, subsequent
    # ``.net.name = target`` calls on the same Net are no-ops (the
    # name already matches).
    name_elements_by_net = {
        e.pins[0].node: e.id
        for e in spec.elements
        if e.type == "NAME" and len(e.pins) == 1
    }
    if name_elements_by_net:
        _apply_name_annotations(
            spec, oa_by_id, name_elements_by_net, set(spec.ground_nodes)
        )
        design.save_design()

    # --- ADS netlist (automated) --------------------------------------
    ads_netlist_path = circuit_dir / "ads_generated_netlist.txt"
    if auto_netlist:
        text = _generate_netlist_text(design)
        if text is not None:
            ads_netlist_path.write_text(text, encoding="utf-8")

    return {
        "workspace_path": str(workspace_dir),
        "ads_netlist_path": str(ads_netlist_path),
        "hpeesofsim_netlist_path": str(hpeesofsim_path),
    }


def emit_diagnostic_workspace(
    diagnostic_id: str,
    workspace_dir: str | Path,
    *,
    library_name: str | None = None,
    cell_name: str | None = None,
    auto_netlist: bool = True,
) -> dict[str, str]:
    """Build an ADS workspace for a diagnostic circuit (D0-D4).

    Thin wrapper over :func:`emit_workspace_from_spec`: it looks up the
    diagnostic's device spec by id and delegates. Kept so the diagnostic
    CLI and tests address diagnostics by name.
    """

    if diagnostic_id not in DIAGNOSTIC_SPECS:
        raise ValueError(
            f"unknown diagnostic {diagnostic_id!r}; "
            f"known: {sorted(DIAGNOSTIC_SPECS)}"
        )
    return emit_workspace_from_spec(
        DIAGNOSTIC_SPECS[diagnostic_id](),
        workspace_dir,
        library_name=library_name or f"{diagnostic_id}_oa_lib",
        cell_name=cell_name or f"{diagnostic_id}_oa_test",
        auto_netlist=auto_netlist,
    )


