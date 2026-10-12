"""Analytical 2-port S-parameter simulator for the ADS-builder-v2 ladder.

Topologies handled, by level:

* Levels 1-2 -- a series cascade of one or more MLIN elements between
  two ports: ``Port -- MLIN -- [MLIN -- ...] -- Port``.
* Level 3 -- a through line with one shunt branch attached at an MTEE
  T-junction: ``Port -- MLIN -- MTEE -- MLIN -- Port`` with the MTEE's
  third pin feeding a shunt branch of MLIN(s) terminating at ground
  (short) or at an open.
* Level 4 -- lumped two-terminal elements (``R``, ``L``, ``C`` and the
  composite ``RLC_SERIES``) anywhere a series MLIN may appear: in the
  series cascade, and as the shunt branch off the MTEE junction.

A two-terminal series element -- an MLIN or a lumped R/L/C/RLC_SERIES --
contributes a 2x2 ABCD matrix. An MLIN's is the frequency-dependent
transmission-line matrix; a lumped element's is the series-impedance
matrix ``[[1, Z(w)], [0, 1]]``. The composite ``RLC_SERIES`` uses the
closed-form series impedance ``Z = R + jwL + 1/(jwC)`` directly --
analytically identical to, and numerically steadier than, cascading
three separate R/L/C matrices. The simulator walks the node graph and
either cascades the elements (Levels 1-2 and the Level-4 series cases)
or cascades the main line with a shunt admittance inserted at the
junction (Level 3 and the Level-4 shunt case).

MTEE junction model
-------------------
At Level 3 the MTEE is treated as an **ideal junction**: the three
pins are electrically the same node, contributing no parasitic L or C.
The branch (shunt) input admittance -- the shorted/open stub
transformed through its MLIN cascade -- is inserted as a shunt at the
merged junction node, between the two main-line MLIN cascades.

The Level-3 discovery decision landed mid-ladder: the ideal junction reproduces ADS within the 0.5 dB
parity envelope for the two non-resonant canonical circuits
(``tee_open_zero_stub``, ``tee_short_paper_dims``) but not for the two
strong-reflection circuits (``tee_short_zero_stub``,
``tee_qw_short_3p1GHz``). Two contributing causes were identified:
(1) the MTEE junction parasitics shift the resonant features, and
(2) ADS's own closed-form MTEE model returns a mildly non-passive
3-port for the strong-reflection geometries (``|S11|^2 + |S21|^2``
exceeds 1), which a passive analytical model cannot match. The
repository's ``tvtl.microstrip.discontinuities.tee_junction`` model
was trialled and over-corrects (it breaks the open-stub cross-check),
so the ideal junction is kept as the cleanest TVTL model pending a
properly fitted, ADS-consistent lumped MTEE model.

A uniform line's ABCD matrix is exactly additive in electrical length,
so splitting one line into collinear segments is electrically
identical -- the Level-2 cross-check relies on this.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tvtl.materials import Substrate as TvtlSubstrate
from tvtl.microstrip.models import calc_gamma_legacy, hammerstad_kirschning
from tvtl.theory.abcd import (
    as_abcd,
    identity_abcd,
    line_abcd,
    series_impedance_abcd,
    shunt_admittance_abcd,
)
from tvtl.theory.sparams import abcd_to_sparameters

from ..device_spec import DeviceSpec, Element, resolve_spec, validate_device_spec


# Cap for a degenerate near-short shunt admittance (zero-length grounded
# stub). The canonical Level-3 circuits use finite stub lengths so this
# is a guard, not a routine path.
_MAX_SHUNT_ADMITTANCE_S = 1.0e6

# Two-terminal series elements: an MLIN or a lumped R/L/C/RLC_SERIES.
# Each contributes one 2x2 ABCD and can sit in a series cascade or in
# the MTEE shunt branch.
_SERIES_2T_TYPES = frozenset({"MLIN", "R", "L", "C", "RLC_SERIES"})

# Every element type the analytical simulator understands.
# ``NAME`` is a spec-level annotation with no electrical
# role; the simulator accepts and ignores it (it is not in
# ``_SERIES_2T_TYPES``, so it contributes no ABCD).
_SIMULATED_ELEMENT_TYPES = _SERIES_2T_TYPES | {"TermG", "MTEE", "NAME"}


def simulate_s_parameters(spec: DeviceSpec) -> dict:
    """Analytically compute the 2-port S-parameters of a device spec.

    Parameters
    ----------
    spec : DeviceSpec
        Device spec for a Level-1/2 MLIN cascade or a Level-3
        through-line-with-shunt-branch circuit. Re-validated here.

    Returns
    -------
    dict
        Keys ``frequencies_Hz``, ``S11``, ``S21``, ``S12``, ``S22``
        (the S-params as complex ndarrays), and ``analysis_label``
        (``"tvtl_analytical"``).
    """

    spec = resolve_spec(spec)
    validate_device_spec(spec)
    if spec.analysis.type != "s_parameters":
        raise ValueError(
            f"simulate_s_parameters cannot handle analysis.type='{spec.analysis.type}'."
        )

    grid = spec.analysis.frequency_grid_Hz
    frequencies_Hz = np.linspace(grid.start, grid.stop, grid.points)
    z0 = float(spec.analysis.port_impedance_ohm)

    mtees = [e for e in spec.elements if e.type == "MTEE"]
    other = [e for e in spec.elements if e.type not in _SIMULATED_ELEMENT_TYPES]
    if other:
        types = sorted({e.type for e in other})
        raise ValueError(
            f"The TVTL simulator does not implement element type(s) {types}; "
            "remove them or wait for the corresponding level."
        )
    termgs = [e for e in spec.elements if e.type == "TermG"]
    if len(termgs) != 2:
        raise ValueError(
            f"The TVTL simulator expects exactly two TermG elements (got {len(termgs)})."
        )

    if not mtees:
        abcd_per_freq = _series_cascade_abcd(spec, frequencies_Hz)
    elif len(mtees) == 1:
        abcd_per_freq = _through_with_shunt_abcd(spec, mtees[0], frequencies_Hz)
    else:
        raise ValueError(
            f"The TVTL simulator handles at most one MTEE (got {len(mtees)})."
        )

    s11 = np.empty(frequencies_Hz.shape, dtype=np.complex128)
    s21 = np.empty(frequencies_Hz.shape, dtype=np.complex128)
    s12 = np.empty(frequencies_Hz.shape, dtype=np.complex128)
    s22 = np.empty(frequencies_Hz.shape, dtype=np.complex128)
    for index in range(frequencies_Hz.size):
        sparams = abcd_to_sparameters(abcd_per_freq[index], reference_impedance_ohm=z0)
        s11[index] = sparams.s11
        s21[index] = sparams.s21
        s12[index] = sparams.s12
        s22[index] = sparams.s22

    return {
        "frequencies_Hz": frequencies_Hz,
        "S11": s11,
        "S21": s21,
        "S12": s12,
        "S22": s22,
        "analysis_label": "tvtl_analytical",
    }


# ---------------------------------------------------------------------------
# Per-element ABCD matrices
# ---------------------------------------------------------------------------


def _lumped_series_impedance(
    element: Element, frequencies_Hz: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Return the series impedance ``Z(w)`` of a lumped element over the grid.

    ``R`` is frequency-flat; ``L`` is ``jwL``; ``C`` is ``1/(jwC)``; the
    composite ``RLC_SERIES`` is ``R + jwL + 1/(jwC)`` -- the closed-form
    series sum, not three cascaded matrices. The frequency grid starts
    strictly above 0 Hz (the validator enforces it) and ``L`` / ``C``
    are strictly positive, so the reactive terms never divide by zero.
    """

    omega = 2.0 * np.pi * np.asarray(frequencies_Hz, dtype=np.float64)
    params = element.parameters
    if element.type == "R":
        return np.full(
            omega.shape, complex(float(params["r_ohm"])), dtype=np.complex128
        )
    if element.type == "L":
        return 1j * omega * float(params["l_h"])
    if element.type == "C":
        return 1.0 / (1j * omega * float(params["c_f"]))
    if element.type == "RLC_SERIES":
        r_ohm = float(params["r_ohm"])
        l_h = float(params["l_h"])
        c_f = float(params["c_f"])
        return r_ohm + 1j * omega * l_h + 1.0 / (1j * omega * c_f)
    raise ValueError(
        f"_lumped_series_impedance: '{element.type}' is not a lumped element."
    )


def _element_abcd_stack(
    spec: DeviceSpec,
    element: Element,
    frequencies_Hz: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return the per-frequency ABCD stack ``(N_freq, 2, 2)`` for one element.

    An ``MLIN`` contributes the frequency-dependent transmission-line
    matrix; a lumped ``R`` / ``L`` / ``C`` / ``RLC_SERIES`` contributes
    the series-impedance matrix ``[[1, Z(w)], [0, 1]]``.
    """

    n_freq = int(frequencies_Hz.size)
    stack = np.empty((n_freq, 2, 2), dtype=np.complex128)
    if element.type == "MLIN":
        z0_line, gamma, length_m = _precompute_lines(
            spec, [element], frequencies_Hz
        )[0]
        for index in range(n_freq):
            stack[index] = line_abcd(
                float(z0_line[index]), gamma[index] * length_m
            )
        return stack
    impedance = _lumped_series_impedance(element, frequencies_Hz)
    for index in range(n_freq):
        stack[index] = series_impedance_abcd(impedance[index])
    return stack


# ---------------------------------------------------------------------------
# Series-cascade topology (Levels 1-2, and the Level-4 lumped cascades)
# ---------------------------------------------------------------------------


def _series_cascade_abcd(
    spec: DeviceSpec, frequencies_Hz: NDArray[np.float64]
) -> list[NDArray[np.complex128]]:
    """Return the per-frequency ABCD product for a series cascade.

    The cascade is any chain of two-terminal series elements -- MLINs
    and/or lumped R/L/C/RLC_SERIES -- connecting port 1 to port 2.
    """

    series_elements = [e for e in spec.elements if e.type in _SERIES_2T_TYPES]
    if not series_elements:
        raise ValueError("A cascade spec needs at least one series element.")
    ordered = _ordered_series_cascade(spec, series_elements)
    stacks = [
        _element_abcd_stack(spec, element, frequencies_Hz) for element in ordered
    ]
    result: list[NDArray[np.complex128]] = []
    for index in range(frequencies_Hz.size):
        abcd = identity_abcd()
        for stack in stacks:
            abcd = as_abcd(abcd @ stack[index])
        result.append(abcd)
    return result


def _ordered_series_cascade(
    spec: DeviceSpec, series_elements: list[Element]
) -> list[Element]:
    """Return the series elements ordered from the input to the output port.

    The Levels 1-2 / Level-4 cascade topology is a strict series chain.
    Raises ``ValueError`` if the topology is not a simple series path.
    """

    input_node = _port_node(spec, 1)
    output_node = _port_node(spec, 2)
    adjacency = _two_terminal_series_adjacency(series_elements)

    if len(series_elements) == 1:
        only = series_elements[0]
        endpoints = {only.pins[0].node, only.pins[1].node}
        if endpoints != {input_node, output_node}:
            raise ValueError(
                f"single-element spec must connect the two ports directly; "
                f"{only.type} '{only.id}' spans {sorted(endpoints)}."
            )
        return [only]

    ordered, _terminus = _traverse_chain(
        start_node=input_node,
        adjacency=adjacency,
        stop_node=output_node,
    )
    if len(ordered) != len(series_elements):
        raise ValueError(
            f"series cascade traversal visited {len(ordered)} of "
            f"{len(series_elements)} elements; the spec contains a "
            "disconnected or branched segment."
        )
    return ordered


# ---------------------------------------------------------------------------
# Through-line-with-shunt-branch topology (Levels 3-4)
# ---------------------------------------------------------------------------


def _through_with_shunt_abcd(
    spec: DeviceSpec, mtee: Element, frequencies_Hz: NDArray[np.float64]
) -> list[NDArray[np.complex128]]:
    """Return the per-frequency ABCD for a through line with one shunt.

    The MTEE is treated as an ideal junction: its three pin nodes are
    electrically merged. The two main pins (1, 2) carry the through
    line; the branch pin (3) feeds the shunt branch. The branch input
    admittance is inserted as a shunt at the merged junction node.

    Every chain -- the two main arms and the branch -- is a series of
    two-terminal elements (MLINs and/or lumped R/L/C/RLC_SERIES). At
    Level 3 all three chains are MLINs; at Level 4 the branch is a
    lumped RLC stub to ground.
    """

    series_elements = [e for e in spec.elements if e.type in _SERIES_2T_TYPES]
    if not series_elements:
        raise ValueError(
            "A through-with-shunt spec needs at least one series element."
        )
    if len(mtee.pins) != 3:
        raise ValueError(f"MTEE '{mtee.id}' must have exactly three pins.")

    node_main_a = mtee.pins[0].node
    node_main_b = mtee.pins[1].node
    node_branch = mtee.pins[2].node
    input_node = _port_node(spec, 1)
    output_node = _port_node(spec, 2)

    adjacency = _two_terminal_series_adjacency(series_elements)

    # Branch chain: from the MTEE branch pin's node outward to a terminus.
    branch_elements, branch_terminus = _traverse_chain(
        start_node=node_branch, adjacency=adjacency, stop_node=None
    )
    if not branch_elements:
        raise ValueError(
            f"MTEE '{mtee.id}' branch pin node '{node_branch}' has no "
            "series element."
        )
    branch_short = branch_terminus in set(spec.ground_nodes)

    # Main left/right chains, traversed junction-outward to a port.
    left_elements, left_terminus = _traverse_chain(
        start_node=node_main_a, adjacency=adjacency, stop_node=None
    )
    right_elements, right_terminus = _traverse_chain(
        start_node=node_main_b, adjacency=adjacency, stop_node=None
    )
    termini = {left_terminus, right_terminus}
    if termini != {input_node, output_node}:
        raise ValueError(
            "through-with-shunt main-line traversal did not reach both ports "
            f"(reached {sorted(termini)}, expected {sorted({input_node, output_node})})."
        )
    if left_terminus == input_node:
        in_to_junction = list(reversed(left_elements))
        junction_to_out = right_elements
    else:
        in_to_junction = list(reversed(right_elements))
        junction_to_out = left_elements

    used = {e.id for e in in_to_junction} | {e.id for e in junction_to_out} | {
        e.id for e in branch_elements
    }
    if len(used) != len(series_elements):
        raise ValueError(
            f"through-with-shunt traversal classified {len(used)} of "
            f"{len(series_elements)} series elements; the spec has a "
            "disconnected or unexpected segment."
        )

    left_stacks = [
        _element_abcd_stack(spec, e, frequencies_Hz) for e in in_to_junction
    ]
    right_stacks = [
        _element_abcd_stack(spec, e, frequencies_Hz) for e in junction_to_out
    ]
    branch_data = _precompute_branch(spec, branch_elements, frequencies_Hz)

    result: list[NDArray[np.complex128]] = []
    for index in range(frequencies_Hz.size):
        z_branch = _branch_input_impedance(branch_data, branch_short, index)
        y_shunt = _safe_reciprocal(z_branch)
        abcd = identity_abcd()
        for stack in left_stacks:
            abcd = as_abcd(abcd @ stack[index])
        abcd = as_abcd(abcd @ shunt_admittance_abcd(y_shunt))
        for stack in right_stacks:
            abcd = as_abcd(abcd @ stack[index])
        result.append(abcd)
    return result


def _precompute_branch(
    spec: DeviceSpec,
    branch_elements: list[Element],
    frequencies_Hz: NDArray[np.float64],
) -> list[tuple[Element, object]]:
    """Precompute per-element data for the shunt branch, junction-outward.

    Each entry is ``(element, data)``: for an ``MLIN``, ``data`` is the
    ``(z0, gamma, length_m)`` line triple; for a lumped element it is
    the per-frequency series-impedance array ``Z(w)``.
    """

    mlins = [e for e in branch_elements if e.type == "MLIN"]
    line_data = dict(
        zip(
            (m.id for m in mlins),
            _precompute_lines(spec, mlins, frequencies_Hz),
        )
    )
    precomputed: list[tuple[Element, object]] = []
    for element in branch_elements:
        if element.type == "MLIN":
            precomputed.append((element, line_data[element.id]))
        else:
            precomputed.append(
                (element, _lumped_series_impedance(element, frequencies_Hz))
            )
    return precomputed


def _series_load(z_load: complex | None, impedance: complex) -> complex | None:
    """Add a series impedance between ``z_load`` and the junction.

    A series element in front of an open termination is still an open
    -- no current path -- so ``None`` (open) stays ``None``; otherwise
    the impedances simply add.
    """

    if z_load is None:
        return None
    return complex(z_load) + complex(impedance)


def _branch_input_impedance(
    branch_data: list[tuple[Element, object]],
    short_terminated: bool,
    freq_index: int,
) -> complex | None:
    """Return the branch input impedance at the branch's top node.

    The branch is a chain of two-terminal series elements terminating
    at ground (short) or at an open. The impedance is transformed
    element by element from the terminus toward the junction: an
    ``MLIN`` transforms it as a transmission line, a lumped element
    adds its series impedance. ``None`` denotes an open.
    """

    z_load: complex | None = 0.0 if short_terminated else None  # None == open
    for element, data in reversed(branch_data):
        if element.type == "MLIN":
            z0_line, gamma, length_m = data
            z0_seg = complex(float(z0_line[freq_index]))
            theta = gamma[freq_index] * length_m
            z_load = _impedance_through_line(z_load, z0_seg, theta)
        else:
            z_load = _series_load(z_load, complex(data[freq_index]))
    return z_load


def _impedance_through_line(
    z_load: complex | None, z0: complex, theta: complex
) -> complex | None:
    """Transform a load impedance through a transmission-line segment.

        Z_in = Z0 * (Z_load + Z0*tanh(theta)) / (Z0 + Z_load*tanh(theta))

    ``z_load = None`` represents an open termination; it stays open
    through a zero-length segment and becomes ``Z0*coth(theta)``
    through a finite one.
    """

    tanh_theta = np.tanh(theta)
    if z_load is None:
        if abs(tanh_theta) < 1.0e-30:
            return None
        return z0 / tanh_theta
    numerator = z_load + z0 * tanh_theta
    denominator = z0 + z_load * tanh_theta
    if abs(denominator) < 1.0e-30:
        return complex(0.0, 0.0)
    return z0 * numerator / denominator


def _safe_reciprocal(z: complex | None) -> complex:
    """Return ``1/z`` as a shunt admittance, capped for a near-short."""

    if z is None:
        return complex(0.0, 0.0)  # open branch -> no shunt loading
    if abs(z) < 1.0 / _MAX_SHUNT_ADMITTANCE_S:
        return complex(_MAX_SHUNT_ADMITTANCE_S, 0.0)
    return 1.0 / z


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _two_terminal_series_adjacency(
    elements: list[Element],
) -> dict[str, list[tuple[Element, str]]]:
    """Build ``node -> [(element, other_node), ...]`` adjacency.

    Every entry is a two-terminal series element -- an MLIN or a lumped
    R/L/C/RLC_SERIES. The adjacency drives the chain traversal that
    orders a series cascade and that finds the MTEE shunt branch.
    """

    adjacency: dict[str, list[tuple[Element, str]]] = {}
    for element in elements:
        if len(element.pins) != 2:
            raise ValueError(
                f"{element.type} '{element.id}' must have exactly two pins."
            )
        node_a = element.pins[0].node
        node_b = element.pins[1].node
        adjacency.setdefault(node_a, []).append((element, node_b))
        adjacency.setdefault(node_b, []).append((element, node_a))
    return adjacency


def _traverse_chain(
    *,
    start_node: str,
    adjacency: dict[str, list[tuple[Element, str]]],
    stop_node: str | None,
) -> tuple[list[Element], str]:
    """Walk a simple MLIN chain from ``start_node``.

    The walk follows MLINs until it reaches ``stop_node`` (if given) or
    a terminus -- a node with no further unused MLIN. It raises if a
    node branches into more than one unused MLIN, i.e. the chain is not
    simple.

    Returns ``(ordered_mlins, terminus_node)``.
    """

    ordered: list[Element] = []
    used: set[str] = set()
    current = start_node
    # Bound the walk by the number of distinct MLINs (each is used once).
    total_mlins = len({line.id for entries in adjacency.values() for line, _ in entries})
    safety = total_mlins + 1
    while True:
        if stop_node is not None and current == stop_node:
            return ordered, current
        candidates = [
            (line, other)
            for line, other in adjacency.get(current, [])
            if line.id not in used
        ]
        if not candidates:
            return ordered, current
        if len(candidates) > 1:
            raise ValueError(
                f"node '{current}' branches into {len(candidates)} unused MLINs; "
                "the simulator only walks a simple chain here."
            )
        line, other = candidates[0]
        ordered.append(line)
        used.add(line.id)
        current = other
        safety -= 1
        if safety < 0:
            raise ValueError("MLIN chain traversal did not terminate (cycle?).")


def _precompute_lines(
    spec: DeviceSpec,
    ordered_mlins: list[Element],
    frequencies_Hz: NDArray[np.float64],
) -> list[tuple[NDArray[np.float64], NDArray[np.complex128], float]]:
    """Return ``(z0_line, gamma, length_m)`` for each MLIN over the grid.

    Returns an empty list when given no MLINs -- a Level-4 shunt branch
    can be entirely lumped, with no microstrip line to characterise.
    """

    if not ordered_mlins:
        return []
    substrate = _substrate_from_spec(spec)
    height_m = float(spec.substrate.parameters["h_m"])
    eps_r = float(spec.substrate.parameters["eps_r"])
    thickness_m = float(spec.substrate.parameters.get("t_m", 0.0))
    per_line: list[tuple[NDArray[np.float64], NDArray[np.complex128], float]] = []
    for line in ordered_mlins:
        width_m, length_m = _mlin_geometry(line)
        z0_line, eps_eff = hammerstad_kirschning(
            width_m=width_m,
            thickness_m=thickness_m,
            height_m=height_m,
            epsilon_r=eps_r,
            frequency_hz=frequencies_Hz,
        )
        gamma = calc_gamma_legacy(
            frequency_hz=frequencies_Hz,
            z0_ohm=z0_line,
            epsilon_eff=eps_eff,
            substrate=substrate,
        )
        per_line.append((z0_line, gamma, length_m))
    return per_line


def _port_node(spec: DeviceSpec, num: int) -> str:
    for port in spec.ports:
        if port.num == num:
            return port.node
    raise ValueError(f"device spec has no port with num={num}.")


def _substrate_from_spec(spec: DeviceSpec) -> TvtlSubstrate:
    p = spec.substrate.parameters
    return TvtlSubstrate(
        epsilon_r=float(p["eps_r"]),
        thickness_m=float(p["h_m"]),
        copper_thickness_m=float(p["t_m"]),
        loss_tangent=float(p["tan_delta"]),
        conductivity_s_per_m=float(p["sigma_S_per_m"]),
    )


def _mlin_geometry(element: Element) -> tuple[float, float]:
    width = float(element.parameters["w_m"])
    length = float(element.parameters["l_m"])
    if width <= 0.0:
        raise ValueError(
            f"MLIN element '{element.id}' has non-positive width (w_m={width})."
        )
    if length < 0.0:
        raise ValueError(
            f"MLIN element '{element.id}' has negative length (l_m={length})."
        )
    return width, length


def s_parameters_to_dB(s21: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert an ``S21`` complex array to magnitude in dB."""

    magnitude = np.abs(s21)
    # Floor magnitude to avoid log(0).
    magnitude = np.where(magnitude > 0.0, magnitude, 1.0e-300)
    return 20.0 * np.log10(magnitude)


