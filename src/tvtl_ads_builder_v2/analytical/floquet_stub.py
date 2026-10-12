"""Per-harmonic stub admittance for the harmonic ABCD lift.

The analytical Floquet solver in :mod:`tvtl.floquet.harmonics` originally
models the shunt branch as a local series RLC with the time-varying
elastance ``S(t) = S0 + dS·cos(ω_m·t + φ)``. The energy-conservation
cross-check surfaced that
this local model omits a ~5 mm stub MLIN that sits in series between
the varactor and ground in the Level-9 canonicals — a divider
``Z_var/(Z_var + Z_stub) ≈ 0.243 ≈ -12.3 dB`` at the signal frequency.

This module evaluates ``Y_stub(ω) = 1 / Z_stub(ω)`` at arbitrary
frequency by reusing the v2 framework's static transmission-line
machinery (Hammerstad-Kirschning Z_0/eps_eff in
:mod:`tvtl.microstrip.models` and the standard input-impedance
recursion). The result is consumed by
:func:`tvtl.floquet.harmonics.build_stub_tridiagonal` via its
``y_stub_fn`` callable — diagonal terms only; off-diagonal modulation
coupling is unchanged because the time-varying capacitance remains
local to the varactor.

The lift consists of one wrapper (:func:`Y_stub_at_freq`), one spec
helper (:func:`extract_level9_stub_segments`), and one fn-builder
convenience (:func:`build_stub_y_fn_from_spec`) — total ~80 lines of
new code, all driven off the existing static machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ..device_spec import DeviceSpec
from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
from tvtl.microstrip.models import hammerstad_kirschning


__all__ = [
    "StubSegment",
    "TerminationImpedance",
    "Y_stub_at_freq",
    "Y_stub_at_freq_with_shunts",
    "build_stub_y_fn_from_spec",
    "build_stub_y_fn_with_mid_shunts",
    "extract_level9_stub_segments",
    "extract_series_via_inductance_H",
    "extract_stub_segments",
]


# A termination impedance is either a complex constant (Z_load at every
# harmonic frequency), one of the named string constants ``"short"`` or
# ``"open"`` (back-compat with the original lift API), or a callable
# ``omega -> Z_load(omega)`` for frequency-dependent loads such as an
# inductive via-to-ground termination (Level-10 paper geometry).
TerminationImpedance = (
    "complex | Callable[[float], complex] | str"  # type alias for docs
)


@dataclass(frozen=True)
class StubSegment:
    """One microstrip section in a stub cascade.

    ``width_m`` and ``length_m`` are the microstrip section's width and
    length in metres; the stub is a series chain of these segments,
    ordered from the varactor terminal toward the termination
    (short to ground for the Level-9 case).
    """

    width_m: float
    length_m: float


def _z0_and_v_phase(
    width_m: float,
    substrate: Mapping[str, float],
    frequency_hz: float,
) -> tuple[float, float]:
    """Return ``(Z_0, v_phase)`` of one microstrip section at one frequency.

    Thin wrapper around :func:`hammerstad_kirschning` matching the
    convention used by :mod:`tvtl.floquet.topologies.open_stub_2segment`
    so the lift produces numerically identical Z_0 / v_phase to the
    static framework on the same geometry.
    """

    eps_r = float(substrate["eps_r"])
    h = float(substrate["h_m"])
    copper_thickness = float(substrate.get("t_m", 17.5e-6))
    z0_arr, eps_eff_arr = hammerstad_kirschning(
        float(width_m), copper_thickness, h, eps_r, [float(frequency_hz)],
    )
    eps_eff = float(eps_eff_arr[0])
    v_phase = SPEED_OF_LIGHT_M_PER_S / math.sqrt(max(eps_eff, 1.0e-30))
    return float(z0_arr[0]), float(v_phase)


def _resolve_termination(
    termination: object, omega_val: float
) -> complex:
    """Map a termination spec to a load impedance at ``omega_val``.

    Accepts the historical strings ``"short"`` / ``"open"`` (original lift
    API), a complex constant (frequency-flat lumped load), or a
    callable ``omega -> Z_load(omega)`` (Level-10's inductive via). The
    Level-9 path keeps ``"short"`` as a default literal so the existing
    callers and regression tests are unchanged.
    """

    if callable(termination):
        return complex(termination(float(omega_val)))
    if isinstance(termination, str):
        if termination == "short":
            return complex(0.0, 0.0)
        if termination == "open":
            return complex("inf")
        raise ValueError(
            f"termination string must be 'short' or 'open' (got "
            f"{termination!r}); use a complex value or a Callable for "
            f"frequency-dependent terminations."
        )
    # Treat any other value as a frequency-flat complex impedance.
    return complex(termination)  # type: ignore[arg-type]


def Y_stub_at_freq(
    omega: float,
    stub_segments: Sequence[StubSegment],
    substrate: Mapping[str, float],
    *,
    termination: object = "short",
) -> complex:
    """Stub-cascade input admittance at angular frequency ``omega``.

    Walks the standard transmission-line input-impedance recursion
    inward from the termination (short for Level-9; ``z_load = 0``)
    through each segment. The Z_0 and effective permittivity are
    evaluated at ``f = omega / (2π)`` per harmonic — the lift's whole
    point is to honour the frequency dependence the static framework
    already understands.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s]. Must be positive.
    stub_segments : sequence of :class:`StubSegment`
        Stub cascade ordered from the varactor outward. The Level-9
        canonical has one segment; the recursion form generalises to
        Level-10 with three.
    substrate : Mapping[str, float]
        Substrate dict carrying at least ``"eps_r"`` and ``"h_m"``; the
        copper thickness ``"t_m"`` defaults to 17.5 μm if absent. Keys
        mirror the v2 :class:`Substrate.parameters` schema.
    termination : str | complex | Callable[[float], complex], optional
        Far-end load presented to the last stub segment.

        - ``"short"`` (the default) — Level-9's grounded VIAGND end —
          starts the recursion at ``Z_load = 0``.
        - ``"open"`` accepts an open-terminated cascade and starts at
          ``Z_load = inf``.
        - A complex value is used directly as a frequency-flat load
          impedance (e.g. a 50 Ω termination).
        - A callable ``omega -> Z_load(omega)`` returns the load
          impedance at each harmonic frequency — the Level-10 paper
          geometry's via-to-ground termination uses
          ``lambda omega: 1j*omega*L_via_eff`` to model the Bahl-formula
          via inductance.

    Returns
    -------
    complex
        ``Y_stub = 1 / Z_in`` at the requested frequency. A finite,
        non-zero value at every physical operating point.
    """

    omega_val = float(omega)
    if not stub_segments:
        raise ValueError("stub_segments must contain at least one segment.")

    # Floquet harmonic frequencies ``ws + n*wm`` can be negative when
    # n*wm > ws. The microstrip Z_0(f)/eps_eff(f) model is defined for
    # |f|; the propagation constant ``beta = omega / v_phase`` carries
    # the sign through ``tan(beta*l)``, which is itself an odd function.
    # We therefore evaluate Hammerstad-Kirschning at |omega|/(2π) and
    # let the sign of beta propagate through tan(beta*l) — the resulting
    # Z_in is the analytic continuation onto the negative-frequency
    # axis, which is exactly what the Floquet system expects.
    if omega_val == 0.0:
        # DC limit: depends on the termination. A short load presents
        # Z_in = 0 -> Y = inf. An inductive load presents Z = 0 at DC
        # (the inductor is a short at DC), same result. The general
        # rule is "evaluate the termination at omega = 0 and propagate"
        # but the closed-form DC limit through a non-zero stub length
        # is just the load impedance itself (the stub is non-dispersive
        # at DC: tan(beta*l) -> 0, so the recursion collapses to
        # z_in = z_load each step).
        z_load_dc = _resolve_termination(termination, 0.0)
        if not math.isfinite(abs(z_load_dc)):
            return complex(0.0, 0.0)
        if abs(z_load_dc) <= 1.0e-300:
            return complex(float("inf"), 0.0)
        return complex(1.0 / z_load_dc)

    frequency_hz = abs(omega_val) / (2.0 * math.pi)

    # Initialise the recursion at the termination and walk inward
    # toward the varactor terminal.
    z_load = _resolve_termination(termination, omega_val)

    for segment in reversed(list(stub_segments)):
        z0, v_phase = _z0_and_v_phase(segment.width_m, substrate, frequency_hz)
        beta = omega_val / v_phase
        tan_term = math.tan(beta * float(segment.length_m))
        if not math.isfinite(abs(z_load)):
            # Open-load limit -> Z_in = Z0 / (j·tan).
            if abs(tan_term) > 1.0e-30:
                z_load = z0 / complex(0.0, tan_term)
            else:
                z_load = complex(0.0, 0.0)
        else:
            denom = z0 + 1j * z_load * tan_term
            if abs(denom) <= 1.0e-300:
                # Numerically singular — treat as open to avoid divide-by-zero.
                z_load = complex("inf")
                continue
            z_load = z0 * (z_load + 1j * z0 * tan_term) / denom

    if not math.isfinite(abs(z_load)) or abs(z_load) <= 1.0e-300:
        # Pathological — return the natural limit so the caller's
        # tridiagonal stays well-conditioned.
        return complex(0.0, 0.0) if not math.isfinite(abs(z_load)) else complex(float("inf"), 0.0)
    return complex(1.0 / z_load)


def Y_stub_at_freq_with_shunts(
    omega: float,
    stub_segments: Sequence[StubSegment],
    substrate: Mapping[str, float],
    *,
    termination: object = "short",
    mid_shunt_loads: Mapping[int, object] | None = None,
) -> complex:
    """Stub-cascade input admittance with shunt loads at INTERMEDIATE nodes.

    :func:`Y_stub_at_freq` walks a strictly *series* input-impedance recursion:
    one far-end load, one chain of :class:`StubSegment`. It therefore cannot
    represent a stub that is grounded (or otherwise loaded) part-way along --
    e.g. the fabricated SMV1245 board, whose 3-section OPEN/MLEF stub carries a
    ``VIAGND`` at each of its two intermediate section junctions. Those shunts
    are simply invisible to the series tracer, which is why that board's branch
    admittance has had to be supplied from an ADS-tabulated ``.s1p`` override.

    This function is the additive generalisation. It performs the identical
    inward recursion, and additionally shunts a load onto the running input
    impedance whenever it arrives at a loaded intermediate node. With
    ``mid_shunt_loads`` empty or ``None`` it reduces to :func:`Y_stub_at_freq`
    exactly (same arithmetic, same order), so it is a strict superset.

    Node numbering. For ``K`` segments ordered from the varactor outward, the
    nodes are ``n_0`` (the varactor terminal, where ``Y_stub`` is presented),
    ``n_1 … n_{K-1}`` (the section junctions), and ``n_K`` (the far end, where
    ``termination`` sits)::

        n_0 --seg[0]-- n_1 --seg[1]-- n_2 -- … --seg[K-1]-- n_K = termination
                        |              |
                    shunt[1]       shunt[2]

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s]; the same negative-frequency convention as
        :func:`Y_stub_at_freq` applies (Hammerstad-Kirschning at ``|omega|``,
        sign carried by ``tan(beta*l)``).
    stub_segments : sequence of :class:`StubSegment`
        Stub cascade ordered from the varactor outward.
    substrate : Mapping[str, float]
        Substrate dict (``"eps_r"``, ``"h_m"``, optional ``"t_m"``).
    termination : str | complex | Callable[[float], complex], optional
        Far-end load at ``n_K``; see :func:`Y_stub_at_freq`.
    mid_shunt_loads : Mapping[int, TerminationImpedance], optional
        Shunt load impedances keyed by intermediate node index ``i`` in
        ``1 … K-1``. Each value takes the same forms as ``termination``: the
        strings ``"short"`` / ``"open"``, a complex constant, or a callable
        ``omega -> Z_shunt(omega)`` -- so a physical via's
        ``ViaModel.analytical_termination(...)`` can be passed straight in.
        Indices outside ``1 … K-1`` raise ``ValueError`` (index ``0`` would
        short the presentation plane and index ``K`` is the far-end
        ``termination``, which has its own argument).

    Returns
    -------
    complex
        ``Y_stub = 1 / Z_in`` looking into the cascade at ``n_0``.
    """

    omega_val = float(omega)
    segments = list(stub_segments)
    if not segments:
        raise ValueError("stub_segments must contain at least one segment.")

    shunts: dict[int, object] = dict(mid_shunt_loads or {})
    n_seg = len(segments)
    for idx in shunts:
        if not 1 <= int(idx) <= n_seg - 1:
            raise ValueError(
                f"mid_shunt_loads index {idx!r} outside the intermediate-node "
                f"range 1..{n_seg - 1} for a {n_seg}-segment stub (index 0 is "
                f"the presentation plane; index {n_seg} is `termination`)."
            )

    if not shunts:
        # Byte-identical fall-through to the original series recursion.
        return Y_stub_at_freq(
            omega_val, segments, substrate, termination=termination,
        )

    def _shunt_admittance(node_index: int) -> complex:
        """``Y_shunt`` at an intermediate node (0 when the node is unloaded)."""
        if node_index not in shunts:
            return complex(0.0, 0.0)
        z_sh = _resolve_termination(shunts[node_index], omega_val)
        if not math.isfinite(abs(z_sh)):
            return complex(0.0, 0.0)          # open shunt -> no load
        if abs(z_sh) <= 1.0e-300:
            return complex(float("inf"), 0.0)  # shorted node
        return 1.0 / z_sh

    if omega_val == 0.0:
        # DC limit. tan(beta*l) -> 0 collapses every segment to a through
        # connection, so the whole chain reduces to the termination in parallel
        # with each loaded intermediate node.
        y_dc = complex(0.0, 0.0)
        z_load_dc = _resolve_termination(termination, 0.0)
        if math.isfinite(abs(z_load_dc)):
            if abs(z_load_dc) <= 1.0e-300:
                return complex(float("inf"), 0.0)
            y_dc += 1.0 / z_load_dc
        for i in range(1, n_seg):
            y_sh = _shunt_admittance(i)
            if not math.isfinite(abs(y_sh)):
                return complex(float("inf"), 0.0)
            y_dc += y_sh
        return y_dc

    frequency_hz = abs(omega_val) / (2.0 * math.pi)

    # Walk inward from the far end, applying each intermediate shunt as the
    # recursion arrives at its node.
    z_load = _resolve_termination(termination, omega_val)

    for seg_index in range(n_seg - 1, -1, -1):
        segment = segments[seg_index]
        z0, v_phase = _z0_and_v_phase(segment.width_m, substrate, frequency_hz)
        beta = omega_val / v_phase
        tan_term = math.tan(beta * float(segment.length_m))
        if not math.isfinite(abs(z_load)):
            if abs(tan_term) > 1.0e-30:
                z_load = z0 / complex(0.0, tan_term)
            else:
                z_load = complex(0.0, 0.0)
        else:
            denom = z0 + 1j * z_load * tan_term
            if abs(denom) <= 1.0e-300:
                z_load = complex("inf")
            else:
                z_load = z0 * (z_load + 1j * z0 * tan_term) / denom

        # ``seg_index`` was just traversed, so the recursion now sits at node
        # ``seg_index`` -- an intermediate node for every index >= 1.
        if seg_index >= 1:
            y_sh = _shunt_admittance(seg_index)
            if y_sh != 0:
                if not math.isfinite(abs(y_sh)):
                    z_load = complex(0.0, 0.0)     # node shorted to ground
                    continue
                y_line = (
                    complex(0.0, 0.0)
                    if not math.isfinite(abs(z_load))
                    else (complex(float("inf"), 0.0)
                          if abs(z_load) <= 1.0e-300
                          else 1.0 / z_load)
                )
                if not math.isfinite(abs(y_line)):
                    z_load = complex(0.0, 0.0)
                    continue
                y_tot = y_line + y_sh
                z_load = (complex("inf") if abs(y_tot) <= 1.0e-300
                          else 1.0 / y_tot)

    if not math.isfinite(abs(z_load)) or abs(z_load) <= 1.0e-300:
        return (complex(0.0, 0.0) if not math.isfinite(abs(z_load))
                else complex(float("inf"), 0.0))
    return complex(1.0 / z_load)


def extract_stub_segments(
    spec: DeviceSpec,
    cell_index: int,
) -> tuple[tuple[StubSegment, ...], Mapping[str, float], str]:
    """Return the cell-k stub MLIN cascade + substrate dict + termination.

    Walks the spec's element list to find the cell's varactor (SDD2P)
    and the series chain reaching from its bottom pin to a ground node,
    accumulating one :class:`StubSegment` per MLIN encountered. I_Probe
    elements in the chain are transparent (zero series impedance) and
    are stepped past with the net pointer advanced. The walk handles
    arbitrary cascade lengths — Level-9 has one MLIN segment in the
    chain, Level-10 has three — so the same extractor serves every
    Level-9-shaped spec without modification.

    Returns
    -------
    tuple
        ``(segments, substrate_dict, termination_string)``. ``segments``
        is the cascade ordered from the varactor outward (the natural
        order to consume in the lifted Floquet's inward recursion).
        ``substrate_dict`` is a shallow copy of
        :attr:`DeviceSpec.substrate.parameters`. ``termination_string``
        is ``"short"`` (the chain ends at a ``ground_nodes`` entry); the
        Level-10 path provides a frequency-dependent inductive load via
        a separate callable, not via this helper.

    Note
    ----
    Originally named ``extract_level9_stub_segments`` (when
    Level-9 was the only client). Renamed to reflect its
    generic role; the old name is kept as a deprecated alias so the
    original tests are unchanged.
    """

    sdd_elements = [e for e in spec.elements if e.type == "SDD2P"]
    if not 0 <= cell_index < len(sdd_elements):
        raise IndexError(
            f"cell_index {cell_index} out of range for {len(sdd_elements)} "
            f"SDD2P element(s) in spec {spec.label!r}."
        )
    sdd = sdd_elements[cell_index]

    # The SDD2P's pin 2 (port-1- net) is the entry into the stub chain.
    sdd_bot = None
    for pin in sdd.pins:
        if int(pin.pin) == 2:
            sdd_bot = pin.node
            break
    if sdd_bot is None:
        raise ValueError(
            f"SDD2P element {sdd.id!r} has no pin 2 — cannot trace stub."
        )

    # Build an adjacency map of nets to series elements so we can walk
    # the chain. An MLIN connects two nets and contributes a stub
    # segment; an I_Probe also connects two nets but with no impedance
    # (transparently skipped); a lumped ``L`` / ``R`` / ``C`` is recognised so
    # the walker steps past a via-to-ground network (a Level-10 paper-geometry
    # lumped ``L`` via, or the ``RlcVia`` series ``R``->``L``->ground +
    # shunt-pad ``C``) and still reaches the ground node -> ``"short"``. Without
    # ``R`` / ``C`` the walker would stop at the RlcVia's series ``R`` (the
    # ``L``-to-ground is one hop behind it) and mis-classify the SHORT cell as an
    # open dead-end. The via element values are recovered separately by
    # :func:`extract_series_via_inductance_H` / the via model's
    # ``analytical_termination`` and folded in by :func:`build_stub_y_fn_from_spec`;
    # this walk only collects the MLIN segments and classifies the termination.
    # (Below the SDD's pin 2 the only lumped R/L/C are the via network -- the
    # package ``L_pkg`` / ``R_series`` sit above the SDD and are unreachable from
    # pin 2 -- so recognising R/C here cannot wander off the stub-to-ground path.)
    ground_nodes = set(spec.ground_nodes)

    def _other_net(element, this_net: str) -> str | None:
        for pin in element.pins:
            if pin.node != this_net:
                return pin.node
        return None

    current_net = sdd_bot
    segments: list[StubSegment] = []
    consumed_ids: set[str] = {sdd.id}
    visited_nets: set[str] = set()
    termination = "short"  # default: the chain reaches a ground node
    while current_net not in ground_nodes:
        if current_net in visited_nets:
            raise ValueError(
                f"Stub-trace loop at net {current_net!r}; spec {spec.label!r} "
                f"cell {cell_index} may be malformed."
            )
        visited_nets.add(current_net)
        # Find the next series element on this net (MLIN, I_Probe, or lumped
        # L/R/C) that we have not yet consumed. Series elements have exactly
        # two pins; a junction (MTEE etc.) would have three+ and is
        # not expected on the stub chain.
        next_element = None
        for element in spec.elements:
            if element.id in consumed_ids:
                continue
            if element.type not in ("MLIN", "I_Probe", "L", "R", "C"):
                continue
            net_ids = [pin.node for pin in element.pins]
            if current_net in net_ids:
                next_element = element
                break
        if next_element is None:
            # Model-convention addition: a ``VIAGND`` is an internally-grounded
            # 1-port (ADS physical via). The stub chain reaches ground *through* it
            # rather than at a ``ground_nodes`` net, so a VIAGND on the current net is
            # a SHORT termination (matching the LumpedLVia ``L``-to-ground case), not an
            # open dead-end. VIAGND is a new element type, so this branch is unreachable
            # for any pre-existing spec (the classification is purely additive).
            via_to_ground = any(
                e.type == "VIAGND" and current_net in {p.node for p in e.pins}
                for e in spec.elements
            )
            if via_to_ground:
                termination = "short"
                break
            # Convention: a degree-1 non-ground leaf is an
            # open-circuited stub end, not a malformed spec. Earlier
            # callers (Level-9/10) only emit via-to-ground / inductive chains,
            # which always reach a ``ground_nodes`` entry and never enter this
            # branch -- so their behaviour is unchanged (still returns
            # ``"short"``). The OPEN-termination topology families
            # terminate here; return ``"open"`` so the caller models an open
            # far end. A dead-end with *no* segments collected is still a
            # genuine malformed-spec error.
            if segments:
                termination = "open"
                break
            raise ValueError(
                f"Stub chain dead-end at net {current_net!r}; spec "
                f"{spec.label!r} cell {cell_index} (no MLIN/I_Probe/L from the "
                f"SDD toward an open end or ground)."
            )
        other = _other_net(next_element, current_net)
        if other is None:
            raise ValueError(
                f"Element {next_element.id!r} has no opposite-side net "
                f"from {current_net!r}."
            )
        consumed_ids.add(next_element.id)
        if next_element.type == "MLIN":
            w = float(next_element.parameters["w_m"])
            l_seg = float(next_element.parameters["l_m"])
            segments.append(StubSegment(width_m=w, length_m=l_seg))
        # I_Probe is transparent — just advance the net pointer.
        current_net = other

    if not segments:
        raise ValueError(
            f"No stub MLIN found between SDD2P pin 2 and ground in spec "
            f"{spec.label!r} cell {cell_index}."
        )

    return tuple(segments), dict(spec.substrate.parameters), termination


# Deprecated alias kept for the original tests and external callers. The
# new generic name ``extract_stub_segments`` is preferred for new code.
extract_level9_stub_segments = extract_stub_segments


def extract_series_via_inductance_H(
    spec: DeviceSpec, cell_index: int
) -> float:
    """Total series-L inductance between the cell's SDD2P pin 2 and ground.

    Walks the same SDD2P → ground chain as :func:`extract_stub_segments`
    and accumulates the inductance value (``l_h``) of every series ``L``
    element it steps past. For Level-9 (no via inductance modelled in the
    spec) this returns ``0.0``. For Level-10 (one or more parallel vias
    to ground modelled by a single lumped L element with the
    parallel-combined Bahl value) this returns the spec's emitted
    inductance, which the analytical-side termination then mirrors
    exactly. This keeps the analytical lift and the ADS netlist
    referencing the same physical L_via value.
    """

    sdd_elements = [e for e in spec.elements if e.type == "SDD2P"]
    if not 0 <= cell_index < len(sdd_elements):
        raise IndexError(
            f"cell_index {cell_index} out of range for {len(sdd_elements)} "
            f"SDD2P element(s) in spec {spec.label!r}."
        )
    sdd = sdd_elements[cell_index]
    sdd_bot = None
    for pin in sdd.pins:
        if int(pin.pin) == 2:
            sdd_bot = pin.node
            break
    if sdd_bot is None:
        return 0.0

    ground_nodes = set(spec.ground_nodes)
    current_net = sdd_bot
    consumed_ids: set[str] = {sdd.id}
    visited_nets: set[str] = set()
    total_l_h = 0.0
    while current_net not in ground_nodes:
        if current_net in visited_nets:
            return total_l_h  # malformed: don't spin
        visited_nets.add(current_net)
        next_element = None
        for element in spec.elements:
            if element.id in consumed_ids:
                continue
            if element.type not in ("MLIN", "I_Probe", "L"):
                continue
            if current_net in {pin.node for pin in element.pins}:
                next_element = element
                break
        if next_element is None:
            return total_l_h
        if next_element.type == "L":
            total_l_h += float(next_element.parameters.get("l_h", 0.0))
        consumed_ids.add(next_element.id)
        other_net = None
        for pin in next_element.pins:
            if pin.node != current_net:
                other_net = pin.node
                break
        if other_net is None:
            return total_l_h
        current_net = other_net
    return float(total_l_h)


def build_stub_y_fn_from_spec(
    spec: DeviceSpec,
    cell_index: int,
    *,
    via_inductance_H: float | None = None,
    termination_override: Callable[[float], complex] | None = None,
) -> Callable[[float], complex]:
    """Build a frequency-to-admittance callable for the cell's stub chain.

    The returned function is ``omega -> Y_stub(omega)`` with the stub
    geometry and substrate captured by closure. It is the natural
    argument to pass as ``y_stub_fn`` into
    :func:`tvtl.floquet.harmonics.build_stub_tridiagonal`.

    Parameters
    ----------
    spec : DeviceSpec
        The Level-9-shaped cascade spec carrying the cell's stub MLIN
        chain.
    cell_index : int
        0-based cell index.
    via_inductance_H : float, optional
        When supplied, the stub terminates at an inductive load
        ``Z_via(omega) = j*omega*via_inductance_H`` rather than at a
        pure short. This is the Level-10 paper-geometry path: the stub
        terminates at one or more parallel vias to the ground plane,
        each modelled by the Bahl formula in
        :func:`tvtl.microstrip.discontinuities.via_inductance_bahl_H`,
        and the parallel combination's effective inductance is passed
        here. When ``None`` (default) the helper auto-detects via
        inductance from any series ``L`` elements in the cell's
        SDD2P → ground chain via
        :func:`extract_series_via_inductance_H` -- so a Level-10 spec
        with an emitted ``L_VIA_C{k}`` lumped inductor produces the
        matching inductive termination automatically. The Level-9 path
        leaves this ``None`` and finds no series L, so the existing
        short-termination behaviour is unchanged.

    Returns
    -------
    Callable[[float], complex]
        ``omega -> Y_stub(omega)`` with the stub geometry, substrate,
        and termination captured by closure.
    """

    segments, substrate, default_termination = extract_stub_segments(
        spec, cell_index,
    )

    if termination_override is not None:
        # Model-registry path: a via / open-end model supplies the
        # far-end ``Z_load(omega)`` directly (e.g. an RLC via, or an MLEF open-end
        # fringe), from the same parameters the emitter rendered. When no model is
        # given (or the default lumped_L / ideal models, whose analytical_termination
        # returns None) this branch is skipped and the byte-identical spec-extraction
        # path below runs unchanged.
        termination: object = termination_override
    elif via_inductance_H is None:
        auto_l = extract_series_via_inductance_H(spec, cell_index)
        l_via = float(auto_l)
        if l_via < 0.0:
            raise ValueError(
                f"via_inductance_H must be non-negative (got {l_via})."
            )
        if l_via == 0.0:
            termination = default_termination
        else:
            l_via_captured = l_via
            termination = lambda omega: complex(0.0, float(omega) * l_via_captured)
    else:
        l_via = float(via_inductance_H)
        if l_via < 0.0:
            raise ValueError(
                f"via_inductance_H must be non-negative (got {l_via})."
            )
        if l_via == 0.0:
            termination = default_termination
        else:
            # An inductive load is ``Z_load(omega) = j * omega * L``. The
            # callable form lets ``Y_stub_at_freq`` evaluate it at each
            # Floquet harmonic frequency.
            l_via_captured = l_via
            termination = lambda omega: complex(0.0, float(omega) * l_via_captured)

    def _y_fn(omega: float) -> complex:
        return Y_stub_at_freq(
            omega, segments, substrate, termination=termination,
        )

    return _y_fn


def build_stub_y_fn_with_mid_shunts(
    spec: DeviceSpec,
    cell_index: int,
    *,
    mid_shunt_loads: Mapping[int, object] | None = None,
    via_inductance_H: float | None = None,
    termination_override: Callable[[float], complex] | None = None,
) -> Callable[[float], complex]:
    """Like :func:`build_stub_y_fn_from_spec`, plus intermediate-node shunts.

    The far-end termination is resolved exactly as
    :func:`build_stub_y_fn_from_spec` resolves it (``termination_override`` →
    explicit ``via_inductance_H`` → auto-detected series ``L`` → the tracer's
    ``"short"``/``"open"`` classification), and the segments come from the same
    :func:`extract_stub_segments` walk. The only addition is
    ``mid_shunt_loads``, forwarded to :func:`Y_stub_at_freq_with_shunts`.

    This is the analytical counterpart of a stub grounded at its *section
    junctions* -- the fabricated SMV1245 board's two mid-stub ``VIAGND``. Those
    elements are spliced into the emitted spec at ``n_c{k}_s1`` / ``n_c{k}_s2``
    and are invisible to the series tracer, so the caller states them here, e.g.

    .. code-block:: python

        z_via = ViagndVia(via_radius_m=0.8e-3).analytical_termination()
        y_fn = build_stub_y_fn_with_mid_shunts(
            spec, 0, mid_shunt_loads={1: z_via, 2: z_via},
            termination_override=MlefOpen().analytical_termination(
                width_m=w_last, substrate=sub, f_ref_Hz=f_signal),
        )

    With ``mid_shunt_loads`` empty or ``None`` the returned callable is
    numerically identical to :func:`build_stub_y_fn_from_spec`'s.
    """

    segments, substrate, default_termination = extract_stub_segments(
        spec, cell_index,
    )

    if termination_override is not None:
        termination: object = termination_override
    else:
        if via_inductance_H is None:
            l_via = float(extract_series_via_inductance_H(spec, cell_index))
        else:
            l_via = float(via_inductance_H)
        if l_via < 0.0:
            raise ValueError(
                f"via_inductance_H must be non-negative (got {l_via})."
            )
        if l_via == 0.0:
            termination = default_termination
        else:
            l_via_captured = l_via
            termination = lambda omega: complex(0.0, float(omega) * l_via_captured)

    shunts = dict(mid_shunt_loads or {})

    def _y_fn(omega: float) -> complex:
        return Y_stub_at_freq_with_shunts(
            omega, segments, substrate, termination=termination,
            mid_shunt_loads=shunts,
        )

    return _y_fn


