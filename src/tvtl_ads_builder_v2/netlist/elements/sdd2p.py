"""Emit an ``SDD2P`` symbolically-defined-device line for hpeesofsim.

``SDD2P`` is the ADS native Symbolically Defined Device, 2-port. The
public builder uses it to represent a nonlinear varactor as a charge-driven device: the
port-1 charge polynomial ``Q(_v1)`` is supplied as a weighted current
equation so that the port-1 current is ``dQ/dt``.

ADS conventions used by the public emitter:

* netlist element token is ``SDD`` (not ``SDD2P``);
* the device is a genuine 2-port -> 4 terminals
  ``[port1+, port1-, port2+, port2-]``;
* a body equation ``I[p,w]=expr`` sets port ``p``'s current; weighting
  index ``w = 0`` means the expression *is* the current, ``w = 1`` means
  the expression is differentiated in time. So ``I[1,1]=Q(_v1)`` makes
  port-1 current ``i1 = dQ/dt`` -- the charge-driven varactor;
* ``_v1`` is port-1 voltage referenced to port 1's own negative
  terminal.

For a 2-terminal varactor port 1 carries the charge polynomial and
port 2 is the KCL closure: port 2's current is the mirror image of
port 1's -- ``I[2,1]=-Q(_v1)``, the negation of port 1's expression at
the same weighting index 1. Port 2's two terminals are tied to a
single shared return net, so the mirror current circulates within the
(shorted) port and is electrically self-consistent (a port driven to a
non-zero current must not terminate on isolated one-pin nets).

The emitted line therefore is::

    SDD:<id> <p1> <p2> <p3> <p4> I[1,1]=+Q(_v1) I[2,1]=-Q(_v1)

with ``<p3>`` and ``<p4>`` the same shared return net.

The polynomial text is produced by :func:`format_charge_polynomial`,
which both this hpeesofsim emitter and the OA emitter call -- the two
backends emit a byte-identical body so the netlist cross-check and the
polynomial content cross-check stay meaningful. Port 2's expression is
the same formatter applied to the sign-negated coefficients.
"""

from __future__ import annotations

from typing import Callable, Sequence

from ...device_spec import Element

# The device-spec parameter key carrying the charge polynomial.
Q_POLY_COEFFICIENTS_KEY = "q_poly_coefficients"
Q_POLY_REFERENCE_KEY = "q_poly_reference"

# Time-varying-capacitor (control-port) parameter keys. When ``tvc_c0_f`` is
# present the SDD is an ideal externally-pumped linear capacitor (the
# design-cross-validation device): port 1 is the in-line shunt and port 2 the
# control port driven by a pump source.
TVC_C0_KEY = "tvc_c0_f"
TVC_K_KEY = "tvc_k_f_per_v"

# Control-port *realistic* time-varying capacitor.
# When ``tvc_poly_coefficients`` is present the SDD imposes the realistic
# varactor's capacitance-vs-pump Taylor law on the control port: the port-1
# charge is ``Q1 = (sum_k a_k * _v2^k) * _v1`` so the time-varying capacitance
# is ``C(t) = C(V_bias + v2(t))`` (``a_k`` the capacitance Taylor coeffs,
# ``a0`` the static C incl. the CP fold). The bilinear ``tvc_c0_f`` / tvc_k
# form above is the 2-term special case. Port 2 is still the zero-current
# voltage-sense control port.
TVC_POLY_KEY = "tvc_poly_coefficients"

# The SDD DSL variables for port-1 / port-2 voltages.
_PORT1_VOLTAGE = "_v1"
_PORT2_VOLTAGE = "_v2"


def _is_tvc(element: Element) -> bool:
    """True iff the SDD2P element is in bilinear time-varying-capacitor mode."""

    return TVC_C0_KEY in element.parameters


def _is_tvc_poly(element: Element) -> bool:
    """True iff the SDD2P element is in control-port realistic-poly mode."""

    return TVC_POLY_KEY in element.parameters


def sdd2p_tvc_poly_coefficients(element: Element) -> tuple[float, ...]:
    """Return the control-port capacitance Taylor coefficients ``[a0, a1, ...]``."""

    raw = element.parameters.get(TVC_POLY_KEY)
    if raw is None:
        raise ValueError(
            f"SDD2P '{element.id}' is missing the required parameter "
            f"'{TVC_POLY_KEY}'."
        )
    return tuple(float(value) for value in raw)


def sdd2p_tvc_poly_body_rows(element: Element) -> list[list[str]]:
    """Return the SDD body rows for a control-port *realistic* time-varying cap.

    The port-1 charge is ``Q1 = (a0 + a1*_v2 + a2*_v2^2 + ...) * _v1`` (weight
    1 -> ``i1 = dQ1/dt``), realising ``C(t) = C(V_bias + v2(t))`` driven by the
    control-port pump ``v2``; port 2 is the zero-current voltage-sense control
    port (``I[2,0] = 0``). With ``v2 = Vp*cos(wm*t + phi)`` this is the
    realistic varactor's capacitance modulation with full C-V curvature, the
    pump imposed directly on the junction (bypassing the package divider).
    """

    coeffs = sdd2p_tvc_poly_coefficients(element)
    terms: list[str] = []
    for k, a in enumerate(coeffs):
        if a == 0.0:
            continue
        if k == 0:
            terms.append(f"({a:.16g})*{_PORT1_VOLTAGE}")
        elif k == 1:
            terms.append(f"({a:.16g})*{_PORT2_VOLTAGE}*{_PORT1_VOLTAGE}")
        else:
            terms.append(
                f"({a:.16g})*{_PORT2_VOLTAGE}**{k}*{_PORT1_VOLTAGE}"
            )
    q_pos = "+".join(terms) if terms else "0"
    return [["1", "1", q_pos], ["2", "0", "0"]]


def sdd2p_tvc_body_rows(element: Element) -> list[list[str]]:
    """Return the SDD body rows for a control-port time-varying capacitor.

    The realised device is ``C(t) = C0 + K*v2(t)`` with port-1 charge
    ``Q1 = C0*_v1 + K*_v2*_v1`` (weight 1 -> ``i1 = dQ1/dt``) and port 2 a
    zero-current voltage-sense control port (``I[2,0] = 0``). With the control
    port driven by a pump ``v2 = Vp*cos(wm*t + phi)`` this is exactly the
    capacitance-linear ideal time-varying capacitor
    ``C(t) = C0 + K*Vp*cos(wm*t + phi)`` (modulation depth ``K*Vp/C0``).
    """

    c0 = float(element.parameters[TVC_C0_KEY])
    k = float(element.parameters[TVC_K_KEY])
    q_pos = (
        f"({c0:.16g})*{_PORT1_VOLTAGE}"
        f"+({k:.16g})*{_PORT2_VOLTAGE}*{_PORT1_VOLTAGE}"
    )
    return [["1", "1", q_pos], ["2", "0", "0"]]


def sdd2p_charge_coefficients(element: Element) -> tuple[float, ...]:
    """Return an SDD2P element's charge-polynomial coefficients as floats.

    The device-spec stores ``q_poly_coefficients`` as a tuple of floats
    ``(q0, q1, q2, ...)`` where ``Q(v) = q0 + q1*v + q2*v**2 + ...``; a
    JSON round-trip turns the tuple into a list, so this coerces either
    form to a ``tuple[float, ...]``.
    """

    if element.type != "SDD2P":
        raise ValueError(
            f"sdd2p_charge_coefficients called on element of type "
            f"'{element.type}'."
        )
    raw = element.parameters.get(Q_POLY_COEFFICIENTS_KEY)
    if raw is None:
        raise ValueError(
            f"SDD2P '{element.id}' is missing the required parameter "
            f"'{Q_POLY_COEFFICIENTS_KEY}'."
        )
    return tuple(float(value) for value in raw)


def format_charge_polynomial(coefficients: Sequence[float]) -> str:
    """Render a charge polynomial ``Q(_v1)`` as an SDD body expression.

    ``coefficients`` is ``(q0, q1, q2, ...)`` -- the coefficient of
    ``v**k`` at index ``k``, in coulombs / V**k. The rendered text is

        ``(q1)*_v1+(q2)*_v1**2+(q3)*_v1**3+...``

    deliberately **space-free** so it is a single netlist token (a netlist
    parser splits on whitespace) and so the hpeesofsim and OA backends
    emit a byte-identical body. ``q0`` contributes no term (the charge
    polynomial passes through the origin); a coefficient of exactly
    ``0.0`` is skipped. An all-zero polynomial renders as ``0``.

    This is the single source of truth for the SDD body text -- both
    SDD2P emitters call it.
    """

    terms: list[str] = []
    for power, coeff in enumerate(coefficients):
        if power == 0:
            # q0 is the v**0 term; the charge polynomial has Q(0) = 0.
            continue
        value = float(coeff)
        if value == 0.0:
            continue
        v_token = _PORT1_VOLTAGE if power == 1 else f"{_PORT1_VOLTAGE}**{power}"
        terms.append(f"({value:.16g})*{v_token}")
    return "+".join(terms) if terms else "0"


def negate_coefficients(coefficients: Sequence[float]) -> tuple[float, ...]:
    """Return the sign-negated charge-polynomial coefficients.

    Port 2's mirror expression ``-Q(_v1)`` is ``Q(_v1)`` with the sign
    of every coefficient flipped; this produces the coefficient tuple
    for that mirror.
    """

    return tuple(-float(value) for value in coefficients)


def sdd2p_body_rows(element: Element) -> list[list[str]]:
    """Return the SDD repeated-parameter rows for an SDD2P element.

    Each row is ``[port, weight, expression]`` and the ADS netlister
    emits it as ``I[port,weight]=expression``. Port 1 carries the
    charge-driven varactor (``I[1,1]=+Q(_v1)``, weight 1 -> dQ/dt);
    port 2 is the KCL closure carrying the mirror current
    (``I[2,1]=-Q(_v1)``, the sign-negated polynomial at the same
    weighting index). The OA emitter assigns this list straight to the
    instance's ``SDD`` repeated parameter.
    """

    if _is_tvc_poly(element):
        return sdd2p_tvc_poly_body_rows(element)
    if _is_tvc(element):
        return sdd2p_tvc_body_rows(element)
    coefficients = sdd2p_charge_coefficients(element)
    q_pos = format_charge_polynomial(coefficients)
    q_neg = format_charge_polynomial(negate_coefficients(coefficients))
    return [["1", "1", q_pos], ["2", "1", q_neg]]


def emit_sdd2p(
    element: Element, resolve_node: Callable[[str], str] = lambda node: node
) -> str:
    """Return a single-line ``SDD`` directive for an ``SDD2P`` element.

    Parameters
    ----------
    element : Element
        Must have ``type == "SDD2P"`` and exactly four pins
        ``[port1+, port1-, port2+, port2-]``.
    resolve_node : callable
        Maps a device-spec node id to the netlist token (translates a
        ground node to the literal ``0``). Defaults to the identity.
    """

    if element.type != "SDD2P":
        raise ValueError(
            f"emit_sdd2p called on element of type '{element.type}'."
        )
    if len(element.pins) != 4:
        raise ValueError(
            f"SDD2P '{element.id}' must declare exactly four pins "
            f"(got {len(element.pins)})."
        )
    by_pin = {pin.pin: pin.node for pin in element.pins}
    if set(by_pin) != {1, 2, 3, 4}:
        raise ValueError(
            f"SDD2P '{element.id}' must declare pins 1, 2, 3 and 4 "
            f"(got {sorted(by_pin)})."
        )
    nets = " ".join(resolve_node(by_pin[k]) for k in (1, 2, 3, 4))
    if _is_tvc_poly(element):
        rows = sdd2p_tvc_poly_body_rows(element)
        body = " ".join(f"I[{p},{w}]={expr}" for p, w, expr in rows)
        return f"SDD:{element.id} {nets} {body}"
    if _is_tvc(element):
        rows = sdd2p_tvc_body_rows(element)
        body = " ".join(f"I[{p},{w}]={expr}" for p, w, expr in rows)
        return f"SDD:{element.id} {nets} {body}"
    coefficients = sdd2p_charge_coefficients(element)
    q_pos = format_charge_polynomial(coefficients)
    q_neg = format_charge_polynomial(negate_coefficients(coefficients))
    return f"SDD:{element.id} {nets} I[1,1]={q_pos} I[2,1]={q_neg}"


