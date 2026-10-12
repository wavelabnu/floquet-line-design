"""SDD2P polynomial-content cross-check -- the SDD2P load-bearing gate.

The structural cross-check (:mod:`netlist_compare`) proves the OA and
hpeesofsim netlists describe the *same circuit topology* and that their
shared parameters agree as text. It does **not** prove that an
``SDD2P``'s body equation is the charge polynomial the device spec
asked for. That gap is exactly how v1's silent-physics bug survived:
v1's SDD2P body was left at the ADS default ``(_v1)/50.0`` -- a 50-ohm
passive resistor -- so the "parametric amplifier" had no nonlinear
element in it at all, and the cross-check of the day did not look
inside the F-equation.

This module closes that gap. For every ``SDD2P`` element in a device
spec it:

1. parses BOTH port equations -- the port-1 charge equation
   ``I[1,1]=+Q(_v1)`` and the port-2 KCL-closure mirror
   ``I[2,1]=-Q(_v1)`` -- out of each backend's netlist into structured
   ``{power: coefficient}`` maps, and flags any other ``I[k,m]`` entry
   as a non-whitelisted equation (a weight-0 ``I[2,0]`` is the known
   port-2 bug pattern);
2. checks each backend's port-1 polynomial against the spec's
   ``q_poly_coefficients``, checks each backend's port-2 polynomial is
   the exact negation of its port-1, and checks the two backends
   against each other -- all within a floating-point tolerance;
3. fails loudly -- with the spec coefficients, the backend
   coefficients and the element name -- on any mismatch.

It also explicitly recognises v1-style bug shapes:

* a body that is a bare non-zero constant (e.g. ``1/50`` evaluated, or
  a placeholder) -> passive-element / placeholder pattern;
* a body that divides by a constant (the literal ``(_v1)/50.0``
  default) -> passive-element pattern;
* a missing port-1 charge equation (only a weight-0 ``I[1,0]``, or
  nothing) -> empty/default F-equation;
* a body referencing an undefined/unexpected variable -> template
  placeholder pattern.

The polynomial is the charge polynomial ``Q(v) = q0 + q1*v + q2*v**2
+ ...`` (coulombs / V**n); ``q0`` is always ``0`` (the validator
enforces ``Q(0) = 0``), so a clean body carries only ``v**k`` terms
with ``k >= 1``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..device_spec import DeviceSpec, Element
from ..netlist.elements.sdd2p import sdd2p_charge_coefficients

# Default floating-point tolerance for coefficient comparison.
#
# The operative gate is the *relative* tolerance: a charge coefficient
# survives the emit path (``{:.16g}`` format -> string -> ``float()``)
# essentially exactly -- a double round-trips through 16 significant
# digits -- so a real coefficient error is many orders of magnitude
# larger than emit noise, and 1e-9 relative catches every meaningful
# error while never tripping on representation noise.
#
# The absolute tolerance is only a noise floor for comparing against a
# coefficient that is *exactly zero* (where the relative tolerance gives
# a zero threshold). It is deliberately set FAR below the charge-
# coefficient scale: the SMV1245 charge coefficients are themselves
# O(1e-12) C/V**n, so an absolute floor anywhere near 1e-12 would swamp
# the coefficients and make the check blind to picocoulomb-scale errors
# (a 2x error on a 1e-12 coefficient would read as "close"). 1e-21 sits
# ~9 orders below the smallest physically meaningful coefficient -- it
# tightens, never loosens, the comparison relative to a 1e-12 floor.
DEFAULT_ABS_TOL = 1.0e-21
DEFAULT_REL_TOL = 1.0e-9

# One polynomial term: ``(<number>)*_v1`` or ``(<number>)*_v1**<power>``.
_TERM_RE = re.compile(
    r"\(\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)\s*\*\s*_v1"
    r"(?:\s*\*\*\s*(\d+))?"
)
# An identifier token, used to spot a foreign/undefined variable in the
# unparsed remainder of a body.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# ``I[port,weight]=`` markers inside an SDD netlist line.
_EQ_MARKER_RE = re.compile(r"I\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*=")

# The only (port, weight) equations a 2-terminal SDD2P varactor body may
# carry: port 1's charge equation and port 2's KCL-closure mirror, both
# at weighting index 1 (-> dQ/dt). Any other entry -- notably a weight-0
# I[2,0] -- is the known port-2 bug pattern and is flagged.
_WHITELISTED_EQUATIONS = frozenset({(1, 1), (2, 1)})


@dataclass(frozen=True)
class ParsedSDDBody:
    """The parsed port-1 charge equation of one SDD2P netlist line.

    ``coefficients`` maps a power of ``_v1`` to its coefficient; it is
    populated only for a body that parses as a polynomial. ``ok`` is
    true only when the body is a clean charge polynomial in ``_v1``
    (possibly the all-zero polynomial). ``diagnostic`` explains why a
    non-clean body is not a charge polynomial.
    """

    raw: str
    coefficients: dict[int, float]
    ok: bool
    diagnostic: str


@dataclass
class SDD2PContentResult:
    """Outcome of :func:`check_sdd2p_content`.

    ``match`` is true only when every SDD2P element's polynomial agrees
    with the spec on every checked backend and the backends agree with
    each other. ``failures`` is the loud, human-readable list of every
    disagreement (empty when ``match`` is true). ``checked`` lists the
    SDD2P element ids that were examined.
    """

    match: bool
    failures: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _logical_lines(text: str) -> list[str]:
    """Join ``\\``-continued physical lines into logical statements."""

    statements: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buffer += line[:-1].rstrip() + " "
            continue
        buffer += line
        statements.append(buffer.strip())
        buffer = ""
    if buffer.strip():
        statements.append(buffer.strip())
    return statements


def _find_sdd_line(netlist_text: str, element_id: str) -> str | None:
    """Return the ``SDD:<element_id>`` logical line, or ``None``.

    Matches the netlist element token ``SDD`` (what both backends emit)
    with the instance name equal to ``element_id``.
    """

    for statement in _logical_lines(netlist_text):
        head = statement.split(None, 1)[0] if statement else ""
        if ":" not in head:
            continue
        token, _, name = head.partition(":")
        if token.upper() == "SDD" and name == element_id:
            return statement
    return None


def _equations(sdd_line: str) -> dict[tuple[int, int], str]:
    """Return ``{(port, weight): expression}`` from an SDD netlist line.

    Each ``I[port,weight]=expression`` runs until the next ``I[..]=``
    marker or the end of the line, so a body that (wrongly) contains
    spaces is still captured whole.
    """

    markers = list(_EQ_MARKER_RE.finditer(sdd_line))
    equations: dict[tuple[int, int], str] = {}
    for index, marker in enumerate(markers):
        port = int(marker.group(1))
        weight = int(marker.group(2))
        start = marker.end()
        end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(sdd_line)
        )
        equations[(port, weight)] = sdd_line[start:end].strip()
    return equations


def parse_charge_polynomial_expression(expr: str) -> ParsedSDDBody:
    """Parse an SDD body string into a structured charge polynomial.

    A clean body is a sum of ``(<number>)*_v1**k`` terms (``k >= 1``);
    the all-zero body ``0`` is also clean (it is the zero polynomial).
    Anything else -- a bare non-zero constant, a division, a foreign
    variable, unparseable junk -- is reported ``ok = False`` with a
    diagnostic naming the v1-style bug pattern.
    """

    text = (expr or "").strip()
    if not text:
        return ParsedSDDBody(
            raw=text,
            coefficients={},
            ok=False,
            diagnostic=(
                "empty/default F-equation detected -- the port-1 charge "
                "equation has no expression"
            ),
        )

    # A whole-expression numeric constant: the zero polynomial is fine;
    # any other constant is a fixed-value placeholder, not a charge law.
    try:
        constant = float(text)
    except ValueError:
        constant = None
    if constant is not None:
        if constant == 0.0:
            return ParsedSDDBody(text, {}, True, "")
        return ParsedSDDBody(
            raw=text,
            coefficients={0: constant},
            ok=False,
            diagnostic=(
                f"constant body detected -- placeholder/passive-element "
                f"pattern: the body {text!r} is a fixed-value current "
                f"source, not a voltage-dependent charge polynomial"
            ),
        )

    coefficients: dict[int, float] = {}
    covered = [False] * len(text)
    for match in _TERM_RE.finditer(text):
        coeff = float(match.group(1))
        power = int(match.group(2)) if match.group(2) else 1
        coefficients[power] = coefficients.get(power, 0.0) + coeff
        for index in range(match.start(), match.end()):
            covered[index] = True

    remainder = "".join(ch for index, ch in enumerate(text) if not covered[index])
    significant = re.sub(r"[\s+]", "", remainder)
    if significant:
        if "/" in significant:
            return ParsedSDDBody(
                raw=text,
                coefficients=coefficients,
                ok=False,
                diagnostic=(
                    f"placeholder/passive-element pattern detected -- the "
                    f"body {text!r} divides by a constant (a fixed I = V/R "
                    f"resistor form, e.g. the ADS default (_v1)/50.0), not "
                    f"a charge polynomial"
                ),
            )
        foreign = [
            token
            for token in _IDENT_RE.findall(remainder)
            if token not in ("e", "E", "_v1")
        ]
        if foreign:
            return ParsedSDDBody(
                raw=text,
                coefficients=coefficients,
                ok=False,
                diagnostic=(
                    f"template placeholder pattern detected -- the body "
                    f"{text!r} references the undefined/unexpected "
                    f"token(s) {sorted(set(foreign))}; a charge polynomial "
                    f"may use only the port-1 voltage _v1"
                ),
            )
        return ParsedSDDBody(
            raw=text,
            coefficients=coefficients,
            ok=False,
            diagnostic=(
                f"non-polynomial body detected -- {text!r} could not be "
                f"parsed as a charge polynomial in _v1"
            ),
        )

    if not coefficients:
        return ParsedSDDBody(
            raw=text,
            coefficients={},
            ok=False,
            diagnostic=(
                f"non-polynomial body detected -- no _v1 charge terms "
                f"found in {text!r}"
            ),
        )
    return ParsedSDDBody(text, coefficients, True, "")


def parse_sdd_body_from_netlist(
    netlist_text: str, element_id: str
) -> ParsedSDDBody | None:
    """Parse one SDD2P element's port-1 charge body out of a netlist.

    Returns ``None`` when the netlist carries no ``SDD:<element_id>``
    line at all. When the line is present but its port-1 charge
    equation ``I[1,1]`` is missing, returns a non-ok :class:`ParsedSDDBody`
    whose diagnostic names the empty/default-F-equation pattern (this is
    v1's other failure shape: a weight-0 ``I[1,0]`` current source, or
    no port-1 equation at all).
    """

    sdd_line = _find_sdd_line(netlist_text, element_id)
    if sdd_line is None:
        return None
    equations = _equations(sdd_line)
    charge = equations.get((1, 1))
    if charge is not None:
        return parse_charge_polynomial_expression(charge)
    if (1, 0) in equations:
        return ParsedSDDBody(
            raw=sdd_line,
            coefficients={},
            ok=False,
            diagnostic=(
                "empty/default F-equation detected -- the SDD line carries "
                "a weight-0 port-1 equation I[1,0] (a passive current-source "
                "form, e.g. the ADS default I[1,0]=(_v1)/50.0) but no "
                "weight-1 charge equation I[1,1]; a charge-driven varactor "
                "must use I[1,1]=Q(_v1)"
            ),
        )
    return ParsedSDDBody(
        raw=sdd_line,
        coefficients={},
        ok=False,
        diagnostic=(
            "empty/default F-equation detected -- the SDD line carries no "
            "port-1 charge equation I[1,1]"
        ),
    )


def expected_charge_coefficients(element: Element) -> dict[int, float]:
    """Return the spec's charge polynomial as ``{power: coefficient}``.

    Built from the SDD2P element's ``q_poly_coefficients`` parameter
    ``(q0, q1, q2, ...)`` -- ``q0`` (the ``v**0`` term) is included as
    ``0.0`` since the validator enforces it.
    """

    return {
        power: coeff
        for power, coeff in enumerate(sdd2p_charge_coefficients(element))
    }


def _coefficient_diffs(
    expected: dict[int, float],
    actual: dict[int, float],
    *,
    abs_tol: float,
    rel_tol: float,
) -> list[str]:
    """Return a per-power list of disagreements (empty when they agree).

    A power present in only one map is compared against an implicit
    ``0.0`` in the other.
    """

    diffs: list[str] = []
    for power in sorted(set(expected) | set(actual)):
        want = float(expected.get(power, 0.0))
        got = float(actual.get(power, 0.0))
        if not math.isclose(want, got, rel_tol=rel_tol, abs_tol=abs_tol):
            diffs.append(f"v**{power}: {want:.9g} vs {got:.9g}")
    return diffs


def check_sdd2p_content(
    spec: DeviceSpec,
    *,
    hpeesofsim_netlist_text: str | None = None,
    ads_netlist_text: str | None = None,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> SDD2PContentResult:
    """Verify every SDD2P element's polynomial content against the spec.

    For each SDD2P element and each supplied backend this checks:

    * the SDD line carries only the whitelisted equations -- ``I[1,1]``
      (port-1 charge) and ``I[2,1]`` (port-2 mirror); any other
      ``I[k,m]`` (notably a weight-0 ``I[2,0]``) is flagged;
    * the port-1 polynomial ``I[1,1]`` matches the spec's
      ``q_poly_coefficients``;
    * the port-2 polynomial ``I[2,1]`` is the exact negation of that
      backend's port-1 polynomial (the KCL closure ``I[2,1]=-Q(_v1)``);
    * the two backends agree on both port-1 and port-2.

    Parameters
    ----------
    spec : DeviceSpec
        The device spec -- the source of truth for the expected
        ``q_poly_coefficients`` of every SDD2P element.
    hpeesofsim_netlist_text, ads_netlist_text : str, optional
        The two backends' netlist texts. At least one should be given;
        when both are given the two are also checked against each other.
    abs_tol, rel_tol : float
        Floating-point tolerance for coefficient comparison.

    Returns
    -------
    SDD2PContentResult
        ``match`` is true only when every check above holds for every
        SDD2P element on every supplied backend. Every disagreement is
        recorded in ``failures`` with the element name and the offending
        coefficients.
    """

    backends: dict[str, str] = {}
    if hpeesofsim_netlist_text is not None:
        backends["hpeesofsim"] = hpeesofsim_netlist_text
    if ads_netlist_text is not None:
        backends["OA/ADS"] = ads_netlist_text

    failures: list[str] = []
    notes: list[str] = []
    checked: list[str] = []

    sdd_elements = [e for e in spec.elements if e.type == "SDD2P"]
    if not sdd_elements:
        notes.append("spec has no SDD2P elements -- content check is a no-op")
        return SDD2PContentResult(match=True, notes=notes)
    if not backends:
        failures.append(
            "no netlist text supplied -- cannot verify SDD2P polynomial "
            "content"
        )
        return SDD2PContentResult(match=False, failures=failures, notes=notes)

    for element in sdd_elements:
        checked.append(element.id)
        try:
            expected_p1 = expected_charge_coefficients(element)
        except ValueError as exc:
            failures.append(f"SDD2P '{element.id}': {exc}")
            continue

        # Per-backend parsed port bodies -- only the cleanly-parsed ones
        # are stored, for the cross-backend and mirror comparisons.
        port1: dict[str, ParsedSDDBody] = {}
        port2: dict[str, ParsedSDDBody] = {}

        for backend_name, text in backends.items():
            sdd_line = _find_sdd_line(text, element.id)
            if sdd_line is None:
                failures.append(
                    f"SDD2P '{element.id}' [{backend_name}]: the netlist "
                    f"carries no SDD line for this element"
                )
                continue
            equations = _equations(sdd_line)

            # (a) no non-whitelisted equations.
            unexpected = sorted(
                key for key in equations if key not in _WHITELISTED_EQUATIONS
            )
            if unexpected:
                pretty = ", ".join(f"I[{p},{w}]" for p, w in unexpected)
                failures.append(
                    f"SDD2P '{element.id}' [{backend_name}]: unexpected "
                    f"non-whitelisted equation(s) {pretty} -- an SDD2P "
                    f"varactor body must carry only I[1,1] (port-1 charge) "
                    f"and I[2,1] (port-2 KCL-closure mirror); a weight-0 "
                    f"entry such as I[2,0] is the known port-2 bug "
                    f"pattern"
                )

            # (b) port 1 -- I[1,1] -- must match the device spec.
            p1_expr = equations.get((1, 1))
            if p1_expr is None:
                failures.append(
                    f"SDD2P '{element.id}' [{backend_name}]: port-1 charge "
                    f"equation I[1,1] is missing -- empty/default "
                    f"F-equation (a charge-driven varactor must carry "
                    f"I[1,1]=Q(_v1))"
                )
            else:
                body1 = parse_charge_polynomial_expression(p1_expr)
                if not body1.ok:
                    failures.append(
                        f"SDD2P '{element.id}' [{backend_name}] port 1: "
                        f"{body1.diagnostic}"
                    )
                else:
                    port1[backend_name] = body1
                    diffs = _coefficient_diffs(
                        expected_p1, body1.coefficients,
                        abs_tol=abs_tol, rel_tol=rel_tol,
                    )
                    if diffs:
                        failures.append(
                            f"SDD2P '{element.id}' [{backend_name}] port 1: "
                            f"emitted polynomial disagrees with the device "
                            f"spec on {len(diffs)} power(s) -- "
                            f"{'; '.join(diffs)} (spec vs netlist)"
                        )

            # (c) port 2 -- I[2,1] -- must be the negation of port 1.
            p2_expr = equations.get((2, 1))
            if p2_expr is None:
                failures.append(
                    f"SDD2P '{element.id}' [{backend_name}]: port-2 charge "
                    f"equation I[2,1] is missing -- a 2-terminal SDD2P "
                    f"varactor must carry the port-2 KCL-closure mirror "
                    f"I[2,1]=-Q(_v1)"
                )
            else:
                body2 = parse_charge_polynomial_expression(p2_expr)
                if not body2.ok:
                    failures.append(
                        f"SDD2P '{element.id}' [{backend_name}] port 2: "
                        f"{body2.diagnostic}"
                    )
                else:
                    port2[backend_name] = body2
                    if backend_name in port1:
                        negated_port1 = {
                            power: -coeff
                            for power, coeff
                            in port1[backend_name].coefficients.items()
                        }
                        diffs = _coefficient_diffs(
                            negated_port1, body2.coefficients,
                            abs_tol=abs_tol, rel_tol=rel_tol,
                        )
                        if diffs:
                            failures.append(
                                f"SDD2P '{element.id}' [{backend_name}]: "
                                f"port-2 polynomial I[2,1] is not the "
                                f"negation of port-1 I[1,1] -- the KCL "
                                f"closure requires I[2,1]=-Q(_v1); "
                                f"disagreement on {len(diffs)} power(s) -- "
                                f"{'; '.join(diffs)} (-port1 vs port2)"
                            )

        # Cross-backend: the two backends must emit the same polynomials.
        if "hpeesofsim" in port1 and "OA/ADS" in port1:
            diffs = _coefficient_diffs(
                port1["hpeesofsim"].coefficients,
                port1["OA/ADS"].coefficients,
                abs_tol=abs_tol, rel_tol=rel_tol,
            )
            if diffs:
                failures.append(
                    f"SDD2P '{element.id}': the hpeesofsim and OA/ADS "
                    f"backends emit different port-1 polynomials on "
                    f"{len(diffs)} power(s) -- {'; '.join(diffs)} "
                    f"(hpeesofsim vs OA/ADS)"
                )
        if "hpeesofsim" in port2 and "OA/ADS" in port2:
            diffs = _coefficient_diffs(
                port2["hpeesofsim"].coefficients,
                port2["OA/ADS"].coefficients,
                abs_tol=abs_tol, rel_tol=rel_tol,
            )
            if diffs:
                failures.append(
                    f"SDD2P '{element.id}': the hpeesofsim and OA/ADS "
                    f"backends emit different port-2 polynomials on "
                    f"{len(diffs)} power(s) -- {'; '.join(diffs)} "
                    f"(hpeesofsim vs OA/ADS)"
                )

    return SDD2PContentResult(
        match=not failures,
        failures=failures,
        checked=checked,
        notes=notes,
    )


def format_content_report(result: SDD2PContentResult) -> str:
    """Render a human-readable SDD2P content-check report."""

    lines: list[str] = []
    verdict = "MATCH" if result.match else "MISMATCH"
    lines.append(f"SDD2P polynomial content check: {verdict}")
    lines.append("=" * 44)
    if result.checked:
        lines.append(f"checked element(s): {', '.join(result.checked)}")
    if result.failures:
        lines.append(f"\n{len(result.failures)} failure(s):")
        lines += [f"  - {item}" for item in result.failures]
    else:
        lines.append("\nevery SDD2P polynomial matches the device spec.")
    if result.notes:
        lines += [f"note: {note}" for note in result.notes]
    return "\n".join(lines) + "\n"


