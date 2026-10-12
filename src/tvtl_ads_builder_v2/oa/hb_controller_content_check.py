"""HB controller content cross-check (single- and two-tone).

The structural cross-check (:mod:`netlist_compare`) proves the OA and
hpeesofsim netlists describe the same circuit topology. It does not, on
its own, prove that the harmonic-balance controller carries the
analysis parameters the device spec asked for.

This module closes that gap. For a device spec carrying an
``hb_single_tone`` directive it parses the ``HB:`` controller line out
of each backend's netlist and checks:

1. each backend's ``Freq[1]`` / ``Order[1]`` / ``MaxOrder`` matches the
   spec's :class:`HBSingleToneSpec` (within a floating-point tolerance);
2. the two backends agree with each other on those parameters;
3. no second tone leaked in -- a ``Freq[2]`` / ``Order[2]`` on a
   single-tone controller is a two-tone artifact and is flagged.

The hpeesofsim ``HB:`` syntax is ``HB:HB1 Freq[1]=<f> GHz Order[1]=<n>
MaxOrder=<n> ...``; ``Freq[1]`` carries an explicit frequency unit,
which is normalised to Hz for comparison against the spec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..device_spec import DeviceSpec, resolve_spec

# Floating-point tolerance for parameter comparison -- the same gate the
# SDD2P and VAR content checks use.
DEFAULT_ABS_TOL = 1.0e-21
DEFAULT_REL_TOL = 1.0e-9

# Frequency-unit suffixes ADS may write after a Freq value, in Hz.
_FREQ_UNIT_HZ = {
    "Hz": 1.0,
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
    "THz": 1.0e12,
}


@dataclass
class HBContentResult:
    """Outcome of :func:`check_hb_controller_content`.

    ``match`` is true only when every backend's HB controller agrees
    with the spec and the backends agree with each other. ``failures``
    is the loud, human-readable list of disagreements.
    """

    match: bool
    failures: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _find_hb_line(netlist_text: str) -> str | None:
    """Return the ``HB:`` controller line of a netlist, or ``None``.

    Skips comment lines (``;`` / ``//``) so a commented-out
    ``; HarmonicBalance:HB1 ...`` review line is never mistaken for the
    real directive.
    """

    for raw in netlist_text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("//"):
            continue
        head = line.split(None, 1)[0]
        token, _, _name = head.partition(":")
        if token == "HB":
            return line
    return None


def parse_hb_line(line: str) -> dict[str, float]:
    """Parse an ``HB:`` line into ``{parameter: value}``.

    Every ``KEY=<number>`` token is recorded. A ``Freq[*]`` value is
    normalised to Hz using the frequency-unit token that follows it
    (``GHz`` etc.); a value with no unit token is taken as-is. A
    non-numeric assignment (``SweepVar="..."``) is skipped.
    """

    tokens = line.split()
    params: dict[str, float] = {}
    index = 1  # token 0 is the ``HB:<name>`` head
    while index < len(tokens):
        token = tokens[index]
        if "=" in token:
            key, _, raw_value = token.partition("=")
            unit_factor = 1.0
            if index + 1 < len(tokens) and tokens[index + 1] in _FREQ_UNIT_HZ:
                unit_factor = _FREQ_UNIT_HZ[tokens[index + 1]]
                index += 1
            try:
                value = float(raw_value)
            except ValueError:
                index += 1
                continue
            params[key] = value * unit_factor if key.startswith("Freq") else value
        index += 1
    return params


def check_hb_controller_content(
    spec: DeviceSpec,
    *,
    hpeesofsim_netlist_text: str | None = None,
    ads_netlist_text: str | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> HBContentResult:
    """Verify a spec's single-tone HB controller content.

    Parameters
    ----------
    spec : DeviceSpec
        The device spec -- the source of truth for the
        ``hb_single_tone`` directive.
    hpeesofsim_netlist_text, ads_netlist_text : str, optional
        The two backends' netlist texts. When both are given the
        controller parameters are checked against each other; either
        may be omitted.
    rel_tol, abs_tol : float
        Floating-point tolerance for parameter comparison.

    Returns
    -------
    HBContentResult
        ``match`` is true only when every supplied backend's HB
        controller agrees with the spec and with the other backend.
    """

    failures: list[str] = []
    notes: list[str] = []

    # Resolve any VAR block so a named-reference frequency is compared
    # as the substituted literal -- the same literal both backends
    # emit. A no-op for a spec with no VAR block.
    spec = resolve_spec(spec)
    hb = spec.hb_single_tone
    if hb is None:
        notes.append(
            "spec has no HB controller -- HB content check is a no-op"
        )
        return HBContentResult(match=True, notes=notes)

    expected = {
        "Freq[1]": float(hb.fundamental_freq),
        "Order[1]": float(hb.fundamental_order),
        "MaxOrder": float(hb.max_order),
    }

    backends: dict[str, str] = {}
    if hpeesofsim_netlist_text is not None:
        backends["hpeesofsim"] = hpeesofsim_netlist_text
    if ads_netlist_text is not None:
        backends["OA/ADS"] = ads_netlist_text
    if not backends:
        failures.append(
            "no netlist text supplied -- cannot verify HB controller content"
        )
        return HBContentResult(match=False, failures=failures, notes=notes)

    parsed: dict[str, dict[str, float]] = {}
    for backend_name, text in backends.items():
        line = _find_hb_line(text)
        if line is None:
            failures.append(
                f"[{backend_name}]: the netlist carries no HB: controller line"
            )
            continue
        params = parse_hb_line(line)
        parsed[backend_name] = params

        for key, want in expected.items():
            if key not in params:
                failures.append(
                    f"[{backend_name}]: HB controller is missing parameter "
                    f"{key!r}"
                )
            elif not math.isclose(
                params[key], want, rel_tol=rel_tol, abs_tol=abs_tol
            ):
                failures.append(
                    f"[{backend_name}]: HB controller {key} = "
                    f"{params[key]:.9g}, spec expects {want:.9g}"
                )
        # A single-tone controller must not carry a second tone.
        for stray in ("Freq[2]", "Order[2]"):
            if stray in params:
                failures.append(
                    f"[{backend_name}]: single-tone HB controller carries "
                    f"{stray!r} -- the spec declares one tone"
                )

    if "hpeesofsim" in parsed and "OA/ADS" in parsed:
        hp = parsed["hpeesofsim"]
        oa = parsed["OA/ADS"]
        for key in expected:
            if key in hp and key in oa and not math.isclose(
                hp[key], oa[key], rel_tol=rel_tol, abs_tol=abs_tol
            ):
                failures.append(
                    f"backend cross-check: HB controller {key} -- "
                    f"hpeesofsim {hp[key]:.9g} vs OA/ADS {oa[key]:.9g}"
                )

    return HBContentResult(
        match=not failures,
        failures=failures,
        checked=["HB1"],
        notes=notes,
    )


def check_hb_two_tone_content(
    spec: DeviceSpec,
    *,
    hpeesofsim_netlist_text: str | None = None,
    ads_netlist_text: str | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> HBContentResult:
    """Verify a spec's two-tone HB controller content.

    Checks both fundamentals (``Freq[1]`` / ``Freq[2]``), both per-tone
    orders (``Order[1]`` / ``Order[2]``) and ``MaxOrder`` against the
    spec's ``hb_two_tone`` directive on each supplied backend, and the
    backends against each other. The ``HB:``-line parsing
    (:func:`parse_hb_line`) is shared with the single-tone check.
    """

    failures: list[str] = []
    notes: list[str] = []

    # Resolve any VAR block so a named-reference frequency is compared
    # as the substituted literal. A no-op for a spec with no VAR block.
    spec = resolve_spec(spec)
    hb = spec.hb_two_tone
    if hb is None:
        notes.append(
            "spec has no two-tone HB controller -- HB content check is a "
            "no-op"
        )
        return HBContentResult(match=True, notes=notes)

    expected = {
        "Freq[1]": float(hb.fundamental_freq_1),
        "Freq[2]": float(hb.fundamental_freq_2),
        "Order[1]": float(hb.fundamental_order_1),
        "Order[2]": float(hb.fundamental_order_2),
        "MaxOrder": float(hb.max_order),
    }

    backends: dict[str, str] = {}
    if hpeesofsim_netlist_text is not None:
        backends["hpeesofsim"] = hpeesofsim_netlist_text
    if ads_netlist_text is not None:
        backends["OA/ADS"] = ads_netlist_text
    if not backends:
        failures.append(
            "no netlist text supplied -- cannot verify HB controller content"
        )
        return HBContentResult(match=False, failures=failures, notes=notes)

    parsed: dict[str, dict[str, float]] = {}
    for backend_name, text in backends.items():
        line = _find_hb_line(text)
        if line is None:
            failures.append(
                f"[{backend_name}]: the netlist carries no HB: controller line"
            )
            continue
        params = parse_hb_line(line)
        parsed[backend_name] = params

        for key, want in expected.items():
            if key not in params:
                failures.append(
                    f"[{backend_name}]: HB controller is missing parameter "
                    f"{key!r}"
                )
            elif not math.isclose(
                params[key], want, rel_tol=rel_tol, abs_tol=abs_tol
            ):
                failures.append(
                    f"[{backend_name}]: HB controller {key} = "
                    f"{params[key]:.9g}, spec expects {want:.9g}"
                )
        # A two-tone controller must not carry a third tone.
        for stray in ("Freq[3]", "Order[3]"):
            if stray in params:
                failures.append(
                    f"[{backend_name}]: two-tone HB controller carries "
                    f"{stray!r} -- the spec declares two tones"
                )

    if "hpeesofsim" in parsed and "OA/ADS" in parsed:
        hp = parsed["hpeesofsim"]
        oa = parsed["OA/ADS"]
        for key in expected:
            if key in hp and key in oa and not math.isclose(
                hp[key], oa[key], rel_tol=rel_tol, abs_tol=abs_tol
            ):
                failures.append(
                    f"backend cross-check: HB controller {key} -- "
                    f"hpeesofsim {hp[key]:.9g} vs OA/ADS {oa[key]:.9g}"
                )

    return HBContentResult(
        match=not failures,
        failures=failures,
        checked=["HB1"],
        notes=notes,
    )


def format_content_report(result: HBContentResult) -> str:
    """Render a human-readable HB controller content-check report."""

    lines: list[str] = []
    verdict = "MATCH" if result.match else "MISMATCH"
    lines.append(f"HB controller content check: {verdict}")
    lines.append("=" * 38)
    if result.checked:
        lines.append(f"checked controller(s): {', '.join(result.checked)}")
    if result.failures:
        lines.append(f"\n{len(result.failures)} failure(s):")
        lines += [f"  - {item}" for item in result.failures]
    else:
        lines.append(
            "\nthe HB controller matches the device spec on every backend."
        )
    if result.notes:
        lines += [f"note: {note}" for note in result.notes]
    return "\n".join(lines) + "\n"


