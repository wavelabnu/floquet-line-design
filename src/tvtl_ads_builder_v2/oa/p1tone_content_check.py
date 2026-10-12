"""``P_1Tone`` power-source content cross-check.

The structural cross-check (:mod:`netlist_compare`) proves the OA and
hpeesofsim netlists describe the same routed topology. It does not
verify the ``P_1Tone`` power sources: a P_1Tone source is an analysis
stimulus, not a routed two-terminal device, so the comparator routes
it to ``ignored`` -- exactly as it does the ``HB:`` controller line.

This module closes that gap, mirroring the HB controller content
check. For a device spec carrying ``sources`` it parses every P_1Tone
source emission out of each backend's netlist and checks:

1. each source named in the spec appears, on each backend, with the
   right driven node, ground return, ``Num``, ``Z``, ``Freq`` and
   ``P`` (power);
2. the two backends agree with each other on every source.

Named references resolve first: :func:`check_p1tone_content` calls
:func:`resolve_spec`, so a ``freq`` / ``power_dBm`` that named a VAR
variable is compared as the substituted literal -- the cross-backend
invariant the named-reference lift must hold.

A P_1Tone source reaches a netlist in two different forms, and the
parser accepts both:

* the **hpeesofsim** flat emitter writes v1's confirmed line::

      P_1Tone:<name> <pos> 0 Num=<n> Z=<ohms> Ohm Freq=<f> <unit> P=polar(dbmtow(<dBm>),0) Noise=no

* **ADS 2026** normalises the OA ``ads_sources/P_1Tone`` cell to a
  ``Port:`` line with array-indexed tone parameters (live probe
  evidence)::

      Port:<name> <pos> 0 Num=<n> Z=<ohms> Ohm P[1]=polar(dbmtow(<dBm>),0) Freq[1]=<f> <unit> Noise=no Pac=polar(dbmtow(0),0)

Both carry the same content; only the element token and the
``Freq``/``Freq[1]``, ``P``/``P[1]`` spellings differ. ``Port:`` also
serves as the *sink* output-port termination token, so the parser
classifies a ``Port:`` line as source-or-sink by its parameter
signature -- a source carries a tone-power parameter, a sink does not.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..device_spec import DeviceSpec, resolve_spec
from .netlist_compare import _net_isomorphism, _pair_elements, parse_netlist_text

# Floating-point tolerance -- the same gate the SDD2P / VAR / HB content
# checks use.
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

# The hpeesofsim ground-reference token for a source's return terminal.
_GROUND_TOKEN = "0"

# Extracts the dBm argument out of ADS's ``polar(dbmtow(<dBm>),0)``
# power wrapper.
_DBMTOW_RE = re.compile(r"dbmtow\(\s*([+\-0-9.eE]+)\s*\)")

# Strips a trailing array index -- ``P[1]`` -> ``P``, ``Freq[1]`` ->
# ``Freq``. ADS 2026 normalises the OA ads_sources/P_1Tone cell to a
# ``Port:`` line whose tone parameters are array-indexed; the
# hpeesofsim ``P_1Tone:`` form writes them unindexed. Stripping the
# index makes the two forms compare on equal keys.
_INDEX_SUFFIX_RE = re.compile(r"\[\d+\]\Z")


@dataclass
class P1ToneContentResult:
    """Outcome of :func:`check_p1tone_content`.

    ``match`` is true only when every source named in the spec agrees
    with the spec on every supplied backend and the backends agree with
    each other. ``failures`` is the loud, human-readable disagreement
    list; ``checked`` lists the source names examined.
    """

    match: bool
    failures: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ParsedSource:
    """One parsed P_1Tone source emission (``P_1Tone:`` or ``Port:``)."""

    name: str
    pos_node: str
    neg_node: str
    num: float | None
    z_ohm: float | None
    freq_Hz: float | None
    power_dBm: float | None


def _parse_source_emission(line: str) -> _ParsedSource | None:
    """Parse a P_1Tone source emission, or return ``None``.

    Recognises both backend forms of a P_1Tone power source:

    * hpeesofsim writes it as a ``P_1Tone:`` line (v1's confirmed
      pattern);
    * ADS 2026 normalises the OA ``ads_sources/P_1Tone`` cell to a
      ``Port:`` line carrying a tone-power parameter (``P[1]``) and an
      array-indexed ``Freq[1]`` -- discovered by a live emission
      probe.

    Returns ``None`` when the line is neither -- in particular for a
    *sink* ``Port:`` (an output-port termination: ``Num`` / ``Z`` /
    ``Noise`` with no tone-power parameter). The source/sink decision
    is by parameter signature, not the element token, because ``Port:``
    serves both roles.

    The two leading bare tokens are the positive and negative nodes;
    ``Num`` / ``Z`` / ``Freq`` (or ``Freq[1]``) carry numeric values
    (``Freq`` is normalised to Hz via the trailing unit token); ``P``
    (or ``P[1]``) is the tone-power wrapper, from which the ``dbmtow``
    dBm argument is extracted. ``Pac`` -- the AC small-signal power ADS
    adds to the normalised form -- is not a tone parameter and is
    ignored.
    """

    tokens = line.split()
    if not tokens:
        return None
    token, _, name = tokens[0].partition(":")
    if token not in ("P_1Tone", "Port"):
        return None
    rest = tokens[1:]

    nodes: list[str] = []
    index = 0
    while index < len(rest) and "=" not in rest[index]:
        nodes.append(rest[index])
        index += 1
    pos_node = nodes[0] if len(nodes) > 0 else ""
    neg_node = nodes[1] if len(nodes) > 1 else ""

    num: float | None = None
    z_ohm: float | None = None
    freq_Hz: float | None = None
    power_dBm: float | None = None
    has_tone_power = False
    while index < len(rest):
        raw_token = rest[index]
        if "=" in raw_token:
            raw_key, _, raw_value = raw_token.partition("=")
            key = _INDEX_SUFFIX_RE.sub("", raw_key)
            unit_factor = 1.0
            if index + 1 < len(rest) and rest[index + 1] in _FREQ_UNIT_HZ:
                unit_factor = _FREQ_UNIT_HZ[rest[index + 1]]
                index += 1
            elif index + 1 < len(rest) and rest[index + 1] == "Ohm":
                index += 1
            if key == "P":
                has_tone_power = True
                hit = _DBMTOW_RE.search(raw_value)
                if hit is not None:
                    power_dBm = float(hit.group(1))
            elif key in ("Num", "Z", "Freq"):
                try:
                    value = float(raw_value)
                except ValueError:
                    index += 1
                    continue
                if key == "Num":
                    num = value
                elif key == "Z":
                    z_ohm = value
                else:  # Freq / Freq[1]
                    freq_Hz = value * unit_factor
            # every other key (Noise, Pac, ...) is not a tone parameter
        index += 1

    # A ``Port:`` line with no tone-power parameter is a sink -- an
    # output-port termination -- not a source; exclude it.
    if token == "Port" and not has_tone_power:
        return None
    return _ParsedSource(
        name=name,
        pos_node=pos_node,
        neg_node=neg_node,
        num=num,
        z_ohm=z_ohm,
        freq_Hz=freq_Hz,
        power_dBm=power_dBm,
    )


def find_p1tone_lines(netlist_text: str) -> dict[str, _ParsedSource]:
    """Return ``{source name: parsed source}`` for a netlist's P_1Tone sources.

    Recognises both the hpeesofsim ``P_1Tone:`` form and ADS's
    OA-normalised ``Port:`` form (see :func:`_parse_source_emission`); a
    *sink* ``Port:`` (an output-port termination) is correctly
    excluded. Comment lines (``;`` / ``//``) are skipped so a
    commented-out review line is never mistaken for a real directive.
    """

    found: dict[str, _ParsedSource] = {}
    for raw in netlist_text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("//"):
            continue
        parsed = _parse_source_emission(line)
        if parsed is not None:
            found[parsed.name] = parsed
    return found


def check_p1tone_content(
    spec: DeviceSpec,
    *,
    hpeesofsim_netlist_text: str | None = None,
    ads_netlist_text: str | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> P1ToneContentResult:
    """Verify a spec's ``P_1Tone`` sources against the emitted netlists.

    Parameters
    ----------
    spec : DeviceSpec
        The device spec -- the source of truth for ``sources``. Passed
        *unresolved*; the check resolves it so a ``freq`` / ``power_dBm``
        that named a VAR variable is compared as the substituted
        literal.
    hpeesofsim_netlist_text, ads_netlist_text : str, optional
        The two backends' netlist texts. When both are given the
        sources are checked against each other; either may be omitted.
    rel_tol, abs_tol : float
        Floating-point tolerance for parameter comparison.

    Returns
    -------
    P1ToneContentResult
        ``match`` is true only when every spec source agrees with every
        supplied backend and the backends agree with each other.
    """

    failures: list[str] = []
    notes: list[str] = []

    resolved = resolve_spec(spec)
    if not resolved.sources:
        notes.append("spec has no P_1Tone sources -- content check is a no-op")
        return P1ToneContentResult(match=True, notes=notes)

    expected = {
        source.name: source for source in resolved.sources
    }

    backends: dict[str, str] = {}
    if hpeesofsim_netlist_text is not None:
        backends["hpeesofsim"] = hpeesofsim_netlist_text
    if ads_netlist_text is not None:
        backends["OA/ADS"] = ads_netlist_text
    if not backends:
        failures.append(
            "no netlist text supplied -- cannot verify P_1Tone sources"
        )
        return P1ToneContentResult(match=False, failures=failures, notes=notes)

    parsed_by_backend: dict[str, dict[str, _ParsedSource]] = {
        name: find_p1tone_lines(text) for name, text in backends.items()
    }

    # ADS auto-renames every named net to ``N__nn`` when it recomputes a
    # source-bearing circuit's connectivity from wire geometry, so the
    # OA/ADS netlist drives a source from e.g. ``N__23`` where the spec
    # says ``n_input``. Reuse the structural audit's net isomorphism --
    # the same A->B mapping Path A's compare_circuits builds via
    # _net_isomorphism -- to translate an OA/ADS net name back to the
    # spec net it corresponds to. A topology-preserving rename then
    # compares equal, while a source on a genuinely different net still
    # maps to a different spec name and is still caught. Built only when
    # both backends are supplied and the mapping is conflict-free (a
    # one-to-one isomorphism); otherwise the literal net name is kept.
    oa_to_spec_net: dict[str, str] = {}
    if hpeesofsim_netlist_text is not None and ads_netlist_text is not None:
        net_map, conflicts = _net_isomorphism(
            _pair_elements(
                parse_netlist_text(ads_netlist_text).elements,
                parse_netlist_text(hpeesofsim_netlist_text).elements,
            )
        )
        if not conflicts:
            oa_to_spec_net = net_map

    def _to_spec_net(node: str, backend: str) -> str:
        """An OA/ADS net name mapped to its spec net via the audit
        isomorphism; an hpeesofsim name already is a spec net name."""

        if backend == "OA/ADS":
            return oa_to_spec_net.get(node, node)
        return node

    for backend_name, parsed in parsed_by_backend.items():
        for source_name, source in expected.items():
            got = parsed.get(source_name)
            if got is None:
                failures.append(
                    f"[{backend_name}]: no P_1Tone: line for source "
                    f"{source_name!r}"
                )
                continue
            if _to_spec_net(got.pos_node, backend_name) != source.port_node:
                failures.append(
                    f"[{backend_name}]: source {source_name!r} drives node "
                    f"{got.pos_node!r}, spec expects {source.port_node!r}"
                )
            if got.neg_node != _GROUND_TOKEN:
                failures.append(
                    f"[{backend_name}]: source {source_name!r} return "
                    f"terminal is {got.neg_node!r}, expected ground "
                    f"{_GROUND_TOKEN!r}"
                )
            _compare_number(
                failures, backend_name, source_name, "Num",
                got.num, float(source.num), rel_tol, abs_tol,
            )
            _compare_number(
                failures, backend_name, source_name, "Z",
                got.z_ohm, float(source.z_ohm), rel_tol, abs_tol,
            )
            _compare_number(
                failures, backend_name, source_name, "Freq",
                got.freq_Hz, float(source.freq), rel_tol, abs_tol,
            )
            _compare_number(
                failures, backend_name, source_name, "P (dBm)",
                got.power_dBm, float(source.power_dBm), rel_tol, abs_tol,
            )

    if "hpeesofsim" in parsed_by_backend and "OA/ADS" in parsed_by_backend:
        hp = parsed_by_backend["hpeesofsim"]
        oa = parsed_by_backend["OA/ADS"]
        for source_name in expected:
            if source_name not in hp or source_name not in oa:
                continue
            for label, hv, ov in (
                ("Num", hp[source_name].num, oa[source_name].num),
                ("Z", hp[source_name].z_ohm, oa[source_name].z_ohm),
                ("Freq", hp[source_name].freq_Hz, oa[source_name].freq_Hz),
                ("P (dBm)", hp[source_name].power_dBm,
                 oa[source_name].power_dBm),
            ):
                if hv is None or ov is None:
                    continue
                if not math.isclose(hv, ov, rel_tol=rel_tol, abs_tol=abs_tol):
                    failures.append(
                        f"backend cross-check: source {source_name!r} "
                        f"{label} -- hpeesofsim {hv:.9g} vs OA/ADS {ov:.9g}"
                    )
            if hp[source_name].pos_node != _to_spec_net(
                oa[source_name].pos_node, "OA/ADS"
            ):
                failures.append(
                    f"backend cross-check: source {source_name!r} driven "
                    f"node -- hpeesofsim {hp[source_name].pos_node!r} vs "
                    f"OA/ADS {oa[source_name].pos_node!r}"
                )

    return P1ToneContentResult(
        match=not failures,
        failures=failures,
        checked=sorted(expected),
        notes=notes,
    )


def _compare_number(
    failures: list[str],
    backend_name: str,
    source_name: str,
    label: str,
    got: float | None,
    want: float,
    rel_tol: float,
    abs_tol: float,
) -> None:
    """Append a failure when a parsed source parameter misses the spec."""

    if got is None:
        failures.append(
            f"[{backend_name}]: source {source_name!r} is missing "
            f"parameter {label!r}"
        )
        return
    if not math.isclose(got, want, rel_tol=rel_tol, abs_tol=abs_tol):
        failures.append(
            f"[{backend_name}]: source {source_name!r} {label} = "
            f"{got:.9g}, spec expects {want:.9g}"
        )


def format_content_report(result: P1ToneContentResult) -> str:
    """Render a human-readable P_1Tone content-check report."""

    lines: list[str] = []
    verdict = "MATCH" if result.match else "MISMATCH"
    lines.append(f"P_1Tone source content check: {verdict}")
    lines.append("=" * 38)
    if result.checked:
        lines.append(f"checked source(s): {', '.join(result.checked)}")
    if result.failures:
        lines.append(f"\n{len(result.failures)} failure(s):")
        lines += [f"  - {item}" for item in result.failures]
    else:
        lines.append(
            "\nevery P_1Tone source matches the device spec on every backend."
        )
    if result.notes:
        lines += [f"note: {note}" for note in result.notes]
    return "\n".join(lines) + "\n"


