"""VAR named-parameter content cross-check.

The structural cross-check (:mod:`netlist_compare`) proves the OA and
hpeesofsim netlists describe the same circuit topology and that their
shared parameters agree as text. For a circuit with a VAR block it
cannot, on its own, prove that a parameter literal traces back to the
*right* named scalar: the var-resolver substitutes a variable's value
into both backends from the same device spec, so a resolver that
substituted a wrong value would put the *same* wrong literal in both
netlists and the structural check would still pass.

This module closes that gap. For a device spec carrying a VAR block it
checks two invariants, mirroring the split the SDD2P content check
established:

1. **The resolver produces correct literals.** Every element parameter
   whose value names a declared variable must, after
   :func:`resolve_spec`, equal that variable's declared literal value
   (within a floating-point tolerance).
2. **Both backends consume the literals identically.** For every
   var-referencing element, the numeric parameter values emitted into
   the hpeesofsim netlist and into the OA/ADS netlist must agree.

VAR is not an emitted ADS element -- ADS VAR/VarEqn is non-functional
in ADS 2026 Update 1.2 (see ``docs/LIMITATIONS.md``) -- so there is no
VAR block to parse out of a netlist; the resolved values appear as
ordinary numeric literals on ordinary elements, and that is what this
check inspects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..device_spec import DeviceSpec, resolve_spec

# Floating-point tolerance for value comparison -- the same gate the
# SDD2P content check uses. A resolved literal survives the emit path
# (format -> string -> float) essentially exactly, so 1e-9 relative
# catches every meaningful error; the 1e-21 absolute floor only guards
# the comparison against an exactly-zero value.
DEFAULT_ABS_TOL = 1.0e-21
DEFAULT_REL_TOL = 1.0e-9


@dataclass(frozen=True)
class VarReference:
    """One element parameter whose value names a declared VAR variable.

    ``expected_value`` is the variable's declared literal -- what the
    resolver must substitute and what both backends must emit.
    """

    element_id: str
    parameter: str
    variable: str
    expected_value: float


@dataclass
class VarContentResult:
    """Outcome of :func:`check_var_content`.

    ``match`` is true only when every var reference resolves to its
    declared literal and every backend supplied agrees with the others.
    ``failures`` is the loud, human-readable list of disagreements;
    ``checked`` lists the var-referencing element ids examined.
    """

    match: bool
    failures: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def find_var_references(spec: DeviceSpec) -> list[VarReference]:
    """Return every element parameter that names a declared VAR variable.

    Scans the *unresolved* spec: a parameter whose value is a string
    equal to a variable name in ``spec.var_block`` is a reference.
    Returns an empty list when the spec has no VAR block.
    """

    if spec.var_block is None:
        return []
    namespace = {
        entry.name: float(entry.value) for entry in spec.var_block.variables
    }
    references: list[VarReference] = []
    for element in spec.elements:
        for key, value in element.parameters.items():
            if isinstance(value, str) and value in namespace:
                references.append(
                    VarReference(
                        element_id=element.id,
                        parameter=key,
                        variable=value,
                        expected_value=namespace[value],
                    )
                )
    return references


def _element_emitted_numbers(
    netlist_text: str, element_id: str
) -> dict[str, float]:
    """Return ``{instance.KEY: value}`` for one element's netlist lines.

    Matches netlist lines whose instance name is ``element_id`` or
    begins with ``element_id + "_"`` -- the latter catches the
    ``<id>_R`` / ``<id>_L`` / ``<id>_C`` sub-elements an ``RLC_SERIES``
    expands into. Every ``KEY=<number>`` token on such a line is
    recorded; a non-numeric assignment (``Noise=no``) is skipped.
    """

    prefix = element_id + "_"
    numbers: dict[str, float] = {}
    for raw in netlist_text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("//"):
            continue
        head = line.split(None, 1)[0]
        if ":" not in head:
            continue
        _, _, name = head.partition(":")
        if name != element_id and not name.startswith(prefix):
            continue
        for token in line.split():
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            try:
                numbers[f"{name}.{key}"] = float(value)
            except ValueError:
                continue
    return numbers


def check_var_content(
    spec: DeviceSpec,
    *,
    hpeesofsim_netlist_text: str | None = None,
    ads_netlist_text: str | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> VarContentResult:
    """Verify a spec's VAR named-parameter resolution and emission.

    Parameters
    ----------
    spec : DeviceSpec
        The device spec -- the source of truth for the VAR block and
        which element parameters reference it. Passed *unresolved*
        (with its ``var_block`` still present).
    hpeesofsim_netlist_text, ads_netlist_text : str, optional
        The two backends' netlist texts. When both are given the
        var-referencing elements' emitted numbers are checked against
        each other; either may be omitted.
    rel_tol, abs_tol : float
        Floating-point tolerance for value comparison.

    Returns
    -------
    VarContentResult
        ``match`` is true only when every var reference resolves to its
        declared literal and every supplied backend agrees.
    """

    failures: list[str] = []
    notes: list[str] = []

    if spec.var_block is None:
        notes.append("spec has no VAR block -- VAR content check is a no-op")
        return VarContentResult(match=True, notes=notes)

    references = find_var_references(spec)
    checked = sorted({ref.element_id for ref in references})
    if not references:
        notes.append(
            f"VAR block '{spec.var_block.name}' declares "
            f"{len(spec.var_block.variables)} variable(s), none referenced "
            f"by an element parameter"
        )

    # Invariant 1 -- the resolver substitutes each reference with the
    # variable's declared literal value.
    resolved = resolve_spec(spec)
    resolved_by_id = {element.id: element for element in resolved.elements}
    for ref in references:
        element = resolved_by_id.get(ref.element_id)
        if element is None:
            failures.append(
                f"resolver: var-referencing element '{ref.element_id}' is "
                f"absent from the resolved spec"
            )
            continue
        actual = element.parameters.get(ref.parameter)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            failures.append(
                f"resolver: '{ref.element_id}'.{ref.parameter} references "
                f"variable '{ref.variable}' but resolved to a non-numeric "
                f"value {actual!r}"
            )
            continue
        if not math.isclose(
            float(actual), ref.expected_value, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            failures.append(
                f"resolver: '{ref.element_id}'.{ref.parameter} references "
                f"variable '{ref.variable}' (declared "
                f"{ref.expected_value:.9g}) but resolved to "
                f"{float(actual):.9g}"
            )

    # Invariant 2 -- both backends emit the resolved literals identically.
    backends: dict[str, str] = {}
    if hpeesofsim_netlist_text is not None:
        backends["hpeesofsim"] = hpeesofsim_netlist_text
    if ads_netlist_text is not None:
        backends["OA/ADS"] = ads_netlist_text

    for element_id in checked:
        per_backend: dict[str, dict[str, float]] = {}
        for backend_name, text in backends.items():
            numbers = _element_emitted_numbers(text, element_id)
            if not numbers:
                failures.append(
                    f"backend [{backend_name}]: no netlist line found for "
                    f"var-referencing element '{element_id}'"
                )
            else:
                per_backend[backend_name] = numbers
        if "hpeesofsim" in per_backend and "OA/ADS" in per_backend:
            hp = per_backend["hpeesofsim"]
            oa = per_backend["OA/ADS"]
            if set(hp) != set(oa):
                failures.append(
                    f"backend cross-check: element '{element_id}' emits "
                    f"different parameter sets -- hpeesofsim {sorted(hp)} "
                    f"vs OA/ADS {sorted(oa)}"
                )
            else:
                for key in sorted(hp):
                    if not math.isclose(
                        hp[key], oa[key], rel_tol=rel_tol, abs_tol=abs_tol
                    ):
                        failures.append(
                            f"backend cross-check: element '{element_id}' "
                            f"parameter {key} disagrees -- hpeesofsim "
                            f"{hp[key]:.9g} vs OA/ADS {oa[key]:.9g}"
                        )

    return VarContentResult(
        match=not failures,
        failures=failures,
        checked=checked,
        notes=notes,
    )


def format_content_report(result: VarContentResult) -> str:
    """Render a human-readable VAR content-check report."""

    lines: list[str] = []
    verdict = "MATCH" if result.match else "MISMATCH"
    lines.append(f"VAR named-parameter content check: {verdict}")
    lines.append("=" * 42)
    if result.checked:
        lines.append(
            f"var-referencing element(s): {', '.join(result.checked)}"
        )
    if result.failures:
        lines.append(f"\n{len(result.failures)} failure(s):")
        lines += [f"  - {item}" for item in result.failures]
    else:
        lines.append(
            "\nevery VAR reference resolves to its declared literal and "
            "every backend agrees."
        )
    if result.notes:
        lines += [f"note: {note}" for note in result.notes]
    return "\n".join(lines) + "\n"


