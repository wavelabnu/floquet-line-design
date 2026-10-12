"""Resolve VAR named-scalar references in a device spec to literals.

ADS's VAR / VarEqn construct is non-functional in ADS 2026 Update 1.2
(see ``docs/LIMITATIONS.md``): neither the OA ``VarEqn`` cell nor the
hpeesofsim ``Var`` / ``Vars`` / ``Eqn`` netlist syntaxes work. A VAR
block is therefore never emitted as an ADS element. Instead this module
*resolves it away*: every element parameter whose value is the name of
a declared variable is replaced with that variable's literal numeric
value, and the VAR block is dropped. Both backends then emit ordinary
numeric literals and never depend on a VAR construct.

:func:`resolve_spec` is the device-spec transform the emitters and the
analytical simulator call at their entry. It is a no-op -- it returns
the spec unchanged -- for any spec with no VAR block, so it is free for
every pre-Level-5 circuit, and it is idempotent.

:func:`resolve_parameter_value` is the pure substitution core, testable
without any emitter.

Scope: a parameter references a variable by giving the variable's name
verbatim as the parameter value (``r_ohm = "R"``). Symbolic arithmetic
expressions over variables (``"2*pi*f0"``) are intentionally not
supported -- a variable's value is a literal number and a reference is
a whole-name lookup.

The resolver reaches three further named-reference
sites beyond element parameters: the HB-controller frequency fields
(:class:`HBSingleToneSpec` ``fundamental_freq``; :class:`HBTwoToneSpec`
``fundamental_freq_1`` / ``fundamental_freq_2``) and the
:class:`P1ToneSource` ``freq`` / ``power_dBm`` fields. Each is a
``float | str`` field; a string is resolved exactly as an element
parameter is, a number passes through unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .schema import (
    DeviceSpec,
    Element,
    HBSingleToneSpec,
    HBTwoToneSpec,
    P1ToneSource,
    VarBlockSpec,
)
from .validator import validate_var_block


def resolved_namespace(var_block: VarBlockSpec) -> dict[str, float]:
    """Return ``{variable name: literal value}`` for a VAR block.

    Variable order is the block's declaration order; a later duplicate
    name would overwrite an earlier one, but :func:`validate_var_block`
    rejects duplicate names before this is reached.
    """

    return {entry.name: float(entry.value) for entry in var_block.variables}


def resolve_parameter_value(value: Any, namespace: Mapping[str, float]) -> Any:
    """Resolve one element-parameter value against a variable namespace.

    A string equal to a declared variable name resolves to that
    variable's literal numeric value. Every other value -- a number, a
    string that is not a variable name, a list, a mapping -- is returned
    unchanged. The function is pure: same inputs, same output, no I/O.
    """

    if isinstance(value, str) and value in namespace:
        return namespace[value]
    return value


def resolve_spec(spec: DeviceSpec) -> DeviceSpec:
    """Return ``spec`` with every VAR reference substituted by a literal.

    For a spec with no VAR block this returns the spec unchanged (the
    same object), so the emitters can call it unconditionally at no cost
    for every pre-Level-5 circuit.

    For a spec with a VAR block: the block is validated; every element
    parameter, every HB-controller frequency field and every
    ``P1ToneSource`` ``freq`` / ``power_dBm`` field that names a
    declared variable is replaced with that variable's literal value,
    and ``var_block`` is cleared on the returned spec. The result
    therefore carries only numeric literals and is identical whichever
    backend consumes it.

    The transform is idempotent: resolving an already-resolved spec
    (``var_block is None``) returns it unchanged.

    Raises
    ------
    DeviceSpecError
        If the VAR block is itself malformed (propagated from
        :func:`validate_var_block`).
    """

    if spec.var_block is None:
        return spec
    validate_var_block(spec.var_block)
    namespace = resolved_namespace(spec.var_block)
    resolved_elements = tuple(
        _resolve_element(element, namespace) for element in spec.elements
    )
    resolved_sources = tuple(
        _resolve_source(source, namespace) for source in spec.sources
    )
    return replace(
        spec,
        elements=resolved_elements,
        sources=resolved_sources,
        hb_single_tone=_resolve_hb_single_tone(spec.hb_single_tone, namespace),
        hb_two_tone=_resolve_hb_two_tone(spec.hb_two_tone, namespace),
        var_block=None,
    )


def _resolve_element(element: Element, namespace: Mapping[str, float]) -> Element:
    """Return ``element`` with every VAR-referencing parameter resolved."""

    resolved = {
        key: resolve_parameter_value(value, namespace)
        for key, value in element.parameters.items()
    }
    return replace(element, parameters=resolved)


def _resolve_source(
    source: P1ToneSource, namespace: Mapping[str, float]
) -> P1ToneSource:
    """Return ``source`` with its ``freq`` / ``power_dBm`` resolved.

    A named reference becomes the variable's literal value; a numeric
    field passes through unchanged. ``num`` / ``z_ohm`` are never
    named-referenced and are left as-is.
    """

    return replace(
        source,
        freq=resolve_parameter_value(source.freq, namespace),
        power_dBm=resolve_parameter_value(source.power_dBm, namespace),
        phase_rad=resolve_parameter_value(source.phase_rad, namespace),
    )


def _resolve_hb_single_tone(
    hb: HBSingleToneSpec | None, namespace: Mapping[str, float]
) -> HBSingleToneSpec | None:
    """Return ``hb`` with its ``fundamental_freq`` resolved (or ``None``)."""

    if hb is None:
        return None
    return replace(
        hb, fundamental_freq=resolve_parameter_value(hb.fundamental_freq, namespace)
    )


def _resolve_hb_two_tone(
    hb: HBTwoToneSpec | None, namespace: Mapping[str, float]
) -> HBTwoToneSpec | None:
    """Return ``hb`` with both fundamental frequencies resolved (or ``None``)."""

    if hb is None:
        return None
    return replace(
        hb,
        fundamental_freq_1=resolve_parameter_value(
            hb.fundamental_freq_1, namespace
        ),
        fundamental_freq_2=resolve_parameter_value(
            hb.fundamental_freq_2, namespace
        ),
    )


