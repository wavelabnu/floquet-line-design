"""Design-layer invariant validator.

:func:`validate_design` enforces the invariants a :class:`TVTLDesign`
must satisfy *before* expansion -- the structural checks that catch a
malformed design with a message at the design layer rather than letting
it surface as a confusing device-spec error after expansion.

It does not simulate or expand anything. The expanded device spec is
re-validated by the existing :func:`device_spec.validate_device_spec`;
this validator covers only the design-layer-specific rules: the cell
count, named-reference resolution, the design-layer element vocabulary,
the analysis controllers, and the branch terminations.
"""

from __future__ import annotations

import math

from .elements import TVTLDesign

# Design-layer element type tags accepted in a branch's element chain.
# These correspond one-to-one to device-spec element types the v2
# emitters support; ``SDD2P_varactor`` expands to the device-spec
# ``SDD2P``.
_BRANCH_ELEMENT_TYPES = frozenset(
    {"R", "L", "C", "MLIN", "MTEE", "SDD2P_varactor"}
)

# Valid branch terminations.
_TERMINATIONS = frozenset({"ground", "floating"})

# The design-layer varactor type and its polynomial-reference key.
_VARACTOR_TYPE = "SDD2P_varactor"
_Q_POLY_REFERENCE_KEY = "q_poly_reference"


class DesignError(ValueError):
    """Raised when a :class:`TVTLDesign` violates a design-layer invariant."""


def _is_number(value: object) -> bool:
    """True for a real, finite numeric value (``bool`` excluded)."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_design(design: TVTLDesign) -> None:
    """Validate a :class:`TVTLDesign` against the design-layer invariants.

    Rules
    -----
    * ``n_cells`` is an integer >= 1;
    * ``parameters`` values are all finite real numbers;
    * every named reference in the cell, the sources and the analysis
      controllers names a declared ``parameters`` entry;
    * every branch element type is one the design layer expands
      (``R`` / ``L`` / ``C`` / ``MLIN`` / ``MTEE`` / ``SDD2P_varactor``);
    * a branch's ``SDD2P_varactor``, if present, is the last element and
      carries a ``q_poly_reference``;
    * ``analysis`` declares the S-parameter controller (the device spec's
      mandatory analysis block) and at least one controller overall;
    * every branch ``termination`` is ``"ground"`` or ``"floating"``.

    Raises
    ------
    DesignError
        On the first violation; the message identifies the field.
    """

    n = design.n_cells
    if isinstance(n, bool) or not isinstance(n, int):
        raise DesignError(
            f"n_cells must be an integer (got {type(n).__name__})."
        )
    if n < 1:
        raise DesignError(f"n_cells must be >= 1 (got {n}).")

    for name, value in design.parameters.items():
        if not _is_number(value):
            raise DesignError(
                f"parameters[{name!r}] must be a finite real number "
                f"(got {value!r})."
            )
    declared = frozenset(design.parameters)

    _check_analysis(design)
    _check_cell(design, declared)
    _check_sources(design, declared)
    _check_analysis_refs(design, declared)


def _check_analysis(design: TVTLDesign) -> None:
    """Validate the analysis controllers' presence."""

    analysis = design.analysis
    if analysis.s_param is None:
        raise DesignError(
            "analysis.s_param is required -- the device spec's analysis "
            "block is the S-parameter analysis."
        )
    sp = analysis.s_param
    if not (_is_number(sp.f_start_Hz) and sp.f_start_Hz > 0.0):
        raise DesignError(
            f"analysis.s_param.f_start_Hz must be positive (got "
            f"{sp.f_start_Hz!r})."
        )
    if not (_is_number(sp.f_stop_Hz) and sp.f_stop_Hz > sp.f_start_Hz):
        raise DesignError(
            "analysis.s_param.f_stop_Hz must be greater than f_start_Hz "
            f"(start={sp.f_start_Hz!r}, stop={sp.f_stop_Hz!r})."
        )
    if not isinstance(sp.n_points, int) or sp.n_points < 2:
        raise DesignError(
            f"analysis.s_param.n_points must be an integer >= 2 (got "
            f"{sp.n_points!r})."
        )


def _check_ref(location: str, value: object, declared: frozenset) -> None:
    """A ``float | str`` field: a string must name a declared parameter."""

    if isinstance(value, str):
        if value not in declared:
            raise DesignError(
                f"{location} references parameter {value!r}, which is not "
                f"declared in parameters (declared: {sorted(declared)})."
            )
    elif not _is_number(value):
        raise DesignError(
            f"{location} must be a finite real number or a parameter name "
            f"(got {value!r})."
        )


def _check_cell(design: TVTLDesign, declared: frozenset) -> None:
    """Validate the unit cell's through-line and branches."""

    cell = design.cell
    _check_ref(
        "cell.through_line.width", cell.through_line.width, declared
    )
    _check_ref(
        "cell.through_line.length", cell.through_line.length, declared
    )
    for b, branch in enumerate(cell.branches):
        if branch.termination not in _TERMINATIONS:
            raise DesignError(
                f"cell.branches[{b}].termination {branch.termination!r} is "
                f"not valid (expected one of {sorted(_TERMINATIONS)})."
            )
        if not branch.elements:
            raise DesignError(
                f"cell.branches[{b}].elements must contain at least one "
                f"element."
            )
        for e, elem in enumerate(branch.elements):
            path = f"cell.branches[{b}].elements[{e}]"
            if elem.type not in _BRANCH_ELEMENT_TYPES:
                raise DesignError(
                    f"{path}.type {elem.type!r} is not a recognised design "
                    f"element type (expected one of "
                    f"{sorted(_BRANCH_ELEMENT_TYPES)})."
                )
            if elem.type == _VARACTOR_TYPE:
                if e != len(branch.elements) - 1:
                    raise DesignError(
                        f"{path} ({_VARACTOR_TYPE}) must be the last element "
                        f"of its branch -- it is the branch terminator."
                    )
                if _Q_POLY_REFERENCE_KEY not in elem.params:
                    raise DesignError(
                        f"{path} ({_VARACTOR_TYPE}) is missing required "
                        f"parameter {_Q_POLY_REFERENCE_KEY!r}."
                    )
            for key, value in elem.params.items():
                # q_poly_reference is a polynomial-registry key, not a
                # VAR reference -- it is resolved by the expander, not
                # the var-resolver, so it is exempt from the named-ref
                # check.
                if key == _Q_POLY_REFERENCE_KEY:
                    continue
                _check_ref(f"{path}.params[{key!r}]", value, declared)


def _check_sources(design: TVTLDesign, declared: frozenset) -> None:
    """Validate the design's ``P_1Tone`` sources' named references."""

    for s, source in enumerate(design.sources):
        _check_ref(f"sources[{s}].freq", source.freq, declared)
        _check_ref(
            f"sources[{s}].power_dBm", source.power_dBm, declared
        )


def _check_analysis_refs(design: TVTLDesign, declared: frozenset) -> None:
    """Validate the HB controllers' frequency named references."""

    hb1 = design.analysis.hb_single_tone
    if hb1 is not None:
        _check_ref(
            "analysis.hb_single_tone.fundamental_freq",
            hb1.fundamental_freq,
            declared,
        )
    hb2 = design.analysis.hb_two_tone
    if hb2 is not None:
        _check_ref(
            "analysis.hb_two_tone.fundamental_freq_1",
            hb2.fundamental_freq_1,
            declared,
        )
        _check_ref(
            "analysis.hb_two_tone.fundamental_freq_2",
            hb2.fundamental_freq_2,
            declared,
        )


