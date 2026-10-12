"""Device-spec invariant validator.

Every device-spec invariant is enforced here.
Errors are raised as :class:`DeviceSpecError` with a path-style message
that identifies the offending field by JSON pointer (e.g.
``elements[1].pins[0].node 'n_mid' not declared in nodes``).

The validator never simulates anything; it only checks that the parsed
spec is structurally consistent. Simulation-side dispatch (e.g. element
type is one of {TermG, MLIN}) lives in the netlist/simulator layers.
"""

from __future__ import annotations

import math
import re

from .schema import (
    DEVICE_SPEC_VERSION,
    DeviceSpec,
    HBSingleToneSpec,
    HBTwoToneSpec,
    P1ToneSource,
    VarBlockSpec,
)


class DeviceSpecError(ValueError):
    """Raised when a device spec violates one of the schema invariants."""


def validate_device_spec(spec: DeviceSpec) -> None:
    """Validate a parsed :class:`DeviceSpec` against the Level-1 schema.

    Parameters
    ----------
    spec : DeviceSpec
        The parsed device spec.

    Raises
    ------
    DeviceSpecError
        If any structural invariant is violated. The message identifies
        the offending field by path.
    """

    _check_version(spec)
    _check_analysis(spec)
    _check_nodes_unique(spec)
    _check_substrate(spec)
    _check_var_block(spec)
    _check_hb_single_tone(spec)
    _check_hb_two_tone(spec)
    _check_elements(spec)
    _check_sources(spec)
    _check_ports(spec)
    _check_ground_nodes(spec)


def _spec_var_names(spec: DeviceSpec) -> frozenset[str]:
    """Return the set of variable names a spec's VAR block declares.

    Empty when the spec carries no VAR block. A ``float | str`` schema
    field whose value is a string must name one of these.
    """

    if spec.var_block is None:
        return frozenset()
    return frozenset(entry.name for entry in spec.var_block.variables)


def _check_number_or_name(
    descr: str,
    value: object,
    field: str,
    var_names: frozenset[str],
    *,
    require_positive: bool,
) -> None:
    """Validate one ``float | str`` schema field (a frequency/power).

    A string value is a VAR-block reference: it must name a declared
    variable -- the var-resolver substitutes the literal before
    emission, so the numeric range check is deferred to the resolved
    spec. A numeric value must be finite, and positive when
    ``require_positive`` is set. Anything else is rejected.
    """

    if isinstance(value, str):
        if value not in var_names:
            raise DeviceSpecError(
                f"{descr} {field} references variable {value!r}, which is "
                f"not declared in the VAR block "
                f"(declared: {sorted(var_names)})."
            )
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeviceSpecError(
            f"{descr} {field} must be a real number or a VAR-variable name "
            f"(got {type(value).__name__})."
        )
    number = float(value)
    if not math.isfinite(number):
        raise DeviceSpecError(
            f"{descr} {field} must be finite (got {value})."
        )
    if require_positive and number <= 0.0:
        raise DeviceSpecError(
            f"{descr} {field} must be positive (got {value})."
        )


# A valid ADS identifier: a letter or underscore, then letters, digits
# or underscores. Used for VAR variable names and the block name.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def validate_var_block(var_block: VarBlockSpec) -> None:
    """Validate a :class:`VarBlockSpec`'s names and values.

    Rules
    -----
    * the block name and every variable name must be a valid ADS
      identifier -- a letter or underscore, then letters/digits/
      underscores -- and names are case-sensitive;
    * variable names must be unique within the block;
    * every variable value must be a finite real number.

    No reserved-name rule is applied. A VAR block is resolved away
    before emission (see :class:`VarBlockSpec`), so it never reaches
    ADS, and there is no ADS-probe-confirmed reserved-name list to
    enforce -- inventing one would be a guess.

    Raises
    ------
    DeviceSpecError
        On any violation; the message identifies the offending field.
    """

    if not isinstance(var_block.name, str) or not _IDENTIFIER_RE.fullmatch(
        var_block.name
    ):
        raise DeviceSpecError(
            f"var_block.name {var_block.name!r} is not a valid identifier "
            f"(a letter or underscore, then letters/digits/underscores)."
        )
    seen: set[str] = set()
    for index, entry in enumerate(var_block.variables):
        path = f"var_block.variables[{index}]"
        if not isinstance(entry.name, str) or not _IDENTIFIER_RE.fullmatch(
            entry.name
        ):
            raise DeviceSpecError(
                f"{path}.name {entry.name!r} is not a valid identifier "
                f"(a letter or underscore, then letters/digits/underscores)."
            )
        if entry.name in seen:
            raise DeviceSpecError(
                f"{path}.name {entry.name!r} duplicates a previous variable "
                f"in the same VAR block."
            )
        seen.add(entry.name)
        value = entry.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DeviceSpecError(
                f"{path} variable {entry.name!r} value must be a real number "
                f"(got {type(value).__name__})."
            )
        if not math.isfinite(float(value)):
            raise DeviceSpecError(
                f"{path} variable {entry.name!r} value must be finite "
                f"(got {value})."
            )


def _check_var_block(spec: DeviceSpec) -> None:
    """Validate ``spec.var_block`` when one is present."""

    if spec.var_block is not None:
        validate_var_block(spec.var_block)


def validate_hb_single_tone(
    hb: HBSingleToneSpec, var_names: frozenset[str] = frozenset()
) -> None:
    """Validate a single-tone Harmonic Balance controller spec.

    Rules
    -----
    * ``fundamental_freq`` is a positive, finite real number, or a
      ``str`` naming a variable in ``var_names`` (the VAR block the
      var-resolver will substitute from). A string reference to a name
      not in ``var_names`` -- including the empty set, when the spec
      has no VAR block -- is rejected;
    * ``fundamental_order`` must be an integer >= 1 (the fundamental
      itself is order 1);
    * ``max_order`` must be an integer >= ``fundamental_order`` -- the
      maximum combined order cannot be below the single tone's own
      harmonic order.

    Parameters
    ----------
    hb : HBSingleToneSpec
    var_names : frozenset[str]
        Names declared in the device's VAR block; a string-valued
        ``fundamental_freq`` must be one of these. Empty by default
        (the standalone, no-VAR-block case).

    Raises
    ------
    DeviceSpecError
        On any violation; the message identifies the offending field.
    """

    _check_number_or_name(
        "hb_single_tone",
        hb.fundamental_freq,
        "fundamental_freq",
        var_names,
        require_positive=True,
    )
    order = hb.fundamental_order
    if isinstance(order, bool) or not isinstance(order, int):
        raise DeviceSpecError(
            f"hb_single_tone.fundamental_order must be an integer "
            f"(got {type(order).__name__})."
        )
    if order < 1:
        raise DeviceSpecError(
            f"hb_single_tone.fundamental_order must be >= 1 (got {order})."
        )
    max_order = hb.max_order
    if isinstance(max_order, bool) or not isinstance(max_order, int):
        raise DeviceSpecError(
            f"hb_single_tone.max_order must be an integer "
            f"(got {type(max_order).__name__})."
        )
    if max_order < order:
        raise DeviceSpecError(
            f"hb_single_tone.max_order ({max_order}) must be >= "
            f"fundamental_order ({order})."
        )


def _check_hb_single_tone(spec: DeviceSpec) -> None:
    """Validate ``spec.hb_single_tone`` when one is present."""

    if spec.hb_single_tone is not None:
        validate_hb_single_tone(spec.hb_single_tone, _spec_var_names(spec))


def validate_hb_two_tone(
    hb: HBTwoToneSpec, var_names: frozenset[str] = frozenset()
) -> None:
    """Validate a two-tone Harmonic Balance controller spec.

    Rules
    -----
    * ``fundamental_freq_1`` and ``fundamental_freq_2`` are each a
      positive, finite real number, or a ``str`` naming a variable in
      ``var_names`` (the var-resolver substitutes it before emission);
    * ``fundamental_order_1`` and ``fundamental_order_2`` must each be
      an integer >= 1;
    * ``max_order`` must be an integer >= the larger of the two
      per-tone orders. This is the loose bound v1's pattern satisfies
      (v1 uses ``Order=[order, order]`` with ``MaxOrder=max(order, 5)``,
      which is below ``order + order``); ``order_1 + order_2`` is *not*
      required.

    Parameters
    ----------
    hb : HBTwoToneSpec
    var_names : frozenset[str]
        Names declared in the device's VAR block; a string-valued
        fundamental frequency must be one of these. Empty by default.

    Raises
    ------
    DeviceSpecError
        On any violation; the message identifies the offending field.
    """

    for label, freq in (
        ("fundamental_freq_1", hb.fundamental_freq_1),
        ("fundamental_freq_2", hb.fundamental_freq_2),
    ):
        _check_number_or_name(
            "hb_two_tone", freq, label, var_names, require_positive=True
        )
    for label, order in (
        ("fundamental_order_1", hb.fundamental_order_1),
        ("fundamental_order_2", hb.fundamental_order_2),
    ):
        if isinstance(order, bool) or not isinstance(order, int):
            raise DeviceSpecError(
                f"hb_two_tone.{label} must be an integer "
                f"(got {type(order).__name__})."
            )
        if order < 1:
            raise DeviceSpecError(
                f"hb_two_tone.{label} must be >= 1 (got {order})."
            )
    max_order = hb.max_order
    if isinstance(max_order, bool) or not isinstance(max_order, int):
        raise DeviceSpecError(
            f"hb_two_tone.max_order must be an integer "
            f"(got {type(max_order).__name__})."
        )
    min_required = max(hb.fundamental_order_1, hb.fundamental_order_2)
    if max_order < min_required:
        raise DeviceSpecError(
            f"hb_two_tone.max_order ({max_order}) must be >= the larger "
            f"per-tone order ({min_required})."
        )


def _check_hb_two_tone(spec: DeviceSpec) -> None:
    """Validate ``spec.hb_two_tone`` and the HB-controller exclusivity.

    A device spec may declare at most one HB controller: setting both
    ``hb_single_tone`` and ``hb_two_tone`` is rejected.
    """

    if spec.hb_two_tone is None:
        return
    if spec.hb_single_tone is not None:
        raise DeviceSpecError(
            "hb_single_tone and hb_two_tone are mutually exclusive -- a "
            "device spec may declare at most one HB controller."
        )
    validate_hb_two_tone(spec.hb_two_tone, _spec_var_names(spec))


def validate_p1tone_source(
    source: P1ToneSource, var_names: frozenset[str] = frozenset()
) -> None:
    """Validate one :class:`P1ToneSource` -- the source-intrinsic rules.

    Rules
    -----
    * ``name`` must be a valid ADS identifier;
    * ``num`` must be an integer >= 1;
    * ``z_ohm`` must be a positive, finite real number;
    * ``freq`` must be a positive, finite real number or a VAR-block
      variable name in ``var_names``;
    * ``power_dBm`` must be a finite real number (any sign -- dBm is a
      logarithmic scale and a negative value is an ordinary small
      power) or a VAR-block variable name.

    The cross-cutting rules -- ``port_node`` is a declared, non-ground
    node, and ``name`` is unique among a spec's sources -- are checked
    by :func:`_check_sources`, which has the whole spec in hand.

    Parameters
    ----------
    source : P1ToneSource
    var_names : frozenset[str]
        VAR-block variable names; a string-valued ``freq`` /
        ``power_dBm`` must be one of these. Empty by default.

    Raises
    ------
    DeviceSpecError
        On any violation; the message identifies the offending field.
    """

    if not isinstance(source.name, str) or not _IDENTIFIER_RE.fullmatch(
        source.name
    ):
        raise DeviceSpecError(
            f"P1ToneSource name {source.name!r} is not a valid identifier "
            f"(a letter or underscore, then letters/digits/underscores)."
        )
    descr = f"P1ToneSource {source.name!r}"
    num = source.num
    if isinstance(num, bool) or not isinstance(num, int):
        raise DeviceSpecError(
            f"{descr} num must be an integer (got {type(num).__name__})."
        )
    if num < 1:
        raise DeviceSpecError(f"{descr} num must be >= 1 (got {num}).")
    z_ohm = source.z_ohm
    if isinstance(z_ohm, bool) or not isinstance(z_ohm, (int, float)):
        raise DeviceSpecError(
            f"{descr} z_ohm must be a real number "
            f"(got {type(z_ohm).__name__})."
        )
    if not (math.isfinite(float(z_ohm)) and float(z_ohm) > 0.0):
        raise DeviceSpecError(
            f"{descr} z_ohm must be positive and finite (got {z_ohm})."
        )
    _check_number_or_name(
        descr, source.freq, "freq", var_names, require_positive=True
    )
    _check_number_or_name(
        descr, source.power_dBm, "power_dBm", var_names, require_positive=False
    )
    _check_number_or_name(
        descr, source.phase_rad, "phase_rad", var_names, require_positive=False
    )


def _check_sources(spec: DeviceSpec) -> None:
    """Validate ``spec.sources`` -- the ``P_1Tone`` power sources.

    Each source is validated for its intrinsic rules
    (:func:`validate_p1tone_source`); additionally every source's
    ``port_node`` must be a declared node that is not a ground node (a
    source drives a signal node, never the ground reference), and
    source names must be unique within the spec.
    """

    node_ids = {n.id for n in spec.nodes}
    ground = set(spec.ground_nodes)
    var_names = _spec_var_names(spec)
    seen: set[str] = set()
    for index, source in enumerate(spec.sources):
        path = f"sources[{index}]"
        validate_p1tone_source(source, var_names)
        if source.name in seen:
            raise DeviceSpecError(
                f"{path} name {source.name!r} duplicates a previous source."
            )
        seen.add(source.name)
        if source.port_node not in node_ids:
            raise DeviceSpecError(
                f"{path} ({source.name!r}) port_node {source.port_node!r} "
                f"is not declared in nodes."
            )
        if source.port_node in ground:
            raise DeviceSpecError(
                f"{path} ({source.name!r}) port_node {source.port_node!r} "
                f"is a ground node; a P_1Tone source drives a signal node, "
                f"not the ground reference."
            )


def _check_version(spec: DeviceSpec) -> None:
    if int(spec.device_spec_version) != int(DEVICE_SPEC_VERSION):
        raise DeviceSpecError(
            f"device_spec_version {spec.device_spec_version} is not supported; "
            f"this builder accepts version {DEVICE_SPEC_VERSION}."
        )


def _check_analysis(spec: DeviceSpec) -> None:
    if spec.analysis.type != "s_parameters":
        raise DeviceSpecError(
            f"analysis.type '{spec.analysis.type}' is not permitted at Level 1; "
            "only 's_parameters' is implemented."
        )
    grid = spec.analysis.frequency_grid_Hz
    if grid.start <= 0.0:
        raise DeviceSpecError(
            f"analysis.frequency_grid_Hz.start must be positive (got {grid.start})."
        )
    if grid.stop <= grid.start:
        raise DeviceSpecError(
            "analysis.frequency_grid_Hz.stop must be greater than start "
            f"(start={grid.start}, stop={grid.stop})."
        )
    if grid.points < 2:
        raise DeviceSpecError(
            f"analysis.frequency_grid_Hz.points must be >= 2 (got {grid.points})."
        )
    if spec.analysis.port_impedance_ohm <= 0.0:
        raise DeviceSpecError(
            "analysis.port_impedance_ohm must be positive "
            f"(got {spec.analysis.port_impedance_ohm})."
        )


def _check_nodes_unique(spec: DeviceSpec) -> None:
    seen: set[str] = set()
    for index, node in enumerate(spec.nodes):
        if not node.id:
            raise DeviceSpecError(f"nodes[{index}].id is empty.")
        if node.id in seen:
            raise DeviceSpecError(f"nodes[{index}].id '{node.id}' duplicates a previous node.")
        seen.add(node.id)


def _check_substrate(spec: DeviceSpec) -> None:
    if not spec.substrate.name:
        raise DeviceSpecError("substrate.name must be non-empty.")
    if spec.substrate.model != "hammerstad_kirschning":
        raise DeviceSpecError(
            f"substrate.model '{spec.substrate.model}' is not implemented; "
            "Level 1 supports only 'hammerstad_kirschning'."
        )
    required = {"eps_r", "h_m", "tan_delta", "sigma_S_per_m", "t_m"}
    missing = required - set(spec.substrate.parameters)
    if missing:
        raise DeviceSpecError(
            f"substrate.parameters is missing required key(s): {sorted(missing)}."
        )
    eps_r = float(spec.substrate.parameters["eps_r"])
    h_m = float(spec.substrate.parameters["h_m"])
    if eps_r <= 0.0 or h_m <= 0.0:
        raise DeviceSpecError(
            "substrate.parameters eps_r and h_m must be positive "
            f"(got eps_r={eps_r}, h_m={h_m})."
        )


_RECOGNIZED_ELEMENT_TYPES = frozenset(
    {
        "TermG", "MLIN", "MLEF", "MLOC", "VIAGND", "MTEE", "R", "L", "C",
        "RLC_SERIES", "SDD2P", "I_Probe", "NAME",
    }
)

# Required parameters for each lumped element type (Level 4). ``R`` may
# be zero (a zero-Ohm resistor is an ideal wire); ``L`` and ``C`` must
# be strictly positive -- the positivity rule per parameter is enforced
# in ``_check_lumped_element``.
_LUMPED_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "R": ("r_ohm",),
    "L": ("l_h",),
    "C": ("c_f",),
    "RLC_SERIES": ("r_ohm", "l_h", "c_f"),
}


def _check_numeric_param(
    descr: str, raw: object, key: str, var_names: set[str]
) -> float | None:
    """Return a numeric element parameter as a float, or ``None``.

    ``None`` means the value is a string naming a declared VAR variable
    -- a valid reference the var-resolver substitutes with a literal
    before emission, so the pre-resolution validator skips the numeric
    range check for it. A string that is *not* a declared variable is
    rejected. When the spec has no VAR block (``var_names`` empty) the
    value is converted with ``float`` exactly as before.
    """

    if var_names and isinstance(raw, str):
        if raw in var_names:
            return None
        raise DeviceSpecError(
            f"{descr} parameter {key!r} references variable {raw!r}, which "
            f"is not declared in the VAR block (declared: {sorted(var_names)})."
        )
    return float(raw)  # type: ignore[arg-type]


def _check_elements(spec: DeviceSpec) -> None:
    node_ids = {n.id for n in spec.nodes}
    seen_ids: set[str] = set()
    var_names = (
        {entry.name for entry in spec.var_block.variables}
        if spec.var_block is not None
        else set()
    )
    for elem_idx, element in enumerate(spec.elements):
        path = f"elements[{elem_idx}]"
        if not element.id:
            raise DeviceSpecError(f"{path}.id is empty.")
        if element.id in seen_ids:
            raise DeviceSpecError(f"{path}.id '{element.id}' duplicates a previous element.")
        seen_ids.add(element.id)
        if not element.type:
            raise DeviceSpecError(f"{path}.type is empty.")
        if element.type not in _RECOGNIZED_ELEMENT_TYPES:
            raise DeviceSpecError(
                f"{path}.type '{element.type}' is not a recognized element type; "
                f"this builder implements {sorted(_RECOGNIZED_ELEMENT_TYPES)}."
            )
        if not element.pins:
            raise DeviceSpecError(f"{path}.pins must contain at least one entry.")
        expected_pin_numbers = list(range(1, len(element.pins) + 1))
        actual_pin_numbers = [pin.pin for pin in element.pins]
        if sorted(actual_pin_numbers) != expected_pin_numbers:
            raise DeviceSpecError(
                f"{path}.pins must be 1-indexed and contiguous (got {actual_pin_numbers})."
            )
        for pin_idx, pin in enumerate(element.pins):
            pin_path = f"{path}.pins[{pin_idx}]"
            if pin.node not in node_ids:
                raise DeviceSpecError(
                    f"{pin_path}.node '{pin.node}' not declared in nodes"
                )
        substrate_ref = element.parameters.get("substrate_ref")
        if substrate_ref is not None and str(substrate_ref) != spec.substrate.name:
            raise DeviceSpecError(
                f"{path}.parameters.substrate_ref '{substrate_ref}' does not match "
                f"substrate.name '{spec.substrate.name}'."
            )
        if element.type == "MTEE":
            _check_mtee_element(path, element, spec)
        elif element.type == "SDD2P":
            _check_sdd2p_element(path, element)
        elif element.type == "I_Probe":
            _check_i_probe_element(path, element)
        elif element.type == "NAME":
            _check_name_element(path, element)
        elif element.type in _LUMPED_REQUIRED_PARAMS:
            _check_lumped_element(path, element, var_names)
    _check_name_uniqueness(spec)


def _check_mtee_element(path: str, element, spec: DeviceSpec) -> None:
    """Validate an MTEE element's pin count and required parameters."""

    if len(element.pins) != 3:
        raise DeviceSpecError(
            f"{path} (MTEE '{element.id}') must declare exactly three pins "
            f"(got {len(element.pins)})."
        )
    required = {"w1_m", "w2_m", "w3_m", "substrate_ref"}
    missing = required - set(element.parameters)
    if missing:
        raise DeviceSpecError(
            f"{path} (MTEE '{element.id}') is missing required parameter(s): "
            f"{sorted(missing)}."
        )
    for key in ("w1_m", "w2_m", "w3_m"):
        if float(element.parameters[key]) <= 0.0:
            raise DeviceSpecError(
                f"{path} (MTEE '{element.id}') parameter '{key}' must be positive."
            )


def _check_sdd2p_element(path: str, element) -> None:
    """Validate an ``SDD2P`` symbolically-defined-device element.

    ``SDD2P`` is the ADS native 2-port symbolically-defined device,
    used here to represent a charge-driven nonlinear varactor.
    It is a genuine 2-port, so it has exactly four pins
    ``[port1+, port1-, port2+, port2-]``.

    The required ``q_poly_coefficients`` parameter is the charge
    polynomial ``Q(v) = c0 + c1*v + c2*v**2 + ...`` (coulombs / V**n).
    It must be a non-empty sequence, and its first coefficient ``c0``
    must be ``0.0`` -- the charge polynomial passes through the origin
    (``Q(0) = 0``). ``q_poly_reference`` is optional (an audit-only
    string identifier traceable to the source polynomial).
    """

    if len(element.pins) != 4:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') must declare exactly four pins "
            f"(got {len(element.pins)})."
        )
    # Control-port realistic time-varying-capacitor mode:
    # the port-1 charge is (sum_k a_k * v2^k) * v1, realising C(t) =
    # C(V_bias + v2). In this mode a 'tvc_poly_coefficients' tuple replaces
    # the charge polynomial.
    if "tvc_poly_coefficients" in element.parameters:
        _check_sdd2p_tvc_poly_params(path, element)
        return
    # Time-varying-capacitor (control-port) mode: an ideal externally-pumped
    # linear capacitor C(t) = tvc_c0_f + tvc_k_f_per_v * v2, with port 1 the
    # in-line shunt and port 2 the control port (driven by a pump source).
    # In this mode the charge polynomial is replaced by the (C0, K) pair.
    if "tvc_c0_f" in element.parameters:
        _check_sdd2p_tvc_params(path, element)
        return
    if "q_poly_coefficients" not in element.parameters:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') is missing required parameter "
            f"'q_poly_coefficients' (or 'tvc_c0_f' for time-varying-C mode)."
        )
    coefficients = element.parameters["q_poly_coefficients"]
    if isinstance(coefficients, (str, bytes)) or not isinstance(
        coefficients, (tuple, list)
    ):
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'q_poly_coefficients' "
            f"must be a tuple/list of floats (got {type(coefficients).__name__})."
        )
    if len(coefficients) == 0:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'q_poly_coefficients' "
            f"must be a non-empty sequence."
        )
    try:
        values = [float(value) for value in coefficients]
    except (TypeError, ValueError) as exc:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'q_poly_coefficients' "
            f"must contain only numbers ({exc})."
        ) from exc
    if values[0] != 0.0:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'q_poly_coefficients' "
            f"first coefficient (c0) must be 0.0 -- the charge polynomial "
            f"passes through the origin, Q(0) = 0 -- (got {values[0]})."
        )
    reference = element.parameters.get("q_poly_reference")
    if reference is not None and not isinstance(reference, str):
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'q_poly_reference' "
            f"must be a string when present (got "
            f"{type(reference).__name__})."
        )


def _check_sdd2p_tvc_params(path: str, element) -> None:
    """Validate an SDD2P in time-varying-capacitor (control-port) mode.

    Requires ``tvc_c0_f`` (static C, positive) and ``tvc_k_f_per_v`` (the
    control-port modulation slope dC/dv2, real). The emitted SDD body is
    ``I[1,1] = (C0)*_v1 + (K)*_v2*_v1`` (port-1 charge, weight 1 -> dQ/dt) and
    ``I[2,0] = 0`` (port 2 is a zero-current voltage-sense control port). The
    optional ``tvc_reference`` is an audit-only string.
    """

    c0 = element.parameters.get("tvc_c0_f")
    try:
        c0_val = float(c0)
    except (TypeError, ValueError) as exc:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'tvc_c0_f' must be a "
            f"number ({exc})."
        ) from exc
    if not c0_val > 0.0:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'tvc_c0_f' must be "
            f"positive (got {c0_val})."
        )
    if "tvc_k_f_per_v" not in element.parameters:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') in time-varying-C mode is missing "
            f"required parameter 'tvc_k_f_per_v'."
        )
    try:
        float(element.parameters["tvc_k_f_per_v"])
    except (TypeError, ValueError) as exc:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'tvc_k_f_per_v' must be "
            f"a number ({exc})."
        ) from exc
    reference = element.parameters.get("tvc_reference")
    if reference is not None and not isinstance(reference, str):
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'tvc_reference' must be "
            f"a string when present (got {type(reference).__name__})."
        )


def _check_sdd2p_tvc_poly_params(path: str, element) -> None:
    """Validate an SDD2P in control-port realistic time-varying-C mode.

    Requires ``tvc_poly_coefficients`` -- a non-empty tuple/list of numbers
    ``[a0, a1, ...]`` (the capacitance-vs-pump Taylor coefficients) whose
    static term ``a0`` is positive. The emitted body is
    ``I[1,1] = (a0)*_v1 + (a1)*_v2*_v1 + (a2)*_v2**2*_v1 + ...`` (port-1
    charge, weight 1 -> dQ/dt) and ``I[2,0] = 0`` (zero-current control port).
    The optional ``tvc_reference`` / ``q_poly_reference`` is an audit string.
    """

    coeffs = element.parameters["tvc_poly_coefficients"]
    if isinstance(coeffs, (str, bytes)) or not isinstance(coeffs, (tuple, list)):
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'tvc_poly_coefficients' "
            f"must be a tuple/list of numbers (got {type(coeffs).__name__})."
        )
    if len(coeffs) == 0:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'tvc_poly_coefficients' "
            f"must be a non-empty sequence."
        )
    try:
        values = [float(v) for v in coeffs]
    except (TypeError, ValueError) as exc:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'tvc_poly_coefficients' "
            f"must contain only numbers ({exc})."
        ) from exc
    if not values[0] > 0.0:
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'tvc_poly_coefficients' "
            f"static term a0 must be positive (got {values[0]})."
        )
    reference = element.parameters.get("tvc_reference")
    if reference is not None and not isinstance(reference, str):
        raise DeviceSpecError(
            f"{path} (SDD2P '{element.id}') parameter 'tvc_reference' must be "
            f"a string when present (got {type(reference).__name__})."
        )


def _check_i_probe_element(path: str, element) -> None:
    """Validate an ``I_Probe`` current-probe element.

    ``I_Probe`` is the native ADS current probe -- an electrically
    transparent (0 ohm) two-terminal element that records the current
    through itself. It is parameterless at the device-spec level: the
    fixed ``Mode=0`` / ``SaveCurrent=yes`` that select the current-probe
    behaviour are emitter constants (see ``netlist/elements/i_probe.py``
    and the live-probe findings), not user-set fields. So
    the only invariant here is the two-terminal pin count; the generic
    pin checks in :func:`_check_elements` already enforce that the pins
    are the contiguous set {1, 2}.
    """

    if len(element.pins) != 2:
        raise DeviceSpecError(
            f"{path} (I_Probe '{element.id}') must declare exactly two pins "
            f"(got {len(element.pins)})."
        )


def _check_name_element(path: str, element) -> None:
    """Validate a ``NAME`` node-annotation element.

    ``NAME`` is a node-label annotation -- a live-probe
    investigation found that ADS HB only writes voltage columns to the
    ``.ds`` for nets the schematic gives an explicit name. The platform
    therefore emits NAME elements as a spec-level marker "this net
    should be exposed in the HB dataset"; the OA emitter walks NAME
    elements after placement and renames the matching OA Net to the
    spec net name.

    NAME has no electrical role. It has exactly one pin (the net to
    expose); no parameters. The net it points at is the dataset
    variable name -- there is no separate ``label`` parameter, the
    pin's ``node`` IS the label. See the per-net-uniqueness check in
    :func:`_check_name_uniqueness` for the "at most one NAME per net"
    rule.
    """

    if len(element.pins) != 1:
        raise DeviceSpecError(
            f"{path} (NAME '{element.id}') must declare exactly one pin "
            f"(got {len(element.pins)})."
        )


def _check_name_uniqueness(spec: DeviceSpec) -> None:
    """Reject specs with two NAME elements on the same net.

    Two NAMEs on one net is an authoring error: the OA rename pass
    would set ``net.name = node`` either way, but multiple NAMEs on
    one net signal a duplicated annotation and the spec author should
    consolidate to a single NAME. A live investigation showed
    multiple-NAME on one net is idempotent at the ADS level, so this
    is a *style* check, not a *correctness* one -- but a clear
    authoring error nonetheless.
    """

    seen: dict[str, str] = {}
    for elem_idx, element in enumerate(spec.elements):
        if element.type != "NAME":
            continue
        net = element.pins[0].node
        previous = seen.get(net)
        if previous is not None:
            raise DeviceSpecError(
                f"elements[{elem_idx}] (NAME '{element.id}') and "
                f"NAME '{previous}' both annotate net {net!r}; at most "
                f"one NAME element is allowed per net."
            )
        seen[net] = element.id


def _check_lumped_element(path: str, element, var_names: set[str]) -> None:
    """Validate a lumped ``R`` / ``L`` / ``C`` / ``RLC_SERIES`` element.

    Each is a two-terminal element. ``R`` (resistance) may be zero -- a
    zero-Ohm resistor is an ideal wire, used by the lossless Level-4
    circuit. ``L`` (inductance) and ``C`` (capacitance) must be strictly
    positive: a non-positive reactance is unphysical, and a zero
    capacitance is an open the spec should model by omission rather than
    by a degenerate element.

    A parameter value may be a string naming a declared VAR variable
    (``var_names``); such a reference is left for the var-resolver to
    substitute, so its numeric range check is deferred to the resolved
    spec. ``var_names`` is empty for any spec with no VAR block, in
    which case every parameter is checked numerically exactly as before.
    """

    if len(element.pins) != 2:
        raise DeviceSpecError(
            f"{path} ({element.type} '{element.id}') must declare exactly "
            f"two pins (got {len(element.pins)})."
        )
    required = _LUMPED_REQUIRED_PARAMS[element.type]
    missing = set(required) - set(element.parameters)
    if missing:
        raise DeviceSpecError(
            f"{path} ({element.type} '{element.id}') is missing required "
            f"parameter(s): {sorted(missing)}."
        )
    descr = f"{path} ({element.type} '{element.id}')"
    if "r_ohm" in required:
        r_ohm = _check_numeric_param(
            descr, element.parameters["r_ohm"], "r_ohm", var_names
        )
        if r_ohm is not None and r_ohm < 0.0:
            raise DeviceSpecError(
                f"{descr} parameter 'r_ohm' must be non-negative (got {r_ohm})."
            )
    if "l_h" in required:
        l_h = _check_numeric_param(
            descr, element.parameters["l_h"], "l_h", var_names
        )
        if l_h is not None and l_h <= 0.0:
            raise DeviceSpecError(
                f"{descr} parameter 'l_h' must be positive (got {l_h})."
            )
    if "c_f" in required:
        c_f = _check_numeric_param(
            descr, element.parameters["c_f"], "c_f", var_names
        )
        if c_f is not None and c_f <= 0.0:
            raise DeviceSpecError(
                f"{descr} parameter 'c_f' must be positive (got {c_f})."
            )


def _check_ground_nodes(spec: DeviceSpec) -> None:
    node_ids = {n.id for n in spec.nodes}
    seen: set[str] = set()
    for index, node_id in enumerate(spec.ground_nodes):
        path = f"ground_nodes[{index}]"
        if node_id not in node_ids:
            raise DeviceSpecError(f"{path} '{node_id}' not declared in nodes")
        if node_id in seen:
            raise DeviceSpecError(f"{path} '{node_id}' duplicates a previous entry.")
        seen.add(node_id)
        for port in spec.ports:
            if port.node == node_id:
                raise DeviceSpecError(
                    f"{path} '{node_id}' is also a port node; a port cannot be grounded."
                )


def _check_ports(spec: DeviceSpec) -> None:
    node_ids = {n.id for n in spec.nodes}
    nums: list[int] = []
    for port_idx, port in enumerate(spec.ports):
        path = f"ports[{port_idx}]"
        if port.num < 1:
            raise DeviceSpecError(f"{path}.num must be >= 1 (got {port.num}).")
        if port.node not in node_ids:
            raise DeviceSpecError(f"{path}.node '{port.node}' not declared in nodes")
        if port.num in nums:
            raise DeviceSpecError(f"{path}.num {port.num} duplicates a previous port.")
        nums.append(port.num)
    if not nums:
        raise DeviceSpecError("ports must declare at least one port for the analysis.")


