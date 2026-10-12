"""Device-spec dataclasses mirroring the YAML/JSON contract one-to-one.

Each dataclass represents a node in the document tree. ``from_dict``
classmethods reconstruct an instance from the JSON/YAML-loaded mapping.
There is no behaviour here beyond construction and round-tripping --
validation lives in ``validator.py`` and simulation lives in the sibling
modules.

The schema is intentionally minimal at this level: only the fields
required for ``TermG``, ``MLIN``, and ``MSUB`` elements with an
S-parameter analysis are present. Later levels will extend the
``Element.type`` vocabulary and ``Analysis.type``; the rest of the
schema stays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


DEVICE_SPEC_VERSION = 1


def _number_or_name(value: Any) -> float | str:
    """Coerce a ``float | str`` schema value from its JSON form.

    A string is a VAR-block variable reference and is kept verbatim;
    anything else is a numeric literal and is coerced with ``float``.
    Used by the HB-controller frequency fields and the ``P1ToneSource``
    ``freq`` / ``power_dBm`` fields, all of which accept either a
    literal number or a named reference the var-resolver substitutes.
    """

    if isinstance(value, str):
        return value
    return float(value)


@dataclass(frozen=True)
class FrequencyGrid:
    """Linear frequency grid in Hertz (start/stop inclusive, ``points`` total)."""

    start: float
    stop: float
    points: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrequencyGrid":
        return cls(
            start=float(data["start"]),
            stop=float(data["stop"]),
            points=int(data["points"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"start": float(self.start), "stop": float(self.stop), "points": int(self.points)}


@dataclass(frozen=True)
class Analysis:
    """Top-level analysis description.

    For Level 1 only ``s_parameters`` is permitted; ``harmonic_balance``
    is reserved for Level 8.
    """

    type: str
    frequency_grid_Hz: FrequencyGrid
    port_impedance_ohm: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Analysis":
        return cls(
            type=str(data["type"]),
            frequency_grid_Hz=FrequencyGrid.from_dict(data["frequency_grid_Hz"]),
            port_impedance_ohm=float(data["port_impedance_ohm"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": str(self.type),
            "frequency_grid_Hz": self.frequency_grid_Hz.to_dict(),
            "port_impedance_ohm": float(self.port_impedance_ohm),
        }


@dataclass(frozen=True)
class Substrate:
    """Substrate stack-up parameters consumed by the microstrip model."""

    name: str
    model: str
    parameters: Mapping[str, float]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Substrate":
        params = {str(key): float(value) for key, value in dict(data["parameters"]).items()}
        return cls(name=str(data["name"]), model=str(data["model"]), parameters=params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "model": str(self.model),
            "parameters": {key: float(value) for key, value in self.parameters.items()},
        }


@dataclass(frozen=True)
class Node:
    """Schematic node (a named net)."""

    id: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        return cls(id=str(data["id"]))

    def to_dict(self) -> dict[str, Any]:
        return {"id": str(self.id)}


@dataclass(frozen=True)
class ElementPin:
    """One ``(pin_number, node_id)`` association on an element."""

    pin: int
    node: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ElementPin":
        return cls(pin=int(data["pin"]), node=str(data["node"]))

    def to_dict(self) -> dict[str, Any]:
        return {"pin": int(self.pin), "node": str(self.node)}


@dataclass(frozen=True)
class Element:
    """A circuit element (TermG, MLIN, etc.) with its pin map and parameters."""

    id: str
    type: str
    pins: tuple[ElementPin, ...]
    parameters: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Element":
        pins = tuple(ElementPin.from_dict(p) for p in data["pins"])
        # A JSON round-trip turns a tuple-valued parameter (e.g. an
        # SDD2P's q_poly_coefficients) into a list; normalise it back to
        # a tuple so a spec equals its own dump/load round-trip and the
        # frozen Element carries only immutable parameter values.
        params = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in dict(data.get("parameters", {})).items()
        }
        return cls(id=str(data["id"]), type=str(data["type"]), pins=pins, parameters=params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "type": str(self.type),
            "pins": [pin.to_dict() for pin in self.pins],
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class Port:
    """An external 2-port reference: ``(num, node_id)``."""

    num: int
    node: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Port":
        return cls(num=int(data["num"]), node=str(data["node"]))

    def to_dict(self) -> dict[str, Any]:
        return {"num": int(self.num), "node": str(self.node)}


@dataclass(frozen=True)
class VarEntry:
    """One named scalar parameter: an ADS identifier bound to a number.

    ``name`` must be a valid ADS identifier (letters/digits/underscore,
    not starting with a digit); ``value`` is numeric. Symbolic
    cross-variable expressions are intentionally not supported -- see
    :class:`VarBlockSpec`.
    """

    name: str
    value: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VarEntry":
        return cls(name=str(data["name"]), value=float(data["value"]))

    def to_dict(self) -> dict[str, Any]:
        return {"name": str(self.name), "value": float(self.value)}


@dataclass(frozen=True)
class VarBlockSpec:
    """A VAR equations block: an ordered set of named scalar parameters.

    A VAR block lets a circuit define named scalars (``R``, ``L``,
    ``Cvar``, ...) centrally; an element parameter elsewhere in the spec
    references one by giving its name as the parameter value.

    VAR is **not** an emitted ADS element. ADS's VAR/VarEqn construct is
    non-functional in ADS 2026 Update 1.2 (see ``docs/LIMITATIONS.md``):
    neither the OA ``VarEqn`` cell nor the hpeesofsim ``Var``/``Vars``/
    ``Eqn`` netlist syntaxes work. Instead the var-resolver
    (:mod:`tvtl_ads_builder_v2.device_spec.var_resolver`) substitutes
    every variable-name reference in an element parameter with the
    variable's literal numeric value before emission, so both backends
    emit ordinary numeric literals and never depend on a VAR construct.

    The block is therefore a spec-level concept only: it is resolved
    away -- it never reaches a netlist or an OA workspace as an element.
    """

    name: str
    variables: tuple[VarEntry, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VarBlockSpec":
        return cls(
            name=str(data["name"]),
            variables=tuple(VarEntry.from_dict(v) for v in data["variables"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "variables": [v.to_dict() for v in self.variables],
        }


@dataclass(frozen=True)
class HBSingleToneSpec:
    """Single-tone Harmonic Balance controller configuration.

    Configures ADS's harmonic-balance solver for a one-fundamental
    nonlinear steady-state analysis: ``fundamental_freq`` (Hz) is the
    single injected tone, ``fundamental_order`` the number of harmonics
    solved, and ``max_order`` the maximum combined order. For a single
    tone there is one frequency axis, so ``max_order`` is at least
    ``fundamental_order``.

    This is a simulation analysis directive, not a connected element,
    so it lives on ``DeviceSpec`` rather than in ``elements`` -- the
    same spec-level placement ``var_block`` uses.

    ``fundamental_freq`` is either a literal in Hz or a ``str`` naming
    a variable in the device's VAR block (the named-reference lift
    removed the earlier literal-only restriction): the var-resolver substitutes the named
    reference with the variable's literal value before emission.
    ``fundamental_order`` / ``max_order`` stay ``int`` -- they are
    small fixed harmonic counts, not swept design variables, so they
    have no named-reference path.
    """

    fundamental_freq: float | str
    fundamental_order: int
    max_order: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HBSingleToneSpec":
        return cls(
            fundamental_freq=_number_or_name(data["fundamental_freq"]),
            fundamental_order=int(data["fundamental_order"]),
            max_order=int(data["max_order"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fundamental_freq": (
                self.fundamental_freq
                if isinstance(self.fundamental_freq, str)
                else float(self.fundamental_freq)
            ),
            "fundamental_order": int(self.fundamental_order),
            "max_order": int(self.max_order),
        }


@dataclass(frozen=True)
class HBTwoToneSpec:
    """Two-tone Harmonic Balance controller configuration.

    Configures ADS's harmonic-balance solver for a two-fundamental
    nonlinear steady-state analysis -- the arrangement a parametric
    amplifier study uses (a signal tone and a pump tone).
    ``fundamental_freq_1`` / ``fundamental_freq_2`` (Hz) are the two
    injected tones, ``fundamental_order_1`` / ``fundamental_order_2``
    the harmonics solved for each, and ``max_order`` the maximum
    combined intermodulation order.

    Like :class:`HBSingleToneSpec` this is a simulation analysis
    directive, not a connected element; it lives on
    ``DeviceSpec.hb_two_tone`` and is mutually exclusive with
    ``hb_single_tone`` -- a spec declares at most one HB controller.

    ``fundamental_freq_1`` / ``fundamental_freq_2`` are each either a
    literal in Hz or a ``str`` naming a variable in the device's VAR
    block (the named-reference lift removed the earlier literal-only
    restriction): the
    var-resolver substitutes a named reference with the variable's
    literal value before emission. The per-tone orders and
    ``max_order`` stay ``int`` -- small fixed harmonic counts with no
    named-reference path.
    """

    fundamental_freq_1: float | str
    fundamental_order_1: int
    fundamental_freq_2: float | str
    fundamental_order_2: int
    max_order: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HBTwoToneSpec":
        return cls(
            fundamental_freq_1=_number_or_name(data["fundamental_freq_1"]),
            fundamental_order_1=int(data["fundamental_order_1"]),
            fundamental_freq_2=_number_or_name(data["fundamental_freq_2"]),
            fundamental_order_2=int(data["fundamental_order_2"]),
            max_order=int(data["max_order"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fundamental_freq_1": (
                self.fundamental_freq_1
                if isinstance(self.fundamental_freq_1, str)
                else float(self.fundamental_freq_1)
            ),
            "fundamental_order_1": int(self.fundamental_order_1),
            "fundamental_freq_2": (
                self.fundamental_freq_2
                if isinstance(self.fundamental_freq_2, str)
                else float(self.fundamental_freq_2)
            ),
            "fundamental_order_2": int(self.fundamental_order_2),
            "max_order": int(self.max_order),
        }


@dataclass(frozen=True)
class P1ToneSource:
    """A single-tone ADS ``P_1Tone`` power source.

    ``P_1Tone`` is ADS's one-tone power source from the ``ads_sources``
    library. It is a port-like stimulus: a positive terminal driving a
    network node, a negative terminal at ground, and a single injected
    tone. A two-tone (signal + pump) stimulus -- the arrangement a
    parametric-amplifier study needs -- is two ``P1ToneSource`` entries
    on the same node, both ``num=1``; ADS's harmonic balance sums them
    as independent sources within one input port. The public v2 builder
    emits that explicit two-source arrangement directly; users who need
    ADS/HB validation should compare the generated flat deck against their
    local ADS installation rather than relying on any bundled private
    evidence archive.

    Fields
    ------
    name : str
        Source instance name, e.g. ``"SIGNAL_SRC"``. A valid ADS
        identifier, unique among a spec's sources.
    port_node : str
        The network node the source's positive terminal drives. The
        negative terminal is the global ground ``0`` -- not a schema
        field, emitted as the literal ``0`` (the ``TermG`` emitter
        does the same for its implicit ground terminal).
    num : int
        ADS port number. v1 sets every source to ``1`` so the two
        tones share one physical input port.
    freq : float | str
        The injected tone frequency in Hz, or a ``str`` naming a
        variable in the device's VAR block (the var-resolver
        substitutes it before emission).
    power_dBm : float | str
        The source power in dBm, or a VAR-block variable name. It is
        emitted inside ADS's ``polar(dbmtow(<dBm>),0)`` wrapper --
        ``dbmtow`` converts dBm to watts -- reproducing v1 verbatim.
    z_ohm : float
        Source impedance in Ohm; defaults to v1's 50 Ohm.
    phase_rad : float | str
        The injected tone phase in **radians**, or a VAR-block variable
        name. Defaults to ``0.0`` (the historical fixed-zero phase). It is
        emitted as the second argument of ADS's ``polar(mag, angle_deg)``
        after conversion to degrees, so a phase sweep (the degenerate
        phase-sensitive-amplification characterisation) drives the
        pump source's ``phase_rad`` -- e.g. via the ``pump_phase_rad`` VAR.
        At ``0.0`` the emitted ``polar(...,0)`` is byte-identical to the
        historical decks.

    ``Noise=no`` is emitted unconditionally (v1's fixed value); it is
    a solver knob, not a circuit parameter, so it is not a field --
    the same treatment :class:`HBSingleToneSpec` gives ``StatusLevel``.
    """

    name: str
    port_node: str
    num: int
    freq: float | str
    power_dBm: float | str
    z_ohm: float = 50.0
    phase_rad: float | str = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "P1ToneSource":
        return cls(
            name=str(data["name"]),
            port_node=str(data["port_node"]),
            num=int(data["num"]),
            freq=_number_or_name(data["freq"]),
            power_dBm=_number_or_name(data["power_dBm"]),
            z_ohm=float(data.get("z_ohm", 50.0)),
            phase_rad=_number_or_name(data.get("phase_rad", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "port_node": str(self.port_node),
            "num": int(self.num),
            "freq": (
                self.freq if isinstance(self.freq, str) else float(self.freq)
            ),
            "power_dBm": (
                self.power_dBm
                if isinstance(self.power_dBm, str)
                else float(self.power_dBm)
            ),
            "z_ohm": float(self.z_ohm),
            "phase_rad": (
                self.phase_rad
                if isinstance(self.phase_rad, str)
                else float(self.phase_rad)
            ),
        }


@dataclass(frozen=True)
class DeviceSpec:
    """Top-level device specification.

    ``device_spec_version`` pins the schema version this dataclass
    represents; the validator refuses anything other than the version
    declared in :data:`DEVICE_SPEC_VERSION`.

    ``ground_nodes`` (added at Level 3) lists node IDs that are the
    simulator's ground reference. The netlist emitter translates any
    pin connected to such a node to the literal ``0``; the TVTL
    analytical simulator treats them as the V = 0 reference. The field
    is a list so circuits with multiple grounded points are
    representable; it defaults to empty for Level-1/2 specs.

    ``open_terminations`` lists node IDs that are intentionally left
    electrically open -- the unconnected far end of an open-circuit
    stub, for example. Such a net carries exactly one element pin by
    design, so declaring it lets a one-pin net be told apart from an
    accidental dangling pin (the adjacency audit and the topology
    renderer both read this field). It defaults to empty. Example: the
    ``tee_open_zero_stub`` canonical circuit declares
    ``open_terminations=("n_branch_end",)`` -- the open end of its
    shunt stub.

    ``var_block`` (added at Level 5) is an optional :class:`VarBlockSpec`
    holding named scalar parameters. It is resolved away before
    emission -- the var-resolver substitutes each variable-name
    reference in an element parameter with the variable's literal value
    -- so it never reaches a netlist or an OA workspace as an element.
    It defaults to ``None``.

    ``hb_single_tone`` (added at Level 6) is an optional
    :class:`HBSingleToneSpec` -- a single-tone Harmonic Balance
    analysis controller. It is a simulation directive, not a connected
    element; when present it is emitted as an extra ``HB:`` analysis
    line alongside the S-parameter directive. It defaults to ``None``.

    ``hb_two_tone`` (added at Level 6) is an optional
    :class:`HBTwoToneSpec` -- the two-fundamental HB controller. It is
    mutually exclusive with ``hb_single_tone`` (the validator rejects a
    spec that sets both); it is emitted the same additive way. It
    defaults to ``None``.

    ``sources`` (added at Level 7) is a tuple of :class:`P1ToneSource`
    -- ADS ``P_1Tone`` power sources. A source is a port-like stimulus
    that drives a network node; it is not a member of ``elements``
    (its dataclass shape differs from :class:`Element`), so it lives
    on its own spec-level field, like the HB controllers. It is
    emitted additively. It defaults to an empty tuple.
    """

    device_spec_version: int
    analysis: Analysis
    substrate: Substrate
    nodes: tuple[Node, ...]
    elements: tuple[Element, ...]
    ports: tuple[Port, ...]
    label: str = ""
    ground_nodes: tuple[str, ...] = ()
    open_terminations: tuple[str, ...] = ()
    var_block: VarBlockSpec | None = None
    hb_single_tone: HBSingleToneSpec | None = None
    hb_two_tone: HBTwoToneSpec | None = None
    sources: tuple[P1ToneSource, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeviceSpec":
        return cls(
            device_spec_version=int(data.get("device_spec_version", 0)),
            analysis=Analysis.from_dict(data["analysis"]),
            substrate=Substrate.from_dict(data["substrate"]),
            nodes=tuple(Node.from_dict(n) for n in data["nodes"]),
            elements=tuple(Element.from_dict(e) for e in data["elements"]),
            ports=tuple(Port.from_dict(p) for p in data["ports"]),
            label=str(data.get("label", "")),
            ground_nodes=tuple(str(g) for g in data.get("ground_nodes", ())),
            open_terminations=tuple(
                str(n) for n in data.get("open_terminations", ())
            ),
            var_block=(
                VarBlockSpec.from_dict(data["var_block"])
                if data.get("var_block")
                else None
            ),
            hb_single_tone=(
                HBSingleToneSpec.from_dict(data["hb_single_tone"])
                if data.get("hb_single_tone")
                else None
            ),
            hb_two_tone=(
                HBTwoToneSpec.from_dict(data["hb_two_tone"])
                if data.get("hb_two_tone")
                else None
            ),
            sources=tuple(
                P1ToneSource.from_dict(s) for s in data.get("sources", ())
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "device_spec_version": int(self.device_spec_version),
            "analysis": self.analysis.to_dict(),
            "substrate": self.substrate.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
            "elements": [e.to_dict() for e in self.elements],
            "ports": [p.to_dict() for p in self.ports],
        }
        if self.label:
            out["label"] = str(self.label)
        if self.ground_nodes:
            out["ground_nodes"] = [str(g) for g in self.ground_nodes]
        if self.open_terminations:
            out["open_terminations"] = [str(n) for n in self.open_terminations]
        if self.var_block is not None:
            out["var_block"] = self.var_block.to_dict()
        if self.hb_single_tone is not None:
            out["hb_single_tone"] = self.hb_single_tone.to_dict()
        if self.hb_two_tone is not None:
            out["hb_two_tone"] = self.hb_two_tone.to_dict()
        if self.sources:
            out["sources"] = [s.to_dict() for s in self.sources]
        return out


