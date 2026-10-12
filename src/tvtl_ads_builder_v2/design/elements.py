"""Design-layer specification dataclasses for the topology-driven API.

These are the *user-facing* specs a caller declares to describe a TVTL
design as data. They are deliberately distinct from the
:mod:`tvtl_ads_builder_v2.device_spec` dataclasses: a device spec is the
fully-explicit, per-instance contract the emitters consume, whereas a
design spec is the compact, topology-level declaration the expander
(:mod:`tvtl_ads_builder_v2.design.expand`) turns into a device spec.

A design declares one unit cell and a cell count; the expander
replicates the cell ``n_cells`` times, generating per-cell element
instances and per-cell net names mechanically.

Named references
----------------
Several fields are ``float | str``. A string is the name of an entry in
:attr:`TVTLDesign.parameters` (the design's VAR block); the expander
carries the name verbatim into the device spec and the existing
var-resolver substitutes the literal before emission. A float is a
literal used as-is. The one exception is an ``SDD2P_varactor``'s
``q_poly_reference`` parameter -- that string is a charge-polynomial
registry key, not a VAR reference (see :mod:`.expand`).

All dataclasses are frozen. Sequence-valued fields (``branches``,
``elements``, ``sources``) must be passed as tuples so equality and
hashing behave; the canonical builders and tests do so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..device_spec import HBSingleToneSpec, HBTwoToneSpec

__all__ = [
    "ElementSpec",
    "MicrostripSpec",
    "BranchSpec",
    "CellSpec",
    "SourceSpec",
    "SParamSpec",
    "MSUBSpec",
    "AnalysisSpec",
    "TVTLDesign",
]


@dataclass(frozen=True)
class ElementSpec:
    """A type-tagged design-layer element with named-reference parameters.

    ``type`` is a design-layer element type tag -- one of ``"R"``,
    ``"L"``, ``"C"``, ``"MLIN"``, ``"MTEE"`` or ``"SDD2P_varactor"``.
    ``params`` maps a parameter name to either a numeric literal or a
    string. A string is a VAR-block reference (resolved later), except
    an ``SDD2P_varactor``'s ``q_poly_reference``, which names a
    charge-polynomial registry entry.
    """

    type: str
    params: Mapping[str, float | str]


@dataclass(frozen=True)
class MicrostripSpec:
    """A microstrip transmission-line spec (the substrate is design-wide).

    ``width`` / ``length`` are in metres -- a numeric literal, or a
    string naming a :attr:`TVTLDesign.parameters` entry. The substrate
    is inherited from the design's :class:`MSUBSpec`, so it is not a
    field here.
    """

    width: float | str
    length: float | str


@dataclass(frozen=True)
class BranchSpec:
    """A shunt branch: a series chain of elements plus a termination.

    ``elements`` is the ordered series chain from the branch tap (an
    MTEE branch pin) toward the termination. ``termination`` is
    ``"ground"`` (the chain's far end is the ground reference -- the
    usual TVTL shunt) or ``"floating"`` (an open-circuit far end).
    """

    elements: Sequence[ElementSpec]
    termination: str = "ground"


@dataclass(frozen=True)
class CellSpec:
    """One unit cell: a through-line plus zero or more shunt branches.

    ``through_line`` is the cell's series microstrip segment.
    ``branches`` are the shunt branches tapped off the through-line,
    ordered along the cell; each is tapped through its own MTEE. A cell
    with no branches is a pure transmission-line segment.
    """

    through_line: MicrostripSpec
    branches: Sequence[BranchSpec] = ()


@dataclass(frozen=True)
class SourceSpec:
    """A ``P_1Tone`` power source at the design's input net.

    ``freq`` (Hz) and ``power_dBm`` are numeric literals or VAR-block
    references. ``num`` is the ADS port number (two tones on one input
    share ``num=1`` -- v1's two-tone HB stimulus). ``z_ohm`` is the
    source impedance. The driven node is always the design input net,
    so it is not a field.
    """

    name: str
    freq: float | str
    power_dBm: float | str
    z_ohm: float = 50.0
    num: int = 1


@dataclass(frozen=True)
class SParamSpec:
    """An S-parameter analysis: a linear frequency grid and port impedance.

    Maps to the device spec's mandatory ``analysis`` block. ``f_start_Hz``
    / ``f_stop_Hz`` bound the grid (inclusive), ``n_points`` is the
    linear sample count.
    """

    f_start_Hz: float
    f_stop_Hz: float
    n_points: int
    port_impedance_ohm: float = 50.0


@dataclass(frozen=True)
class MSUBSpec:
    """The microstrip substrate, shared across every cell.

    Maps to the device spec's ``substrate`` block. ``t_m`` (conductor
    thickness), ``rough_m`` and ``mur`` carry generic defaults; ``name``
    and ``model`` match the only substrate model the v2 stack supports.
    """

    eps_r: float
    h_m: float
    tan_delta: float
    sigma_S_per_m: float
    t_m: float = 17.5e-6
    rough_m: float = 0.0
    mur: float = 1.0
    name: str = "msub1"
    model: str = "hammerstad_kirschning"


@dataclass(frozen=True)
class AnalysisSpec:
    """The analysis controllers a design declares.

    ``s_param`` is the S-parameter analysis; it is required because the
    device spec's ``analysis`` block is mandatory. ``hb_single_tone`` /
    ``hb_two_tone`` are optional Harmonic Balance controllers (mutually
    exclusive -- the device-spec validator rejects both at once) reused
    verbatim from the device-spec schema.
    """

    s_param: SParamSpec | None = None
    hb_single_tone: HBSingleToneSpec | None = None
    hb_two_tone: HBTwoToneSpec | None = None


@dataclass(frozen=True)
class TVTLDesign:
    """A topology-driven TVTL design -- the design API's top-level object.

    ``n_cells`` identical copies of ``cell`` are cascaded by the
    expander. ``parameters`` is the design's VAR block (name -> literal);
    every named reference in ``cell`` / ``sources`` / ``analysis`` must
    resolve here. ``sources`` are the ``P_1Tone`` sources at the design
    input; ``substrate`` is shared across all cells.
    """

    n_cells: int
    cell: CellSpec
    parameters: Mapping[str, float]
    sources: Sequence[SourceSpec]
    analysis: AnalysisSpec
    substrate: MSUBSpec


