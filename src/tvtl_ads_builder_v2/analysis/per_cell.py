"""Per-cell query API for an N-cell TVTL cascade HB result.

Gain analysis needs per-cell quantities -- "cell 2's varactor voltage at
the signal harmonic", "cell 4's branch current at the idler". This
module resolves those queries.

DeviceSpec-driven resolution
----------------------------
The per-cell net names are NOT reconstructed from a naming template.
They are read off the *expanded* :class:`DeviceSpec`: the expander emits
the cells in cascade order, so the k-th ``MTEE`` element is cell k's
junction, the k-th ``SDD2P`` is cell k's varactor, and so on. A net name
is then whatever that element's pin actually carries. This keeps the
module decoupled from ``design/`` -- it imports only ``device_spec/``
and ``analysis``-internal code, and it works for any single-branch-per-
cell cascade whether or not it came from the design API.

:func:`build_cell_map` does the DeviceSpec walk once; the query
functions take the resulting :class:`CellMap`.

I_Probe-aware extension
-----------------------
Level-9 probed cascades carry native ADS ``I_Probe`` elements at every
cell boundary AND in series with every varactor. :func:`build_cell_map`
resolves which I_Probe sits where by net topology, not by element-id
convention: a probe whose pin shares a net with cell k's SDD2P pin 2
is cell k's *varactor probe*; a probe whose pin nets match cell k-1's
output and cell k's input is the *boundary probe* at that seam.

The query functions :func:`cell_varactor_current` and
:func:`cell_boundary_current` look those resolved probe ids up in the
HB dataset's ``<probe_id>.i`` columns -- the live-probe-confirmed form. The
resolution is empty for unprobed cascades (Level-8 and earlier); the
varactor/boundary helpers then raise :class:`CellResolutionError`.

Scope
-----
The walk supports the single-branch-per-cell cascade -- one through-line
``MLIN``, one ``MTEE``, one shunt branch ending in one ``SDD2P`` per
cell (the design-API canonical shape). A spec that does not
match raises :class:`CellResolutionError`; multi-branch cells are future
scope. I_Probe resolution is optional: probed cascades
populate the ``CellInfo`` probe-id fields; unprobed cascades leave
them ``None`` and the legacy per-cell queries are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..device_spec import DeviceSpec, Element
from .hb_dataset import HBDataset, HBSpectrum

__all__ = [
    "CellInfo",
    "CellMap",
    "CellResolutionError",
    "CellLocation",
    "build_cell_map",
    "cell_count",
    "cell_net",
    "cell_node_voltage",
    "cell_branch_current",
    "cell_boundary_current",
    "cell_varactor_current",
]

# The per-cell node locations a query may name.
CellLocation = Literal[
    "input", "junction", "output", "branch_top", "varactor_signal"
]
_CELL_LOCATIONS: frozenset[str] = frozenset(
    {"input", "junction", "output", "branch_top", "varactor_signal"}
)


class CellResolutionError(ValueError):
    """Raised when a DeviceSpec is not a resolvable single-branch cascade."""


@dataclass(frozen=True)
class CellInfo:
    """The resolved per-cell elements and nets of one unit cell.

    All fields are read off the expanded DeviceSpec; ``index`` is the
    0-based cell position in the cascade. The ``*_probe_id`` fields are
    populated when the spec carries Level-9-style ``I_Probe`` elements
    and are ``None`` otherwise (Level-8 and earlier).
    """

    index: int
    through_mlin_id: str
    mtee_id: str
    varactor_id: str
    input_net: str
    junction_net: str
    output_net: str
    branch_top_net: str
    varactor_signal_net: str
    varactor_probe_id: str | None = None
    boundary_in_probe_id: str | None = None
    boundary_out_probe_id: str | None = None


@dataclass(frozen=True)
class CellMap:
    """The per-cell resolution of a whole cascade DeviceSpec."""

    cells: tuple[CellInfo, ...]
    design_input_net: str
    design_output_net: str


def _pin_net(element: Element, pin_number: int) -> str:
    """Return the net id on ``element``'s pin ``pin_number``."""

    for pin in element.pins:
        if pin.pin == pin_number:
            return pin.node
    raise CellResolutionError(
        f"element {element.id!r} ({element.type}) has no pin {pin_number}."
    )


def build_cell_map(spec: DeviceSpec) -> CellMap:
    """Walk a cascade :class:`DeviceSpec` into a :class:`CellMap`.

    The DeviceSpec is treated as a single-branch-per-cell cascade: the
    k-th ``MTEE`` (in element order) is cell k's junction and the k-th
    ``SDD2P`` (likewise) is cell k's varactor. The cell's through-line
    MLINs are found by net topology -- the MLIN whose pin 2 lands on the
    MTEE's pin 1 (if any) is the cell's left through-line MLIN; the MLIN
    whose pin 1 lands on the MTEE's pin 2 (if any) is the cell's right
    through-line MLIN. The Level-8 design canonicals have only the left
    MLIN; the Level-9 probed canonicals have both, which is why this
    walk is topology-based rather than counting one MLIN per cell.

    The walk also *verifies* the cascade topology -- each cell's output
    must feed the next cell's input, modulo any inter-cell ``I_Probe``
    which is treated as electrically transparent (a 0-ohm short between
    its two pin nets) -- so a spec that merely has the right element
    counts but is wired differently is rejected rather than silently
    mis-resolved.

    For probed cascades (Level 9 and later) the walk also identifies
    which ``I_Probe`` element is each cell's *varactor probe* (the one
    whose pin shares a net with the cell's SDD2P pin 2) and which
    I_Probes sit on the through-line boundaries between cells (or at the
    design input / output). The resolved ids are recorded on each
    :class:`CellInfo`'s ``*_probe_id`` fields. For unprobed cascades
    those fields stay ``None``.

    Raises
    ------
    CellResolutionError
        If the spec carries no MTEE; if the SDD2P count does not match
        the MTEE count; if more than one MLIN feeds a given MTEE pin;
        or if the through-path wiring does not chain cell to cell.
    """

    mtees = [e for e in spec.elements if e.type == "MTEE"]
    sdds = [e for e in spec.elements if e.type == "SDD2P"]
    mlins = [e for e in spec.elements if e.type == "MLIN"]
    i_probes = [e for e in spec.elements if e.type == "I_Probe"]
    n_cells = len(mtees)
    if n_cells == 0:
        raise CellResolutionError(
            "DeviceSpec carries no MTEE elements -- not a TVTL cascade with "
            "tapped shunt branches."
        )
    if len(sdds) != n_cells:
        raise CellResolutionError(
            f"single-branch-per-cell cascade expected one MTEE and one "
            f"SDD2P per cell; got {n_cells} MTEE, {len(sdds)} SDD2P. A "
            f"multi-branch cell or a branch with no SDD2P is outside "
            f"the supported per-cell resolution scope."
        )

    # Index MLINs by their pin-1 and pin-2 nets so the per-cell
    # through-line MLINs can be found by topology (Level-9 has two
    # through-line MLINs per cell; Level-8 has one).
    mlin_by_pin1_net: dict[str, list[Element]] = {}
    mlin_by_pin2_net: dict[str, list[Element]] = {}
    for m in mlins:
        mlin_by_pin1_net.setdefault(_pin_net(m, 1), []).append(m)
        mlin_by_pin2_net.setdefault(_pin_net(m, 2), []).append(m)
    # All cell-junction nets, so a candidate "downstream" MLIN whose pin
    # 2 lands on another cell's junction can be recognised as that other
    # cell's *upstream* MLIN rather than this cell's right MLIN. This is
    # what distinguishes Level-8 (no right MLIN; MTEE pin 2 == next
    # cell's input) from Level-9 (right MLIN spans MTEE pin 2 to the
    # cell's output net).
    junction_nets = {_pin_net(mtee, 1) for mtee in mtees}

    cells: list[CellInfo] = []
    for k in range(n_cells):
        mtee, sdd = mtees[k], sdds[k]
        junction_net = _pin_net(mtee, 1)
        mtee_out_net = _pin_net(mtee, 2)
        branch_top_net = _pin_net(mtee, 3)

        upstream = mlin_by_pin2_net.get(junction_net, [])
        if len(upstream) > 1:
            raise CellResolutionError(
                f"cell {k}: multiple MLINs feed MTEE {mtee.id!r} pin 1 "
                f"({junction_net!r}); the cascade is ambiguous."
            )
        if upstream:
            through_mlin_id = upstream[0].id
            input_net = _pin_net(upstream[0], 1)
        else:
            # Degenerate case: no upstream MLIN -- the cell's input net
            # is the MTEE's pin 1 directly.
            through_mlin_id = mtee.id
            input_net = junction_net

        downstream_candidates = mlin_by_pin1_net.get(mtee_out_net, [])
        # A candidate is "the next cell's upstream MLIN" iff its pin 2
        # net is another cell's junction net. Drop those: they belong to
        # the next cell, not to cell k.
        downstream = [
            m for m in downstream_candidates
            if _pin_net(m, 2) not in junction_nets
        ]
        if len(downstream) > 1:
            raise CellResolutionError(
                f"cell {k}: multiple MLINs lead from MTEE {mtee.id!r} pin "
                f"2 ({mtee_out_net!r}); the cascade is ambiguous."
            )
        output_net = (
            _pin_net(downstream[0], 2) if downstream else mtee_out_net
        )

        cells.append(
            CellInfo(
                index=k,
                through_mlin_id=through_mlin_id,
                mtee_id=mtee.id,
                varactor_id=sdd.id,
                input_net=input_net,
                junction_net=junction_net,
                output_net=output_net,
                branch_top_net=branch_top_net,
                varactor_signal_net=_pin_net(sdd, 1),
            )
        )

    # Each I_Probe in the spec connects two nets as a 0-ohm short
    # (emitted as ``Short:<id> <n1> <n2> Mode=0 SaveCurrent=yes``).
    # Build the resulting net equivalence so the chain check below can
    # tolerate inter-cell probes.
    net_to_class = _net_equivalence_through_i_probes(i_probes)
    for k in range(n_cells - 1):
        cell_out = cells[k].output_net
        next_in = cells[k + 1].input_net
        if (
            cell_out != next_in
            and net_to_class.get(cell_out, cell_out) != net_to_class.get(
                next_in, next_in
            )
        ):
            raise CellResolutionError(
                f"cell {k} output {cell_out!r} does not feed cell "
                f"{k + 1} input {next_in!r} (no I_Probe bridges them); "
                f"the cascade is not chained cell to cell."
            )

    # Resolve I_Probes against per-cell topology: a probe whose pin sits
    # on a cell's SDD2P-pin-2 net is the cell's varactor probe; a probe
    # whose pin sits on a cell's input_net is that cell's boundary-in
    # probe; likewise for output_net / boundary-out.
    if i_probes:
        sdd_pin2_to_cell: dict[str, int] = {
            _pin_net(sdds[k], 2): k for k in range(n_cells)
        }
        input_net_to_cell: dict[str, int] = {
            cells[k].input_net: k for k in range(n_cells)
        }
        output_net_to_cell: dict[str, int] = {
            cells[k].output_net: k for k in range(n_cells)
        }
        varactor_probe_id: dict[int, str] = {}
        boundary_in_probe_id: dict[int, str] = {}
        boundary_out_probe_id: dict[int, str] = {}
        for probe in i_probes:
            nets = {_pin_net(probe, 1), _pin_net(probe, 2)}
            matched_var = next(
                (sdd_pin2_to_cell[n] for n in nets if n in sdd_pin2_to_cell),
                None,
            )
            if matched_var is not None:
                varactor_probe_id[matched_var] = probe.id
                continue
            for net in nets:
                if net in input_net_to_cell:
                    boundary_in_probe_id[input_net_to_cell[net]] = probe.id
                if net in output_net_to_cell:
                    boundary_out_probe_id[output_net_to_cell[net]] = probe.id
        cells = [
            _replace_cell(
                cells[k],
                varactor_probe_id=varactor_probe_id.get(k),
                boundary_in_probe_id=boundary_in_probe_id.get(k),
                boundary_out_probe_id=boundary_out_probe_id.get(k),
            )
            for k in range(n_cells)
        ]

    return CellMap(
        cells=tuple(cells),
        design_input_net=cells[0].input_net,
        design_output_net=cells[-1].output_net,
    )


def _net_equivalence_through_i_probes(
    i_probes: list[Element],
) -> dict[str, str]:
    """Return a net -> canonical-representative map merged across I_Probes.

    Each I_Probe is electrically a 0-ohm short between its two pin nets
    (emitted as ``Short:<id> <n1> <n2> Mode=0 SaveCurrent=yes``). The
    cascade chain check uses the resulting equivalence so an inter-cell
    probe is transparent to "cell k's output feeds cell k+1's input".
    """

    parent: dict[str, str] = {}

    def find(net: str) -> str:
        parent.setdefault(net, net)
        root = net
        while parent[root] != root:
            root = parent[root]
        # Path compression.
        cur = net
        while parent[cur] != root:
            parent[cur], cur = root, parent[cur]
        return root

    for probe in i_probes:
        a = _pin_net(probe, 1)
        b = _pin_net(probe, 2)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {net: find(net) for net in parent}


def _replace_cell(cell: CellInfo, **changes: object) -> CellInfo:
    """Return ``cell`` with ``**changes`` applied -- a frozen-safe update."""

    import dataclasses

    return dataclasses.replace(cell, **changes)


def cell_count(cell_map: CellMap) -> int:
    """Return the number of unit cells in the cascade."""

    return len(cell_map.cells)


def _cell(cell_map: CellMap, cell_index: int) -> CellInfo:
    """Return cell ``cell_index`` (0-based), or raise :class:`IndexError`."""

    if not 0 <= cell_index < len(cell_map.cells):
        raise IndexError(
            f"cell_index {cell_index} out of range for a "
            f"{len(cell_map.cells)}-cell cascade."
        )
    return cell_map.cells[cell_index]


def cell_net(cell_map: CellMap, cell_index: int, location: CellLocation) -> str:
    """Return the net name of one per-cell location.

    ``location`` is one of ``"input"``, ``"junction"``, ``"output"``,
    ``"branch_top"``, ``"varactor_signal"`` (the cell's SDD2P pin-1 net).
    """

    if location not in _CELL_LOCATIONS:
        raise ValueError(
            f"location {location!r} is not one of {sorted(_CELL_LOCATIONS)}."
        )
    cell = _cell(cell_map, cell_index)
    return {
        "input": cell.input_net,
        "junction": cell.junction_net,
        "output": cell.output_net,
        "branch_top": cell.branch_top_net,
        "varactor_signal": cell.varactor_signal_net,
    }[location]


def cell_node_voltage(
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    location: CellLocation,
    mix: tuple[int, ...],
) -> complex:
    """Return a per-cell node voltage at one harmonic.

    Resolves the net at ``location`` for cell ``cell_index`` from the
    DeviceSpec walk, then reads that node's spectrum at mixing tuple
    ``mix``. Raises :class:`KeyError` if the dataset has no spectrum for
    the net or no harmonic at ``mix``.
    """

    net = cell_net(cell_map, cell_index, location)
    return dataset.node_voltage(net).at_mix(mix)


def _lenient_current_spectrum(dataset: HBDataset, element_id: str) -> HBSpectrum:
    """Look up a branch-current spectrum by element id, leniently.

    The exact ADS HB current-column spelling is confirmed against a real
    dataset elsewhere; this tries the bare element id and the
    common ADS current-flavoured variants so a caller is not broken by a
    naming surprise.
    """

    for key in (
        element_id,
        f"I_{element_id}",
        f"I:{element_id}",
        f"{element_id}.i",
        f"{element_id}:1.i",
    ):
        spectrum = dataset.branch_currents.get(key)
        if spectrum is not None:
            return spectrum
    raise KeyError(
        f"no branch-current spectrum for element {element_id!r} (tried the "
        f"bare id and the I_/I:/.i variants); available: "
        f"{sorted(dataset.branch_currents)}"
    )


def cell_branch_current(
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    mix: tuple[int, ...],
) -> complex:
    """Return a per-cell shunt-branch current at one harmonic.

    The shunt branch is a single series chain, so its current is the
    current into the cell's varactor (the branch's terminating SDD2P).
    The current spectrum is looked up by the varactor element id (see
    :func:`_lenient_current_spectrum`).
    """

    cell = _cell(cell_map, cell_index)
    return _lenient_current_spectrum(dataset, cell.varactor_id).at_mix(mix)


def cell_varactor_current(
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    mix: tuple[int, ...],
) -> complex:
    """Return cell ``cell_index``'s varactor-side I_Probe current at ``mix``.

    The Level-9 probed cascades insert an ``I_Probe`` in series with each
    cell's SDD2P varactor, on the SDD2P-pin-2 side (between the
    varactor's port-1 net and the cell's stub). :func:`build_cell_map`
    has resolved which I_Probe element id sits there for each cell; this
    accessor reads its ``<probe_id>.i`` current spectrum at the requested
    mixing tuple.

    Raises
    ------
    CellResolutionError
        If the cell has no resolved varactor probe -- e.g. an unprobed
        Level-8 cascade.
    KeyError
        If the dataset has no ``<probe_id>.i`` column for the resolved
        probe id, or if the spectrum has no harmonic at ``mix``.
    """

    cell = _cell(cell_map, cell_index)
    if cell.varactor_probe_id is None:
        raise CellResolutionError(
            f"cell {cell_index}: no varactor I_Probe resolved (the spec "
            f"carries no I_Probe in series with the SDD2P pin-2 net). "
            f"cell_varactor_current is for probed cascades (Level 9+); "
            f"unprobed cascades have no per-varactor current observable."
        )
    return dataset.i_probe_current(cell.varactor_probe_id).at_mix(mix)


def cell_boundary_current(
    dataset: HBDataset,
    cell_map: CellMap,
    boundary_index: int,
    mix: tuple[int, ...],
) -> complex:
    """Return the through-line I_Probe current at boundary ``boundary_index``.

    For an ``N``-cell cascade the boundary indices run from 0 to ``N``
    inclusive::

        boundary 0      -- upstream of cell 0 (the design input probe)
        boundary k      -- between cell k - 1 and cell k  (1 <= k < N)
        boundary N      -- downstream of cell N - 1 (the design output
                           probe)

    The probe id at each boundary is resolved by :func:`build_cell_map`:
    boundary 0 is ``cells[0].boundary_in_probe_id``; an interior boundary
    k uses ``cells[k].boundary_in_probe_id`` (which is the same physical
    probe as ``cells[k - 1].boundary_out_probe_id``); boundary N uses
    ``cells[N - 1].boundary_out_probe_id``. The ``<probe_id>.i`` spectrum
    is then read at the requested mixing tuple.

    Raises
    ------
    IndexError
        If ``boundary_index`` is outside ``[0, N]``.
    CellResolutionError
        If the boundary has no resolved I_Probe -- e.g. a cascade probed
        only at the cell interiors with no boundary probes.
    KeyError
        If the dataset has no ``<probe_id>.i`` column, or if the spectrum
        has no harmonic at ``mix``.
    """

    n_cells = len(cell_map.cells)
    if not 0 <= boundary_index <= n_cells:
        raise IndexError(
            f"boundary_index {boundary_index} out of range for a "
            f"{n_cells}-cell cascade (valid: 0 .. {n_cells})."
        )
    if boundary_index < n_cells:
        probe_id = cell_map.cells[boundary_index].boundary_in_probe_id
        position = (
            "design input"
            if boundary_index == 0
            else f"between cells {boundary_index - 1} and {boundary_index}"
        )
    else:
        probe_id = cell_map.cells[-1].boundary_out_probe_id
        position = "design output"
    if probe_id is None:
        raise CellResolutionError(
            f"boundary {boundary_index} ({position}): no through-line "
            f"I_Probe resolved for this boundary. cell_boundary_current "
            f"is for probed cascades (Level 9+) with I_Probes at every "
            f"cell seam."
        )
    return dataset.i_probe_current(probe_id).at_mix(mix)


