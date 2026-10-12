"""Tests for the per-cell query API (analysis/per_cell.py).

ADS-free: build_cell_map walks the expanded DeviceSpec; the query
functions read synthetic in-memory HBDatasets. The Level-8 design-API
canonicals are the realistic cascade specs.
"""

from __future__ import annotations

import pytest

from tvtl_ads_builder_v2.analysis import (
    HBDataset,
    HBHarmonic,
    HBSpectrum,
    IDLER_MIX_2TONE,
    PUMP_MIX_2TONE,
    SIGNAL_MIX_2TONE,
    build_cell_map,
    cell_boundary_current,
    cell_branch_current,
    cell_count,
    cell_net,
    cell_node_voltage,
    cell_varactor_current,
)
from tvtl_ads_builder_v2.analysis.per_cell import CellResolutionError
from tvtl_ads_builder_v2.design.canonicals import LEVEL_8_CANONICAL_CIRCUITS
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    LEVEL_3_CANONICAL_CIRCUITS,
    LEVEL_9_CANONICAL_CIRCUITS,
)

_SPEC_3CELL = LEVEL_8_CANONICAL_CIRCUITS[0]
_SPEC_5CELL = LEVEL_8_CANONICAL_CIRCUITS[1]
_SPEC_L9_2CELL = LEVEL_9_CANONICAL_CIRCUITS[0]
_SPEC_L9_3CELL = LEVEL_9_CANONICAL_CIRCUITS[1]

_LOCATIONS = ("input", "junction", "output", "branch_top", "varactor_signal")


# --------------------------------------------------------------------------
# Synthetic HBDataset keyed by a cascade's actual nets
# --------------------------------------------------------------------------


def _spectrum(signal: complex, idler: complex = 0j, pump: complex = 0j) -> HBSpectrum:
    return HBSpectrum(
        quantity="",
        harmonics=(
            HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=signal),
            HBHarmonic(mix=(-1, 1), frequency_Hz=3e9, amplitude=idler),
            HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=pump),
        ),
    )


def _hb_for(cell_map, *, current_key=lambda vid: vid):
    """Return ``(hb, signal_by_net)`` for a cascade.

    Each distinct net gets one spectrum (a cascade shares a net at every
    cell seam -- cell k's output IS cell k+1's input -- so seeding has
    to be per-net, not per (cell, location)). ``signal_by_net`` records
    the signal-harmonic amplitude assigned to each net; idler = signal*j
    and pump = signal*2. Branch currents are keyed via ``current_key`` so
    the lenient varactor-id lookup can be exercised with ADS spellings.
    """

    signal_by_net: dict[str, complex] = {}
    for k in range(cell_count(cell_map)):
        for location in _LOCATIONS:
            net = cell_net(cell_map, k, location)
            if net not in signal_by_net:
                signal_by_net[net] = complex(len(signal_by_net) + 1)
    node_voltages = {
        net: _spectrum(signal=sig, idler=sig * 1j, pump=sig * 2)
        for net, sig in signal_by_net.items()
    }
    branch_currents = {
        current_key(cell_map.cells[k].varactor_id): _spectrum(
            signal=complex(0, -(k + 1))
        )
        for k in range(cell_count(cell_map))
    }
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )
    return hb, signal_by_net


# --------------------------------------------------------------------------
# build_cell_map
# --------------------------------------------------------------------------


def test_build_cell_map_three_cell() -> None:
    """The 3-cell canonical resolves to three cells."""

    cm = build_cell_map(_SPEC_3CELL)
    assert cell_count(cm) == 3
    assert cm.design_input_net == "n_input"
    assert cm.design_output_net == "n_output"


def test_build_cell_map_five_cell() -> None:
    """The 5-cell canonical resolves to five cells -- the scaling check."""

    cm = build_cell_map(_SPEC_5CELL)
    assert cell_count(cm) == 5


def test_cell_info_fields_are_consistent() -> None:
    """Each CellInfo's nets are read off the right elements."""

    cm = build_cell_map(_SPEC_3CELL)
    cell0 = cm.cells[0]
    assert cell0.index == 0
    assert cell0.through_mlin_id == "MLIN_c0"
    assert cell0.mtee_id == "MTEE_c0_0"
    assert cell0.varactor_id == "SDD_c0_b0_3"
    assert cell0.input_net == "n_input"
    assert cell0.junction_net == "n_cell0_junc"
    assert cell0.varactor_signal_net == "n_cell0_br0_n3"


def test_cascade_is_chained_cell_to_cell() -> None:
    """Cell k's output net is cell k+1's input net."""

    cm = build_cell_map(_SPEC_5CELL)
    for k in range(cell_count(cm) - 1):
        assert cm.cells[k].output_net == cm.cells[k + 1].input_net
    assert cm.cells[-1].output_net == "n_output"


def test_build_cell_map_rejects_spec_with_no_mtee() -> None:
    """A through-line-only spec (no MTEE) is not a tapped-branch cascade."""

    spec = LEVEL_1_CANONICAL_CIRCUITS[0]  # TermG + MLIN + TermG
    with pytest.raises(CellResolutionError, match="no MTEE"):
        build_cell_map(spec)


def test_build_cell_map_rejects_inconsistent_element_counts() -> None:
    """An MTEE circuit with no SDD2P fails the one-per-cell count check."""

    spec = LEVEL_3_CANONICAL_CIRCUITS[0]  # MTEE + stub MLIN, no SDD2P
    with pytest.raises(CellResolutionError, match="per cell"):
        build_cell_map(spec)


# --------------------------------------------------------------------------
# cell_net
# --------------------------------------------------------------------------


def test_cell_net_all_locations_resolve() -> None:
    """Every documented location resolves to a declared net."""

    cm = build_cell_map(_SPEC_3CELL)
    declared = {n.id for n in _SPEC_3CELL.nodes}
    for k in range(cell_count(cm)):
        for location in _LOCATIONS:
            assert cell_net(cm, k, location) in declared


def test_cell_net_rejects_unknown_location() -> None:
    cm = build_cell_map(_SPEC_3CELL)
    with pytest.raises(ValueError, match="location"):
        cell_net(cm, 0, "nowhere")  # type: ignore[arg-type]


def test_cell_net_rejects_out_of_range_cell() -> None:
    cm = build_cell_map(_SPEC_3CELL)
    with pytest.raises(IndexError):
        cell_net(cm, 7, "junction")


# --------------------------------------------------------------------------
# cell_node_voltage
# --------------------------------------------------------------------------


def test_cell_node_voltage_returns_seeded_harmonic() -> None:
    """A per-cell node voltage reads back the seeded harmonic amplitude."""

    cm = build_cell_map(_SPEC_3CELL)
    hb, signal_by_net = _hb_for(cm)
    for k in range(cell_count(cm)):
        for location in _LOCATIONS:
            net = cell_net(cm, k, location)
            seeded = signal_by_net[net]
            assert cell_node_voltage(hb, cm, k, location, SIGNAL_MIX_2TONE) == seeded
            assert cell_node_voltage(hb, cm, k, location, IDLER_MIX_2TONE) == (
                seeded * 1j
            )
            assert cell_node_voltage(hb, cm, k, location, PUMP_MIX_2TONE) == (
                seeded * 2
            )


def test_cell_output_and_next_input_query_the_same_net() -> None:
    """Cell k's output and cell k+1's input are one shared net -- same value."""

    cm = build_cell_map(_SPEC_3CELL)
    hb, _ = _hb_for(cm)
    for k in range(cell_count(cm) - 1):
        assert cell_node_voltage(hb, cm, k, "output", SIGNAL_MIX_2TONE) == (
            cell_node_voltage(hb, cm, k + 1, "input", SIGNAL_MIX_2TONE)
        )


def test_cell_node_voltage_five_cell() -> None:
    """Per-cell queries scale to the 5-cell cascade."""

    cm = build_cell_map(_SPEC_5CELL)
    hb, signal_by_net = _hb_for(cm)
    assert cell_count(cm) == 5
    assert cell_node_voltage(hb, cm, 4, "input", SIGNAL_MIX_2TONE) == (
        signal_by_net[cell_net(cm, 4, "input")]
    )


def test_cell_node_voltage_missing_net_raises() -> None:
    """Querying a cell whose net is absent from the dataset raises KeyError."""

    cm = build_cell_map(_SPEC_3CELL)
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9), node_voltages={}, branch_currents={}
    )
    with pytest.raises(KeyError):
        cell_node_voltage(hb, cm, 0, "junction", SIGNAL_MIX_2TONE)


# --------------------------------------------------------------------------
# cell_branch_current
# --------------------------------------------------------------------------


def test_cell_branch_current_returns_seeded_harmonic() -> None:
    """The per-cell branch current reads back via the bare varactor id."""

    cm = build_cell_map(_SPEC_3CELL)
    hb, _ = _hb_for(cm)
    for k in range(cell_count(cm)):
        assert cell_branch_current(hb, cm, k, SIGNAL_MIX_2TONE) == (
            complex(0, -(k + 1))
        )


def test_cell_branch_current_lenient_lookup_variants() -> None:
    """The branch-current lookup tolerates ADS current-name variants."""

    cm = build_cell_map(_SPEC_3CELL)
    for spelling in (lambda v: f"I_{v}", lambda v: f"{v}.i", lambda v: f"I:{v}"):
        hb, _ = _hb_for(cm, current_key=spelling)
        assert cell_branch_current(hb, cm, 1, SIGNAL_MIX_2TONE) == complex(0, -2)


def test_cell_branch_current_missing_raises() -> None:
    """A varactor with no current spectrum raises KeyError."""

    cm = build_cell_map(_SPEC_3CELL)
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9), node_voltages={}, branch_currents={}
    )
    with pytest.raises(KeyError, match="branch-current"):
        cell_branch_current(hb, cm, 0, SIGNAL_MIX_2TONE)


# --------------------------------------------------------------------------
# Level-9 build_cell_map: I_Probe resolution by net topology
# --------------------------------------------------------------------------


def test_build_cell_map_resolves_level9_two_cell() -> None:
    """Level-9 two-cell cascade walks cleanly despite multiple MLINs/cell."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    assert cell_count(cm) == 2
    # Level-9's per-cell ids follow the canonical naming.
    assert cm.cells[0].through_mlin_id == "LINE_LEFT_C1"
    assert cm.cells[0].mtee_id == "TEE_C1"
    assert cm.cells[0].varactor_id == "VARACTOR_C1"
    assert cm.cells[1].through_mlin_id == "LINE_LEFT_C2"
    # The cascade's design-output net is LINE_RIGHT pin 2 of the last
    # cell -- the right-MLIN-aware walk picks it up.
    assert cm.cells[1].output_net == "n_c2_out"


def test_build_cell_map_resolves_i_probe_ids_for_level9() -> None:
    """Every I_Probe in the Level-9 canonical is resolved by topology."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # Varactor probes: pin sits on the SDD2P-pin-2 net.
    assert cm.cells[0].varactor_probe_id == "IPR_VAR_C1"
    assert cm.cells[1].varactor_probe_id == "IPR_VAR_C2"
    # Boundary probes: IPR_IN upstream of cell 0; IPR_C1C2 between cells
    # 0 and 1 (so it is both cell 0's boundary-out and cell 1's
    # boundary-in); IPR_OUT downstream of cell 1.
    assert cm.cells[0].boundary_in_probe_id == "IPR_IN"
    assert cm.cells[0].boundary_out_probe_id == "IPR_C1C2"
    assert cm.cells[1].boundary_in_probe_id == "IPR_C1C2"
    assert cm.cells[1].boundary_out_probe_id == "IPR_OUT"


def test_build_cell_map_resolves_level9_three_cell() -> None:
    """The three-cell Level-9 canonical resolves every probe slot."""

    cm = build_cell_map(_SPEC_L9_3CELL)
    assert cell_count(cm) == 3
    for k, expected_var in enumerate(
        ("IPR_VAR_C1", "IPR_VAR_C2", "IPR_VAR_C3")
    ):
        assert cm.cells[k].varactor_probe_id == expected_var
    # Cell 0's boundary-in is IPR_IN; cells 1 and 2 inherit the inter-
    # cell probes from their upstream cells' boundary-out. Cell 2's
    # boundary-out is IPR_OUT.
    assert cm.cells[0].boundary_in_probe_id == "IPR_IN"
    assert cm.cells[0].boundary_out_probe_id == "IPR_C1C2"
    assert cm.cells[1].boundary_in_probe_id == "IPR_C1C2"
    assert cm.cells[1].boundary_out_probe_id == "IPR_C2C3"
    assert cm.cells[2].boundary_in_probe_id == "IPR_C2C3"
    assert cm.cells[2].boundary_out_probe_id == "IPR_OUT"


def test_inter_cell_iprobe_is_transparent_in_chain_check() -> None:
    """The chain check accepts an I_Probe between cells (a 0-ohm short)."""

    # An inter-cell probe means cells[k].output_net != cells[k+1].input_net
    # in Level-9; the chain check must accept that *because* an I_Probe
    # spans them.
    cm = build_cell_map(_SPEC_L9_2CELL)
    assert cm.cells[0].output_net == "n_c1_out"
    assert cm.cells[1].input_net == "n_pc1c2"
    assert cm.cells[0].output_net != cm.cells[1].input_net


def test_unprobed_cascade_has_none_probe_ids() -> None:
    """An unprobed Level-8 cascade leaves probe-id fields ``None``."""

    cm = build_cell_map(_SPEC_3CELL)
    for cell in cm.cells:
        assert cell.varactor_probe_id is None
        assert cell.boundary_in_probe_id is None
        assert cell.boundary_out_probe_id is None


# --------------------------------------------------------------------------
# cell_varactor_current
# --------------------------------------------------------------------------


def _level9_hb(cell_map) -> tuple[HBDataset, dict[str, complex]]:
    """Return ``(hb, expected)`` for the Level-9 cascade.

    Seeds one distinct signal-harmonic amplitude per resolved probe id so
    each lookup can verify it reads the right column. The dataset uses
    the ``<probe_id>.i`` column-name convention confirmed against a live
    HB run.
    """

    expected: dict[str, complex] = {}
    branch_currents: dict[str, HBSpectrum] = {}
    counter = 0
    for k, cell in enumerate(cell_map.cells):
        for probe_id in (
            cell.varactor_probe_id,
            cell.boundary_in_probe_id,
            cell.boundary_out_probe_id,
        ):
            if probe_id is None or probe_id in expected:
                continue
            counter += 1
            amplitude = complex(counter, -counter)
            expected[probe_id] = amplitude
            branch_currents[f"{probe_id}.i"] = _spectrum(signal=amplitude)
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9),
        node_voltages={},
        branch_currents=branch_currents,
    )
    return hb, expected


def test_cell_varactor_current_reads_resolved_probe_for_each_cell() -> None:
    """Every cell's varactor probe reads back the seeded harmonic."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb, expected = _level9_hb(cm)
    for k, cell in enumerate(cm.cells):
        got = cell_varactor_current(hb, cm, k, SIGNAL_MIX_2TONE)
        assert got == expected[cell.varactor_probe_id]


def test_cell_varactor_current_three_cell_scales() -> None:
    """Three-cell cascade exercises three distinct varactor probes."""

    cm = build_cell_map(_SPEC_L9_3CELL)
    hb, expected = _level9_hb(cm)
    for k in range(cell_count(cm)):
        got = cell_varactor_current(hb, cm, k, SIGNAL_MIX_2TONE)
        assert got == expected[cm.cells[k].varactor_probe_id]


def test_cell_varactor_current_on_unprobed_cascade_raises() -> None:
    """An unprobed Level-8 cascade raises CellResolutionError."""

    cm = build_cell_map(_SPEC_3CELL)
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9), node_voltages={}, branch_currents={}
    )
    with pytest.raises(CellResolutionError, match="varactor I_Probe"):
        cell_varactor_current(hb, cm, 0, SIGNAL_MIX_2TONE)


def test_cell_varactor_current_missing_column_raises_keyerror() -> None:
    """A probed cascade with no ``<id>.i`` column raises KeyError."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9), node_voltages={}, branch_currents={}
    )
    with pytest.raises(KeyError, match="IPR_VAR_C1"):
        cell_varactor_current(hb, cm, 0, SIGNAL_MIX_2TONE)


def test_cell_varactor_current_out_of_range_cell_raises_indexerror() -> None:
    """A cell index past the cascade end raises IndexError."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb, _ = _level9_hb(cm)
    with pytest.raises(IndexError):
        cell_varactor_current(hb, cm, 9, SIGNAL_MIX_2TONE)


# --------------------------------------------------------------------------
# cell_boundary_current
# --------------------------------------------------------------------------


def test_cell_boundary_current_reads_each_boundary() -> None:
    """Boundaries 0..N read the seeded harmonic of the resolved probe."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb, expected = _level9_hb(cm)
    # Boundary 0 = IPR_IN (cell 0's boundary-in).
    assert cell_boundary_current(hb, cm, 0, SIGNAL_MIX_2TONE) == expected["IPR_IN"]
    # Boundary 1 = IPR_C1C2 (cell 1's boundary-in).
    assert cell_boundary_current(hb, cm, 1, SIGNAL_MIX_2TONE) == expected["IPR_C1C2"]
    # Boundary 2 = IPR_OUT (cell 1's boundary-out).
    assert cell_boundary_current(hb, cm, 2, SIGNAL_MIX_2TONE) == expected["IPR_OUT"]


def test_cell_boundary_current_three_cell_indexes_all_seams() -> None:
    """Three-cell cascade exposes boundaries 0..3 covering every seam."""

    cm = build_cell_map(_SPEC_L9_3CELL)
    hb, expected = _level9_hb(cm)
    assert cell_boundary_current(hb, cm, 0, SIGNAL_MIX_2TONE) == expected["IPR_IN"]
    assert cell_boundary_current(hb, cm, 1, SIGNAL_MIX_2TONE) == expected["IPR_C1C2"]
    assert cell_boundary_current(hb, cm, 2, SIGNAL_MIX_2TONE) == expected["IPR_C2C3"]
    assert cell_boundary_current(hb, cm, 3, SIGNAL_MIX_2TONE) == expected["IPR_OUT"]


def test_cell_boundary_current_inter_cell_is_one_physical_probe() -> None:
    """The inter-cell probe seen from cell k-1's right == cell k's left."""

    cm = build_cell_map(_SPEC_L9_3CELL)
    hb, _ = _level9_hb(cm)
    # IPR_C1C2 sits between cells 0 and 1; querying boundary 1 reads the
    # same physical probe whether the implementation routes via cell 0's
    # boundary_out or cell 1's boundary_in.
    boundary_1 = cell_boundary_current(hb, cm, 1, SIGNAL_MIX_2TONE)
    # The IPR_C1C2 is recorded as both cells[0].boundary_out_probe_id
    # and cells[1].boundary_in_probe_id; verify directly.
    assert cm.cells[0].boundary_out_probe_id == cm.cells[1].boundary_in_probe_id
    assert boundary_1 == hb.i_probe_current(
        cm.cells[0].boundary_out_probe_id
    ).at_mix(SIGNAL_MIX_2TONE)


def test_cell_boundary_current_out_of_range_raises_indexerror() -> None:
    """Boundary index outside ``[0, N]`` raises IndexError."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb, _ = _level9_hb(cm)
    with pytest.raises(IndexError, match="boundary_index"):
        cell_boundary_current(hb, cm, -1, SIGNAL_MIX_2TONE)
    with pytest.raises(IndexError, match="boundary_index"):
        cell_boundary_current(hb, cm, 99, SIGNAL_MIX_2TONE)


def test_cell_boundary_current_on_unprobed_cascade_raises() -> None:
    """An unprobed Level-8 cascade raises CellResolutionError."""

    cm = build_cell_map(_SPEC_3CELL)
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9), node_voltages={}, branch_currents={}
    )
    with pytest.raises(CellResolutionError, match="through-line"):
        cell_boundary_current(hb, cm, 0, SIGNAL_MIX_2TONE)


