"""Tests for the HB dataset reader (analysis/hb_dataset.py).

ADS-free: every test builds a synthetic in-memory RawDataset -- no
real ``.ds`` file and no ADS runtime. The structuring, harmonic
indexing, voltage/current classification, and spectrum accessors are
all pure Python.
"""

from __future__ import annotations

import pytest

from tvtl_ads_builder_v2.analysis import (
    HBDataset,
    HBDatasetError,
    HBHarmonic,
    HBSpectrum,
    IDLER_MIX_2TONE,
    PUMP_MIX_2TONE,
    SIGNAL_MIX_2TONE,
    hb_dataset_from_raw,
    read_hb_dataset,
)
from tvtl_ads_builder_v2.simulate.parser import RawDataset, RawVarBlock


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _hb_block(name: str = "HB1", *, freq_column: bool = True) -> RawVarBlock:
    """A synthetic two-tone HB varblock: signal, pump, idler, DC rows."""

    columns: dict[str, tuple[complex, ...]] = {
        "Mix[1]": (1 + 0j, 0 + 0j, -1 + 0j, 0 + 0j),
        "Mix[2]": (0 + 0j, 1 + 0j, 1 + 0j, 0 + 0j),
    }
    if freq_column:
        columns["freq"] = (3e9 + 0j, 6e9 + 0j, 3e9 + 0j, 0j)
    columns["n_input"] = (2.0 + 0j, 0.5 + 0j, 0.1 + 0j, 0.0 + 0j)
    columns["n_cell0_junc"] = (1.5 + 0j, 0.4 + 0j, 0.2 + 0j, 0.0 + 0j)
    columns["I_SDD_c0_b0_3"] = (0.01 + 0j, 0.02 + 0j, 0.005 + 0j, 0.0 + 0j)
    return RawVarBlock(name=name, columns=columns)


def _raw(*blocks: RawVarBlock) -> RawDataset:
    return RawDataset(blocks=tuple(blocks))


# --------------------------------------------------------------------------
# RawDataset / RawVarBlock
# --------------------------------------------------------------------------


def test_raw_dataset_block_names() -> None:
    """RawDataset.block_names returns every varblock name in order."""

    raw = _raw(_hb_block("HB1"), RawVarBlock(name="DC", columns={}))
    assert raw.block_names() == ("HB1", "DC")


def test_raw_dataset_is_frozen() -> None:
    """RawDataset / RawVarBlock are immutable value objects."""

    raw = _raw(_hb_block())
    with pytest.raises((AttributeError, TypeError)):
        raw.blocks = ()  # type: ignore[misc]


# --------------------------------------------------------------------------
# hb_dataset_from_raw -- structuring
# --------------------------------------------------------------------------


def test_structures_voltages_and_currents() -> None:
    """Columns split into node voltages and branch currents by name."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    assert set(hb.node_voltages) == {"n_input", "n_cell0_junc"}
    assert set(hb.branch_currents) == {"I_SDD_c0_b0_3"}
    assert hb.fundamental_freqs_Hz == (3e9, 6e9)


def test_mixing_tuples_are_indexed() -> None:
    """Each harmonic carries its (m, n) mixing tuple and frequency."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    spectrum = hb.node_voltage("n_input")
    by_mix = {h.mix: h for h in spectrum.harmonics}
    assert set(by_mix) == {(1, 0), (0, 1), (-1, 1), (0, 0)}
    assert by_mix[(1, 0)].frequency_Hz == 3e9
    assert by_mix[(0, 1)].frequency_Hz == 6e9
    assert by_mix[(-1, 1)].frequency_Hz == 3e9


def test_signal_pump_idler_amplitudes() -> None:
    """The named two-tone mixing products read back the seeded values."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    v_in = hb.node_voltage("n_input")
    assert v_in.at_mix(SIGNAL_MIX_2TONE) == 2.0 + 0j
    assert v_in.at_mix(PUMP_MIX_2TONE) == 0.5 + 0j
    assert v_in.at_mix(IDLER_MIX_2TONE) == 0.1 + 0j


def test_frequency_derived_when_no_freq_column() -> None:
    """With no freq column the frequency is m*f1 + n*f2 from the mix tuple."""

    hb = hb_dataset_from_raw(_raw(_hb_block(freq_column=False)), (3e9, 6e9))
    by_mix = {h.mix: h for h in hb.node_voltage("n_input").harmonics}
    assert by_mix[(1, 0)].frequency_Hz == pytest.approx(3e9)
    assert by_mix[(0, 1)].frequency_Hz == pytest.approx(6e9)
    assert by_mix[(-1, 1)].frequency_Hz == pytest.approx(3e9)  # 6e9 - 3e9
    assert by_mix[(0, 0)].frequency_Hz == pytest.approx(0.0)


def test_mix_column_index_and_paren_spellings() -> None:
    """Mix(1) / Mix1 spellings are recognised alongside Mix[1]."""

    block = RawVarBlock(
        name="HB1",
        columns={
            "Mix(1)": (1 + 0j, 0 + 0j),
            "Mix(2)": (0 + 0j, 1 + 0j),
            "n_out": (3.0 + 0j, 1.0 + 0j),
        },
    )
    hb = hb_dataset_from_raw(_raw(block), (3e9, 6e9))
    assert hb.node_voltage("n_out").at_mix((1, 0)) == 3.0 + 0j


def test_hb_block_chosen_among_several_varblocks() -> None:
    """The HB block is the one carrying Mix columns, not a sibling block."""

    dc_block = RawVarBlock(name="DC", columns={"node": (1 + 0j,)})
    hb = hb_dataset_from_raw(_raw(dc_block, _hb_block("HB1")), (3e9, 6e9))
    assert set(hb.node_voltages) == {"n_input", "n_cell0_junc"}


def test_no_hb_block_raises() -> None:
    """A dataset with no Mix column is not an HB result."""

    plain = RawVarBlock(name="SP", columns={"freq": (1e9 + 0j,), "S": (0j,)})
    with pytest.raises(HBDatasetError, match="no Harmonic-Balance varblock"):
        hb_dataset_from_raw(_raw(plain), (3e9, 6e9))


def test_empty_fundamentals_rejected() -> None:
    """At least one fundamental tone must be supplied."""

    with pytest.raises(HBDatasetError, match="at least one tone"):
        hb_dataset_from_raw(_raw(_hb_block()), ())


# --------------------------------------------------------------------------
# HBSpectrum accessors
# --------------------------------------------------------------------------


def test_at_mix_missing_raises_keyerror() -> None:
    """Querying a mixing tuple the spectrum lacks raises KeyError."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    with pytest.raises(KeyError, match="no harmonic at mix"):
        hb.node_voltage("n_input").at_mix((5, 5))


def test_at_frequency_sums_collisions() -> None:
    """Two mixing tuples at one frequency sum (signal + idler at 3 GHz)."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    # signal (1,0)=2.0 and idler (-1,1)=0.1 both land at 3 GHz.
    assert hb.node_voltage("n_input").at_frequency(3e9) == pytest.approx(2.1)


def test_at_frequency_first_match_without_summing() -> None:
    """sum_collisions=False returns the first matching harmonic only."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    value = hb.node_voltage("n_input").at_frequency(3e9, sum_collisions=False)
    assert value == 2.0 + 0j


def test_at_frequency_absent_returns_zero() -> None:
    """A frequency no harmonic lands on returns 0j, not an error."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    assert hb.node_voltage("n_input").at_frequency(99e9) == 0j


def test_hbspectrum_constructed_directly() -> None:
    """HBSpectrum is a usable public value object."""

    spectrum = HBSpectrum(
        quantity="probe",
        harmonics=(
            HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=1 + 1j),
        ),
    )
    assert spectrum.at_mix((1, 0)) == 1 + 1j


# --------------------------------------------------------------------------
# HBDataset accessors
# --------------------------------------------------------------------------


def test_node_voltage_missing_raises() -> None:
    """Querying a net with no spectrum raises KeyError listing the nets."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    with pytest.raises(KeyError, match="no node-voltage spectrum"):
        hb.node_voltage("n_nonexistent")


def test_branch_current_missing_raises() -> None:
    """Querying an absent branch current raises KeyError."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    with pytest.raises(KeyError, match="no branch-current spectrum"):
        hb.branch_current("I_nonexistent")


def test_branch_current_accessor_returns_spectrum() -> None:
    """The branch-current accessor returns the seeded current spectrum."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    current = hb.branch_current("I_SDD_c0_b0_3")
    assert current.at_mix(PUMP_MIX_2TONE) == 0.02 + 0j


def test_hbdataset_default_sweep_point_is_empty() -> None:
    """A single-point HB result carries an empty sweep_point."""

    hb = hb_dataset_from_raw(_raw(_hb_block()), (3e9, 6e9))
    assert hb.sweep_point == {}


# --------------------------------------------------------------------------
# Named mixing-product constants
# --------------------------------------------------------------------------


def test_named_mix_constants() -> None:
    """The two-tone mix constants follow the (signal, pump) ordering."""

    assert SIGNAL_MIX_2TONE == (1, 0)
    assert PUMP_MIX_2TONE == (0, 1)
    assert IDLER_MIX_2TONE == (-1, 1)  # idler at f_pump - f_signal


# --------------------------------------------------------------------------
# read_hb_dataset (file-path entry point)
# --------------------------------------------------------------------------


def test_read_hb_dataset_missing_file_raises(tmp_path) -> None:
    """The file-path entry point raises FileNotFoundError for a missing .ds."""

    with pytest.raises(FileNotFoundError):
        read_hb_dataset(tmp_path / "absent.ds", (3e9, 6e9))


# --------------------------------------------------------------------------
# I_Probe current columns
# --------------------------------------------------------------------------


def _hb_block_with_i_probe_currents(
    *, with_through_probes: bool = True
) -> RawVarBlock:
    """A synthetic HB varblock carrying ``<id>.i`` I_Probe current columns.

    Mirrors the column shape established
    against a live HB run: each I_Probe element produces a column whose
    name is the bare element id with the ``.i`` suffix
    appended. The block here mirrors what the
    Level-9 two-cell probed cascade writes -- ``IPR_IN.i`` /
    ``IPR_C1C2.i`` / ``IPR_OUT.i`` on the through-line plus
    ``IPR_VAR_C1.i`` / ``IPR_VAR_C2.i`` on the varactor branches.
    """

    columns: dict[str, tuple[complex, ...]] = {
        "Mix[1]": (1 + 0j, 0 + 0j, -1 + 0j),
        "Mix[2]": (0 + 0j, 1 + 0j, 1 + 0j),
        "freq": (3e9 + 0j, 6e9 + 0j, 3e9 + 0j),
        "n_input": (1.0 + 0j, 0.5 + 0j, 0.1 + 0j),
        "IPR_VAR_C1.i": (0.10 + 0j, 0.02 + 0j, 0.005 + 0j),
        "IPR_VAR_C2.i": (0.20 + 0j, 0.04 + 0j, 0.010 + 0j),
    }
    if with_through_probes:
        columns.update({
            "IPR_IN.i": (0.50 + 0j, 0.10 + 0j, 0.025 + 0j),
            "IPR_C1C2.i": (0.40 + 0j, 0.08 + 0j, 0.020 + 0j),
            "IPR_OUT.i": (0.30 + 0j, 0.06 + 0j, 0.015 + 0j),
        })
    return RawVarBlock(name="HB1.HB", columns=columns)


def test_i_probe_current_columns_classify_as_branch_currents() -> None:
    """``<id>.i`` columns end up in ``branch_currents``, keyed by full name."""

    hb = hb_dataset_from_raw(
        _raw(_hb_block_with_i_probe_currents()), (3e9, 6e9)
    )
    assert "IPR_IN.i" in hb.branch_currents
    assert "IPR_C1C2.i" in hb.branch_currents
    assert "IPR_OUT.i" in hb.branch_currents
    assert "IPR_VAR_C1.i" in hb.branch_currents
    assert "IPR_VAR_C2.i" in hb.branch_currents
    # The lone voltage column does not leak into branch_currents.
    assert hb.node_voltages == {
        "n_input": hb.node_voltages["n_input"],
    }


def test_i_probe_current_accessor_returns_spectrum() -> None:
    """``i_probe_current(probe_id)`` resolves the ``.i`` suffix internally."""

    hb = hb_dataset_from_raw(
        _raw(_hb_block_with_i_probe_currents()), (3e9, 6e9)
    )
    spectrum = hb.i_probe_current("IPR_IN")
    assert spectrum.at_mix(SIGNAL_MIX_2TONE) == 0.50 + 0j
    assert spectrum.at_mix(PUMP_MIX_2TONE) == 0.10 + 0j
    assert spectrum.at_mix(IDLER_MIX_2TONE) == 0.025 + 0j


def test_i_probe_current_at_multiple_probes() -> None:
    """Each probe's spectrum is independent and looked up by its own id."""

    hb = hb_dataset_from_raw(
        _raw(_hb_block_with_i_probe_currents()), (3e9, 6e9)
    )
    assert hb.i_probe_current("IPR_VAR_C1").at_mix(SIGNAL_MIX_2TONE) == 0.10 + 0j
    assert hb.i_probe_current("IPR_VAR_C2").at_mix(SIGNAL_MIX_2TONE) == 0.20 + 0j


def test_i_probe_current_missing_raises_keyerror_with_available_ids() -> None:
    """Asking for an absent probe lists the available I_Probe ids cleanly."""

    hb = hb_dataset_from_raw(
        _raw(_hb_block_with_i_probe_currents()), (3e9, 6e9)
    )
    with pytest.raises(KeyError) as excinfo:
        hb.i_probe_current("IPR_GHOST")
    message = str(excinfo.value)
    # The error names the probe id, the column it looked up, and lists
    # the I_Probe ids the dataset actually carries (stripped of ``.i``).
    assert "IPR_GHOST" in message
    assert "IPR_GHOST.i" in message
    assert "IPR_IN" in message
    assert "IPR_VAR_C1" in message
    # The listed ids are stripped of the ``.i`` suffix.
    assert "IPR_IN.i" not in message.split("available I_Probe ids:")[1]


def test_i_probe_current_present_only_through_probes() -> None:
    """A dataset that omits boundary probes still resolves varactor probes."""

    hb = hb_dataset_from_raw(
        _raw(_hb_block_with_i_probe_currents(with_through_probes=False)),
        (3e9, 6e9),
    )
    assert hb.i_probe_current("IPR_VAR_C1").at_mix(SIGNAL_MIX_2TONE) == 0.10 + 0j
    with pytest.raises(KeyError):
        hb.i_probe_current("IPR_IN")


