"""Tests for the per-cell power and gain helpers.

ADS-free: builds synthetic in-memory HBDatasets keyed by the actual
nets the Level-9 canonicals carry (resolved via `build_cell_map`), then
asserts the power and gain formulas read back hand-computed values.
Each test isolates one decision -- peak-phasor factor of 1/2, sign of
``Re(V * conj(I))``, dB conversion factor 10 vs 20, NaN/-inf edge cases,
cascade aggregation == sum of per-cell, etc.

The integration counterpart
(``tests/parity/test_level9_power_analysis.py``) exercises the same
helpers against a live HB run and skips on hosts without ADS.
"""

from __future__ import annotations

import math

import pytest

from tvtl_ads_builder_v2.analysis import (
    HBDataset,
    HBHarmonic,
    HBSpectrum,
    PUMP_MIX_2TONE,
    SIGNAL_MIX_2TONE,
    build_cell_map,
    cascade_power_gain_dB,
    cascade_voltage_ratio_gain_dB,
    cell_boundary_power,
    cell_count,
    cell_node_voltage,
    cell_power_gain_dB,
    cell_voltage_ratio_gain_dB,
    varactor_power,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_9_CANONICAL_CIRCUITS,
)


_SPEC_L9_2CELL = LEVEL_9_CANONICAL_CIRCUITS[0]
_SPEC_L9_3CELL = LEVEL_9_CANONICAL_CIRCUITS[1]


# --------------------------------------------------------------------------
# Synthetic-dataset helpers
# --------------------------------------------------------------------------


def _spectrum(signal: complex, *, pump: complex = 0j) -> HBSpectrum:
    """Two-harmonic spectrum at signal (1, 0) and pump (0, 1)."""

    return HBSpectrum(
        quantity="",
        harmonics=(
            HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=signal),
            HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=pump),
        ),
    )


def _build_dataset(
    cell_map,
    *,
    boundary_signal_voltages: dict[int, complex] | None = None,
    boundary_signal_currents: dict[int, complex] | None = None,
    branch_top_voltages: dict[int, complex] | None = None,
    varactor_currents: dict[int, complex] | None = None,
) -> HBDataset:
    """Build a synthetic HBDataset keyed by the cascade's actual nets.

    Each per-cell quantity is seeded at the signal mixing tuple only --
    one signal-harmonic complex amplitude per parameter -- so the test
    arithmetic is straightforward and the named-mix lookups exercise
    the same code paths the integration test does. Defaults fill in
    benign placeholder values so a test that only sets one map gets
    finite (but irrelevant) values for the others.
    """

    node_voltages: dict[str, HBSpectrum] = {}
    branch_currents: dict[str, HBSpectrum] = {}
    n_cells = cell_count(cell_map)

    boundary_signal_voltages = boundary_signal_voltages or {}
    boundary_signal_currents = boundary_signal_currents or {}
    branch_top_voltages = branch_top_voltages or {}
    varactor_currents = varactor_currents or {}

    # Boundary voltages -- seed every cell-facing boundary net, including
    # the design-output net (which is the last cell's output_net).
    for k in range(n_cells):
        net = cell_map.cells[k].input_net
        amp = boundary_signal_voltages.get(k, 1.0 + 0j)
        node_voltages[net] = _spectrum(signal=amp)
    last_out_net = cell_map.cells[-1].output_net
    node_voltages.setdefault(
        last_out_net, _spectrum(signal=boundary_signal_voltages.get(n_cells, 1.0 + 0j)),
    )

    # Branch-top voltages (varactor_signal_net == branch_top in Level 9).
    for k in range(n_cells):
        net = cell_map.cells[k].varactor_signal_net
        amp = branch_top_voltages.get(k, 0.1 + 0j)
        node_voltages.setdefault(net, _spectrum(signal=amp))

    # Boundary I_Probe currents.
    for k in range(n_cells + 1):
        amp = boundary_signal_currents.get(k, 0.01 + 0j)
        if k < n_cells:
            probe_id = cell_map.cells[k].boundary_in_probe_id
        else:
            probe_id = cell_map.cells[-1].boundary_out_probe_id
        if probe_id is not None:
            branch_currents[f"{probe_id}.i"] = _spectrum(signal=amp)

    # Varactor I_Probe currents.
    for k in range(n_cells):
        probe_id = cell_map.cells[k].varactor_probe_id
        amp = varactor_currents.get(k, 0.001 + 0j)
        if probe_id is not None:
            branch_currents[f"{probe_id}.i"] = _spectrum(signal=amp)

    return HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )


# --------------------------------------------------------------------------
# _time_avg_real_power -- the peak-phasor formula
# --------------------------------------------------------------------------


def test_power_formula_uses_one_half_peak_phasor() -> None:
    """V = 2, I = 1 (both real) -> P = (1/2) * 2 * 1 = 1.0 W."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 2.0 + 0j},
        boundary_signal_currents={0: 1.0 + 0j},
    )
    p = cell_boundary_power(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert p == pytest.approx(1.0)


def test_power_formula_uses_conjugate_of_current() -> None:
    """V = 1 + 1j, I = 1 - 1j -> Re((1+j)(1-j)*) = Re((1+j)(1+j)) = Re(2j) = 0."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 1.0j},
        boundary_signal_currents={0: 1.0 - 1.0j},
    )
    # V * conj(I) = (1+j) * (1+j) = 2j; Re = 0; P = 0.
    p = cell_boundary_power(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert p == pytest.approx(0.0, abs=1e-12)


def test_power_formula_real_aligned_phasors() -> None:
    """V and I in phase -> peak-phasor P = (1/2) |V||I| for real-aligned."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # V and I real -> Re(V * conj(I)) = V * I; P = V*I/2.
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 3.0 + 0j},
        boundary_signal_currents={0: 2.0 + 0j},
    )
    p = cell_boundary_power(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert p == pytest.approx(3.0)  # (1/2) * 3 * 2


def test_power_formula_orthogonal_phasors_yields_zero() -> None:
    """V purely real, I purely imaginary (90 deg apart) -> P = 0."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j},
        boundary_signal_currents={0: 0.0 + 1.0j},
    )
    # V * conj(I) = 1 * -j = -j; Re = 0.
    p = cell_boundary_power(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert p == pytest.approx(0.0, abs=1e-12)


def test_power_can_be_negative_when_phasors_anti_aligned() -> None:
    """V real positive, I real negative -> Re(V*I.conj) < 0; P < 0 ."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j},
        boundary_signal_currents={0: -2.0 + 0j},
    )
    p = cell_boundary_power(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert p == pytest.approx(-1.0)  # (1/2) * 1 * (-2)


# --------------------------------------------------------------------------
# cell_boundary_power -- boundary indexing, V-node selection
# --------------------------------------------------------------------------


def test_boundary_power_uses_cell_input_net_for_interior_boundaries() -> None:
    """For boundary k < N, V is read from cells[k].input_net."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # cell 1's input net has its own V (the inter-cell probe seam).
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={1: 0.5 + 0j},
        boundary_signal_currents={1: 0.1 + 0j},
    )
    p = cell_boundary_power(hb, cm, 1, SIGNAL_MIX_2TONE)
    assert p == pytest.approx(0.5 * 0.5 * 0.1)


def test_boundary_power_uses_last_output_net_for_design_output() -> None:
    """For boundary N, V is read from cells[N-1].output_net."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    n = cell_count(cm)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={n: 0.7 + 0j},
        boundary_signal_currents={n: 0.2 + 0j},
    )
    p = cell_boundary_power(hb, cm, n, SIGNAL_MIX_2TONE)
    assert p == pytest.approx(0.5 * 0.7 * 0.2)


def test_boundary_power_out_of_range_raises_indexerror() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(cm)
    with pytest.raises(IndexError, match="boundary_index"):
        cell_boundary_power(hb, cm, -1, SIGNAL_MIX_2TONE)
    with pytest.raises(IndexError, match="boundary_index"):
        cell_boundary_power(hb, cm, 9, SIGNAL_MIX_2TONE)


# --------------------------------------------------------------------------
# varactor_power -- branch-top V plus varactor probe I
# --------------------------------------------------------------------------


def test_varactor_power_reads_branch_top_voltage() -> None:
    """varactor_power pairs V_branch_top with I_varactor_probe."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        branch_top_voltages={0: 0.4 + 0j},
        varactor_currents={0: 0.05 + 0j},
    )
    # V * conj(I) = 0.4 * 0.05 (real); P = 0.5 * 0.02 = 0.01
    p = varactor_power(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert p == pytest.approx(0.01)


def test_varactor_power_scales_with_cell_count() -> None:
    """Each cell's varactor probe is independent; per-cell P seeds distinctly."""

    cm = build_cell_map(_SPEC_L9_3CELL)
    hb = _build_dataset(
        cm,
        branch_top_voltages={0: 1.0 + 0j, 1: 0.5 + 0j, 2: 0.25 + 0j},
        varactor_currents={0: 0.1 + 0j, 1: 0.1 + 0j, 2: 0.1 + 0j},
    )
    powers = [varactor_power(hb, cm, k, SIGNAL_MIX_2TONE) for k in range(3)]
    assert powers[0] == pytest.approx(0.5 * 1.0 * 0.1)
    assert powers[1] == pytest.approx(0.5 * 0.5 * 0.1)
    assert powers[2] == pytest.approx(0.5 * 0.25 * 0.1)


# --------------------------------------------------------------------------
# cell_voltage_ratio_gain_dB
# --------------------------------------------------------------------------


def test_voltage_ratio_gain_zero_dB_when_V_out_equals_V_in() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    # Make cells[0].output_net == cells[1].input_net's amplitude
    # automatically -- but the existing _build_dataset seeds them per
    # boundary index, so set both to the same magnitude.
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 2.0 + 0j, 1: 2.0 + 0j},
    )
    # Cell 0: V_in = boundary[0] (cells[0].input_net), V_out is
    # cells[0].output_net which is not the same net as cells[1].input_net
    # in Level-9 (the inter-cell I_Probe spans them). We set the
    # cells[0].output_net amplitude explicitly via the dataset.
    # For this test, write directly into node_voltages:
    hb.node_voltages[cm.cells[0].output_net] = _spectrum(signal=2.0 + 0j)
    g = cell_voltage_ratio_gain_dB(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert g == pytest.approx(0.0, abs=1e-12)


def test_voltage_ratio_gain_six_dB_for_two_to_one() -> None:
    """V_out / V_in = 2 -> 20*log10(2) ~= 6.0206 dB."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(cm, boundary_signal_voltages={0: 1.0 + 0j})
    hb.node_voltages[cm.cells[0].output_net] = _spectrum(signal=2.0 + 0j)
    g = cell_voltage_ratio_gain_dB(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert g == pytest.approx(20.0 * math.log10(2.0))


def test_voltage_ratio_gain_minus_inf_on_zero_input() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(cm, boundary_signal_voltages={0: 0.0 + 0j})
    hb.node_voltages[cm.cells[0].output_net] = _spectrum(signal=1.0 + 0j)
    g = cell_voltage_ratio_gain_dB(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert g == float("-inf")


def test_voltage_ratio_gain_uses_magnitude_not_complex() -> None:
    """Phase difference between V_in and V_out does not affect |V| ratio."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(cm, boundary_signal_voltages={0: 1.0 + 0j})
    # Same magnitude (1.0) as V_in but rotated 90 degrees -- gain = 0 dB.
    hb.node_voltages[cm.cells[0].output_net] = _spectrum(signal=0.0 + 1.0j)
    g = cell_voltage_ratio_gain_dB(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert g == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# cell_power_gain_dB
# --------------------------------------------------------------------------


def test_power_gain_zero_dB_when_powers_equal() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    # Set P_in = P_out by seeding both boundary V/I with equal real values.
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.5 + 0j},
    )
    g = cell_power_gain_dB(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert g == pytest.approx(0.0, abs=1e-12)


def test_power_gain_three_dB_for_two_to_one() -> None:
    """P_out / P_in = 2 -> 10*log10(2) ~= 3.0103 dB."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 1.0 + 0j},
    )
    g = cell_power_gain_dB(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert g == pytest.approx(10.0 * math.log10(2.0))


def test_power_gain_nan_when_input_power_nonpositive() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    # P_in negative: V and I anti-aligned at boundary 0.
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: -0.5 + 0j, 1: 0.5 + 0j},
    )
    g = cell_power_gain_dB(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert math.isnan(g)


def test_power_gain_nan_when_output_power_nonpositive() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: -0.5 + 0j},
    )
    g = cell_power_gain_dB(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert math.isnan(g)


# --------------------------------------------------------------------------
# cascade aggregations -- self-consistency with per-cell sums
# --------------------------------------------------------------------------


def test_cascade_voltage_ratio_gain_equals_sum_of_per_cell() -> None:
    """The dB voltage-ratio gain telescopes across cells when each cell's
    output net amplitude equals the next cell's input net amplitude (the
    inter-cell I_Probe is a 0-ohm short; ADS reports identical amplitudes
    on both sides of it)."""

    cm = build_cell_map(_SPEC_L9_3CELL)
    n = cell_count(cm)
    # Chain V_in[0] -> V_out[0] = V_in[1] -> V_out[1] = V_in[2] -> V_out[2].
    chain = [1.0 + 0j, 0.9 + 0j, 0.8 + 0j, 0.7 + 0j]
    boundary_voltages = {k: chain[k] for k in range(n + 1)}
    hb = _build_dataset(cm, boundary_signal_voltages=boundary_voltages)
    # Force each cell's output_net to match the next cell's input net
    # amplitude (i.e. treat the inter-cell probe as a true 0-ohm short).
    for k in range(n):
        next_amp = chain[k + 1]
        hb.node_voltages[cm.cells[k].output_net] = _spectrum(signal=next_amp)
    per_cell_gains = sum(
        cell_voltage_ratio_gain_dB(hb, cm, k, SIGNAL_MIX_2TONE) for k in range(n)
    )
    cascade = cascade_voltage_ratio_gain_dB(hb, cm, SIGNAL_MIX_2TONE)
    assert per_cell_gains == pytest.approx(cascade, abs=1e-12)


def test_cascade_power_gain_equals_sum_of_per_cell() -> None:
    """The dB power gain telescopes across cells under uniform boundary V."""

    cm = build_cell_map(_SPEC_L9_3CELL)
    n = cell_count(cm)
    # Seed P[k] = 0.5 * V * I[k] with V = 1, I[k] linearly varying so
    # P_out[k] / P_in[k] varies per cell.
    voltages = {k: 1.0 + 0j for k in range(n + 1)}
    currents = {0: 0.5 + 0j, 1: 0.6 + 0j, 2: 0.4 + 0j, 3: 0.7 + 0j}
    hb = _build_dataset(
        cm,
        boundary_signal_voltages=voltages,
        boundary_signal_currents=currents,
    )
    per_cell = sum(
        cell_power_gain_dB(hb, cm, k, SIGNAL_MIX_2TONE) for k in range(n)
    )
    cascade = cascade_power_gain_dB(hb, cm, SIGNAL_MIX_2TONE)
    assert per_cell == pytest.approx(cascade, abs=1e-12)


def test_cascade_power_gain_telescopes_arbitrary_currents() -> None:
    """Random per-boundary I yields cascade == sum(per-cell)."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    n = cell_count(cm)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={k: 1.5 + 0j for k in range(n + 1)},
        boundary_signal_currents={0: 0.10 + 0j, 1: 0.20 + 0j, 2: 0.05 + 0j},
    )
    per_cell = sum(
        cell_power_gain_dB(hb, cm, k, SIGNAL_MIX_2TONE) for k in range(n)
    )
    cascade = cascade_power_gain_dB(hb, cm, SIGNAL_MIX_2TONE)
    assert per_cell == pytest.approx(cascade, abs=1e-12)


def test_cascade_voltage_ratio_gain_minus_inf_on_zero_input() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(cm, boundary_signal_voltages={0: 0.0 + 0j})
    g = cascade_voltage_ratio_gain_dB(hb, cm, SIGNAL_MIX_2TONE)
    assert g == float("-inf")


def test_cascade_power_gain_nan_on_nonpositive_input_power() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j},
        boundary_signal_currents={0: -0.5 + 0j},
    )
    g = cascade_power_gain_dB(hb, cm, SIGNAL_MIX_2TONE)
    assert math.isnan(g)


# --------------------------------------------------------------------------
# Mix indexing -- power at the pump tone differs from power at signal
# --------------------------------------------------------------------------


def test_power_at_pump_mix_is_independent_of_signal_mix() -> None:
    """The power helpers thread the requested mix to the underlying reads."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # Seed signal and pump on the input boundary with different
    # amplitudes so their power values are distinguishable.
    n_in_net = cm.cells[0].input_net
    probe_id = cm.cells[0].boundary_in_probe_id
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9),
        node_voltages={
            n_in_net: HBSpectrum(
                quantity="",
                harmonics=(
                    HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=2.0 + 0j),
                    HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=4.0 + 0j),
                ),
            ),
        },
        branch_currents={
            f"{probe_id}.i": HBSpectrum(
                quantity="",
                harmonics=(
                    HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=1.0 + 0j),
                    HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=2.0 + 0j),
                ),
            ),
        },
    )
    p_signal = cell_boundary_power(hb, cm, 0, SIGNAL_MIX_2TONE)
    p_pump = cell_boundary_power(hb, cm, 0, PUMP_MIX_2TONE)
    assert p_signal == pytest.approx(1.0)  # (1/2)*2*1
    assert p_pump == pytest.approx(4.0)    # (1/2)*4*2


