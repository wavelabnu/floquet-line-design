"""Tests for the energy-conservation cross-check.

ADS-free: builds synthetic in-memory HBDatasets keyed by the actual
nets of the Level-9 canonicals (resolved via :func:`build_cell_map`),
then asserts that :func:`cell_energy_residual` and
:func:`cascade_energy_residuals` evaluate the cell-level conservation
law correctly. The companion integration test
``tests/parity/test_level9_energy_conservation.py`` exercises the same
helpers against a live HB run.

Conservation law under test:

    ΔP_signal(k) ≡ P_boundary_out(k) - P_boundary_in(k) ≈ -P_var(k)

i.e. the residual ``r(k) = ΔP_signal(k) + P_var(k)`` should be ``~0``.
"""

from __future__ import annotations

import math

import pytest

from tvtl_ads_builder_v2.analysis import (
    DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT,
    HBDataset,
    HBHarmonic,
    HBSpectrum,
    PUMP_MIX_2TONE,
    SIGNAL_MIX_2TONE,
    build_cell_map,
    cascade_energy_residuals,
    cell_count,
    cell_energy_residual,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_9_CANONICAL_CIRCUITS,
)


_SPEC_L9_2CELL = LEVEL_9_CANONICAL_CIRCUITS[0]
_SPEC_L9_3CELL = LEVEL_9_CANONICAL_CIRCUITS[1]


# --------------------------------------------------------------------------
# Synthetic-dataset helper (mirrors the test_analysis_power.py shape)
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
    """Build a synthetic HBDataset keyed by the cascade's actual nets."""

    node_voltages: dict[str, HBSpectrum] = {}
    branch_currents: dict[str, HBSpectrum] = {}
    n_cells = cell_count(cell_map)

    boundary_signal_voltages = boundary_signal_voltages or {}
    boundary_signal_currents = boundary_signal_currents or {}
    branch_top_voltages = branch_top_voltages or {}
    varactor_currents = varactor_currents or {}

    for k in range(n_cells):
        net = cell_map.cells[k].input_net
        amp = boundary_signal_voltages.get(k, 1.0 + 0j)
        node_voltages[net] = _spectrum(signal=amp)
    last_out_net = cell_map.cells[-1].output_net
    node_voltages.setdefault(
        last_out_net,
        _spectrum(signal=boundary_signal_voltages.get(n_cells, 1.0 + 0j)),
    )

    for k in range(n_cells):
        net = cell_map.cells[k].varactor_signal_net
        amp = branch_top_voltages.get(k, 0.1 + 0j)
        node_voltages.setdefault(net, _spectrum(signal=amp))

    for k in range(n_cells + 1):
        amp = boundary_signal_currents.get(k, 0.01 + 0j)
        if k < n_cells:
            probe_id = cell_map.cells[k].boundary_in_probe_id
        else:
            probe_id = cell_map.cells[-1].boundary_out_probe_id
        if probe_id is not None:
            branch_currents[f"{probe_id}.i"] = _spectrum(signal=amp)

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
# cell_energy_residual -- exact-zero residual when conservation holds
# --------------------------------------------------------------------------


def test_passive_cell_perfect_conservation_yields_zero_residual() -> None:
    """Hand-construct a cell where ΔP_signal = -P_var exactly.

    Boundary: V_in = 1, I_in = 0.5 -> P_in = 0.25 W.
              V_out = 1, I_out = 0.4 -> P_out = 0.20 W.
              ΔP = -0.05 W.
    Branch:   V_btop = 0.5, I_var = 0.2 -> P_var = 0.05 W.
    Residual: ΔP + P_var = -0.05 + 0.05 = 0.
    """

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.4 + 0j},
        branch_top_voltages={0: 0.5 + 0j},
        varactor_currents={0: 0.2 + 0j},
    )
    result = cell_energy_residual(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert result["P_boundary_in"] == pytest.approx(0.25)
    assert result["P_boundary_out"] == pytest.approx(0.20)
    assert result["delta_P_signal"] == pytest.approx(-0.05)
    assert result["P_var"] == pytest.approx(0.05)
    assert result["residual"] == pytest.approx(0.0, abs=1e-15)
    assert result["within_tolerance"] is True


def test_active_cell_negative_P_var_perfect_conservation() -> None:
    """Parametric-gain cell: P_var < 0 sources power; ΔP > 0.

    Boundary: V_in = 1, I_in = 0.5 -> P_in = 0.25 W.
              V_out = 1, I_out = 0.7 -> P_out = 0.35 W.
              ΔP = +0.10 W.
    Branch:   V_btop = 0.5, I_var = -0.4 -> P_var = -0.10 W.
    Residual: 0.10 + (-0.10) = 0.
    """

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.7 + 0j},
        branch_top_voltages={0: 0.5 + 0j},
        varactor_currents={0: -0.4 + 0j},
    )
    result = cell_energy_residual(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert result["P_var"] == pytest.approx(-0.10)
    assert result["delta_P_signal"] == pytest.approx(0.10)
    assert result["residual"] == pytest.approx(0.0, abs=1e-15)


def test_residual_picks_up_nonzero_imbalance() -> None:
    """Construct a 10% imbalance and verify it surfaces in W and dB."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # P_in = 0.25 W. Choose residual = 0.025 W -> 10% of input -> -20 dB.
    # ΔP = -0.05 W; P_var = 0.075 W; residual = +0.025 W.
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.4 + 0j},
        branch_top_voltages={0: 0.5 + 0j},
        varactor_currents={0: 0.3 + 0j},
    )
    result = cell_energy_residual(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert result["P_var"] == pytest.approx(0.075)
    assert result["residual"] == pytest.approx(0.025)
    assert result["residual_dB_rel_input"] == pytest.approx(-20.0, abs=1e-9)
    # Default budget -40 dB; -20 dB exceeds it.
    assert result["within_tolerance"] is False


def test_residual_within_tolerance_obeys_dB_threshold() -> None:
    """A 0.1% residual is -60 dB -- inside the default -40 dB budget."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # P_in = 0.25 W. Residual = 2.5e-4 -> -60 dB rel input.
    # ΔP = -0.05 W; P_var = 0.05025 W; residual = +2.5e-4 W.
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.4 + 0j},
        branch_top_voltages={0: 0.5 + 0j},
        varactor_currents={0: 0.201 + 0j},
    )
    result = cell_energy_residual(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert result["residual"] == pytest.approx(0.00025)
    assert result["residual_dB_rel_input"] == pytest.approx(-60.0, abs=1e-9)
    assert result["within_tolerance"] is True


def test_tolerance_argument_is_honoured() -> None:
    """A custom looser tolerance can accept a residual the default rejects."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.4 + 0j},
        branch_top_voltages={0: 0.5 + 0j},
        varactor_currents={0: 0.3 + 0j},
    )
    # Default -40 dB rejects.
    r_default = cell_energy_residual(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert r_default["within_tolerance"] is False
    # Loosened to -10 dB accepts.
    r_loose = cell_energy_residual(
        hb, cm, 0, SIGNAL_MIX_2TONE, tolerance_dB_rel_input=-10.0,
    )
    assert r_loose["within_tolerance"] is True


def test_default_tolerance_constant_is_minus_40_dB() -> None:
    """1% of input power -> 20*log10(0.01) = -40 dB."""

    assert DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT == pytest.approx(-40.0)


def test_cell_index_out_of_range_raises_indexerror() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(cm)
    with pytest.raises(IndexError, match="cell_index"):
        cell_energy_residual(hb, cm, -1, SIGNAL_MIX_2TONE)
    with pytest.raises(IndexError, match="cell_index"):
        cell_energy_residual(hb, cm, 7, SIGNAL_MIX_2TONE)


def test_residual_zero_residual_dB_is_extremely_negative() -> None:
    """A residual at FP noise floor reports a very negative dB ratio.

    Floating-point subtraction of equal floats yields ``0.0`` exactly,
    but ``-(-x) + x`` (the residual computation through two boundary
    powers and one varactor power) leaves an O(eps) artifact. We
    therefore assert the dB-rel-input value is far below the default
    -40 dB budget rather than exactly ``-inf``. A literal ``-inf`` is
    still returned when ``r == 0.0`` exactly -- that branch is covered
    by the helper's docstring and indirectly verified at the W-level
    by the other perfect-conservation tests.
    """

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.4 + 0j},
        branch_top_voltages={0: 0.5 + 0j},
        varactor_currents={0: 0.2 + 0j},
    )
    result = cell_energy_residual(hb, cm, 0, SIGNAL_MIX_2TONE)
    # Floating-point cancellation noise leaves a residual at ~ 1e-17 W
    # which translates to ~ -300 dB rel input -- well below any
    # physically-meaningful budget.
    assert result["residual_dB_rel_input"] < -200.0


def test_residual_at_pump_mix_independent_of_signal_mix() -> None:
    """The helper threads the requested mix through to every read."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # For boundary_index = 1 (out of cell 0) the V comes from cells[1].input_net
    # (n_pc1c2), and the current from cells[1].boundary_in_probe_id (which
    # equals cells[0].boundary_out_probe_id -- same physical probe).
    n_in_net = cm.cells[0].input_net  # n_pin
    next_in_net = cm.cells[1].input_net  # n_pc1c2 -- the boundary-1 V net
    btop_net = cm.cells[0].varactor_signal_net  # n_c1_b
    b_in_probe = cm.cells[0].boundary_in_probe_id  # IPR_IN
    next_b_in_probe = cm.cells[1].boundary_in_probe_id  # IPR_C1C2
    v_probe = cm.cells[0].varactor_probe_id  # IPR_VAR_C1

    # Signal at cell 0: V_in=V_out=1, I_in=0.5, I_out=0.4 -> ΔP=-0.05;
    # V_btop=0.5, I_var=0.2 -> P_var=0.05. Residual=0.
    # Pump at cell 0:   V_in=V_out=2, I_in=0.4, I_out=0.3 -> ΔP=-0.10;
    # V_btop=0.5, I_var=0.4 -> P_var=0.10. Residual=0.
    def _two_mix(signal: complex, pump: complex) -> HBSpectrum:
        return HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=pump),
            ),
        )

    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9),
        node_voltages={
            n_in_net: _two_mix(1.0 + 0j, 2.0 + 0j),
            next_in_net: _two_mix(1.0 + 0j, 2.0 + 0j),
            btop_net: _two_mix(0.5 + 0j, 0.5 + 0j),
        },
        branch_currents={
            f"{b_in_probe}.i": _two_mix(0.5 + 0j, 0.4 + 0j),
            f"{next_b_in_probe}.i": _two_mix(0.4 + 0j, 0.3 + 0j),
            f"{v_probe}.i": _two_mix(0.2 + 0j, 0.4 + 0j),
        },
    )
    r_signal = cell_energy_residual(hb, cm, 0, SIGNAL_MIX_2TONE)
    r_pump = cell_energy_residual(hb, cm, 0, PUMP_MIX_2TONE)
    assert r_signal["delta_P_signal"] == pytest.approx(-0.05)
    assert r_pump["delta_P_signal"] == pytest.approx(-0.10)
    assert r_signal["P_var"] == pytest.approx(0.05)
    assert r_pump["P_var"] == pytest.approx(0.10)
    assert r_signal["residual"] == pytest.approx(0.0, abs=1e-15)
    assert r_pump["residual"] == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------
# cascade_energy_residuals -- aggregation
# --------------------------------------------------------------------------


def test_cascade_residuals_returns_one_entry_per_cell() -> None:
    cm = build_cell_map(_SPEC_L9_3CELL)
    hb = _build_dataset(cm)
    result = cascade_energy_residuals(hb, cm, SIGNAL_MIX_2TONE)
    assert len(result["per_cell"]) == cell_count(cm) == 3


def test_cascade_residuals_within_tolerance_aggregates_correctly() -> None:
    """A cascade whose every cell satisfies conservation passes overall."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # Chain V_in=V_out=1 for every boundary; I varying so per-cell ΔP varies.
    # For each cell k, set varactor_currents[k] so that ΔP + P_var = 0.
    # Boundary I (k=0, 1, 2): 0.5, 0.4, 0.3 -> P_in_0=0.25, P_out_0=0.20,
    # P_in_1=0.20, P_out_1=0.15.
    # ΔP_0 = -0.05; P_var_0 must be +0.05; V_btop=0.5 -> I_var=0.2.
    # ΔP_1 = -0.05; P_var_1 must be +0.05; V_btop=0.5 -> I_var=0.2.
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j, 2: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.4 + 0j, 2: 0.3 + 0j},
        branch_top_voltages={0: 0.5 + 0j, 1: 0.5 + 0j},
        varactor_currents={0: 0.2 + 0j, 1: 0.2 + 0j},
    )
    result = cascade_energy_residuals(hb, cm, SIGNAL_MIX_2TONE)
    assert result["within_tolerance"] is True
    for entry in result["per_cell"]:
        assert entry["residual"] == pytest.approx(0.0, abs=1e-15)


def test_cascade_residuals_flags_failing_cell() -> None:
    """A single failing cell flips the cascade flag to False."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # Cell 0 conserved; cell 1 has a +25% residual that exceeds -40 dB.
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j, 2: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.4 + 0j, 2: 0.3 + 0j},
        branch_top_voltages={0: 0.5 + 0j, 1: 0.5 + 0j},
        # Cell 1's correct I_var is 0.2; use 0.5 instead -> residual large.
        varactor_currents={0: 0.2 + 0j, 1: 0.5 + 0j},
    )
    result = cascade_energy_residuals(hb, cm, SIGNAL_MIX_2TONE)
    assert result["within_tolerance"] is False
    assert result["per_cell"][0]["within_tolerance"] is True
    assert result["per_cell"][1]["within_tolerance"] is False
    assert result["max_residual_dB_rel_input"] > -40.0


def test_cascade_total_combines_design_endpoints_and_per_cell_pvar() -> None:
    """cascade_total = (P_design_out - P_design_in) + sum_k P_var(k)."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j, 2: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.4 + 0j, 2: 0.3 + 0j},
        branch_top_voltages={0: 0.5 + 0j, 1: 0.5 + 0j},
        varactor_currents={0: 0.2 + 0j, 1: 0.2 + 0j},
    )
    result = cascade_energy_residuals(hb, cm, SIGNAL_MIX_2TONE)
    # P_design_in = 0.25, P_design_out = 0.15 (boundary 2 I = 0.3).
    # sum P_var = 0.05 + 0.05 = 0.10.
    # cascade_total = -0.10 + 0.10 = 0.
    assert result["cascade_total"] == pytest.approx(0.0, abs=1e-15)


def test_cascade_empty_raises() -> None:
    """A 0-cell cascade is not meaningful; aggregator rejects it."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # Cell-count is structural; we cannot construct a 0-cell L9 spec, so
    # build an artificial CellMap with no cells via the dataclass.
    from tvtl_ads_builder_v2.analysis.per_cell import CellMap

    empty = CellMap(cells=(), design_input_net="x", design_output_net="x")
    hb = _build_dataset(cm)  # any HB; helper rejects before reading
    with pytest.raises(ValueError, match="no cells"):
        cascade_energy_residuals(hb, empty, SIGNAL_MIX_2TONE)


def test_residual_imports_cleanly_from_facade() -> None:
    """The new symbols are reachable from the v2 facade ``__all__``."""

    import tvtl_ads_builder_v2 as v2

    assert hasattr(v2, "cell_energy_residual")
    assert hasattr(v2, "cascade_energy_residuals")
    assert hasattr(v2, "DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT")
    assert "cell_energy_residual" in v2.__all__
    assert "cascade_energy_residuals" in v2.__all__
    assert "DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT" in v2.__all__


def test_per_cell_entries_carry_expected_keys() -> None:
    """Verify the result dict's documented schema."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(cm)
    r = cell_energy_residual(hb, cm, 0, SIGNAL_MIX_2TONE)
    expected = {
        "residual",
        "residual_dB_rel_input",
        "P_boundary_in",
        "P_boundary_out",
        "P_var",
        "delta_P_signal",
        "P_input_reference",
        "within_tolerance",
    }
    assert set(r.keys()) == expected


def test_cascade_result_carries_expected_keys() -> None:
    cm = build_cell_map(_SPEC_L9_3CELL)
    hb = _build_dataset(cm)
    r = cascade_energy_residuals(hb, cm, SIGNAL_MIX_2TONE)
    expected = {
        "per_cell",
        "cascade_total",
        "cascade_total_dB_rel_input",
        "max_abs_residual",
        "max_residual_dB_rel_input",
        "within_tolerance",
    }
    assert set(r.keys()) == expected
    assert isinstance(r["per_cell"], tuple)
    assert len(r["per_cell"]) == 3


def test_residual_pure_loss_no_branch_returns_negative_dP_positive_pvar() -> None:
    """Sanity: in pure-loss regime, ΔP < 0 and P_var > 0 (passive shunt)."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={0: 1.0 + 0j, 1: 1.0 + 0j},
        boundary_signal_currents={0: 0.5 + 0j, 1: 0.4 + 0j},
        branch_top_voltages={0: 0.5 + 0j},
        varactor_currents={0: 0.2 + 0j},
    )
    r = cell_energy_residual(hb, cm, 0, SIGNAL_MIX_2TONE)
    assert r["delta_P_signal"] < 0
    assert r["P_var"] > 0


def test_residual_sums_finite_per_cascade() -> None:
    """sum(per-cell residual) should be finite even with imbalances."""

    cm = build_cell_map(_SPEC_L9_3CELL)
    hb = _build_dataset(
        cm,
        boundary_signal_voltages={k: 1.0 + 0j for k in range(4)},
        boundary_signal_currents={0: 0.5, 1: 0.4, 2: 0.3, 3: 0.2},
        branch_top_voltages={k: 0.5 + 0j for k in range(3)},
        varactor_currents={0: 0.21, 1: 0.19, 2: 0.18},
    )
    result = cascade_energy_residuals(hb, cm, SIGNAL_MIX_2TONE)
    total = sum(float(e["residual"]) for e in result["per_cell"])
    assert math.isfinite(total)
    # max_abs_residual is one of the per-cell |r|.
    assert result["max_abs_residual"] >= max(
        abs(float(e["residual"])) for e in result["per_cell"]
    ) - 1e-15


